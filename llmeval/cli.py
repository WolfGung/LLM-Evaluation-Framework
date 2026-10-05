"""The command line: `llmeval` (or `python -m llmeval`).

- `eval`: replay the recorded run, judge calls included, and write
  `results/`. Without `cassettes/manifest.json` it prints "pending first
  recorded run" and writes nothing. It always replays, whatever LLMEVAL_MODE
  says: recording (with the budget guard and the free-quota handling) is
  `make record`.
- `estimate`: the call plan (by function, version, repeat and judge), the
  free-quota arithmetic and the estimated cost of the calls still to record.
  Refuses (exit 1) above MAX_RUN_COST_USD. Needs no key; for a paid model
  without recorded costs it reads the public price list.
- `status`: calls planned and recorded, whether the manifest is there, and,
  with OPENROUTER_API_KEY set, the free requests the key has left today.
- `record`: record every planned call with the real API (needs the key),
  resumably and inside the free limits; see `llmeval.recording`.
- `prune`: list the recorded entries the current plan no longer has, and
  with `--yes` remove them; see `llmeval.prune`.
- `retrieval`: the retrieval layer over the RAG dataset, offline. BM25 is
  deterministic, so this needs no model and no recording.
- `baseline`: write `results/baseline.json` from the replay results in
  `results/` and the manifest; refuses missing or stale results.
- `gate`: compare the key rates of a results directory with the baseline,
  within the tolerances of `config/gate.yaml`; exit 1 on a regression. See
  `llmeval.gate`.
- `sample`: write `labels/sample.json`, the judged answers the owner labels
  by hand; see `llmeval.labels`.

The API key comes only from the environment and is never printed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Annotated

import httpx
import typer

from app.assistant import DEFAULT_K
from app.prompting import PromptError
from app.retrieval import search
from llmeval.baseline import (
    BASELINE_PATH,
    BaselineError,
    CurrentInputs,
    PairwiseMetrics,
    build_baseline,
    load_baseline,
    load_run_results,
    result_label,
    stale_reasons,
    write_baseline,
)
from llmeval.callplan import (
    PlanCounts,
    PlanInputs,
    count_plan,
    estimate_remaining_cost,
    full_plan,
    plan_lines,
    plural,
    quota_lines,
    replay_plan,
    unplanned,
)
from llmeval.cassettes import (
    PENDING_RECORDED_RUN,
    CassetteError,
    CassetteStore,
    RunManifest,
    load_manifest,
)
from llmeval.checks.judge import RUBRIC_PATH, RubricError, load_rubric
from llmeval.checks.retrieval import retrieval_recall
from llmeval.client import MissingRecording, ModelClient
from llmeval.config import (
    DEFAULT_MODELS_PATH,
    Config,
    ConfigError,
    Mode,
    check_stability_cases,
    load_config,
)
from llmeval.datasets import (
    DATASETS_DIR,
    RAG_PATH,
    DatasetError,
    file_sha256,
    load_rag,
    load_triage,
)
from llmeval.gate import (
    GATE_CONFIG_PATH,
    GateError,
    format_report,
    gate,
    load_tolerances,
    results_outside,
)
from llmeval.labels import (
    SAMPLE_PATH,
    LabelError,
    build_sample,
    runs_by_answer,
    write_sample,
)
from llmeval.openrouter import MissingAPIKey, OpenRouterError
from llmeval.perf import Performance
from llmeval.pricing import BudgetExceeded, PricingError, check_budget, format_usd
from llmeval.prune import PruneRefused, prune
from llmeval.quota import free_daily_quota
from llmeval.recording import (
    EXIT_INTERRUPTED,
    EXIT_STOPPED,
    PlanMismatch,
    RecordLocked,
    record_all,
)
from llmeval.results import RESULTS_DIR, FunctionResults, PairwiseResults
from llmeval.runner import CASSETTES_DIR, EVAL_FUNCTIONS, prompt_sha256, run, versions_of
from llmeval.stability import Stability

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Evaluate the Toolshop LLM features from recorded calls.",
)


def _fail(message: str) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(code=1)


@dataclass(frozen=True)
class Network:
    """How the commands reach the API: the real network unless a test swaps it."""

    transport: httpx.BaseTransport | None = None
    limiter: object | None = None
    sleep: Callable[[float], None] = field(default=time.sleep)


def _network() -> Network:
    return Network()


# Errors a planning command reports in one line (exit code 1).
PLAN_ERRORS = (
    CassetteError,
    ConfigError,
    DatasetError,
    RubricError,
    PricingError,
    OpenRouterError,
    OSError,
)

ConfigOption = Annotated[Path, typer.Option("--config", help="Model config file.")]
DatasetsOption = Annotated[Path, typer.Option(help="Directory with the datasets.")]
CassettesOption = Annotated[Path, typer.Option(help="Recorded calls.")]
RubricOption = Annotated[Path, typer.Option(help="The judge rubric.")]
ResultsOption = Annotated[Path, typer.Option(help="Directory with the replay results.")]
BaselineOption = Annotated[Path, typer.Option(help="The baseline file.")]
SampleOption = Annotated[Path, typer.Option("--sample", help="The label sample.")]


def _echo_notices(store: CassetteStore) -> None:
    for notice in store.notices:
        typer.echo(f"notice: {notice}")


def _dataset_paths(datasets_dir: Path) -> dict[str, Path]:
    return {"rag": datasets_dir / "rag.jsonl", "triage": datasets_dir / "triage.jsonl"}


def _plan_inputs(config: Path, datasets_dir: Path, rubric: Path) -> tuple[Config, PlanInputs]:
    """The config (with the environment) and what the full recording covers."""
    loaded = load_config(config)
    paths = _dataset_paths(datasets_dir)
    rag_cases, triage_cases = load_rag(paths["rag"]), load_triage(paths["triage"])
    check_stability_cases(loaded.models, {case.id for case in (*rag_cases, *triage_cases)})
    inputs = PlanInputs(
        models=loaded.models,
        rag_cases=rag_cases,
        triage_cases=triage_cases,
        rubric=load_rubric(rubric),
        versions={function: versions_of(function) for function in EVAL_FUNCTIONS},
    )
    return loaded, inputs


def _line(result: FunctionResults) -> str:
    parts = [
        f"{layer} {rate.passed}/{rate.total} ({rate.rate:.1%})"
        for layer, rate in result.summary.layers.items()
        if rate.rate is not None
    ]
    total = result.summary.all_checks
    parts.append(f"all checks {total.passed}/{total.total}")
    if (stable := result.summary.stability) is not None:
        parts.append(f"stable {stable.stable}/{stable.repeated} ({stable.stable_share:.1%})")
    return f"{result.function} {result.version}: " + ", ".join(parts)


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


def _tokens(value: float | None) -> str:
    return "n/a" if value is None else str(value)


def _perf_line(label: str, perf: Performance) -> str:
    """One line of performance: latency, mean tokens and cost (unknown costs named)."""
    cost = perf.cost
    spent = f"cost ${cost.total_usd:.6f}"
    if cost.unknown_calls:
        spent += f" known, {cost.unknown_calls} calls with unknown cost"
    elif cost.per_case_usd is not None:
        spent += f" (${cost.per_case_usd:.6f} per case)"
    return (
        f"  {label} calls {perf.calls}, latency p50 {_ms(perf.latency.p50_ms)}, "
        f"p95 {_ms(perf.latency.p95_ms)}, mean tokens in {_tokens(perf.mean_prompt_tokens)} "
        f"out {_tokens(perf.mean_completion_tokens)}, {spent}"
    )


MAX_UNSTABLE_LINES = 10


def _unstable_lines(stability: Stability | None) -> list[str]:
    """The unstable cases and what flipped, at most `MAX_UNSTABLE_LINES` of them."""
    if stability is None:
        return []
    lines = [
        f"  unstable {case.id}: {', '.join([*case.checks, *case.labels])}"
        for case in stability.unstable[:MAX_UNSTABLE_LINES]
    ]
    if (more := len(stability.unstable) - MAX_UNSTABLE_LINES) > 0:
        lines.append(f"  and {more} more unstable {'case' if more == 1 else 'cases'}")
    return lines


def _result_lines(result: FunctionResults) -> list[str]:
    lines = [_line(result), *_unstable_lines(result.summary.stability)]
    lines.append(_perf_line("system", result.summary.performance))
    if result.summary.judge is not None:
        lines.append(_perf_line("judge", result.summary.judge.performance))
    return lines


def _pairwise_lines(result: PairwiseResults) -> list[str]:
    first, second = result.versions
    counts = ", ".join(f"{outcome} {n}" for outcome, n in result.summary.outcomes.items())
    return [
        f"{result.function} {first} vs {second}: {counts}",
        _perf_line("pairwise", result.summary.performance),
    ]


@app.command("eval")
def eval_command(
    function: Annotated[
        list[str] | None,
        typer.Option("--function", help=f"Function to evaluate ({', '.join(EVAL_FUNCTIONS)})."),
    ] = None,
    config: Annotated[Path, typer.Option(help="Model config file.")] = DEFAULT_MODELS_PATH,
    datasets_dir: Annotated[Path, typer.Option(help="Directory with the datasets.")] = DATASETS_DIR,
    cassettes_dir: Annotated[Path, typer.Option(help="Recorded calls.")] = CASSETTES_DIR,
    results_dir: Annotated[Path, typer.Option(help="Where results go.")] = RESULTS_DIR,
    rubric: Annotated[Path, typer.Option(help="The judge rubric.")] = RUBRIC_PATH,
) -> None:
    """Replay the recorded run, apply every check and write results/."""
    functions = function or list(EVAL_FUNCTIONS)
    if unknown := [f for f in functions if f not in EVAL_FUNCTIONS]:
        raise typer.BadParameter(
            f"unknown function: {', '.join(unknown)} (known: {', '.join(EVAL_FUNCTIONS)})",
            param_hint="--function",
        )
    try:
        manifest = load_manifest(cassettes_dir)
        if manifest is None:
            typer.echo("pending first recorded run")
            return
        replay_config = load_config(config, env={})
        manifest.check_models(
            system=replay_config.models.system.model, judge=replay_config.models.judge.model
        )
        paths = _dataset_paths(datasets_dir)
        rag_path, triage_path = paths["rag"], paths["triage"]
        current = {p.name: file_sha256(p) for p in paths.values() if p.is_file()}
        if notice := manifest.dataset_notice(current):
            typer.echo(notice)
        judge_rubric = load_rubric(rubric)
        if notice := manifest.rubric_notice(judge_rubric.sha256):
            typer.echo(notice)
        rag_cases = load_rag(rag_path) if "rag" in functions else ()
        triage_cases = load_triage(triage_path) if "triage" in functions else ()
        store = CassetteStore(cassettes_dir)
        _echo_notices(store)
        client = ModelClient(Mode.REPLAY, store, replay_config)
        outcome = run(
            client,
            replay_config.models.system,
            judge=replay_config.models.judge,
            rubric=judge_rubric,
            mode=Mode.REPLAY,
            cassettes_dir=cassettes_dir,
            results_dir=results_dir,
            rag_cases=rag_cases,
            triage_cases=triage_cases,
            dataset_paths=paths,
            versions={f: v for f, v in manifest.prompt_versions.items() if f in EVAL_FUNCTIONS},
            repeats=manifest.repeats,
            stability_cases=manifest.stability_cases,
            judge_repeats=manifest.judge_repeats,
        )
    except (
        MissingRecording,
        CassetteError,
        ConfigError,
        DatasetError,
        RubricError,
        OSError,
    ) as exc:
        raise _fail(str(exc)) from None
    blocks = [_result_lines(result) for result in outcome.results]
    blocks += [_pairwise_lines(result) for result in outcome.pairwise]
    for block, path in zip(blocks, outcome.written, strict=True):
        for line in block:
            typer.echo(line)
        typer.echo(f"  wrote {path}")


@app.command("estimate")
def estimate_command(
    config: ConfigOption = DEFAULT_MODELS_PATH,
    datasets_dir: DatasetsOption = DATASETS_DIR,
    cassettes_dir: CassettesOption = CASSETTES_DIR,
    rubric: RubricOption = RUBRIC_PATH,
) -> None:
    """Print the call plan, the free-quota arithmetic and the estimated cost.

    Refuses (exit 1) when the cost of the calls still to record is above
    MAX_RUN_COST_USD (default 1.00). `:free` model ids cost $0.00.
    """
    try:
        loaded, inputs = _plan_inputs(config, datasets_dir, rubric)
        store = CassetteStore(cassettes_dir)
        _echo_notices(store)
        plan = full_plan(inputs, store)
        counts = count_plan(plan, store, loaded.models)
        for line in plan_lines(counts, loaded.models) + quota_lines(counts, loaded.models):
            typer.echo(line)
        estimate = estimate_remaining_cost(
            plan, store, loaded.models, transport=_network().transport
        )
    except PLAN_ERRORS as exc:
        raise _fail(str(exc)) from None
    for line in estimate.lines:
        typer.echo(f"  {line}")
    typer.echo(estimate.headline())
    limit = loaded.settings.max_run_cost_usd
    try:
        check_budget(estimate.usd, limit)
    except BudgetExceeded as exc:
        raise _fail(f"refused: {exc}") from None
    typer.echo(f"spend limit MAX_RUN_COST_USD: {format_usd(limit)}; the estimate is within it")


@app.command("status")
def status_command(
    config: ConfigOption = DEFAULT_MODELS_PATH,
    datasets_dir: DatasetsOption = DATASETS_DIR,
    cassettes_dir: CassettesOption = CASSETTES_DIR,
    rubric: RubricOption = RUBRIC_PATH,
) -> None:
    """Calls planned and recorded, the manifest, and the free quota left today.

    The free quota is read from GET /api/v1/key only when OPENROUTER_API_KEY
    is set; the key itself is never printed.
    """
    try:
        loaded, inputs = _plan_inputs(config, datasets_dir, rubric)
        manifest = load_manifest(cassettes_dir)
        store = CassetteStore(cassettes_dir)
        _echo_notices(store)
        plan = full_plan(inputs, store)
    except PLAN_ERRORS as exc:
        raise _fail(str(exc)) from None
    counts = count_plan(plan, store, loaded.models)
    typer.echo(f"cassettes: {cassettes_dir}")
    if manifest is None:
        typer.echo(f"manifest: absent, so the evaluation is {PENDING_RECORDED_RUN}")
    else:
        replay = count_plan(replay_plan(inputs, manifest, store), store, loaded.models)
        for line in _manifest_lines(manifest, replay, counts):
            typer.echo(line)
    for line in plan_lines(counts, loaded.models):
        typer.echo(line)
    if outside := len(unplanned(plan, store)):
        typer.echo(
            f"cassette entries outside the current plan: {outside} "
            "(llmeval prune lists them; make prune removes them)"
        )
    typer.echo(_quota_today(loaded))
    for line in quota_lines(counts, loaded.models):
        typer.echo(line)


def _manifest_lines(manifest: RunManifest, replay: PlanCounts, current: PlanCounts) -> list[str]:
    """What the manifest declares, and whether make eval can still replay it.

    `replay` counts the calls make eval would replay (`replay_plan`), and
    `current` the current plan. A manifest is stale when some of the replay
    calls are not in the cassettes: a config change gave them new keys, or
    `prune` removed them. Only `record` rewrites the manifest.
    """
    start, end = (t.astimezone(UTC) for t in (manifest.recorded_from, manifest.recorded_to))
    span = f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC"
    if replay.to_record:
        return [
            f"manifest: present but stale: written for a complete recording of "
            f"{manifest.planned_calls} calls, {span}",
            f"  make eval would replay {replay.total} calls, and {replay.to_record} of them are "
            "not in the cassettes: it fails until make record completes the current plan and "
            "rewrites the manifest",
        ]
    lines = [f"manifest: present: a complete recording of {manifest.planned_calls} calls, {span}"]
    if replay.total != manifest.planned_calls:
        lines.append(
            f"  the manifest counts {manifest.planned_calls} calls; make eval now replays "
            f"{replay.total}, all recorded"
        )
    if current.to_record:
        lines.append(
            "  the current plan is not fully recorded; make eval replays the manifest's plan"
        )
    return lines


def _quota_today(loaded: Config) -> str:
    """The live free-quota line: read from the key endpoint when a key is set."""
    if loaded.settings.api_key is None:
        return (
            "free quota today: OPENROUTER_API_KEY is not set, so the remaining free requests "
            "are not read"
        )
    try:
        quota = free_daily_quota(loaded.settings.api_key, transport=_network().transport)
    except OpenRouterError as exc:
        return f"free quota today: not read ({exc})"
    if quota is None:
        return "free quota today: the key endpoint did not report free_model_daily_requests"
    return (
        f"free quota today (GET /api/v1/key, read live): used {quota.used}, "
        f"limit {quota.limit}, remaining {quota.remaining}"
    )


@app.command("record")
def record_command(
    config: ConfigOption = DEFAULT_MODELS_PATH,
    datasets_dir: DatasetsOption = DATASETS_DIR,
    cassettes_dir: CassettesOption = CASSETTES_DIR,
    rubric: RubricOption = RUBRIC_PATH,
) -> None:
    """Record every planned call with the real API. Needs OPENROUTER_API_KEY.

    The budget guard runs first: nothing is sent above MAX_RUN_COST_USD.
    Recorded calls are skipped, so a rerun continues. System calls come
    first, then the judge calls planned from the recorded answers; each
    judge kind not yet proven is probed as soon as the answers it needs are
    recorded, so a judge config problem shows on the first day. The run
    keeps to the configured rpm and stops cleanly on the free daily quota.
    After a 429 without a reset time on a free model it waits 30 s, 60 s,
    120 s and 240 s, sending the call again after each wait, while the key
    has free requests left; then it stops. A request the API refuses is
    skipped (its other repeats are not sent) and listed at the end. The run
    stops on three different failures in a row of kinds not yet proven (a
    wrong model id or config), on twenty in a row with no response or a 5xx
    (an outage), and at once on HTTP 401 or 402. The manifest is written only
    when every planned call is recorded. Ctrl-C stops it cleanly, also
    during a wait. Exit codes: 0 complete, 75 stopped on the quota or a rate
    limit (rerun later), 130 stopped with Ctrl-C (rerun to continue), 1 an
    error or skipped calls (rerun to retry them).
    """
    net = _network()
    try:
        loaded, inputs = _plan_inputs(config, datasets_dir, rubric)
        outcome = record_all(
            inputs,
            loaded,
            cassettes_dir,
            _dataset_paths(datasets_dir),
            echo=typer.echo,
            transport=net.transport,
            limiter=net.limiter,
            sleep=net.sleep,
        )
    except BudgetExceeded as exc:
        raise _fail(f"refused: {exc}") from None
    except (MissingAPIKey, PlanMismatch, RecordLocked, *PLAN_ERRORS) as exc:
        raise _fail(str(exc)) from None
    if outcome.interrupted:
        raise typer.Exit(code=EXIT_INTERRUPTED)
    if outcome.stopped is not None:
        raise typer.Exit(code=EXIT_STOPPED)
    if not outcome.complete:
        raise typer.Exit(code=1)


@app.command("prune")
def prune_command(
    yes: Annotated[
        bool, typer.Option("--yes", help="Remove the listed entries. Without it nothing changes.")
    ] = False,
    config: ConfigOption = DEFAULT_MODELS_PATH,
    datasets_dir: DatasetsOption = DATASETS_DIR,
    cassettes_dir: CassettesOption = CASSETTES_DIR,
    rubric: RubricOption = RUBRIC_PATH,
) -> None:
    """List the recorded entries the current plan no longer has; --yes removes them.

    The plan is computed as record and status compute it. Prune refuses
    while a record run holds the lock, when the config, a dataset, the
    rubric or a cassette does not load, and while judge calls wait for their
    answers. With --yes each affected cassette file is rewritten atomically;
    a file left with no entries is deleted. The manifest is never changed.
    Needs no key and calls nothing.
    """
    try:
        _, inputs = _plan_inputs(config, datasets_dir, rubric)
        prune(inputs, cassettes_dir, apply=yes, echo=typer.echo)
    except (RecordLocked, PruneRefused, *PLAN_ERRORS) as exc:
        raise _fail(str(exc)) from None


def _current_inputs(manifest: RunManifest, datasets_dir: Path, rubric: Path) -> CurrentInputs:
    """sha256 of the prompts, datasets and rubric a replay would read now."""
    prompts = {
        (function, version): prompt_sha256(function, version)
        for function, versions in manifest.prompt_versions.items()
        if function in EVAL_FUNCTIONS
        for version in versions
    }
    paths = _dataset_paths(datasets_dir).values()
    datasets = {path.name: file_sha256(path) for path in paths if path.is_file()}
    return CurrentInputs(prompts=prompts, datasets=datasets, rubric=load_rubric(rubric).sha256)


@app.command("baseline")
def baseline_command(
    results_dir: ResultsOption = RESULTS_DIR,
    baseline: BaselineOption = BASELINE_PATH,
    cassettes_dir: CassettesOption = CASSETTES_DIR,
    datasets_dir: DatasetsOption = DATASETS_DIR,
    rubric: RubricOption = RUBRIC_PATH,
) -> None:
    """Write the baseline of the recorded run from the replay results.

    Reads results/ (one file per function and prompt version of the manifest,
    plus the pairwise comparison) and writes results/baseline.json: per case,
    whether it passed and which checks failed, and the key metrics the gate
    compares. Refuses (exit 1) and keeps the old baseline when a results file
    is missing, or when the results are stale: other models, repeats,
    judge_repeats or rubric than the manifest, or a prompt, dataset or rubric
    changed since the results were written. Run make eval first. Without a
    manifest it prints "pending first recorded run" and writes nothing.
    Needs no key and calls nothing.
    """
    try:
        manifest = load_manifest(cassettes_dir)
        if manifest is None:
            typer.echo(PENDING_RECORDED_RUN)
            return
        current = _current_inputs(manifest, datasets_dir, rubric)
        results = load_run_results(results_dir, manifest.prompt_versions)
        reasons = stale_reasons(results, manifest, current)
        if reasons:
            lines = "\n".join(f"  {reason}" for reason in reasons)
            raise _fail(
                f"refused: the results are stale:\n{lines}\nrun make eval, then make baseline"
            )
        built = build_baseline(results.functions, manifest, results.pairwise)
        path = write_baseline(built, baseline)
    except (BaselineError, CassetteError, PromptError, RubricError, OSError) as exc:
        raise _fail(str(exc)) from None
    for result in results.functions:
        cases = built.functions[result.function][result.version].cases.values()
        passed = sum(case.passed for case in cases)
        typer.echo(
            f"{result_label(result)}: {plural(len(cases), 'case')}: {passed} pass, "
            f"{len(cases) - passed} known failures"
        )
    for result in results.pairwise:
        consistent = PairwiseMetrics.of(result.summary).consistent
        share = "n/a" if consistent is None else f"{consistent:.1%}"
        typer.echo(f"{result_label(result)}: position consistency {share}")
    typer.echo(f"wrote {path}")


@app.command("gate")
def gate_command(
    results_dir: ResultsOption = RESULTS_DIR,
    baseline: BaselineOption = BASELINE_PATH,
    tolerances: Annotated[Path, typer.Option(help="The gate tolerances.")] = GATE_CONFIG_PATH,
    cassettes_dir: CassettesOption = CASSETTES_DIR,
) -> None:
    """Compare the key rates of the results with the baseline; exit 1 on a regression.

    Checks every function, prompt version and pairwise comparison in the
    baseline: every baseline case is present, all checks, each layer, new
    safety failures, triage accuracy, the stable share, the judge's rule
    pass rate and valid verdicts, and pairwise position consistency. A rate
    may drop by its tolerance in config/gate.yaml; any new safety failure
    fails. A replay equals the baseline exactly, so the tolerances matter
    for live (drift) results. Without a baseline: "pending first recorded
    run" when there is no manifest either, otherwise exit 1. Needs no key
    and calls nothing.
    """
    try:
        expected = load_baseline(baseline)
        if expected is None:
            if load_manifest(cassettes_dir) is None:
                typer.echo(PENDING_RECORDED_RUN)
                return
            raise _fail(f"no baseline in {baseline}: run make baseline")
        allowed = load_tolerances(tolerances)
        results = load_run_results(results_dir, expected.provenance.prompt_versions)
        report = gate(expected, results, allowed, outside=results_outside(results_dir, expected))
    except (BaselineError, GateError, CassetteError, OSError) as exc:
        raise _fail(str(exc)) from None
    modes = sorted({result.mode for result in (*results.functions, *results.pairwise)})
    typer.echo(f"gate: {results_dir} ({', '.join(modes)} results) against {baseline}")
    for line in format_report(report):
        typer.echo(line)
    if not report.passed:
        raise typer.Exit(code=1)


@app.command("sample")
def sample_command(
    results_dir: ResultsOption = RESULTS_DIR,
    cassettes_dir: CassettesOption = CASSETTES_DIR,
    sample: SampleOption = SAMPLE_PATH,
) -> None:
    """Write labels/sample.json: the judged answers the owner labels by hand.

    Takes every judged answer of repeat 0 the judge failed, plus judge-passed
    answers drawn with a fixed seed in strata of prompt version and category,
    30 in all (see llmeval.labels). Reads the RAG results of the manifest's
    prompt versions. Without a manifest it prints "pending first recorded
    run" and writes nothing. The same results always give the same sample.
    Needs no key and calls nothing.
    """
    try:
        manifest = load_manifest(cassettes_dir)
        if manifest is None:
            typer.echo(PENDING_RECORDED_RUN)
            return
        rag_versions = {"rag": manifest.prompt_versions.get("rag", ())}
        results = load_run_results(results_dir, rag_versions).functions
        built = build_sample(results)
        path = write_sample(built, sample)
    except (BaselineError, CassetteError, LabelError, OSError) as exc:
        raise _fail(str(exc)) from None
    runs = runs_by_answer(results)
    failed = sum(runs[item.ref].verdict is False for item in built.items)
    typer.echo(
        f"wrote {path}: {built.size} answers of repeat 0, {failed} the judge failed and "
        f"{built.size - failed} it passed, seed {built.seed}"
    )


@app.command("retrieval")
def retrieval_command(
    dataset: Annotated[Path, typer.Option(help="RAG dataset.")] = RAG_PATH,
    k: Annotated[int, typer.Option(min=1, help="Documents retrieved per question.")] = DEFAULT_K,
) -> None:
    """Show which expected documents the search returns for each RAG case."""
    try:
        cases = [case for case in load_rag(dataset) if case.expected_docs]
    except DatasetError as exc:
        raise _fail(str(exc)) from None
    complete = expected_total = found_total = 0
    for case in cases:
        retrieved = [hit.doc_id for hit in search(case.question, k)]
        check = retrieval_recall(case.expected_docs, retrieved)
        complete += check.passed
        expected_total += len(case.expected_docs)
        found_total += sum(doc in retrieved for doc in case.expected_docs)
        mark = "ok  " if check.passed else "MISS"
        typer.echo(f"{case.id}  {mark}  {check.detail}; got {', '.join(retrieved) or 'nothing'}")
    share = found_total / expected_total if expected_total else 0.0
    typer.echo(f"{found_total}/{expected_total} expected documents retrieved ({share:.1%})")
    typer.echo(f"{complete}/{len(cases)} cases with every expected document retrieved (k={k})")


def main() -> None:
    app()
