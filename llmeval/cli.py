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
- `retrieval`: the retrieval layer over the RAG dataset, offline. BM25 is
  deterministic, so this needs no model and no recording.

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
from app.retrieval import search
from llmeval.callplan import (
    PlanInputs,
    count_plan,
    estimate_remaining_cost,
    full_plan,
    plan_lines,
    quota_lines,
)
from llmeval.cassettes import PENDING_RECORDED_RUN, CassetteError, CassetteStore, load_manifest
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
from llmeval.openrouter import MissingAPIKey, OpenRouterError
from llmeval.perf import Performance
from llmeval.pricing import BudgetExceeded, PricingError, check_budget, format_usd
from llmeval.quota import free_daily_quota
from llmeval.recording import EXIT_STOPPED, PlanMismatch, RecordLocked, record_all
from llmeval.results import RESULTS_DIR, FunctionResults, PairwiseResults
from llmeval.runner import CASSETTES_DIR, EVAL_FUNCTIONS, run, versions_of
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
        start, end = (t.astimezone(UTC) for t in (manifest.recorded_from, manifest.recorded_to))
        typer.echo(
            f"manifest: present: a complete recording of {manifest.planned_calls} calls, "
            f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC"
        )
        if counts.to_record:
            typer.echo(
                "  the current plan is not fully recorded; make eval replays the manifest's plan"
            )
    for line in plan_lines(counts, loaded.models):
        typer.echo(line)
    if outside := len(store) - counts.recorded:
        typer.echo(f"cassette entries outside the current plan: {outside}")
    typer.echo(_quota_today(loaded))
    for line in quota_lines(counts, loaded.models):
        typer.echo(line)


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
    first, then the judge calls planned from the recorded answers. The run
    keeps to the configured rpm and stops cleanly on the free daily quota.
    After a 429 without a reset time on a free model it waits 30 s, 60 s,
    120 s and 240 s, sending the call again after each wait, while the key
    has free requests left; then it stops. A request the API refuses is
    skipped (its other repeats are not sent) and listed at the end. The run
    stops on three different failures in a row of kinds not yet proven (a
    wrong model id or config), on twenty in a row with no response or a 5xx
    (an outage), and at once on HTTP 401 or 402. The manifest is written only
    when every planned call is recorded. Exit codes:
    0 complete, 75 stopped on the quota or a rate limit (rerun later), 1 an
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
    if outcome.stopped is not None:
        raise typer.Exit(code=EXIT_STOPPED)
    if not outcome.complete:
        raise typer.Exit(code=1)


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
