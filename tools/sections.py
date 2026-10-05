"""The generated blocks other than the main table, rendered from results/.

Each block function takes the recorded run (`Recorded`) and returns parts
(`tools.formatting`): tables, headings and paragraphs. `tools.render` puts
them between their markers in README.md and docs/, and `tools.site` shows
some of them on the published page. Every number comes from the results
files, the run manifest, `results/judge-agreement.json`, the gate tolerances
in `config/gate.yaml` or the datasets; the same files give the same text.

- `pairwise`: the judge's choice between two prompt versions. Outcomes are
  counted over every case; position consistency only over the pairs the
  judge really compared (two different answers, two valid verdicts), so an
  identical or invalid pair never counts as consistent.
- `agreement`: the judge against the author's labels on the label sample:
  `pending human labels` with the sample until there are labels, then
  percent agreement, Cohen's kappa, the confusion and the disagreements,
  always with the note that the sample oversamples judge failures.
- `judge`: the judge as an instrument, per graded version: valid verdicts,
  the rubric rule's pass rate, how often its own `pass` disagrees with the
  rule, mean scores and the Spearman correlation of answer length with each
  score; then, from the pairwise comparison, how often it chose the answer
  shown first and the longer answer; and whether the system and the judge
  come from different vendors (the prefix of the model id).
- `safety`: each safety case, with its attack and the safe answer from
  `datasets/rag.jsonl`, and per version on how many runs every safety check
  passed, which safety checks failed, and which other checks failed.
- `cost`: calls, mean tokens, latency and cost of each kind of model call:
  the system's answers, the judge's grades, the pairwise questions. The
  rule-based layers call no model.
- `gate`: each tolerance of `config/gate.yaml` next to the widest gap
  between the same rate on single repeats of the recorded run, the noise a
  live run has to stay within.
- `scope`: the cases by function and category, how often each ran, what the
  judge graded and the recording's size and date.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from pydantic import ValidationError

from llmeval.agreement import AGREEMENT_FILE, PENDING_HUMAN_LABELS, AgreementReport
from llmeval.baseline import RunResults
from llmeval.callplan import plural
from llmeval.cassettes import RunManifest
from llmeval.checks.judge import CRITERIA
from llmeval.datasets import DatasetError, RagCase, load_rag
from llmeval.gate import GateError, Tolerances, load_tolerances
from llmeval.labels import LABELER
from llmeval.perf import Performance
from llmeval.pricing import format_usd
from llmeval.results import CaseRecord, FunctionResults, PairwiseResults, Rate, Summary, summarise
from tools.formatting import NONE, Heading, Part, RenderError, Table, decimal, percent, seconds

FREE_SUFFIX = ":free"
SAFETY = "safety"  # the safety layer, and the RAG category of the attack cases


@dataclass(frozen=True)
class Recorded:
    """The recorded run and where the blocks read the rest from."""

    manifest: RunManifest
    run: RunResults
    results_dir: Path
    gate_config: Path
    rag_dataset: Path

    def agreement(self) -> AgreementReport:
        """`results/judge-agreement.json`; missing or broken is an error, never pending."""
        path = self.results_dir / AGREEMENT_FILE
        if not path.is_file():
            raise RenderError(f"results missing: {path}: run make eval")
        try:
            return AgreementReport.model_validate_json(path.read_bytes())
        except ValidationError as exc:
            raise RenderError(f"{path.name}: not a valid agreement report: {exc}") from None

    def graded(self) -> list[FunctionResults]:
        return [result for result in self.run.functions if result.judge_model is not None]

    def tolerances(self) -> Tolerances:
        try:
            return load_tolerances(self.gate_config)
        except GateError as exc:
            raise RenderError(str(exc)) from None

    def rag_cases(self) -> dict[str, RagCase]:
        try:
            return {case.id: case for case in load_rag(self.rag_dataset)}
        except DatasetError as exc:
            raise RenderError(str(exc)) from None
        except OSError as exc:
            raise RenderError(f"{self.rag_dataset.name}: {exc.strerror}") from None


# --- pairwise ---------------------------------------------------------------------------

# Below this share of compared pairs with the same verdict in both orders, the
# pairwise block says the comparison picks no winner: too many verdicts
# depend on which answer came first.
PAIRWISE_TRUSTED_CONSISTENCY = Fraction(4, 5)
LOW_CONSISTENCY = (
    "With this many flips, the comparison says more about the judge's position bias than about "
    "the two prompts, so it picks no winner. The main table rests on the rules and the "
    "per-answer grades."
)

KINDS = {
    "same_position_a": "it chose the answer shown first in both orders",
    "same_position_b": "it chose the answer shown second in both orders",
    "tie_in_one_order": "it called a tie in one order and chose a side in the other",
}


def _outcome_rows(result: PairwiseResults) -> list[tuple[str, str]]:
    first, second = result.versions
    outcomes = result.summary.outcomes
    rows = [
        (f"{first} preferred in both orders", outcomes.get(first, 0)),
        (f"{second} preferred in both orders", outcomes.get(second, 0)),
        ("A tie in both orders", outcomes.get("tie", 0)),
        ("Inconsistent: the two orders disagree", outcomes.get("inconsistent", 0)),
    ]
    if outcomes.get("identical"):
        rows.append(("Identical answers, not compared", outcomes["identical"]))
    if outcomes.get("invalid"):
        rows.append(("An invalid verdict, not compared", outcomes["invalid"]))
    return [(label, str(count)) for label, count in rows]


def _flips(result: PairwiseResults) -> str:
    summary = result.summary
    flipped, compared = summary.inconsistent.count, summary.inconsistent.total
    if not flipped:
        return (
            f"In none of the {compared} compared pairs did the judge's preference change "
            "when the two answers swapped places."
        )
    kinds = [
        f"{plural(count, 'time')} {KINDS.get(kind, kind)}"
        for kind, count in summary.inconsistent_kinds.items()
        if count
    ]
    listed = ", ".join(kinds[:-1]) + f", and {kinds[-1]}" if len(kinds) > 1 else kinds[0]
    return (
        f"In {flipped} of the {compared} compared pairs, the judge's preference changed when "
        f"the two answers swapped places: {listed}. An inconsistent pair is never settled by "
        "picking one order."
    )


def pairwise_parts(result: PairwiseResults) -> list[Part]:
    """One pairwise comparison: the outcomes, position consistency and the flips."""
    first, second = result.versions
    summary = result.summary
    compared = summary.inconsistent.total
    consistent = compared - summary.inconsistent.count
    if compared:
        consistency = (
            f"Position consistency: {consistent} of {compared} compared pairs "
            f"({percent(consistent, compared)}) got the same verdict in both orders."
        )
        flips = [_flips(result)]
        if Fraction(consistent, compared) < PAIRWISE_TRUSTED_CONSISTENCY:
            flips.append(LOW_CONSISTENCY)
    else:
        consistency = "Position consistency: no pair was compared."
        flips = []
    return [
        f"The judge compared the first answers of {result.function} {first} and "
        f"{result.function} {second} case by case, asked twice with the order of the two "
        "answers swapped.",
        Table(
            header=(f"Outcome over {plural(summary.pairs, 'case')}", "Cases"),
            rows=tuple(_outcome_rows(result)),
            caption=f"The judge's choice between {result.function} {first} and {second}",
        ),
        consistency,
        *flips,
    ]


def pairwise(recorded: Recorded) -> list[Part]:
    """Every pairwise comparison of the run."""
    if not recorded.run.pairwise:
        return ["No pairwise comparison in this run."]
    return [part for result in recorded.run.pairwise for part in pairwise_parts(result)]


# --- agreement --------------------------------------------------------------------------


def _judge_failures(recorded: Recorded) -> int:
    """Valid verdicts that fail the rubric rule, over every graded version."""
    total = 0
    for result in recorded.graded():
        judge = result.summary.judge
        if judge is not None:
            total += judge.rule_pass.total - judge.rule_pass.passed
    return total


def _sample_sentence(report: AgreementReport, failures: int) -> str:
    sample = report.sample
    failed = (
        f"all {sample.judge_fail} answers the judge failed"
        if sample.judge_fail == failures
        else f"{sample.judge_fail} of the {failures} answers the judge failed"
    )
    labels = "labelled" if report.status == "complete" else "labels"
    return (
        f"{LABELER}, the author, {labels} {sample.size} judged answers by hand, blind to the "
        f"judge's verdict (make label): {failed} and {sample.judge_pass} it passed. The sample "
        "oversamples judge failures, so agreement on it is not the agreement over all answers."
    )


def agreement_parts(report: AgreementReport, failures: int) -> list[Part]:
    """The judge's agreement with the author's labels (see the module docstring)."""
    sample = _sample_sentence(report, failures)
    if report.status == PENDING_HUMAN_LABELS:
        return [PENDING_HUMAN_LABELS, sample]
    cells = report.confusion
    confusion = Table(
        header=("Judge's verdict", "Author: pass", "Author: fail"),
        rows=tuple(
            (
                f"Judge: {verdict}",
                str(cells[f"judge_{verdict}"]["human_pass"]),
                str(cells[f"judge_{verdict}"]["human_fail"]),
            )
            for verdict in ("pass", "fail")
        ),
        caption="The judge's verdict by the author's label, on the labelled sample answers",
    )
    disagreements = cells["judge_pass"]["human_fail"] + cells["judge_fail"]["human_pass"]
    kappa = decimal(report.kappa, 2) if report.kappa is not None else report.kappa_note
    listed = (
        f"{disagreements}, listed in results/judge-agreement.json with the judge's reasons "
        "and the author's comments"
        if disagreements
        else "0"
    )
    parts: list[Part] = [
        confusion,
        f"Percent agreement: {report.agreed} of {report.labelled} "
        f"({percent(report.agreed, report.labelled)}). Cohen's kappa: {kappa}. "
        f"Disagreements: {listed}.",
    ]
    if report.status == "complete":
        parts.append(f"Labelled: {report.labelled} of {report.sample.size} sample answers.")
    else:
        parts.append(
            f"Labelled so far: {report.labelled} of {report.sample.size} sample answers "
            f"({report.status})."
        )
    if unused := len(report.stale) + len(report.unjudged):
        parts.append(f"Labels not used: {unused}, listed in results/judge-agreement.json.")
    parts.append(sample)
    return parts


def agreement(recorded: Recorded) -> list[Part]:
    if not recorded.graded():
        return ["No answer in this run was graded by the judge."]
    return agreement_parts(recorded.agreement(), _judge_failures(recorded))


# --- judge -------------------------------------------------------------------------------


def _label(result: FunctionResults | PairwiseResults) -> str:
    if isinstance(result, PairwiseResults):
        first, second = result.versions
        return f"{result.function} {first} vs {second}"
    return f"{result.function} {result.version}"


def _of(passed: int, total: int) -> str:
    return f"{passed} of {total}"


def _mean(value: float | None) -> str:
    return NONE if value is None else decimal(value, 2)


def _times(count: int, total: int) -> str:
    return f"{count} of {total} times ({percent(count, total)})"


def _vendor(model: str) -> str:
    return model.split("/", 1)[0]


def _vendors(manifest: RunManifest) -> str:
    system, judge = manifest.models["system"], manifest.models["judge"]
    if _vendor(system) == _vendor(judge):
        return (
            f"The system model is {system} and the judge is {judge}, both from the same vendor "
            f"({_vendor(system)}): the judge may prefer answers written by its own model family."
        )
    return (
        f"The system model is {system} (vendor {_vendor(system)}) and the judge is {judge} "
        f"(vendor {_vendor(judge)}): different vendors, so the judge does not grade answers "
        "written by its own model family."
    )


def _length_sentences(graded: Sequence[FunctionResults]) -> list[str]:
    correlations = [
        (criterion, _label(result), judge.length_score_correlation[criterion].spearman)
        for criterion in CRITERIA
        for result in graded
        if (judge := result.summary.judge) is not None
    ]
    sentences = []
    if any(value is None for _, _, value in correlations):
        sentences.append(
            "A dash: no correlation can be computed, because every graded answer got the same "
            "score or fewer than three answers were graded."
        )
    positive: dict[str, list[str]] = {}
    for criterion, label, value in correlations:
        if value is not None and value > 0:
            positive.setdefault(criterion, []).append(f"{label} ({decimal(value, 2)})")
    if not positive:
        sentences.append(
            "No criterion has a positive correlation: longer answers did not get higher scores."
        )
    else:
        named = "; ".join(
            f"{criterion} in {' and '.join(where)}" for criterion, where in positive.items()
        )
        sentences.append(
            f"A positive correlation: {named}. It can mean the judge rewards length, or that "
            "the longer answers were more complete."
        )
    return sentences


def _pairwise_sentences(result: PairwiseResults) -> list[str]:
    summary = result.summary
    bias, longer = summary.position_bias, summary.longer_preferred
    if not bias.total:
        return [
            f"Position ({_label(result)}): no pair where the judge chose a side in both orders."
        ]
    position = (
        "Position: where the judge chose a side in both orders, it chose the answer shown first "
        f"{_times(bias.count, bias.total)}. Half would mean no lean."
    )
    if longer.total:
        length = (
            "Length: of those choices between answers of different lengths, it chose the longer "
            f"answer {_times(longer.count, longer.total)}."
        )
    else:
        length = "Length: no side choice was between answers of different lengths."
    return [position, length]


def judge(recorded: Recorded) -> list[Part]:
    """The judge as a measuring instrument (see the module docstring)."""
    graded = [r for r in recorded.graded() if r.summary.judge is not None]
    if not graded:
        return ["No answer in this run was graded by the judge."]

    def row(label: str, cell: Callable[[FunctionResults], str]) -> tuple[str, ...]:
        return (label, *(cell(result) for result in graded))

    def scores(result: FunctionResults):
        assert result.summary.judge is not None
        return result.summary.judge

    rows = [
        row("Answers graded", lambda r: str(scores(r).judged)),
        row("Valid verdicts", lambda r: _of(scores(r).valid.count, scores(r).valid.total)),
        row(
            "Pass by the rubric rule",
            lambda r: _of(scores(r).rule_pass.passed, scores(r).rule_pass.total),
        ),
        row(
            "The judge's own pass differs from the rule",
            lambda r: str(scores(r).pass_disagreements),
        ),
    ]
    rows += [
        row(f"Mean {criterion}", lambda r, c=criterion: _mean(scores(r).mean_scores.get(c)))
        for criterion in CRITERIA
    ]
    rows += [
        row(
            f"Answer length and {criterion} (Spearman)",
            lambda r, c=criterion: _mean(scores(r).length_score_correlation[c].spearman),
        )
        for criterion in CRITERIA
    ]
    parts: list[Part] = [
        Table(
            header=("Measure", *(_label(result) for result in graded)),
            rows=tuple(rows),
            caption="The judge's grades of each prompt version",
        ),
        *_length_sentences(graded),
    ]
    for result in recorded.run.pairwise:
        parts += _pairwise_sentences(result)
    parts.append(_vendors(recorded.manifest))
    return parts


# --- safety ------------------------------------------------------------------------------


def _attack(case: RagCase) -> str:
    return (case.attack_type or "").replace("_", " ")


def _safe_runs(record: CaseRecord) -> int:
    return sum(
        all(check.passed for check in run.checks if check.layer == SAFETY) for run in record.runs
    )


def _verdict(record: CaseRecord) -> str:
    safe, runs = _safe_runs(record), len(record.runs)
    if safe == runs:
        return f"passed {safe} of {plural(runs, 'run')}"
    return f"failed {runs - safe} of {plural(runs, 'run')}"


def _failed(record: CaseRecord, *, safety: bool) -> str:
    counts = Counter(
        check.name
        for run in record.runs
        for check in run.checks
        if not check.passed and (check.layer == SAFETY) == safety
    )
    return ", ".join(f"{name} ({plural(n, 'run')})" for name, n in counts.items())


def _result_line(label: str, record: CaseRecord) -> str:
    text = f"**{label}:** {_verdict(record)}."
    if failed := _failed(record, safety=True):
        text += f" Failed safety checks: {failed}."
    if other := _failed(record, safety=False):
        text += f" Other failed checks: {other}."
    return text


def safety(recorded: Recorded) -> list[Part]:
    """Each safety case: the attack, the safe answer and each version's result."""
    rag = [result for result in recorded.run.functions if result.function == "rag"]
    if not rag or not any(record.category == SAFETY for record in rag[0].cases):
        return ["No safety case in this run."]
    by_version = [
        (_label(result), {record.id: record for record in result.cases}) for result in rag
    ]
    ids = [record.id for record in rag[0].cases if record.category == SAFETY]
    cases = recorded.rag_cases()
    if missing := [case_id for case_id in ids if case_id not in cases]:
        raise RenderError(f"{missing[0]} is not in {recorded.rag_dataset.name}")
    rows = tuple(
        (
            case_id,
            _attack(cases[case_id]),
            *(_verdict(records[case_id]) for _, records in by_version),
        )
        for case_id in ids
    )
    parts: list[Part] = [
        Table(
            header=("Case", "Attack", *(label for label, _ in by_version)),
            rows=rows,
            numeric=False,
            caption="Each safety case and its result",
        ),
        "A run passes when every safety check passed on it. Other checks, such as required "
        "facts or length, are named under each case.",
    ]
    for case_id in ids:
        case = cases[case_id]
        first = by_version[0][1][case_id]
        parts += [
            Heading(3, f"{case_id}: {_attack(case)}"),
            f"**The message:** {first.input}",
            f"**The attack:** {case.attack}",
            f"**A safe answer:** {case.expected_behaviour}",
            *(_result_line(label, records[case_id]) for label, records in by_version),
        ]
    return parts


# --- cost --------------------------------------------------------------------------------


def _tokens(perf: Performance) -> str:
    if perf.mean_prompt_tokens is None or perf.mean_completion_tokens is None:
        return NONE
    text = f"{decimal(perf.mean_prompt_tokens, 0)} / {decimal(perf.mean_completion_tokens, 0)}"
    if perf.mean_reasoning_tokens:
        text += f" ({decimal(perf.mean_reasoning_tokens, 0)} reasoning)"
    return text


def _spent(perf: Performance, model: str | None) -> str:
    cost = perf.cost
    if cost.unknown_calls:
        return f"{format_usd(cost.total_usd)} known, {plural(cost.unknown_calls, 'call')} unknown"
    if cost.total_usd == 0 and model is not None and model.endswith(FREE_SUFFIX):
        return "$0.00 (free model)"
    return format_usd(cost.total_usd)


def _cost_row(label: str, perf: Performance, model: str | None) -> tuple[str, ...]:
    latency = f"{seconds(perf.latency.p50_ms)} / {seconds(perf.latency.p95_ms)}"
    return (label, str(perf.calls), _tokens(perf), latency, _spent(perf, model))


def _shared_gradings(recorded: Recorded) -> int:
    """Gradings counted in more than one version's row: one recording, counted again."""
    keys = Counter(
        run.judge.call.key
        for result in recorded.graded()
        for case in result.cases
        for run in case.runs
        if run.judge is not None
    )
    return sum(count - 1 for count in keys.values())


def cost(recorded: Recorded) -> list[Part]:
    """Calls, tokens, latency and cost of each kind of model call."""
    rows = [
        _cost_row(f"{_label(r)} answers (system)", r.summary.performance, r.model)
        for r in recorded.run.functions
    ]
    rows += [
        _cost_row(f"Judge grades of {_label(r)}", r.summary.judge.performance, r.judge_model)
        for r in recorded.run.functions
        if r.summary.judge is not None
    ]
    rows += [
        _cost_row(f"Pairwise questions, {_label(r)}", r.summary.performance, r.judge_model)
        for r in recorded.run.pairwise
    ]
    total = sum(int(row[1]) for row in rows)
    held = recorded.manifest.recorded_calls
    if total == held:
        added = f"The rows add up to {plural(total, 'call')}, as many as the recording holds."
    else:
        added = f"The rows add up to {plural(total, 'call')}; the recording holds {held}."
        shared = _shared_gradings(recorded)
        if shared and total - shared == held:
            verb = "is" if shared == 1 else "are"
            added += (
                f" On {plural(shared, 'case')} both prompt versions wrote the same answer, so "
                f"{plural(shared, 'recorded grading')} {verb} counted in both versions' rows."
            )
    return [
        Table(
            header=("Calls", "Count", "Mean tokens in / out", "Latency p50 / p95", "Cost"),
            rows=tuple(rows),
            caption="What each kind of model call cost in the recorded run",
        ),
        "The rule-based layers (retrieval, deterministic, reference, safety and stability) call "
        "no model: they read the answers above, so they add no calls and no cost.",
        added,
    ]


# --- gate --------------------------------------------------------------------------------

Pick = Callable[[Summary], Rate | None]


def _single_repeats(result: FunctionResults) -> list[Summary]:
    """The summary of each repeat on its own (repeat 0 of every case, then repeat 1, ...)."""
    repeats = sorted({run.repeat for case in result.cases for run in case.runs})
    summaries = []
    for repeat in repeats:
        cases = [
            case.model_copy(update={"runs": [run for run in case.runs if run.repeat == repeat]})
            for case in result.cases
        ]
        summaries.append(summarise(result.function, [case for case in cases if case.runs]))
    return summaries


def _spread(rates: Sequence[Rate | None]) -> Fraction | None:
    """The widest gap between rates measured on more than one repeat; None otherwise."""
    shares = [Fraction(rate.passed, rate.total) for rate in rates if rate and rate.total]
    return max(shares) - min(shares) if len(shares) > 1 else None


def _pp(value: Fraction | float) -> str:
    share = Fraction(value) if not isinstance(value, Fraction) else value
    return f"{decimal(Decimal(share.numerator) * 100 / Decimal(share.denominator))} pp"


def _widest(repeats: Sequence[tuple[str, list[Summary]]], pick: Pick) -> str:
    best: tuple[Fraction, str] | None = None
    for label, summaries in repeats:
        spread = _spread([pick(summary) for summary in summaries])
        if spread is not None and (best is None or spread > best[0]):
            best = (spread, label)
    return NONE if best is None else f"{_pp(best[0])} ({best[1]})"


def gate(recorded: Recorded) -> list[Part]:
    """Each gate tolerance next to the spread between single repeats of the run."""
    tolerances = recorded.tolerances()
    repeats = [(_label(r), _single_repeats(r)) for r in recorded.run.functions]
    layers = tolerances.layers.model_dump()
    rows: list[tuple[str, ...]] = [
        ("All checks", _pp(tolerances.all_checks), _widest(repeats, lambda s: s.all_checks))
    ]
    rows += [
        (
            f"{layer.capitalize()} layer",
            _pp(allowed),
            _widest(repeats, lambda s, layer=layer: s.layers.get(layer)),
        )
        for layer, allowed in layers.items()
    ]
    rows += [
        (
            f"{label.capitalize()} accuracy",
            _pp(allowed),
            _widest(repeats, lambda s, label=label: s.checks.get(f"{label}_match")),
        )
        for label, allowed in tolerances.accuracy.model_dump().items()
    ]
    rows += [
        ("Stable cases", _pp(tolerances.stable_share), NONE),
        (
            "Pass by the rubric rule",
            _pp(tolerances.judge.rule_pass),
            _judge_spread(repeats, "rule_pass"),
        ),
        ("Valid judge verdicts", _pp(tolerances.judge.valid), _judge_spread(repeats, "valid")),
        ("Pairwise position consistency", _pp(tolerances.pairwise.consistent), NONE),
        ("Pairs with two valid verdicts", _pp(tolerances.pairwise.valid), NONE),
        ("New safety failures", "none allowed", NONE),
    ]
    return [
        Table(
            header=("Gated rate", "Allowed drop", "Largest move between single repeats"),
            rows=tuple(rows),
            caption="The gate's tolerances and the noise between repeats of the recorded run",
        ),
        "Allowed drop: how far a rate may fall below the baseline before the gate fails "
        "(config/gate.yaml); a rise always passes. Largest move: the widest gap between the "
        "same rate on single repeats of the recorded run, over every function and version, "
        "with the version it was seen in.",
        "A dash: the rate is measured once per run, so single repeats cannot be compared. The "
        "judge grades the first run of each case only, the pairwise question is asked once per "
        "case, and the stable share needs every repeat at once.",
    ]


def _judge_spread(repeats: Sequence[tuple[str, list[Summary]]], measure: str) -> str:
    def pick(summary: Summary) -> Rate | None:
        judge = summary.judge
        if judge is None:
            return None
        if measure == "valid":
            return Rate(passed=judge.valid.count, total=judge.valid.total, rate=judge.valid.rate)
        return judge.rule_pass

    return _widest(repeats, pick)


# --- scope -------------------------------------------------------------------------------


def _times_ran(manifest: RunManifest) -> str:
    times = "once" if manifest.repeats == 1 else f"{manifest.repeats} times"
    if manifest.stability_cases is None:
        return f"Each case ran {times}."
    return f"{plural(len(manifest.stability_cases), 'case')} ran {times}, the others once."


def _dates(manifest: RunManifest) -> str:
    first, last = (t.astimezone(UTC).date() for t in (manifest.recorded_from, manifest.recorded_to))
    if first == last:
        return f"on {first.isoformat()} (UTC)"
    return f"from {first.isoformat()} to {last.isoformat()} (UTC)"


def _judged_sentence(recorded: Recorded) -> str:
    graded = [(r, r.summary.judge) for r in recorded.graded() if r.summary.judge is not None]
    if not graded:
        return ""
    counts = {judge.judged for _, judge in graded}
    if len(counts) == 1 and len(graded) > 1:
        function = graded[0][0].function
        text = f"The judge graded {plural(counts.pop(), 'answer')} of each {function} version"
    else:
        text = "The judge graded " + ", ".join(
            f"{plural(judge.judged, 'answer')} of {_label(r)}" for r, judge in graded
        )
    pairs = recorded.run.pairwise
    if len(pairs) == 1:
        text += f", and compared the two versions on {_compared(pairs[0])}"
    elif pairs:
        text += ", and compared " + ", ".join(f"{_label(r)} on {_compared(r)}" for r in pairs)
    return text + ". "


def _compared(result: PairwiseResults) -> str:
    """The cases the judge compared, and the ones it did not and why."""
    outcomes = result.summary.outcomes
    left_out = []
    if outcomes.get("identical"):
        left_out.append((outcomes["identical"], "identical answers"))
    if outcomes.get("invalid"):
        left_out.append((outcomes["invalid"], "an invalid verdict"))
    text = plural(result.summary.inconsistent.total, "case")
    if not left_out:
        return text
    named = " and ".join(f"{count} more with {what}" for count, what in left_out)
    verb = "was" if sum(count for count, _ in left_out) == 1 else "were"
    return f"{text} ({named} {verb} not compared)"


def scope(recorded: Recorded) -> list[Part]:
    """What the run covers: cases by category, repeats, judged answers, the recording."""
    rows = []
    seen: set[str] = set()
    for result in recorded.run.functions:
        if result.function in seen:
            continue
        seen.add(result.function)
        categories = Counter(record.category for record in result.cases)
        listed = ", ".join(f"{category} {n}" for category, n in categories.items())
        rows.append((result.function, str(len(result.cases)), listed))
    manifest = recorded.manifest
    return [
        Table(
            header=("Function", "Cases", "By category"),
            rows=tuple(rows),
            numeric=False,
            caption="The cases of the recorded run",
        ),
        f"{_times_ran(manifest)} {_judged_sentence(recorded)}Recorded {_dates(manifest)}: "
        f"{plural(manifest.recorded_calls, 'call')}.",
    ]
