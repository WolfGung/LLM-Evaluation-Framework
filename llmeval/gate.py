"""The regression gate: key rates of new results against the baseline.

`llmeval gate` (`make gate`) compares the results in a directory with
`results/baseline.json`, for every function, prompt version and pairwise
comparison the baseline has:

- the cases: every case of the baseline must be in the results (missing
  ones are named);
- the all-checks pass rate and each layer's pass rate;
- with a safety layer (RAG): every safety check a case fails that the
  baseline does not list for that case, as a count of new safety failures;
- triage: category and priority accuracy;
- the stable share, when cases repeat;
- with the judge: the rubric rule's pass rate and the share of valid
  verdicts;
- each pairwise comparison: position consistency and the share of valid
  pairs.

A rate fails when it drops below the baseline by more than its tolerance in
`config/gate.yaml`; a rise always passes. Any new safety failure fails,
whatever the rates say. A metric the baseline has and the results lack fails
as missing. The gate exits 1 when anything fails.

A replay of the recorded run is deterministic: it reproduces the results byte
for byte, so it always equals the baseline. The tolerances matter for live
(drift) runs, where the models answer again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llmeval.baseline import (
    Baseline,
    FunctionBaseline,
    Metrics,
    PairwiseMetrics,
    RunResults,
    failed_checks,
    pair_name,
)
from llmeval.results import FunctionResults

GATE_CONFIG_PATH = Path("config/gate.yaml")
SAFETY = "safety"

Tolerance = Annotated[float, Field(ge=0, le=1)]


class GateError(ValueError):
    """The tolerances file is broken, or a metric has no tolerance."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LayerTolerances(_Strict):
    retrieval: Tolerance
    deterministic: Tolerance
    reference: Tolerance
    safety: Tolerance
    judge: Tolerance


class AccuracyTolerances(_Strict):
    category: Tolerance
    priority: Tolerance


class JudgeTolerances(_Strict):
    rule_pass: Tolerance
    valid: Tolerance


class PairwiseTolerances(_Strict):
    consistent: Tolerance
    valid: Tolerance


class Tolerances(_Strict):
    """The largest allowed drop of each rate below the baseline, as a share
    (0.05 is 5 percentage points). Every key is required."""

    all_checks: Tolerance
    layers: LayerTolerances
    accuracy: AccuracyTolerances
    stable_share: Tolerance
    judge: JudgeTolerances
    pairwise: PairwiseTolerances


def load_tolerances(path: Path | str = GATE_CONFIG_PATH) -> Tolerances:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GateError(f"{path}: cannot read the gate tolerances: {exc.strerror}") from None
    except yaml.YAMLError as exc:
        raise GateError(f"{path.name}: not valid YAML: {exc}") from None
    try:
        return Tolerances.model_validate(data)
    except ValidationError as exc:
        raise GateError(f"{path.name}: not valid gate tolerances: {exc}") from None


Verdict = Literal["ok", "REGRESSION", "missing", "n/a"]
FAILING: frozenset[Verdict] = frozenset({"REGRESSION", "missing"})


@dataclass(frozen=True)
class Row:
    """One compared metric. `baseline` and `now` are rates, or counts when
    `count` is set. `allowed` is how far the metric may move in the direction
    `may` names: a rate may drop, the count of new safety failures may rise."""

    metric: str
    baseline: float | None
    now: float | None
    allowed: float
    verdict: Verdict
    count: bool = False
    may: Literal["drop", "rise"] = "drop"


def rate_row(metric: str, baseline: float | None, now: float | None, allowed: float) -> Row:
    """Compare one rate: it fails when it drops by more than `allowed`."""
    if baseline is None:
        verdict: Verdict = "n/a"
    elif now is None:
        verdict = "missing"
    elif round(baseline - now, 4) > allowed:
        verdict = "REGRESSION"
    else:
        verdict = "ok"
    return Row(metric, baseline, now, allowed, verdict)


def _missing(metric: str) -> Row:
    return Row(metric, None, None, 0, "missing")


@dataclass(frozen=True)
class GateReport:
    rows: tuple[Row, ...]
    # One line per new safety failure, for the reader of a failing gate.
    notes: tuple[str, ...] = ()

    @property
    def compared(self) -> tuple[Row, ...]:
        return tuple(row for row in self.rows if row.verdict != "n/a")

    @property
    def failed(self) -> tuple[Row, ...]:
        return tuple(row for row in self.rows if row.verdict in FAILING)

    @property
    def passed(self) -> bool:
        return not self.failed


def _layer_rows(label: str, base: Metrics, now: Metrics, tolerances: Tolerances) -> list[Row]:
    """All checks and each layer the baseline has."""
    rows = [rate_row(f"{label} all checks", base.all_checks, now.all_checks, tolerances.all_checks)]
    by_layer = tolerances.layers.model_dump()
    for layer, rate in base.layers.items():
        allowed = by_layer.get(layer)
        if allowed is None:
            raise GateError(f"{label}: no tolerance for the {layer} layer in the gate tolerances")
        rows.append(rate_row(f"{label} {layer} layer", rate, now.layers.get(layer), allowed))
    return rows


def _other_rows(label: str, base: Metrics, now: Metrics, tolerances: Tolerances) -> list[Row]:
    """Triage accuracy, the stable share and the judge, where they apply."""
    rows = []
    if base.accuracy is not None:
        now_accuracy = now.accuracy or {}
        for name in ("category", "priority"):
            rates = (base.accuracy.get(name), now_accuracy.get(name))
            allowed = getattr(tolerances.accuracy, name)
            rows.append(rate_row(f"{label} {name} accuracy", *rates, allowed))
    if base.stable_share is not None or now.stable_share is not None:
        allowed = tolerances.stable_share
        rows.append(rate_row(f"{label} stable share", base.stable_share, now.stable_share, allowed))
    if base.judge_valid is not None or now.judge_valid is not None:
        judge = tolerances.judge
        rule_pass = (base.judge_rule_pass, now.judge_rule_pass, judge.rule_pass)
        rows.append(rate_row(f"{label} judge rule pass", *rule_pass))
        valid = (base.judge_valid, now.judge_valid, judge.valid)
        rows.append(rate_row(f"{label} judge valid verdicts", *valid))
    return rows


MAX_NAMED_CASES = 5


def _coverage(
    label: str, expected: FunctionBaseline, result: FunctionResults
) -> tuple[Row, list[str]]:
    """The cases row: every baseline case must be in the results. The note
    names up to `MAX_NAMED_CASES` missing case ids."""
    present = {record.id for record in result.cases}
    missing = [case_id for case_id in expected.cases if case_id not in present]
    verdict: Verdict = "REGRESSION" if missing else "ok"
    row = Row(
        f"{label} cases",
        len(expected.cases),
        len(expected.cases) - len(missing),
        0,
        verdict,
        count=True,
    )
    if not missing:
        return row, []
    named = ", ".join(missing[:MAX_NAMED_CASES])
    if (more := len(missing) - MAX_NAMED_CASES) > 0:
        named += f" and {more} more"
    return row, [f"missing cases: {label} {named}"]


def new_safety_failures(expected: FunctionBaseline, result: FunctionResults) -> list[str]:
    """`<case> safety/<check>` for every safety check a case fails (on any
    repeat) that its baseline entry does not list."""
    new = []
    for record in result.cases:
        known = expected.cases.get(record.id)
        listed = set(known.failed_checks) if known else set()
        for layer, name in failed_checks(record):
            if layer == SAFETY and (layer, name) not in listed:
                new.append(f"{record.id} {layer}/{name}")
    return new


def gate(baseline: Baseline, run: RunResults, tolerances: Tolerances) -> GateReport:
    """Compare `run` with `baseline` (see the module docstring)."""
    rows: list[Row] = []
    notes: list[str] = []
    results = {(result.function, result.version): result for result in run.functions}
    for function, versions in baseline.functions.items():
        for version, expected in versions.items():
            label = f"{function} {version}"
            result = results.get((function, version))
            if result is None:
                rows.append(_missing(f"{label} results"))
                continue
            coverage, missing = _coverage(label, expected, result)
            rows.append(coverage)
            notes += missing
            now = Metrics.of(result.summary)
            rows += _layer_rows(label, expected.metrics, now, tolerances)
            if SAFETY in expected.metrics.layers:
                new = new_safety_failures(expected, result)
                verdict: Verdict = "REGRESSION" if new else "ok"
                metric = f"{label} new safety failures"
                rows.append(Row(metric, 0, len(new), 0, verdict, count=True, may="rise"))
                notes += [f"new safety failure: {label} {failure}" for failure in new]
            rows += _other_rows(label, expected.metrics, now, tolerances)
    comparisons = {(p.function, pair_name(p.versions)): p for p in run.pairwise}
    for function, pairs in baseline.pairwise.items():
        for name, expected_pair in pairs.items():
            label = f"{function} {name.replace('-vs-', ' vs ')}"
            result_pair = comparisons.get((function, name))
            if result_pair is None:
                rows.append(_missing(f"{label} results"))
                continue
            now_pair = PairwiseMetrics.of(result_pair.summary)
            allowed = tolerances.pairwise
            rows.append(
                rate_row(
                    f"{label} position consistency",
                    expected_pair.consistent,
                    now_pair.consistent,
                    allowed.consistent,
                )
            )
            rows.append(
                rate_row(f"{label} valid pairs", expected_pair.valid, now_pair.valid, allowed.valid)
            )
    return GateReport(rows=tuple(rows), notes=tuple(notes))


def _value(value: float | None, count: bool) -> str:
    if value is None:
        return "n/a"
    return str(int(value)) if count else f"{value:.2%}"


def _allowed(row: Row) -> str:
    amount = str(int(row.allowed)) if row.count else f"{row.allowed * 100:.2f} pp"
    return f"{row.may} {amount}"


def format_report(report: GateReport) -> list[str]:
    """The table (metric, baseline, now, allowed move, verdict), the notes and
    one verdict line."""
    header = ("metric", "baseline", "now", "allowed", "verdict")
    cells = [
        (
            row.metric,
            _value(row.baseline, row.count),
            _value(row.now, row.count),
            _allowed(row),
            row.verdict,
        )
        for row in report.rows
    ]
    widths = [max(len(line[i]) for line in (header, *cells)) for i in range(4)]

    def line(cell: tuple[str, ...]) -> str:
        metric, *numbers, verdict = cell
        columns = zip(numbers, widths[1:], strict=True)
        right = "  ".join(text.rjust(width) for text, width in columns)
        return f"{metric.ljust(widths[0])}  {right}  {verdict}".rstrip()

    lines = [line(header), *(line(cell) for cell in cells), *report.notes]
    compared, failed = len(report.compared), len(report.failed)
    noun = "metric" if compared == 1 else "metrics"
    if failed:
        lines.append(
            f"gate failed: {failed} of {compared} compared {noun} regressed beyond tolerance "
            "or are missing"
        )
    else:
        lines.append(f"gate passed: {compared} compared {noun} within tolerance")
    return lines
