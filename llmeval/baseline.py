"""The baseline: what the recorded run measured, case by case.

An evaluation outcome is a measurement, not a code failure. The baseline
stores, per function, prompt version and case, whether the case passed and
which checks failed, plus the key metrics and where they came from. It is
built from a replay of the recorded run (stage 8 writes `results/baseline.json`
with `make baseline`), and the eval tests compare every replayed case with it:

- passed in the baseline and passes now: pass;
- passed in the baseline and fails now: a regression, the test fails;
- failed in the baseline and fails the same checks (or fewer): xfail naming
  the checks;
- failed in the baseline and fails a check the baseline did not: a
  regression, the test fails;
- failed in the baseline and passes now: a strict XPASS. The baseline is
  updated deliberately, never silently;
- not in the baseline: skipped as "pending baseline".
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from llmeval.cassettes import RunManifest
from llmeval.results import RESULTS_DIR, CaseRecord, FunctionResults

BASELINE_PATH = RESULTS_DIR / "baseline.json"
PENDING_BASELINE = "pending baseline"
UPDATE_HINT = "now passes; update the baseline deliberately (make baseline)"

Check = tuple[str, str]  # (layer, name)


class BaselineError(ValueError):
    """The baseline file is broken, or it would be built from the wrong results."""


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CaseBaseline(_Record):
    """One case: passed on every repeat, or the checks that failed on any repeat."""

    passed: bool
    failed_checks: tuple[Check, ...] = ()

    @model_validator(mode="after")
    def _passed_means_no_failed_checks(self) -> CaseBaseline:
        if self.passed == bool(self.failed_checks):
            raise ValueError("a case passed exactly when no check failed")
        return self


class Metrics(_Record):
    """Key rates of one function and version. `stable_share` comes with the stability layer."""

    all_checks: float | None
    layers: dict[str, float | None]
    accuracy: dict[str, float | None] | None = None
    retrieval_recall: float | None = None
    stable_share: float | None = None


class FunctionBaseline(_Record):
    metrics: Metrics
    cases: dict[str, CaseBaseline]


class Provenance(_Record):
    """The recorded run the baseline was measured on (from its manifest)."""

    recorded_from: datetime
    recorded_to: datetime
    models: dict[str, str]
    repeats: int
    prompt_versions: dict[str, tuple[str, ...]]
    stability_cases: tuple[str, ...] | None = None


class Baseline(_Record):
    schema_version: Literal[1] = 1
    provenance: Provenance
    functions: dict[str, dict[str, FunctionBaseline]]  # function -> version -> baseline

    def case(self, function: str, version: str, case_id: str) -> CaseBaseline | None:
        try:
            return self.functions[function][version].cases.get(case_id)
        except KeyError:
            return None


def failed_checks(record: CaseRecord) -> tuple[Check, ...]:
    """Checks that failed on any repeat, in order of first failure."""
    seen: dict[Check, None] = {}
    for run in record.runs:
        for check in run.checks:
            if not check.passed:
                seen.setdefault((check.layer, check.name), None)
    return tuple(seen)


def case_baseline(record: CaseRecord) -> CaseBaseline:
    failed = failed_checks(record)
    return CaseBaseline(passed=not failed, failed_checks=failed)


def build_baseline(results: Iterable[FunctionResults], manifest: RunManifest) -> Baseline:
    """The baseline of a replay of the recorded run described by `manifest`."""
    functions: dict[str, dict[str, FunctionBaseline]] = {}
    for result in results:
        if result.mode != "replay":
            raise BaselineError(
                f"a baseline is built from replay results only; {result.function} "
                f"{result.version} comes from a {result.mode} run"
            )
        summary = result.summary
        metrics = Metrics(
            all_checks=summary.all_checks.rate,
            layers={layer: rate.rate for layer, rate in summary.layers.items()},
            accuracy=summary.accuracy,
            retrieval_recall=summary.retrieval_recall,
        )
        cases = {record.id: case_baseline(record) for record in result.cases}
        functions.setdefault(result.function, {})[result.version] = FunctionBaseline(
            metrics=metrics, cases=cases
        )
    provenance = Provenance(
        recorded_from=manifest.recorded_from,
        recorded_to=manifest.recorded_to,
        models=dict(manifest.models),
        repeats=manifest.repeats,
        prompt_versions=dict(manifest.prompt_versions),
        stability_cases=manifest.stability_cases,
    )
    return Baseline(provenance=provenance, functions=functions)


def load_baseline(path: Path | str = BASELINE_PATH) -> Baseline | None:
    """The baseline in `path`, or None when there is none. A broken file raises."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return Baseline.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        raise BaselineError(f"{path.name}: not a valid baseline: {exc}") from None


def format_checks(checks: Iterable[Check]) -> str:
    return ", ".join(f"{layer}/{name}" for layer, name in checks)


def explain(record: CaseRecord) -> str:
    """The failed checks of a case, by repeat and layer, for a failure message."""
    lines = [f"{record.id}: {record.input}"]
    for run in record.runs:
        failed = [check for check in run.checks if not check.passed]
        for check in failed:
            lines.append(f"  repeat {run.repeat} [{check.layer}] {check.name}: {check.detail}")
        if any(check.layer == "retrieval" for check in failed):
            lines.append(
                f"  repeat {run.repeat}: retrieval miss: the search did not return an expected "
                "document, so later failures may not be generation failures"
            )
    return "\n".join(lines)


Outcome = Literal["pass", "fail", "xfail", "xpass", "pending"]


@dataclass(frozen=True)
class Verdict:
    outcome: Outcome
    message: str = ""


def compare(record: CaseRecord, expected: CaseBaseline | None) -> Verdict:
    """How a replayed case compares with its baseline entry (see the module docstring)."""
    if expected is None:
        return Verdict("pending", PENDING_BASELINE)
    current = case_baseline(record)
    if expected.passed:
        return Verdict("pass") if current.passed else Verdict("fail", explain(record))
    if current.passed:
        return Verdict("xpass", UPDATE_HINT)
    new = [check for check in current.failed_checks if check not in expected.failed_checks]
    if new:
        return Verdict(
            "fail",
            f"fails checks the baseline did not: {format_checks(new)}\n{explain(record)}",
        )
    return Verdict(
        "xfail", f"known failure in the baseline: {format_checks(current.failed_checks)}"
    )
