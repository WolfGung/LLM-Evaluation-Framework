"""The baseline: what the recorded run measured, case by case.

An evaluation outcome is a measurement, not a code failure. The baseline
stores, per function, prompt version and case, whether the case passed and
which checks failed, plus the key metrics (also of each pairwise comparison)
and where they came from. `make baseline` builds it from the replay results
in `results/` and writes `results/baseline.json`; it refuses results that are
missing or stale (`stale_reasons`). The eval tests compare every replayed
case with it:

- passed in the baseline and passes now: pass;
- passed in the baseline and fails now: a regression, the test fails;
- failed in the baseline and fails the same checks (or fewer): xfail naming
  the checks;
- failed in the baseline and fails a check the baseline did not: a
  regression, the test fails;
- failed in the baseline and passes now: a strict XPASS. The baseline is
  updated deliberately, never silently;
- not in the baseline: skipped as "pending baseline".

The regression gate (`llmeval.gate`) compares the key metrics.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from llmeval.cassettes import RunManifest
from llmeval.config import JudgeRepeats
from llmeval.results import (
    RESULTS_DIR,
    AnyResults,
    CaseRecord,
    FunctionResults,
    PairwiseResults,
    PairwiseSummary,
    Summary,
    pairwise_path,
    results_path,
)
from llmeval.runner import EVAL_FUNCTIONS, pairs_of

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
    """Key rates of one function and version. `stable_share` is the stability
    layer's share of stable cases (None when no case repeats). For graded
    runs, `judge_rule_pass` is the rubric rule's pass rate over valid
    verdicts and `judge_valid` the share of valid verdicts (None without the
    judge)."""

    all_checks: float | None
    layers: dict[str, float | None]
    accuracy: dict[str, float | None] | None = None
    retrieval_recall: float | None = None
    stable_share: float | None = None
    judge_rule_pass: float | None = None
    judge_valid: float | None = None

    @classmethod
    def of(cls, summary: Summary) -> Metrics:
        judge = summary.judge
        return cls(
            all_checks=summary.all_checks.rate,
            layers={layer: rate.rate for layer, rate in summary.layers.items()},
            accuracy=summary.accuracy,
            retrieval_recall=summary.retrieval_recall,
            stable_share=summary.stability.stable_share if summary.stability else None,
            judge_rule_pass=judge.rule_pass.rate if judge else None,
            judge_valid=judge.valid.rate if judge else None,
        )


class PairwiseMetrics(_Record):
    """Key rates of one pairwise comparison of two prompt versions.

    - `consistent`: position consistency, the share of compared pairs whose
      two orders agree (one minus the inconsistent share);
    - `valid`: the share of asked pairs with two valid verdicts.
    """

    consistent: float | None
    valid: float | None

    @classmethod
    def of(cls, summary: PairwiseSummary) -> PairwiseMetrics:
        compared = summary.inconsistent
        consistent = (
            round((compared.total - compared.count) / compared.total, 4) if compared.total else None
        )
        return cls(consistent=consistent, valid=summary.valid.rate)


class FunctionBaseline(_Record):
    metrics: Metrics
    cases: dict[str, CaseBaseline]


class Provenance(_Record):
    """The recorded run the baseline was measured on (from its manifest)."""

    recorded_from: datetime
    recorded_to: datetime
    models: dict[str, str]
    repeats: int
    judge_repeats: JudgeRepeats
    prompt_versions: dict[str, tuple[str, ...]]
    stability_cases: tuple[str, ...] | None = None


class Baseline(_Record):
    schema_version: Literal[1] = 1
    provenance: Provenance
    functions: dict[str, dict[str, FunctionBaseline]]  # function -> version -> baseline
    # function -> "<first>-vs-<second>" -> metrics of that pairwise comparison
    pairwise: dict[str, dict[str, PairwiseMetrics]] = Field(default_factory=dict)

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


def pair_name(versions: tuple[str, str]) -> str:
    """How the baseline names a pairwise comparison: `v1-vs-v2`."""
    return f"{versions[0]}-vs-{versions[1]}"


def result_label(result: AnyResults) -> str:
    """`rag v1` for one function and version, `rag v1 vs v2` for a comparison."""
    if isinstance(result, PairwiseResults):
        return f"{result.function} {result.versions[0]} vs {result.versions[1]}"
    return f"{result.function} {result.version}"


def _replay_only(result: AnyResults) -> None:
    if result.mode != "replay":
        raise BaselineError(
            f"a baseline is built from replay results only; {result_label(result)} "
            f"comes from a {result.mode} run"
        )


def build_baseline(
    results: Iterable[FunctionResults],
    manifest: RunManifest,
    pairwise: Iterable[PairwiseResults] = (),
) -> Baseline:
    """The baseline of a replay of the recorded run described by `manifest`."""
    functions: dict[str, dict[str, FunctionBaseline]] = {}
    for result in results:
        _replay_only(result)
        cases = {record.id: case_baseline(record) for record in result.cases}
        functions.setdefault(result.function, {})[result.version] = FunctionBaseline(
            metrics=Metrics.of(result.summary), cases=cases
        )
    comparisons: dict[str, dict[str, PairwiseMetrics]] = {}
    for result in pairwise:
        _replay_only(result)
        comparisons.setdefault(result.function, {})[pair_name(result.versions)] = (
            PairwiseMetrics.of(result.summary)
        )
    provenance = Provenance(
        recorded_from=manifest.recorded_from,
        recorded_to=manifest.recorded_to,
        models=dict(manifest.models),
        repeats=manifest.repeats,
        judge_repeats=manifest.judge_repeats,
        prompt_versions=dict(manifest.prompt_versions),
        stability_cases=manifest.stability_cases,
    )
    return Baseline(provenance=provenance, functions=functions, pairwise=comparisons)


def _flat(value: Any, prefix: str = "") -> dict[str, Any]:
    """Nested dicts as one dict with dotted keys (`layers.safety`)."""
    if not isinstance(value, dict):
        return {prefix: value}
    flat: dict[str, Any] = {}
    for key, inner in value.items():
        flat.update(_flat(inner, f"{prefix}.{key}" if prefix else str(key)))
    return flat


def _show(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def _value_differences(label: str, committed: BaseModel, rebuilt: BaseModel) -> list[str]:
    first, second = _flat(committed.model_dump(mode="json")), _flat(rebuilt.model_dump(mode="json"))
    keys = [*first, *(key for key in second if key not in first)]
    return [
        f"{label} {key}: committed {_show(first.get(key))}, rebuilt {_show(second.get(key))}"
        for key in keys
        if first.get(key) != second.get(key)
    ]


def _verdict_text(case: CaseBaseline) -> str:
    return "passes" if case.passed else f"fails {format_checks(case.failed_checks)}"


def _both_sides[T](
    label: str, committed: Mapping[str, T], rebuilt: Mapping[str, T]
) -> tuple[list[str], list[tuple[str, T, T]]]:
    """Keys on one side only (as lines), and the keys on both with their values."""
    lines: list[str] = []
    pairs: list[tuple[str, T, T]] = []
    for key in [*committed, *(key for key in rebuilt if key not in committed)]:
        if key not in committed:
            lines.append(f"{label}{key}: not in the committed baseline")
        elif key not in rebuilt:
            lines.append(f"{label}{key}: not in the results")
        else:
            pairs.append((key, committed[key], rebuilt[key]))
    return lines, pairs


def baseline_differences(committed: Baseline, rebuilt: Baseline) -> list[str]:
    """What differs between a committed baseline and one rebuilt from the
    results, one line each: provenance fields, metrics, cases (a known failure
    added or removed, a case missing on either side) and pairwise metrics.
    Empty when they are equal."""
    lines = _value_differences("provenance", committed.provenance, rebuilt.provenance)
    for function in [*committed.functions, *rebuilt.functions.keys() - committed.functions.keys()]:
        missing, versions = _both_sides(
            f"{function} ",
            committed.functions.get(function, {}),
            rebuilt.functions.get(function, {}),
        )
        lines += missing
        for version, first, second in versions:
            label = f"{function} {version}"
            lines += _value_differences(f"{label} metric", first.metrics, second.metrics)
            missing_cases, cases = _both_sides(f"{label} ", first.cases, second.cases)
            lines += missing_cases
            lines += [
                f"{label} {case_id}: committed {_verdict_text(a)}; rebuilt {_verdict_text(b)}"
                for case_id, a, b in cases
                if a != b
            ]
    for function in [*committed.pairwise, *rebuilt.pairwise.keys() - committed.pairwise.keys()]:
        missing, pairs = _both_sides(
            f"{function} ", committed.pairwise.get(function, {}), rebuilt.pairwise.get(function, {})
        )
        lines += missing
        for name, first, second in pairs:
            label = f"{function} {name.replace('-vs-', ' vs ')} metric"
            lines += _value_differences(label, first, second)
    return lines


def write_baseline(baseline: Baseline, path: Path | str = BASELINE_PATH) -> Path:
    """Write the baseline as indented JSON; the same baseline gives the same bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(baseline.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class RunResults:
    """The results files of one run: one `FunctionResults` per function and
    version, and the pairwise comparisons."""

    functions: tuple[FunctionResults, ...]
    pairwise: tuple[PairwiseResults, ...] = ()


def _read[R: (FunctionResults, PairwiseResults)](path: Path, model: type[R]) -> R:
    try:
        return model.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        raise BaselineError(f"{path.name}: not valid results: {exc}") from None


def load_run_results(results_dir: Path | str, versions: Mapping[str, Sequence[str]]) -> RunResults:
    """Read the results of a run with these prompt versions from `results_dir`.

    One file per evaluated function and version; for a function graded by
    the judge, also one per compared version pair (`runner.pairs_of`). Every
    missing file is named in one `BaselineError`, and a file that is not
    valid results raises too.
    """
    missing: list[Path] = []
    functions: list[FunctionResults] = []
    pairwise: list[PairwiseResults] = []
    for function, chosen in versions.items():
        if function not in EVAL_FUNCTIONS:
            continue
        loaded = []
        for version in chosen:
            path = results_path(results_dir, function, version)
            if path.is_file():
                loaded.append(_read(path, FunctionResults))
            else:
                missing.append(path)
        functions += loaded
        if not any(result.judge_model is not None for result in loaded):
            continue
        for pair in pairs_of(chosen):
            path = pairwise_path(results_dir, function, pair)
            if path.is_file():
                pairwise.append(_read(path, PairwiseResults))
            else:
                missing.append(path)
    if missing:
        raise BaselineError(
            f"results missing: {', '.join(str(path) for path in missing)}: run make eval"
        )
    return RunResults(functions=tuple(functions), pairwise=tuple(pairwise))


@dataclass(frozen=True)
class CurrentInputs:
    """sha256 of what a replay reads now: each prompt by (function, version),
    each dataset by file name, and the rubric (`Rubric.sha256`)."""

    prompts: Mapping[tuple[str, str], str]
    datasets: Mapping[str, str]
    rubric: str | None


def stale_reasons(run: RunResults, manifest: RunManifest, current: CurrentInputs) -> list[str]:
    """Why `run` is not a replay of the recorded run with the current files.

    Empty when it is. Checked against the manifest: the system and judge
    models, repeats, judge_repeats and the rubric the judge layer was
    recorded with, and that the results come from a replay. Checked against
    the current files: the prompt, the dataset and the rubric hashes, so
    results written before one of them changed are stale. The manifest's
    dataset hashes are not compared: they are information, not a lock (see
    `RunManifest`), so expectations edited after the recording are re-checked
    by a new replay instead.
    """
    reasons: list[str] = []
    for result in run.functions:
        label = result_label(result)
        reasons += _common_reasons(result, label, manifest, current)
        if result.model != manifest.models["system"]:
            reasons.append(
                f"{label}: system model {result.model}, "
                f"the recording used {manifest.models['system']}"
            )
        if result.repeats != manifest.repeats:
            reasons.append(
                f"{label}: repeats {result.repeats}, the recording has {manifest.repeats}"
            )
        if result.prompt_sha256 != current.prompts.get((result.function, result.version)):
            reasons.append(f"{label}: the prompt changed since these results were written")
        if result.judge_model is not None and result.judge_repeats != manifest.judge_repeats:
            reasons.append(
                f"{label}: judge_repeats {result.judge_repeats}, "
                f"the recording has {manifest.judge_repeats}"
            )
    for result in run.pairwise:
        reasons += _common_reasons(result, result_label(result), manifest, current)
    return reasons


def _common_reasons(
    result: AnyResults, label: str, manifest: RunManifest, current: CurrentInputs
) -> list[str]:
    """The mode, the dataset and, for graded results, the judge model and rubric."""
    reasons = []
    if result.mode != "replay":
        reasons.append(
            f"{label}: comes from a {result.mode} run; a baseline is built from replay results only"
        )
    if result.dataset_sha256 != current.datasets.get(result.dataset):
        reasons.append(f"{label}: {result.dataset} changed since these results were written")
    if result.judge_model is None:
        return reasons
    if result.judge_model != manifest.models["judge"]:
        reasons.append(
            f"{label}: judge model {result.judge_model}, "
            f"the recording used {manifest.models['judge']}"
        )
    if result.rubric_sha256 != manifest.rubric_sha256:
        reasons.append(f"{label}: graded with another rubric than the recording")
    if result.rubric_sha256 != current.rubric:
        reasons.append(f"{label}: the rubric changed since these results were written")
    return reasons


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
