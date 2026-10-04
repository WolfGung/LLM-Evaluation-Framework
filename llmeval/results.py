"""Result records, summaries and the results files.

`results/<function>-<version>.json` holds one `FunctionResults`: a record per
case and repeat with every check, and pass rates per layer. Only replay
results of a recorded run go to `results/`; live results go to the
git-ignored `results-live/`, so every number in `results/` can be reproduced
from the cassettes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.triage import Category, Priority
from llmeval.checks import reference as ref
from llmeval.checks.retrieval import retrieval_recall_value
from llmeval.client import CallResult

LAYERS = ("retrieval", "deterministic", "reference")
RESULTS_DIR = Path("results")
LIVE_RESULTS_DIR = Path("results-live")
SCHEMA_VERSION = 1

CATEGORY_LABELS = tuple(str(c) for c in Category)
PRIORITY_LABELS = tuple(str(p) for p in Priority)


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CheckRecord(_Record):
    layer: str
    name: str
    passed: bool
    detail: str


class CallRecord(_Record):
    key: str
    model_used: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float | None
    cost_source: str
    finish_reason: str | None
    empty_reason: str | None
    recorded_at: datetime

    @classmethod
    def from_call(cls, call: CallResult) -> CallRecord:
        return cls(
            key=call.key,
            model_used=call.model_used,
            latency_ms=call.latency_ms,
            prompt_tokens=call.usage.prompt_tokens,
            completion_tokens=call.usage.completion_tokens,
            reasoning_tokens=call.usage.reasoning_tokens,
            cost_usd=call.cost_usd,
            cost_source=call.cost_source,
            finish_reason=call.finish_reason,
            empty_reason=call.empty_reason,
            recorded_at=call.recorded_at,
        )


class RunRecord(_Record):
    """One repeat of one case: the output, the call and every check."""

    repeat: int
    output: str
    error: str | None = None
    retrieved: list[str] | None = None
    cited: list[str] | None = None
    call: CallRecord
    checks: list[CheckRecord]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


class CaseRecord(_Record):
    id: str
    category: str
    input: str
    expected: dict[str, Any]
    runs: list[RunRecord]


class Rate(_Record):
    passed: int
    total: int
    rate: float | None

    @classmethod
    def of(cls, outcomes: Iterable[bool]) -> Rate:
        outcomes = list(outcomes)
        passed = sum(outcomes)
        return cls(
            passed=passed,
            total=len(outcomes),
            rate=round(passed / len(outcomes), 4) if outcomes else None,
        )


class Summary(_Record):
    """Pass rates over all runs (every repeat of every case counts once).

    - `layers`: a run passes a layer when every check of that layer passes;
      only runs that have a check in the layer count.
    - `checks`: per check, over the runs it applies to.
    - `all_checks` and `by_category`: runs that pass every check.
    - RAG: `retrieval_recall` is expected documents retrieved / expected
      documents, over cases (the search does not depend on the model).
    - Triage: `accuracy` per label and `confusion` matrices (expected label by
      predicted label, with `invalid` for no usable value).
    """

    cases: int
    runs: int
    all_checks: Rate
    layers: dict[str, Rate]
    checks: dict[str, Rate]
    by_category: dict[str, Rate]
    retrieval_recall: float | None = None
    accuracy: dict[str, float | None] | None = None
    confusion: dict[str, dict[str, dict[str, int]]] | None = None


class FunctionResults(_Record):
    """The content of `results/<function>-<version>.json`."""

    schema_version: int = SCHEMA_VERSION
    function: str
    version: str
    mode: str
    model: str
    prompt_sha256: str
    dataset: str
    dataset_sha256: str
    repeats: int
    summary: Summary
    cases: list[CaseRecord]


def summarise(function: str, cases: Sequence[CaseRecord]) -> Summary:
    runs = [(case, run) for case in cases for run in case.runs]
    by_layer: dict[str, list[bool]] = defaultdict(list)
    by_check: dict[str, list[bool]] = {}
    by_category: dict[str, list[bool]] = {}
    for case, run in runs:
        layers: dict[str, bool] = {}
        for check in run.checks:
            layers[check.layer] = layers.get(check.layer, True) and check.passed
            by_check.setdefault(check.name, []).append(check.passed)
        for layer, passed in layers.items():
            by_layer[layer].append(passed)
        by_category.setdefault(case.category, []).append(run.passed)
    ordered_layers = [layer for layer in LAYERS if layer in by_layer] + sorted(
        set(by_layer) - set(LAYERS)
    )
    summary: dict[str, Any] = {
        "cases": len(cases),
        "runs": len(runs),
        "all_checks": Rate.of(run.passed for _, run in runs),
        "layers": {layer: Rate.of(by_layer[layer]) for layer in ordered_layers},
        "checks": {name: Rate.of(outcomes) for name, outcomes in by_check.items()},
        "by_category": {name: Rate.of(outcomes) for name, outcomes in by_category.items()},
    }
    if function == "rag":
        expected = [
            case.expected["expected_docs"] for case in cases if case.expected["expected_docs"]
        ]
        retrieved = [
            case.runs[0].retrieved or [] for case in cases if case.expected["expected_docs"]
        ]
        total = sum(len(docs) for docs in expected)
        found = sum(
            (retrieval_recall_value(docs, got) or 0) * len(docs)
            for docs, got in zip(expected, retrieved, strict=True)
        )
        summary["retrieval_recall"] = round(found / total, 4) if total else None
    if function == "triage":
        summary["accuracy"] = {
            label: Rate.of(by_check.get(f"{label}_match", [])).rate
            for label in ("category", "priority", "order_id")
        }
        summary["confusion"] = {
            label: ref.confusion_matrix(
                [
                    (case.expected[label], ref.predicted(run.output, label, labels))
                    for case, run in runs
                ],
                labels,
            )
            for label, labels in (("category", CATEGORY_LABELS), ("priority", PRIORITY_LABELS))
        }
    return Summary(**summary)


def results_path(results_dir: Path | str, function: str, version: str) -> Path:
    return Path(results_dir) / f"{function}-{version}.json"


def _write(results: FunctionResults, directory: Path | str) -> Path:
    path = results_path(directory, results.function, results.version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(results.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def write_results(results: FunctionResults, results_dir: Path | str) -> Path:
    """Write replay results of a recorded run (the only kind `results/` takes)."""
    if results.mode != "replay":
        raise ValueError(
            f"only replay results go to {results_dir}; got {results.mode} results "
            f"(live results go to {LIVE_RESULTS_DIR}/, a recording is replayed first)"
        )
    return _write(results, results_dir)


def write_live_results(results: FunctionResults, live_dir: Path | str) -> Path:
    """Write results of a live run to the git-ignored live directory."""
    if results.mode != "live":
        raise ValueError(f"only live results go to {live_dir}; got {results.mode} results")
    return _write(results, live_dir)
