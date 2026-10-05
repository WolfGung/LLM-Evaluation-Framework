"""Synthetic results for the baseline and gate tests.

Synthetic data: every check, judge grade, pairwise preference, result and
manifest built here is made up by the test that asks for it, never a model
output. Results are written only into pytest's `tmp_path`, never into
`results/`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from llmeval.cassettes import RunManifest
from llmeval.checks.judge import RUBRIC_PATH, load_rubric
from llmeval.datasets import file_sha256
from llmeval.results import (
    CallRecord,
    CaseRecord,
    CheckRecord,
    FunctionResults,
    JudgeRecord,
    PairwiseCaseRecord,
    PairwiseOrderRecord,
    PairwiseResults,
    RunRecord,
    summarise,
    summarise_pairwise,
)
from llmeval.runner import prompt_sha256

ROOT = Path(__file__).resolve().parents[2]
RUBRIC_SHA256 = load_rubric(ROOT / RUBRIC_PATH).sha256
TIME = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
SYSTEM_MODEL = "synthetic/system:free"
JUDGE_MODEL = "synthetic/judge:free"
CALL = CallRecord(
    key="k" * 64,
    model_used=SYSTEM_MODEL,
    latency_ms=1.0,
    prompt_tokens=1,
    completion_tokens=1,
    reasoning_tokens=0,
    cost_usd=0.0,
    cost_source="provider",
    finish_reason="stop",
    empty_reason=None,
    recorded_at=TIME,
)
# Checks per function, as "layer/name".
RAG_CHECKS = (
    "retrieval/retrieval_recall",
    "deterministic/cites_retrieved",
    "reference/required_facts",
    "safety/no_trap_leak",
)
TRIAGE_CHECKS = (
    "deterministic/valid_json",
    "reference/category_match",
    "reference/priority_match",
    "reference/order_id_match",
)
TRIAGE_LABELS = {"category": "order_status", "priority": "normal", "order_id": "TS-111111"}


Grade = Literal["pass", "fail", "invalid"]


def judge_record(grade: Grade) -> JudgeRecord:
    """A valid grade that passes or fails the rubric rule, or an invalid one."""
    if grade == "invalid":
        return JudgeRecord(
            scores=None,
            judge_pass=None,
            rule_pass=None,
            reasons=None,
            error="invalid_json",
            detail="synthetic invalid verdict",
            raw="not json",
            call=CALL,
        )
    passed = grade == "pass"
    score = 5 if passed else 2
    return JudgeRecord(
        scores={"groundedness": score, "helpfulness": score, "tone": score},
        judge_pass=passed,
        rule_pass=passed,
        reasons="Synthetic reasons.",
        error=None,
        detail=None,
        raw=None,
        call=CALL,
    )


def case_record(
    case_id: str,
    *failing_per_run: Iterable[str],
    category: str = "answerable",
    checks: Sequence[str] = RAG_CHECKS,
    judge: Grade | None = None,
) -> CaseRecord:
    """A case with one run per argument; each names the checks ("layer/name")
    that fail on that run. `judge` grades repeat 0 (None: not graded)."""
    runs = []
    for repeat, failing in enumerate(failing_per_run or ((),)):
        failing = set(failing)
        records = []
        for spec in checks:
            layer, name = spec.split("/")
            records.append(
                CheckRecord(
                    layer=layer, name=name, passed=spec not in failing, detail=f"synthetic {name}"
                )
            )
        graded = None
        if judge is not None and repeat == 0:
            graded = judge_record(judge)
            valid = judge != "invalid"
            records.append(
                CheckRecord(layer="judge", name="verdict_valid", passed=valid, detail="")
            )
            if valid:
                records.append(
                    CheckRecord(
                        layer="judge", name="groundedness", passed=judge == "pass", detail=""
                    )
                )
        runs.append(
            RunRecord(
                repeat=repeat, output="Synthetic answer.", call=CALL, checks=records, judge=graded
            )
        )
    expected = TRIAGE_LABELS if checks == TRIAGE_CHECKS else {}
    return CaseRecord(
        id=case_id, category=category, input="Synthetic question?", expected=expected, runs=runs
    )


def function_results(
    function: str,
    version: str,
    cases: Sequence[CaseRecord],
    *,
    dataset_sha256: str = "0" * 64,
    mode: str = "replay",
    repeats: int = 1,
    graded: bool | None = None,
) -> FunctionResults:
    """Results of one function and version; RAG results are graded by default."""
    graded = function == "rag" if graded is None else graded
    return FunctionResults(
        function=function,
        version=version,
        mode=mode,
        model=SYSTEM_MODEL,
        prompt_sha256=prompt_sha256(function, version),
        dataset=f"{function}.jsonl",
        dataset_sha256=dataset_sha256,
        repeats=repeats,
        judge_model=JUDGE_MODEL if graded else None,
        judge_repeats="first" if graded else None,
        rubric_sha256=RUBRIC_SHA256 if graded else None,
        summary=summarise(function, list(cases)),
        cases=list(cases),
    )


def _order(shown_as_a: str, other: str, preferred: str | None) -> PairwiseOrderRecord:
    winner = {"A": shown_as_a, "B": other, "tie": "tie"}.get(preferred) if preferred else None
    return PairwiseOrderRecord(
        shown_as_a=shown_as_a,
        preferred=preferred,
        winner=winner,
        reasons="Synthetic reasons." if preferred else None,
        error=None if preferred else "invalid_json",
        detail=None if preferred else "synthetic invalid verdict",
        raw=None if preferred else "not json",
        call=CALL,
    )


def pair_case(
    case_id: str, first: str | None, second: str | None, versions: tuple[str, str] = ("v1", "v2")
) -> PairwiseCaseRecord:
    """One compared case: `first` is the preference ("A", "B", "tie", or None
    for an invalid verdict) with the first version shown as A, `second` with
    the second version shown as A."""
    orders = [
        _order(versions[0], versions[1], first),
        _order(versions[1], versions[0], second),
    ]
    winners = {order.winner for order in orders}
    if None in winners:
        outcome = "invalid"
    elif len(winners) == 1:
        outcome = winners.pop()
    else:
        outcome = "inconsistent"
    return PairwiseCaseRecord(
        id=case_id,
        category="answerable",
        input="Synthetic question?",
        answers={versions[0]: "Synthetic one.", versions[1]: "Synthetic two."},
        outcome=outcome,
        orders=orders,
    )


def pairwise_results(
    cases: Sequence[PairwiseCaseRecord],
    *,
    versions: tuple[str, str] = ("v1", "v2"),
    dataset_sha256: str = "0" * 64,
    mode: str = "replay",
) -> PairwiseResults:
    return PairwiseResults(
        function="rag",
        versions=versions,
        mode=mode,
        judge_model=JUDGE_MODEL,
        rubric_sha256=RUBRIC_SHA256,
        dataset="rag.jsonl",
        dataset_sha256=dataset_sha256,
        summary=summarise_pairwise(list(cases), versions),
        cases=list(cases),
    )


def manifest(
    versions: dict[str, tuple[str, ...]] | None = None, *, repeats: int = 1
) -> RunManifest:
    return RunManifest(
        models={"system": SYSTEM_MODEL, "judge": JUDGE_MODEL},
        prompt_versions=versions or {"rag": ("v1", "v2"), "triage": ("v1",)},
        repeats=repeats,
        datasets={"rag.jsonl": "0" * 64, "triage.jsonl": "1" * 64},
        recorded_from=TIME,
        recorded_to=TIME,
        planned_calls=2,
        recorded_calls=2,
        judge_repeats="first",
        rubric_sha256=RUBRIC_SHA256,
    )


def write_datasets(directory: Path) -> dict[str, str]:
    """Synthetic dataset files (their bytes only matter for the hash); their sha256."""
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for function in ("rag", "triage"):
        path = directory / f"{function}.jsonl"
        path.write_text(f'{{"synthetic": "{function}"}}\n', encoding="utf-8")
        hashes[function] = file_sha256(path)
    return hashes
