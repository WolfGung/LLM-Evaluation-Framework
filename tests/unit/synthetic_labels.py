"""Synthetic judged RAG results for the label sample and agreement tests.

Synthetic data: every case, answer, cassette key and judge grade built here
is made up for the test that asks for it, never a model output. Nothing is
written outside pytest's `tmp_path`.
"""

from __future__ import annotations

from llmeval.results import CaseRecord, FunctionResults, RunRecord
from tests.unit import synthetic_results as syn


def key_of(case_id: str, version: str, repeat: int) -> str:
    return f"{case_id}-{version}-{repeat}".ljust(64, "0")


def rag_case(case_id: str, version: str, category: str, *grades: str | None) -> CaseRecord:
    """A RAG case with one run per grade ("pass", "fail", "invalid" or None:
    not graded), each run with its own synthetic answer key."""
    runs = []
    for repeat, grade in enumerate(grades or ("pass",)):
        runs.append(
            RunRecord(
                repeat=repeat,
                output=f"Synthetic answer {case_id} {version} {repeat}.",
                call=syn.CALL.model_copy(update={"key": key_of(case_id, version, repeat)}),
                checks=[],
                judge=None if grade is None else syn.judge_record(grade),
            )
        )
    return CaseRecord(
        id=case_id, category=category, input="Synthetic question?", expected={}, runs=runs
    )


def population(version: str, counts: dict[str, tuple[int, int]], start: int = 1) -> FunctionResults:
    """Cases per category: (judge passes, judge failures), numbered from `start`."""
    cases = []
    number = start
    for category, (passes, failures) in counts.items():
        for grade in ["pass"] * passes + ["fail"] * failures:
            cases.append(rag_case(f"rag-{number:03d}", version, category, grade))
            number += 1
    return syn.function_results("rag", version, cases)


def two_versions() -> list[FunctionResults]:
    counts = {"answerable": (24, 1), "multi_doc": (6, 1), "unanswerable": (6, 2)}
    later = {"answerable": (24, 1), "multi_doc": (6, 1), "unanswerable": (7, 1)}
    return [population("v1", counts), population("v2", later)]
