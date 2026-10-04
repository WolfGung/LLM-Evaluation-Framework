"""The retrieval layer: did the search return the documents that hold the answer?

Retrieval is part of the system under test. Checking it separately tells a
retrieval miss (the right document never reached the model) apart from a
generation failure (the document was there and the answer still went wrong).
"""

from __future__ import annotations

from collections.abc import Sequence

from llmeval.checks import CheckResult


def retrieval_recall_value(expected: Sequence[str], retrieved: Sequence[str]) -> float | None:
    """Share of expected documents that were retrieved; None when none are expected."""
    if not expected:
        return None
    return sum(doc in retrieved for doc in expected) / len(expected)


def retrieval_recall(expected: Sequence[str], retrieved: Sequence[str]) -> CheckResult:
    """Passes when every expected document is among the retrieved ones."""
    missing = [doc for doc in expected if doc not in retrieved]
    found = len(expected) - len(missing)
    detail = f"retrieved {found} of {len(expected)} expected documents"
    if missing:
        detail += f"; missing {', '.join(missing)}"
    return CheckResult("retrieval_recall", not missing, detail)
