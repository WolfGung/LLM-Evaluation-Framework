"""Checks: pure functions that look at one output and return a `CheckResult`.

Layers, in the order they are applied:

- retrieval (`retrieval.py`): did the search return the documents that hold
  the answer? A miss here explains a failure further down.
- deterministic (`deterministic.py`): rules a program can decide, such as
  valid JSON, citations of retrieved documents, forbidden phrases, length,
  declining to answer.
- reference (`reference.py`): comparison with the authored expectation, such
  as labels and required facts.
- judge (`judge.py`): a second model grades what rules cannot check, with a
  rubric: groundedness to the retrieved documents, helpfulness and tone. It
  never grades safety or required facts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one check: its name, pass or fail, and a short reason."""

    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = ["CheckResult"]
