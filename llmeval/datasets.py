"""The evaluation datasets: authored test cases with their expected behaviour.

The cases are test design written by a person, not model output.

RAG cases (`datasets/rag.jsonl`):

- `answerable`: one document answers the question; `expected: answer`.
- `multi_doc`: the answer needs two or more documents; `expected: answer`.
- `unanswerable`: no document answers it; `expected: dont_know`, no facts and
  no expected documents.
- `safety`: attacks and traps, added in Task 5.

`required_facts` are short facts the answer must state. A fact may list
alternatives separated by `|` ("60 minutes|1 hour"); one of them is enough.
`expected_docs` are the documents that hold the answer; the retrieval layer
checks that the search returned them. `forbidden` are phrases the answer must
not contain (wrong or invented claims).

Triage labels follow `datasets/triage-guideline.md`; every triage case names
the priority rule that decides it, and the loader refuses a case whose
priority does not match its rule.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from app.triage import Category, Priority

DATASETS_DIR = Path("datasets")
RAG_PATH = DATASETS_DIR / "rag.jsonl"
TRIAGE_PATH = DATASETS_DIR / "triage.jsonl"

# Priority rules of the guideline, by id. The guideline's text must define
# exactly these (checked in tests/repo).
PRIORITY_RULES: dict[str, Priority] = {
    "U1": Priority.URGENT,
    "U2": Priority.URGENT,
    "H1": Priority.HIGH,
    "H2": Priority.HIGH,
    "H3": Priority.HIGH,
    "N1": Priority.NORMAL,
    "L1": Priority.LOW,
}

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
DocId = Annotated[str, Field(pattern=r"^kb-[a-z0-9]+(?:-[a-z0-9]+)*$")]
RagCategory = Literal["answerable", "multi_doc", "unanswerable", "safety"]
Expected = Literal["answer", "dont_know", "refuse"]


class DatasetError(ValueError):
    """A dataset file is broken or holds an invalid case."""


class _Case(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RagCase(_Case):
    """One question for the support assistant and the behaviour expected from it."""

    id: str = Field(pattern=r"^rag-\d{3}$")
    category: RagCategory
    question: Text
    expected: Expected
    required_facts: tuple[Text, ...] = ()
    expected_docs: tuple[DocId, ...] = ()
    forbidden: tuple[Text, ...] = ()
    note: str | None = None

    @model_validator(mode="after")
    def _expectation_fits_the_category(self) -> RagCase:
        if len(set(self.expected_docs)) != len(self.expected_docs):
            raise ValueError("expected_docs lists a document twice")
        if self.category in ("answerable", "multi_doc"):
            least = 2 if self.category == "multi_doc" else 1
            if self.expected != "answer":
                raise ValueError(f"{self.category} cases expect an answer")
            if len(self.expected_docs) < least or len(self.required_facts) < least:
                raise ValueError(
                    f"{self.category} cases need at least {least} expected documents "
                    f"and {least} required facts"
                )
        if self.category == "unanswerable":
            if self.expected != "dont_know":
                raise ValueError("unanswerable cases expect dont_know")
            if self.required_facts or self.expected_docs:
                raise ValueError("unanswerable cases have no required facts or expected documents")
        return self

    @property
    def fact_alternatives(self) -> tuple[tuple[str, ...], ...]:
        """Each required fact as its alternatives (split on `|`)."""
        return tuple(
            tuple(alt.strip() for alt in fact.split("|") if alt.strip())
            for fact in self.required_facts
        )


class TriageCase(_Case):
    """One support ticket and its labels from the guideline."""

    id: str = Field(pattern=r"^tri-\d{3}$")
    text: Text
    category: Category
    priority: Priority
    order_id: str | None = Field(pattern=r"^TS-\d{6}$")
    priority_rule: str
    note: str | None = None

    @model_validator(mode="after")
    def _priority_follows_its_rule(self) -> TriageCase:
        rule = PRIORITY_RULES.get(self.priority_rule)
        if rule is None:
            raise ValueError(f"unknown priority rule {self.priority_rule!r}")
        if rule is not self.priority:
            raise ValueError(
                f"rule {self.priority_rule} gives priority {rule}, "
                f"but the case says {self.priority}"
            )
        return self


def _rows(path: Path) -> Iterator[tuple[int, object]]:
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path.name}:{number}: not valid JSON: {exc.msg}") from None


def _load[CaseT: _Case](path: Path | str, model: type[CaseT]) -> tuple[CaseT, ...]:
    path = Path(path)
    cases: list[CaseT] = []
    seen: set[str] = set()
    for number, row in _rows(path):
        try:
            case = model.model_validate(row)
        except ValidationError as exc:
            raise DatasetError(f"{path.name}:{number}: invalid case: {exc}") from None
        case_id = case.id  # type: ignore[attr-defined]
        if case_id in seen:
            raise DatasetError(f"{path.name}:{number}: id {case_id} is used twice")
        seen.add(case_id)
        cases.append(case)
    return tuple(cases)


def load_rag(path: Path | str = RAG_PATH) -> tuple[RagCase, ...]:
    return _load(path, RagCase)


def load_triage(path: Path | str = TRIAGE_PATH) -> tuple[TriageCase, ...]:
    return _load(path, TriageCase)


def file_sha256(path: Path | str) -> str:
    """sha256 of a file's bytes, as stored in the run manifest."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
