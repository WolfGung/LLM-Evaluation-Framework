"""The evaluation datasets: authored test cases with their expected behaviour.

The cases are test design written by a person, not model output. Triage
labels follow `datasets/triage-guideline.md`; every triage case names the
priority rule that decides it, and the loader refuses a case whose priority
does not match its rule.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

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


class DatasetError(ValueError):
    """A dataset file is broken or holds an invalid case."""


class _Case(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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


def load_triage(path: Path | str = TRIAGE_PATH) -> tuple[TriageCase, ...]:
    return _load(path, TriageCase)


def file_sha256(path: Path | str) -> str:
    """sha256 of a file's bytes, as stored in the run manifest."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
