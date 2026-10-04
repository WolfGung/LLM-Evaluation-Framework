"""The stability layer: do repeated runs of a case reach the same verdicts?

The system model runs at a non-zero temperature, so one case can pass on one
run and fail on the next. A case run more than once (see `repeats` and
`stability_cases` in `config/models.yaml`) is stable when every repeat has the
same verdict on every deterministic, reference and safety check, and, for
triage, the same predicted category and priority (two different wrong
categories fail the same check, but the answer still changed).

Left out on purpose:

- the retrieval layer: BM25 is deterministic and does not depend on the
  model, so it cannot flip;
- the judge layer: a second model's verdict measures the judge, not the
  system, and by default the judge grades repeat 0 only.

The layer reports the share of stable cases and, for each unstable case,
which checks and labels flipped, with the value on every repeat.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from app.triage import Category, Priority
from llmeval.checks import reference as ref

if TYPE_CHECKING:
    from llmeval.results import CaseRecord

STABILITY_LAYERS = ("deterministic", "reference", "safety")
# The triage labels compared across repeats, with the values they can take.
TRIAGE_LABELS: dict[str, tuple[str, ...]] = {
    "category": tuple(str(c) for c in Category),
    "priority": tuple(str(p) for p in Priority),
}


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UnstableCase(_Record):
    """One case whose repeats disagree.

    - `checks`: per flipped check (`layer/name`), the outcome on each repeat
      (None when the check did not run on that repeat);
    - `labels`: per flipped triage label, the predicted value on each repeat
      (`invalid` when the reply gave no usable value).
    """

    id: str
    category: str
    checks: dict[str, list[bool | None]]
    labels: dict[str, list[str]]


class Stability(_Record):
    """Stability over the cases run more than once.

    `repeated` cases ran two or more times; `stable` of them reached the same
    verdicts on every repeat; `stable_share` is `stable / repeated`.
    """

    repeated: int
    stable: int
    stable_share: float | None
    unstable: list[UnstableCase]


def _check_flips(record: CaseRecord) -> dict[str, list[bool | None]]:
    outcomes: list[dict[str, bool]] = [
        {
            f"{check.layer}/{check.name}": check.passed
            for check in run.checks
            if check.layer in STABILITY_LAYERS
        }
        for run in record.runs
    ]
    names = list(dict.fromkeys(name for run in outcomes for name in run))
    flips = {}
    for name in names:
        values = [run.get(name) for run in outcomes]
        if len(set(values)) > 1:
            flips[name] = values
    return flips


def _label_flips(function: str, record: CaseRecord) -> dict[str, list[str]]:
    if function != "triage":
        return {}
    flips = {}
    for label, values in TRIAGE_LABELS.items():
        predicted = [ref.predicted(run.output, label, values) for run in record.runs]
        if len(set(predicted)) > 1:
            flips[label] = predicted
    return flips


def stability(function: str, cases: Sequence[CaseRecord]) -> Stability | None:
    """The stability layer of one function and version, or None when no case repeats."""
    repeated = [record for record in cases if len(record.runs) > 1]
    if not repeated:
        return None
    unstable = []
    for record in repeated:
        checks = _check_flips(record)
        labels = _label_flips(function, record)
        if checks or labels:
            unstable.append(
                UnstableCase(id=record.id, category=record.category, checks=checks, labels=labels)
            )
    stable = len(repeated) - len(unstable)
    return Stability(
        repeated=len(repeated),
        stable=stable,
        stable_share=round(stable / len(repeated), 4),
        unstable=unstable,
    )
