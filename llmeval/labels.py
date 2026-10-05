"""The owner's labels of judged answers: the sample, the labels file and the tool.

The judge is a measuring instrument too, so the owner labels a fixed sample
of the answers it graded, and `llmeval.agreement` compares the two.

- `labels/sample.json` (`build_sample`, written by `llmeval sample`): about
  30 judged answers of repeat 0. Every answer the judge failed is in it, so
  both classes are present, plus judge-passed answers drawn with a seed, in
  strata of prompt version and category. Each item names its answer by the
  cassette key of the system call that wrote it. The file says nothing about
  the judge's verdicts.
- `labels/human.jsonl`: one line per labelled answer (`HumanLabel`): the
  answer (case, version, repeat and cassette key), `pass` or `fail`, a
  comment, the labeler and the time in UTC. Only `llmeval label`
  (`make label`) writes it, and only the owner runs it. Tests use their own
  temporary files. Each label is appended as one complete line, or not at
  all (`append_label`), so Ctrl-C never leaves half a line.
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from llmeval.results import CaseRecord, FunctionResults, RunRecord

SAMPLE_PATH = Path("labels/sample.json")
LABELS_PATH = Path("labels/human.jsonl")
SAMPLE_SIZE = 30
SAMPLE_SEED = 2026
SAMPLE_SCHEMA_VERSION = 1
RAG_FUNCTION = "rag"
LABELER = "Pavel Zhukov Atum"

SAMPLE_RULE = (
    "Every judged answer of repeat 0 whose valid verdict fails the rubric's pass rule. "
    "Then judge-passed answers of repeat 0, drawn with Python's random.Random(seed) in "
    "strata of prompt version and category, to the sample size in all. Each stratum gets "
    "a share proportional to its judge-passed answers (largest remainder; a tie goes to "
    "the earlier stratum). Answers without a valid verdict are left out. The labelling "
    "order is shuffled with the same generator, so the judge failures are spread out."
)


class LabelError(ValueError):
    """The sample or the labels file is missing or broken, or cannot be built."""


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SampleItem(_Record):
    """One answer to label: the case, the prompt version, the repeat, the
    cassette key of the answer and the case's category."""

    case: str
    version: str
    repeat: int = Field(ge=0)
    answer_key: str
    category: str

    @property
    def ref(self) -> tuple[str, str, int]:
        return (self.case, self.version, self.repeat)


class Sample(_Record):
    """The content of `labels/sample.json`, items in labelling order."""

    schema_version: int = SAMPLE_SCHEMA_VERSION
    seed: int
    rule: str
    size: int
    items: list[SampleItem]

    @model_validator(mode="after")
    def _consistent(self) -> Sample:
        if self.size != len(self.items):
            raise ValueError(f"size is {self.size} but there are {len(self.items)} items")
        refs = [item.ref for item in self.items]
        if len(set(refs)) != len(refs):
            raise ValueError("an answer is in the sample twice")
        return self


@dataclass(frozen=True)
class AnswerRun:
    """One run of a RAG case, as the results hold it."""

    case: CaseRecord
    run: RunRecord

    @property
    def verdict(self) -> bool | None:
        """The judge's verdict by the rubric's pass rule; None when the run
        was not graded or the verdict is not valid."""
        return None if self.run.judge is None else self.run.judge.rule_pass


def runs_by_answer(results: Iterable[FunctionResults]) -> dict[tuple[str, str, int], AnswerRun]:
    """Every RAG run in `results`, by (case, version, repeat)."""
    return {
        (case.id, result.version, run.repeat): AnswerRun(case=case, run=run)
        for result in results
        if result.function == RAG_FUNCTION
        for case in result.cases
        for run in case.runs
    }


def stratum_quotas[K: Hashable](counts: Mapping[K, int], need: int) -> dict[K, int]:
    """How many of `need` items each stratum gives, in proportion to its size.

    Largest remainder: each stratum gets the whole part of its share, and the
    items left go to the largest fractional parts; a tie goes to the stratum
    that comes first in `counts`. When the strata hold no more than `need`
    items, all of them are taken.
    """
    total = sum(counts.values())
    if need >= total:
        return dict(counts)
    exact = {key: Fraction(need * n, total) for key, n in counts.items()}
    quotas = {key: math.floor(share) for key, share in exact.items()}
    left = need - sum(quotas.values())
    # sorted() is stable, so equal remainders keep the order of `counts`.
    for key in sorted(counts, key=lambda k: quotas[k] - exact[k])[:left]:
        quotas[key] += 1
    return quotas


def build_sample(
    results: Iterable[FunctionResults], *, size: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED
) -> Sample:
    """The label sample of these results, by `SAMPLE_RULE`.

    Strata go by prompt version (in the order of `results`), then category
    (alphabetical). Within a stratum the answers are sorted by case id
    before the draw, so the sample depends only on the results and the seed.
    """
    failures: list[SampleItem] = []
    passes: dict[tuple[str, str], list[SampleItem]] = {}
    versions: list[str] = []
    for result in results:
        if result.function != RAG_FUNCTION:
            continue
        versions.append(result.version)
        for case in result.cases:
            for run in case.runs:
                if run.repeat != 0 or run.judge is None or run.judge.rule_pass is None:
                    continue
                item = SampleItem(
                    case=case.id,
                    version=result.version,
                    repeat=run.repeat,
                    answer_key=run.call.key,
                    category=case.category,
                )
                if run.judge.rule_pass:
                    passes.setdefault((result.version, case.category), []).append(item)
                else:
                    failures.append(item)
    if not failures and not passes:
        raise LabelError("no judged answer with a valid verdict in the results: nothing to sample")
    if len(failures) > size:
        raise LabelError(f"{len(failures)} judge failures do not fit a sample of {size}")
    order = {version: at for at, version in enumerate(versions)}
    strata = dict(sorted(passes.items(), key=lambda kv: (order[kv[0][0]], kv[0][1])))
    quotas = stratum_quotas(
        {key: len(items) for key, items in strata.items()}, size - len(failures)
    )
    rng = random.Random(seed)
    chosen = list(failures)
    for key, items in strata.items():
        chosen += rng.sample(sorted(items, key=lambda item: item.case), quotas[key])
    rng.shuffle(chosen)
    return Sample(seed=seed, rule=SAMPLE_RULE, size=len(chosen), items=chosen)


def write_sample(sample: Sample, path: Path | str = SAMPLE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sample.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_sample(path: Path | str = SAMPLE_PATH) -> Sample:
    path = Path(path)
    if not path.is_file():
        raise LabelError(f"no sample in {path}: run llmeval sample")
    try:
        return Sample.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        raise LabelError(f"{path.name}: not a valid sample: {exc}") from None


class HumanLabel(_Record):
    """One line of `labels/human.jsonl`: the owner's label of one answer.

    `answer_key` is the cassette key of the answer when it was labelled. If
    the sample now names another key for the same case, version and repeat,
    the label is stale: it labels an answer that is no longer measured.
    """

    case: str
    version: str
    repeat: int = Field(ge=0)
    answer_key: str
    label: Literal["pass", "fail"]
    comment: str
    labeler: Literal["Pavel Zhukov Atum"]
    labeled_at: datetime

    @field_validator("labeled_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("labeled_at must be in UTC (ISO 8601 with Z or +00:00)")
        return value

    @property
    def ref(self) -> tuple[str, str, int]:
        return (self.case, self.version, self.repeat)


def describe(ref: tuple[str, str, int]) -> str:
    case, version, repeat = ref
    return f"{case} {version} repeat {repeat}"


def load_labels(path: Path | str = LABELS_PATH) -> list[HumanLabel]:
    """The labels in `path`, in file order; no file means no labels.

    Refuses a broken file and names the line: a line that is not a valid
    label, an empty line, a last line without its newline (it may be cut
    off), or two labels of the same answer.
    """
    path = Path(path)
    if not path.is_file():
        return []
    data = path.read_bytes()
    lines = data.split(b"\n")
    if lines[-1]:
        raise LabelError(
            f"{path} line {len(lines)} has no newline at the end: it may be cut off; "
            "check it, then end it with a newline or remove it"
        )
    labels: list[HumanLabel] = []
    seen: dict[tuple[tuple[str, str, int], str], int] = {}
    for number, line in enumerate(lines[:-1], start=1):
        if not line.strip():
            raise LabelError(f"{path} line {number} is empty")
        try:
            item = HumanLabel.model_validate_json(line)
        except ValidationError as exc:
            problem = exc.errors()[0]
            where = ".".join(str(part) for part in problem["loc"]) or "line"
            raise LabelError(
                f"{path} line {number}: not a valid label: {where}: {problem['msg']}"
            ) from None
        answer = (item.ref, item.answer_key)
        if answer in seen:
            raise LabelError(
                f"{path} lines {seen[answer]} and {number} label the same answer: "
                f"{describe(item.ref)}; keep one"
            )
        seen[answer] = number
        labels.append(item)
    return labels


Writer = Callable[[int, bytes], int]


def append_label(path: Path | str, label: HumanLabel, *, write: Writer = os.write) -> None:
    """Append `label` to `path` as one complete line, or leave the file as it was.

    The line goes out in one write call on a file opened for appending. If
    the write is short or interrupted (Ctrl-C), the file is cut back to its
    old length, so it never holds half a line. Refuses a file whose last line
    has no newline, because the label would be glued to it.
    """
    path = Path(path)
    line = (label.model_dump_json() + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        size = os.fstat(fd).st_size
        if size and os.pread(fd, 1, size - 1) != b"\n":
            raise LabelError(f"{path} does not end with a newline: check its last line first")
        written = 0
        try:
            written = write(fd, line)
        finally:
            if written != len(line):
                os.ftruncate(fd, size)
        if written != len(line):
            raise OSError(f"{path}: wrote {written} of {len(line)} bytes; the label was not saved")
        os.fsync(fd)
    finally:
        os.close(fd)
