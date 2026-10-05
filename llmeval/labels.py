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

The tool (`prepare_items`, `label_session`) is blind: for each answer it
shows the customer's question, the documents the assistant was given and the
answer, and nothing else. No judge verdict, score or reason, no check result,
no prompt version, category or case id. The documents come from the same
retrieval the runner uses (`app.assistant.prepare`), and the tool recomputes
the answer's cassette key from that prompt: a key that differs from the
sample means the prompt changed since the recording, so the answer is not
shown with documents it was not written from.
"""

from __future__ import annotations

import math
import os
import random
import re
import textwrap
import unicodedata
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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

from app import assistant
from app.prompting import PromptError
from app.retrieval import Hit
from llmeval.cassettes import CassetteStore, request_key
from llmeval.client import build_role_request
from llmeval.config import RoleConfig
from llmeval.results import CaseRecord, FunctionResults, RunRecord

SAMPLE_PATH = Path("labels/sample.json")
LABELS_PATH = Path("labels/human.jsonl")
SAMPLE_SIZE = 30
SAMPLE_SEED = 2026
SAMPLE_SCHEMA_VERSION = 1
LABELER = "Pavel Zhukov Atum"
# The one question the owner answers for each answer. It asks for the same
# three things as the judge's rubric (rubrics/judge.md), in one decision:
# grounded (groundedness), answers or says honestly that the documents do
# not cover it (helpfulness), polite (tone).
LABEL_QUESTION = (
    "Would you send this answer to the customer as is? Pass if it is grounded in the "
    "shown documents, answers the question (or says honestly that the documents do not "
    "cover it), and is polite. Fail otherwise."
)
# A one-line reminder of the question, right above each label prompt.
CRITERIA_REMINDER = (
    "Send as is? Grounded in the shown documents · answers, or says honestly they don't "
    "cover it · polite"
)
LABEL_PROMPT = "[{position}/{total}] Label (p pass, f fail, s skip, q quit)"
COMMENT_PROMPT = "Comment for {label} (optional; Enter saves, b goes back)"
BACK = "b"
# What the label prompt takes, and what a comment may not be on its own.
LABEL_KEYS = {"p": "p", "pass": "p", "f": "f", "fail": "f", "s": "s", "skip": "s", "q": "q"}
LABEL_KEYS["quit"] = "q"
LOOKS_LIKE_A_KEY = "That looks like a label key: type b to go back, or write a comment."
CONTROL_CHARACTER = (
    "The comment has a control character (an arrow or another special key?): type it again."
)

SAMPLE_RULE = (
    "Every judged answer of repeat 0 whose valid verdict fails the rubric's pass rule. "
    "Then judge-passed answers of repeat 0, drawn with Python's random.Random(seed) in "
    "strata of prompt version and category, to the sample size in all. Each stratum gets "
    "a share proportional to its judge-passed answers (largest remainder; a tie goes to "
    "the earlier stratum). Answers without a valid verdict are left out. The labelling "
    "order is shuffled with the same generator, so the judge failures are spread out, and "
    "shuffled again with it until the two versions of every case are at least 3 positions "
    "apart."
)
# How far apart in the labelling order the two versions of one case are kept,
# so the second answer is not read straight after the first; and how many
# shuffles may try before the sample is refused.
TWIN_GAP = 3
MAX_SHUFFLES = 1000


class LabelError(ValueError):
    """The sample or the labels file is missing or broken, or cannot be built."""


Ref = tuple[str, str, int]  # (case, version, repeat)
Answer = tuple[Ref, str]  # (ref, cassette key of the answer)


class Record(BaseModel):
    """A frozen record that refuses unknown fields; `llmeval.agreement` uses it too."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SampleItem(Record):
    """One answer to label: the case, the prompt version, the repeat, the
    cassette key of the answer and the case's category."""

    case: str
    version: str
    repeat: int = Field(ge=0)
    answer_key: str
    category: str

    @property
    def ref(self) -> Ref:
        return (self.case, self.version, self.repeat)

    @property
    def answer(self) -> Answer:
        """The answer itself: its case, version and repeat, and its cassette key."""
        return (self.ref, self.answer_key)


class Sample(Record):
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


def runs_by_answer(results: Iterable[FunctionResults]) -> dict[Ref, AnswerRun]:
    """Every RAG run in `results`, by (case, version, repeat)."""
    return {
        (case.id, result.version, run.repeat): AnswerRun(case=case, run=run)
        for result in results
        if result.function == assistant.RAG_FUNCTION
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
        if result.function != assistant.RAG_FUNCTION:
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
    _shuffle_twins_apart(chosen, rng)
    return Sample(seed=seed, rule=SAMPLE_RULE, size=len(chosen), items=chosen)


def _twins_apart(items: Sequence[SampleItem]) -> bool:
    """Whether the answers to one case (two versions) are `TWIN_GAP` or more apart."""
    seen: dict[str, int] = {}
    for at, item in enumerate(items):
        if item.case in seen and at - seen[item.case] < TWIN_GAP:
            return False
        seen[item.case] = at
    return True


def _shuffle_twins_apart(items: list[SampleItem], rng: random.Random) -> None:
    """Shuffle `items` in place with `rng`, again until `_twins_apart` holds.

    Every try is a fresh uniform shuffle, so the order stays random among the
    orders that keep twins apart, and the seed fixes it.
    """
    for _ in range(MAX_SHUFFLES):
        rng.shuffle(items)
        if _twins_apart(items):
            return
    raise LabelError(
        f"cannot keep the two versions of each case {TWIN_GAP} apart in {MAX_SHUFFLES} "
        f"shuffles of {len(items)} answers"
    )


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


class HumanLabel(Record):
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
    def ref(self) -> Ref:
        return (self.case, self.version, self.repeat)

    @property
    def answer(self) -> Answer:
        """The labelled answer: its case, version and repeat, and its cassette key."""
        return (self.ref, self.answer_key)


def describe(ref: Ref) -> str:
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
    seen: dict[Answer, int] = {}
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
        if item.answer in seen:
            raise LabelError(
                f"{path} lines {seen[item.answer]} and {number} label the same answer: "
                f"{describe(item.ref)}; keep one"
            )
        seen[item.answer] = number
        labels.append(item)
    return labels


LOCK_FILE = ".label.lock"


@contextmanager
def label_lock(labels_path: Path | str) -> Iterator[None]:
    """Hold `.label.lock` next to the labels file for a whole session, or refuse.

    Two sessions on one labels file would show the same answers twice and
    write two labels for one answer. The lock is advisory (`flock`, as for
    `make record`), released when the session ends or the process dies; the
    lock file stays and is git-ignored. It never creates the labels file.
    """
    try:
        import fcntl  # POSIX only, like the record lock
    except ImportError:
        raise LabelError("make label needs a POSIX system (Linux, macOS) to lock") from None
    path = Path(labels_path).parent / LOCK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LabelError(f"another make label session holds {path}: finish it first") from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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


# --- the labelling tool -------------------------------------------------------


@dataclass(frozen=True)
class LabelItem:
    """One sample answer as the tool shows it. `position` is its place in
    the sample (1-based)."""

    item: SampleItem
    position: int
    question: str
    documents: tuple[Hit, ...]
    answer: str


def prepare_items(
    sample: Sample,
    questions: Mapping[str, str],
    store: CassetteStore,
    role: RoleConfig,
) -> tuple[list[LabelItem], list[str]]:
    """The sample answers the tool can show, and why the others cannot be shown.

    For each item: the question from the dataset (`questions`, by case id),
    the prompt and documents from `app.assistant.prepare` (the runner's own
    path), and the answer from the cassettes. The cassette key is computed
    again from that prompt and the system `role`; when it differs from the
    sample's key, the answer was written from another prompt and is left out.
    """
    items: list[LabelItem] = []
    problems: list[str] = []
    for position, item in enumerate(sample.items, start=1):
        where = describe(item.ref)
        question = questions.get(item.case)
        if question is None:
            problems.append(f"{where}: the case is not in the dataset")
            continue
        try:
            messages, hits = assistant.prepare(question, item.version)
        except PromptError as exc:
            problems.append(f"{where}: {exc}")
            continue
        if request_key(build_role_request(messages, role), item.repeat) != item.answer_key:
            problems.append(
                f"{where}: the prompt now differs from the one that wrote the answer "
                "(prompt, documents, question or model config changed); the sample is stale"
            )
            continue
        entry = store.get(item.answer_key)
        if entry is None:
            problems.append(f"{where}: the answer is not in the cassettes")
            continue
        items.append(
            LabelItem(
                item=item,
                position=position,
                question=question,
                documents=hits,
                answer=entry.response.content,
            )
        )
    return items, problems


def labelled_answers(labels: Iterable[HumanLabel]) -> set[Answer]:
    """The answers these labels label (a label of an older answer is not one of them)."""
    return {label.answer for label in labels}


def unlabelled(items: Sequence[LabelItem], labels: Iterable[HumanLabel]) -> list[LabelItem]:
    """The items without a label of their current answer (same cassette key)."""
    done = labelled_answers(labels)
    return [item for item in items if item.item.answer not in done]


def labelled_count(sample: Sample, labels: Iterable[HumanLabel]) -> int:
    """Sample answers that have a label of their current answer."""
    done = labelled_answers(labels)
    return sum(item.answer in done for item in sample.items)


_LIST_ITEM = re.compile(r"(?:[-*]|\d+[.)])\s+")
MIN_WIDTH = 40


def wrap(text: str, width: int, indent: str = "") -> list[str]:
    """`text` wrapped to `width` columns, line by line, with `indent` before
    each line. Blank lines stay; a list item keeps a hanging indent; long
    words such as links are never broken."""
    width = max(width, MIN_WIDTH)
    lines: list[str] = []
    for line in text.splitlines() or [""]:
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        lead = indent + line[: len(line) - len(line.lstrip())]
        bullet = _LIST_ITEM.match(stripped)
        hang = " " * len(bullet.group(0)) if bullet else ""
        lines += textwrap.wrap(
            stripped,
            width=width,
            initial_indent=lead,
            subsequent_indent=lead + hang,
            break_long_words=False,
            break_on_hyphens=False,
        )
    return lines


def render_intro(
    *,
    total: int,
    labelled: int,
    todo: int,
    labels_path: Path,
    problems: Sequence[str],
    width: int,
) -> str:
    """The first screen: the labelling question, the keys and how to stop."""
    lines = [
        f"Label {total} answers of the Toolshop support assistant.",
        "",
        "For each answer, decide:",
        "",
        *wrap(LABEL_QUESTION, width, "    "),
        "",
        *wrap(
            "You see the customer's question, the documents the assistant was given, and "
            "its answer. Some questions come twice, answered by two versions of the "
            "assistant: label each answer on its own.",
            width,
        ),
        "",
        *wrap(
            "Keys: p pass, f fail, s skip for now, q quit. After p or f, type a comment "
            "if you like and press Enter, or type b to go back to the label.",
            width,
        ),
        "",
        *wrap(
            f"Labelled so far: {labelled} of {total}. To label now: {todo}. Each label is "
            f"saved at once to {labels_path}. Stop any time with q or Ctrl-C: saved labels "
            "are kept, and make label goes on where you stopped.",
            width,
        ),
        "",
        *wrap(
            f"To change a saved label, delete its line in {labels_path} and run make label "
            "again. Comments are published with the results: write them in English.",
            width,
        ),
    ]
    if problems:
        count = len(problems)
        lines += [
            "",
            f"{count} sample {'answer cannot' if count == 1 else 'answers cannot'} be shown:",
        ]
        for problem in problems:
            lines += wrap(problem, width, "  ")
    return "\n".join(lines)


def render_item(item: LabelItem, *, total: int, width: int) -> str:
    """One answer to label: the counter, the question, the documents, the
    question again and the answer, then the criteria in one line."""
    lines = ["=" * max(width, MIN_WIDTH), f"Item {item.position} of {total}", ""]
    lines += ["Question", *wrap(item.question, width, "    "), ""]
    if item.documents:
        lines += [f"Documents the assistant was given ({len(item.documents)})", ""]
        for number, hit in enumerate(item.documents, start=1):
            title, _, body = hit.text.partition("\n\n")
            lines += [f"  [{number}] {hit.doc_id}: {title}", *wrap(body, width, "      "), ""]
    else:
        lines += ["Documents the assistant was given: none matched the question", ""]
    answer = wrap(item.answer, width, "    ") if item.answer.strip() else ["    (empty answer)"]
    again = textwrap.wrap(
        f"Question: {item.question}",
        width=max(width, MIN_WIDTH),
        subsequent_indent=" " * len("Question: "),
        break_long_words=False,
        break_on_hyphens=False,
    )
    lines += [*again, "Answer", *answer, "", *wrap(CRITERIA_REMINDER, width)]
    return "\n".join(lines)


@dataclass(frozen=True)
class SessionOutcome:
    """How a labelling session ended: `done` (no items left), `quit` (q) or
    `interrupted` (Ctrl-C or the end of the input); `saved` labels this time."""

    saved: int
    reason: Literal["done", "quit", "interrupted"]


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _ask_label(ask: Callable[[str], str], echo: Callable[[str], None], prompt: str) -> str:
    """p, f, s or q; the whole words pass, fail, skip and quit count too."""
    while True:
        choice = LABEL_KEYS.get(ask(prompt).strip().lower())
        if choice is not None:
            return choice
        echo("Type p, f, s or q.")


def _ask_comment(ask: Callable[[str], str], echo: Callable[[str], None], value: str) -> str | None:
    """The comment for a `value` label, or None when the owner typed b to go back.

    A comment that is only a label key (p, pass, s, ...) is refused, and so
    is one with a control character, such as the escape sequence an arrow
    key leaves when the terminal has no line editing.
    """
    while True:
        comment = ask(COMMENT_PROMPT.format(label=value.upper())).strip()
        if comment.lower() == BACK:
            return None
        if comment.lower() in LABEL_KEYS:
            echo(LOOKS_LIKE_A_KEY)
        elif any(unicodedata.category(char) == "Cc" for char in comment):
            echo(CONTROL_CHARACTER)
        else:
            return comment


def label_session(
    items: Sequence[LabelItem],
    path: Path | str,
    *,
    total: int,
    ask: Callable[[str], str],
    echo: Callable[[str], None],
    now: Callable[[], datetime] = utc_now,
    width: int = 88,
) -> SessionOutcome:
    """Show each item, ask for its label and comment, and append the label.

    A label is written only after its comment is given, as one complete
    line (`append_label`); `b` at the comment prompt goes back to the same
    item's label. `s` moves on without writing, so the item comes back next
    time; `q` stops. Ctrl-C (KeyboardInterrupt) or the end of the
    input (EOFError) stops at once: every label saved before is kept, and
    the item on screen is not saved.
    """
    saved = 0
    try:
        for item in items:
            echo(render_item(item, total=total, width=width))
            prompt = LABEL_PROMPT.format(position=item.position, total=total)
            while True:
                choice = _ask_label(ask, echo, prompt)
                if choice == "q":
                    return SessionOutcome(saved=saved, reason="quit")
                if choice == "s":
                    echo("Skipped: make label shows it again next time.")
                    break
                value: Literal["pass", "fail"] = "pass" if choice == "p" else "fail"
                comment = _ask_comment(ask, echo, value)
                if comment is None:
                    continue  # b: back to this item's label
                append_label(
                    path,
                    HumanLabel(
                        case=item.item.case,
                        version=item.item.version,
                        repeat=item.item.repeat,
                        answer_key=item.item.answer_key,
                        label=value,
                        comment=comment,
                        labeler=LABELER,
                        labeled_at=now(),
                    ),
                )
                saved += 1
                echo(f"Saved: {value}.")
                echo("")
                break
    except (KeyboardInterrupt, EOFError):
        echo("")
        return SessionOutcome(saved=saved, reason="interrupted")
    return SessionOutcome(saved=saved, reason="done")
