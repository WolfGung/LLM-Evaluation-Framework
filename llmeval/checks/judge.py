"""The judge layer: a second model grades what rules cannot check.

The judge grades RAG answers on three criteria: groundedness to the retrieved
documents, helpfulness and tone. It never grades safety or required facts;
the rule-based layers own those.

The rubric lives in `rubrics/judge.md`. Its body is the judge's instructions
(criteria, anchors for scores 1, 3 and 5, the pass rule, the pairwise rule).
Its front matter gives the code the criteria and the minimum score per
criterion. The loader refuses a rubric whose prose pass rule and front matter
disagree, so the judge and the code apply one rule.

Judge calls go through the same model client as the system under test, in the
`judge` role of `config/models.yaml` (another model, temperature 0, a fixed
seed, structured output). They are recorded and replayed like any other call.
The prompt is built from the rubric, the question, the retrieved documents and
the answer only, so its request key is deterministic.

The verdict is requested with a strict JSON schema and validated with Pydantic
after parsing. An empty or invalid verdict is an invalid judgement: it is kept,
counted and reported, never dropped and never asked again.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from app.assistant import format_documents
from app.prompting import ChatModel
from app.retrieval import Hit
from app.triage import unwrap_reply
from llmeval.cassettes import CallTag
from llmeval.checks import CheckResult
from llmeval.client import CallResult
from llmeval.config import RoleConfig

RUBRIC_PATH = Path("rubrics/judge.md")
CRITERIA = ("groundedness", "helpfulness", "tone")
SCORES = (1, 2, 3, 4, 5)
ANCHORS = (5, 3, 1)

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)
_KEYS = frozenset({"criteria", "pass_rule"})
_PASS_RULE_SECTION = "Pass rule"


class RubricError(ValueError):
    """The rubric file is missing, broken, or its prose and front matter disagree."""


@dataclass(frozen=True)
class Rubric:
    """A loaded rubric.

    - `text`: the body the judge reads (the front matter is left out);
    - `criteria`: the graded criteria, in order;
    - `pass_rule`: the minimum score per criterion for a pass;
    - `sha256`: of the file's text, so results can name the rubric they used.
    """

    text: str
    criteria: tuple[str, ...]
    pass_rule: Mapping[str, int]
    sha256: str

    def short_of(self, scores: Mapping[str, int]) -> tuple[str, ...]:
        """The criteria whose score is below the minimum, in rubric order."""
        return tuple(name for name in self.criteria if scores[name] < self.pass_rule[name])

    def passes(self, scores: Mapping[str, int]) -> bool:
        """The pass rule: every criterion at least its minimum."""
        return not self.short_of(scores)


def _sections(body: str) -> dict[str, str]:
    """`## Title` sections of the body, by title."""
    parts = re.split(r"^## +(.+?) *$", body, flags=re.MULTILINE)
    return {title: text for title, text in zip(parts[1::2], parts[2::2], strict=True)}


def _check_front_matter(meta: Any, name: str) -> dict[str, int]:
    if not isinstance(meta, dict):
        raise RubricError(f"{name}: front matter is not a mapping")
    if unknown := sorted(set(meta) - _KEYS):
        raise RubricError(f"{name}: unknown front matter key: {', '.join(unknown)}")
    if tuple(meta.get("criteria") or ()) != CRITERIA:
        raise RubricError(f"{name}: criteria must be {', '.join(CRITERIA)}, in that order")
    rule = meta.get("pass_rule")
    if not isinstance(rule, dict) or set(rule) != set(CRITERIA):
        raise RubricError(
            f"{name}: pass_rule must give a minimum score for each of {', '.join(CRITERIA)}"
        )
    for criterion, minimum in rule.items():
        # bool is an int in Python; a minimum of `true` is a typo, not a 1.
        if type(minimum) is not int or minimum not in SCORES:
            raise RubricError(
                f"{name}: pass_rule.{criterion} must be a whole number from 1 to 5, got {minimum!r}"
            )
    return {criterion: rule[criterion] for criterion in CRITERIA}


def _check_body(body: str, rule: Mapping[str, int], name: str) -> None:
    sections = _sections(body)
    for criterion in CRITERIA:
        title = criterion.capitalize()
        if title not in sections:
            raise RubricError(f"{name}: no '## {title}' section")
        for score in ANCHORS:
            if not re.search(rf"^- {score}: \S", sections[title], flags=re.MULTILINE):
                raise RubricError(f"{name}: {title} has no anchor for score {score}")
    if _PASS_RULE_SECTION not in sections:
        raise RubricError(f"{name}: no '## {_PASS_RULE_SECTION}' section")
    prose = " ".join(sections[_PASS_RULE_SECTION].split())
    for criterion, minimum in rule.items():
        stated = re.findall(rf"\b{criterion} is at least (\d)\b", prose)
        if not stated:
            raise RubricError(
                f"{name}: the pass rule does not say '{criterion} is at least {minimum}'"
            )
        for value in stated:
            if int(value) != minimum:
                raise RubricError(
                    f"{name}: the pass rule says {criterion} is at least {value}; "
                    f"the front matter says {minimum}"
                )


def parse_rubric(source: str, *, name: str) -> Rubric:
    """Parse and check a rubric; `name` is used in error messages."""
    text = source.replace("\r\n", "\n")
    match = _FRONT_MATTER.match(text)
    if match is None:
        raise RubricError(f"{name}: no front matter between '---' lines")
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise RubricError(f"{name}: front matter is not valid YAML: {exc}") from None
    rule = _check_front_matter(meta, name)
    body = match.group(2).strip()
    _check_body(body, rule, name)
    return Rubric(
        text=body,
        criteria=CRITERIA,
        pass_rule=rule,
        sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )


def load_rubric(path: Path | str = RUBRIC_PATH) -> Rubric:
    path = Path(path)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RubricError(f"cannot read the rubric {path}: {exc.strerror}") from None
    return parse_rubric(source, name=path.name)


# --- the verdict --------------------------------------------------------------


def _whole_score(value: object) -> object:
    # Literal[1..5] alone accepts 4.0 and true (true == 1 in Python).
    if type(value) is not int:
        raise ValueError("a score must be a whole number from 1 to 5")
    return value


Score = Annotated[Literal[1, 2, 3, 4, 5], BeforeValidator(_whole_score)]


class _Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("reasons", check_fields=False)
    @classmethod
    def _reasons_have_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reasons must not be blank")
        return value


class JudgeVerdict(_Verdict):
    """The judge's grade of one answer.

    The schema has no length, pattern or range keywords, so strict structured
    output modes accept it; scores are an enum of 1 to 5.
    """

    groundedness: Score
    helpfulness: Score
    tone: Score
    passed: bool = Field(alias="pass")
    reasons: str

    @property
    def scores(self) -> dict[str, int]:
        return {criterion: getattr(self, criterion) for criterion in CRITERIA}


VERDICT_SCHEMA_NAME = "judge_verdict"


def _response_format(name: str, model: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": model.model_json_schema()},
    }


def verdict_schema() -> dict[str, Any]:
    return JudgeVerdict.model_json_schema()


def verdict_format() -> dict[str, Any]:
    """`response_format` for a verdict: the strict JSON schema of `JudgeVerdict`."""
    return _response_format(VERDICT_SCHEMA_NAME, JudgeVerdict)


Preference = Literal["A", "B", "tie"]


class PairwiseVerdict(_Verdict):
    """The judge's choice between two answers shown as A and B."""

    preferred: Preference
    reasons: str


PAIRWISE_SCHEMA_NAME = "pairwise_verdict"


def pairwise_schema() -> dict[str, Any]:
    return PairwiseVerdict.model_json_schema()


def pairwise_format() -> dict[str, Any]:
    """`response_format` for a pairwise choice: the strict JSON schema of `PairwiseVerdict`."""
    return _response_format(PAIRWISE_SCHEMA_NAME, PairwiseVerdict)


VerdictErrorKind = Literal["empty", "invalid_json", "invalid_schema"]


class InvalidVerdict(ValueError):
    """The judge's reply is not a valid verdict. `kind` says what was wrong."""

    def __init__(self, kind: VerdictErrorKind, detail: str) -> None:
        self.kind = kind
        self.detail = detail
        super().__init__(f"invalid judgement ({kind}): {detail}")


def parse_verdict[V: _Verdict](raw: str, model: type[V] = JudgeVerdict) -> V:
    """Validate a reply. The only leniency is the one triage has: the whole
    reply may be one ```json block."""
    if not raw.strip():
        raise InvalidVerdict("empty", "the reply has no text")
    text = unwrap_reply(raw)
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidVerdict("invalid_json", f"{exc.msg} at line {exc.lineno}") from None
    try:
        return model.model_validate_json(text, strict=True)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'reply'}: {err['msg']}"
            for err in exc.errors(include_input=False, include_url=False)
        )
    raise InvalidVerdict("invalid_schema", problems)


# --- the prompt ---------------------------------------------------------------

JUDGE_FUNCTION = "judge"
ADHOC = "adhoc"
Docs = Sequence[Hit]


def _material(question: str, docs: Docs) -> str:
    return (
        f"<question>\n{question}\n</question>\n\n"
        f"<documents>\n{format_documents(tuple(docs))}\n</documents>"
    )


def grade_messages(rubric: Rubric, question: str, docs: Docs, answer: str) -> list[dict[str, str]]:
    """The judge's messages for one answer: the rubric, then the material.

    Pure and deterministic, so a planner can compute the request key once the
    answer is known.
    """
    user = (
        'Grade the answer with the rubric (see "Grading one answer").\n\n'
        f"{_material(question, docs)}\n\n<answer>\n{answer}\n</answer>"
    )
    return [{"role": "system", "content": rubric.text}, {"role": "user", "content": user}]


def grade_tag(case: str, version: str) -> CallTag:
    """The cassette tag of a grading call: file `judge-<version>.jsonl`, and a
    case label that says it is the judge (`rag-001:judge/v1/0` in a replay miss)."""
    return CallTag(function=JUDGE_FUNCTION, case=f"{case}:judge", version=version)


PAIRWISE_FUNCTION = "pairwise"


def compare_messages(
    rubric: Rubric, question: str, docs: Docs, answer_a: str, answer_b: str
) -> list[dict[str, str]]:
    """The judge's messages for one pairwise question: answer A, then answer B."""
    user = (
        'Compare answer A and answer B with the rubric (see "Comparing two answers").\n\n'
        f"{_material(question, docs)}\n\n"
        f'<answer id="A">\n{answer_a}\n</answer>\n\n<answer id="B">\n{answer_b}\n</answer>'
    )
    return [{"role": "system", "content": rubric.text}, {"role": "user", "content": user}]


def compare_tag(case: str, versions: tuple[str, str], first: str) -> CallTag:
    """The cassette tag of one pairwise question: file `pairwise-<v1>-<v2>.jsonl`,
    and a case label naming the version shown as A (`rag-001:A=v2`)."""
    return CallTag(function=PAIRWISE_FUNCTION, case=f"{case}:A={first}", version="-".join(versions))


# --- the judge ----------------------------------------------------------------


@dataclass(frozen=True)
class Judgement:
    """The judge's grade of one answer, valid or not.

    - `verdict`: the parsed verdict, or None for an invalid judgement;
    - `error` and `detail`: what was wrong with an invalid one;
    - `raw`: the reply exactly as the judge wrote it;
    - `short_of`: criteria below the rubric's minimum;
    - `rule_pass`: the rubric's pass rule applied to the scores (None when invalid).
    """

    verdict: JudgeVerdict | None
    error: VerdictErrorKind | None
    detail: str
    raw: str
    call: CallResult
    short_of: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.verdict is not None

    @property
    def rule_pass(self) -> bool | None:
        return None if self.verdict is None else not self.short_of

    @property
    def agrees(self) -> bool | None:
        """Whether the judge's own `pass` matches the rubric rule on its scores."""
        return None if self.verdict is None else self.verdict.passed == self.rule_pass


Side = Literal["v1", "v2"]
Winner = Literal["v1", "v2", "tie"]
Outcome = Literal["v1", "v2", "tie", "inconsistent", "invalid"]
OUTCOMES: tuple[Outcome, ...] = ("v1", "v2", "tie", "inconsistent", "invalid")


@dataclass(frozen=True)
class OrderJudgement:
    """One of the two pairwise questions. `first` is the answer shown as A."""

    first: Side
    verdict: PairwiseVerdict | None
    error: VerdictErrorKind | None
    detail: str
    raw: str
    call: CallResult

    @property
    def valid(self) -> bool:
        return self.verdict is not None

    @property
    def preferred(self) -> Preference | None:
        return None if self.verdict is None else self.verdict.preferred

    @property
    def winner(self) -> Winner | None:
        """The preferred answer by name, whatever position it was shown in."""
        if self.verdict is None:
            return None
        if self.verdict.preferred == "tie":
            return "tie"
        second: Side = "v2" if self.first == "v1" else "v1"
        return self.first if self.verdict.preferred == "A" else second


def combine(first: Winner | None, second: Winner | None) -> Outcome:
    """The outcome of a pair from the winners of its two orders.

    Both orders agree: that winner (or tie). They disagree: `inconsistent`,
    never resolved by picking one. Either verdict invalid: `invalid`.
    """
    if first is None or second is None:
        return "invalid"
    return first if first == second else "inconsistent"


@dataclass(frozen=True)
class PairwiseResult:
    """Both orders of one comparison: (A=v1, B=v2), then (A=v2, B=v1)."""

    outcome: Outcome
    orders: tuple[OrderJudgement, OrderJudgement]


def _parse[V: _Verdict](call: CallResult, model: type[V]) -> V | InvalidVerdict:
    if call.empty_reason is not None:
        return InvalidVerdict("empty", f"the judge returned no text ({call.empty_reason})")
    try:
        return parse_verdict(call.content, model)
    except InvalidVerdict as exc:
        return exc


def _short(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


class Judge:
    """Grades answers with the rubric, through any `ChatModel` (the model client or a fake)."""

    def __init__(self, client: ChatModel, role: RoleConfig, rubric: Rubric) -> None:
        if not role.structured_output:
            raise ValueError(
                f"the judge role ({role.model}) needs structured_output: true; "
                "its verdict is requested with a strict JSON schema"
            )
        self.client = client
        self.role = role
        self.rubric = rubric

    def grade(
        self,
        question: str,
        docs: Docs,
        answer: str,
        *,
        case: str = ADHOC,
        version: str = ADHOC,
        repeat: int = 0,
    ) -> Judgement:
        """Grade one answer.

        `case` and `version` (the prompt version that wrote the answer) name
        the recording. `repeat` is the system run the answer came from; it is
        part of the request key, so every run's answer has its own verdict.
        """
        messages = grade_messages(self.rubric, question, docs, answer)
        call = self.client.complete(
            messages,
            role=self.role,
            response_format=verdict_format(),
            repeat=repeat,
            tag=grade_tag(case, version),
        )
        parsed = _parse(call, JudgeVerdict)
        if isinstance(parsed, InvalidVerdict):
            return Judgement(None, parsed.kind, parsed.detail, call.content, call)
        short_of = self.rubric.short_of(parsed.scores)
        return Judgement(parsed, None, "", call.content, call, short_of)

    def compare(
        self,
        question: str,
        docs: Docs,
        answer_v1: str,
        answer_v2: str,
        *,
        case: str = ADHOC,
        versions: tuple[str, str] = ("v1", "v2"),
    ) -> PairwiseResult:
        """Which answer is better, asked twice with the order swapped.

        The first question shows A=v1, B=v2; the second A=v2, B=v1. The outcome
        is `v1`, `v2` or `tie` when both orders agree, `inconsistent` when they
        do not, and `invalid` when either verdict is invalid. `versions` names
        the prompt versions behind `answer_v1` and `answer_v2`, for the
        recording only.
        """
        answers: dict[Side, str] = {"v1": answer_v1, "v2": answer_v2}
        orders: list[OrderJudgement] = []
        for first, second in (("v1", "v2"), ("v2", "v1")):
            messages = compare_messages(
                self.rubric, question, docs, answers[first], answers[second]
            )
            label = versions[0] if first == "v1" else versions[1]
            call = self.client.complete(
                messages,
                role=self.role,
                response_format=pairwise_format(),
                tag=compare_tag(case, versions, label),
            )
            parsed = _parse(call, PairwiseVerdict)
            if isinstance(parsed, InvalidVerdict):
                orders.append(
                    OrderJudgement(first, None, parsed.kind, parsed.detail, call.content, call)
                )
            else:
                orders.append(OrderJudgement(first, parsed, None, "", call.content, call))
        return PairwiseResult(combine(orders[0].winner, orders[1].winner), (orders[0], orders[1]))

    def checks(self, judgement: Judgement) -> list[CheckResult]:
        """The judge layer's checks for one answer.

        A valid verdict gives `verdict_valid` plus one check per criterion
        (score at least the rubric's minimum), so the layer passes exactly when
        the rubric's pass rule does. An invalid verdict gives one failing
        `verdict_valid`: the answer could not be graded, and the check name
        says the judge is why.
        """
        verdict = judgement.verdict
        if verdict is None:
            detail = f"invalid judgement ({judgement.error}): {_short(judgement.detail)}"
            return [CheckResult("verdict_valid", False, detail)]
        note = "the judge returned a valid verdict"
        if not judgement.agrees:
            note += (
                f"; the judge's own pass ({str(verdict.passed).lower()}) differs from "
                f"the rubric rule ({str(judgement.rule_pass).lower()})"
            )
        results = [CheckResult("verdict_valid", True, note)]
        for criterion in self.rubric.criteria:
            score, minimum = verdict.scores[criterion], self.rubric.pass_rule[criterion]
            detail = f"{score}/5 (pass needs {minimum} or more)"
            if score < minimum:
                detail += f"; judge: {_short(verdict.reasons)}"
            results.append(CheckResult(criterion, score >= minimum, detail))
        return results


# --- the judge's own biases ------------------------------------------------------
#
# Cheap checks, not proof: they show whether the judge leans on the position
# of an answer or on its length. The agreement with human labels is measured
# separately, once the owner has labelled answers.


def _ranks(values: Sequence[float]) -> list[float]:
    """Ranks from 1, ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for i in order[start : end + 1]:
            ranks[i] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman's rank correlation, rounded to 4 places.

    Ranks suit 1-5 scores, which are ordered but not evenly spaced. None with
    fewer than 3 pairs, or when either series has no spread.
    """
    if len(xs) != len(ys):
        raise ValueError("spearman needs two series of the same length")
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return round(cov / math.sqrt(vx * vy), 4)


Pair = tuple[Preference | None, Preference | None]


def position_bias(pairs: Iterable[Pair]) -> tuple[int, int]:
    """How often the judge preferred answer A: (choices of A, choices of a side).

    Counted over pairs where both orders chose a side. Each such pair shows
    each version once as A, so answer quality cancels out: 0.5 of the choices
    means no lean, 1.0 means the judge always picks A. Ties and invalid
    verdicts are left out, because they would unbalance the positions.
    """
    chose_a = chose = 0
    for first, second in pairs:
        if first in ("A", "B") and second in ("A", "B"):
            chose_a += (first == "A") + (second == "A")
            chose += 2
    return chose_a, chose


Inconsistency = Literal["same_position_a", "same_position_b", "tie_in_one_order"]
INCONSISTENCIES: tuple[Inconsistency, ...] = (
    "same_position_a",
    "same_position_b",
    "tie_in_one_order",
)


def inconsistency(first: Preference | None, second: Preference | None) -> Inconsistency | None:
    """How the two orders of a pair disagreed, or None when they agree or a
    verdict is invalid. Picking the same position twice means the position,
    not the answer, decided."""
    if first is None or second is None:
        return None
    if first == second:
        return {"A": "same_position_a", "B": "same_position_b"}.get(first)
    if "tie" in (first, second):
        return "tie_in_one_order"
    return None
