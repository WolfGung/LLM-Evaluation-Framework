"""How far the judge agrees with the owner's labels.

The judge is a measuring instrument, so it is checked against the owner's
own labels. `llmeval agreement` (also run by `make eval`) takes the labelled
answers of `labels/sample.json`, compares the judge's verdict by the rubric's
pass rule (`rule_pass` in the results) with the owner's label in
`labels/human.jsonl`, and writes `results/judge-agreement.json`:

- percent agreement (`agreed` of `labelled`, as `agreement_rate`) and Cohen's
  kappa (`cohen_kappa`), the agreement beyond what the two pass rates give
  by chance;
- a 2x2 confusion: the judge's pass or fail by the owner's pass or fail;
- the disagreements: case, version, the judge's scores and reasons, and the
  owner's comment;
- stale labels, never used: the sample now names another answer for the
  same case, version and repeat (its cassette key changed), the answer left
  the sample, or the results hold another answer than the sample. Labels of
  answers without a valid judge verdict are listed as `unjudged`;
- with no usable label, the status `pending human labels` and the sample size.

Agreement is measured on this sample, which oversamples judge failures so
both classes are present; it is not the population rate.

The two standards: the owner answers one question per answer
(`labels.LABEL_QUESTION`): would you send it to the customer as is, that is,
is it grounded in the shown documents, does it answer the question (or say
honestly that the documents do not cover it), and is it polite. The judge
passes an answer when its scores meet the rubric's minimums (groundedness 4,
helpfulness 3, tone 3 in `rubrics/judge.md`). Both ask the same three things.
The owner's single decision can be stricter: an answer that misses part of
the question can score helpfulness 3 and pass the judge, yet not be sent as
is.

The file is a pure function of the results, the sample, the labels and the
rubric, with no clock in it, so a fresh replay plus the committed labels
reproduces it byte for byte.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from llmeval.checks.judge import Rubric
from llmeval.labels import (
    LABEL_QUESTION,
    HumanLabel,
    Record,
    Ref,
    Sample,
    describe,
    runs_by_answer,
)
from llmeval.results import FunctionResults

AGREEMENT_FILE = "judge-agreement.json"
AGREEMENT_SCHEMA_VERSION = 1
PENDING_HUMAN_LABELS = "pending human labels"
SAMPLE_NOTE = (
    "Agreement is measured on labels/sample.json, which oversamples judge failures so both "
    "classes are present; it is not the population rate."
)
STALE_KEY = "stale: the sample names another answer now (its cassette key changed)"
STALE_OUTSIDE = "stale: the answer is not in the sample"
STALE_RESULTS = "stale: results/ hold another answer than the sample (run llmeval sample)"
UNJUDGED = "the judge verdict for this answer is not valid"

Verdict = Literal["pass", "fail"]


def cohen_kappa(pairs: Sequence[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa of two raters' pass or fail on the same answers.

    kappa = (po - pe) / (1 - pe), where po is the share of answers they
    agree on and pe the agreement expected by chance from each rater's pass
    share p1 and p2: pe = p1 * p2 + (1 - p1) * (1 - p2). 1 is perfect
    agreement, 0 no better than chance, below 0 worse. None when there are
    no answers, or when pe is 1 (both raters gave every answer the same one
    label): kappa is undefined then. Computed on counts, so a hand-checked
    table gives the exact fraction.
    """
    n = len(pairs)
    if n == 0:
        return None
    agreed = sum(first == second for first, second in pairs)
    first_pass = sum(first for first, _ in pairs)
    second_pass = sum(second for _, second in pairs)
    chance = first_pass * second_pass + (n - first_pass) * (n - second_pass)
    if chance == n * n:
        return None
    return (agreed * n - chance) / (n * n - chance)


class SampleSummary(Record):
    """The sample: its size and seed, and the judge's verdicts on it."""

    size: int
    seed: int
    judge_pass: int
    judge_fail: int


class Disagreement(Record):
    case: str
    version: str
    repeat: int
    category: str
    judge: Verdict
    human: Verdict
    judge_scores: dict[str, int]
    judge_reasons: str | None
    human_comment: str


class UnusedLabel(Record):
    """A label that is not compared, and why."""

    case: str
    version: str
    repeat: int
    reason: str


class AgreementReport(Record):
    """The content of `results/judge-agreement.json` (see the module docstring).

    `status` is `pending human labels` (no usable label), `partial` or
    `complete` (every sample answer labelled). `confusion` is keyed by the
    judge's verdict, then the owner's.
    """

    schema_version: int = AGREEMENT_SCHEMA_VERSION
    status: str
    note: str = SAMPLE_NOTE
    question: str = LABEL_QUESTION
    standards: str
    judge_model: str | None
    rubric_sha256: str | None
    sample: SampleSummary
    labelled: int
    agreed: int
    agreement_rate: float | None
    kappa: float | None
    kappa_note: str | None
    confusion: dict[str, dict[str, int]]
    disagreements: list[Disagreement]
    stale: list[UnusedLabel]
    unjudged: list[UnusedLabel]


def standards(rubric: Rubric) -> str:
    """How the owner's question and the judge's pass rule line up."""
    minimums = [f"{name} at least {rubric.pass_rule[name]}" for name in rubric.criteria]
    rule = ", ".join(minimums[:-1]) + f" and {minimums[-1]}" if len(minimums) > 1 else minimums[0]
    return (
        "The owner answers the question above for each answer. The judge passes an answer "
        f"when it scores {rule} (rubrics/judge.md). Both ask the same three things: grounded "
        "is groundedness, answering the question or saying honestly that the documents do "
        "not cover it is helpfulness, polite is tone. The owner's one decision can be "
        "stricter: an answer that misses part of the question can score helpfulness 3 and "
        "pass the judge, yet not be sent as is."
    )


def _word(passed: bool) -> Verdict:
    return "pass" if passed else "fail"


def agreement_report(
    sample: Sample,
    labels: Iterable[HumanLabel],
    results: Sequence[FunctionResults],
    rubric: Rubric,
) -> AgreementReport:
    """Compare the judge with the owner's labels on the sample answers."""
    runs = runs_by_answer(results)
    items = {item.ref: item for item in sample.items}

    def current(ref: Ref):
        """The results' run of a sample answer, when it is the sample's answer."""
        answer = runs.get(ref)
        return answer if answer and answer.run.call.key == items[ref].answer_key else None

    stale: list[UnusedLabel] = []
    unjudged: list[UnusedLabel] = []
    compared = []
    for label in labels:
        item = items.get(label.ref)
        reason = None
        if item is None:
            reason = STALE_OUTSIDE
        elif label.answer_key != item.answer_key:
            reason = STALE_KEY
        elif (answer := current(label.ref)) is None:
            reason = STALE_RESULTS
        elif answer.verdict is None:
            unjudged.append(UnusedLabel(**_ref(label.ref), reason=UNJUDGED))
            continue
        if reason is not None:
            stale.append(UnusedLabel(**_ref(label.ref), reason=reason))
            continue
        compared.append((item, label, answer))
    compared.sort(key=lambda entry: entry[0].ref)
    pairs = [(bool(answer.verdict), label.label == "pass") for _, label, answer in compared]
    confusion = {
        f"judge_{_word(judge)}": {
            f"human_{_word(human)}": sum(pair == (judge, human) for pair in pairs)
            for human in (True, False)
        }
        for judge in (True, False)
    }
    disagreements = [
        Disagreement(
            **_ref(item.ref),
            category=item.category,
            judge=_word(bool(answer.verdict)),
            human=label.label,
            judge_scores=dict(answer.run.judge.scores or {}),
            judge_reasons=answer.run.judge.reasons,
            human_comment=label.comment,
        )
        for item, label, answer in compared
        if bool(answer.verdict) != (label.label == "pass")
    ]
    verdicts = [answer.verdict for ref in items if (answer := current(ref)) is not None]
    agreed = sum(judge == human for judge, human in pairs)
    kappa = cohen_kappa(pairs)
    if not pairs:
        status, kappa_note = PENDING_HUMAN_LABELS, "no labelled answers"
    else:
        status = "complete" if len(pairs) == sample.size else "partial"
        kappa_note = (
            None if kappa is not None else "undefined: both gave every answer the same single label"
        )
    graded = [result for result in results if result.judge_model is not None]
    return AgreementReport(
        status=status,
        standards=standards(rubric),
        judge_model=graded[0].judge_model if graded else None,
        rubric_sha256=graded[0].rubric_sha256 if graded else None,
        sample=SampleSummary(
            size=sample.size,
            seed=sample.seed,
            judge_pass=sum(verdict is True for verdict in verdicts),
            judge_fail=sum(verdict is False for verdict in verdicts),
        ),
        labelled=len(pairs),
        agreed=agreed,
        agreement_rate=round(agreed / len(pairs), 4) if pairs else None,
        kappa=round(kappa, 4) if kappa is not None else None,
        kappa_note=kappa_note,
        confusion=confusion,
        disagreements=disagreements,
        stale=sorted(stale, key=_order),
        unjudged=sorted(unjudged, key=_order),
    )


def _ref(ref: Ref) -> dict[str, str | int]:
    case, version, repeat = ref
    return {"case": case, "version": version, "repeat": repeat}


def _order(label: UnusedLabel) -> tuple[str, str, int, str]:
    return (label.case, label.version, label.repeat, label.reason)


def write_agreement(report: AgreementReport, results_dir: Path | str) -> Path:
    path = Path(results_dir) / AGREEMENT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def report_lines(report: AgreementReport) -> list[str]:
    """The report for the terminal."""
    sample = report.sample
    if report.status == PENDING_HUMAN_LABELS:
        lines = [
            f"judge agreement: {PENDING_HUMAN_LABELS}: 0 of {sample.size} sample answers "
            f"labelled ({sample.judge_fail} the judge failed, {sample.judge_pass} it passed)"
        ]
    else:
        rate = report.agreement_rate or 0.0
        kappa = f"{report.kappa:.3f}" if report.kappa is not None else f"n/a ({report.kappa_note})"
        cells = report.confusion
        lines = [
            f"judge agreement ({report.status}): {report.labelled} of {sample.size} sample "
            "answers labelled",
            f"  percent agreement {rate:.1%} ({report.agreed} of {report.labelled}), "
            f"Cohen's kappa {kappa}",
            "  confusion (judge by human): "
            f"judge pass: human pass {cells['judge_pass']['human_pass']}, "
            f"human fail {cells['judge_pass']['human_fail']}; "
            f"judge fail: human pass {cells['judge_fail']['human_pass']}, "
            f"human fail {cells['judge_fail']['human_fail']}",
        ]
        for item in report.disagreements:
            scores = ", ".join(f"{name} {score}" for name, score in item.judge_scores.items())
            comment = f": {item.human_comment}" if item.human_comment else ""
            lines.append(
                f"  disagreement {describe((item.case, item.version, item.repeat))}: judge "
                f"{item.judge} ({scores}), human {item.human}{comment}"
            )
    for unused in (*report.stale, *report.unjudged):
        lines.append(
            f"  label not used, {describe((unused.case, unused.version, unused.repeat))}: "
            f"{unused.reason}"
        )
    lines.append(f"  {report.note}")
    return lines
