"""Judge agreement: percent agreement, Cohen's kappa, the confusion and the disagreements.

Synthetic data: every judge grade, sample and human label below is made up
for the test, never a model output or a label the owner set. Nothing is
written outside pytest's `tmp_path`.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from llmeval.agreement import (
    AGREEMENT_FILE,
    PENDING_HUMAN_LABELS,
    SAMPLE_NOTE,
    agreement_report,
    cohen_kappa,
    write_agreement,
)
from llmeval.checks.judge import RUBRIC_PATH, load_rubric
from llmeval.labels import LABEL_QUESTION, LABELER, HumanLabel, build_sample
from tests.unit import synthetic_results as syn
from tests.unit.synthetic_labels import population, rag_case

ROOT = Path(__file__).resolve().parents[2]
RUBRIC = load_rubric(ROOT / RUBRIC_PATH)
NOW = datetime(2026, 1, 2, 9, 30, tzinfo=UTC)


def pairs(both_pass, judge_only, human_only, both_fail):
    """(judge pass, human pass) pairs from the four cells of a 2x2 table."""
    return (
        [(True, True)] * both_pass
        + [(True, False)] * judge_only
        + [(False, True)] * human_only
        + [(False, False)] * both_fail
    )


def test_kappa_matches_a_hand_computed_two_by_two_table():
    # 30 answers: both pass 20, judge pass and human fail 3, judge fail and
    # human pass 2, both fail 5. Observed agreement po = 25/30. The judge
    # passes 23/30 and the human 22/30, so chance agreement
    # pe = (23 * 22 + 7 * 8) / 900 = 562/900, and
    # kappa = (po - pe) / (1 - pe) = (750 - 562) / (900 - 562) = 188/338.
    assert cohen_kappa(pairs(20, 3, 2, 5)) == pytest.approx(188 / 338)
    assert round(188 / 338, 4) == 0.5562


def test_kappa_edge_cases():
    assert cohen_kappa(pairs(3, 0, 0, 2)) == 1.0  # perfect agreement, both classes
    assert cohen_kappa(pairs(0, 1, 1, 0)) == -1.0  # perfect disagreement
    assert cohen_kappa(pairs(4, 1, 0, 0)) == 0.0  # the judge passes all: no better than chance
    assert cohen_kappa(pairs(5, 0, 0, 0)) is None  # all one class for both: undefined
    assert cohen_kappa(pairs(0, 0, 0, 5)) is None
    assert cohen_kappa([]) is None  # no labels


def results():
    return [
        population("v1", {"answerable": (6, 2), "unanswerable": (2, 2)}),
        population("v2", {"answerable": (6, 1), "unanswerable": (3, 1)}),
    ]


def label(item, value, comment="", key=None):
    return HumanLabel(
        case=item.case,
        version=item.version,
        repeat=item.repeat,
        answer_key=key or item.answer_key,
        label=value,
        comment=comment,
        labeler=LABELER,
        labeled_at=NOW,
    )


def verdicts(sample, run_results):
    graded = {
        (case.id, result.version): case.runs[0].judge.rule_pass
        for result in run_results
        for case in result.cases
    }
    return [graded[(item.case, item.version)] for item in sample.items]


def test_without_labels_the_report_is_pending_with_the_sample_size():
    run_results = results()
    sample = build_sample(run_results, size=12, seed=3)
    report = agreement_report(sample, [], run_results, RUBRIC)
    assert report.status == PENDING_HUMAN_LABELS
    assert report.sample.size == 12 and report.sample.seed == 3
    assert (report.sample.judge_fail, report.sample.judge_pass) == (6, 6)
    assert report.labelled == 0 and report.agreed == 0
    assert report.agreement_rate is None and report.kappa is None
    assert report.kappa_note == "no labelled answers"
    assert report.confusion == {
        "judge_pass": {"human_pass": 0, "human_fail": 0},
        "judge_fail": {"human_pass": 0, "human_fail": 0},
    }
    assert report.disagreements == [] and report.stale == [] and report.unjudged == []
    assert report.note == SAMPLE_NOTE
    assert "oversamples judge failures" in SAMPLE_NOTE
    assert "not the population rate" in SAMPLE_NOTE
    assert report.question == LABEL_QUESTION
    assert "groundedness at least 4, helpfulness at least 3 and tone at least 3" in (
        report.standards
    )
    assert report.judge_model == syn.JUDGE_MODEL
    assert report.rubric_sha256 == syn.RUBRIC_SHA256


def test_labels_give_the_rate_kappa_confusion_and_disagreements():
    run_results = results()
    sample = build_sample(run_results, size=12, seed=3)
    judged = verdicts(sample, run_results)
    # The human agrees on every answer except the first judge pass and the
    # first judge fail.
    first_pass = judged.index(True)
    first_fail = judged.index(False)
    labels = []
    for at, (item, passed) in enumerate(zip(sample.items, judged, strict=True)):
        human = passed if at not in (first_pass, first_fail) else not passed
        labels.append(label(item, "pass" if human else "fail", comment=f"Synthetic {at}."))
    report = agreement_report(sample, labels, run_results, RUBRIC)
    assert report.status == "complete"
    assert (report.labelled, report.agreed) == (12, 10)
    assert report.agreement_rate == round(10 / 12, 4)
    assert report.confusion == {
        "judge_pass": {"human_pass": 5, "human_fail": 1},
        "judge_fail": {"human_pass": 1, "human_fail": 5},
    }
    # po = 10/12, pe = (6*6 + 6*6)/144 = 1/2: kappa = (120 - 72) / (144 - 72) = 2/3.
    assert report.kappa == round(2 / 3, 4)
    assert report.kappa_note is None
    wrong = sorted([sample.items[first_pass], sample.items[first_fail]], key=lambda i: i.ref)
    assert [(d.case, d.version, d.repeat) for d in report.disagreements] == [i.ref for i in wrong]
    for disagreement in report.disagreements:
        assert disagreement.judge != disagreement.human
        assert disagreement.judge_reasons == "Synthetic reasons."
        assert set(disagreement.judge_scores) == {"groundedness", "helpfulness", "tone"}
        assert disagreement.human_comment.startswith("Synthetic ")
        assert disagreement.category in {"answerable", "unanswerable"}


def test_some_labels_give_a_partial_report():
    run_results = results()
    sample = build_sample(run_results, size=12, seed=3)
    report = agreement_report(sample, [label(sample.items[0], "pass")], run_results, RUBRIC)
    assert report.status == "partial"
    assert report.labelled == 1


def test_a_stale_label_is_named_and_never_used():
    run_results = results()
    sample = build_sample(run_results, size=12, seed=3)
    first, second = sample.items[0], sample.items[1]
    outside = rag_case("rag-099", "v1", "answerable", "pass").runs[0]
    labels = [
        label(first, "pass", key="e" * 64),  # the answer changed after labelling
        label(second, "fail"),
        HumanLabel(
            case="rag-099",
            version="v1",
            repeat=0,
            answer_key=outside.call.key,
            label="pass",
            comment="",
            labeler=LABELER,
            labeled_at=NOW,
        ),
    ]
    report = agreement_report(sample, labels, run_results, RUBRIC)
    assert report.labelled == 1
    assert sorted((s.case, s.version, s.reason) for s in report.stale) == sorted(
        [
            (
                first.case,
                first.version,
                "stale: the sample names another answer now (its cassette key changed)",
            ),
            ("rag-099", "v1", "stale: the answer is not in the sample"),
        ]
    )


def test_a_label_is_stale_when_the_results_hold_another_answer():
    run_results = results()
    sample = build_sample(run_results, size=12, seed=3)
    item = sample.items[0]
    changed = []
    for result in run_results:
        cases = []
        for case in result.cases:
            if (case.id, result.version) == (item.case, item.version):
                run = case.runs[0]
                call = run.call.model_copy(update={"key": "d" * 64})
                case = case.model_copy(update={"runs": [run.model_copy(update={"call": call})]})
            cases.append(case)
        changed.append(syn.function_results("rag", result.version, cases))
    report = agreement_report(sample, [label(item, "pass")], changed, RUBRIC)
    assert report.labelled == 0 and report.status == PENDING_HUMAN_LABELS
    assert [s.reason for s in report.stale] == [
        "stale: results/ hold another answer than the sample (run llmeval sample)"
    ]


def test_a_label_of_an_answer_without_a_valid_verdict_is_not_compared():
    run_results = [
        syn.function_results(
            "rag",
            "v1",
            [
                rag_case("rag-001", "v1", "answerable", "pass"),
                rag_case("rag-002", "v1", "answerable", "fail"),
            ],
        )
    ]
    sample = build_sample(run_results, size=2, seed=1)
    invalid = [
        syn.function_results(
            "rag",
            "v1",
            [
                rag_case("rag-001", "v1", "answerable", "invalid"),
                rag_case("rag-002", "v1", "answerable", "fail"),
            ],
        )
    ]
    item = next(i for i in sample.items if i.case == "rag-001")
    report = agreement_report(sample, [label(item, "pass")], invalid, RUBRIC)
    assert report.labelled == 0
    assert [(u.case, u.reason) for u in report.unjudged] == [
        ("rag-001", "the judge verdict for this answer is not valid")
    ]
    assert (report.sample.judge_pass, report.sample.judge_fail) == (0, 1)


def test_the_report_file_is_the_same_whatever_the_label_order(tmp_path):
    run_results = results()
    sample = build_sample(run_results, size=12, seed=3)
    labels = [label(item, "pass", comment="Synthetic.") for item in sample.items]
    first = write_agreement(agreement_report(sample, labels, run_results, RUBRIC), tmp_path / "a")
    second = write_agreement(
        agreement_report(sample, labels[::-1], run_results, RUBRIC), tmp_path / "b"
    )
    assert first.name == AGREEMENT_FILE == "judge-agreement.json"
    assert first.read_bytes() == second.read_bytes()
    assert first.read_text(encoding="utf-8").endswith("}\n")
