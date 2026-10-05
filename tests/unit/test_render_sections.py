"""The generated blocks after the main table: pairwise, agreement and the docs blocks.

Synthetic data: every result, verdict, label count, manifest and tolerance
below is made up for the test and written only into `tmp_path`, never into
the repository.
"""

import pytest

from llmeval.agreement import AgreementReport, SampleSummary, write_agreement
from llmeval.cassettes import write_manifest
from llmeval.results import PairwiseCaseRecord, write_results
from tests.unit import synthetic_results as syn
from tools import render


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "cassettes").mkdir()
    return tmp_path


def sources(ws) -> render.Sources:
    return render.Sources(results_dir=ws / "results", cassettes_dir=ws / "cassettes")


def body(ws, name: str) -> str:
    return render.render_bodies([name], sources(ws))[name]


def identical(case_id: str) -> PairwiseCaseRecord:
    return PairwiseCaseRecord(
        id=case_id,
        category="answerable",
        input="Synthetic question?",
        answers={"v1": "Same.", "v2": "Same."},
        outcome="identical",
        orders=[],
    )


def graded_rag(version: str, grades: list[str]):
    cases = [
        syn.case_record(f"rag-{n:03d}", (), judge=grade) for n, grade in enumerate(grades, 1)
    ]
    return syn.function_results("rag", version, cases)


def record_rag(ws, pairs, grades=("pass", "fail")):
    write_manifest(ws / "cassettes", syn.manifest({"rag": ("v1", "v2")}))
    for version in ("v1", "v2"):
        write_results(graded_rag(version, list(grades)), ws / "results")
    write_results(syn.pairwise_results(pairs), ws / "results")


PAIRS = [
    syn.pair_case("rag-001", "B", "A"),  # v2 in both orders
    syn.pair_case("rag-002", "A", "B"),  # v1 in both orders
    syn.pair_case("rag-003", "tie", "tie"),
    syn.pair_case("rag-004", "A", "A"),  # answer A both times: inconsistent
    syn.pair_case("rag-005", "tie", "A"),  # a tie in one order: inconsistent
    identical("rag-006"),
    syn.pair_case("rag-007", None, "A"),  # an invalid verdict
]


def test_the_pairwise_block_counts_outcomes_over_cases_and_consistency_over_compared_pairs(ws):
    record_rag(ws, PAIRS)
    assert body(ws, "pairwise") == (
        "\nThe judge compared the first answers of rag v1 and rag v2 case by case, asked "
        "twice with the order of the two answers swapped.\n\n"
        "| Outcome over 7 cases | Cases |\n"
        "|---|---:|\n"
        "| v1 preferred in both orders | 1 |\n"
        "| v2 preferred in both orders | 1 |\n"
        "| A tie in both orders | 1 |\n"
        "| Inconsistent: the two orders disagree | 2 |\n"
        "| Identical answers, not compared | 1 |\n"
        "| An invalid verdict, not compared | 1 |\n\n"
        "Position consistency: 3 of 5 compared pairs (60.0%) got the same verdict in both "
        "orders.\n\n"
        "In 2 of the 5 compared pairs, the judge's preference changed when the two answers "
        "swapped places: 1 time it chose the answer shown first in both orders, and 1 time it "
        "called a tie in one order and chose a side in the other. An inconsistent pair is never "
        "settled by picking one order.\n\n"
    )


def test_the_pairwise_block_leaves_out_outcomes_that_did_not_happen(ws):
    record_rag(ws, [syn.pair_case("rag-001", "B", "A"), syn.pair_case("rag-002", "B", "B")])
    text = body(ws, "pairwise")
    assert "Identical answers" not in text and "invalid" not in text
    assert "| A tie in both orders | 0 |" in text  # the four main outcomes always show
    assert "Position consistency: 1 of 2 compared pairs (50.0%)" in text
    assert "1 time it chose the answer shown second in both orders." in text


def test_with_every_pair_consistent_the_block_says_so(ws):
    record_rag(ws, [syn.pair_case("rag-001", "tie", "tie")])
    text = body(ws, "pairwise")
    assert "Position consistency: 1 of 1 compared pairs (100.0%)" in text
    assert "In none of the 1 compared pairs did the judge's preference change" in text


def sample_report(**changes) -> AgreementReport:
    fields = dict(
        status="pending human labels",
        standards="Synthetic standards.",
        judge_model=syn.JUDGE_MODEL,
        rubric_sha256=syn.RUBRIC_SHA256,
        sample=SampleSummary(size=3, seed=1, judge_pass=1, judge_fail=2),
        labelled=0,
        agreed=0,
        agreement_rate=None,
        kappa=None,
        kappa_note="no labelled answers",
        confusion={
            "judge_pass": {"human_pass": 0, "human_fail": 0},
            "judge_fail": {"human_pass": 0, "human_fail": 0},
        },
        disagreements=[],
        stale=[],
        unjudged=[],
    )
    fields.update(changes)
    return AgreementReport(**fields)


def test_without_labels_the_agreement_block_is_pending_with_the_sample(ws):
    record_rag(ws, PAIRS)
    write_agreement(sample_report(), ws / "results")
    assert body(ws, "agreement") == (
        "\npending human labels\n\n"
        "The owner labels 3 judged answers by hand, blind to the judge's verdict (make label): "
        "all 2 answers the judge failed and 1 it passed. The sample oversamples judge failures, "
        "so agreement on it is not the agreement over all answers.\n\n"
    )


def test_the_agreement_block_says_when_the_sample_misses_judge_failures(ws):
    record_rag(ws, PAIRS, grades=("fail", "fail", "pass"))
    write_agreement(sample_report(), ws / "results")
    assert "2 of the 4 answers the judge failed and 1 it passed" in body(ws, "agreement")


def labelled_report(**changes) -> AgreementReport:
    fields = dict(
        status="complete",
        labelled=3,
        agreed=2,
        agreement_rate=0.6667,
        kappa=0.4,
        kappa_note=None,
        confusion={
            "judge_pass": {"human_pass": 1, "human_fail": 1},
            "judge_fail": {"human_pass": 0, "human_fail": 1},
        },
    )
    fields.update(changes)
    return sample_report(**fields)


def test_with_labels_the_agreement_block_shows_agreement_kappa_and_the_confusion(ws):
    record_rag(ws, PAIRS)
    write_agreement(labelled_report(), ws / "results")
    assert body(ws, "agreement") == (
        "\n| Judge's verdict | Owner: pass | Owner: fail |\n"
        "|---|---:|---:|\n"
        "| Judge: pass | 1 | 1 |\n"
        "| Judge: fail | 0 | 1 |\n\n"
        "Percent agreement: 2 of 3 (66.7%). Cohen's kappa: 0.40. Disagreements: 1, listed in "
        "results/judge-agreement.json with the judge's reasons and the owner's comments.\n\n"
        "Labelled: 3 of 3 sample answers.\n\n"
        "The owner labels 3 judged answers by hand, blind to the judge's verdict (make label): "
        "all 2 answers the judge failed and 1 it passed. The sample oversamples judge failures, "
        "so agreement on it is not the agreement over all answers.\n\n"
    )


def test_a_partial_labelling_and_an_undefined_kappa_are_named(ws):
    record_rag(ws, PAIRS)
    report = labelled_report(
        status="partial",
        labelled=2,
        agreed=2,
        agreement_rate=1.0,
        kappa=None,
        kappa_note="undefined: both gave every answer the same single label",
        confusion={
            "judge_pass": {"human_pass": 2, "human_fail": 0},
            "judge_fail": {"human_pass": 0, "human_fail": 0},
        },
        stale=[{"case": "rag-001", "version": "v1", "repeat": 0, "reason": "stale: x"}],
    )
    write_agreement(report, ws / "results")
    text = body(ws, "agreement")
    assert "Cohen's kappa: undefined: both gave every answer the same single label." in text
    assert "Disagreements: 0." in text
    assert "Labelled so far: 2 of 3 sample answers (partial)." in text
    assert "Labels not used: 1, listed in results/judge-agreement.json." in text


def test_a_recorded_run_without_the_agreement_file_is_an_error(ws):
    record_rag(ws, PAIRS)
    with pytest.raises(render.RenderError, match="judge-agreement.json: run make eval"):
        body(ws, "agreement")


def test_a_run_without_judged_answers_needs_no_agreement_file(ws):
    write_manifest(ws / "cassettes", syn.manifest({"triage": ("v1",)}))
    case = syn.case_record("tri-001", (), checks=syn.TRIAGE_CHECKS)
    write_results(syn.function_results("triage", "v1", [case]), ws / "results")
    assert body(ws, "agreement") == "\nNo answer in this run was graded by the judge.\n\n"
    assert body(ws, "pairwise") == "\nNo pairwise comparison in this run.\n\n"


def test_both_blocks_are_pending_without_a_recorded_run(ws):
    assert body(ws, "pairwise") == "\npending first recorded run\n\n"
    assert body(ws, "agreement") == "\npending first recorded run\n\n"
