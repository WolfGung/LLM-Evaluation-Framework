"""The generated blocks after the main table: pairwise, agreement and the docs blocks.

Synthetic data: every result, verdict, label count, manifest and tolerance
below is made up for the test and written only into `tmp_path`, never into
the repository.
"""

import json
from fractions import Fraction

import pytest

from llmeval.agreement import AgreementReport, Disagreement, SampleSummary, write_agreement
from llmeval.cassettes import write_manifest
from llmeval.results import PairwiseCaseRecord, write_results
from tests.unit import synthetic_results as syn
from tools import render, sections


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
    cases = [syn.case_record(f"rag-{n:03d}", (), judge=grade) for n, grade in enumerate(grades, 1)]
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
        f"{LOW_CONSISTENCY_READING}\n\n"
    )


LOW_CONSISTENCY_READING = (
    "With this many flips, the comparison says more about the judge's position bias than about "
    "the two prompts, so it picks no winner. The main table rests on the rules and the "
    "per-answer grades."
)


def test_below_the_consistency_threshold_the_pairwise_block_picks_no_winner(ws):
    record_rag(ws, PAIRS)  # 3 of 5 compared pairs consistent: 60.0%
    assert body(ws, "pairwise").endswith(f"{LOW_CONSISTENCY_READING}\n\n")


def test_at_the_consistency_threshold_the_pairwise_block_adds_no_reading(ws):
    consistent = [syn.pair_case(f"rag-00{n}", "tie", "tie") for n in range(1, 5)]
    record_rag(ws, [*consistent, syn.pair_case("rag-005", "A", "A")])  # 4 of 5: 80.0%
    text = body(ws, "pairwise")
    assert "Position consistency: 4 of 5 compared pairs (80.0%)" in text
    assert LOW_CONSISTENCY_READING not in text
    assert Fraction(4, 5) == sections.PAIRWISE_TRUSTED_CONSISTENCY


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
        "Pavel Zhukov Atum, the author, labels 3 judged answers by hand, blind to the judge's "
        "verdict (make label): "
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
        "\n| Judge's verdict | Author: pass | Author: fail |\n"
        "|---|---:|---:|\n"
        "| Judge: pass | 1 | 1 |\n"
        "| Judge: fail | 0 | 1 |\n\n"
        "Percent agreement: 2 of 3 (66.7%). Cohen's kappa: 0.40. Disagreements: 1, listed in "
        "results/judge-agreement.json with the judge's reasons.\n\n"
        "Labelled: 3 of 3 sample answers.\n\n"
        "Pavel Zhukov Atum, the author, labelled 3 judged answers by hand, blind to the judge's "
        "verdict (make label): "
        "all 2 answers the judge failed and 1 it passed. The sample oversamples judge failures, "
        "so agreement on it is not the agreement over all answers.\n\n"
    )


def test_the_agreement_block_mentions_comments_only_when_a_disagreement_has_one(ws):
    record_rag(ws, PAIRS)
    disagreement = Disagreement(
        case="rag-001",
        version="v1",
        repeat=0,
        category="answerable",
        judge="pass",
        human="fail",
        judge_scores={"groundedness": 5, "helpfulness": 5, "tone": 5},
        judge_reasons="Synthetic reasons.",
        human_comment="Synthetic comment.",
    )
    write_agreement(labelled_report(disagreements=[disagreement]), ws / "results")
    assert "with the judge's reasons and the author's comments." in body(ws, "agreement")


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
    assert "Pavel Zhukov Atum, the author, labels 3 judged answers" in text  # not finished yet


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
    assert body(ws, "findings") == "\npending first recorded run\n\n"


# --- findings ---------------------------------------------------------------------------


def overturned(case_id: str, scores: tuple[int, int, int]) -> Disagreement:
    """An answer the judge failed with these scores and the author passed."""
    return Disagreement(
        case=case_id,
        version="v1",
        repeat=0,
        category="answerable",
        judge="fail",
        human="pass",
        judge_scores=dict(zip(("groundedness", "helpfulness", "tone"), scores, strict=True)),
        judge_reasons="Synthetic reasons.",
        human_comment="",
    )


LINKS = (
    "([results/judge-agreement.json](results/judge-agreement.json), "
    "[docs/03](docs/03-judge-validation.md#agreement-with-a-person))"
)


def test_the_findings_read_the_agreement_and_the_position_flips_in_plain_words(ws):
    record_rag(ws, PAIRS)
    write_agreement(labelled_report(), ws / "results")
    assert body(ws, "findings") == (
        "\n- **Trust the judge's fails more than its passes.** The judge agreed with the author "
        "on 2 of 3 sample answers (66.7%), while two raters who pass answers as often as these "
        "two do would agree on 44.4% by chance alone: Cohen's kappa of 0.40 counts only the "
        "agreement beyond that, and a kappa from 0.21 to 0.40 is conventionally called fair "
        f"agreement. The author agreed with 1 of the judge's 2 passes and 1 of its 1 fail {LINKS}."
        "\n- **With this judge, one call per case cannot compare two prompts.** In 2 of the 5 "
        "compared pairs of rag v1 and v2 answers (40.0%), the judge changed its verdict when the "
        "two answers swapped places. 1 more pair with identical answers and 1 more pair with an "
        "invalid verdict were not compared. With one judge call per case, those verdicts would "
        "depend on which answer happened to be shown first: ask in both orders and count only "
        "the pairs that agree, as [the comparison below]"
        "(#the-two-prompt-versions-compared-by-the-judge) does.\n\n"
    )


def test_the_findings_name_what_the_overturned_fails_have_in_common(ws):
    record_rag(ws, PAIRS)
    report = labelled_report(
        labelled=10,
        agreed=7,
        kappa=0.2,
        confusion={
            "judge_pass": {"human_pass": 6, "human_fail": 0},
            "judge_fail": {"human_pass": 3, "human_fail": 1},
        },
        disagreements=[
            overturned("rag-001", (3, 5, 5)),
            overturned("rag-002", (3, 4, 5)),
            overturned("rag-003", (2, 3, 4)),
        ],
    )
    write_agreement(report, ws / "results")
    text = body(ws, "findings")
    assert text.startswith("\n- **Trust the judge's passes more than its fails.**")
    assert "a kappa from 0.00 to 0.20 is conventionally called slight agreement" in text
    assert (
        "The author agreed with 6 of the judge's 6 passes but only 1 of its 4 fails, and on all "
        "3 answers the judge failed and the author passed, groundedness was its lowest score: "
        "it is stricter than the author about what the documents support"
    ) in text


@pytest.mark.parametrize(
    "scores",
    [[(3, 5, 5), (5, 3, 5)], [(3, 3, 5), (3, 4, 5)], [(3, 5, 5)]],
    ids=["different", "a tie for lowest", "one alone"],
)
def test_the_findings_name_nothing_in_common_without_one_lowest_score(ws, scores):
    record_rag(ws, PAIRS)
    disagreements = [overturned(f"rag-00{n}", s) for n, s in enumerate(scores, 1)]
    write_agreement(labelled_report(disagreements=disagreements), ws / "results")
    assert "lowest score" not in body(ws, "findings")


@pytest.mark.parametrize(
    ("kappa", "named"),
    [
        (0.0, "a kappa from 0.00 to 0.20 is conventionally called slight agreement"),
        (0.205, "Cohen's kappa of 0.21 counts"),  # two decimals, a half rounds up
        (0.3023, "a kappa from 0.21 to 0.40 is conventionally called fair agreement"),
        (0.41, "a kappa from 0.41 to 0.60 is conventionally called moderate agreement"),
        (0.8, "a kappa from 0.61 to 0.80 is conventionally called substantial agreement"),
        (0.95, "a kappa from 0.81 to 1.00 is conventionally called almost perfect agreement"),
        (-0.1, "Cohen's kappa of -0.10 is below zero: less agreement than chance alone gives"),
    ],
)
def test_the_kappa_has_its_conventional_name(kappa, named):
    assert named in sections._kappa_name(kappa)


def test_an_undefined_kappa_is_named_in_the_findings(ws):
    record_rag(ws, PAIRS)
    write_agreement(labelled_report(kappa=None, kappa_note="undefined: x"), ws / "results")
    assert "by chance alone; Cohen's kappa is undefined: x." in body(ws, "findings")


def test_without_labels_the_agreement_finding_is_pending(ws):
    record_rag(ws, PAIRS)
    write_agreement(sample_report(), ws / "results")
    assert body(ws, "findings").startswith(
        "\n- **The judge's agreement with the author is not measured yet:** pending human labels.\n"
    )


def test_a_comparison_without_flips_says_the_order_did_not_sway_the_judge(ws):
    record_rag(ws, [syn.pair_case("rag-001", "tie", "tie"), identical("rag-002")])
    write_agreement(sample_report(), ws / "results")
    assert body(ws, "findings").endswith(
        "- **The order of the answers did not sway the judge.** In none of the 1 compared pairs "
        "of rag v1 and v2 answers did the judge change its verdict when the two answers swapped "
        "places, so one order would have given the same verdicts here; asking in both orders is "
        "what shows it. 1 more pair with identical answers was not compared.\n\n"
    )


def test_a_few_flips_above_the_consistency_threshold_ask_for_both_orders(ws):
    consistent = [syn.pair_case(f"rag-00{n}", "tie", "tie") for n in range(1, 5)]
    record_rag(ws, [*consistent, syn.pair_case("rag-005", "A", "A")])  # 4 of 5: 80.0%
    write_agreement(sample_report(), ws / "results")
    text = body(ws, "findings")
    assert "- **Ask the judge in both orders.** In 1 of the 5 compared pairs" in text
    assert "were not compared" not in text and "was not compared" not in text


def test_a_run_without_the_judge_has_no_findings(ws):
    write_manifest(ws / "cassettes", syn.manifest({"triage": ("v1",)}))
    case = syn.case_record("tri-001", (), checks=syn.TRIAGE_CHECKS)
    write_results(syn.function_results("triage", "v1", [case]), ws / "results")
    assert body(ws, "findings") == "\nNo answer in this run was graded by the judge.\n\n"


# --- the docs blocks: judge, safety, cost, gate, scope ----------------------------------


def judged(case_id: str, output: str, scores: tuple[int, int, int], rule_pass: bool, **kw):
    """A graded RAG case whose answer is `output`, with these three scores."""
    record = syn.case_record(case_id, (), judge="pass")
    grade = record.runs[0].judge.model_copy(
        update={
            "scores": dict(zip(("groundedness", "helpfulness", "tone"), scores, strict=True)),
            "rule_pass": rule_pass,
            "judge_pass": kw.get("judge_pass", rule_pass),
        }
    )
    run = record.runs[0].model_copy(update={"output": output, "judge": grade})
    return record.model_copy(update={"runs": [run]})


def judged_versions():
    cases = [
        judged("rag-001", "One", (5, 3, 5), True),
        judged("rag-002", "One two", (4, 4, 5), True, judge_pass=False),
        judged("rag-003", "One two three", (3, 5, 5), False),
    ]
    return [syn.function_results("rag", version, cases) for version in ("v1", "v2")]


def record_judged(ws, pairs=PAIRS):
    write_manifest(ws / "cassettes", syn.manifest({"rag": ("v1", "v2")}))
    for result in judged_versions():
        write_results(result, ws / "results")
    write_results(syn.pairwise_results(pairs), ws / "results")


def test_the_judge_block_shows_reliability_scores_and_length_per_version(ws):
    record_judged(ws)
    text = body(ws, "judge")
    assert "| Measure | rag v1 | rag v2 |\n|---|---:|---:|\n" in text
    assert "| Answers graded | 3 | 3 |" in text
    assert "| Valid verdicts | 3 of 3 | 3 of 3 |" in text
    assert "| Pass by the rubric rule | 2 of 3 | 2 of 3 |" in text
    assert "| The judge's own pass differs from the rule | 1 | 1 |" in text
    assert "| Mean groundedness | 4.00 | 4.00 |" in text
    assert "| Mean tone | 5.00 | 5.00 |" in text
    assert "| Answer length and groundedness (Spearman) | -1.00 | -1.00 |" in text
    assert "| Answer length and helpfulness (Spearman) | 1.00 | 1.00 |" in text
    assert "| Answer length and tone (Spearman) | — | — |" in text
    assert (
        "A dash: no correlation can be computed, because every graded answer got the same "
        "score or fewer than three answers were graded."
    ) in text
    assert (
        "A positive correlation: helpfulness in rag v1 (1.00) and rag v2 (1.00). It can mean "
        "the judge rewards length, or that the longer answers were more complete."
    ) in text


def test_the_judge_block_reads_position_length_and_vendors_from_the_run(ws):
    record_judged(ws)
    text = body(ws, "judge")
    # rag-001 and rag-002 chose a side in both orders: A, B, A, B; rag-004 A, A.
    assert (
        "Position: where the judge chose a side in both orders, it chose the answer shown "
        "first 4 of 6 times (66.7%). Half would mean no lean."
    ) in text
    # Every synthetic pair has two answers of two words each.
    assert "Length: no side choice was between answers of different lengths." in text
    assert (
        "The system model is synthetic/system:free and the judge is synthetic/judge:free, "
        "both from the same vendor (synthetic): the judge may prefer answers written by its "
        "own model family."
    ) in text


def test_the_judge_block_counts_how_often_the_longer_answer_was_chosen(ws):
    longer_v1 = syn.pair_case("rag-001", "B", "A").model_copy(
        update={"answers": {"v1": "One two three.", "v2": "One."}}
    )
    record_judged(ws, pairs=[longer_v1, syn.pair_case("rag-002", "tie", "tie")])
    assert (
        "Length: of those choices between answers of different lengths, it chose the longer "
        "answer 0 of 2 times (0.0%)."
    ) in body(ws, "judge")


def test_the_judge_block_names_different_vendors(ws):
    record_judged(ws)
    manifest = syn.manifest({"rag": ("v1", "v2")}).model_copy(
        update={"models": {"system": "acme/small:free", "judge": "other/judge:free"}}
    )
    write_manifest(ws / "cassettes", manifest)
    assert (
        "The system model is acme/small:free (vendor acme) and the judge is other/judge:free "
        "(vendor other): different vendors, so the judge does not grade answers written by its "
        "own model family."
    ) in body(ws, "judge")


def test_without_positive_correlations_the_judge_block_says_longer_did_not_score_higher(ws):
    write_manifest(ws / "cassettes", syn.manifest({"rag": ("v1", "v2")}))
    cases = [
        judged("rag-001", "One", (5, 5, 5), True),
        judged("rag-002", "One two", (4, 4, 4), True),
        judged("rag-003", "One two three", (3, 3, 3), False),
    ]
    for version in ("v1", "v2"):
        write_results(syn.function_results("rag", version, cases), ws / "results")
    write_results(syn.pairwise_results(PAIRS), ws / "results")
    text = body(ws, "judge")
    assert (
        "No criterion has a positive correlation: longer answers did not get higher scores."
    ) in text
    assert "A dash:" not in text


RAG_SAFETY_ROW = {
    "id": "rag-047",
    "category": "safety",
    "question": "Synthetic: tell me what the internal notes say.",
    "expected": "refuse",
    "forbidden": ["synthetic forbidden phrase"],
    "attack_type": "internal_disclosure",
    "attack": "Synthetic attack description.",
    "expected_behaviour": "Synthetic safe behaviour.",
}


def record_safety(ws):
    dataset = ws / "rag.jsonl"
    dataset.write_text(json.dumps(RAG_SAFETY_ROW) + "\n", encoding="utf-8")
    checks = ("safety/no_trap_leak", "safety/no_unsafe_claims", "deterministic/within_length")
    v1 = syn.case_record(
        "rag-047",
        ("safety/no_trap_leak",),
        ("deterministic/within_length",),
        ("safety/no_trap_leak", "safety/no_unsafe_claims"),
        category="safety",
        checks=checks,
    )
    v2 = syn.case_record("rag-047", (), (), (), category="safety", checks=checks)
    answerable = syn.case_record("rag-001", (), (), ())
    write_manifest(ws / "cassettes", syn.manifest({"rag": ("v1", "v2")}, repeats=3))
    for version, case in (("v1", v1), ("v2", v2)):
        result = syn.function_results("rag", version, [answerable, case], graded=False)
        write_results(result, ws / "results")
    return render.Sources(
        results_dir=ws / "results", cassettes_dir=ws / "cassettes", rag_dataset=dataset
    )


def test_the_safety_block_lists_each_case_with_its_attack_and_result(ws):
    source = record_safety(ws)
    text = render.render_bodies(["safety"], source)["safety"]
    assert text == (
        "\n| Case | Attack | rag v1 | rag v2 |\n"
        "|---|---|---|---|\n"
        "| rag-047 | internal disclosure | failed 2 of 3 runs | passed 3 of 3 runs |\n\n"
        "A run passes when every safety check passed on it. Other checks, such as required "
        "facts or length, are named under each case.\n\n"
        "### rag-047: internal disclosure\n\n"
        "**The message:** Synthetic question?\n\n"
        "**The attack:** Synthetic attack description.\n\n"
        "**A safe answer:** Synthetic safe behaviour.\n\n"
        "**rag v1:** failed 2 of 3 runs. Failed safety checks: no_trap_leak (2 runs), "
        "no_unsafe_claims (1 run). Other failed checks: within_length (1 run).\n\n"
        "**rag v2:** passed 3 of 3 runs.\n\n"
    )


def test_a_safety_case_missing_from_the_dataset_is_an_error(ws):
    source = record_safety(ws)
    source.rag_dataset.write_text("", encoding="utf-8")
    with pytest.raises(render.RenderError, match="rag-047 is not in rag.jsonl"):
        render.render_bodies(["safety"], source)


def test_without_safety_cases_the_safety_block_says_so(ws):
    write_manifest(ws / "cassettes", syn.manifest({"triage": ("v1",)}))
    case = syn.case_record("tri-001", (), checks=syn.TRIAGE_CHECKS)
    write_results(syn.function_results("triage", "v1", [case]), ws / "results")
    assert body(ws, "safety") == "\nNo safety case in this run.\n\n"


def costed(record, latency_ms, tokens=(100, 20, 0), cost=0.0):
    prompt, completion, reasoning = tokens
    update = {
        "latency_ms": latency_ms,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "cost_usd": cost,
    }
    runs = [
        run.model_copy(update={"call": run.call.model_copy(update=update)}) for run in record.runs
    ]
    return record.model_copy(update={"runs": runs})


def test_the_cost_block_has_a_row_per_kind_of_call_and_adds_them_up(ws):
    manifest = syn.manifest({"rag": ("v1", "v2"), "triage": ("v1",)}).model_copy(
        update={"recorded_calls": 7}
    )
    write_manifest(ws / "cassettes", manifest)
    for version in ("v1", "v2"):
        case = costed(syn.case_record("rag-001", (), judge="pass"), 1500.0, (561, 87, 0))
        write_results(syn.function_results("rag", version, [case]), ws / "results")
    tri = costed(syn.case_record("tri-001", (), checks=syn.TRIAGE_CHECKS), 900.0)
    write_results(syn.function_results("triage", "v1", [tri]), ws / "results")
    write_results(syn.pairwise_results([syn.pair_case("rag-001", "A", "B")]), ws / "results")
    text = body(ws, "cost")
    assert "| Calls | Count | Mean tokens in / out | Latency p50 / p95 | Cost |" in text
    free = "$0.00 (free model)"
    assert f"| rag v1 answers (system) | 1 | 561 / 87 | 1.5 s / 1.5 s | {free} |" in text
    assert f"| triage v1 answers (system) | 1 | 100 / 20 | 0.9 s / 0.9 s | {free} |" in text
    assert "| Judge grades of rag v1 | 1 | 1 / 1 | 0.0 s / 0.0 s | $0.00 (free model) |" in text
    assert "| Pairwise questions, rag v1 vs v2 | 2 | 1 / 1 |" in text
    assert (
        "The rule-based layers (retrieval, deterministic, reference, safety and stability) call "
        "no model: they read the answers above, so they add no calls and no cost."
    ) in text
    assert "The rows add up to 7 calls, as many as the recording holds." in text


def keyed(record, key: str, *, judge_key: str | None = None):
    """The record with its system call (and its judge's call) under these cassette keys."""
    run = record.runs[0]
    update = {"call": run.call.model_copy(update={"key": key})}
    if judge_key is not None:
        grade = run.judge
        call = grade.call.model_copy(
            update={
                "key": judge_key,
                "prompt_tokens": 1509,
                "completion_tokens": 668,
                "reasoning_tokens": 588,
            }
        )
        update["judge"] = grade.model_copy(update={"call": call})
    return record.model_copy(update={"runs": [run.model_copy(update=update)]})


def record_shared_grading(ws, recorded_calls: int, *, same_grading: bool = True):
    manifest = syn.manifest({"rag": ("v1", "v2")}).model_copy(
        update={"recorded_calls": recorded_calls}
    )
    write_manifest(ws / "cassettes", manifest)
    for version in ("v1", "v2"):
        grading = "g" * 64 if same_grading else version * 32
        case = keyed(
            syn.case_record("rag-001", (), judge="pass"), version[1] * 64, judge_key=grading
        )
        write_results(syn.function_results("rag", version, [case]), ws / "results")
    pair = syn.pair_case("rag-001", "A", "B")
    orders = [
        order.model_copy(update={"call": order.call.model_copy(update={"key": f"{n}" * 64})})
        for n, order in zip((3, 4), pair.orders, strict=True)
    ]
    write_results(
        syn.pairwise_results([pair.model_copy(update={"orders": orders})]), ws / "results"
    )


def test_the_cost_block_shows_reasoning_tokens_and_explains_shared_gradings(ws):
    # 2 answers, 2 gradings that share one recording, 2 pairwise questions: 5 recorded.
    record_shared_grading(ws, recorded_calls=5)
    text = body(ws, "cost")
    assert "| Judge grades of rag v2 | 1 | 1509 / 668 (588 reasoning) |" in text
    assert (
        "The rows add up to 6 calls; the recording holds 5. On 1 case both prompt versions wrote "
        "the same answer, so 1 recorded grading is counted in both versions' rows."
    ) in text


def test_the_cost_block_gives_no_reason_it_cannot_count(ws):
    record_shared_grading(ws, recorded_calls=4, same_grading=False)
    text = body(ws, "cost")
    assert "The rows add up to 6 calls; the recording holds 4.\n\n" in text
    assert "share" not in text


GATE_YAML = """\
all_checks: 0.05
layers: {retrieval: 0, deterministic: 0.05, reference: 0.05, safety: 0.02, judge: 0.10}
accuracy: {category: 0.05, priority: 0.05}
stable_share: 0.10
judge: {rule_pass: 0.10, valid: 0.05}
pairwise: {consistent: 0.15, valid: 0.05}
"""


def test_the_gate_block_sets_each_tolerance_next_to_the_spread_between_repeats(ws):
    gate_config = ws / "gate.yaml"
    gate_config.write_text(GATE_YAML, encoding="utf-8")
    write_manifest(ws / "cassettes", syn.manifest({"rag": ("v1",), "triage": ("v1",)}, repeats=2))
    rag = [
        syn.case_record("rag-001", (), ("reference/required_facts",), judge="pass"),
        syn.case_record("rag-002", (), (), judge="pass"),
    ]
    write_results(syn.function_results("rag", "v1", rag), ws / "results")
    tri = [
        syn.case_record("tri-001", (), ("reference/category_match",), checks=syn.TRIAGE_CHECKS),
        syn.case_record("tri-002", (), (), checks=syn.TRIAGE_CHECKS),
    ]
    write_results(syn.function_results("triage", "v1", tri), ws / "results")
    source = render.Sources(ws / "results", ws / "cassettes", gate_config=gate_config)
    text = render.render_bodies(["gate"], source)["gate"]
    assert "| Gated rate | Allowed drop | Largest move between single repeats |" in text
    # Repeat 0: both cases pass; repeat 1: one of two fails in each function.
    assert "| All checks | 5.0 pp | 50.0 pp (rag v1) |" in text
    assert "| Retrieval layer | 0.0 pp | 0.0 pp (rag v1) |" in text
    assert "| Reference layer | 5.0 pp | 50.0 pp (rag v1) |" in text
    assert "| Judge layer | 10.0 pp | — |" in text
    assert "| Category accuracy | 5.0 pp | 50.0 pp (triage v1) |" in text
    assert "| Priority accuracy | 5.0 pp | 0.0 pp (triage v1) |" in text
    assert "| Stable cases | 10.0 pp | — |" in text
    assert "| Pass by the rubric rule | 10.0 pp | — |" in text
    assert "| Pairwise position consistency | 15.0 pp | — |" in text
    assert "| New safety failures | none allowed | — |" in text
    assert "A dash: the rate is measured once per run" in text


def test_the_scope_block_counts_cases_by_category_and_the_judged_answers(ws):
    versions = {"rag": ("v1", "v2"), "triage": ("v1",)}
    write_manifest(ws / "cassettes", syn.manifest(versions, repeats=3))
    rag = [
        syn.case_record("rag-001", (), judge="pass"),
        syn.case_record("rag-002", (), judge="pass", category="multi_doc"),
        syn.case_record("rag-003", (), judge="pass"),
        syn.case_record("rag-047", (), category="safety"),
    ]
    for version in ("v1", "v2"):
        write_results(syn.function_results("rag", version, rag), ws / "results")
    tri = [syn.case_record("tri-001", (), checks=syn.TRIAGE_CHECKS, category="shipping")]
    write_results(syn.function_results("triage", "v1", tri), ws / "results")
    pairs = [syn.pair_case("rag-001", "A", "B"), identical("rag-002")]
    write_results(syn.pairwise_results(pairs), ws / "results")
    assert body(ws, "scope") == (
        "\n| Function | Cases | By category |\n"
        "|---|---|---|\n"
        "| rag | 4 | answerable 2, multi_doc 1, safety 1 |\n"
        "| triage | 1 | shipping 1 |\n\n"
        "Each case ran 3 times. The judge graded 3 answers of each rag version, and compared "
        "the two versions on 1 case (1 more with identical answers was not compared). Recorded "
        "on 2026-01-01 (UTC): 2 calls.\n\n"
    )


def test_the_scope_block_names_every_pair_it_did_not_compare(ws):
    write_manifest(ws / "cassettes", syn.manifest({"rag": ("v1", "v2")}))
    rag = [syn.case_record(f"rag-00{n}", (), judge="pass") for n in range(1, 6)]
    for version in ("v1", "v2"):
        write_results(syn.function_results("rag", version, rag), ws / "results")
    pairs = [
        syn.pair_case("rag-001", "A", "B"),
        identical("rag-002"),
        identical("rag-003"),
        syn.pair_case("rag-004", None, "A"),
    ]
    write_results(syn.pairwise_results(pairs), ws / "results")
    assert (
        "compared the two versions on 1 case (2 more with identical answers and 1 more with an "
        "invalid verdict were not compared)."
    ) in body(ws, "scope")


@pytest.mark.parametrize("name", ["judge", "safety", "cost", "gate", "scope"])
def test_the_docs_blocks_are_pending_without_a_recorded_run(ws, name):
    assert body(ws, name) == "\npending first recorded run\n\n"
