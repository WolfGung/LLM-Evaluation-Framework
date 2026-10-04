"""Cheap checks of the judge's own biases: position bias, length bias, score-length correlation.

Synthetic data: every preference, score and answer below is made up for the
test. Nothing is written anywhere.
"""

from datetime import UTC, datetime

import pytest

from llmeval.checks.judge import inconsistency, position_bias, spearman
from llmeval.results import CallRecord, JudgeRecord, summarise_judge

CALL = CallRecord(
    key="0" * 64,
    model_used="synthetic/judge:free",
    latency_ms=1.0,
    prompt_tokens=10,
    completion_tokens=5,
    reasoning_tokens=0,
    cost_usd=0.0,
    cost_source="provider",
    finish_reason="stop",
    empty_reason=None,
    recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
)


def graded(groundedness, helpfulness=4, tone=5):
    return JudgeRecord(
        scores={"groundedness": groundedness, "helpfulness": helpfulness, "tone": tone},
        judge_pass=True,
        rule_pass=True,
        reasons="Synthetic.",
        error=None,
        detail=None,
        raw=None,
        call=CALL,
    )


INVALID = JudgeRecord(
    scores=None,
    judge_pass=None,
    rule_pass=None,
    reasons=None,
    error="empty",
    detail="the judge returned no text (length)",
    raw="",
    call=CALL,
)


# --- Spearman correlation -------------------------------------------------------


def test_spearman_of_a_monotone_relation_is_one():
    assert spearman([10, 20, 30, 40], [1, 2, 4, 5]) == 1.0
    assert spearman([10, 20, 30, 40], [5, 4, 2, 1]) == -1.0


def test_spearman_ranks_ties_by_their_average():
    # Ranks: x = 1, 2, 3, 4; y = 1.5, 1.5, 3.5, 3.5. Pearson of the ranks = 0.8944.
    assert spearman([10, 20, 30, 40], [3, 3, 5, 5]) == 0.8944


def test_spearman_is_undefined_without_spread_or_with_too_few_points():
    assert spearman([10, 20, 30], [4, 4, 4]) is None  # every score the same
    assert spearman([7, 7, 7], [1, 2, 3]) is None  # every length the same
    assert spearman([10, 20], [1, 2]) is None  # two points always line up
    assert spearman([], []) is None


def test_spearman_needs_pairs():
    with pytest.raises(ValueError, match="same length"):
        spearman([1, 2, 3], [1, 2])


# --- position bias ----------------------------------------------------------------


def test_a_judge_without_position_bias_scores_one_half():
    # Consistent pairs: the better answer wins in both positions.
    assert position_bias([("A", "B"), ("B", "A"), ("B", "A")]) == (3, 6)


def test_a_judge_that_always_picks_a_scores_one():
    assert position_bias([("A", "A"), ("A", "A")]) == (4, 4)
    assert position_bias([("B", "B")]) == (0, 2)


def test_position_bias_leaves_out_ties_and_invalid_verdicts():
    # Only pairs where both orders chose a side keep the two positions balanced.
    assert position_bias([("A", "tie"), ("tie", "tie"), (None, "A"), ("A", "A")]) == (2, 2)


def test_inconsistent_pairs_say_how_they_disagree():
    assert inconsistency("A", "A") == "same_position_a"
    assert inconsistency("B", "B") == "same_position_b"
    assert inconsistency("A", "tie") == inconsistency("tie", "B") == "tie_in_one_order"
    assert inconsistency("A", "B") is None  # consistent: v1 both times
    assert inconsistency("tie", "tie") is None
    assert inconsistency(None, "A") is None  # invalid, not inconsistent


# --- length and score -------------------------------------------------------------


def test_the_judge_summary_reports_the_length_score_correlation():
    answers = ["one two", "one two three four", "a b c d e f", "a b c d e f g h"]
    records = [graded(2, helpfulness=3), graded(3), graded(4, helpfulness=3), graded(5)]
    summary = summarise_judge(list(zip(answers, records, strict=True)))
    correlation = summary.length_score_correlation
    assert correlation["groundedness"].model_dump() == {"n": 4, "spearman": 1.0}
    assert correlation["tone"].model_dump() == {"n": 4, "spearman": None}  # all 5
    assert correlation["helpfulness"].n == 4


def test_empty_answers_and_invalid_verdicts_stay_out_of_the_correlation():
    # An empty answer is short and unhelpful for a reason that is not verbosity.
    pairs = [
        ("", graded(5, helpfulness=1)),
        ("one", graded(2)),
        ("one two", graded(3)),
        ("one two three", graded(4)),
        ("one two three four", INVALID),
    ]
    summary = summarise_judge(pairs)
    assert summary.length_score_correlation["groundedness"].model_dump() == {
        "n": 3,
        "spearman": 1.0,
    }
    assert summary.judged == 5


def test_empty_answers_are_counted_apart_from_the_scores():
    # The rubric scores an empty answer 1 on every criterion; that says nothing
    # about the answers the system did write, so the scores leave it out.
    pairs = [
        ("", graded(1, helpfulness=1, tone=1)),
        ("one two", graded(5, helpfulness=3, tone=4)),
        ("   ", INVALID),
    ]
    summary = summarise_judge(pairs)
    assert summary.judged == 3
    assert summary.empty_answers == 2
    assert summary.mean_scores == {"groundedness": 5.0, "helpfulness": 3.0, "tone": 4.0}
    assert summary.score_counts["groundedness"] == {"1": 0, "2": 0, "3": 0, "4": 0, "5": 1}
    # The pass rule still counts the empty answer's verdict: that run failed.
    assert summary.rule_pass.total == 2
