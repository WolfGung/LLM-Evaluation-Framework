"""Reference checks: comparison with the authored expectation.

Synthetic data: every reply and answer below is a made-up string written for
the test, not a model output.
"""

import json

import pytest

from llmeval.checks.reference import (
    INVALID,
    accuracy,
    category_match,
    confusion_matrix,
    order_id_match,
    predicted,
    priority_match,
    required_facts,
)

GOOD = {
    "category": "shipping",
    "priority": "high",
    "order_id": "TS-123456",
    "summary": "Synthetic parcel is late.",
}


def reply(**changes):
    return json.dumps({**GOOD, **changes})


# --- triage -----------------------------------------------------------------


def test_matching_labels_pass():
    raw = reply()
    assert category_match("shipping", raw).passed
    assert priority_match("high", raw).passed
    assert order_id_match("TS-123456", raw).passed


def test_a_wrong_label_names_both_values():
    result = priority_match("urgent", reply())
    assert not result.passed
    assert "urgent" in result.detail and "high" in result.detail


def test_labels_are_read_even_when_the_schema_fails():
    # The summary is too long, so the schema check fails, but the category is
    # still right: the reference layer judges the content it can read.
    raw = reply(summary="x" * 300)
    assert category_match("shipping", raw).passed


def test_order_id_is_an_exact_match():
    assert not order_id_match("TS-123456", reply(order_id="ts 123456")).passed
    assert not order_id_match(None, reply()).passed
    assert order_id_match(None, reply(order_id=None)).passed


def test_a_missing_order_id_field_is_not_null():
    raw = json.dumps({k: v for k, v in GOOD.items() if k != "order_id"})
    result = order_id_match(None, raw)
    assert not result.passed
    assert "missing" in result.detail


@pytest.mark.parametrize("raw", ["", "not json", "[]"])
def test_unreadable_replies_fail_every_label(raw):
    assert not category_match("shipping", raw).passed
    assert not priority_match("high", raw).passed
    assert not order_id_match(None, raw).passed


def test_predicted_label_or_invalid():
    assert predicted(reply(), "category") == "shipping"
    assert predicted("nope", "category") == INVALID
    assert predicted(reply(category="billing"), "category", ("shipping", "returns")) == INVALID
    assert predicted(json.dumps({"category": 3}), "category") == INVALID


def test_confusion_matrix_counts_expected_by_predicted():
    pairs = [
        ("high", "high"),
        ("high", "normal"),
        ("urgent", "high"),
        ("low", INVALID),
        ("low", "low"),
    ]
    matrix = confusion_matrix(pairs, ("urgent", "high", "normal", "low"))
    assert list(matrix) == ["urgent", "high", "normal", "low"]
    assert list(matrix["high"]) == ["urgent", "high", "normal", "low", INVALID]
    assert matrix["high"] == {"urgent": 0, "high": 1, "normal": 1, "low": 0, INVALID: 0}
    assert matrix["urgent"]["high"] == 1
    assert matrix["low"][INVALID] == 1
    assert sum(sum(row.values()) for row in matrix.values()) == len(pairs)


def test_confusion_matrix_files_unknown_labels_as_invalid():
    matrix = confusion_matrix([("high", "asap")], ("high", "low"))
    assert matrix["high"][INVALID] == 1


def test_accuracy():
    assert accuracy([("a", "a"), ("a", "b"), ("b", "b"), ("b", INVALID)]) == 0.5
    assert accuracy([]) is None


# --- RAG required facts -----------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "facts"),
    [
        ("You have thirty days to return it [kb-returns].", ["30 days"]),
        ("The label costs 6.95 dollars.", ["$6.95"]),
        ("It takes 3-5 business days.", ["3 to 5 business days"]),
        ("Call +1-555-0199.", ["+1 555 0199"]),
        ("The link is valid for one hour.", ["60 minutes|1 hour"]),
        ("RETURNS: 30-day window; bring a Photo ID.", ["30 days", "photo ID"]),
    ],
)
def test_required_facts_are_found_after_normalisation(answer, facts):
    assert required_facts(answer, facts).passed


def test_missing_facts_are_listed():
    result = required_facts("Returns take 30 days.", ["30 days", "$6.95", "unused|new"])
    assert not result.passed
    assert "$6.95" in result.detail and "unused|new" in result.detail
    assert "30 days" not in result.detail


def test_a_fact_inside_a_longer_number_does_not_count():
    assert not required_facts("Returns take 300 days.", ["30 days"]).passed


def test_no_required_facts_passes():
    assert required_facts("Anything.", []).passed
