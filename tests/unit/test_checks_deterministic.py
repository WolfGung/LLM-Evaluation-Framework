"""Deterministic checks: rules a program can decide without a model.

Synthetic data: every reply and answer below is a made-up string written for
the test, not a model output.
"""

import json

import pytest

from llmeval.checks import CheckResult
from llmeval.checks.deterministic import (
    MAX_ANSWER_WORDS,
    cites_retrieved,
    declines,
    dont_know,
    has_text,
    invented_specifics,
    no_forbidden,
    no_unretrieved_citations,
    redirects,
    triage_checks,
    triage_enums_valid,
    triage_json_valid,
    triage_required_fields,
    triage_schema_valid,
    within_length,
)
from llmeval.checks.retrieval import retrieval_recall, retrieval_recall_value

GOOD = {
    "category": "shipping",
    "priority": "high",
    "order_id": "TS-123456",
    "summary": "Synthetic parcel is late.",
}


def reply(**changes):
    data = {**GOOD, **changes}
    return json.dumps({k: v for k, v in data.items() if v is not ...})


# --- triage -----------------------------------------------------------------


def test_a_valid_reply_passes_every_triage_check():
    results = triage_checks(reply())
    assert [r.name for r in results] == [
        "json_valid",
        "required_fields",
        "enums_valid",
        "schema_valid",
    ]
    assert all(r.passed for r in results)


def test_a_fenced_json_reply_is_read_like_the_parser_reads_it():
    assert all(r.passed for r in triage_checks(f"```json\n{reply()}\n```"))


@pytest.mark.parametrize(
    ("raw", "detail"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("Sure! Here it is: {}", "not valid JSON"),
        ("[1, 2]", "not a JSON object"),
        ("{'category': 'shipping'}", "not valid JSON"),
    ],
)
def test_json_valid_fails_with_a_reason(raw, detail):
    result = triage_json_valid(raw)
    assert not result.passed
    assert detail in result.detail


def test_required_fields_names_what_is_missing():
    result = triage_required_fields(reply(order_id=..., summary=...))
    assert not result.passed
    assert "order_id" in result.detail and "summary" in result.detail


def test_a_null_order_id_counts_as_present():
    assert triage_required_fields(reply(order_id=None)).passed


def test_enums_valid_names_the_bad_values():
    result = triage_enums_valid(reply(category="billing", priority="asap"))
    assert not result.passed
    assert "billing" in result.detail and "asap" in result.detail


def test_enums_valid_fails_when_a_field_is_missing():
    assert not triage_enums_valid(reply(priority=...)).passed


def test_schema_valid_catches_what_the_other_checks_allow():
    # Valid JSON, all fields, valid enums, but an order id in the wrong form
    # and an extra field: only the full schema check fails.
    raw = json.dumps({**GOOD, "order_id": "ts 123456", "confidence": 0.9})
    results = {r.name: r for r in triage_checks(raw)}
    assert results["json_valid"].passed
    assert results["required_fields"].passed
    assert results["enums_valid"].passed
    assert not results["schema_valid"].passed
    assert "order_id" in results["schema_valid"].detail


def test_checks_on_non_json_fail_without_raising():
    results = triage_checks("I cannot triage this.")
    assert not any(r.passed for r in results)


def test_triage_schema_valid_alone():
    assert triage_schema_valid(reply()).passed
    assert not triage_schema_valid(reply(summary="x" * 201)).passed


# --- RAG answers ------------------------------------------------------------


def test_has_text():
    assert has_text("Returns take 30 days [kb-returns].").passed
    result = has_text("  ", "length")
    assert not result.passed
    assert "length" in result.detail


def test_cites_retrieved():
    assert cites_retrieved(("kb-returns",), ("kb-returns", "kb-refunds")).passed
    assert not cites_retrieved((), ("kb-returns",)).passed
    assert not cites_retrieved(("kb-warranty",), ("kb-returns",)).passed


def test_no_unretrieved_citations():
    assert no_unretrieved_citations(("kb-returns",), ("kb-returns",)).passed
    assert no_unretrieved_citations((), ("kb-returns",)).passed
    result = no_unretrieved_citations(("kb-returns", "kb-made-up"), ("kb-returns",))
    assert not result.passed
    assert "kb-made-up" in result.detail


def test_forbidden_phrases_take_alternatives():
    result = no_forbidden("We'll match the price.", ("we price match|we will match the price",))
    assert not result.passed
    assert "we price match|we will match the price" in result.detail


@pytest.mark.parametrize(
    "text",
    [
        "I don't know whether we sharpen blades.",
        "I'm not sure if we price match.",
    ],
)
def test_a_forbidden_phrase_inside_a_whether_or_if_clause_is_not_a_claim(text):
    assert no_forbidden(text, ("we sharpen", "we price match")).passed


@pytest.mark.parametrize(
    "text",
    [
        "I don't know if we price match.",
        "I can't confirm that we price match.",
        "I'm not sure we price match.",
        "Sorry, I'm not sure, but I do not think we price match.",
        # Review round 4 (L2): the new decline idioms hedge their own clause.
        "I'm unsure whether we price match.",
        "I have no idea if we price match.",
        # Review round 4 (L1): "and" with no subject after it is not a break.
        "I'm not sure we price match and refund the difference.",
    ],
)
def test_a_hedged_mention_is_not_a_claim(text):
    assert no_forbidden(text, ("we price match",)).passed


@pytest.mark.parametrize(
    "text",
    [
        "If so, we price match.",
        "If you like, we price match.",
        "I'm not sure about chisels. We price match.",
        "We price match, if I remember right.",
        "Whether or not you ask, we price match.",
        "I'm not sure about online orders, but we price match in store.",
        "I don't know, however we price match.",
        "Note that we price match.",
        "I can confirm that we price match.",
        # Review round 4 (L1): a dash, or "and" plus a subject, starts a clause.
        "I'm not sure about online orders and we price match in store.",
        "I don't know the details and you can trust that we price match.",
        "I don't know the exact price — we price match in store.",
        "I'm not sure about online orders – we price match in store.",
        "I'm not sure about online orders - we price match in store.",
        # Review round 4 (L2): a decline idiom hedges only its own clause.
        "I'm unsure about online orders, but we price match in store.",
    ],
)
def test_an_unhedged_clause_is_a_claim(text):
    assert not no_forbidden(text, ("we price match",)).passed


def test_a_forbidden_claim_after_a_hedge_elsewhere_still_counts():
    text = "We sharpen saw blades for free. Not sure if chisels too."
    assert not no_forbidden(text, ("we sharpen",)).passed


def test_no_forbidden_matches_after_normalisation():
    result = no_forbidden("Use code tools50 for 50 % off!", ("TOOLS50", "50%"))
    assert not result.passed
    assert "TOOLS50" in result.detail and "50%" in result.detail
    assert no_forbidden("Returns take 30 days.", ("60 days",)).passed
    assert no_forbidden("anything", ()).passed


def test_within_length():
    assert within_length("word " * MAX_ANSWER_WORDS).passed
    result = within_length("word " * (MAX_ANSWER_WORDS + 1))
    assert not result.passed
    assert str(MAX_ANSWER_WORDS + 1) in result.detail
    assert within_length("a b c", limit=2).passed is False


# --- "I don't know" ---------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "I don't know.",
        "I do not know whether we offer that.",
        "I'm not sure, sorry.",
        "I can't find that in our documents.",
        "I couldn't find any information about price matching.",
        "The documents don't say anything about rentals.",
        "Our information does not mention student discounts.",
        "I don't have information about that.",
        "That isn't covered in the documents I have.",
        "This is not mentioned in the documents.",
        "There is nothing in my documents about tool rental.",
        "That information is unavailable.",
        "I’m not sure — please contact support.",
        # Review round 2 (N1): honest declines in v1-style wording.
        "There's no mention of price matching in our documents",
        "I don't see anything about tool rental in the documents",
        "Sorry, I don't have that information",
        "I don't have enough information to answer that",
        "That information isn't in the documents I have",
        "This is not covered in the provided documents",
        "Our help pages don't say",
        "Our documentation doesn't say",
        "None of the documents mention tool rental",
        "There is nothing about tool rental in my documents",
        "I'm not aware of a price-match policy",
        "I have no details about rentals",
        # Review round 3 (NB5).
        "I don't have pricing for blade sharpening in my documents.",
        "There are no details about refurbished tools in the documents.",
        "As far as I can tell from the documents, no price-match policy is mentioned.",
        "The documents don't seem to mention rentals.",
        "That's not something I have information about",
        "I'm sorry, I can't help with that question based on the documents I have",
        "Our records don't show a sharpening service",
        # Review round 4 (L2).
        "I'm unsure whether Toolshop price matches.",
        "I have no idea whether students get a discount.",
        "I wasn't able to find anything about Sunday delivery.",
        "We were not able to confirm that.",
    ],
)
def test_decline_phrasings_are_recognised(answer):
    assert declines(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "Yes, we price match any competitor.",
        "Returns are accepted within 30 days [kb-returns].",
        "I know exactly: it costs $10.",
        "",
        # A redirect alone is not a decline (review round 1).
        "Please contact Toolshop support for help with that.",
        "You could reach out to our customer support team.",
        "Yes! Tile cutters can be rented. Ask our support team to book one.",
        # A bare negation is a policy statement, not a decline.
        "Refurbished tools are not covered by the warranty [kb-warranty].",
        "Rust is not covered.",
    ],
)
def test_answers_that_do_not_decline(answer):
    assert not declines(answer)


def test_a_redirect_is_a_separate_signal():
    assert redirects("Please reach out to our customer support team.")
    assert redirects("Contact support.")
    assert not redirects("I don't know.")
    result = dont_know("Please contact support.", ())
    assert not result.passed
    assert "redirects to support" in result.detail


SOURCES = (
    "Contacting support\n\nphone: +1 555 0199, Monday to Friday 8:00 to 20:00.",
    "Do you deliver on Sundays?",
)


def test_specifics_from_the_sources_or_the_question_are_not_invented():
    answer = "I don't know. Call +1-555-0199, Monday to Friday from 8 am."
    assert invented_specifics(answer, SOURCES) == []


def test_invented_numbers_and_days_are_listed():
    answer = "I'm not sure, but delivery on Sunday costs $9.99 and takes 2 hours."
    assert invented_specifics(answer, SOURCES) == ["$9.99", "2"]


def test_dont_know_passes_a_decline_without_invented_specifics():
    result = dont_know("I don't know. Please contact support at +1 555 0199.", SOURCES)
    assert result.passed


def test_dont_know_fails_when_nothing_is_declined():
    result = dont_know("Yes, we deliver on Sundays.", SOURCES)
    assert not result.passed
    assert "no decline" in result.detail


def test_dont_know_fails_a_decline_that_still_invents():
    result = dont_know("I'm not sure, but Sunday delivery is $9.99.", SOURCES)
    assert not result.passed
    assert "$9.99" in result.detail


def test_dont_know_does_not_depend_on_one_wording():
    # The same behaviour in different words passes; the exact sentence of a
    # prompt version is not what is measured.
    for answer in ("I don't know.", "The documents don't say.", "I can't find that."):
        assert dont_know(answer, SOURCES).passed


def test_dont_know_fails_an_empty_answer():
    assert not dont_know("", SOURCES).passed


# --- retrieval --------------------------------------------------------------


def test_retrieval_recall_passes_when_every_expected_document_is_retrieved():
    assert retrieval_recall(("kb-a", "kb-b"), ("kb-b", "kb-c", "kb-a")).passed


def test_retrieval_recall_names_the_missing_documents():
    result = retrieval_recall(("kb-a", "kb-b"), ("kb-b",))
    assert not result.passed
    assert "kb-a" in result.detail
    assert "1 of 2" in result.detail


def test_retrieval_recall_value():
    assert retrieval_recall_value(("kb-a", "kb-b"), ("kb-b",)) == 0.5
    assert retrieval_recall_value((), ("kb-b",)) is None


def test_check_results_are_plain_records():
    result = CheckResult("x", True, "fine")
    assert (result.name, result.passed, result.detail) == ("x", True, "fine")
    assert result.to_dict() == {"name": "x", "passed": True, "detail": "fine"}


def test_the_detectors_say_what_they_do_not_measure():
    # Declines plus numeric or listed invention; other wording is the judge's.
    for check in (dont_know, no_forbidden):
        assert "judge layer" in " ".join(check.__doc__.split())
