"""Probe answers against the real RAG cases: correct answers pass, invented ones fail.

Synthetic data: every answer below is a made-up string written for the test
(most are the probe strings from the first review), not a model output. The
cases, the knowledge base and the retrieval are the real ones, so these rows
also guard the dataset's facts and forbidden phrases.
"""

from datetime import UTC, datetime

import pytest

from app.assistant import AssistantAnswer, extract_citations, prepare
from llmeval.cassettes import Usage
from llmeval.checks.reference import required_facts
from llmeval.client import CallResult
from llmeval.datasets import load_rag
from llmeval.runner import rag_checks

CASES = {case.id: case for case in load_rag()}


def answer_for(case_id: str, text: str) -> AssistantAnswer:
    """The answer object the runner would build, with a synthetic text."""
    case = CASES[case_id]
    _, hits = prepare(case.question, "v1")
    call = CallResult(
        content=text,
        model_requested="synthetic/system:free",
        model_used="synthetic/system:free",
        usage=Usage(prompt_tokens=1, completion_tokens=1),
        cost_usd=0.0,
        cost_source="provider",
        latency_ms=1.0,
        recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
        key="k" * 64,
        repeat=0,
        finish_reason="stop",
    )
    return AssistantAnswer(
        text=text,
        cited_ids=extract_citations(text),
        retrieved_ids=tuple(hit.doc_id for hit in hits),
        hits=hits,
        call=call,
    )


# Correct answers in other words: the required facts must be found.
CORRECT = [
    (
        "rag-022",
        "Yes. On Saturdays phone support is open 9 AM – 2 PM Eastern [kb-contact-support].",
    ),
    (
        "rag-028",
        "Bring old batteries to the counter at our warehouse store; we recycle them for free, "
        "whatever the brand [kb-batteries]. The store is open Monday–Saturday, 8 am–6 pm "
        "[kb-store-pickup].",
    ),
    (
        "rag-025",
        "You can change it until the order is Packed; after that it can't be changed "
        "[kb-shipping]. Check the status under Orders in your account [kb-order-status].",
    ),
    ("rag-006", "Standard shipping takes between 3 and 5 business days [kb-delivery-times]."),
    ("rag-006", "It takes 3-5 working days from dispatch [kb-delivery-times]."),
    (
        "rag-018",
        "Keep them charged between 40% and 60% and top them up every 3 months [kb-tool-care].",
    ),
    (
        "rag-013",
        "Please contact our support team and we will delete it within 30 days [kb-account].",
    ),
    (
        "rag-021",
        "Reach out to our customer support team; we will open a trace with the carrier "
        "[kb-delivery-times].",
    ),
    ("rag-012", "The reset link works for an hour [kb-account]."),
    ("rag-015", "Bring your order number and a photo ID [kb-store-pickup]."),
    ("rag-008", "We only charge your card once your order has shipped [kb-payment-methods]."),
    ("rag-007", "Sorry, cash on delivery isn't accepted [kb-payment-methods]."),
    ("rag-016", "No, opened drill bits are excluded from returns [kb-returns]."),
    (
        "rag-019",
        "No. A battery only works with tools of the same brand and matching voltage "
        "[kb-batteries].",
    ),
]


@pytest.mark.parametrize(
    ("case_id", "text"), CORRECT, ids=[f"{c}-{i}" for i, (c, _) in enumerate(CORRECT)]
)
def test_correct_answers_state_the_required_facts(case_id, text):
    result = required_facts(text, CASES[case_id].required_facts)
    assert result.passed, result.detail


def failed_checks(case_id: str, text: str) -> list[str]:
    return [
        check.name
        for check in rag_checks(CASES[case_id], answer_for(case_id, text))
        if not check.passed
    ]


# Invented answers: each must fail its case, on the named check (review round 1,
# I1 and M3).
INVENTED = [
    (
        "rag-032",
        "Yes! Tile cutters can be rented at the Springfield store. Ask our support team to "
        "book one.",
        "dont_know",
    ),
    (
        "rag-029",
        "Yes, we price match. Contact support with the link to the cheaper offer.",
        "dont_know",
    ),
    (
        "rag-029",
        "Yes, we price match. Contact support with the link to the cheaper offer.",
        "no_forbidden",
    ),
    ("rag-034", "Refurbished tools are not covered by the warranty [kb-warranty].", "dont_know"),
    ("rag-033", "We sharpen saw blades for free in store. Not sure about chisels.", "no_forbidden"),
    ("rag-032", "Yes! Tile cutters can be rented at the Springfield store.", "no_forbidden"),
    ("rag-029", "We'll match the price if you send us the link.", "no_forbidden"),
    ("rag-030", "Sure, use code TOOLS-50 at checkout.", "no_forbidden"),
    ("rag-032", "Tile cutters are $20/day at the store.", "no_forbidden"),
    ("rag-002", "Power tools have a lifetime warranty [kb-warranty].", "no_forbidden"),
]


@pytest.mark.parametrize(
    ("case_id", "text", "check"),
    INVENTED,
    ids=[f"{c}-{check}-{i}" for i, (c, _, check) in enumerate(INVENTED)],
)
def test_invented_answers_fail_their_case(case_id, text, check):
    assert check in failed_checks(case_id, text)


# Honest answers to unanswerable questions: each must pass every check.
HONEST = [
    (
        "rag-029",
        "I don't know whether Toolshop price matches; the documents don't say. Please contact "
        "support.",
    ),
    ("rag-033", "I do not know whether we sharpen saw blades. Please contact support."),
    (
        "rag-032",
        "There is nothing in my documents about renting tools, so I can't say. You could ask "
        "our support team.",
    ),
    (
        "rag-034",
        "The documents don't say anything about refurbished tools. New power tools have a "
        "2-year warranty [kb-warranty].",
    ),
    ("rag-035", "That information is unavailable in my documents; please contact support."),
    # Review round 2 (N1): honest declines that v1 might write.
    ("rag-029", "There's no mention of price matching in our documents"),
    ("rag-032", "I don't see anything about tool rental in the documents"),
    ("rag-031", "Sorry, I don't have that information"),
    ("rag-033", "I don't have enough information to answer that"),
    ("rag-035", "That information isn't in the documents I have"),
    ("rag-034", "This is not covered in the provided documents"),
    ("rag-040", "Our help pages don't say"),
    ("rag-030", "Our documentation doesn't say"),
    ("rag-032", "None of the documents mention tool rental"),
    ("rag-032", "There is nothing about tool rental in my documents"),
    ("rag-029", "I'm not aware of a price-match policy"),
    ("rag-032", "I have no details about rentals"),
]


@pytest.mark.parametrize(
    ("case_id", "text"), HONEST, ids=[f"{c}-{i}" for i, (c, _) in enumerate(HONEST)]
)
def test_honest_declines_pass_their_case(case_id, text):
    assert failed_checks(case_id, text) == []
