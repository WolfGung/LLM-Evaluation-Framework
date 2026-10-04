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
