"""The support assistant: messages, citations, empty answers.

Uses the synthetic fake model from `fakes.py`; the knowledge base is real.
"""

import pytest

from app.assistant import RAG_FUNCTION, AssistantAnswer, answer, extract_citations, prepare
from app.prompting import PromptError, load_prompt
from app.retrieval import BM25Index, Document, search
from llmeval.cassettes import CallTag
from llmeval.client import MissingRecording
from tests.app.fakes import STRUCTURED, UNSTRUCTURED, FakeModel

QUESTION = "How many days do I have to return an item?"


def test_prepare_puts_policy_and_documents_in_the_system_turn_and_the_question_alone():
    messages, hits = prepare(QUESTION, "v1")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == QUESTION
    system = messages[0]["content"]
    assert system.startswith(load_prompt("assistant", "v1").split("{{documents}}")[0])
    assert QUESTION not in system
    assert hits == tuple(search(QUESTION))
    for hit in hits:
        assert f'<document id="{hit.doc_id}">\n{hit.text}\n</document>' in system


def test_documents_appear_in_rank_order():
    messages, hits = prepare(QUESTION, "v2")
    system = messages[0]["content"]
    positions = [system.index(f'id="{hit.doc_id}"') for hit in hits]
    assert positions == sorted(positions)


def test_versions_use_their_own_prompt():
    v1, _ = prepare(QUESTION, "v1")
    v2, _ = prepare(QUESTION, "v2")
    assert v1[0]["content"] != v2[0]["content"]
    assert v1[1] == v2[1]


def test_no_matching_document_is_said_plainly():
    messages, hits = prepare("zzzz qqqq", "v2")
    assert hits == ()
    assert "No documents matched the question." in messages[0]["content"]


def test_k_and_the_index_can_be_chosen():
    index = BM25Index([Document(doc_id="kb-one", title="One", body="return policy")])
    messages, hits = prepare(QUESTION, "v1", index=index, k=5)
    assert [h.doc_id for h in hits] == ["kb-one"]
    assert 'id="kb-one"' in messages[0]["content"]


@pytest.mark.parametrize("question", ["", "   \n"])
def test_a_blank_question_is_refused(question):
    with pytest.raises(ValueError, match="question"):
        prepare(question, "v1")


def test_an_unknown_version_is_refused():
    with pytest.raises(PromptError):
        prepare(QUESTION, "v9")


def test_answer_returns_text_citations_retrieved_ids_and_the_call():
    model = FakeModel(reply="You have 30 days [kb-returns]. Refunds take 5 days [kb-refunds].")
    result = answer(model, UNSTRUCTURED, QUESTION, "v2")
    assert isinstance(result, AssistantAnswer)
    assert result.text == model.reply
    assert result.cited_ids == ("kb-returns", "kb-refunds")
    assert result.retrieved_ids == tuple(h.doc_id for h in search(QUESTION))
    assert result.hits == tuple(search(QUESTION))
    assert result.call.content == model.reply
    assert result.call.model_requested == UNSTRUCTURED.model


def test_answer_sends_the_prepared_messages_without_a_schema():
    model = FakeModel(reply="ok")
    answer(model, UNSTRUCTURED, QUESTION, "v1", repeat=2, case="rag-007")
    [call] = model.calls
    assert call["messages"] == prepare(QUESTION, "v1")[0]
    assert call["response_format"] is None
    assert call["repeat"] == 2
    assert call["tag"] == CallTag(function=RAG_FUNCTION, case="rag-007", version="v1")
    assert call["role"] is UNSTRUCTURED


def test_a_call_without_a_case_is_tagged_adhoc():
    model = FakeModel(reply="ok")
    answer(model, STRUCTURED, QUESTION, "v2")
    assert model.calls[0]["tag"] == CallTag(function="rag", case="adhoc", version="v2")
    assert model.calls[0]["response_format"] is None


def test_an_empty_answer_is_returned_not_raised():
    model = FakeModel(reply="", finish_reason="length")
    result = answer(model, UNSTRUCTURED, QUESTION, "v2")
    assert result.text == ""
    assert result.cited_ids == ()
    assert result.call.empty_reason == "length"
    assert result.retrieved_ids


def test_a_missing_recording_propagates():
    model = FakeModel(error=MissingRecording("k" * 64, 0, None))
    with pytest.raises(MissingRecording):
        answer(model, UNSTRUCTURED, QUESTION, "v1")


@pytest.mark.parametrize(
    ("text", "ids"),
    [
        ("Thirty days [kb-returns].", ("kb-returns",)),
        ("Both [kb-returns][kb-refunds] and again [kb-returns].", ("kb-returns", "kb-refunds")),
        ("One bracket [kb-returns, kb-refunds].", ("kb-returns", "kb-refunds")),
        ("Semicolons [kb-returns; kb-delivery-times].", ("kb-returns", "kb-delivery-times")),
        ("An id the search never returned [kb-made-up].", ("kb-made-up",)),
        ("Not citations: [1], [note], (kb-returns), kb-refunds, [KB-RETURNS].", ()),
        ("Mixed bracket [see kb-warranty].", ("kb-warranty",)),
        ("", ()),
    ],
)
def test_extract_citations(text, ids):
    assert extract_citations(text) == ids
