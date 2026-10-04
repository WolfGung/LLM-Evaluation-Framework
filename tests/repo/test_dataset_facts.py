"""Required facts in the RAG dataset come from the knowledge base.

A required fact must be stated in the case's expected documents (after the
same normalisation the reference check uses), not taken from the author's
head. No model is called.
"""

from pathlib import Path

import pytest

from app.retrieval import load_kb
from llmeval.checks.text import contains, normalise
from llmeval.datasets import RAG_PATH, load_rag

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def rag_cases():
    return load_rag(ROOT / RAG_PATH)


@pytest.fixture(scope="module")
def kb():
    return {doc.doc_id: normalise(doc.text) for doc in load_kb()}


def test_every_required_fact_is_stated_in_an_expected_document(rag_cases, kb):
    # A required fact must come from the documents, not from the author's
    # head: one of its alternatives appears, after normalisation, in one of
    # the case's expected documents.
    for case in rag_cases:
        sources = [kb[doc_id] for doc_id in case.expected_docs]
        for alternatives in case.fact_alternatives:
            assert any(
                contains(source, normalise(alt)) for alt in alternatives for source in sources
            ), f"{case.id}: {' | '.join(alternatives)}"


def test_multi_document_facts_need_every_document(rag_cases, kb):
    # In a multi-document case, each expected document holds at least one
    # required fact; otherwise the case is answerable from fewer documents.
    for case in (c for c in rag_cases if c.category == "multi_doc"):
        for doc_id in case.expected_docs:
            assert any(
                contains(kb[doc_id], normalise(alt))
                for alternatives in case.fact_alternatives
                for alt in alternatives
            ), f"{case.id}: {doc_id} holds none of the required facts"


def _holds(doc: str, alternatives: tuple[str, ...]) -> bool:
    return any(contains(doc, normalise(alt)) for alt in alternatives)


def test_no_single_document_answers_a_multi_document_case(rag_cases, kb):
    # Otherwise the case is answerable from one document and measures nothing
    # a single-document case does not.
    for case in (c for c in rag_cases if c.category == "multi_doc"):
        for doc_id in case.expected_docs:
            assert not all(_holds(kb[doc_id], alts) for alts in case.fact_alternatives), (
                f"{case.id}: {doc_id} alone holds every required fact"
            )


def test_required_facts_are_not_words_of_the_question(rag_cases):
    # A fact the question already states is passed by echoing the question.
    for case in rag_cases:
        question = normalise(case.question)
        for alternatives in case.fact_alternatives:
            echoed = [alt for alt in alternatives if contains(question, normalise(alt))]
            assert not echoed, f"{case.id}: {', '.join(echoed)} is in the question"
