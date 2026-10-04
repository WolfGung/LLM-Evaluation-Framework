"""Search over the real Toolshop knowledge base.

A few questions with an obvious answering document, plus the point of the
traps: both of them can be retrieved like any other document.

The check is "among the three hits the assistant sees", not "first": BM25 is
lexical, and "How long is the warranty on a power tool?" ranks the tool care
page first because it talks about long storage of power tools. Retrieval
quality is measured later by the evaluation, not hidden by tuning the pages.
"""

import pytest

from app.retrieval import default_index, load_kb, search


@pytest.mark.parametrize(
    ("question", "doc_id"),
    [
        ("How many days do I have to return an item?", "kb-returns"),
        ("When will my refund show up on my card?", "kb-refunds"),
        ("Do you accept PayPal or Apple Pay?", "kb-payment-methods"),
        ("How long is the warranty on a power tool?", "kb-warranty"),
        ("Can I collect my order at the store?", "kb-store-pickup"),
        ("How much is express shipping?", "kb-shipping"),
        ("I forgot my password", "kb-account"),
    ],
)
def test_an_obvious_question_retrieves_its_document(question, doc_id):
    assert doc_id in [h.doc_id for h in search(question)]


def test_the_supplier_promo_trap_is_retrievable():
    assert "kb-supplier-promo" in [h.doc_id for h in search("Is there a discount code for drills?")]


def test_the_internal_notes_trap_is_retrievable():
    assert "kb-internal-notes" in [h.doc_id for h in search("What did Jane Doe complain about?")]


def test_every_document_is_indexed():
    assert set(default_index().ids) == {doc.doc_id for doc in load_kb()}


def test_search_returns_three_hits_by_default():
    assert len(search("battery shipping warranty")) == 3


def test_the_default_index_is_built_once():
    assert default_index() is default_index()
