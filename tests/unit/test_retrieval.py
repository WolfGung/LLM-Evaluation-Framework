"""BM25 search over a tiny corpus.

Synthetic data: the documents below are made up for these tests and have
nothing to do with the Toolshop knowledge base.
"""

import math

import pytest

from app.retrieval import BM25Index, Document, Hit, tokenize


def doc(doc_id: str, body: str, title: str = "Note") -> Document:
    return Document(doc_id=doc_id, title=title, body=body)


def test_tokenize_lowercases_splits_and_drops_stop_words():
    assert tokenize("How do I RETURN a drill-driver?") == ["return", "drill", "driver"]


@pytest.mark.parametrize(
    ("word", "token"),
    [
        ("returns", "return"),
        ("batteries", "battery"),
        ("boxes", "box"),
        ("watches", "watch"),
        ("glass", "glass"),
        ("status", "status"),
        ("gas", "gas"),
        ("30", "30"),
    ],
)
def test_tokenize_folds_simple_plurals(word, token):
    assert tokenize(word) == [token]


def _bm25(tf: int, df: int, n: int, dl: int, avgdl: float, k1: float = 1.5, b: float = 0.75):
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))


# Titles are part of the indexed text, so these use a stop word as the title
# to keep the token counts obvious: "the" is dropped.
CORPUS = [
    doc("d-red", "red apple", title="the"),
    doc("d-green", "green apple pie", title="the"),
    doc("d-sky", "blue sky", title="the"),
]


def test_scores_follow_the_bm25_formula():
    hits = BM25Index(CORPUS).search("apple", k=3)
    avgdl = (2 + 3 + 2) / 3
    assert [h.doc_id for h in hits] == ["d-red", "d-green"]
    assert hits[0].score == pytest.approx(_bm25(tf=1, df=2, n=3, dl=2, avgdl=avgdl))
    assert hits[1].score == pytest.approx(_bm25(tf=1, df=2, n=3, dl=3, avgdl=avgdl))


def test_a_query_sums_its_terms():
    hits = BM25Index(CORPUS).search("green apple", k=1)
    avgdl = 7 / 3
    expected = _bm25(1, 1, 3, 3, avgdl) + _bm25(1, 2, 3, 3, avgdl)
    assert hits == [
        Hit(doc_id="d-green", score=pytest.approx(expected), text="the\n\ngreen apple pie")
    ]


def test_the_default_parameters_are_k1_1_5_and_b_0_75():
    index = BM25Index(CORPUS)
    assert (index.k1, index.b) == (1.5, 0.75)


def test_more_occurrences_score_higher_but_saturate():
    corpus = [doc("once", "saw x y z"), doc("twice", "saw saw y z"), doc("many", "saw saw saw saw")]
    scores = {h.doc_id: h.score for h in BM25Index(corpus).search("saw", k=3)}
    assert scores["many"] > scores["twice"] > scores["once"]
    assert scores["many"] - scores["twice"] < scores["twice"] - scores["once"]


def test_a_shorter_document_wins_at_equal_term_frequency():
    corpus = [doc("short", "chisel"), doc("long", "chisel wood metal stone glass"), doc("x", "y")]
    assert [h.doc_id for h in BM25Index(corpus).search("chisel")] == ["short", "long"]


def test_b_zero_turns_off_length_normalisation():
    corpus = [doc("short", "chisel"), doc("long", "chisel wood metal stone glass"), doc("x", "y")]
    hits = BM25Index(corpus, b=0).search("chisel")
    assert hits[0].score == pytest.approx(hits[1].score)


def test_a_rare_term_outweighs_a_common_one():
    corpus = [doc("a", "tool rare"), doc("b", "tool common"), doc("c", "tool common")]
    hits = BM25Index(corpus).search("rare common", k=3)
    assert hits[0].doc_id == "a"


def test_a_term_in_every_document_still_scores_above_zero():
    corpus = [doc("a", "tool one"), doc("b", "tool two")]
    hits = BM25Index(corpus).search("tool", k=2)
    assert len(hits) == 2
    assert all(h.score > 0 for h in hits)


def test_the_title_is_searchable():
    corpus = [doc("a", "body only", title="Warranty"), doc("b", "other body", title="Shipping")]
    assert [h.doc_id for h in BM25Index(corpus).search("warranty")] == ["a"]


def test_documents_without_a_query_term_are_not_returned():
    assert BM25Index(CORPUS).search("blue", k=3) == [
        Hit(doc_id="d-sky", score=pytest.approx(_bm25(1, 1, 3, 2, 7 / 3)), text="the\n\nblue sky")
    ]


@pytest.mark.parametrize("query", ["", "   ", "the and of", "banana"])
def test_a_query_without_known_terms_finds_nothing(query):
    assert BM25Index(CORPUS).search(query) == []


def test_k_limits_the_hits():
    assert len(BM25Index(CORPUS).search("apple", k=1)) == 1


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k"):
        BM25Index(CORPUS).search("apple", k=0)


def test_ties_are_broken_by_document_id():
    corpus = [doc("b", "drill"), doc("a", "drill"), doc("c", "saw")]
    assert [h.doc_id for h in BM25Index(corpus).search("drill")] == ["a", "b"]


def test_a_repeated_query_term_counts_once():
    index = BM25Index(CORPUS)
    assert index.search("apple apple apple") == index.search("apple")


def test_an_empty_corpus_is_refused():
    with pytest.raises(ValueError, match="no documents"):
        BM25Index([])


def test_duplicate_ids_are_refused():
    with pytest.raises(ValueError, match="d-red"):
        BM25Index([CORPUS[0], CORPUS[0]])


def test_documents_are_looked_up_by_id():
    index = BM25Index(CORPUS)
    assert index.get("d-sky") is CORPUS[2]
    assert index.ids == ("d-red", "d-green", "d-sky")
