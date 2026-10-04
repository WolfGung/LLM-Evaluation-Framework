"""The authored datasets: they parse, their labels are consistent, sizes are as documented.

These are checks on test design, not on model output. No model is called.
"""

import re
from collections import Counter
from pathlib import Path

import pytest

from app.retrieval import load_kb
from llmeval.config import check_stability_cases, load_models_config
from llmeval.datasets import PRIORITY_RULES, RAG_PATH, TRIAGE_PATH, load_rag, load_triage

ROOT = Path(__file__).resolve().parents[2]
GUIDELINE = ROOT / "datasets" / "triage-guideline.md"

# Documented sizes: about 40 triage cases. RAG: 25 answerable, 7 multi-document
# and 8 unanswerable cases (5 of them deliberately messy), plus 10-12 safety
# cases, so about 50 in all.
TRIAGE_RANGE = range(35, 46)
RAG_RANGES = {
    "answerable": range(18, 27),
    "multi_doc": range(5, 9),
    "unanswerable": range(5, 9),
    "safety": range(10, 13),
}
RAG_TOTAL = range(30, 56)


@pytest.fixture(scope="module")
def triage_cases():
    return load_triage(ROOT / TRIAGE_PATH)


@pytest.fixture(scope="module")
def rag_cases():
    return load_rag(ROOT / RAG_PATH)


def test_rag_sizes_are_in_the_documented_ranges(rag_cases):
    counts = Counter(c.category for c in rag_cases)
    assert len(rag_cases) in RAG_TOTAL
    for category, allowed in RAG_RANGES.items():
        assert counts.get(category, 0) in allowed, category


def test_rag_ids_are_numbered_in_order(rag_cases):
    assert [c.id for c in rag_cases] == [f"rag-{n:03d}" for n in range(1, len(rag_cases) + 1)]


def test_rag_questions_are_unique(rag_cases):
    questions = [c.question.casefold() for c in rag_cases]
    assert len(questions) == len(set(questions))


def test_expected_documents_exist_in_the_knowledge_base(rag_cases):
    kb = {doc.doc_id: doc for doc in load_kb()}
    for case in rag_cases:
        for doc_id in case.expected_docs:
            assert doc_id in kb, f"{case.id}: {doc_id}"
            assert kb[doc_id].visibility == "public", f"{case.id}: {doc_id} is internal"


def test_triage_size_is_in_the_documented_range(triage_cases):
    assert len(triage_cases) in TRIAGE_RANGE


def test_triage_ids_are_numbered_in_order(triage_cases):
    assert [c.id for c in triage_cases] == [f"tri-{n:03d}" for n in range(1, len(triage_cases) + 1)]


def test_every_category_and_priority_is_covered(triage_cases):
    categories = Counter(str(c.category) for c in triage_cases)
    priorities = Counter(str(c.priority) for c in triage_cases)
    assert set(categories) == {
        "shipping",
        "returns",
        "payment",
        "warranty",
        "order_status",
        "product_question",
        "other",
    }
    assert set(priorities) == {"urgent", "high", "normal", "low"}
    assert min(categories.values()) >= 3
    assert min(priorities.values()) >= 4


def test_order_ids_are_present_and_absent(triage_cases):
    with_id = sum(c.order_id is not None for c in triage_cases)
    assert 0.3 * len(triage_cases) <= with_id <= 0.7 * len(triage_cases)


def test_the_guideline_defines_every_rule_under_its_priority():
    text = GUIDELINE.read_text(encoding="utf-8")
    sections = re.split(r"^### ", text, flags=re.MULTILINE)
    defined = {}
    for section in sections[1:]:
        heading = section.splitlines()[0].strip().lower()
        for rule in re.findall(r"^- \*\*([A-Z]\d)\*\*", section, flags=re.MULTILINE):
            defined[rule] = heading
    assert defined == {rule: str(priority) for rule, priority in PRIORITY_RULES.items()}


def test_the_guideline_says_v2_encodes_the_rules():
    text = GUIDELINE.read_text(encoding="utf-8")
    assert "partly by construction" in " ".join(text.split())


def test_a_written_order_id_is_extracted_in_normal_form(triage_cases):
    # Labels follow the guideline: a six-digit id in the text, written any
    # common way, is the expected order id in TS-###### form.
    pattern = re.compile(r"\bts[\s-]?(\d{6})\b", re.IGNORECASE)
    for case in triage_cases:
        found = [f"TS-{digits}" for digits in pattern.findall(case.text)]
        if not found:
            assert case.order_id is None, case.id
        else:
            assert case.order_id in found, case.id


def test_the_guideline_counts_its_divergences_from_v2_exactly():
    text = GUIDELINE.read_text(encoding="utf-8")
    section = text.split("## How this guideline relates to the prompts", 1)[1]
    stated = re.search(r"differs from the v2 prompt in (\d+) places", section)
    assert stated, "the section states the number of divergences"
    listed = re.findall(r"^\d+\. ", section, flags=re.MULTILINE)
    assert len(listed) == int(stated.group(1))
    assert "almost word for word" not in text
    for case_id in ("tri-012", "tri-030"):
        assert case_id in section


BORDERLINE = ("tri-010", "tri-012", "tri-014", "tri-019", "tri-030", "tri-033")


def test_borderline_cases_explain_their_labels(triage_cases, rag_cases):
    notes = {c.id: c.note for c in (*triage_cases, *rag_cases)}
    for case_id in (*BORDERLINE, "rag-034", "rag-035"):
        assert notes[case_id] and len(notes[case_id]) > 40, case_id
    for case_id in ("rag-034", "rag-035"):
        assert "correct answer" in notes[case_id].lower(), case_id


def test_tri_010_and_tri_033_notes_follow_the_ticket(triage_cases):
    notes = {c.id: c.note.lower() for c in triage_cases if c.note}
    # The ticket says the sander was sent two weeks ago, not that two weeks
    # passed since it arrived.
    assert "sent" in notes["tri-010"] and "two weeks have passed" not in notes["tri-010"]
    # The ticket mentions the customer's account, so the note weighs `other`.
    assert "other" in notes["tri-033"] and "account" in notes["tri-033"]


def test_the_configured_stability_subset_names_cases_of_the_datasets(rag_cases, triage_cases):
    models = load_models_config(ROOT / "config" / "models.yaml")
    check_stability_cases(models, {case.id for case in (*rag_cases, *triage_cases)})
