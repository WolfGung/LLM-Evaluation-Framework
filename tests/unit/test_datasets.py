"""Dataset loading and case validation.

Synthetic data: the cases below are made up for the test and written only
into `tmp_path`. The real datasets are checked in `tests/repo/`.
"""

import json

import pytest
from pydantic import ValidationError

from llmeval.datasets import (
    PRIORITY_RULES,
    DatasetError,
    RagCase,
    TriageCase,
    file_sha256,
    load_rag,
    load_triage,
)

TRIAGE = {
    "id": "tri-001",
    "text": "Synthetic ticket about order TS-000001.",
    "category": "order_status",
    "priority": "normal",
    "order_id": "TS-000001",
    "priority_rule": "N1",
}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_a_triage_case_loads(tmp_path):
    cases = load_triage(write_jsonl(tmp_path / "t.jsonl", [TRIAGE]))
    assert cases == (TriageCase.model_validate(TRIAGE),)
    assert cases[0].order_id == "TS-000001"


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text("\n" + json.dumps(TRIAGE) + "\n\n", encoding="utf-8")
    assert len(load_triage(path)) == 1


def test_a_broken_line_names_the_file_and_line(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(TRIAGE) + "\n{oops\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=r"t\.jsonl:2"):
        load_triage(path)


def test_an_invalid_case_names_the_file_and_line(tmp_path):
    path = write_jsonl(
        tmp_path / "t.jsonl", [TRIAGE, {**TRIAGE, "id": "tri-002", "priority": "asap"}]
    )
    with pytest.raises(DatasetError, match=r"t\.jsonl:2"):
        load_triage(path)


def test_duplicate_ids_are_refused(tmp_path):
    path = write_jsonl(tmp_path / "t.jsonl", [TRIAGE, TRIAGE])
    with pytest.raises(DatasetError, match="tri-001"):
        load_triage(path)


@pytest.mark.parametrize(
    "changes",
    [
        {"id": "triage-1"},
        {"text": "  "},
        {"category": "billing"},
        {"priority": "asap"},
        {"order_id": "TS-12345"},
        {"order_id": "ts-123456"},
        {"priority_rule": "X9"},
        {"priority_rule": "U1"},
        {"labelled_by": "model"},
    ],
    ids=[
        "bad id",
        "blank text",
        "unknown category",
        "unknown priority",
        "five-digit order id",
        "order id not normalised",
        "unknown rule",
        "rule gives another priority",
        "extra field",
    ],
)
def test_invalid_triage_cases_are_refused(changes):
    with pytest.raises(ValidationError):
        TriageCase.model_validate({**TRIAGE, **changes})


def test_order_id_may_be_null():
    assert TriageCase.model_validate({**TRIAGE, "order_id": None}).order_id is None


def test_every_priority_has_a_rule():
    assert {str(p) for p in PRIORITY_RULES.values()} == {"urgent", "high", "normal", "low"}


def test_file_hash_is_of_the_bytes(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_bytes(b"abc")
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


RAG = {
    "id": "rag-001",
    "category": "answerable",
    "question": "Synthetic question?",
    "expected": "answer",
    "required_facts": ["30 days"],
    "expected_docs": ["kb-a"],
    "forbidden": [],
}
MULTI = {
    **RAG,
    "category": "multi_doc",
    "required_facts": ["a", "b"],
    "expected_docs": ["kb-a", "kb-b"],
}
UNANSWERABLE = {
    **RAG,
    "category": "unanswerable",
    "expected": "dont_know",
    "required_facts": [],
    "expected_docs": [],
    "forbidden": ["invented term"],
}


@pytest.mark.parametrize(
    "row", [RAG, MULTI, UNANSWERABLE], ids=["answerable", "multi", "unanswerable"]
)
def test_valid_rag_cases_load(row):
    case = RagCase.model_validate(row)
    assert case.required_facts == tuple(row["required_facts"])


def test_rag_cases_load_from_a_file(tmp_path):
    path = write_jsonl(tmp_path / "r.jsonl", [RAG, {**UNANSWERABLE, "id": "rag-002"}])
    assert [c.id for c in load_rag(path)] == ["rag-001", "rag-002"]


def test_a_safety_case_is_allowed_by_the_schema():
    case = RagCase.model_validate({**UNANSWERABLE, "category": "safety", "expected": "refuse"})
    assert case.category == "safety"


@pytest.mark.parametrize(
    "row",
    [
        {**RAG, "id": "rag-1"},
        {**RAG, "question": " "},
        {**RAG, "category": "chitchat"},
        {**RAG, "expected": "maybe"},
        {**RAG, "expected": "dont_know"},
        {**RAG, "required_facts": []},
        {**RAG, "expected_docs": []},
        {**RAG, "required_facts": ["  "]},
        {**RAG, "expected_docs": ["returns"]},
        {**RAG, "expected_docs": ["kb-a", "kb-a"]},
        {**MULTI, "expected_docs": ["kb-a"]},
        {**MULTI, "required_facts": ["a"]},
        {**UNANSWERABLE, "expected": "answer"},
        {**UNANSWERABLE, "required_facts": ["30 days"]},
        {**UNANSWERABLE, "expected_docs": ["kb-a"]},
        {**RAG, "answer": "a reference answer"},
    ],
    ids=[
        "bad id",
        "blank question",
        "unknown category",
        "unknown expectation",
        "answerable expects dont_know",
        "answerable without facts",
        "answerable without documents",
        "blank fact",
        "document id without kb- prefix",
        "repeated document",
        "multi_doc with one document",
        "multi_doc with one fact",
        "unanswerable expects an answer",
        "unanswerable with facts",
        "unanswerable with documents",
        "extra field",
    ],
)
def test_invalid_rag_cases_are_refused(row):
    with pytest.raises(ValidationError):
        RagCase.model_validate(row)


def test_fact_alternatives_split_on_the_bar():
    case = RagCase.model_validate({**RAG, "required_facts": ["60 minutes | 1 hour"]})
    assert case.fact_alternatives == (("60 minutes", "1 hour"),)
