"""The support assistant, case by case, on the recorded run, compared with the baseline.

A case passes when every check passes on every repeat: retrieval (the
expected documents were found), deterministic rules (citations, forbidden
phrases, length, declining when the documents do not answer), reference
facts and, except for safety cases, the judge's rubric (groundedness,
helpfulness, tone; replayed from the recording). Whether a failing case fails
the test depends on the baseline (see `llmeval.baseline`).
"""

from llmeval.baseline import compare
from llmeval.datasets import RAG_PATH, load_rag
from tests.eval.support import ROOT, apply_verdict

FUNCTION = "rag"
CASES = load_rag(ROOT / RAG_PATH)


def test_rag_case(request, replay, baseline, case, version):
    record = replay.case(FUNCTION, case, version)
    expected = baseline.case(FUNCTION, version, case.id) if baseline else None
    apply_verdict(request, record, compare(record, expected))
