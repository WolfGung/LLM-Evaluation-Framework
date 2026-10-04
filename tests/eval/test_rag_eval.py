"""The support assistant, case by case, on the recorded run.

A case passes when every check passes on every repeat: retrieval (the
expected documents were found), deterministic rules (citations, forbidden
phrases, length, declining when the documents do not answer) and reference
facts.
"""

import pytest

from llmeval.datasets import RAG_PATH, load_rag
from llmeval.runner import versions_of
from tests.eval.conftest import ROOT, assert_case_passes

CASES = load_rag(ROOT / RAG_PATH)


@pytest.mark.parametrize("version", versions_of("rag"))
@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_rag_case(replay, case, version):
    assert_case_passes(replay.rag(case, version))
