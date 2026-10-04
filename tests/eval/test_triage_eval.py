"""Ticket triage, case by case, on the recorded run.

A case passes when every check passes on every repeat: the reply is valid
JSON matching the schema, and category, priority and order id equal the
labels from `datasets/triage-guideline.md`.
"""

import pytest

from llmeval.datasets import TRIAGE_PATH, load_triage
from llmeval.runner import versions_of
from tests.eval.conftest import ROOT, assert_case_passes

CASES = load_triage(ROOT / TRIAGE_PATH)


@pytest.mark.parametrize("version", versions_of("triage"))
@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_triage_case(replay, case, version):
    assert_case_passes(replay.triage(case, version))
