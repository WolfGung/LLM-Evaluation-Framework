"""Ticket triage, case by case, on the recorded run, compared with the baseline.

A case passes when every check passes on every repeat: the reply is valid
JSON matching the schema, and category, priority and order id equal the
labels from `datasets/triage-guideline.md`. Whether a failing case fails the
test depends on the baseline (see `llmeval.baseline`).
"""

from llmeval.baseline import compare
from llmeval.datasets import TRIAGE_PATH, load_triage
from tests.eval.support import ROOT, apply_verdict

FUNCTION = "triage"
CASES = load_triage(ROOT / TRIAGE_PATH)


def test_triage_case(request, replay, baseline, case, version):
    record = replay.case(FUNCTION, case, version)
    expected = baseline.case(FUNCTION, version, case.id) if baseline else None
    apply_verdict(request, record, compare(record, expected))
