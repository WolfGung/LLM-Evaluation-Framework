"""Pairwise comparison: the judge picks the better of two prompt versions' answers.

One test per judged RAG case (safety cases are left to the rules) and
compared version pair: the first version with each later one, on repeat 0,
replayed from the recording. The judge sees each answer once as A and once
as B, and the Allure report attaches both orders, so a preference that
flips with the position shows ("inconsistent").

A preference is a measurement, not a pass or a fail: the test fails only
when the replay misses a call. The gate checks position consistency and
valid verdicts against the baseline.
"""

from llmeval.datasets import RAG_PATH, load_rag
from llmeval.runner import judged
from tests.eval.report import show_comparison, show_pairwise
from tests.eval.support import ROOT

FUNCTION = "rag"
CASES = [case for case in load_rag(ROOT / RAG_PATH) if judged(case)]


def test_pairwise_case(replay, case, pair):
    show_pairwise(case, pair)
    show_comparison(replay.pairwise(case, pair))
