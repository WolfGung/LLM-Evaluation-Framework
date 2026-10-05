"""Helpers for the replay evaluation suite: where the recording is, and how to replay a case.

LLMEVAL_CASSETTES_DIR and LLMEVAL_BASELINE point the suite at another
recording and baseline; the framework's own tests use them to run this suite
on synthetic recordings.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llmeval.baseline import BASELINE_PATH, Verdict
from llmeval.cassettes import CassetteStore, RunManifest
from llmeval.checks.judge import RUBRIC_PATH, Judge, load_rubric
from llmeval.client import ModelClient
from llmeval.config import Mode, load_config
from llmeval.results import CaseRecord, PairwiseCaseRecord
from llmeval.runner import compare_case, run_rag, run_triage
from tests.eval.report import REGRESSION, STRICT_XPASS

ROOT = Path(__file__).resolve().parents[2]
CASSETTES = Path(os.environ.get("LLMEVAL_CASSETTES_DIR") or ROOT / "cassettes")
BASELINE = Path(os.environ.get("LLMEVAL_BASELINE") or ROOT / BASELINE_PATH)
CONFIG = ROOT / "config" / "models.yaml"
DATASETS = ROOT / "datasets"
# The version id used when there is no recorded run (every test skips then).
UNRECORDED = "unrecorded"


class Replay:
    """Replays one case at a time from the recorded run.

    A replay is deterministic, so each case and version is replayed once per
    session: the per-case tests and the layer tests share the records.
    """

    def __init__(self, manifest: RunManifest) -> None:
        self.config = load_config(CONFIG, env={})
        manifest.check_models(
            system=self.config.models.system.model, judge=self.config.models.judge.model
        )
        self.manifest = manifest
        self.client = ModelClient(Mode.REPLAY, CassetteStore(CASSETTES), self.config)
        self.judge = Judge(self.client, self.config.models.judge, load_rubric(ROOT / RUBRIC_PATH))
        self._records: dict[tuple[str, str, str], CaseRecord] = {}

    def case(self, function: str, case, version: str) -> CaseRecord:
        """Replay one case; RAG answers are graded by the recorded judge too."""
        key = (function, case.id, version)
        if key not in self._records:
            self._records[key] = self._replay(function, case, version)
        return self._records[key]

    def pairwise(self, case, versions: tuple[str, str]) -> PairwiseCaseRecord:
        """Replay the judge's comparison of two versions' answers (repeat 0) to one RAG case."""
        answers = tuple(self.case("rag", case, version).runs[0].output for version in versions)
        return compare_case(self.judge, case, answers, versions)

    def _replay(self, function: str, case, version: str) -> CaseRecord:
        options = {
            "repeats": self.manifest.repeats,
            "stability_cases": self.manifest.stability_cases,
        }
        if function == "rag":
            options["judge"] = self.judge
            options["judge_repeats"] = self.manifest.judge_repeats
            return run_rag(self.client, self.config.models.system, [case], version, **options)[0]
        return run_triage(self.client, self.config.models.system, [case], version, **options)[0]


def apply_verdict(verdict: Verdict) -> None:
    """Turn a baseline comparison into the pytest outcome.

    A regression fails with a message that starts with "regression:". A
    known failure that now passes fails as a strict XPASS, in pytest's own
    words: pytest's strict xfail marker would report it without a message,
    and the Allure report could not tell it from a regression then.
    """
    if verdict.outcome == "pending":
        pytest.skip(verdict.message)
    if verdict.outcome == "xfail":
        pytest.xfail(verdict.message)
    if verdict.outcome == "xpass":
        pytest.fail(f"{STRICT_XPASS} {verdict.message}", pytrace=False)
    if verdict.outcome == "fail":
        pytest.fail(f"{REGRESSION}: {verdict.message}", pytrace=False)
