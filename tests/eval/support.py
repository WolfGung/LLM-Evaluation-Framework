"""Helpers for the replay evaluation suite: where the recording is, and how to replay a case.

LLMEVAL_CASSETTES_DIR and LLMEVAL_BASELINE point the suite at another
recording and baseline; the framework's own tests use them to run this suite
on synthetic recordings.
"""

from __future__ import annotations

import os
from pathlib import Path

import allure
import pytest

from llmeval.baseline import BASELINE_PATH, Verdict
from llmeval.cassettes import CassetteStore, RunManifest
from llmeval.client import ModelClient
from llmeval.config import Mode, load_config
from llmeval.results import CaseRecord
from llmeval.runner import run_rag, run_triage

ROOT = Path(__file__).resolve().parents[2]
CASSETTES = Path(os.environ.get("LLMEVAL_CASSETTES_DIR") or ROOT / "cassettes")
BASELINE = Path(os.environ.get("LLMEVAL_BASELINE") or ROOT / BASELINE_PATH)
CONFIG = ROOT / "config" / "models.yaml"
DATASETS = ROOT / "datasets"
# The version id used when there is no recorded run (every test skips then).
UNRECORDED = "unrecorded"


class Replay:
    """Replays one case at a time from the recorded run."""

    def __init__(self, manifest: RunManifest) -> None:
        self.config = load_config(CONFIG, env={})
        manifest.check_models(
            system=self.config.models.system.model, judge=self.config.models.judge.model
        )
        self.manifest = manifest
        self.client = ModelClient(Mode.REPLAY, CassetteStore(CASSETTES), self.config)

    def case(self, function: str, case, version: str) -> CaseRecord:
        run = {"rag": run_rag, "triage": run_triage}[function]
        return run(
            self.client,
            self.config.models.system,
            [case],
            version,
            repeats=self.manifest.repeats,
            stability_cases=self.manifest.stability_cases,
        )[0]


def apply_verdict(request: pytest.FixtureRequest, record: CaseRecord, verdict: Verdict) -> None:
    """Turn a baseline comparison into the pytest outcome."""
    allure.attach(
        record.model_dump_json(indent=2),
        name=f"{record.id} record",
        attachment_type=allure.attachment_type.JSON,
    )
    if verdict.outcome == "pending":
        pytest.skip(verdict.message)
    if verdict.outcome == "xfail":
        pytest.xfail(verdict.message)
    if verdict.outcome == "xpass":
        # The case passes now; strict xfail turns that into a failing XPASS.
        request.node.add_marker(pytest.mark.xfail(strict=True, reason=verdict.message))
        return
    if verdict.outcome == "fail":
        pytest.fail(verdict.message, pytrace=False)
