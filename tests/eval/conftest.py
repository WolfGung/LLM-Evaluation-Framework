"""Replay evaluation: every case of every dataset, per prompt version.

These tests read the recorded run in `cassettes/`. Without
`cassettes/manifest.json` they skip with "pending first recorded run". With
it, a call missing from the cassettes fails its test with the replay-miss
message ("no recording for <case>/<version>/<repeat>: run make record").

No model is called: the client is in replay mode and needs no key.

LLMEVAL_CASSETTES_DIR points the suite at another cassette directory; the
framework's own tests use it to run this suite on synthetic recordings.
"""

from __future__ import annotations

import os
from pathlib import Path

import allure
import pytest

from llmeval.cassettes import PENDING_RECORDED_RUN, CassetteStore, load_manifest
from llmeval.client import ModelClient
from llmeval.config import Mode, load_config
from llmeval.runner import CaseRecord, run_rag, run_triage

ROOT = Path(__file__).resolve().parents[2]
CASSETTES = Path(os.environ.get("LLMEVAL_CASSETTES_DIR") or ROOT / "cassettes")
CONFIG = ROOT / "config" / "models.yaml"


class Replay:
    """Runs one case at a time through the recorded calls."""

    def __init__(self, repeats: int) -> None:
        self.config = load_config(CONFIG, env={})
        self.client = ModelClient(Mode.REPLAY, CassetteStore(CASSETTES), self.config)
        self.repeats = repeats

    def rag(self, case, version: str) -> CaseRecord:
        role = self.config.models.system
        return run_rag(self.client, role, [case], version, repeats=self.repeats)[0]

    def triage(self, case, version: str) -> CaseRecord:
        role = self.config.models.system
        return run_triage(self.client, role, [case], version, repeats=self.repeats)[0]


@pytest.fixture(scope="session")
def replay() -> Replay:
    manifest = load_manifest(CASSETTES)
    if manifest is None:
        pytest.skip(PENDING_RECORDED_RUN)
    return Replay(manifest.repeats)


def explain(record: CaseRecord) -> str:
    """The failed checks of a case, by repeat and layer, for the failure message."""
    lines = [f"{record.id}: {record.input}"]
    for run in record.runs:
        failed = [check for check in run.checks if not check.passed]
        for check in failed:
            lines.append(f"  repeat {run.repeat} [{check.layer}] {check.name}: {check.detail}")
        if any(check.layer == "retrieval" for check in failed):
            lines.append(
                f"  repeat {run.repeat}: the search did not return an expected document, "
                "so later failures may be retrieval misses, not generation failures"
            )
    return "\n".join(lines)


def assert_case_passes(record: CaseRecord) -> None:
    allure.attach(
        record.model_dump_json(indent=2),
        name=f"{record.id} record",
        attachment_type=allure.attachment_type.JSON,
    )
    assert all(run.passed for run in record.runs), explain(record)
