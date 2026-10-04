"""The replay evaluation suite (`tests/eval`) skips, fails and passes for the right reasons.

Synthetic data: the manifest and the recorded replies are made up for the
test. They are recorded through the real client with a mock transport into
`tmp_path`, and the eval suite is pointed there with LLMEVAL_CASSETTES_DIR.
Nothing is written to the repository's `cassettes/` or `results/`.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from app.triage import triage
from llmeval.cassettes import MANIFEST_FILE, CassetteStore
from llmeval.client import ModelClient
from llmeval.config import Config, Mode, Settings, load_models_config
from llmeval.datasets import load_triage

ROOT = Path(__file__).resolve().parents[2]


def run_eval_suite(cassettes: Path, *selection: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "LLMEVAL_CASSETTES_DIR": str(cassettes)}
    for name in ("OPENROUTER_API_KEY", "LLMEVAL_MODE"):
        env.pop(name, None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-W",
            "error",
            "-p",
            "no:cacheprovider",
            "-q",
            "-rs",
            "tests/eval/test_triage_eval.py",
            *selection,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def write_manifest(cassettes: Path) -> None:
    models = load_models_config(ROOT / "config" / "models.yaml")
    manifest = {
        "models": {"system": models.system.model, "judge": models.judge.model},
        "prompt_versions": {"triage": ["v1", "v2"]},
        "repeats": 1,
        "datasets": {"triage.jsonl": "0" * 64},
        "recorded_from": "2026-01-01T10:00:00Z",
        "recorded_to": "2026-01-01T10:01:00Z",
        "planned_calls": 2,
        "recorded_calls": 2,
    }
    (cassettes / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def cassettes(tmp_path):
    path = tmp_path / "cassettes"
    path.mkdir()
    (path / ".gitkeep").write_text("", encoding="utf-8")
    return path


def test_without_a_manifest_every_case_skips_as_pending(cassettes):
    done = run_eval_suite(cassettes)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "pending first recorded run" in done.stdout
    assert "80 skipped" in done.stdout


def test_with_a_manifest_a_missing_recording_fails_loudly(cassettes):
    write_manifest(cassettes)
    done = run_eval_suite(cassettes, "-k", "tri-001 and v1")
    assert done.returncode == 1
    assert "no recording for tri-001/v1/0: run make record" in done.stdout
    assert "1 failed" in done.stdout


class SyntheticOpenRouter:
    """Answers every triage call with the labels of tri-001, as JSON."""

    def __init__(self, reply: dict) -> None:
        self.reply = reply

    def __call__(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-synthetic",
                "model": json.loads(request.content)["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(self.reply)},
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0},
            },
        )


class NoWait:
    def acquire(self) -> float:
        return 0.0


def test_a_recorded_case_passes_or_fails_on_its_checks(cassettes):
    models = load_models_config(ROOT / "config" / "models.yaml")
    config = Config(models=models, settings=Settings(api_key=SecretStr("synthetic-key-123")))
    case = load_triage(ROOT / "datasets" / "triage.jsonl")[0]
    right = {
        "category": str(case.category),
        "priority": str(case.priority),
        "order_id": case.order_id,
        "summary": "Synthetic summary.",
    }
    wrong = {**right, "priority": "low"}
    for version, reply in (("v1", right), ("v2", wrong)):
        with ModelClient(
            Mode.RECORD,
            CassetteStore(cassettes),
            config,
            httpx.MockTransport(SyntheticOpenRouter(reply)),
            limiter=NoWait(),
        ) as client:
            triage(client, models.system, case.text, version, case=case.id)
    write_manifest(cassettes)

    done = run_eval_suite(cassettes, "-k", case.id)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "1 passed" in done.stdout and "1 failed" in done.stdout
    assert f"test_triage_case[{case.id}-v2]" in done.stdout
    assert "priority_match" in done.stdout
