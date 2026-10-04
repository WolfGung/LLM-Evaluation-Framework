"""The replay evaluation suite (`tests/eval`) reports each outcome for the right reason.

Outcomes: pending first recorded run, a loud replay miss, pending baseline,
pass, xfail (a known failure), strict XPASS (a known failure that now
passes), and a regression.

Synthetic data: the manifest, the baselines and the recorded replies are made
up for the test. Replies are recorded through the real client with a mock
transport into `tmp_path`; the suite is pointed there with
LLMEVAL_CASSETTES_DIR and LLMEVAL_BASELINE. Nothing is written to the
repository's `cassettes/` or `results/`.
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
from llmeval.baseline import Baseline, CaseBaseline, FunctionBaseline, Metrics, Provenance
from llmeval.cassettes import MANIFEST_FILE, CassetteStore
from llmeval.client import ModelClient
from llmeval.config import Config, Mode, Settings, load_models_config
from llmeval.datasets import load_triage

ROOT = Path(__file__).resolve().parents[2]
MODELS = load_models_config(ROOT / "config" / "models.yaml")
CASE = load_triage(ROOT / "datasets" / "triage.jsonl")[0]
RIGHT = {
    "category": str(CASE.category),
    "priority": str(CASE.priority),
    "order_id": CASE.order_id,
    "summary": "Synthetic summary.",
}
WRONG_PRIORITY = {**RIGHT, "priority": "low"}
WRONG_TWICE = {**RIGHT, "priority": "low", "order_id": None}
KNOWN_PRIORITY_FAILURE = CaseBaseline(
    passed=False, failed_checks=(("reference", "priority_match"),)
)


def run_eval_suite(tmp_path: Path, *selection: str) -> tuple[int, str]:
    """Run the triage eval suite in a subprocess; return the exit code and the output."""
    env = {
        **os.environ,
        "LLMEVAL_CASSETTES_DIR": str(tmp_path / "cassettes"),
        "LLMEVAL_BASELINE": str(tmp_path / "baseline.json"),
    }
    for name in ("OPENROUTER_API_KEY", "LLMEVAL_MODE"):
        env.pop(name, None)
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-W",
            "error",
            "-p",
            "no:cacheprovider",
            "-q",
            "-rsxX",
            "tests/eval/test_triage_eval.py",
            *selection,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode, done.stdout + done.stderr


def write_manifest(tmp_path: Path, system_model: str = MODELS.system.model) -> None:
    manifest = {
        "models": {"system": system_model, "judge": MODELS.judge.model},
        "prompt_versions": {"triage": ["v1"]},
        "repeats": 1,
        "datasets": {"triage.jsonl": "0" * 64},
        "recorded_from": "2026-01-01T10:00:00Z",
        "recorded_to": "2026-01-01T10:01:00Z",
        "planned_calls": 1,
        "recorded_calls": 1,
    }
    (tmp_path / "cassettes" / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")


class SyntheticOpenRouter:
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


def record_reply(tmp_path: Path, reply: dict) -> None:
    config = Config(models=MODELS, settings=Settings(api_key=SecretStr("synthetic-key-123")))
    with ModelClient(
        Mode.RECORD,
        CassetteStore(tmp_path / "cassettes"),
        config,
        httpx.MockTransport(SyntheticOpenRouter(reply)),
        limiter=NoWait(),
    ) as client:
        triage(client, MODELS.system, CASE.text, "v1", case=CASE.id)


def write_baseline(tmp_path: Path, case: CaseBaseline) -> None:
    baseline = Baseline(
        provenance=Provenance(
            recorded_from="2026-01-01T10:00:00Z",
            recorded_to="2026-01-01T10:01:00Z",
            models={"system": MODELS.system.model, "judge": MODELS.judge.model},
            repeats=1,
            prompt_versions={"triage": ("v1",)},
        ),
        functions={
            "triage": {
                "v1": FunctionBaseline(
                    metrics=Metrics(all_checks=None, layers={}), cases={CASE.id: case}
                )
            }
        },
    )
    (tmp_path / "baseline.json").write_text(baseline.model_dump_json(indent=2), encoding="utf-8")


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "cassettes").mkdir()
    (tmp_path / "cassettes" / ".gitkeep").write_text("", encoding="utf-8")
    return tmp_path


def test_without_a_manifest_every_case_skips_as_pending(ws):
    code, out = run_eval_suite(ws)
    assert code == 0, out
    assert "pending first recorded run" in out
    assert "40 skipped" in out


def test_with_a_manifest_a_missing_recording_fails_loudly(ws):
    write_manifest(ws)
    code, out = run_eval_suite(ws, "-k", CASE.id)
    assert code == 1, out
    assert f"no recording for {CASE.id}/v1/0: run make record" in out
    assert "1 failed" in out


def test_versions_come_from_the_manifest(ws):
    # Only v1 was recorded, so there is no v2 test although a v2 prompt exists.
    write_manifest(ws)
    code, out = run_eval_suite(ws, "-k", CASE.id, "--collect-only")
    assert code == 0, out
    assert f"test_triage_case[{CASE.id}-v1]" in out
    assert f"{CASE.id}-v2" not in out


def test_recordings_without_a_baseline_skip_as_pending_baseline(ws):
    write_manifest(ws)
    record_reply(ws, RIGHT)
    code, out = run_eval_suite(ws, "-k", CASE.id)
    assert code == 0, out
    assert "pending baseline" in out
    assert "1 skipped" in out


def test_a_case_that_passed_and_passes(ws):
    write_manifest(ws)
    record_reply(ws, RIGHT)
    write_baseline(ws, CaseBaseline(passed=True))
    code, out = run_eval_suite(ws, "-k", CASE.id)
    assert code == 0, out
    assert "1 passed" in out


def test_a_known_failure_is_an_xfail(ws):
    write_manifest(ws)
    record_reply(ws, WRONG_PRIORITY)
    write_baseline(ws, KNOWN_PRIORITY_FAILURE)
    code, out = run_eval_suite(ws, "-k", CASE.id)
    assert code == 0, out
    assert "1 xfailed" in out
    assert "reference/priority_match" in out


def test_a_known_failure_that_now_passes_is_a_strict_xpass(ws):
    write_manifest(ws)
    record_reply(ws, RIGHT)
    write_baseline(ws, KNOWN_PRIORITY_FAILURE)
    code, out = run_eval_suite(ws, "-k", CASE.id)
    assert code == 1, out
    assert "XPASS(strict)" in out
    assert "now passes; update the baseline deliberately (make baseline)" in out


def test_a_case_that_passed_and_now_fails_is_a_regression(ws):
    write_manifest(ws)
    record_reply(ws, WRONG_PRIORITY)
    write_baseline(ws, CaseBaseline(passed=True))
    code, out = run_eval_suite(ws, "-k", CASE.id)
    assert code == 1, out
    assert "1 failed" in out
    assert "[reference] priority_match: expected 'high', got 'low'" in out


def test_a_known_failure_with_a_new_failing_check_is_a_regression(ws):
    write_manifest(ws)
    record_reply(ws, WRONG_TWICE)
    write_baseline(ws, KNOWN_PRIORITY_FAILURE)
    code, out = run_eval_suite(ws, "-k", CASE.id)
    assert code == 1, out
    assert "reference/order_id_match" in out
    assert "baseline did not" in out


def test_changed_datasets_are_noticed_once(ws):
    write_manifest(ws)
    record_reply(ws, RIGHT)
    write_baseline(ws, CaseBaseline(passed=True))
    code, out = run_eval_suite(ws, "-k", CASE.id)
    assert code == 0, out
    assert out.count("notice: datasets changed since the recording: triage.jsonl") == 1


def test_a_manifest_recorded_with_other_models_is_an_error(ws):
    write_manifest(ws, system_model="synthetic/old-model:free")
    code, out = run_eval_suite(ws, "-k", CASE.id)
    assert code == 1, out
    assert "recorded with synthetic/old-model:free" in out
    assert "re-record or restore the config" in out


def test_a_broken_manifest_fails_with_one_clear_line(ws):
    (ws / "cassettes" / MANIFEST_FILE).write_text("{not json", encoding="utf-8")
    code, out = run_eval_suite(ws)
    assert code != 0, out
    assert "INTERNALERROR" not in out
    assert "manifest error: manifest.json: not a valid run manifest" in out


def test_a_function_missing_from_the_recorded_run_skips_with_a_reason(ws):
    write_manifest(ws)
    manifest_path = ws / "cassettes" / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prompt_versions"] = {"rag": ["v1"]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    code, out = run_eval_suite(ws)
    assert code == 0, out
    assert "40 skipped" in out
    assert "triage is not in the recorded run" in out
    assert "empty parameter set" not in out
