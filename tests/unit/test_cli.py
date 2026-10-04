"""The command line: `llmeval eval` and `llmeval retrieval`.

Synthetic data: the datasets, the model config, the manifest and every model
reply below are made up for the test. The end-to-end test records synthetic
replies through the real client with a mock transport, into `tmp_path` only;
nothing is written to the repository's `cassettes/` or `results/`.
"""

import json
import subprocess
import sys

import httpx
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from llmeval.cassettes import MANIFEST_FILE, CassetteStore
from llmeval.cli import app
from llmeval.client import ModelClient
from llmeval.config import Config, Mode, Settings, load_models_config
from llmeval.datasets import load_rag, load_triage
from llmeval.runner import run

runner = CliRunner()

CONFIG_YAML = """\
system:
  model: synthetic/system:free
  temperature: 0.2
  max_tokens: 200
  structured_output: false
judge:
  model: synthetic/judge:free
  temperature: 0
  seed: 7
  max_tokens: 200
  structured_output: true
repeats: 1
rpm: 18
"""

RAG_ROW = {
    "id": "rag-001",
    "category": "answerable",
    "question": "How many days do I have to return an item?",
    "expected": "answer",
    "required_facts": ["30 days"],
    "expected_docs": ["kb-returns"],
    "forbidden": [],
}
MISS_ROW = {
    "id": "rag-002",
    "category": "answerable",
    "question": "What warranty does a cordless drill have?",
    "expected": "answer",
    "required_facts": ["2 years"],
    "expected_docs": ["kb-warranty"],
    "forbidden": [],
}
TRIAGE_ROW = {
    "id": "tri-001",
    "text": "Synthetic: where is order TS-111111?",
    "category": "order_status",
    "priority": "normal",
    "order_id": "TS-111111",
    "priority_rule": "N1",
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("OPENROUTER_API_KEY", "LLMEVAL_MODE", "MAX_RUN_COST_USD"):
        monkeypatch.delenv(name, raising=False)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "config.yaml").write_text(CONFIG_YAML, encoding="utf-8")
    write_jsonl(tmp_path / "datasets" / "rag.jsonl", [RAG_ROW])
    write_jsonl(tmp_path / "datasets" / "triage.jsonl", [TRIAGE_ROW])
    (tmp_path / "cassettes").mkdir()
    (tmp_path / "cassettes" / ".gitkeep").write_text("", encoding="utf-8")
    return tmp_path


def eval_args(ws, *extra):
    return [
        "eval",
        "--config",
        str(ws / "config.yaml"),
        "--datasets-dir",
        str(ws / "datasets"),
        "--cassettes-dir",
        str(ws / "cassettes"),
        "--results-dir",
        str(ws / "results"),
        *extra,
    ]


def write_manifest(ws, versions):
    manifest = {
        "models": {"system": "synthetic/system:free", "judge": "synthetic/judge:free"},
        "prompt_versions": versions,
        "repeats": 1,
        "datasets": {"rag.jsonl": "0" * 64, "triage.jsonl": "1" * 64},
        "recorded_from": "2026-01-01T10:00:00Z",
        "recorded_to": "2026-01-01T10:05:00Z",
        "planned_calls": 2,
        "recorded_calls": 2,
    }
    (ws / "cassettes" / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")


def test_eval_without_a_manifest_is_pending(workspace):
    result = runner.invoke(app, eval_args(workspace))
    assert result.exit_code == 0, result.output
    assert "pending first recorded run" in result.output
    assert not (workspace / "results").exists()


def test_eval_with_a_manifest_and_no_recordings_fails_loudly(workspace):
    write_manifest(workspace, {"rag": ["v1"], "triage": ["v1"]})
    result = runner.invoke(app, eval_args(workspace))
    assert result.exit_code == 1
    assert "no recording for rag-001/v1/0: run make record" in result.output
    assert not (workspace / "results").exists()


def test_eval_ignores_a_live_mode_in_the_environment(workspace, monkeypatch):
    # eval only replays; recording and live calls belong to `make record`.
    monkeypatch.setenv("LLMEVAL_MODE", "live")
    result = runner.invoke(app, eval_args(workspace))
    assert result.exit_code == 0, result.output
    assert "pending first recorded run" in result.output


def test_eval_refuses_an_unknown_function(workspace):
    result = runner.invoke(app, eval_args(workspace, "--function", "chat"))
    assert result.exit_code != 0
    assert "chat" in result.output


class Recorder:
    """Mock OpenRouter: answers each synthetic question with a fixed reply."""

    def __init__(self):
        self.requests = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        question = json.loads(request.content)["messages"][-1]["content"]
        if question.startswith("Synthetic:"):
            content = json.dumps(
                {
                    "category": "order_status",
                    "priority": "normal",
                    "order_id": "TS-111111",
                    "summary": "Synthetic status question.",
                }
            )
        else:
            content = "You have 30 days [kb-returns]."
        return httpx.Response(
            200,
            json={
                "id": "gen-synthetic",
                "model": "synthetic/system:free",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"content": content}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10, "cost": 0.0},
            },
        )


class NoWait:
    def acquire(self) -> float:
        return 0.0


def test_record_then_replay_end_to_end(workspace):
    ws = workspace
    models = load_models_config(ws / "config.yaml")
    config = Config(models=models, settings=Settings(api_key=SecretStr("synthetic-key-123")))
    recorder = Recorder()
    versions = {"rag": ["v1"], "triage": ["v2"]}
    with ModelClient(
        Mode.RECORD,
        CassetteStore(ws / "cassettes"),
        config,
        httpx.MockTransport(recorder),
        limiter=NoWait(),
    ) as client:
        recorded = run(
            client,
            models.system,
            mode=Mode.RECORD,
            cassettes_dir=ws / "cassettes",
            results_dir=ws / "recorded-results",
            rag_cases=load_rag(ws / "datasets" / "rag.jsonl"),
            triage_cases=load_triage(ws / "datasets" / "triage.jsonl"),
            dataset_paths={
                "rag": ws / "datasets" / "rag.jsonl",
                "triage": ws / "datasets" / "triage.jsonl",
            },
            versions=versions,
        )
    assert recorder.requests == 2
    write_manifest(ws, versions)

    result = runner.invoke(app, eval_args(ws))
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (ws / "results").iterdir()) == ["rag-v1.json", "triage-v2.json"]
    assert "rag v1" in result.output and "triage v2" in result.output

    replayed = json.loads((ws / "results" / "rag-v1.json").read_text(encoding="utf-8"))
    original = recorded.results[0].model_dump(mode="json")
    assert replayed["mode"] == "replay" and original["mode"] == "record"
    assert replayed["cases"] == original["cases"]
    assert replayed["summary"] == original["summary"]
    assert recorder.requests == 2  # replay made no request


def test_eval_can_select_a_function(workspace):
    write_manifest(workspace, {"rag": ["v1"], "triage": ["v1"]})
    result = runner.invoke(app, eval_args(workspace, "--function", "triage"))
    # Triage is selected, so the first missing recording is a triage call.
    assert result.exit_code == 1
    assert "no recording for tri-001/v1/0" in result.output


def test_retrieval_reports_misses_and_recall(workspace):
    write_jsonl(workspace / "datasets" / "rag.jsonl", [RAG_ROW, MISS_ROW])
    result = runner.invoke(
        app, ["retrieval", "--dataset", str(workspace / "datasets" / "rag.jsonl")]
    )
    assert result.exit_code == 0, result.output
    assert "rag-001" in result.output and "rag-002" in result.output
    assert "missing kb-warranty" in result.output
    assert "1/2 expected documents retrieved (50.0%)" in result.output
    assert "1/2 cases with every expected document" in result.output


def test_python_dash_m_runs_the_cli():
    done = subprocess.run(
        [sys.executable, "-m", "llmeval", "--help"], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "eval" in done.stdout and "retrieval" in done.stdout
