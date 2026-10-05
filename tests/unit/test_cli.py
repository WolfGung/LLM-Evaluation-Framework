"""The command line: `llmeval eval` and `llmeval retrieval`.

Synthetic data: the datasets, the model config, the manifest and every model
reply below are made up for the test. The end-to-end test records synthetic
replies through the real client with a mock transport, into `tmp_path` only;
nothing is written to the repository's `cassettes/` or `results/`.
"""

import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from llmeval.cassettes import MANIFEST_FILE, CassetteStore
from llmeval.checks.judge import RUBRIC_PATH, load_rubric
from llmeval.cli import app
from llmeval.client import ModelClient
from llmeval.config import Config, Mode, Settings, load_models_config
from llmeval.datasets import load_rag, load_triage
from llmeval.runner import run

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[2]
RUBRIC = ROOT / RUBRIC_PATH

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
        "--rubric",
        str(RUBRIC),
        *extra,
    ]


def write_manifest(ws, versions, rubric_sha256=None, repeats=1, judge_repeats="first"):
    manifest = {
        "models": {"system": "synthetic/system:free", "judge": "synthetic/judge:free"},
        "prompt_versions": versions,
        "repeats": repeats,
        "judge_repeats": judge_repeats,
        "datasets": {"rag.jsonl": "0" * 64, "triage.jsonl": "1" * 64},
        "recorded_from": "2026-01-01T10:00:00Z",
        "recorded_to": "2026-01-01T10:05:00Z",
        "planned_calls": 2,
        "recorded_calls": 2,
        "rubric_sha256": rubric_sha256 or load_rubric(RUBRIC).sha256,
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


def test_eval_refuses_a_manifest_recorded_with_other_models(workspace):
    write_manifest(workspace, {"rag": ["v1"], "triage": ["v1"]})
    (workspace / "config.yaml").write_text(
        CONFIG_YAML.replace("synthetic/system:free", "synthetic/other:free"), encoding="utf-8"
    )
    result = runner.invoke(app, eval_args(workspace))
    assert result.exit_code == 1
    assert (
        "system model: recorded with synthetic/system:free, config says synthetic/other:free: "
        "re-record or restore the config"
    ) in result.output


def test_eval_refuses_an_unknown_function(workspace):
    result = runner.invoke(app, eval_args(workspace, "--function", "chat"))
    assert result.exit_code != 0
    assert "chat" in result.output


class Recorder:
    """Mock OpenRouter: answers each synthetic question, and each judge question,
    with a fixed reply."""

    def __init__(self):
        self.requests = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        body = json.loads(request.content)
        question = body["messages"][-1]["content"]
        schema = body.get("response_format", {}).get("json_schema", {}).get("name")
        if schema == "judge_verdict":
            content = json.dumps(
                {
                    "groundedness": 5,
                    "helpfulness": 4,
                    "tone": 5,
                    "pass": True,
                    "reasons": "Synthetic reasons.",
                }
            )
        elif schema == "pairwise_verdict":
            content = json.dumps({"preferred": "tie", "reasons": "Synthetic reasons."})
        elif question.startswith("Synthetic:"):
            content = json.dumps(
                {
                    "category": "order_status",
                    "priority": "normal",
                    "order_id": "TS-111111",
                    "summary": "Synthetic status question.",
                }
            )
        elif "Follow these rules" in body["messages"][0]["content"]:  # assistant v2
            content = "Returns are accepted within 30 days [kb-returns]."
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
    versions = {"rag": ["v1", "v2"], "triage": ["v2"]}
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
            judge=models.judge,
            rubric=load_rubric(RUBRIC),
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
    # rag 2 answers + 2 gradings + 2 pairwise questions, triage 1.
    assert recorder.requests == 7
    write_manifest(ws, versions)

    result = runner.invoke(app, eval_args(ws))
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (ws / "results").iterdir()) == [
        "rag-v1-vs-v2.json",
        "rag-v1.json",
        "rag-v2.json",
        "triage-v2.json",
    ]
    assert "rag v1" in result.output and "triage v2" in result.output
    assert "judge 1/1 (100.0%)" in result.output
    assert "stable" not in result.output  # one repeat: no stability layer
    assert "  pairwise calls 2, latency p50 " in result.output
    assert (
        "rag v1 vs v2: v1 0, v2 0, tie 1, inconsistent 0, identical 0, invalid 0" in result.output
    )
    # The synthetic manifest holds placeholder hashes, so both datasets differ.
    assert "notice: datasets changed since the recording: rag.jsonl, triage.jsonl" in result.output

    replayed = json.loads((ws / "results" / "rag-v1.json").read_text(encoding="utf-8"))
    original = recorded.results[0].model_dump(mode="json")
    assert replayed["mode"] == "replay" and original["mode"] == "record"
    assert replayed["cases"] == original["cases"]
    assert replayed["summary"] == original["summary"]
    assert replayed["cases"][0]["runs"][0]["judge"]["scores"]["helpfulness"] == 4
    assert recorder.requests == 7  # replay made no request
    assert "rubric changed" not in result.output


@pytest.mark.parametrize(
    ("manifest_says", "missing"),
    [("first", None), ("all", "rag-001:judge/v1/1")],
)
def test_eval_grades_the_repeats_the_manifest_names(workspace, manifest_says, missing):
    ws = workspace
    models = load_models_config(ws / "config.yaml")
    config = Config(models=models, settings=Settings(api_key=SecretStr("synthetic-key-123")))
    versions = {"rag": ["v1"], "triage": ["v1"]}
    with ModelClient(
        Mode.RECORD,
        CassetteStore(ws / "cassettes"),
        config,
        httpx.MockTransport(Recorder()),
        limiter=NoWait(),
    ) as client:
        run(
            client,
            models.system,
            judge=models.judge,
            rubric=load_rubric(RUBRIC),
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
            repeats=2,
            judge_repeats="first",
        )
    write_manifest(ws, versions, repeats=2, judge_repeats=manifest_says)
    result = runner.invoke(app, eval_args(ws))
    if missing is None:
        assert result.exit_code == 0, result.output
        # Two repeats with the same reply: every repeated case is stable.
        assert "all checks 2/2, stable 1/1 (100.0%)" in result.output
        assert "  system calls 2, latency p50 " in result.output
        assert "  judge calls 1, latency p50 " in result.output
        assert "cost $0.000000 ($0.000000 per case)" in result.output
        rag = json.loads((ws / "results" / "rag-v1.json").read_text(encoding="utf-8"))
        assert rag["judge_repeats"] == "first"
        assert [run["judge"] is not None for run in rag["cases"][0]["runs"]] == [True, False]
    else:
        assert result.exit_code == 1
        assert f"no recording for {missing}: run make record" in result.output


def test_eval_says_when_the_rubric_changed_since_the_recording(workspace):
    write_manifest(workspace, {"rag": ["v1"], "triage": ["v1"]}, rubric_sha256="0" * 64)
    result = runner.invoke(app, eval_args(workspace))
    assert result.exit_code == 1  # nothing is recorded in this workspace
    assert (
        result.output.count("notice: rubric changed since the recording: re-record the judge layer")
        == 1
    )


def test_a_front_matter_edit_gives_no_rubric_notice(workspace):
    write_manifest(workspace, {"rag": ["v1"], "triage": ["v1"]})
    rubric = workspace / "judge.md"
    text = RUBRIC.read_text(encoding="utf-8")
    rubric.write_text(text.replace("---\n", "---\n# Synthetic comment.\n", 1), encoding="utf-8")
    args = eval_args(workspace)
    args[args.index("--rubric") + 1] = str(rubric)
    result = runner.invoke(app, args)
    assert result.exit_code == 1  # nothing is recorded in this workspace
    assert "rubric changed" not in result.output


def test_eval_refuses_a_broken_rubric(workspace):
    write_manifest(workspace, {"rag": ["v1"], "triage": ["v1"]})
    rubric = workspace / "judge.md"
    rubric.write_text(RUBRIC.read_text(encoding="utf-8").replace("is at least 4", "is at least 2"))
    args = eval_args(workspace)
    args[args.index("--rubric") + 1] = str(rubric)
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "the pass rule says groundedness is at least 2" in result.output


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


def test_the_performance_line_says_n_a_without_calls():
    from llmeval.cli import _perf_line
    from llmeval.perf import performance

    line = _perf_line("judge", performance([], cases=0))
    assert line == (
        "  judge calls 0, latency p50 n/a, p95 n/a, mean tokens in n/a out n/a, cost $0.000000"
    )
    assert "None" not in line


def test_eval_lists_at_most_ten_unstable_cases_with_their_flips():
    from llmeval.cli import _unstable_lines
    from llmeval.stability import Stability, UnstableCase

    unstable = [
        UnstableCase(
            id=f"tri-{n:03d}",
            category="payment",
            checks={"reference/priority_match": [True, False]} if n % 2 else {},
            labels={} if n % 2 else {"category": ["payment", "shipping"]},
        )
        for n in range(1, 13)
    ]
    lines = _unstable_lines(Stability(repeated=20, stable=8, stable_share=0.4, unstable=unstable))
    assert lines[0] == "  unstable tri-001: reference/priority_match"
    assert lines[1] == "  unstable tri-002: category"
    assert len(lines) == 11
    assert lines[-1] == "  and 2 more unstable cases"
    assert _unstable_lines(None) == []
    eleven = Stability(repeated=20, stable=9, stable_share=0.45, unstable=unstable[:11])
    assert _unstable_lines(eleven)[-1] == "  and 1 more unstable case"


GONE = (
    "the manifest names prompt version rag v9, but its prompt file is gone: restore it or "
    "record again"
)


def status_args(ws):
    return [
        "status",
        "--config",
        str(ws / "config.yaml"),
        "--datasets-dir",
        str(ws / "datasets"),
        "--cassettes-dir",
        str(ws / "cassettes"),
        "--rubric",
        str(RUBRIC),
    ]


@pytest.mark.parametrize("command", ["eval", "status"])
def test_a_manifest_naming_a_deleted_prompt_version_fails_in_one_line(workspace, command):
    write_manifest(workspace, {"rag": ["v1", "v9"], "triage": ["v1"]})
    args = eval_args(workspace) if command == "eval" else status_args(workspace)
    result = runner.invoke(app, args)
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)  # no traceback
    assert [line for line in result.output.splitlines() if "prompt" in line] == [GONE]
    assert not (workspace / "results").exists()
