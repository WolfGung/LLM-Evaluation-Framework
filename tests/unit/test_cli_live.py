"""`llmeval live`: every case again against the API, results to results-live/ only.

Synthetic data: the config, the datasets, every reply, price, quota number
and error are made up for the test (see `synthetic_openrouter`). Requests go
to an httpx MockTransport, never to the network. Everything is written only
into `tmp_path`; nothing is written to the repository's `cassettes/`,
`results/` or `results-live/`.
"""

import json

import httpx
import pytest
from typer.testing import CliRunner

from llmeval import cli
from llmeval.checks.judge import RUBRIC_PATH
from llmeval.cli import app
from tests.unit.synthetic_judge import ROOT
from tests.unit.synthetic_openrouter import (
    FAKE_KEY,
    NoWait,
    SyntheticOpenRouter,
    everything_under,
    make_workspace,
)

runner = CliRunner()
RUBRIC = ROOT / RUBRIC_PATH
# The synthetic workspace: 12 system calls, 4 gradings, 4 pairwise questions.
ALL_CALLS = 20
LIVE_FILES = ["rag-v1-vs-v2.json", "rag-v1.json", "rag-v2.json", "triage-v1.json", "triage-v2.json"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("OPENROUTER_API_KEY", "LLMEVAL_MODE", "MAX_RUN_COST_USD"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    return make_workspace(tmp_path)


@pytest.fixture
def network(monkeypatch):
    def use(router):
        net = cli.Network(httpx.MockTransport(router), limiter=NoWait(), sleep=lambda _: None)
        monkeypatch.setattr(cli, "_network", lambda: net)
        return router

    return use


def live(ws):
    return runner.invoke(
        app,
        [
            "live",
            "--config",
            str(ws / "config.yaml"),
            "--datasets-dir",
            str(ws / "datasets"),
            "--rubric",
            str(RUBRIC),
            "--results-dir",
            str(ws / "results-live"),
        ],
    )


def test_live_calls_every_case_again_and_writes_only_results_live(ws, network):
    router = network(SyntheticOpenRouter())
    result = live(ws)
    assert result.exit_code == 0, result.output
    assert len(router.chat_bodies) == ALL_CALLS
    assert sorted(p.name for p in (ws / "results-live").iterdir()) == LIVE_FILES
    for name in LIVE_FILES:
        written = json.loads((ws / "results-live" / name).read_text(encoding="utf-8"))
        assert written["mode"] == "live"
    # Nothing is recorded, and the replay results are not touched.
    assert sorted(p.name for p in (ws / "cassettes").iterdir()) == [".gitkeep"]
    assert not (ws / "results").exists()
    out = result.output
    assert "spend limit MAX_RUN_COST_USD: $1.00; the estimate is within it" in out
    assert "free requests left today (GET /api/v1/key): 1000" in out
    assert f"live run: up to {ALL_CALLS} calls, at most 18 per minute; nothing is recorded" in out
    assert "rag v1 vs v2: v1 0, v2 0, tie 2" in out
    assert f"wrote {ws / 'results-live' / 'rag-v1.json'}" in out
    assert f"compare with the baseline: llmeval gate --results-dir {ws / 'results-live'}" in out
    assert FAKE_KEY not in out


def test_live_without_a_key_sends_nothing(ws, network, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    router = network(SyntheticOpenRouter())
    result = live(ws)
    assert result.exit_code == 1
    assert "OPENROUTER_API_KEY is not set" in result.output
    assert router.paths == []
    assert not (ws / "results-live").exists()


def test_live_refuses_a_run_above_the_spend_limit_before_any_call(ws, network, monkeypatch):
    config = (ws / "config.yaml").read_text(encoding="utf-8")
    config = config.replace("synthetic/system:free", "synthetic/system-paid")
    config = config.replace("synthetic/judge:free", "synthetic/judge-paid")
    (ws / "config.yaml").write_text(config, encoding="utf-8")
    monkeypatch.setenv("MAX_RUN_COST_USD", "0.10")
    prices = {
        "synthetic/system-paid": ("0.001", "0.001"),
        "synthetic/judge-paid": ("0.001", "0.001"),
    }
    router = network(SyntheticOpenRouter(prices=prices))
    result = live(ws)
    assert result.exit_code == 1
    assert "refused:" in result.output
    assert "exceeds MAX_RUN_COST_USD=$0.10; nothing was sent" in result.output
    assert router.chat_bodies == []
    assert not (ws / "results-live").exists()


def test_live_refuses_when_the_free_requests_left_cannot_cover_the_run(ws, network):
    router = network(SyntheticOpenRouter(remaining=5))
    result = live(ws)
    assert result.exit_code == 1
    assert (
        f"refused: a live run sends up to {ALL_CALLS} free-model requests, and the key has 5 "
        "left today (GET /api/v1/key); nothing was sent"
    ) in result.output
    assert router.chat_bodies == []
    assert not (ws / "results-live").exists()


def test_live_goes_on_when_the_key_endpoint_does_not_report_free_requests(ws, network):
    network(SyntheticOpenRouter(remaining=None))
    result = live(ws)
    assert result.exit_code == 0, result.output
    assert "free requests left today: not reported, so a 429 is the stop" in result.output


def test_a_failed_call_stops_the_live_run_and_nothing_is_written(ws, network):
    def fail_later(request, body):
        if len(router.chat_bodies) >= 5:
            return httpx.Response(500, json={"error": {"message": f"upstream down {FAKE_KEY}"}})
        return None

    router = network(SyntheticOpenRouter(chat_override=fail_later))
    result = live(ws)
    assert result.exit_code == 1
    assert "live run stopped:" in result.output
    assert f"nothing written to {ws / 'results-live'}" in result.output
    assert FAKE_KEY not in result.output
    assert not (ws / "results-live").exists()
    assert FAKE_KEY not in everything_under(ws)


def test_a_429_without_a_reset_time_stops_the_live_run(ws, network):
    def busy(request, body):
        return httpx.Response(429, json={"error": {"message": "Provider returned error"}})

    network(SyntheticOpenRouter(chat_override=busy))
    result = live(ws)
    assert result.exit_code == 1
    assert "live run stopped:" in result.output
    assert not (ws / "results-live").exists()
