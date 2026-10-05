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

from llmeval import cli, recording
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
def sleeps():
    """The seconds of every wait the run asked for, in order (nothing really waits)."""
    return []


@pytest.fixture
def network(monkeypatch, sleeps):
    def use(router):
        net = cli.Network(httpx.MockTransport(router), limiter=NoWait(), sleep=sleeps.append)
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


BUSY_REPLY = {"error": {"message": "Provider returned error"}}
BUSY = "upstream provider busy (HTTP 429, API: Provider returned error)"


def test_an_upstream_429_waits_as_recording_does_and_the_run_completes(ws, network, sleeps):
    tries = {"n": 0}

    def busy_twice(request, body):
        tries["n"] += 1
        return httpx.Response(429, json=BUSY_REPLY) if tries["n"] in (2, 3) else None

    router = network(SyntheticOpenRouter(chat_override=busy_twice))
    result = live(ws)
    out = result.output
    assert result.exit_code == 0, out
    assert sleeps == [30, 60]
    assert f"{BUSY}; waiting 30 s, then retrying (1 of 4)" in out
    assert f"{BUSY}; waiting 60 s, then retrying (2 of 4)" in out
    assert len(router.chat_bodies) == ALL_CALLS
    assert sorted(p.name for p in (ws / "results-live").iterdir()) == LIVE_FILES


def test_a_success_resets_the_live_backoff(ws, network, sleeps):
    tries = {"n": 0}

    def two_calls_busy_once(request, body):
        tries["n"] += 1
        return httpx.Response(429, json=BUSY_REPLY) if tries["n"] in (2, 6) else None

    network(SyntheticOpenRouter(chat_override=two_calls_busy_once))
    result = live(ws)
    assert result.exit_code == 0, result.output
    assert sleeps == [30, 30]


def test_an_upstream_429_that_persists_stops_the_live_run_after_four_waits(ws, network, sleeps):
    def busy(request, body):
        return httpx.Response(429, json=BUSY_REPLY)

    network(SyntheticOpenRouter(chat_override=busy))
    result = live(ws)
    out = result.output
    assert result.exit_code == 1
    assert sleeps == [30, 60, 120, 240]
    assert "live run stopped:" in out
    assert (
        "HTTP 429 without a reset time came back after 4 waits (30 s, 60 s, 120 s and 240 s; "
        "7.5 minutes in all), so the run stopped"
    ) in out
    assert f"nothing written to {ws / 'results-live'}" in out
    assert not (ws / "results-live").exists()


def test_after_a_429_without_reset_a_zero_key_count_stops_the_live_run_at_once(ws, network, sleeps):
    def quota_gone(request, body):
        router.remaining = 0
        return httpx.Response(429, json=BUSY_REPLY)

    router = network(SyntheticOpenRouter(chat_override=quota_gone))
    result = live(ws)
    out = result.output
    assert result.exit_code == 1
    assert sleeps == []
    assert "stopped at once" in out
    assert "the key has no free requests left today, so this is the daily quota" in out
    assert not (ws / "results-live").exists()


def test_a_429_without_reset_on_a_paid_model_stops_the_live_run_at_once(
    ws, network, sleeps, monkeypatch
):
    config = (ws / "config.yaml").read_text(encoding="utf-8")
    (ws / "config.yaml").write_text(
        config.replace("synthetic/system:free", "synthetic/system-paid"), encoding="utf-8"
    )
    monkeypatch.setenv("MAX_RUN_COST_USD", "100")

    def busy(request, body):
        return httpx.Response(429, json=BUSY_REPLY)

    prices = {"synthetic/system-paid": ("0.0001", "0.0001")}
    network(SyntheticOpenRouter(prices=prices, chat_override=busy))
    result = live(ws)
    assert result.exit_code == 1
    assert sleeps == []
    assert "live run stopped:" in result.output


def test_the_live_docstring_says_an_upstream_429_waits_as_recording_does():
    flat = " ".join(cli.live_command.__doc__.split())
    waits = [f"{wait} s" for wait in recording.UPSTREAM_BACKOFF_S]
    assert f"{', '.join(waits[:-1])} and {waits[-1]}" in flat
    assert "without a reset time" in flat
