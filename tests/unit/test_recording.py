"""`llmeval record`: two passes, resumable, inside the free limits, honest about progress.

Synthetic data: the config, the datasets, every reply, price, quota number
and rate-limit header are made up for the test (see `synthetic_openrouter`).
Requests go to an httpx MockTransport, never to the network. Cassettes and
results are written only into `tmp_path`; nothing is written to the
repository's `cassettes/` or `results/`.
"""

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from typer.testing import CliRunner

from llmeval import cli, recording
from llmeval.cassettes import MANIFEST_FILE, CassetteStore, load_manifest
from llmeval.checks.judge import RUBRIC_PATH, load_rubric
from llmeval.cli import app
from llmeval.datasets import file_sha256
from llmeval.recording import EXIT_STOPPED
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
# The synthetic workspace: rag 2 cases x 2 versions x 2 repeats, triage
# 1 x 2 x 2 (12 system calls); the judge grades both rag cases on repeat 0
# per version (4) and compares them in two orders (4): 20 calls.
SYSTEM_CALLS = 12
ALL_CALLS = 20


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
    """Send the CLI's API calls to a synthetic OpenRouter, with no rpm waits."""

    def use(router):
        net = cli.Network(httpx.MockTransport(router), limiter=NoWait(), sleep=lambda s: None)
        monkeypatch.setattr(cli, "_network", lambda: net)
        return router

    return use


def args(command, ws):
    paths = [
        "--config",
        str(ws / "config.yaml"),
        "--datasets-dir",
        str(ws / "datasets"),
        "--cassettes-dir",
        str(ws / "cassettes"),
        "--rubric",
        str(RUBRIC),
    ]
    if command == "eval":
        paths += ["--results-dir", str(ws / "results")]
    return [command, *paths]


def judge_body(body):
    return "response_format" in body


# --- a complete recording ------------------------------------------------------------


def test_record_runs_two_passes_and_writes_the_manifest(ws, network):
    router = network(SyntheticOpenRouter())
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 0, result.output
    assert len(router.chat_bodies) == ALL_CALLS
    # Pass 1 records every system call before the first judge call.
    kinds = [judge_body(body) for body in router.chat_bodies]
    assert kinds == [False] * SYSTEM_CALLS + [True] * (ALL_CALLS - SYSTEM_CALLS)
    out = result.output
    assert f"recorded 1/{ALL_CALLS}  rag-001/v1/0" in out
    assert f"recorded {ALL_CALLS}/{ALL_CALLS}" in out
    assert "pass 2: judge calls planned from the recorded answers: 8 to record" in out
    manifest = load_manifest(ws / "cassettes")
    assert manifest.planned_calls == manifest.recorded_calls == ALL_CALLS
    assert manifest.models == {"system": "synthetic/system:free", "judge": "synthetic/judge:free"}
    assert manifest.prompt_versions == {"rag": ("v1", "v2"), "triage": ("v1", "v2")}
    assert manifest.repeats == 2 and manifest.judge_repeats == "first"
    assert manifest.stability_cases is None
    assert manifest.rubric_sha256 == load_rubric(RUBRIC).sha256
    assert manifest.datasets == {
        "rag.jsonl": file_sha256(ws / "datasets" / "rag.jsonl"),
        "triage.jsonl": file_sha256(ws / "datasets" / "triage.jsonl"),
    }
    assert f"every planned call is recorded ({ALL_CALLS}); wrote" in out
    assert not (ws / "results").exists()  # recording writes no results


def test_a_recording_replays_without_the_network(ws, network):
    router = network(SyntheticOpenRouter())
    assert runner.invoke(app, args("record", ws)).exit_code == 0
    sent = len(router.paths)
    result = runner.invoke(app, args("eval", ws))
    assert result.exit_code == 0, result.output
    assert len(router.paths) == sent  # replay sent nothing
    assert sorted(p.name for p in (ws / "results").iterdir()) == [
        "rag-v1-vs-v2.json",
        "rag-v1.json",
        "rag-v2.json",
        "triage-v1.json",
        "triage-v2.json",
    ]


def test_the_manifest_names_the_stability_subset(tmp_path, network, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    ws = make_workspace(tmp_path, stability_cases="[rag-001]")
    router = network(SyntheticOpenRouter())
    assert runner.invoke(app, args("record", ws)).exit_code == 0
    manifest = load_manifest(ws / "cassettes")
    assert manifest.stability_cases == ("rag-001",)
    # rag-001 twice, rag-002 once, tri-001 once, per version: 8 system calls.
    assert manifest.planned_calls == 8 + 8 == len(router.chat_bodies)


def test_a_complete_recording_records_nothing_again(ws, network):
    router = network(SyntheticOpenRouter())
    assert runner.invoke(app, args("record", ws)).exit_code == 0
    first = (ws / "cassettes" / MANIFEST_FILE).read_text(encoding="utf-8")
    sent = len(router.chat_bodies)
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 0, result.output
    assert len(router.chat_bodies) == sent
    assert "nothing left to record" in result.output
    assert (ws / "cassettes" / MANIFEST_FILE).read_text(encoding="utf-8") == first


# --- the key and the budget come first --------------------------------------------------


def test_record_needs_the_key_and_sends_nothing_without_it(ws, network, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    router = network(SyntheticOpenRouter())
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 1
    assert "OPENROUTER_API_KEY is not set" in result.output
    assert router.paths == []
    assert sorted(p.name for p in (ws / "cassettes").iterdir()) == [".gitkeep"]


def test_the_budget_guard_runs_before_any_model_call(ws, network, monkeypatch):
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
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 1
    assert "refused: estimated cost $" in result.output
    assert "exceeds MAX_RUN_COST_USD=$0.10; nothing was sent" in result.output
    assert router.paths == ["/api/v1/models"]  # prices only, no model call
    assert router.chat_bodies == []


# --- stopping and resuming ------------------------------------------------------------


def rate_limited(after, reset):
    """A chat override: answer `after` calls, then HTTP 429 with `reset` headers."""
    state = {"answered": 0, "limited": 0}

    def override(request, body):
        if state["answered"] < after:
            state["answered"] += 1
            return None
        state["limited"] += 1
        return httpx.Response(429, headers=reset, json={"error": {"message": "rate limited"}})

    return override, state


def test_a_daily_429_stops_cleanly_and_a_rerun_continues(ws, network):
    tomorrow = datetime.now(UTC) + timedelta(hours=10)
    reset = {"X-RateLimit-Reset": str(int(tomorrow.timestamp() * 1000))}
    override, _ = rate_limited(5, reset)
    router = network(SyntheticOpenRouter(chat_override=override))
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == EXIT_STOPPED == 75
    assert f"free daily quota reached; 5 of {ALL_CALLS} calls recorded; rerun after" in (
        result.output
    )
    assert "rerun make record to continue: the 5 recorded calls are kept and skipped" in (
        result.output
    )
    # Stopped in pass 1: the judge calls are still counted at their upper bound.
    assert "the total counts judge calls at their upper bound until the answers exist" in (
        result.output
    )
    assert load_manifest(ws / "cassettes") is None  # incomplete: no manifest
    assert len(CassetteStore(ws / "cassettes")) == 5

    router = network(SyntheticOpenRouter())
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 0, result.output
    assert len(router.chat_bodies) == ALL_CALLS - 5  # the first five were skipped
    assert f"recorded 6/{ALL_CALLS}" in result.output
    assert load_manifest(ws / "cassettes").planned_calls == ALL_CALLS


def upstream_limited(after):
    """A chat override: answer `after` calls, then HTTP 429 without reset headers
    and with the provider's message (which echoes the key, to test the scrub)."""
    state = {"answered": 0, "limited": 0}

    def override(request, body):
        if state["answered"] < after:
            state["answered"] += 1
            return None
        state["limited"] += 1
        message = (
            f"qwen is temporarily rate-limited upstream ({request.headers['authorization']}). "
            "Please retry shortly."
        )
        return httpx.Response(429, json={"error": {"message": message}})

    return override, state


def test_a_429_without_a_reset_time_stops_at_once_and_is_not_the_daily_quota(ws, network):
    override, state = upstream_limited(3)
    network(SyntheticOpenRouter(remaining=47, chat_override=override))
    result = runner.invoke(app, args("record", ws))
    out = result.output
    assert result.exit_code == EXIT_STOPPED
    assert state["limited"] == 1  # no retry, no guessed wait
    assert "daily quota reached" not in out
    assert (
        "rate limited: HTTP 429 without a reset time (API: qwen is temporarily rate-limited "
        "upstream (Bearer [redacted]). Please retry shortly.); 3 of 20 calls recorded; rerun later"
    ) in out
    assert "HTTP 429 came without a reset time, so the run stopped at once" in out
    # The key endpoint is read again: 47 at the start, 3 used since.
    assert (
        "the key still has 44 free requests today, so this is not the daily quota: "
        "rerun in a few minutes"
    ) in out
    assert FAKE_KEY not in out
    assert len(CassetteStore(ws / "cassettes")) == 3


def test_after_a_429_without_reset_a_zero_key_count_points_at_the_daily_reset(ws, network):
    router = SyntheticOpenRouter(remaining=10)

    def override(request, body):
        if len(router.chat_bodies) < 2:
            return None
        router.remaining = 0  # the quota ran out elsewhere meanwhile
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    router.chat_override = override
    network(router)
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == EXIT_STOPPED
    assert (
        "the key has no free requests left today, so this is the daily quota: "
        "rerun after the daily reset"
    ) in result.output


def test_after_a_429_without_reset_an_unreadable_key_endpoint_is_said_plainly(ws, network):
    router = SyntheticOpenRouter(remaining=10)
    reads = {"key": 0}

    def handler(request):
        if request.url.path.endswith("/key"):
            reads["key"] += 1
            if reads["key"] > 1:
                return httpx.Response(503, json={"error": {"message": "key service down"}})
        return router(request)

    def override(request, body):
        if len(router.chat_bodies) < 2:
            return None
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    router.chat_override = override
    network(handler)
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == EXIT_STOPPED
    assert (
        "the key endpoint could not be read (/api/v1/key returned HTTP 503: key service down), "
        "so it is unknown whether the daily quota is used up: rerun later"
    ) in result.output


def test_a_used_up_key_stops_before_the_first_call(ws, network):
    router = network(SyntheticOpenRouter(remaining=0))
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == EXIT_STOPPED
    assert (
        "the key has no free-model requests left today (GET /api/v1/key: 0 remaining); "
        f"0 of {ALL_CALLS} calls recorded; rerun after the daily reset"
    ) in result.output
    assert router.chat_bodies == []


def test_the_key_endpoint_count_stops_the_run_before_a_429(ws, network):
    router = network(SyntheticOpenRouter(remaining=4))
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == EXIT_STOPPED
    assert "the key has 4 free-model requests left today; the recording stops after them" in (
        result.output
    )
    assert len(router.chat_bodies) == 4
    # It asked the key endpoint again before stopping, in case more were allowed.
    assert router.paths.count("/api/v1/key") == 2
    assert f"4 of {ALL_CALLS} calls recorded; rerun after the daily reset" in result.output


def test_an_unknown_free_quota_leaves_the_stop_to_the_429(ws, network):
    router = network(SyntheticOpenRouter(remaining=None))
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 0, result.output
    assert "free quota today: unknown, so the recording relies on HTTP 429 to stop" in (
        result.output
    )
    assert len(router.chat_bodies) == ALL_CALLS


def test_an_api_error_stops_with_progress_and_the_key_scrubbed(ws, network):
    def broken(request, body):
        if judge_body(body):
            message = f"upstream failed for {request.headers['authorization']}"
            return httpx.Response(500, json={"error": {"message": message}})
        return None

    network(SyntheticOpenRouter(chat_override=broken))
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 1
    assert "stopped: /api/v1/chat/completions returned HTTP 500" in result.output
    assert f"{SYSTEM_CALLS} of {ALL_CALLS} calls recorded" in result.output
    assert "rerun make record to retry: recorded calls are kept and skipped" in result.output
    assert "[redacted]" in result.output
    assert FAKE_KEY not in result.output
    assert load_manifest(ws / "cassettes") is None


# --- the rpm limit and the key ------------------------------------------------------------


def test_the_client_keeps_the_configured_rpm(ws, network, monkeypatch):
    built = []

    class CountingLimiter:
        def __init__(self, rpm, **kwargs):
            built.append(rpm)
            self.calls = 0

        def acquire(self):
            self.calls += 1
            return 0.0

    monkeypatch.setattr("llmeval.client.RateLimiter", CountingLimiter)
    router = SyntheticOpenRouter()
    monkeypatch.setattr(cli, "_network", lambda: cli.Network(httpx.MockTransport(router)))
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 0, result.output
    assert built == [18]  # rpm from the config


def test_the_key_never_reaches_the_output_or_the_files(ws, network):
    network(SyntheticOpenRouter())
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 0, result.output
    assert FAKE_KEY not in result.output
    assert FAKE_KEY not in everything_under(ws)


def test_record_writes_only_into_the_given_cassettes_directory(ws, network):
    network(SyntheticOpenRouter())
    assert runner.invoke(app, args("record", ws)).exit_code == 0
    names = sorted(p.name for p in (ws / "cassettes").iterdir())
    assert MANIFEST_FILE in names and ".gitkeep" in names
    assert all(n.endswith(".jsonl") for n in names if n not in (MANIFEST_FILE, ".gitkeep"))
    assert not list(ws.glob("*.tmp")) and not list((ws / "cassettes").glob("*.tmp"))


def test_the_record_docstring_says_a_429_without_a_reset_stops_at_once():
    for text in (recording.__doc__, cli.record_command.__doc__):
        assert "without a reset time" in " ".join(text.split())
