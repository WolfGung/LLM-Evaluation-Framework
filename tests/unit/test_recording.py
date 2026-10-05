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
    unanswerable_rows,
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
    # The cost per role and the limit, as estimate prints them.
    assert "  system (synthetic/system:free): $0.00 for 12 calls (a :free model id)" in out
    assert "  judge (synthetic/judge:free): $0.00 for up to 8 calls (a :free model id)" in out
    assert "spend limit MAX_RUN_COST_USD: $1.00; the estimate is within it" in out
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
    assert "from the published prices (GET /api/v1/models)" in result.output  # per role
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
    out = result.output
    assert result.exit_code == 1
    # Every judge call fails: three in a row stop the run.
    assert (
        "skipped rag-001:judge/v1/0: /api/v1/chat/completions returned HTTP 500: "
        "upstream failed for Bearer [redacted]"
    ) in out
    assert (
        "stopped: 3 different requests in a row failed, and none of their kinds has succeeded "
        "in this run: check the model id and config"
    ) in out
    assert f"{SYSTEM_CALLS} of {ALL_CALLS} calls recorded" in out
    assert "rerun make record to retry: recorded calls are kept and skipped" in out
    assert FAKE_KEY not in out
    assert load_manifest(ws / "cassettes") is None


def test_day_one_records_one_call_of_each_kind_first(ws, network):
    network(SyntheticOpenRouter())
    out = runner.invoke(app, args("record", ws)).output
    progress = [line.split("  ")[1] for line in out.splitlines() if line.startswith("recorded ")]
    # Pass 1: rag v1, rag v2, triage v1, triage v2, then the rest in plan order.
    assert progress[:5] == [
        "rag-001/v1/0",
        "rag-001/v2/0",
        "tri-001/v1/0",
        "tri-001/v2/0",
        "rag-001/v1/1",
    ]
    # Pass 2: a grading of each version's answers and one pairwise question first.
    assert progress[SYSTEM_CALLS : SYSTEM_CALLS + 3] == [
        "rag-001:judge/v1/0",
        "rag-001:judge/v2/0",
        "rag-001:A=v1/v1-v2/0",
    ]
    assert len(progress) == len(set(progress)) == ALL_CALLS


def test_a_refused_request_is_skipped_and_the_rest_is_recorded(ws, network):
    question = "Do you rent out ladders by the day?"  # rag-002

    def moderated(request, body):
        v1 = "Follow these rules" not in body["messages"][0]["content"]
        if v1 and body["messages"][-1]["content"] == question and not judge_body(body):
            return httpx.Response(403, json={"error": {"message": "flagged by moderation"}})
        return None

    router = network(SyntheticOpenRouter(chat_override=moderated))
    result = runner.invoke(app, args("record", ws))
    out = result.output
    assert result.exit_code == 1
    # rag-002 v1 is tried once; its second repeat is not sent; the rest is recorded.
    assert "skipped rag-002/v1/0: /api/v1/chat/completions returned HTTP 403" in out
    assert len(router.chat_bodies) == ALL_CALLS - 2 - 3
    assert "skipped 1 call (rerun make record to retry it):" in out
    assert "not sent 1 call (rerun make record to send it):" in out
    assert (
        "  rag-002/v1/1: an earlier repeat failed "
        "(/api/v1/chat/completions returned HTTP 403: flagged by moderation)"
    ) in out
    # Its grading and both pairwise questions wait for the missing answer.
    assert "3 judge calls wait for answers that were skipped" in out
    assert load_manifest(ws / "cassettes") is None

    router = network(SyntheticOpenRouter())
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 0, result.output
    assert len(router.chat_bodies) == 2 + 3
    assert load_manifest(ws / "cassettes").planned_calls == ALL_CALLS


def test_failures_that_are_not_in_a_row_do_not_stop_the_run(ws, network):
    calls = {"n": 0, "failed": 0}

    def every_other(request, body):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            calls["failed"] += 1
            return httpx.Response(502, json={"error": {"message": "bad gateway"}})
        return None

    network(SyntheticOpenRouter(chat_override=every_other))
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 1
    assert "calls in a row failed" not in result.output
    assert calls["failed"] >= 3
    assert f"skipped {calls['failed']} calls (rerun make record to retry them):" in result.output
    assert "not sent " in result.output  # the other repeats of the failed requests


def test_a_torn_last_line_is_recorded_again(ws, network):
    tomorrow = datetime.now(UTC) + timedelta(hours=10)
    override, _ = rate_limited(5, {"X-RateLimit-Reset": str(int(tomorrow.timestamp() * 1000))})
    network(SyntheticOpenRouter(chat_override=override))
    assert runner.invoke(app, args("record", ws)).exit_code == EXIT_STOPPED
    # As if the last call's write had been cut short: keep a third of its line.
    path = ws / "cassettes" / "rag-v1.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:-1]) + lines[-1][: len(lines[-1]) // 3], encoding="utf-8")

    status = runner.invoke(app, args("status", ws))
    notice = (
        "ignored an unfinished last line in rag-v1.jsonl "
        "(or a record run is writing it now); that call will be recorded again"
    )
    assert status.exit_code == 0 and notice in status.output
    router = network(SyntheticOpenRouter())
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 0, result.output
    assert notice in result.output
    assert len(router.chat_bodies) == ALL_CALLS - 4  # the torn call is asked again
    assert all(line.startswith("{") for line in path.read_text(encoding="utf-8").splitlines())
    assert CassetteStore(ws / "cassettes").notices == []


def test_record_cuts_a_torn_tail_even_when_nothing_more_goes_to_that_file(ws, network):
    network(SyntheticOpenRouter())
    assert runner.invoke(app, args("record", ws)).exit_code == 0
    (ws / "cassettes" / MANIFEST_FILE).unlink()
    path = ws / "cassettes" / "triage-v2.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"key": "unfinished')  # a fragment of no planned call
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 0, result.output
    assert "ignored an unfinished last line in triage-v2.jsonl" in result.output
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert CassetteStore(ws / "cassettes").notices == []


# --- one record run at a time ----------------------------------------------------------


def test_a_second_record_run_is_refused_while_one_is_active(ws, network):
    from llmeval.recording import RecordLocked, record_all

    router = SyntheticOpenRouter()
    second = {}

    def start_a_second_session(request, body):
        if not second:  # during the first session's first call
            loaded, inputs = cli._plan_inputs(ws / "config.yaml", ws / "datasets", RUBRIC)
            lines = []
            try:
                record_all(
                    inputs,
                    loaded,
                    ws / "cassettes",
                    cli._dataset_paths(ws / "datasets"),
                    echo=lines.append,
                    transport=httpx.MockTransport(router),
                    limiter=NoWait(),
                )
            except RecordLocked as exc:
                second["refused"] = str(exc)
            second["lines"] = lines
        return None

    router.chat_override = start_a_second_session
    network(router)
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 0, result.output
    assert second["refused"].startswith("another record run is active on ")
    assert second["lines"] == []  # refused before it planned, estimated or sent anything
    assert len(router.chat_bodies) == ALL_CALLS  # only the first session's calls
    assert len(CassetteStore(ws / "cassettes")) == ALL_CALLS


def test_a_held_lock_refuses_record_in_the_cli(ws, network):
    import fcntl

    router = network(SyntheticOpenRouter())
    with (ws / "cassettes" / ".record.lock").open("a") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 1
    assert "another record run is active" in result.output
    assert router.paths == []
    # Released: the next run goes ahead.
    assert runner.invoke(app, args("record", ws)).exit_code == 0


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
    assert {MANIFEST_FILE, ".gitkeep", ".record.lock"} <= set(names)
    others = set(names) - {MANIFEST_FILE, ".gitkeep", ".record.lock"}
    assert others and all(name.endswith(".jsonl") for name in others)
    assert sorted(p.name for p in ws.iterdir()) == ["cassettes", "config.yaml", "datasets"]


def test_the_record_docstring_says_a_429_without_a_reset_stops_at_once():
    for text in (recording.__doc__, cli.record_command.__doc__):
        assert "without a reset time" in " ".join(text.split())


# --- the running spending cap ----------------------------------------------------------


def paid_system(ws):
    """Make the system model a paid one; the judge stays free."""
    config = (ws / "config.yaml").read_text(encoding="utf-8")
    (ws / "config.yaml").write_text(
        config.replace("synthetic/system:free", "synthetic/system-paid"), encoding="utf-8"
    )


def test_the_cap_stops_before_a_call_could_pass_the_limit(ws, network, monkeypatch):
    # The reviewer's probe B: two cheap recorded calls make the history estimate
    # $0.0000, but later calls cost $0.05 each and the limit is $0.0001.
    paid_system(ws)
    prices = {"synthetic/system-paid": ("0.0001", "0.0001")}
    tomorrow = datetime.now(UTC) + timedelta(hours=10)
    override, _ = rate_limited(2, {"X-RateLimit-Reset": str(int(tomorrow.timestamp() * 1000))})
    monkeypatch.setenv("MAX_RUN_COST_USD", "100")
    network(SyntheticOpenRouter(cost=0.0, prices=prices, chat_override=override))
    assert runner.invoke(app, args("record", ws)).exit_code == EXIT_STOPPED

    monkeypatch.setenv("MAX_RUN_COST_USD", "0.0001")
    router = network(SyntheticOpenRouter(cost=0.05, prices=prices))
    result = runner.invoke(app, args("record", ws))
    out = result.output
    assert result.exit_code == 1
    assert "spend limit MAX_RUN_COST_USD: $0.0001; the estimate is within it" in out
    assert "system (synthetic/system-paid): $0.0000 for 10 calls" in out  # the history guess
    assert "spending cap: this run spent $0.00 of MAX_RUN_COST_USD=$0.0001" in out
    assert "the next call could cost up to $" in out
    assert router.chat_bodies == []  # stopped before sending
    assert "2 of 20 calls recorded" in out
    assert "raise MAX_RUN_COST_USD or rerun make record later to continue" in out


def test_the_cap_stops_when_real_costs_pass_the_published_bound(ws, network, monkeypatch):
    paid_system(ws)
    # Published prices far below what the calls really cost.
    prices = {"synthetic/system-paid": ("0.0000000001", "0.0000000001")}
    monkeypatch.setenv("MAX_RUN_COST_USD", "0.06")
    router = network(SyntheticOpenRouter(cost=0.05, prices=prices))
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 1
    assert len(router.chat_bodies) == 2  # 0.05, then 0.10 > 0.06: stop
    assert "spending cap: this run spent $0.10 of MAX_RUN_COST_USD=$0.06" in result.output
    assert "cost more than its published-price bound" in result.output


def test_free_calls_never_touch_the_cap(ws, network, monkeypatch):
    monkeypatch.setenv("MAX_RUN_COST_USD", "0")
    router = network(SyntheticOpenRouter(cost=0.0))
    assert runner.invoke(app, args("record", ws)).exit_code == 0
    assert "/api/v1/models" not in router.paths


def test_a_paid_model_without_a_published_price_is_refused_before_any_call(
    ws, network, monkeypatch
):
    paid_system(ws)
    tomorrow = datetime.now(UTC) + timedelta(hours=10)
    override, _ = rate_limited(2, {"X-RateLimit-Reset": str(int(tomorrow.timestamp() * 1000))})
    monkeypatch.setenv("MAX_RUN_COST_USD", "100")
    prices = {"synthetic/system-paid": ("0.0001", "0.0001")}
    network(SyntheticOpenRouter(cost=0.0, prices=prices, chat_override=override))
    assert runner.invoke(app, args("record", ws)).exit_code == EXIT_STOPPED
    # Now the price list no longer has the model; the history alone passes the guard.
    router = network(SyntheticOpenRouter(cost=0.0, prices={}))
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 1
    assert "no published price for: synthetic/system-paid" in result.output
    assert "the running spending cap needs it" in result.output
    assert router.chat_bodies == []


def test_an_unknown_call_cost_counts_at_its_published_bound():
    from llmeval.recording import SpendCap

    cap = SpendCap(limit=1.0, prices={})
    cap.add(None, bound=0.3)
    cap.add(0.1, bound=0.3)
    assert cap.spent == pytest.approx(0.4)
    assert cap.unknown == 1


def test_one_skipped_call_is_counted_in_the_singular(ws, network):
    calls = {"n": 0}

    def seventh_fails(request, body):
        calls["n"] += 1
        if calls["n"] == 7:
            return httpx.Response(403, json={"error": {"message": "flagged by moderation"}})
        return None

    network(SyntheticOpenRouter(chat_override=seventh_fails))
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 1
    assert "skipped 1 call (rerun make record to retry it):" in result.output
    assert "  rag-002/v1/1: /api/v1/chat/completions returned HTTP 403" in result.output


# --- refused prompts under repeats, blocks of them, and outages ---------------------------


def refuse_questions(*questions):
    """A chat override: HTTP 403 for system calls asking any of `questions`."""

    def override(request, body):
        if not judge_body(body) and body["messages"][-1]["content"] in questions:
            return httpx.Response(403, json={"error": {"message": "flagged by moderation"}})
        return None

    return override


def test_a_refused_prompt_under_three_repeats_is_tried_once_and_the_rest_recorded(
    tmp_path, network, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    ws = make_workspace(tmp_path, repeats=3)
    # rag-002 is refused in both versions. Plan: 18 system calls, 4 gradings,
    # 4 pairwise questions = 26.
    attempts = []
    refuse = refuse_questions("Do you rent out ladders by the day?")

    def counting(request, body):
        response = refuse(request, body)
        if response is not None:
            attempts.append(body["messages"][0]["content"][:20])
        return response

    router = network(SyntheticOpenRouter(chat_override=counting))
    result = runner.invoke(app, args("record", ws))
    out = result.output
    assert result.exit_code == 1
    assert "stopped:" not in out
    assert len(attempts) == 2  # one try per version; its other repeats are not sent
    assert len(router.chat_bodies) == 26 - 6 - 4  # all but rag-002's 6 and the 4 judge calls
    assert "skipped 2 calls (rerun make record to retry them):" in out
    assert "not sent 4 calls (rerun make record to send them):" in out
    assert (
        "  rag-002/v1/1: an earlier repeat failed "
        "(/api/v1/chat/completions returned HTTP 403: flagged by moderation)"
    ) in out
    assert "4 judge calls wait for answers that were skipped" in out
    assert load_manifest(ws / "cassettes") is None

    router = network(SyntheticOpenRouter())
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 0, result.output
    assert len(router.chat_bodies) == 6 + 4
    assert load_manifest(ws / "cassettes").planned_calls == 26


def test_a_block_of_refused_prompts_after_their_kind_succeeded_does_not_stop(
    tmp_path, network, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    rows = unanswerable_rows(7)
    ws = make_workspace(tmp_path, repeats=1, rag_rows=rows)
    blocked = [row["question"] for row in rows[1:6]]  # rag-002..rag-006, contiguous
    router = network(SyntheticOpenRouter(chat_override=refuse_questions(*blocked)))
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 1
    assert "stopped:" not in result.output
    assert "skipped 10 calls (rerun make record to retry them):" in result.output
    # rag-001 and rag-007 in both versions, triage in both, and their judge calls.
    recorded = {entry.tag.label(entry.repeat) for entry in CassetteStore(ws / "cassettes")}
    assert {"rag-001/v1/0", "rag-007/v1/0", "rag-007/v2/0", "tri-001/v2/0"} <= recorded
    assert len(router.chat_bodies) == len(recorded)


def test_three_failures_of_a_kind_that_never_succeeded_stop_the_run(
    tmp_path, network, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    ws = make_workspace(tmp_path, repeats=1, rag_rows=unanswerable_rows(6))

    def v1_broken(request, body):  # every rag v1 call fails, as with a broken prompt
        system = body["messages"][0]["content"]
        question = body["messages"][-1]["content"]
        rag_v1 = question.startswith("Synthetic question") and "Follow these rules" not in system
        if rag_v1 and not judge_body(body):
            return httpx.Response(400, json={"error": {"message": "bad request"}})
        return None

    network(SyntheticOpenRouter(chat_override=v1_broken))
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 1
    assert (
        "stopped: 3 different requests in a row failed, and none of their kinds has succeeded "
        "in this run: check the model id and config"
    ) in result.output
    # The day-1 probe of rag v1 failed; rag v2 and triage succeeded in between;
    # then rag-002, rag-003 and rag-004 v1 failed in a row.
    assert "skipped 4 calls" in result.output


def outage_after(answered, failure):
    """A transport handler: the synthetic API answers `answered` chat calls,
    then every chat call gets `failure(request)` (a response, or it raises)."""
    router = SyntheticOpenRouter()

    def handler(request):
        if request.url.path.endswith("/chat/completions") and len(router.chat_bodies) >= answered:
            return failure(request)
        return router(request)

    return handler


def connection_refused(request):
    raise httpx.ConnectError("connection refused", request=request)


def bad_gateway(request):
    return httpx.Response(502, json={"error": {"message": "bad gateway"}})


@pytest.mark.parametrize("failure", [connection_refused, bad_gateway], ids=["no response", "5xx"])
def test_an_outage_stops_the_run_after_twenty_different_requests(
    tmp_path, network, monkeypatch, failure
):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    ws = make_workspace(tmp_path, repeats=1, rag_rows=unanswerable_rows(25))
    # The four day-1 probes succeed, so every kind is proven before the outage.
    network(outage_after(4, failure))
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 1
    assert (
        "stopped: 20 different requests in a row failed: the API or network may be down; "
        "rerun later"
    ) in result.output
    assert "skipped 20 calls (rerun make record to retry them):" in result.output


def test_refusals_on_a_proven_kind_never_stop_the_run(tmp_path, network, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    rows = unanswerable_rows(25)
    ws = make_workspace(tmp_path, repeats=1, rag_rows=rows)
    blocked = [row["question"] for row in rows[1:]]  # 24 prompts, both versions
    network(SyntheticOpenRouter(chat_override=refuse_questions(*blocked)))
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 1
    assert "stopped:" not in result.output
    assert "skipped 48 calls (rerun make record to retry them):" in result.output


def test_a_refused_block_on_a_rerun_does_not_stop_the_run(tmp_path, network, monkeypatch):
    # The reviewer's second gap: on a rerun only the 12 refused prompts are left,
    # 24 refusals back to back (both versions), and nothing succeeds in the run.
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    rows = unanswerable_rows(14)
    ws = make_workspace(tmp_path, repeats=1, rag_rows=rows)
    blocked = [row["question"] for row in rows[1:13]]
    network(SyntheticOpenRouter(chat_override=refuse_questions(*blocked)))
    assert runner.invoke(app, args("record", ws)).exit_code == 1
    router = network(SyntheticOpenRouter(chat_override=refuse_questions(*blocked)))
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 1
    assert "stopped:" not in result.output
    assert "skipped 24 calls (rerun make record to retry them):" in result.output
    assert router.chat_bodies == []  # every call left was refused again


def daily_quota(per_day, *, refused=()):
    """A chat override for one day: 403 for `refused` questions, and HTTP 429
    with tomorrow's reset once `per_day` requests (refused ones too) were made."""
    state = {"requests": 0}
    tomorrow = datetime.now(UTC) + timedelta(hours=10)
    reset = {"X-RateLimit-Reset": str(int(tomorrow.timestamp() * 1000))}
    refuse = refuse_questions(*refused)

    def override(request, body):
        state["requests"] += 1
        if state["requests"] > per_day:
            return httpx.Response(429, headers=reset, json={"error": {"message": "daily"}})
        return refuse(request, body)

    return override


def test_a_later_day_records_fresh_calls_after_a_refused_block(tmp_path, network, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    rows = unanswerable_rows(8)
    ws = make_workspace(tmp_path, repeats=1, rag_rows=rows)
    blocked = [row["question"] for row in rows[1:6]]  # rag-002..rag-006
    # Day 1: the probes, the five refusals in v1, rag-007/v1, then the quota.
    network(SyntheticOpenRouter(chat_override=daily_quota(10, refused=blocked)))
    day1 = runner.invoke(app, args("record", ws))
    assert day1.exit_code == EXIT_STOPPED and "free daily quota reached" in day1.output
    # Day 2: the kinds are proven by yesterday's recordings, so the refusals
    # do not read as a broken config, and fresh calls are recorded.
    router = network(SyntheticOpenRouter(chat_override=daily_quota(10, refused=blocked)))
    day2 = runner.invoke(app, args("record", ws))
    assert day2.exit_code == EXIT_STOPPED
    assert "stopped:" not in day2.output  # before: "3 different requests ... check the config"
    assert "free daily quota reached" in day2.output
    assert len(router.chat_bodies) >= 1  # a fresh call was recorded


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (401, "stopped: the API key was not accepted (/api/v1/chat/completions returned HTTP 401"),
        (402, "stopped: the account has no credit for this request (/api/v1/chat/completions "),
    ],
)
def test_401_and_402_stop_at_once(ws, network, status, message):
    attempts = []

    def refused(request, body):
        attempts.append(1)
        detail = f"rejected {request.headers['authorization']}"
        return httpx.Response(status, json={"error": {"message": detail}})

    network(SyntheticOpenRouter(chat_override=refused))
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 1
    assert message in result.output
    assert len(attempts) == 1
    assert FAKE_KEY not in result.output


def test_a_changed_judge_model_is_unproven_and_stops_after_three(ws, network):
    network(SyntheticOpenRouter())
    assert runner.invoke(app, args("record", ws)).exit_code == 0
    config = (ws / "config.yaml").read_text(encoding="utf-8")
    (ws / "config.yaml").write_text(
        config.replace("synthetic/judge:free", "synthetic/judge-renamed:free"), encoding="utf-8"
    )

    def unknown_model(request, body):
        if body["model"] == "synthetic/judge-renamed:free":
            return httpx.Response(404, json={"error": {"message": "model not found"}})
        return None

    network(SyntheticOpenRouter(chat_override=unknown_model))
    result = runner.invoke(app, args("record", ws))
    # The system calls are recorded, but no key of the renamed judge's kinds is.
    assert result.exit_code == 1
    assert (
        "stopped: 3 different requests in a row failed, and none of their kinds has succeeded "
        "in this run: check the model id and config"
    ) in result.output


def test_the_skip_summary_is_printed_when_a_stop_ends_the_run(ws, network):
    calls = {"n": 0}
    tomorrow = datetime.now(UTC) + timedelta(hours=10)
    reset = {"X-RateLimit-Reset": str(int(tomorrow.timestamp() * 1000))}

    def refuse_then_quota(request, body):
        calls["n"] += 1
        if calls["n"] == 6:  # rag-002/v1/0: refused, so rag-002/v1/1 is not sent
            return httpx.Response(403, json={"error": {"message": "flagged by moderation"}})
        if calls["n"] == 9:
            return httpx.Response(429, headers=reset, json={})
        return None

    network(SyntheticOpenRouter(chat_override=refuse_then_quota))
    result = runner.invoke(app, args("record", ws))
    out = result.output
    assert result.exit_code == EXIT_STOPPED
    assert "free daily quota reached" in out
    assert "skipped 1 call (rerun make record to retry it):" in out
    assert "not sent 1 call (rerun make record to send it):" in out
    assert out.index("free daily quota reached") < out.index("skipped 1 call")
    assert out.rstrip().endswith("kept and skipped")


def test_without_fcntl_record_says_it_needs_posix_and_other_commands_work(
    ws, network, monkeypatch
):
    import sys

    router = network(SyntheticOpenRouter())
    monkeypatch.setitem(sys.modules, "fcntl", None)  # as on a platform without it
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 1
    assert (
        "recording needs a POSIX system (Linux, macOS, or the Docker image) "
        "to lock the cassettes directory"
    ) in result.output
    assert router.chat_bodies == []
    assert runner.invoke(app, args("status", ws)).exit_code == 0


def test_the_cli_imports_without_fcntl():
    import subprocess
    import sys

    code = "import sys; sys.modules['fcntl'] = None; import llmeval.cli, llmeval.recording"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
