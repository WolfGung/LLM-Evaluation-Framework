"""Superseded recordings: a changed judge config, status, `llmeval prune` and the record hint.

Synthetic data: the config, the datasets, every reply, quota number and
cassette entry here are made up for the test (see `synthetic_openrouter`).
Every cassette is written by the repository's own `record` command against an
httpx MockTransport, never by hand and never from the network. Workspaces
live in `tmp_path`; nothing touches the repository's `cassettes/` or
`results/`.
"""

import httpx
import pytest
from typer.testing import CliRunner

from llmeval import cli
from llmeval.checks.judge import RUBRIC_PATH
from llmeval.cli import app
from tests.unit.synthetic_judge import ROOT
from tests.unit.synthetic_openrouter import FAKE_KEY, NoWait, SyntheticOpenRouter, make_workspace

runner = CliRunner()
RUBRIC = ROOT / RUBRIC_PATH
# The synthetic workspace (see test_recording): 12 system calls, 4 gradings
# and 4 pairwise questions, 20 in all.
SYSTEM_CALLS = 12
JUDGE_CALLS = 8
ALL_CALLS = SYSTEM_CALLS + JUDGE_CALLS


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
    """Send the CLI's API calls to a synthetic OpenRouter, with no waits."""

    def use(router):
        net = cli.Network(httpx.MockTransport(router), limiter=NoWait(), sleep=lambda _: None)
        monkeypatch.setattr(cli, "_network", lambda: net)
        return router

    return use


def args(command, ws, *extra):
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
    return [command, *paths, *extra]


def set_judge_budget(ws, max_tokens):
    """Change only the judge's max_tokens in the workspace config."""
    path = ws / "config.yaml"
    text = path.read_text(encoding="utf-8")
    judge = text.index("judge:")
    old = "  max_tokens: 200\n"
    at = text.index(old, judge)
    path.write_text(
        text[:at] + f"  max_tokens: {max_tokens}\n" + text[at + len(old) :], encoding="utf-8"
    )


def recorded(ws, network):
    """A complete synthetic recording made by `record`."""
    network(SyntheticOpenRouter())
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 0, result.output
    return result


# --- a judge budget change, and what status and eval say about the old manifest ----------


def test_a_judge_budget_change_leaves_the_manifest_stale_and_status_says_so(ws, network):
    recorded(ws, network)
    set_judge_budget(ws, 400)
    result = runner.invoke(app, args("status", ws))
    assert result.exit_code == 0, result.output
    out = result.output
    stale = f"manifest: present but stale: written for a complete recording of {ALL_CALLS} calls"
    assert stale in out
    assert (
        f"  make eval would replay {ALL_CALLS} calls, and {JUDGE_CALLS} of them are not in the "
        "cassettes: it fails until make record completes the current plan and rewrites the "
        "manifest"
    ) in out
    assert "make eval replays the manifest's plan" not in out
    # The system answers stay planned; the judge calls are planned again from them.
    assert (
        f"total: {SYSTEM_CALLS} system calls + {JUDGE_CALLS} judge calls = {ALL_CALLS} distinct "
        f"calls; recorded {SYSTEM_CALLS}, to record {JUDGE_CALLS}"
    ) in out
    replay = runner.invoke(app, args("eval", ws))
    assert replay.exit_code == 1
    assert "no recording for rag-001:judge/v1/0: run make record" in replay.output
    assert not (ws / "results").exists()


def test_a_manifest_whose_plan_is_recorded_stays_replayable_when_the_config_plans_more(
    ws, network
):
    recorded(ws, network)
    config = (ws / "config.yaml").read_text(encoding="utf-8")
    (ws / "config.yaml").write_text(config.replace("repeats: 2", "repeats: 3"), encoding="utf-8")
    out = runner.invoke(app, args("status", ws)).output
    assert f"manifest: present: a complete recording of {ALL_CALLS} calls" in out
    assert "the current plan is not fully recorded; make eval replays the manifest's plan" in out
    assert "stale" not in out
    assert runner.invoke(app, args("eval", ws)).exit_code == 0


def test_status_says_when_make_eval_replays_fewer_calls_than_the_manifest_counts(ws, network):
    recorded(ws, network)
    path = ws / "datasets" / "rag.jsonl"
    path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    out = runner.invoke(app, args("status", ws)).output
    # rag-001 alone: 4 rag and 4 triage calls, 2 gradings and 2 pairwise questions.
    assert f"manifest: present: a complete recording of {ALL_CALLS} calls" in out
    assert f"  the manifest counts {ALL_CALLS} calls; make eval now replays 12, all recorded" in out
