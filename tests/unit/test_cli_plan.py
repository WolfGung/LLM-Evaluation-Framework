"""`llmeval estimate` and `llmeval status`: the plan, the free quota and the cost, without a key.

Synthetic data: the config, the datasets, the manifest, prices and quota
numbers are made up for the test (see `synthetic_openrouter`). Every request
goes to an httpx MockTransport; workspaces live in `tmp_path`, and nothing is
written to the repository's `cassettes/` or `results/`.
"""

import json

import httpx
import pytest
from typer.testing import CliRunner

from llmeval import cli
from llmeval.cassettes import MANIFEST_FILE
from llmeval.checks.judge import RUBRIC_PATH
from llmeval.cli import app
from tests.unit.synthetic_judge import ROOT
from tests.unit.synthetic_openrouter import (
    FAKE_KEY,
    SyntheticOpenRouter,
    everything_under,
    make_workspace,
)

runner = CliRunner()
RUBRIC = ROOT / RUBRIC_PATH


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("OPENROUTER_API_KEY", "LLMEVAL_MODE", "MAX_RUN_COST_USD"):
        monkeypatch.delenv(name, raising=False)


def refuse(request):
    raise AssertionError(f"no request expected: {request.url}")


@pytest.fixture
def network(monkeypatch):
    """Route the CLI's API calls to a mock; by default any request fails the test."""

    def use(handler=refuse):
        monkeypatch.setattr(cli, "_network", lambda: cli.Network(httpx.MockTransport(handler)))
        return handler

    use()
    return use


def args(command, ws):
    return [
        command,
        "--config",
        str(ws / "config.yaml"),
        "--datasets-dir",
        str(ws / "datasets"),
        "--cassettes-dir",
        str(ws / "cassettes"),
        "--rubric",
        str(RUBRIC),
    ]


# --- estimate ---------------------------------------------------------------------


def test_estimate_prints_the_plan_the_quota_days_and_a_free_cost(tmp_path, network):
    ws = make_workspace(tmp_path)
    result = runner.invoke(app, args("estimate", ws))
    assert result.exit_code == 0, result.output
    out = result.output
    assert "call plan: repeats 2, judge_repeats first, stability_cases: every case" in out
    # rag: 2 cases x 2 repeats; the judge grades both cases on repeat 0.
    assert (
        "  rag v1: 4 system calls (repeat 0: 2, repeat 1: 2) "
        "+ up to 2 judge calls (repeat 0: 2) = up to 6; recorded 0"
    ) in out
    assert "  triage v2: 2 system calls (repeat 0: 1, repeat 1: 1) = 2; recorded 0" in out
    assert "  rag v1 vs v2: up to 4 pairwise judge calls (repeat 0: 4) = up to 4; recorded 0" in out
    assert (
        "total: 12 system calls + up to 8 judge calls = up to 20 distinct calls; "
        "recorded 0, to record up to 20"
    ) in out
    assert "free-model limits (OpenRouter limits documentation, checked 2026-10-04" in out
    assert "free-model calls to record: up to 20: 1 day at 50 a day, 1 day at 1000 a day" in out
    assert "estimated cost of the calls still to record: $0.00" in out
    assert "spend limit MAX_RUN_COST_USD: $1.00; the estimate is within it" in out


def test_estimate_writes_nothing_and_needs_no_key(tmp_path, network):
    ws = make_workspace(tmp_path)
    before = sorted(p.name for p in ws.rglob("*"))
    result = runner.invoke(app, args("estimate", ws))
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in ws.rglob("*")) == before


def test_estimate_follows_the_config_levers(tmp_path, network):
    ws = make_workspace(tmp_path, repeats=3, judge_repeats="all", stability_cases="[rag-001]")
    result = runner.invoke(app, args("estimate", ws))
    assert result.exit_code == 0, result.output
    # rag-001 three times, rag-002 once; every repeat graded.
    assert (
        "  rag v1: 4 system calls (repeat 0: 2, repeat 1: 1, repeat 2: 1) "
        "+ up to 4 judge calls (repeat 0: 2, repeat 1: 1, repeat 2: 1) = up to 8; recorded 0"
    ) in result.output
    assert "stability_cases: rag-001" in result.output


def test_estimate_refuses_a_subset_with_unknown_cases(tmp_path, network):
    ws = make_workspace(tmp_path, stability_cases="[rag-001, rag-404]")
    result = runner.invoke(app, args("estimate", ws))
    assert result.exit_code == 1
    assert "stability_cases names cases that are not in the datasets: rag-404" in result.output


def paid(ws):
    config = (ws / "config.yaml").read_text(encoding="utf-8")
    config = config.replace("synthetic/system:free", "synthetic/system-paid")
    (ws / "config.yaml").write_text(
        config.replace("synthetic/judge:free", "synthetic/judge-paid"), encoding="utf-8"
    )


def test_estimate_prices_paid_models_and_refuses_above_the_limit(tmp_path, network, monkeypatch):
    ws = make_workspace(tmp_path)
    paid(ws)
    network(
        SyntheticOpenRouter(
            prices={
                "synthetic/system-paid": ("0.001", "0.001"),
                "synthetic/judge-paid": ("0.001", "0.001"),
            }
        )
    )
    monkeypatch.setenv("MAX_RUN_COST_USD", "0.50")
    result = runner.invoke(app, args("estimate", ws))
    assert result.exit_code == 1
    assert "from the published prices (GET /api/v1/models)" in result.output
    assert "refused: estimated cost $" in result.output
    assert "exceeds MAX_RUN_COST_USD=$0.50; nothing was sent" in result.output


def test_estimate_without_a_price_for_a_paid_model_fails(tmp_path, network):
    ws = make_workspace(tmp_path)
    paid(ws)
    network(SyntheticOpenRouter(prices={}))
    result = runner.invoke(app, args("estimate", ws))
    assert result.exit_code == 1
    assert "no published price for: synthetic/" in result.output


# --- status -----------------------------------------------------------------------


def test_status_without_a_key_still_prints_planned_and_recorded(tmp_path, network):
    ws = make_workspace(tmp_path)
    result = runner.invoke(app, args("status", ws))
    assert result.exit_code == 0, result.output
    out = result.output
    assert "manifest: absent, so the evaluation is pending first recorded run" in out
    assert "total: 12 system calls + up to 8 judge calls" in out
    assert (
        "free quota today: OPENROUTER_API_KEY is not set, so the remaining free requests "
        "are not read"
    ) in out
    assert "free-model limits (OpenRouter limits documentation, checked 2026-10-04" in out


def test_status_with_a_key_reads_the_live_quota_and_never_prints_the_key(
    tmp_path, network, monkeypatch
):
    ws = make_workspace(tmp_path)
    router = network(SyntheticOpenRouter(remaining=30))
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    result = runner.invoke(app, args("status", ws))
    assert result.exit_code == 0, result.output
    assert (
        "free quota today (GET /api/v1/key, read live): used 20, limit 50, remaining 30"
    ) in result.output
    assert router.paths == ["/api/v1/key"]
    assert FAKE_KEY not in result.output


def test_status_scrubs_the_key_from_a_failed_quota_read(tmp_path, network, monkeypatch):
    ws = make_workspace(tmp_path)

    def echo_key(request):
        message = f"invalid key {request.headers['authorization']}"
        return httpx.Response(401, json={"error": {"message": message}})

    network(echo_key)
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    result = runner.invoke(app, args("status", ws))
    assert result.exit_code == 0, result.output
    assert "free quota today: not read (/api/v1/key returned HTTP 401" in result.output
    assert "[redacted]" in result.output
    assert FAKE_KEY not in result.output


def test_status_reports_a_manifest(tmp_path, network):
    ws = make_workspace(tmp_path)
    manifest = {
        "models": {"system": "synthetic/system:free", "judge": "synthetic/judge:free"},
        "prompt_versions": {"rag": ["v1"]},
        "repeats": 1,
        "datasets": {"rag.jsonl": "0" * 64},
        "recorded_from": "2026-01-01T10:00:00Z",
        "recorded_to": "2026-01-02T18:30:00Z",
        "planned_calls": 7,
        "recorded_calls": 7,
        "judge_repeats": "first",
    }
    (ws / "cassettes" / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    result = runner.invoke(app, args("status", ws))
    assert result.exit_code == 0, result.output
    assert (
        "manifest: present: a complete recording of 7 calls, "
        "2026-01-01 10:00 to 2026-01-02 18:30 UTC"
    ) in result.output
    # The current config plans more than that recording covers.
    assert "the current plan is not fully recorded; make eval replays the manifest's plan" in (
        result.output
    )


def test_status_fails_on_a_broken_manifest(tmp_path, network):
    ws = make_workspace(tmp_path)
    (ws / "cassettes" / MANIFEST_FILE).write_text("{broken", encoding="utf-8")
    result = runner.invoke(app, args("status", ws))
    assert result.exit_code == 1
    assert "manifest.json: not a valid run manifest" in result.output


def test_no_command_writes_the_key_anywhere(tmp_path, network, monkeypatch):
    ws = make_workspace(tmp_path)
    network(SyntheticOpenRouter())
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    outputs = [runner.invoke(app, args(command, ws)).output for command in ("estimate", "status")]
    assert all(FAKE_KEY not in output for output in outputs)
    assert FAKE_KEY not in everything_under(ws)
