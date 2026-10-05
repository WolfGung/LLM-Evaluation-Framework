"""Superseded recordings: a changed judge config, status, `llmeval prune` and the record hint.

Synthetic data: the config, the datasets, every reply, quota number and
cassette entry here are made up for the test (see `synthetic_openrouter`).
Every cassette entry is written by the repository's own `record` command
against an httpx MockTransport, never from the network; a few tests damage a
file on purpose (a broken or an unfinished line), and the label test builds
one synthetic entry in memory. Workspaces live in `tmp_path`; nothing touches
the repository's `cassettes/` or `results/`.
"""

import fcntl
import json

import httpx
import pytest
from typer.testing import CliRunner

from llmeval import cli
from llmeval.cassettes import MANIFEST_FILE, CassetteStore, load_manifest
from llmeval.checks.judge import RUBRIC_PATH
from llmeval.cli import app
from llmeval.recording import EXIT_STOPPED, LOCK_FILE, unplanned_hint
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


# --- llmeval prune ----------------------------------------------------------------------


def snapshot(ws):
    """Every file in the cassettes directory, by name, with its bytes."""
    return {path.name: path.read_bytes() for path in sorted((ws / "cassettes").iterdir())}


JUDGE_FILES = ("judge-v1.jsonl", "judge-v2.jsonl", "pairwise-v1-v2.jsonl")


def test_prune_with_nothing_unplanned_says_so_and_changes_nothing(ws, network):
    recorded(ws, network)
    before = snapshot(ws)
    result = runner.invoke(app, args("prune", ws, "--yes"))
    assert result.exit_code == 0, result.output
    assert "every recorded entry is in the current plan: nothing to prune" in result.output
    assert snapshot(ws) == before


def test_a_dry_run_lists_the_unplanned_entries_and_changes_nothing(ws, network):
    recorded(ws, network)
    set_judge_budget(ws, 400)
    (ws / "cassettes" / LOCK_FILE).unlink()  # left by record; a dry run must not create it
    before = snapshot(ws)
    result = runner.invoke(app, args("prune", ws))
    assert result.exit_code == 0, result.output
    out = result.output
    assert f"current plan: {ALL_CALLS} distinct calls, {SYSTEM_CALLS} of them recorded" in out
    assert (
        f"{JUDGE_CALLS} recorded entries are not in the current plan: "
        "judge-v1.jsonl 2, judge-v2.jsonl 2, pairwise-v1-v2.jsonl 4"
    ) in out
    # One line per entry: file, call (case/version/repeat) and when it was recorded.
    listed = [line for line in out.splitlines() if line.startswith("  ") and "recorded 20" in line]
    assert len(listed) == JUDGE_CALLS
    grading = "  judge-v1.jsonl: rag-001:judge/v1/0, recorded "
    assert any(line.startswith(grading) for line in listed)
    assert any(line.startswith("  pairwise-v1-v2.jsonl: rag-002:A=v2/v1-v2/0, ") for line in listed)
    assert all(line.endswith(" UTC") for line in listed)
    assert "dry run: nothing was changed; make prune (llmeval prune --yes) removes them" in out
    assert snapshot(ws) == before


def test_yes_removes_exactly_the_unplanned_entries_and_keeps_the_rest_byte_for_byte(ws, network):
    recorded(ws, network)
    # Only rag-001 repeats now: the repeat-1 runs of rag-002 and tri-001 leave the plan.
    config = (ws / "config.yaml").read_text(encoding="utf-8")
    config = config.replace("stability_cases: null", "stability_cases: [rag-001]")
    (ws / "config.yaml").write_text(config, encoding="utf-8")
    before = snapshot(ws)
    store = CassetteStore(ws / "cassettes")
    gone = {
        entry.key
        for entry in store
        if entry.repeat == 1 and entry.tag.case in ("rag-002", "tri-001")
    }
    assert len(gone) == 4
    result = runner.invoke(app, args("prune", ws, "--yes"))
    assert result.exit_code == 0, result.output
    for name in ("rag-v1.jsonl", "rag-v2.jsonl", "triage-v1.jsonl", "triage-v2.jsonl"):
        assert f"removed 1 entry from {name}" in result.output
    assert "removed 4 entries in all" in result.output
    after = snapshot(ws)
    assert set(after) == set(before)
    for name, data in before.items():
        if not name.endswith(".jsonl"):
            assert after[name] == data, name
            continue
        lines = data.splitlines(keepends=True)
        kept = [line for line in lines if not line.strip() or json.loads(line)["key"] not in gone]
        assert after[name] == b"".join(kept), name
    assert after[MANIFEST_FILE] == before[MANIFEST_FILE]  # prune never edits the manifest
    status = runner.invoke(app, args("status", ws)).output
    assert "outside the current plan" not in status


def test_a_judge_budget_change_prunes_the_old_judge_entries_and_keeps_the_system_answers(
    ws, network
):
    recorded(ws, network)
    set_judge_budget(ws, 400)
    status = runner.invoke(app, args("status", ws)).output
    assert (
        f"cassette entries outside the current plan: {JUDGE_CALLS} "
        "(llmeval prune lists them; make prune removes them)"
    ) in status
    before = snapshot(ws)
    result = runner.invoke(app, args("prune", ws, "--yes"))
    assert result.exit_code == 0, result.output
    out = result.output
    assert "removed 2 entries from judge-v1.jsonl and deleted the file: no entries left" in out
    assert "removed 4 entries from pairwise-v1-v2.jsonl and deleted the file" in out
    assert f"removed {JUDGE_CALLS} entries in all" in out
    assert "manifest.json is unchanged: prune never edits it" in out
    after = snapshot(ws)
    # Every emptied file is removed; the system answers and the manifest stay byte for byte.
    assert set(before) - set(after) == set(JUDGE_FILES)
    assert all(after[name] == data for name, data in before.items() if name in after)
    status = runner.invoke(app, args("status", ws)).output
    assert "manifest: present but stale" in status
    assert f"recorded {SYSTEM_CALLS}, to record {JUDGE_CALLS}" in status
    # The next record run sends only the judge calls, with the new budget, and
    # rewrites the manifest.
    router = network(SyntheticOpenRouter())
    again = runner.invoke(app, args("record", ws))
    assert again.exit_code == 0, again.output
    assert len(router.chat_bodies) == JUDGE_CALLS
    assert {body["max_tokens"] for body in router.chat_bodies} == {400}
    assert load_manifest(ws / "cassettes").planned_calls == ALL_CALLS
    assert "not in the current plan" not in again.output
    status = runner.invoke(app, args("status", ws)).output
    assert f"manifest: present: a complete recording of {ALL_CALLS} calls" in status


def test_prune_refuses_while_a_record_run_holds_the_lock(ws, network):
    recorded(ws, network)
    set_judge_budget(ws, 400)
    before = snapshot(ws)
    with (ws / "cassettes" / LOCK_FILE).open("a") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        applied = runner.invoke(app, args("prune", ws, "--yes"))
        dry = runner.invoke(app, args("prune", ws))
    for result in (applied, dry):
        assert result.exit_code == 1
        assert "another record run is active" in result.output
        assert "not in the current plan" not in result.output
    assert snapshot(ws) == before
    # Released: prune goes ahead.
    assert runner.invoke(app, args("prune", ws, "--yes")).exit_code == 0


def test_without_fcntl_prune_says_it_needs_posix(ws, network, monkeypatch):
    import sys

    recorded(ws, network)
    monkeypatch.setitem(sys.modules, "fcntl", None)  # as on a platform without it
    result = runner.invoke(app, args("prune", ws))
    assert result.exit_code == 1
    assert "pruning needs a POSIX system (Linux, macOS, or the Docker image)" in result.output


def test_an_entry_without_a_tag_is_listed_by_its_key():
    from datetime import UTC, datetime

    from llmeval.cassettes import CassetteEntry
    from llmeval.prune import _label

    moment = datetime(2026, 1, 1, tzinfo=UTC)
    entry = CassetteEntry(
        key="ab" * 32,
        repeat=1,
        request={"model": "synthetic/system:free", "messages": []},
        response={"model": "synthetic/system:free", "content": "Synthetic."},
        usage={"prompt_tokens": 1, "completion_tokens": 1},
        cost_usd=0.0,
        cost_source="provider",
        latency_ms=1.0,
        requested_at=moment,
        recorded_at=moment,
    )
    assert _label(entry) == "key abababababab/1"


def break_config(ws):
    path = ws / "config.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("rpm: 18", "rpm: 0"), "utf-8")


def drop_dataset(ws):
    (ws / "datasets" / "triage.jsonl").unlink()


def break_cassette(ws):
    with (ws / "cassettes" / "rag-v1.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (break_config, "is invalid"),
        (drop_dataset, "triage.jsonl"),
        (break_cassette, "rag-v1.jsonl:5: not a valid entry"),
    ],
)
def test_prune_refuses_when_the_plan_cannot_be_computed(ws, network, damage, message):
    recorded(ws, network)
    set_judge_budget(ws, 400)
    damage(ws)
    before = snapshot(ws)
    result = runner.invoke(app, args("prune", ws, "--yes"))
    assert result.exit_code == 1
    assert message in result.output
    assert "removed" not in result.output
    assert snapshot(ws) == before


def test_prune_refuses_while_judge_calls_wait_for_their_answers(ws, network):
    # Stopped on day one: some answers are not recorded, so some judge keys are unknown.
    network(SyntheticOpenRouter(remaining=6))
    assert runner.invoke(app, args("record", ws)).exit_code == EXIT_STOPPED
    before = snapshot(ws)
    result = runner.invoke(app, args("prune", ws, "--yes"))
    assert result.exit_code == 1
    assert "judge calls wait for answers that are not recorded, so their keys are unknown" in (
        result.output
    )
    assert "run make record until every answer is recorded, then prune" in result.output
    assert snapshot(ws) == before


def test_a_torn_last_line_follows_the_store_rule(ws, network):
    recorded(ws, network)
    set_judge_budget(ws, 400)
    path = ws / "cassettes" / "rag-v1.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"key": "unfinished')
    torn = path.read_bytes()
    result = runner.invoke(app, args("prune", ws, "--yes"))
    assert result.exit_code == 0, result.output
    assert "notice: ignored an unfinished last line in rag-v1.jsonl" in result.output
    assert f"removed {JUDGE_CALLS} entries in all" in result.output
    assert path.read_bytes() == torn  # not pruned, not repaired: record cuts it off


# --- the hint after a complete record run ------------------------------------------------


def test_a_complete_record_names_the_entries_outside_the_plan(ws, network):
    recorded(ws, network)
    set_judge_budget(ws, 400)
    router = network(SyntheticOpenRouter())
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == 0, result.output
    assert len(router.chat_bodies) == JUDGE_CALLS
    lines = result.output.splitlines()
    assert lines[-1] == (
        f"{JUDGE_CALLS} recorded entries are not in the current plan: "
        "run make prune to remove them"
    )
    assert lines[-2].startswith(f"every planned call is recorded ({ALL_CALLS}); wrote ")
    status = runner.invoke(app, args("status", ws)).output
    assert f"manifest: present: a complete recording of {ALL_CALLS} calls" in status


def test_the_hint_counts_one_entry_in_the_singular():
    assert unplanned_hint(1) == (
        "1 recorded entry is not in the current plan: run make prune to remove it"
    )
    assert unplanned_hint(2) == (
        "2 recorded entries are not in the current plan: run make prune to remove them"
    )


def test_an_incomplete_record_gives_no_prune_hint(ws, network):
    recorded(ws, network)
    set_judge_budget(ws, 400)
    network(SyntheticOpenRouter(remaining=3))
    result = runner.invoke(app, args("record", ws))
    assert result.exit_code == EXIT_STOPPED
    assert "make prune" not in result.output
