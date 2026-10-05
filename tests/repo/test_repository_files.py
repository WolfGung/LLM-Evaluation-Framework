"""Repository files that keep the evaluation honest: ignores, make targets, CI steps."""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_live_results_are_not_committed():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "results-live/" in ignored
    assert "results/" not in ignored  # replay results are committed


def make_targets():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = {}
    current = None
    for line in makefile.splitlines():
        if line and not line[0].isspace() and line.endswith(":"):
            current = line[:-1]
        elif line.startswith("\t") and current:
            targets.setdefault(current, []).append(line.strip())
    phony = next(line for line in makefile.splitlines() if line.startswith(".PHONY:"))
    return targets, set(phony.split(":", 1)[1].split())


def test_make_test_leaves_out_the_evaluation_and_make_eval_runs_it():
    targets, _ = make_targets()
    assert any("--ignore=tests/eval" in step for step in targets["test"])
    assert any("tests/eval" in step and "--ignore" not in step for step in targets["eval"])
    # make eval also writes results/ from the replay (pending without a manifest).
    assert any(step.endswith("llmeval eval") for step in targets["eval"])


def test_make_eval_writes_the_judge_agreement_after_the_replay():
    targets, _ = make_targets()
    # The agreement reads the fresh results, so it comes after the replay.
    assert targets["eval"][:2] == ["$(BIN)/llmeval eval", "$(BIN)/llmeval agreement"]


def test_make_has_the_recording_targets():
    targets, phony = make_targets()
    for name in ("estimate", "status", "record"):
        assert targets[name] == [f"$(BIN)/llmeval {name}"]
    assert targets["prune"] == ["$(BIN)/llmeval prune --yes"]
    assert {"test", "eval", "estimate", "status", "record", "prune", "lint"} <= phony


def test_make_has_the_baseline_and_gate_targets():
    targets, phony = make_targets()
    assert targets["baseline"] == ["$(BIN)/llmeval baseline"]
    assert targets["gate"] == ["$(BIN)/llmeval gate"]
    assert {"baseline", "gate"} <= phony


def test_make_readme_writes_the_results_block():
    targets, phony = make_targets()
    assert targets["readme"] == ["$(BIN)/python -m tools.render --write"]
    assert "readme" in phony


def test_make_prune_says_it_removes_and_how_to_look_first():
    lines = (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
    at = lines.index("prune:")
    comment = []
    for line in reversed(lines[:at]):
        if not line.startswith("#"):
            break
        comment.insert(0, line)
    text = " ".join(comment)
    assert "llmeval prune" in text  # the dry run that only lists
    assert "git" in text  # the cassettes are in git, so a removal can be undone


def test_ci_runs_tests_and_the_evaluation_as_separate_steps():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    test_step = workflow.split("- name: Test", 1)[1].split("- name:", 1)[0]
    eval_step = workflow.split("- name: Eval", 1)[1].split("- name:", 1)[0]
    assert "--ignore=tests/eval" in test_step
    assert "pytest -W error tests/eval" in eval_step
    assert "${{ secrets" not in workflow  # replay only: no key in CI


def test_ci_replays_into_a_temporary_directory_and_runs_the_gate():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    eval_step = workflow.split("- name: Eval", 1)[1].split("- name:", 1)[0]
    # CI never rewrites the committed results/: the replay goes to the runner's temp.
    assert "REPLAY_DIR: ${{ runner.temp }}/" in eval_step
    commands = [line.strip() for line in eval_step.splitlines()]
    start = commands.index("rc=0")
    # A failed replay ends the step: there is nothing to gate or compare.
    replay = commands.index('llmeval eval --results-dir "$REPLAY_DIR"')
    # The judge agreement on the replay, as make eval writes it.
    agreement = commands.index('llmeval agreement --results-dir "$REPLAY_DIR" || rc=1')
    # A failing gate still lets the per-case tests run; the step fails at the end.
    gate = commands.index('llmeval gate --results-dir "$REPLAY_DIR" || rc=1')
    cases = next(
        i
        for i, line in enumerate(commands)
        if line.startswith("pytest -W error tests/eval") and line.endswith("|| rc=1")
    )
    end = commands.index("exit $rc")
    assert start < replay < agreement < gate < cases < end


def test_a_half_written_manifest_and_the_record_lock_are_never_committed():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "cassettes/.manifest.json.partial" in ignored
    assert "cassettes/.record.lock" in ignored
    # A cassette file prune is rewriting (renamed into place when written).
    assert "cassettes/.*.jsonl.partial" in ignored


def test_the_built_page_and_report_are_not_committed():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "site/" in ignored  # python -m tools.site and allure generate write it in CI
    assert "allure-results/" in ignored


def workflow_jobs() -> dict:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8"))
    return workflow["jobs"]


def test_ci_publishes_the_page_and_the_allure_report_from_main_only():
    jobs = workflow_jobs()
    pages = jobs["pages"]
    assert pages["if"] == "github.event_name == 'push' && github.ref == 'refs/heads/main'"
    assert pages["needs"] == "test"  # only a green run is published
    assert pages["permissions"] == {"contents": "read", "pages": "write", "id-token": "write"}
    assert pages["concurrency"] == {"group": "pages", "cancel-in-progress": False}
    assert pages["env"]["ALLURE_VERSION"] == "2.30.0"
    uses = [step.get("uses", "") for step in pages["steps"]]
    assert "actions/download-artifact@v8" in uses
    assert "actions/upload-pages-artifact@v5" in uses
    assert "actions/deploy-pages@v5" in uses
    runs = "\n".join(step.get("run", "") for step in pages["steps"])
    assert "github.com/allure-framework/allure2/releases/download/$ALLURE_VERSION/" in runs
    assert "python -m tools.site" in runs
    assert "allure generate allure-results --clean --output site/report" in runs
    # Every other job stays read-only.
    for name, job in jobs.items():
        if name != "pages":
            assert "permissions" not in job, name


def test_the_label_lock_is_ignored_but_the_owner_labels_are_not():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "labels/.label.lock" in ignored
    # The owner commits labels/human.jsonl: nothing may ignore it.
    done = subprocess.run(
        ["git", "check-ignore", "labels/human.jsonl", "labels/sample.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (done.returncode, done.stdout) == (1, ""), f"git ignores: {done.stdout}"


def test_make_label_runs_the_labelling_tool():
    targets, phony = make_targets()
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
    # bash waits for llmeval on Ctrl-C and runs the || part; dash dies at once.
    assert "label: SHELL := /bin/bash" in makefile
    assert targets["label"] == ["@$(BIN)/llmeval label || [ $$? -eq 130 ]"]
    assert "label" in phony


def make_label(bin_dir: Path, **popen) -> subprocess.Popen:
    return subprocess.Popen(
        ["make", "--no-print-directory", "label", f"BIN={bin_dir}"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **popen,
    )


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
@pytest.mark.parametrize(("code", "make_fails"), [(0, False), (1, True)])
def test_make_label_passes_on_other_exit_codes(tmp_path, code, make_fails):
    # A stand-in for llmeval that exits with `code`: an error still fails make.
    fake = tmp_path / "llmeval"
    fake.write_text(f"#!/bin/sh\nexit {code}\n", encoding="utf-8")
    fake.chmod(0o755)
    process = make_label(tmp_path)
    out, err = process.communicate(timeout=60)
    assert (process.returncode != 0) == make_fails, out + err
    assert ("Error" in err) == make_fails, err


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_a_real_ctrl_c_under_make_label_is_a_clean_stop(tmp_path):
    """Ctrl-C reaches make, the recipe's shell and llmeval together (one
    process group). llmeval says it stopped and exits 130; make must add no
    "Interrupt" or "Error" line after it."""
    ready = tmp_path / "ready"
    fake = tmp_path / "llmeval"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys, time\n"
        f"pathlib.Path({str(ready)!r}).write_text('ready')\n"
        "try:\n"
        "    time.sleep(60)\n"
        "except KeyboardInterrupt:\n"
        "    print('Stopped. Saved 0 labels this time.')\n"
        "    sys.exit(130)\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    process = make_label(tmp_path, start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < deadline, "the stand-in llmeval did not start"
            time.sleep(0.05)
        os.killpg(process.pid, signal.SIGINT)
        out, err = process.communicate(timeout=30)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    assert "Stopped. Saved 0 labels this time." in out
    assert "Interrupt" not in err and "Error" not in err, err
    # make got the Ctrl-C too: it lets the recipe end cleanly, then stops on it.
    assert process.returncode in (0, -signal.SIGINT), out + err
    assert "llmeval label" not in out  # @: make does not echo the command
