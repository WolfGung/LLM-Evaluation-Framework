"""Repository files that keep the evaluation honest: ignores, make targets, CI steps."""

from pathlib import Path

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


def test_make_has_the_recording_targets():
    targets, phony = make_targets()
    for name in ("estimate", "status", "record"):
        assert targets[name] == [f"$(BIN)/llmeval {name}"]
    assert targets["prune"] == ["$(BIN)/llmeval prune --yes"]
    assert {"test", "eval", "estimate", "status", "record", "prune", "lint"} <= phony


def test_make_baseline_writes_the_baseline_from_the_committed_results():
    targets, phony = make_targets()
    assert targets["baseline"] == ["$(BIN)/llmeval baseline"]
    assert "baseline" in phony


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


def test_a_half_written_manifest_and_the_record_lock_are_never_committed():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "cassettes/.manifest.json.partial" in ignored
    assert "cassettes/.record.lock" in ignored
    # A cassette file prune is rewriting (renamed into place when written).
    assert "cassettes/.*.jsonl.partial" in ignored
