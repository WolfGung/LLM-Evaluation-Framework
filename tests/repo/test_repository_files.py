"""Repository files that keep the evaluation honest: ignores, make targets, CI steps."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_live_results_are_not_committed():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "results-live/" in ignored
    assert "results/" not in ignored  # replay results are committed


def test_make_test_leaves_out_the_evaluation_and_make_eval_runs_it():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = {}
    current = None
    for line in makefile.splitlines():
        if line and not line[0].isspace() and line.endswith(":"):
            current = line[:-1]
        elif line.startswith("\t") and current:
            targets.setdefault(current, []).append(line.strip())
    assert any("--ignore=tests/eval" in step for step in targets["test"])
    assert any("tests/eval" in step and "--ignore" not in step for step in targets["eval"])


def test_ci_runs_tests_and_the_evaluation_as_separate_steps():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    test_step = workflow.split("- name: Test", 1)[1].split("- name:", 1)[0]
    eval_step = workflow.split("- name: Eval", 1)[1].split("- name:", 1)[0]
    assert "--ignore=tests/eval" in test_step
    assert "pytest -W error tests/eval" in eval_step
    assert "${{ secrets" not in workflow  # replay only: no key in CI
