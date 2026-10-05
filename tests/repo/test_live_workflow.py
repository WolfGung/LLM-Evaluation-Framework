""".github/workflows/live.yml: a manual live run that never fails for a missing key.

The owner's ruling: no scheduled runs, so the workflow starts only by hand.
With the OPENROUTER_API_KEY secret it runs every case again against the API
into results-live/, gates the live results against the committed baseline
and uploads them with the gate's report as an artifact. It commits nothing.
Without the secret, a notice says so and the live job is skipped.

The key check's shell script is run here with and without a key (a made-up
value), so "skipped with a notice, never failed" is tested, not only read.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "live.yml"
SECRET = "${{ secrets.OPENROUTER_API_KEY }}"


def text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def workflow() -> dict:
    return yaml.safe_load(text())


def triggers(data: dict) -> dict:
    # YAML 1.1 reads the bare key `on` as true.
    return data.get("on", data.get(True))


def steps(job: dict) -> list[dict]:
    return job["steps"]


def step_named(job: dict, words: str) -> dict:
    return next(step for step in steps(job) if words in step.get("name", ""))


def test_live_runs_only_when_started_by_hand():
    assert triggers(workflow()) == {"workflow_dispatch": None}
    assert "schedule:" not in text()


def test_live_can_read_the_repository_and_commits_nothing():
    data = workflow()
    assert data["permissions"] == {"contents": "read"}
    for job in data["jobs"].values():
        assert "permissions" not in job
    assert "git commit" not in text() and "git push" not in text()


def test_the_secret_reaches_only_the_key_check_and_the_live_run():
    jobs = workflow()["jobs"]
    users = [
        (name, step.get("name"))
        for name, job in jobs.items()
        for step in steps(job)
        if SECRET in str(step)
    ]
    assert users == [("key", "Check for the API key"), ("live", "Live run")]
    assert all(SECRET not in str(job.get("env", {})) for job in jobs.values())
    assert text().count("secrets.") == 2


def test_the_live_job_is_skipped_without_the_key():
    jobs = workflow()["jobs"]
    assert jobs["key"]["outputs"] == {"present": "${{ steps.check.outputs.present }}"}
    assert jobs["live"]["needs"] == "key"
    assert jobs["live"]["if"] == "needs.key.outputs.present == 'true'"


def test_the_live_job_runs_live_gates_and_uploads_the_drift_report():
    live = workflow()["jobs"]["live"]
    run = step_named(live, "Live run")["run"]
    assert "llmeval live --results-dir results-live" in run
    gate = step_named(live, "Gate")
    assert gate["shell"] == "bash"  # pipefail: a regression fails the step through tee
    assert "llmeval gate --results-dir results-live" in gate["run"]
    assert "results-live/drift-report.txt" in gate["run"]
    upload = step_named(live, "Upload")
    assert upload["if"] == "always()"
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert upload["with"]["path"] == "results-live"


def key_check_script() -> str:
    return next(step for step in steps(workflow()["jobs"]["key"]) if step.get("id") == "check")[
        "run"
    ]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash, as the runner has")
@pytest.mark.parametrize(
    ("key", "present", "notice"),
    [("", "false", True), ("synthetic-not-a-key", "true", False)],
    ids=["no-secret", "secret"],
)
def test_the_key_check_never_fails_and_says_why_it_skips(tmp_path, key, present, notice):
    output = tmp_path / "github_output"
    env = {
        "PATH": os.environ["PATH"],
        "GITHUB_OUTPUT": str(output),
        "OPENROUTER_API_KEY": key,
    }
    done = subprocess.run(
        ["bash", "-e", "-c", key_check_script()],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert output.read_text(encoding="utf-8").splitlines() == [f"present={present}"]
    assert ("::notice" in done.stdout) is notice
    assert "synthetic-not-a-key" not in done.stdout + done.stderr
