"""The Allure report of the recorded run: what the published report will show.

Runs the replay evaluation for one RAG case and one triage case, with the
pairwise comparison and the layer results, into `--alluredir` in a
temporary directory, and reads the JSON files allure-pytest writes there
(the Allure command line is not needed). The report must agree with the
committed results: each layer's rate in its title, and the pairwise outcome
in its tag. Nothing key-like may appear in it.

Without a manifest there is no recorded run, and this is skipped. No model
is called and nothing in the repository is written.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from llmeval.baseline import load_baseline
from llmeval.cassettes import PENDING_RECORDED_RUN, load_manifest
from llmeval.results import FunctionResults, PairwiseResults
from tests.allure_files import (
    KEY_LIKE,
    attachments,
    categories_of,
    labels,
    parameters,
    results_of,
    written_text,
)
from tools.render import percent

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def first_known_failure() -> str:
    """The first RAG v1 case the baseline lists as a known failure."""
    baseline = load_baseline(RESULTS / "baseline.json")
    assert baseline is not None, "results/baseline.json is missing: run make baseline"
    cases = baseline.functions["rag"]["v1"].cases
    return next(case_id for case_id, case in cases.items() if not case.passed)


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> Path:
    if load_manifest(ROOT / "cassettes") is None:
        pytest.skip(PENDING_RECORDED_RUN)
    out_dir = tmp_path_factory.mktemp("allure") / "allure-results"
    selection = f"rag-001 or tri-001 or {first_known_failure()} or layer_pass_rate"
    env = {**os.environ}
    for name in ("OPENROUTER_API_KEY", "LLMEVAL_MODE", "LLMEVAL_CASSETTES_DIR", "LLMEVAL_BASELINE"):
        env.pop(name, None)
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-W",
            "error",
            "-p",
            "no:cacheprovider",
            "-q",
            "tests/eval",
            "-k",
            selection,
            f"--alluredir={out_dir}",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return out_dir


def named(out_dir: Path, prefix: str) -> dict:
    (result,) = [r for r in results_of(out_dir) if r["name"].startswith(prefix)]
    return result


def committed(name: str):
    path = RESULTS / f"{name}.json"
    model = PairwiseResults if "-vs-" in name else FunctionResults
    return model.model_validate_json(path.read_bytes())


def test_a_rag_case_shows_its_layers_answers_and_verdict(report):
    result = named(report, "rag-001 v1: ")
    assert labels(result, "epic") == ["rag"]
    assert labels(result, "story") == ["answerable"]
    assert labels(result, "feature")
    assert parameters(result) == {"case": "'rag-001'", "version": "'v1'"}
    assert result["description"].startswith("One document (kb-returns) answers this question.")
    shown = attachments(report, result)
    assert {"question", "retrieved documents", "judge verdict", "record"} <= shown.keys()
    assert [name for name in shown if name.startswith("answer, repeat")] == [
        "answer, repeat 0",
        "answer, repeat 1",
        "answer, repeat 2",
    ]
    record = committed("rag-v1").cases[0]
    assert shown["answer, repeat 0"] == record.runs[0].output
    assert "kb-returns: Returns (expected)" in shown["retrieved documents"].splitlines()
    assert shown["judge verdict"].startswith("repeat 0: groundedness ")


def test_a_triage_case_shows_its_ticket_and_replies(report):
    result = named(report, "tri-001 v2: ")
    assert labels(result, "epic") == ["triage"]
    assert parameters(result) == {"case": "'tri-001'", "version": "'v2'"}
    assert "rule H1" in result["description"]
    shown = attachments(report, result)
    assert {"ticket", "reply, repeat 0", "record"} <= shown.keys()


def test_the_pairwise_outcome_is_the_committed_one(report):
    result = named(report, "rag-001: v1 vs v2")
    assert labels(result, "feature") == ["pairwise comparison"]
    outcome = committed("rag-v1-vs-v2").cases[0].outcome
    assert labels(result, "tag") == [outcome]
    shown = attachments(report, result)
    assert {"order 1: v1 shown as A", "order 2: v2 shown as A"} <= shown.keys()


def test_each_layer_title_shows_the_committed_rate(report):
    titles = {r["name"] for r in results_of(report) if labels(r, "story") == ["pass rate"]}
    expected = set()
    for name in ("rag-v1", "rag-v2", "triage-v1", "triage-v2"):
        result = committed(name)
        for layer, rate in result.summary.layers.items():
            share = percent(rate.passed, rate.total)
            expected.add(
                f"{result.function} {result.version}: {layer} layer, {share} of runs pass "
                f"({rate.passed} of {rate.total})"
            )
    assert titles == expected


def test_a_known_failure_falls_in_its_category(report):
    result = named(report, f"{first_known_failure()} v1: ")
    assert result["status"] == "skipped"
    assert categories_of(report, result) == ["Known failure in the baseline"]


def test_the_environment_names_the_recording(report):
    manifest = load_manifest(ROOT / "cassettes")
    environment = (report / "environment.properties").read_text("utf-8").splitlines()
    assert f"system.model={manifest.models['system']}" in environment
    assert f"judge.model={manifest.models['judge']}" in environment
    assert f"calls={manifest.recorded_calls}" in environment
    assert f"repeats={manifest.repeats}" in environment
    assert json.loads((report / "categories.json").read_text("utf-8"))


def test_nothing_key_like_reaches_the_report(report):
    for result in results_of(report):
        assert not KEY_LIKE.findall(written_text(report, result)), result["name"]
    for path in report.iterdir():
        text = path.read_text("utf-8")
        assert "sk-or-" not in text and "Authorization" not in text, path.name
