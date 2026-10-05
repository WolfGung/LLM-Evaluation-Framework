"""`llmeval sample`, `llmeval label` and `llmeval agreement` on synthetic files.

Synthetic data: the results, judge grades, manifests, answers and labels
below are made up for the test. Every file lives in pytest's `tmp_path`;
nothing is written to the repository's `labels/`, `results/` or `cassettes/`.
"""

import json

import pytest
from typer.testing import CliRunner

from llmeval.cassettes import write_manifest
from llmeval.cli import app
from llmeval.labels import load_sample
from llmeval.results import write_results
from tests.unit import synthetic_results as syn
from tests.unit.synthetic_labels import population

runner = CliRunner()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("OPENROUTER_API_KEY", "LLMEVAL_MODE", "MAX_RUN_COST_USD"):
        monkeypatch.delenv(name, raising=False)


def recorded_run(ws, *, rag=None):
    """A manifest and graded RAG results (v1, v2) in the workspace."""
    rag = rag or [
        population("v1", {"answerable": (10, 1), "unanswerable": (4, 2)}),
        population("v2", {"answerable": (10, 1), "unanswerable": (5, 1)}),
    ]
    write_manifest(ws / "cassettes", syn.manifest({"rag": ("v1", "v2")}))
    for result in rag:
        write_results(result, ws / "results")
    write_results(syn.pairwise_results([syn.pair_case("rag-001", "A", "B")]), ws / "results")


def sample_args(ws):
    return [
        "sample",
        "--results-dir",
        str(ws / "results"),
        "--cassettes-dir",
        str(ws / "cassettes"),
        "--sample",
        str(ws / "labels" / "sample.json"),
    ]


def test_sample_without_a_recorded_run_is_pending(tmp_path):
    (tmp_path / "cassettes").mkdir()
    result = runner.invoke(app, sample_args(tmp_path))
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "pending first recorded run"
    assert not (tmp_path / "labels").exists()


def test_sample_writes_the_label_sample_of_the_results(tmp_path):
    recorded_run(tmp_path)
    result = runner.invoke(app, sample_args(tmp_path))
    assert result.exit_code == 0, result.output
    sample = load_sample(tmp_path / "labels" / "sample.json")
    assert sample.size == 30
    assert sample.seed == 2026
    assert result.output.splitlines() == [
        f"wrote {tmp_path / 'labels' / 'sample.json'}: 30 answers of repeat 0, "
        "5 the judge failed and 25 it passed, seed 2026",
    ]
    again = runner.invoke(app, sample_args(tmp_path))
    assert again.exit_code == 0
    assert load_sample(tmp_path / "labels" / "sample.json") == sample


def test_sample_names_missing_results(tmp_path):
    recorded_run(tmp_path)
    (tmp_path / "results" / "rag-v2.json").unlink()
    result = runner.invoke(app, sample_args(tmp_path))
    assert result.exit_code == 1
    assert "results missing: " in result.output and "rag-v2.json" in result.output
    assert not (tmp_path / "labels").exists()


def test_sample_refuses_results_without_judged_answers(tmp_path):
    ungraded = [
        syn.function_results("rag", version, [syn.case_record("rag-001")], graded=False)
        for version in ("v1", "v2")
    ]
    recorded_run(tmp_path, rag=ungraded)
    result = runner.invoke(app, sample_args(tmp_path))
    assert result.exit_code == 1
    assert "no judged answer with a valid verdict in the results" in result.output
    assert not (tmp_path / "labels").exists()


def test_sample_file_is_plain_json(tmp_path):
    recorded_run(tmp_path)
    runner.invoke(app, sample_args(tmp_path))
    data = json.loads((tmp_path / "labels" / "sample.json").read_text(encoding="utf-8"))
    assert data["size"] == len(data["items"]) == 30
