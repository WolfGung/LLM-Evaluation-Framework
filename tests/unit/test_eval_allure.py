"""The replay evaluation suite in the Allure report: labels, parameters, attachments.

The suite runs in a subprocess with `--alluredir` in `tmp_path`, and these
tests read the JSON files allure-pytest writes there (the Allure command line
is not needed).

Synthetic data: the manifest, the baselines and the recorded replies are made
up for the test (see `tests/unit/test_eval_suite.py`), recorded through the
real client with a mock transport into `tmp_path`. Nothing is written to the
repository's `cassettes/` or `results/`.
"""

import json
import re
from pathlib import Path

import pytest

from llmeval.baseline import CaseBaseline
from llmeval.cassettes import MANIFEST_FILE
from tests.unit.test_eval_suite import (
    CASE,
    KNOWN_PRIORITY_FAILURE,
    MODELS,
    RAG_CASE,
    RIGHT,
    WRONG_PRIORITY,
    record_rag_with_judge,
    record_reply,
    run_eval_suite,
    write_baseline,
    write_manifest,
)

RAG_SUITE = "tests/eval/test_rag_eval.py"
# What must never reach a report: the synthetic key, request headers, and
# long hex strings such as cassette keys (sha256 of a request).
KEY_LIKE = re.compile(r"synthetic-key-123|sk-or-|Bearer|Authorization|\b[0-9a-f]{40,}\b")


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "cassettes").mkdir()
    return tmp_path


def run_with_allure(ws: Path, *selection: str, suite: str = "tests/eval/test_triage_eval.py"):
    out_dir = ws / "allure-results"
    code, out = run_eval_suite(ws, *selection, f"--alluredir={out_dir}", suite=suite)
    return code, out, out_dir


def results_of(out_dir: Path) -> list[dict]:
    return [json.loads(path.read_text("utf-8")) for path in sorted(out_dir.glob("*-result.json"))]


def only_result(out_dir: Path) -> dict:
    (result,) = results_of(out_dir)
    return result


def labels(result: dict, name: str) -> list[str]:
    return [label["value"] for label in result["labels"] if label["name"] == name]


def parameters(result: dict) -> dict[str, str]:
    return {param["name"]: param["value"] for param in result.get("parameters", [])}


def attachments(out_dir: Path, result: dict) -> dict[str, str]:
    return {
        item["name"]: (out_dir / item["source"]).read_text("utf-8")
        for item in result.get("attachments", [])
    }


def write_rag_manifest(ws: Path) -> None:
    manifest = {
        "models": {"system": MODELS.system.model, "judge": MODELS.judge.model},
        "prompt_versions": {"rag": ["v1"]},
        "repeats": 1,
        "datasets": {"rag.jsonl": "0" * 64},
        "recorded_from": "2026-01-01T10:00:00Z",
        "recorded_to": "2026-01-01T10:01:00Z",
        "planned_calls": 2,
        "recorded_calls": 2,
        "judge_repeats": "first",
    }
    (ws / "cassettes" / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")


def test_a_triage_case_shows_its_function_failed_layer_category_and_case(ws):
    write_manifest(ws)
    record_reply(ws, WRONG_PRIORITY)
    write_baseline(ws, KNOWN_PRIORITY_FAILURE)
    code, out, out_dir = run_with_allure(ws, "-k", CASE.id)
    assert code == 0, out
    result = only_result(out_dir)
    assert labels(result, "epic") == ["triage"]
    # Listed under the layer it fails, not under the deterministic layer it passes.
    assert labels(result, "feature") == ["reference"]
    assert labels(result, "story") == [str(CASE.category)]
    assert parameters(result) == {"case": f"'{CASE.id}'", "version": "'v1'"}
    assert result["name"].startswith(f"{CASE.id} v1: ")
    assert f"priority {CASE.priority} (rule {CASE.priority_rule})" in result["description"]
    shown = attachments(out_dir, result)
    assert list(shown) == ["ticket", "reply, repeat 0", "failed checks", "record"]
    assert shown["ticket"] == CASE.text
    assert json.loads(shown["reply, repeat 0"]) == WRONG_PRIORITY
    assert "[reference] priority_match: expected 'high', got 'low'" in shown["failed checks"]
    assert json.loads(shown["record"])["id"] == CASE.id


def test_a_passing_case_is_listed_under_every_layer_it_was_checked_on(ws):
    write_manifest(ws)
    record_reply(ws, RIGHT)
    write_baseline(ws, CaseBaseline(passed=True))
    code, out, out_dir = run_with_allure(ws, "-k", CASE.id)
    assert code == 0, out
    result = only_result(out_dir)
    assert result["status"] == "passed"
    assert labels(result, "feature") == ["deterministic", "reference"]
    assert "failed checks" not in attachments(out_dir, result)


def test_a_graded_rag_case_attaches_the_retrieval_the_answer_and_the_verdict(ws):
    write_rag_manifest(ws)
    verdict = {"groundedness": 5, "helpfulness": 4, "tone": 5, "pass": True, "reasons": "Fine."}
    record_rag_with_judge(ws, verdict)
    code, out, out_dir = run_with_allure(ws, "-k", RAG_CASE.id, suite=RAG_SUITE)
    assert code == 0, out  # no baseline: pending baseline, after the replay
    result = only_result(out_dir)
    assert labels(result, "epic") == ["rag"]
    assert labels(result, "story") == ["answerable"]
    assert "30 days" in result["description"]
    shown = attachments(out_dir, result)
    assert list(shown) == [
        "question",
        "retrieved documents",
        "answer, repeat 0",
        "judge verdict",
        "failed checks",
        "record",
    ]
    assert shown["question"] == RAG_CASE.question
    assert "kb-returns: Returns (expected)" in shown["retrieved documents"].splitlines()
    assert shown["answer, repeat 0"] == "Synthetic answer [kb-warranty]."
    assert shown["judge verdict"] == (
        "repeat 0: groundedness 5, helpfulness 4, tone 5; the rubric rule passes; "
        "the judge said pass\nreasons: Fine."
    )


def test_nothing_key_like_reaches_the_report(ws):
    write_rag_manifest(ws)
    verdict = {"groundedness": 2, "helpfulness": 4, "tone": 5, "pass": False, "reasons": "No."}
    record_rag_with_judge(ws, verdict)
    code, out, out_dir = run_with_allure(ws, "-k", RAG_CASE.id, suite=RAG_SUITE)
    assert code == 0, out
    result = only_result(out_dir)
    shown = " ".join(attachments(out_dir, result).values())
    written = " ".join(
        [result["name"], result["description"], json.dumps(result["labels"]), shown]
    )
    assert not KEY_LIKE.findall(written)
    for path in out_dir.iterdir():
        assert "synthetic-key-123" not in path.read_text("utf-8")
