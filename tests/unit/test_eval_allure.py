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

import httpx
import pytest
from pydantic import SecretStr

from app.assistant import prepare
from app.triage import triage
from llmeval.baseline import CaseBaseline
from llmeval.cassettes import MANIFEST_FILE, CassetteStore
from llmeval.checks.judge import RUBRIC_PATH, Judge, load_rubric
from llmeval.client import ModelClient
from llmeval.config import Config, Mode, Settings
from llmeval.datasets import load_triage
from llmeval.runner import run_rag
from tests.unit.test_eval_suite import (
    CASE,
    KNOWN_PRIORITY_FAILURE,
    MODELS,
    RAG_CASE,
    RIGHT,
    ROOT,
    WRONG_PRIORITY,
    WRONG_TWICE,
    NoWait,
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
    written = " ".join([result["name"], result["description"], json.dumps(result["labels"]), shown])
    assert not KEY_LIKE.findall(written)
    for path in out_dir.iterdir():
        assert "synthetic-key-123" not in path.read_text("utf-8")


# --- categories and environment ------------------------------------------------

REGRESSION = "Regression against the baseline"
FIXED = "Known failure that now passes: update the baseline"
KNOWN = "Known failure in the baseline"
PENDING = "Pending: no recorded run or no baseline"


def categories_of(out_dir: Path, result: dict) -> list[str]:
    """The categories of categories.json a result falls in, by Allure's rule:
    the status is one of `matchedStatuses`, and `messageRegex` matches the
    whole message (Java's Pattern.matches with DOTALL)."""
    categories = json.loads((out_dir / "categories.json").read_text("utf-8"))
    message = (result.get("statusDetails") or {}).get("message")
    found = []
    for category in categories:
        if result["status"] not in category["matchedStatuses"]:
            continue
        regex = category.get("messageRegex")
        if regex is None or (message is not None and re.fullmatch(regex, message, re.DOTALL)):
            found.append(category["name"])
    return found


@pytest.mark.parametrize(
    ("baseline", "reply", "status", "expected"),
    [
        pytest.param(CaseBaseline(passed=True), RIGHT, "passed", [], id="pass"),
        pytest.param(
            CaseBaseline(passed=True), WRONG_PRIORITY, "failed", [REGRESSION], id="regression"
        ),
        pytest.param(KNOWN_PRIORITY_FAILURE, WRONG_TWICE, "failed", [REGRESSION], id="new-check"),
        pytest.param(KNOWN_PRIORITY_FAILURE, WRONG_PRIORITY, "skipped", [KNOWN], id="known"),
        pytest.param(KNOWN_PRIORITY_FAILURE, RIGHT, "failed", [FIXED], id="fixed"),
        pytest.param(None, RIGHT, "skipped", [PENDING], id="pending-baseline"),
    ],
)
def test_each_outcome_falls_in_exactly_its_category(ws, baseline, reply, status, expected):
    write_manifest(ws)
    record_reply(ws, reply)
    if baseline is not None:
        write_baseline(ws, baseline)
    _, out, out_dir = run_with_allure(ws, "-k", CASE.id)
    result = only_result(out_dir)
    assert result["status"] == status, out
    assert categories_of(out_dir, result) == expected


def test_without_a_recorded_run_every_case_is_pending(ws):
    _, out, out_dir = run_with_allure(ws, "-k", CASE.id)
    result = only_result(out_dir)
    assert categories_of(out_dir, result) == [PENDING], out


def test_the_environment_names_the_recording(ws):
    write_manifest(ws)
    record_reply(ws, RIGHT)
    _, out, out_dir = run_with_allure(ws, "-k", CASE.id)
    assert (out_dir / "environment.properties").read_text("utf-8") == (
        f"system.model={MODELS.system.model}\n"
        f"judge.model={MODELS.judge.model}\n"
        "recorded.from=2026-01-01 10:00 UTC\n"
        "recorded.to=2026-01-01 10:01 UTC\n"
        "repeats=1\n"
        "judge_repeats=first\n"
        "calls=1\n"
        "prompt.versions=triage v1\n"
    ), out


def test_without_a_recorded_run_the_environment_says_pending(ws):
    _, out, out_dir = run_with_allure(ws, "-k", CASE.id)
    environment = (out_dir / "environment.properties").read_text("utf-8")
    assert environment == "recording=pending first recorded run\n", out


def test_report_files_are_written_only_with_an_alluredir(ws):
    write_manifest(ws)
    record_reply(ws, RIGHT)
    code, out = run_eval_suite(ws, "-k", CASE.id)
    assert code == 0, out
    assert not list(ws.rglob("categories.json"))
    assert not list(ws.rglob("environment.properties"))


# --- layer results ---------------------------------------------------------------

LAYER_SUITE = "tests/eval/test_layers.py"
TRIAGE_CASES = load_triage(ROOT / "datasets" / "triage.jsonl")


def record_every_ticket(ws: Path, wrong: frozenset[str] = frozenset()) -> None:
    """Record a synthetic reply to every triage case: its own labels, except a
    wrong priority for the case ids in `wrong`."""
    cases = {case.text: case for case in TRIAGE_CASES}

    def reply(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        case = cases[body["messages"][-1]["content"]]
        priority = str(case.priority)
        if case.id in wrong:
            priority = "urgent" if priority == "low" else "low"
        labels = {
            "category": str(case.category),
            "priority": priority,
            "order_id": case.order_id,
            "summary": "Synthetic summary.",
        }
        return httpx.Response(
            200,
            json={
                "id": "gen-synthetic",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(labels)},
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0},
            },
        )

    config = Config(models=MODELS, settings=Settings(api_key=SecretStr("synthetic-key-123")))
    with ModelClient(
        Mode.RECORD,
        CassetteStore(ws / "cassettes"),
        config,
        httpx.MockTransport(reply),
        limiter=NoWait(),
    ) as client:
        for case in TRIAGE_CASES:
            triage(client, MODELS.system, case.text, "v1", case=case.id)


def write_layer_baseline(ws: Path) -> None:
    """A baseline where every triage v1 run passed both layers."""
    write_baseline(ws, CaseBaseline(passed=True))
    path = ws / "baseline.json"
    data = json.loads(path.read_text("utf-8"))
    data["functions"]["triage"]["v1"]["metrics"]["layers"] = {
        "deterministic": 1.0,
        "reference": 1.0,
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def layer_result(out_dir: Path, layer: str) -> dict:
    (result,) = [r for r in results_of(out_dir) if labels(r, "feature") == [layer]]
    return result


def test_a_layer_drop_within_its_tolerance_passes_and_shows_the_rate(ws):
    write_manifest(ws)
    record_every_ticket(ws, wrong=frozenset({TRIAGE_CASES[0].id}))
    write_layer_baseline(ws)
    code, out, out_dir = run_with_allure(ws, suite=LAYER_SUITE)
    assert code == 0, out
    assert "2 passed" in out
    result = layer_result(out_dir, "reference")
    assert result["status"] == "passed"
    assert result["name"] == "triage v1: reference layer, 97.5% of runs pass (39 of 40)"
    assert labels(result, "epic") == ["triage"]
    assert labels(result, "story") == ["pass rate"]
    shown = attachments(out_dir, result)
    assert "priority_match: 97.5% (39 of 40)" in shown["checks"].splitlines()
    assert "category_match: 100.0% (40 of 40)" in shown["checks"].splitlines()
    (failing,) = shown["failing runs"].splitlines()
    assert failing.startswith(f"{TRIAGE_CASES[0].id} repeat 0: priority_match: ")


def test_a_layer_drop_beyond_its_tolerance_is_a_regression(ws):
    write_manifest(ws)
    record_every_ticket(ws, wrong=frozenset(case.id for case in TRIAGE_CASES[:3]))
    write_layer_baseline(ws)
    code, out, out_dir = run_with_allure(ws, suite=LAYER_SUITE)
    assert code == 1, out
    result = layer_result(out_dir, "reference")
    assert result["statusDetails"]["message"] == (
        "Failed: regression: triage v1 reference layer 92.5% (37 of 40 runs), "
        "baseline 100.0%, allowed drop 5.0 pp"
    )
    assert categories_of(out_dir, result) == [REGRESSION]
    assert layer_result(out_dir, "deterministic")["status"] == "passed"


def test_a_layer_without_a_baseline_is_pending_after_its_rate(ws):
    write_manifest(ws)
    record_every_ticket(ws)
    code, out, out_dir = run_with_allure(ws, suite=LAYER_SUITE)
    assert code == 0, out
    result = layer_result(out_dir, "reference")
    assert result["name"] == "triage v1: reference layer, 100.0% of runs pass (40 of 40)"
    assert categories_of(out_dir, result) == [PENDING]
    assert attachments(out_dir, result)["failing runs"] == "none"


# --- pairwise comparison -----------------------------------------------------------

PAIRWISE_SUITE = "tests/eval/test_pairwise_eval.py"
ANSWERS = {"v1": "Synthetic answer one [kb-returns].", "v2": "Synthetic answer two [kb-returns]."}


def completion(body: dict, content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "gen-synthetic",
            "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.0},
        },
    )


def record_comparison(ws: Path, preferences: tuple[str, str], answers=ANSWERS) -> None:
    """Record RAG_CASE answered by v1 and v2 (repeat 0), graded, and compared:
    `preferences` are the judge's picks with v1 shown as A, then v2 shown as A."""
    manifest = {
        "models": {"system": MODELS.system.model, "judge": MODELS.judge.model},
        "prompt_versions": {"rag": ["v1", "v2"]},
        "repeats": 1,
        "datasets": {"rag.jsonl": "0" * 64},
        "recorded_from": "2026-01-01T10:00:00Z",
        "recorded_to": "2026-01-01T10:01:00Z",
        "planned_calls": 6,
        "recorded_calls": 6,
        "judge_repeats": "first",
    }
    (ws / "cassettes" / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    grade = {"groundedness": 5, "helpfulness": 5, "tone": 5, "pass": True, "reasons": "Fine."}

    def reply_for(version: str):
        def reply(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            schema = (body.get("response_format") or {}).get("json_schema", {}).get("name")
            if schema == "pairwise_verdict":
                shown_a = body["messages"][-1]["content"].split('<answer id="A">\n', 1)[1]
                first = shown_a.startswith(answers["v1"])
                pick = preferences[0] if first else preferences[1]
                return completion(
                    body, json.dumps({"preferred": pick, "reasons": f"Picked {pick}."})
                )
            if schema is not None:
                return completion(body, json.dumps(grade))
            return completion(body, answers[version])

        return reply

    config = Config(models=MODELS, settings=Settings(api_key=SecretStr("synthetic-key-123")))
    records = {}
    for version in ("v1", "v2"):
        with ModelClient(
            Mode.RECORD,
            CassetteStore(ws / "cassettes"),
            config,
            httpx.MockTransport(reply_for(version)),
            limiter=NoWait(),
        ) as client:
            judge = Judge(client, MODELS.judge, load_rubric(ROOT / RUBRIC_PATH))
            records[version] = run_rag(
                client, MODELS.system, [RAG_CASE], version, repeats=1, judge=judge
            )
    with ModelClient(
        Mode.RECORD,
        CassetteStore(ws / "cassettes"),
        config,
        httpx.MockTransport(reply_for("v1")),
        limiter=NoWait(),
    ) as client:
        judge = Judge(client, MODELS.judge, load_rubric(ROOT / RUBRIC_PATH))
        pair = (records["v1"][0].runs[0].output, records["v2"][0].runs[0].output)
        _, hits = prepare(RAG_CASE.question, "v1")
        judge.compare(RAG_CASE.question, hits, *pair, case=RAG_CASE.id, versions=("v1", "v2"))


def test_a_comparison_shows_both_orders_and_never_fails_on_a_preference(ws):
    record_comparison(ws, ("A", "A"))  # each order picks the answer shown first
    code, out, out_dir = run_with_allure(ws, "-k", RAG_CASE.id, suite=PAIRWISE_SUITE)
    assert code == 0, out
    result = only_result(out_dir)
    assert result["status"] == "passed"
    assert result["name"] == (
        f"{RAG_CASE.id}: v1 vs v2, inconsistent: the preference changed with the position"
    )
    assert labels(result, "epic") == ["rag"]
    assert labels(result, "feature") == ["pairwise comparison"]
    assert labels(result, "story") == ["answerable"]
    assert labels(result, "tag") == ["inconsistent"]
    assert parameters(result) == {"case": f"'{RAG_CASE.id}'", "pair": "'v1 vs v2'"}
    shown = attachments(out_dir, result)
    assert list(shown) == [
        "question",
        "answer v1",
        "answer v2",
        "order 1: v1 shown as A",
        "order 2: v2 shown as A",
    ]
    assert shown["answer v2"] == ANSWERS["v2"]
    assert shown["order 1: v1 shown as A"] == "preferred: A, which is v1\nreasons: Picked A."
    assert shown["order 2: v2 shown as A"] == "preferred: A, which is v2\nreasons: Picked A."


def test_a_consistent_preference_is_named_in_the_title(ws):
    record_comparison(ws, ("B", "A"))  # v2 in both orders
    code, out, out_dir = run_with_allure(ws, "-k", RAG_CASE.id, suite=PAIRWISE_SUITE)
    assert code == 0, out
    assert only_result(out_dir)["name"] == f"{RAG_CASE.id}: v1 vs v2, v2 preferred in both orders"


def test_identical_answers_are_not_compared(ws):
    same = {"v1": ANSWERS["v1"], "v2": ANSWERS["v1"]}
    record_comparison(ws, ("A", "A"), answers=same)
    code, out, out_dir = run_with_allure(ws, "-k", RAG_CASE.id, suite=PAIRWISE_SUITE)
    assert code == 0, out
    result = only_result(out_dir)
    assert result["name"] == f"{RAG_CASE.id}: v1 vs v2, identical answers: the judge is not asked"
    assert list(attachments(out_dir, result)) == ["question", "answer v1", "answer v2"]


def test_one_recorded_version_has_nothing_to_compare(ws):
    write_rag_manifest(ws)
    code, out, out_dir = run_with_allure(ws, "-k", RAG_CASE.id, suite=PAIRWISE_SUITE)
    assert code == 0, out
    assert "nothing to compare" in out
    assert categories_of(out_dir, only_result(out_dir)) == [PENDING]
