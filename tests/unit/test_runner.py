"""The runner: plan calls, run cases through the model client, apply checks, write results.

Synthetic data: the knowledge base, the cases, the manifest and every model
reply below are made up for the test. Results and cassettes are written only
into `tmp_path`.
"""

import json
from datetime import UTC, datetime

import pytest

from app.retrieval import BM25Index, Document
from llmeval.cassettes import MANIFEST_FILE, CassetteStore
from llmeval.checks.reference import INVALID
from llmeval.client import MissingRecording, ModelClient
from llmeval.config import Config, Mode, ModelsConfig, Settings
from llmeval.datasets import RagCase, TriageCase
from llmeval.results import FunctionResults, results_path, write_live_results, write_results
from llmeval.runner import plan_requests, run
from tests.app.fakes import STRUCTURED, UNSTRUCTURED, FakeModel

INDEX = BM25Index(
    [
        Document("kb-alpha", "Returns", "Synthetic shop: returns are accepted within 30 days."),
        Document("kb-beta", "Shipping", "Synthetic shop: standard shipping costs $5.95."),
        Document("kb-gamma", "Pickup", "Synthetic shop: pickup orders are held for 7 days."),
    ]
)

RAG_CASES = (
    RagCase(
        id="rag-001",
        category="answerable",
        question="How many days do I have for returns?",
        expected="answer",
        required_facts=("30 days",),
        expected_docs=("kb-alpha",),
    ),
    RagCase(
        id="rag-002",
        category="unanswerable",
        question="Do you rent ladders?",
        expected="dont_know",
        forbidden=("rental fee",),
    ),
    RagCase(
        id="rag-003",
        category="answerable",
        question="What does standard shipping cost?",
        expected="answer",
        required_facts=("$5.95",),
        expected_docs=("kb-gamma",),  # deliberately not what the search returns
    ),
)

TRIAGE_CASES = (
    TriageCase(
        id="tri-001",
        text="Synthetic: my parcel TS-123456 is two weeks late.",
        category="shipping",
        priority="high",
        order_id="TS-123456",
        priority_rule="H1",
    ),
    TriageCase(
        id="tri-002",
        text="Synthetic: do you take gift cards?",
        category="payment",
        priority="low",
        order_id=None,
        priority_rule="L1",
    ),
)

RAG_REPLIES = {
    "How many days do I have for returns?": "You have 30 days [kb-alpha].",
    "Do you rent ladders?": "I don't know. Please contact support.",
    "What does standard shipping cost?": "It costs $5.95 [kb-beta].",
}
TRIAGE_REPLIES = {
    TRIAGE_CASES[0].text: json.dumps(
        {"category": "shipping", "priority": "urgent", "order_id": "TS-123456", "summary": "Late."}
    ),
    TRIAGE_CASES[1].text: "Payment question, low priority.",
}


def reply(messages):
    question = messages[-1]["content"]
    return RAG_REPLIES.get(question) or TRIAGE_REPLIES[question]


def write_manifest(cassettes, repeats=1):
    cassettes.mkdir(exist_ok=True)
    manifest = {
        "models": {"system": UNSTRUCTURED.model, "judge": STRUCTURED.model},
        "prompt_versions": {"rag": ["v1", "v2"], "triage": ["v1", "v2"]},
        "repeats": repeats,
        "datasets": {"rag.jsonl": "0" * 64, "triage.jsonl": "1" * 64},
        "recorded_from": "2026-01-01T10:00:00Z",
        "recorded_to": "2026-01-01T11:00:00Z",
        "planned_calls": 20,
        "recorded_calls": 20,
    }
    (cassettes / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")


def run_all(client, tmp_path, mode=Mode.REPLAY, repeats=1, versions=None):
    datasets = tmp_path / "datasets"
    datasets.mkdir(exist_ok=True)
    (datasets / "rag.jsonl").write_text("synthetic\n", encoding="utf-8")
    (datasets / "triage.jsonl").write_text("synthetic\n", encoding="utf-8")
    return run(
        client,
        UNSTRUCTURED,
        mode=mode,
        cassettes_dir=tmp_path / "cassettes",
        results_dir=tmp_path / "results",
        live_results_dir=tmp_path / "results-live",
        rag_cases=RAG_CASES,
        triage_cases=TRIAGE_CASES,
        dataset_paths={"rag": datasets / "rag.jsonl", "triage": datasets / "triage.jsonl"},
        versions=versions or {"rag": ("v1",), "triage": ("v1",)},
        repeats=repeats,
        index=INDEX,
    )


# --- plan -------------------------------------------------------------------


def test_the_plan_covers_datasets_versions_and_repeats():
    plan = plan_requests(
        UNSTRUCTURED,
        rag_cases=RAG_CASES,
        triage_cases=TRIAGE_CASES,
        versions={"rag": ("v1", "v2"), "triage": ("v1", "v2")},
        repeats=3,
        index=INDEX,
    )
    assert len(plan) == (3 + 2) * 2 * 3
    assert len({p.key for p in plan}) == len(plan)
    assert plan[0].function == "rag" and plan[0].case_id == "rag-001" and plan[0].repeat == 0
    assert {p.tag.file_stem for p in plan} == {"rag-v1", "rag-v2", "triage-v1", "triage-v2"}


def test_planned_keys_are_the_keys_the_run_uses(tmp_path):
    write_manifest(tmp_path / "cassettes")
    fake = FakeModel(reply=reply)
    run_all(fake, tmp_path, repeats=2, versions={"rag": ("v1", "v2"), "triage": ("v2",)})
    plan = plan_requests(
        UNSTRUCTURED,
        rag_cases=RAG_CASES,
        triage_cases=TRIAGE_CASES,
        versions={"rag": ("v1", "v2"), "triage": ("v2",)},
        repeats=2,
        index=INDEX,
    )
    from llmeval.cassettes import request_key

    used = [request_key(call["body"], call["repeat"]) for call in fake.calls]
    assert sorted(used) == sorted(p.key for p in plan)


def test_the_plan_refuses_an_unknown_version():
    with pytest.raises(ValueError, match="v9"):
        plan_requests(UNSTRUCTURED, rag_cases=RAG_CASES, versions={"rag": ("v9",)}, index=INDEX)


# --- pending and loud failures ----------------------------------------------


def test_replay_without_a_manifest_is_pending_and_writes_nothing(tmp_path):
    (tmp_path / "cassettes").mkdir()
    (tmp_path / "cassettes" / ".gitkeep").write_text("", encoding="utf-8")
    fake = FakeModel(reply=reply)
    outcome = run_all(fake, tmp_path)
    assert outcome.pending
    assert outcome.reason == "pending first recorded run"
    assert outcome.written == ()
    assert fake.calls == []
    assert not (tmp_path / "results").exists()


def test_replay_with_a_manifest_and_a_missing_entry_fails_loudly(tmp_path):
    write_manifest(tmp_path / "cassettes")
    config = Config(
        models=ModelsConfig(system=UNSTRUCTURED, judge=STRUCTURED, repeats=1, rpm=18),
        settings=Settings(),
    )
    client = ModelClient(Mode.REPLAY, CassetteStore(tmp_path / "cassettes"), config)
    with pytest.raises(MissingRecording, match=r"no recording for rag-001/v1/0: run make record"):
        run_all(client, tmp_path)
    assert not (tmp_path / "results").exists()


def test_a_record_run_writes_no_results(tmp_path):
    # Recording fills the cassettes; results come from a replay of the
    # recorded run, never from the recording itself.
    outcome = run_all(FakeModel(reply=reply), tmp_path, mode=Mode.RECORD)
    assert not outcome.pending
    assert outcome.written == ()
    assert {r.mode for r in outcome.results} == {"record"}
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "results-live").exists()


def test_a_live_run_writes_only_to_the_live_directory(tmp_path):
    outcome = run_all(FakeModel(reply=reply), tmp_path, mode=Mode.LIVE)
    assert {p.parent for p in outcome.written} == {tmp_path / "results-live"}
    assert {p.name for p in outcome.written} == {"rag-v1.json", "triage-v1.json"}
    stored = json.loads((tmp_path / "results-live" / "rag-v1.json").read_text(encoding="utf-8"))
    assert stored["mode"] == "live"
    assert not (tmp_path / "results").exists()


def test_results_take_replay_results_only(tmp_path):
    live = run_all(FakeModel(reply=reply), tmp_path, mode=Mode.LIVE).results[0]
    with pytest.raises(ValueError, match="only replay results"):
        write_results(live, tmp_path / "results")
    write_manifest(tmp_path / "cassettes")
    replayed = run_all(FakeModel(reply=reply), tmp_path).results[0]
    with pytest.raises(ValueError, match="only live results"):
        write_live_results(replayed, tmp_path / "results-live")


# --- results ----------------------------------------------------------------


@pytest.fixture
def results(tmp_path):
    write_manifest(tmp_path / "cassettes")
    outcome = run_all(FakeModel(reply=reply), tmp_path)
    return {r.function: r for r in outcome.results}, tmp_path


def checks_of(record, repeat=0):
    return {c.name: c for c in record.runs[repeat].checks}


def test_results_files_hold_what_the_run_returned(results):
    by_function, tmp_path = results
    path = results_path(tmp_path / "results", "rag", "v1")
    assert path == tmp_path / "results" / "rag-v1.json"
    stored = FunctionResults.model_validate_json(path.read_text(encoding="utf-8"))
    assert stored == by_function["rag"]
    assert stored.function == "rag" and stored.version == "v1" and stored.mode == "replay"
    assert stored.model == UNSTRUCTURED.model
    assert len(stored.prompt_sha256) == 64 and len(stored.dataset_sha256) == 64


def test_results_are_byte_for_byte_reproducible(tmp_path):
    write_manifest(tmp_path / "cassettes")
    run_all(FakeModel(reply=reply), tmp_path)
    first = (tmp_path / "results" / "triage-v1.json").read_bytes()
    run_all(FakeModel(reply=reply), tmp_path)
    assert (tmp_path / "results" / "triage-v1.json").read_bytes() == first


def test_an_answerable_case_gets_retrieval_deterministic_and_reference_checks(results):
    rag = results[0]["rag"]
    record = rag.cases[0]
    assert record.id == "rag-001" and record.input == RAG_CASES[0].question
    assert [(c.layer, c.name) for c in record.runs[0].checks] == [
        ("retrieval", "retrieval_recall"),
        ("deterministic", "has_text"),
        ("deterministic", "cites_retrieved"),
        ("deterministic", "no_unretrieved_citations"),
        ("deterministic", "no_forbidden"),
        ("deterministic", "within_length"),
        ("reference", "required_facts"),
    ]
    assert record.runs[0].passed
    assert record.runs[0].cited == ["kb-alpha"]
    assert record.runs[0].retrieved[0] == "kb-alpha"


def test_an_unanswerable_case_is_checked_for_declining(results):
    record = results[0]["rag"].cases[1]
    checks = checks_of(record)
    assert "dont_know" in checks and checks["dont_know"].passed
    assert "required_facts" not in checks and "retrieval_recall" not in checks
    assert "cites_retrieved" not in checks


def test_a_retrieval_miss_is_told_apart_from_the_answer(results):
    # rag-003 expects kb-gamma, which the search does not return; the answer
    # itself states the fact. The retrieval layer fails, the reference passes.
    checks = checks_of(results[0]["rag"].cases[2])
    assert not checks["retrieval_recall"].passed
    assert "kb-gamma" in checks["retrieval_recall"].detail
    assert checks["required_facts"].passed


def test_rag_summary_rates(results):
    summary = results[0]["rag"].summary
    assert summary.runs == 3 and summary.cases == 3
    assert summary.layers["retrieval"].model_dump() == {"passed": 1, "total": 2, "rate": 0.5}
    assert summary.layers["reference"].model_dump() == {"passed": 2, "total": 2, "rate": 1.0}
    assert summary.layers["deterministic"].passed == 3
    assert summary.checks["dont_know"].model_dump() == {"passed": 1, "total": 1, "rate": 1.0}
    assert summary.all_checks.model_dump() == {"passed": 2, "total": 3, "rate": 0.6667}
    assert summary.by_category["answerable"].model_dump() == {"passed": 1, "total": 2, "rate": 0.5}
    assert summary.retrieval_recall == 0.5


def test_a_triage_reply_with_a_wrong_label(results):
    record = results[0]["triage"].cases[0]
    checks = checks_of(record)
    assert all(checks[n].passed for n in ("json_valid", "required_fields", "enums_valid"))
    assert checks["category_match"].passed and checks["order_id_match"].passed
    assert not checks["priority_match"].passed
    assert record.runs[0].error is None


def test_an_invalid_triage_reply_is_recorded_not_raised(results):
    record = results[0]["triage"].cases[1]
    run0 = record.runs[0]
    assert run0.output == "Payment question, low priority."
    assert run0.error == "invalid_json"
    assert not any(c.passed for c in run0.checks)


def test_triage_summary_accuracy_and_confusion(results):
    summary = results[0]["triage"].summary
    assert summary.accuracy == {"category": 0.5, "priority": 0.0, "order_id": 0.5}
    priority = summary.confusion["priority"]
    assert priority["high"]["urgent"] == 1
    assert priority["low"][INVALID] == 1
    assert summary.layers["deterministic"].model_dump() == {"passed": 1, "total": 2, "rate": 0.5}
    assert summary.layers["reference"].model_dump() == {"passed": 0, "total": 2, "rate": 0.0}


def test_repeats_are_separate_runs(tmp_path):
    write_manifest(tmp_path / "cassettes", repeats=2)
    fake = FakeModel(reply=reply)
    outcome = run_all(fake, tmp_path, repeats=2)
    rag = next(r for r in outcome.results if r.function == "rag")
    assert [run.repeat for run in rag.cases[0].runs] == [0, 1]
    assert rag.summary.runs == 6
    assert sorted({call["repeat"] for call in fake.calls}) == [0, 1]


def test_call_details_are_kept_per_run(results):
    call = results[0]["rag"].cases[0].runs[0].call
    assert call.model_used == UNSTRUCTURED.model
    assert (call.prompt_tokens, call.completion_tokens) == (10, 5)
    assert call.latency_ms == 1.0 and call.cost_usd == 0.0
    assert call.recorded_at == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert len(call.key) == 64
