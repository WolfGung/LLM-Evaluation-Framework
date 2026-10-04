"""The baseline: what the recorded run measured, and how a replay compares with it.

Synthetic data: every case record, result and manifest below is made up for
the test. No baseline is written into the repository.
"""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from llmeval.baseline import (
    PENDING_BASELINE,
    UPDATE_HINT,
    Baseline,
    BaselineError,
    CaseBaseline,
    build_baseline,
    case_baseline,
    compare,
    explain,
    load_baseline,
)
from llmeval.cassettes import RunManifest
from llmeval.results import (
    CallRecord,
    CaseRecord,
    CheckRecord,
    FunctionResults,
    RunRecord,
    summarise,
)

TIME = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
CALL = CallRecord(
    key="k" * 64,
    model_used="synthetic/system:free",
    latency_ms=1.0,
    prompt_tokens=1,
    completion_tokens=1,
    reasoning_tokens=0,
    cost_usd=0.0,
    cost_source="provider",
    finish_reason="stop",
    empty_reason=None,
    recorded_at=TIME,
)


def record(*runs_failing: tuple[str, ...], case_id="rag-001", category="answerable"):
    """A case record with one run per argument; each names the checks that fail."""
    names = ("retrieval_recall", "cites_retrieved", "required_facts")
    layers = {"retrieval_recall": "retrieval", "cites_retrieved": "deterministic"}
    runs = []
    for repeat, failing in enumerate(runs_failing):
        checks = [
            CheckRecord(
                layer=layers.get(name, "reference"),
                name=name,
                passed=name not in failing,
                detail=f"synthetic detail for {name}",
            )
            for name in names
        ]
        runs.append(RunRecord(repeat=repeat, output="synthetic", call=CALL, checks=checks))
    return CaseRecord(
        id=case_id, category=category, input="Synthetic question?", expected={}, runs=runs
    )


PASSING = record(())
FAILING = record(("required_facts",))


def test_a_case_baseline_lists_failed_checks_across_repeats():
    flaky = record((), ("required_facts",), ("cites_retrieved", "required_facts"))
    baseline = case_baseline(flaky)
    assert not baseline.passed
    assert baseline.failed_checks == (
        ("reference", "required_facts"),
        ("deterministic", "cites_retrieved"),
    )
    assert case_baseline(PASSING) == CaseBaseline(passed=True)


def test_passed_and_failed_checks_must_agree():
    with pytest.raises(ValidationError):
        CaseBaseline(passed=True, failed_checks=(("reference", "required_facts"),))
    with pytest.raises(ValidationError):
        CaseBaseline(passed=False)


def test_no_baseline_entry_is_pending():
    verdict = compare(PASSING, None)
    assert (verdict.outcome, verdict.message) == ("pending", PENDING_BASELINE)
    assert PENDING_BASELINE == "pending baseline"


def test_a_passing_case_that_still_passes():
    assert compare(PASSING, CaseBaseline(passed=True)).outcome == "pass"


def test_a_passing_case_that_now_fails_is_a_regression():
    verdict = compare(FAILING, CaseBaseline(passed=True))
    assert verdict.outcome == "fail"
    assert verdict.message == explain(FAILING)
    assert "[reference] required_facts: synthetic detail for required_facts" in verdict.message


def test_a_known_failure_is_an_xfail_naming_the_checks():
    expected = case_baseline(FAILING)
    verdict = compare(FAILING, expected)
    assert verdict.outcome == "xfail"
    assert "reference/required_facts" in verdict.message


def test_a_known_failure_that_now_passes_asks_for_a_deliberate_update():
    verdict = compare(PASSING, case_baseline(FAILING))
    assert verdict.outcome == "xpass"
    assert verdict.message == UPDATE_HINT
    assert UPDATE_HINT == "now passes; update the baseline deliberately (make baseline)"


def test_a_known_failure_with_a_new_failing_check_is_a_regression():
    worse = record(("required_facts", "cites_retrieved"))
    verdict = compare(worse, case_baseline(FAILING))
    assert verdict.outcome == "fail"
    assert "deterministic/cites_retrieved" in verdict.message
    assert "baseline did not" in verdict.message


def test_explain_says_when_retrieval_missed():
    text = explain(record(("retrieval_recall", "required_facts")))
    assert "retrieval miss" in text


MANIFEST = RunManifest(
    models={"system": "synthetic/system:free", "judge": "synthetic/judge:free"},
    prompt_versions={"rag": ("v1",)},
    repeats=1,
    datasets={"rag.jsonl": "0" * 64},
    recorded_from=TIME,
    recorded_to=TIME,
    planned_calls=2,
    recorded_calls=2,
)


def results_of(*records):
    return FunctionResults(
        function="rag",
        version="v1",
        mode="replay",
        model="synthetic/system:free",
        prompt_sha256="p" * 64,
        dataset="rag.jsonl",
        dataset_sha256="0" * 64,
        repeats=1,
        summary=summarise("rag", list(records)),
        cases=list(records),
    )


def test_build_baseline_from_replay_results():
    baseline = build_baseline(
        [results_of(PASSING, record(("required_facts",), case_id="rag-002"))], MANIFEST
    )
    assert baseline.case("rag", "v1", "rag-001") == CaseBaseline(passed=True)
    assert baseline.case("rag", "v1", "rag-002").failed_checks == (("reference", "required_facts"),)
    assert baseline.case("rag", "v2", "rag-001") is None
    assert baseline.case("rag", "v1", "rag-999") is None
    metrics = baseline.functions["rag"]["v1"].metrics
    assert metrics.all_checks == 0.5
    assert metrics.layers["reference"] == 0.5
    assert metrics.stable_share is None
    assert baseline.provenance.models == MANIFEST.models
    assert baseline.provenance.recorded_to == TIME


def test_build_baseline_refuses_live_results():
    live = results_of(PASSING).model_copy(update={"mode": "live"})
    with pytest.raises(BaselineError, match="replay"):
        build_baseline([live], MANIFEST)


def test_load_baseline(tmp_path):
    assert load_baseline(tmp_path / "baseline.json") is None
    baseline = build_baseline([results_of(PASSING)], MANIFEST)
    path = tmp_path / "baseline.json"
    path.write_text(baseline.model_dump_json(indent=2), encoding="utf-8")
    assert load_baseline(path) == baseline
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(BaselineError, match="baseline.json"):
        load_baseline(path)


def test_the_baseline_round_trips_through_json():
    baseline = build_baseline(
        [results_of(PASSING, FAILING.model_copy(update={"id": "rag-002"}))], MANIFEST
    )
    assert Baseline.model_validate_json(baseline.model_dump_json()) == baseline
