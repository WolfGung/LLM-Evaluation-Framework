"""Performance: latency percentiles, mean tokens and cost, by function and version.

Synthetic data: every call record below is made up for the test (latencies,
token counts and costs are chosen numbers, not measurements).
"""

from datetime import UTC, datetime

import pytest

from llmeval.perf import Performance, cost_sum, percentile, performance
from llmeval.results import (
    CallRecord,
    CaseRecord,
    CheckRecord,
    JudgeRecord,
    PairwiseCaseRecord,
    PairwiseOrderRecord,
    RunRecord,
    summarise,
    summarise_judge,
    summarise_pairwise,
)

TIME = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def call(latency=100.0, prompt=400, completion=50, reasoning=0, cost=0.0, source="provider"):
    return CallRecord(
        key="k" * 64,
        model_used="synthetic/model:free",
        latency_ms=latency,
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        cost_usd=cost,
        cost_source=source,
        finish_reason="stop",
        empty_reason=None,
        recorded_at=TIME,
    )


UNKNOWN = call(cost=None, source="unknown")


def test_percentiles_use_the_nearest_rank():
    values = [400.0, 100.0, 300.0, 200.0]
    assert percentile(values, 50) == 200.0
    assert percentile(values, 95) == 400.0
    assert percentile([250.0], 95) == 250.0
    assert percentile(list(range(1, 101)), 95) == 95
    assert percentile([], 50) is None


def test_percentile_refuses_a_rank_outside_0_to_100():
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_costs_sum_the_known_ones_and_count_the_unknown():
    assert cost_sum([0.25, None, 0.5, None]) == (0.75, 2)
    assert cost_sum([]) == (0.0, 0)


def test_performance_of_a_set_of_calls():
    calls = [
        call(latency=100, prompt=400, completion=40, reasoning=10, cost=0.001),
        call(latency=300, prompt=600, completion=60, reasoning=0, cost=0.003),
    ]
    perf = performance(calls, cases=1)
    assert perf.calls == 2
    assert perf.latency.p50_ms == 100.0 and perf.latency.p95_ms == 300.0
    assert perf.mean_prompt_tokens == 500.0
    assert perf.mean_completion_tokens == 50.0
    assert perf.mean_reasoning_tokens == 5.0
    assert perf.cost.model_dump() == {
        "total_usd": 0.004,
        "unknown_calls": 0,
        "per_case_usd": 0.004,
        "per_run_usd": 0.002,
    }


def test_an_unknown_cost_is_never_counted_as_zero():
    perf = performance([call(cost=0.002), UNKNOWN], cases=2)
    assert perf.cost.total_usd == 0.002
    assert perf.cost.unknown_calls == 1
    # A mean over a partly unknown total would understate the cost.
    assert perf.cost.per_case_usd is None and perf.cost.per_run_usd is None


def test_no_calls_give_empty_measures():
    perf = performance([], cases=0)
    assert perf == Performance.model_validate(
        {
            "calls": 0,
            "latency": {"p50_ms": None, "p95_ms": None},
            "mean_prompt_tokens": None,
            "mean_completion_tokens": None,
            "mean_reasoning_tokens": None,
            "cost": {
                "total_usd": 0.0,
                "unknown_calls": 0,
                "per_case_usd": None,
                "per_run_usd": None,
            },
        }
    )


def test_without_a_case_count_there_is_no_per_case_cost():
    assert performance([call(cost=0.001)]).cost.per_case_usd is None


def run(repeat, latency, cost=0.0, judge=None):
    check = CheckRecord(layer="deterministic", name="has_text", passed=True, detail="synthetic")
    return RunRecord(
        repeat=repeat,
        output="Synthetic answer.",
        call=call(latency=latency, cost=cost),
        checks=[check],
        judge=judge,
    )


def judge_record(latency, cost):
    return JudgeRecord(
        scores={"groundedness": 5, "helpfulness": 5, "tone": 5},
        judge_pass=True,
        rule_pass=True,
        reasons="Synthetic.",
        error=None,
        detail=None,
        raw=None,
        call=call(latency=latency, cost=cost),
    )


def test_the_summary_measures_the_system_calls_and_the_judge_calls_apart():
    cases = [
        CaseRecord(
            id="rag-001",
            category="answerable",
            input="Synthetic?",
            expected={},
            runs=[run(0, 100, 0.001, judge_record(900, 0.01)), run(1, 300, 0.003)],
        ),
        CaseRecord(
            id="rag-002",
            category="answerable",
            input="Synthetic?",
            expected={},
            runs=[run(0, 200, 0.002, judge_record(1100, 0.03))],
        ),
    ]
    summary = summarise("rag", cases)
    assert summary.performance.calls == 3
    assert summary.performance.latency.p50_ms == 200.0
    assert summary.performance.cost.total_usd == 0.006
    assert summary.performance.cost.per_case_usd == 0.003
    assert summary.performance.cost.per_run_usd == 0.002
    judge = summary.judge.performance
    assert judge.calls == 2
    assert judge.latency.p95_ms == 1100.0
    assert judge.cost.total_usd == 0.04
    assert judge.cost.per_case_usd == 0.02


def test_the_judge_summary_alone_has_no_case_count():
    summary = summarise_judge([("Synthetic answer.", judge_record(500, 0.01))])
    assert summary.performance.calls == 1
    assert summary.performance.cost.per_case_usd is None


def order(first, latency, cost):
    return PairwiseOrderRecord(
        shown_as_a=first,
        preferred="tie",
        winner="tie",
        reasons="Synthetic.",
        error=None,
        detail=None,
        raw=None,
        call=call(latency=latency, cost=cost),
    )


def test_the_pairwise_summary_measures_the_questions_asked():
    cases = [
        PairwiseCaseRecord(
            id="rag-001",
            category="answerable",
            input="Synthetic?",
            answers={"v1": "One.", "v2": "Two words."},
            outcome="tie",
            orders=[order("v1", 400, 0.002), order("v2", 600, 0.004)],
        ),
        PairwiseCaseRecord(
            id="rag-002",
            category="answerable",
            input="Synthetic?",
            answers={"v1": "Same.", "v2": "Same."},
            outcome="identical",
            orders=[],
        ),
    ]
    perf = summarise_pairwise(cases, ("v1", "v2")).performance
    assert perf.calls == 2
    assert perf.cost.total_usd == 0.006
    # The identical pair is a compared case that cost nothing.
    assert perf.cost.per_case_usd == 0.003
