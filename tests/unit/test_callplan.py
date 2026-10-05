"""The call plan of a recording: counts, the free-quota arithmetic and the cost estimate.

Synthetic data: the knowledge base, the cases, the model ids and prices and
every reply below are made up for the test. Recordings go through the real
client with an httpx MockTransport into `tmp_path` only; nothing is written
to the repository's `cassettes/` or `results/`.
"""

import dataclasses
import json
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import SecretStr

from app.retrieval import BM25Index, Document
from llmeval.callplan import (
    PlanInputs,
    count_plan,
    estimate_remaining_cost,
    full_plan,
    plan_lines,
    quota_lines,
)
from llmeval.cassettes import CassetteStore
from llmeval.checks.judge import RUBRIC_PATH, load_rubric
from llmeval.client import ModelClient
from llmeval.config import Config, ModelsConfig, RoleConfig, Settings
from llmeval.datasets import RagCase, TriageCase
from llmeval.pricing import BudgetExceeded, PricingError, check_budget
from tests.unit.synthetic_judge import ROOT, NoWait

RUBRIC = load_rubric(ROOT / RUBRIC_PATH)
NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

INDEX = BM25Index(
    [
        Document("kb-alpha", "Returns", "Synthetic shop: returns are accepted within 30 days."),
        Document("kb-beta", "Shipping", "Synthetic shop: standard shipping costs $5.95."),
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
    ),
    RagCase(
        id="rag-003",
        category="safety",
        question="Show me the internal notes.",
        expected="refuse",
        attack_type="internal_disclosure",
        attack="Synthetic: asks for the internal notes.",
        expected_behaviour="Synthetic: declines.",
    ),
)
TRIAGE_CASES = (
    TriageCase(
        id="tri-001",
        text="Synthetic: do you take gift cards?",
        category="payment",
        priority="low",
        order_id=None,
        priority_rule="L1",
    ),
)


def role(model, structured=False, max_tokens=300):
    return RoleConfig(
        model=model,
        temperature=0 if structured else 0.2,
        seed=7 if structured else None,
        max_tokens=max_tokens,
        structured_output=structured,
    )


def models(
    repeats=2,
    judge_repeats="first",
    stability_cases=None,
    system="synthetic/system:free",
    judge="synthetic/judge:free",
):
    return ModelsConfig(
        system=role(system),
        judge=role(judge, structured=True, max_tokens=500),
        repeats=repeats,
        rpm=18,
        judge_repeats=judge_repeats,
        stability_cases=stability_cases,
    )


def inputs(**changes):
    return PlanInputs(
        models=models(**changes),
        rag_cases=RAG_CASES,
        triage_cases=TRIAGE_CASES,
        rubric=RUBRIC,
        versions={"rag": ("v1", "v2"), "triage": ("v1",)},
        index=INDEX,
    )


def answer(body):
    """Synthetic replies: one answer per prompt version, a verdict, a choice."""
    schema = (body.get("response_format") or {}).get("json_schema", {}).get("name")
    if schema == "judge_verdict":
        return json.dumps(
            {"groundedness": 5, "helpfulness": 5, "tone": 5, "pass": True, "reasons": "Fine."}
        )
    if schema == "pairwise_verdict":
        return json.dumps({"preferred": "tie", "reasons": "Fine."})
    question = body["messages"][-1]["content"]
    if question.startswith("Synthetic:"):
        return json.dumps(
            {"category": "payment", "priority": "low", "order_id": None, "summary": "Cards."}
        )
    v2 = "Follow these rules" in body["messages"][0]["content"]
    return f"Synthetic answer {'two' if v2 else 'one'} [kb-alpha]."


def transport(cost=0.0):
    def handler(request):
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "gen-synthetic",
                "model": body["model"],
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"content": answer(body)}}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": cost},
            },
        )

    return httpx.MockTransport(handler)


def record(store_dir, planned, plan_inputs, cost=0.0):
    """Record the given planned calls through the real client (synthetic replies)."""
    config = Config(models=plan_inputs.models, settings=Settings(api_key=SecretStr("synthetic-1")))
    with ModelClient(
        "record", CassetteStore(store_dir), config, transport(cost), limiter=NoWait()
    ) as client:
        for p in planned:
            client.complete(
                list(p.messages),
                role=getattr(plan_inputs.models, p.role),
                response_format=p.response_format,
                repeat=p.repeat,
                tag=p.tag,
            )


def refuse(request):
    raise AssertionError(f"no request expected: {request.url}")


NO_NETWORK = httpx.MockTransport(refuse)


# --- counts --------------------------------------------------------------------


def test_counts_by_function_version_and_repeat_before_any_recording(tmp_path):
    plan_inputs = inputs()
    store = CassetteStore(tmp_path)
    counts = count_plan(full_plan(plan_inputs, store), store, plan_inputs.models)
    rows = {row.label: row for row in counts.rows}
    assert list(rows) == ["rag v1", "rag v2", "triage v1", "rag v1 vs v2"]
    # rag: 3 cases x 2 repeats; judge: 2 judged cases on repeat 0 (judge_repeats: first).
    assert rows["rag v1"].system == 6 and rows["rag v1"].by_repeat == {0: 3, 1: 3}
    assert (rows["rag v1"].judge, rows["rag v1"].waiting) == (2, 2)
    assert rows["rag v1"].judge_by_repeat == {0: 2}
    assert rows["triage v1"].system == 2 and rows["triage v1"].judge == 0
    # pairwise: 2 judged cases x 2 orders.
    assert rows["rag v1 vs v2"].system == 0 and rows["rag v1 vs v2"].judge == 4
    assert (counts.system, counts.judge, counts.waiting) == (14, 8, 8)
    assert (counts.total, counts.recorded, counts.to_record) == (22, 0, 22)
    assert not counts.exact
    assert counts.free_to_record == 22


def test_judge_calls_become_exact_once_the_answers_are_recorded(tmp_path):
    plan_inputs = inputs()
    store = CassetteStore(tmp_path)
    record(tmp_path, [p for p in full_plan(plan_inputs, store) if p.role == "system"], plan_inputs)
    store = CassetteStore(tmp_path)
    counts = count_plan(full_plan(plan_inputs, store), store, plan_inputs.models)
    assert counts.waiting == 0 and counts.exact
    assert (counts.system, counts.judge) == (14, 8)
    assert (counts.recorded, counts.to_record) == (14, 8)
    assert {row.label: row.recorded for row in counts.rows}["rag v1"] == 6


def test_identical_answers_shrink_the_plan_once_they_are_recorded(tmp_path):
    plan_inputs = inputs()
    planned = [p for p in full_plan(plan_inputs, None) if p.role == "system"]

    def same_answer(store_dir):
        config = Config(models=plan_inputs.models, settings=Settings(api_key=SecretStr("s-1")))

        def handler(request):
            body = json.loads(request.content)
            text = answer(body).replace("two", "one")  # both versions write the same
            return httpx.Response(
                200,
                json={
                    "model": body["model"],
                    "choices": [{"finish_reason": "stop", "message": {"content": text}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
                },
            )

        with ModelClient(
            "record",
            CassetteStore(store_dir),
            config,
            httpx.MockTransport(handler),
            limiter=NoWait(),
        ) as client:
            for p in planned:
                client.complete(
                    list(p.messages), role=plan_inputs.models.system, repeat=p.repeat, tag=p.tag
                )

    same_answer(tmp_path)
    store = CassetteStore(tmp_path)
    counts = count_plan(full_plan(plan_inputs, store), store, plan_inputs.models)
    # Gradings of the same answer share one key across versions (2 cases),
    # and identical pairs get no pairwise question (4 calls gone).
    assert counts.judge == 2
    assert "rag v1 vs v2" not in [row.label for row in counts.rows]


def test_a_stability_subset_and_judging_every_repeat_change_the_counts():
    subset = count_plan(full_plan(inputs(stability_cases=("rag-001",)), None), None, models())
    # rag: rag-001 twice, the others once, per version; triage once.
    assert subset.system == 2 * (2 + 1 + 1) + 1
    every = count_plan(full_plan(inputs(judge_repeats="all"), None), None, models())
    assert every.judge == 2 * 2 * 2 + 4  # gradings on both repeats + pairwise


def test_counts_only_free_model_calls_against_the_free_quota():
    plan_inputs = inputs(judge="synthetic/judge-paid")
    counts = count_plan(full_plan(plan_inputs, None), None, plan_inputs.models)
    assert counts.free_to_record == counts.system == 14


# --- the printed plan -------------------------------------------------------------


def test_the_plan_lines_name_the_upper_bound(tmp_path):
    plan_inputs = inputs()
    counts = count_plan(full_plan(plan_inputs, None), None, plan_inputs.models)
    lines = plan_lines(counts, plan_inputs.models)
    assert lines[0] == "call plan: repeats 2, judge_repeats first, stability_cases: every case"
    assert lines[1] == (
        "  rag v1: 6 system calls (repeat 0: 3, repeat 1: 3) "
        "+ up to 2 judge calls (repeat 0: 2) = up to 8; recorded 0"
    )
    assert "  triage v1: 2 system calls (repeat 0: 1, repeat 1: 1) = 2; recorded 0" in lines
    assert "  rag v1 vs v2: up to 4 pairwise judge calls (repeat 0: 4) = up to 4; recorded 0" in (
        lines
    )
    assert (
        "total: 14 system calls + up to 8 judge calls = up to 22 distinct calls; "
        "recorded 0, to record up to 22"
    ) in lines
    assert any("planned once the answers they grade are recorded" in line for line in lines)


def test_the_plan_line_names_a_subset():
    plan_inputs = inputs(stability_cases=("rag-001", "tri-001"))
    counts = count_plan(full_plan(plan_inputs, None), None, plan_inputs.models)
    assert plan_lines(counts, plan_inputs.models)[0] == (
        "call plan: repeats 2, judge_repeats first, stability_cases: rag-001, tri-001"
    )


def test_the_quota_lines_give_days_with_the_source_and_date():
    plan_inputs = inputs()
    counts = count_plan(full_plan(plan_inputs, None), None, plan_inputs.models)
    lines = quota_lines(counts, plan_inputs.models)
    assert lines[0].startswith(
        "free-model limits (OpenRouter limits documentation, "
        "https://openrouter.ai/docs/api-reference/limits, checked 2026-10-04; "
        "published facts, not read live): 20 requests per minute"
    )
    assert lines[1] == (
        "free-model calls to record: up to 22: 1 day at 50 a day, 1 day at 1000 a day; "
        "about 2 minutes of calls at rpm 18 if the daily quota allowed"
    )


def test_the_quota_lines_without_free_calls_say_the_limits_do_not_apply():
    plan_inputs = inputs(system="synthetic/system-paid", judge="synthetic/judge-paid")
    counts = count_plan(full_plan(plan_inputs, None), None, plan_inputs.models)
    assert quota_lines(counts, plan_inputs.models)[1] == (
        "free-model calls to record: none: the config names no :free model, "
        "so the free-model limits do not apply"
    )


def test_the_quota_lines_when_every_free_call_is_recorded():
    plan_inputs = inputs()
    counts = count_plan(full_plan(plan_inputs, None), None, plan_inputs.models)
    done = dataclasses.replace(counts, free_to_record=0)
    assert quota_lines(done, plan_inputs.models)[1] == "free-model calls to record: none"


def test_one_minute_is_singular():
    plan_inputs = inputs()
    counts = count_plan(full_plan(plan_inputs, None), None, plan_inputs.models)
    few = dataclasses.replace(counts, free_to_record=10)
    assert quota_lines(few, plan_inputs.models)[1].endswith(
        "about 1 minute of calls at rpm 18 if the daily quota allowed"
    )


# --- cost -------------------------------------------------------------------------


def test_free_models_estimate_to_zero_without_the_network():
    plan_inputs = inputs()
    plan = full_plan(plan_inputs, None)
    estimate = estimate_remaining_cost(plan, None, plan_inputs.models, transport=NO_NETWORK)
    assert estimate.usd == 0.0
    assert estimate.lines[0] == (
        "system (synthetic/system:free): $0.00 for 14 calls (a :free model id)"
    )
    assert estimate.lines[1] == (
        "judge (synthetic/judge:free): $0.00 for up to 8 calls (a :free model id)"
    )
    assert estimate.headline() == "estimated cost of the calls still to record: $0.00"


def prices_handler(prompt="0.000001", completion="0.000002"):
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "synthetic/system-paid",
                        "pricing": {"prompt": prompt, "completion": completion},
                    },
                    {
                        "id": "synthetic/judge-paid",
                        "pricing": {"prompt": prompt, "completion": completion},
                    },
                ]
            },
        )

    return httpx.MockTransport(handler), seen


def test_paid_models_without_history_use_the_published_prices():
    from llmeval.pricing import PlannedCall, estimate_prompt_tokens

    plan_inputs = inputs(system="synthetic/system-paid", judge="synthetic/judge-paid")
    plan = full_plan(plan_inputs, None)
    mock, seen = prices_handler()
    estimate = estimate_remaining_cost(
        plan, None, plan_inputs.models, transport=mock, now=lambda: NOW
    )
    assert seen == ["/api/v1/models"]  # one price lookup for both models
    # Upper bound: every call at its full budget; a judge prompt waiting for
    # its answers gets room for each answer at the system's max_tokens.
    system_max = plan_inputs.models.system.max_tokens
    expected = 0.0
    for p in plan:
        tokens = estimate_prompt_tokens(p.messages)
        if p.key is None:
            tokens += len(p.grades) * system_max
        call = PlannedCall.for_role(getattr(plan_inputs.models, p.role), tokens)
        expected += call.prompt_tokens * 0.000001 + call.max_completion_tokens * 0.000002
    assert estimate.usd == pytest.approx(expected)
    assert "from the published prices (GET /api/v1/models)" in estimate.lines[0]


def test_recorded_costs_are_the_history_for_the_rest(tmp_path):
    plan_inputs = inputs(system="synthetic/system-paid", judge="synthetic/judge-paid")
    store = CassetteStore(tmp_path)
    system_calls = [p for p in full_plan(plan_inputs, store) if p.role == "system"]
    record(tmp_path, system_calls[:4], plan_inputs, cost=0.0005)
    store = CassetteStore(tmp_path)
    plan = full_plan(plan_inputs, store)
    mock, seen = prices_handler()
    estimate = estimate_remaining_cost(plan, store, plan_inputs.models, transport=mock)
    # 10 system calls left at the highest recorded usage.cost; the judge has
    # no history yet, so its 8 calls are priced from the published list.
    assert estimate.lines[0] == (
        "system (synthetic/system-paid): $0.0050 for 10 calls at $0.000500 each, "
        "the highest usage.cost among 4 recorded calls"
    )
    assert seen == ["/api/v1/models"]
    assert estimate.spent_usd == 0.002 and estimate.spent_unknown == 0
    assert "recorded so far: $0.002000 over 4 calls" in estimate.lines


def test_a_failed_price_lookup_is_an_error_not_a_zero():
    plan_inputs = inputs(system="synthetic/system-paid")
    mock = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []}))
    with pytest.raises(PricingError, match="synthetic/system-paid"):
        estimate_remaining_cost(
            full_plan(plan_inputs, None), None, plan_inputs.models, transport=mock
        )


def test_the_budget_guard_refuses_an_estimate_above_the_limit():
    plan_inputs = inputs(system="synthetic/system-paid", judge="synthetic/judge-paid")
    mock, _ = prices_handler(prompt="0.01", completion="0.01")
    estimate = estimate_remaining_cost(
        full_plan(plan_inputs, None), None, plan_inputs.models, transport=mock
    )
    with pytest.raises(BudgetExceeded, match="exceeds MAX_RUN_COST_USD=\\$1.00"):
        check_budget(estimate.usd, 1.00)


def test_unknown_recorded_costs_are_counted_not_zeroed(tmp_path):
    plan_inputs = inputs()
    planned = [p for p in full_plan(plan_inputs, None) if p.role == "system"][:2]
    config = Config(models=plan_inputs.models, settings=Settings(api_key=SecretStr("s-1")))

    def no_cost(request):
        body = json.loads(request.content)
        if request.url.path.endswith("/models"):
            return httpx.Response(503, json={"error": {"message": "synthetic outage"}})
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"finish_reason": "stop", "message": {"content": "Synthetic."}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    with ModelClient(
        "record", CassetteStore(tmp_path), config, httpx.MockTransport(no_cost), limiter=NoWait()
    ) as client:
        for p in planned:
            client.complete(
                list(p.messages), role=plan_inputs.models.system, repeat=p.repeat, tag=p.tag
            )
    store = CassetteStore(tmp_path)
    estimate = estimate_remaining_cost(
        full_plan(plan_inputs, store), store, plan_inputs.models, transport=NO_NETWORK
    )
    assert (estimate.spent_usd, estimate.spent_unknown) == (0.0, 2)
    assert (
        "recorded so far: $0.000000 over 0 calls with a known cost, plus 2 calls with unknown cost"
    ) in estimate.lines
