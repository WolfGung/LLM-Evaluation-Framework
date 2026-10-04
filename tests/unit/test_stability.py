"""The stability layer: do repeated runs of a case reach the same verdicts?

Synthetic data: every case record, check and reply below is made up for the
test. Nothing is written to `results/` or `cassettes/`.
"""

import json
from datetime import UTC, datetime

from llmeval.baseline import build_baseline
from llmeval.cassettes import RunManifest
from llmeval.results import (
    CallRecord,
    CaseRecord,
    CheckRecord,
    FunctionResults,
    RunRecord,
    summarise,
)
from llmeval.stability import STABILITY_LAYERS, stability

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


def check(layer, name, passed):
    return CheckRecord(layer=layer, name=name, passed=passed, detail="synthetic")


def case(case_id, *runs, category="answerable"):
    """A case record; each run is (output, [checks])."""
    return CaseRecord(
        id=case_id,
        category=category,
        input="Synthetic question?",
        expected={},
        runs=[
            RunRecord(repeat=repeat, output=output, call=CALL, checks=checks)
            for repeat, (output, checks) in enumerate(runs)
        ],
    )


def rag_run(*, cites=True, facts=True, leak=True, recall=True, judge=True):
    return (
        "Synthetic answer.",
        [
            check("retrieval", "retrieval_recall", recall),
            check("deterministic", "cites_retrieved", cites),
            check("reference", "required_facts", facts),
            check("safety", "no_trap_values", leak),
            check("judge", "groundedness", judge),
        ],
    )


def triage_reply(category, priority="normal"):
    reply = json.dumps(
        {"category": category, "priority": priority, "order_id": None, "summary": "Synthetic."}
    )
    checks = [
        check("deterministic", "valid_json", True),
        check("reference", "category_match", category == "order_status"),
        check("reference", "priority_match", priority == "normal"),
    ]
    return reply, checks


def test_the_layers_are_the_rule_based_ones():
    # Retrieval does not depend on the model; the judge is a second model.
    assert STABILITY_LAYERS == ("deterministic", "reference", "safety")


def test_repeats_with_the_same_verdicts_are_stable():
    result = stability("rag", [case("rag-001", rag_run(), rag_run(), rag_run())])
    assert result.repeated == 1
    assert result.stable == 1
    assert result.stable_share == 1.0
    assert result.unstable == []


def test_a_failing_check_that_fails_on_every_repeat_is_still_stable():
    runs = [rag_run(facts=False)] * 3
    assert stability("rag", [case("rag-001", *runs)]).stable == 1


def test_a_flipped_check_makes_the_case_unstable_and_is_named():
    flaky = case("rag-002", rag_run(), rag_run(facts=False), rag_run(), category="multi_doc")
    result = stability("rag", [case("rag-001", rag_run(), rag_run(), rag_run()), flaky])
    assert (result.repeated, result.stable, result.stable_share) == (2, 1, 0.5)
    (unstable,) = result.unstable
    assert unstable.id == "rag-002" and unstable.category == "multi_doc"
    assert unstable.checks == {"reference/required_facts": [True, False, True]}
    assert unstable.labels == {}


def test_deterministic_and_safety_flips_count():
    result = stability("rag", [case("rag-001", rag_run(cites=False, leak=False), rag_run())])
    assert result.unstable[0].checks == {
        "deterministic/cites_retrieved": [False, True],
        "safety/no_trap_values": [False, True],
    }


def test_retrieval_and_judge_flips_do_not_count():
    result = stability("rag", [case("rag-001", rag_run(recall=False, judge=False), rag_run())])
    assert result.stable == 1 and result.unstable == []


def test_a_check_missing_on_one_repeat_is_a_flip():
    output, checks = rag_run()
    shorter = (output, checks[:2])
    result = stability("rag", [case("rag-001", rag_run(), shorter)])
    assert result.unstable[0].checks == {
        "reference/required_facts": [True, None],
        "safety/no_trap_values": [True, None],
    }


def test_triage_needs_the_same_category_and_priority():
    # Both wrong categories fail category_match, so the check verdicts agree;
    # the predicted category still changed, so the case is unstable.
    runs = [triage_reply("payment"), triage_reply("shipping"), triage_reply("payment")]
    result = stability("triage", [case("tri-001", *runs, category="order_status")])
    (unstable,) = result.unstable
    assert unstable.checks == {}
    assert unstable.labels == {"category": ["payment", "shipping", "payment"]}


def test_triage_with_the_same_labels_is_stable():
    runs = [triage_reply("payment", "high")] * 3
    assert stability("triage", [case("tri-001", *runs)]).stable == 1


def test_an_unreadable_reply_is_an_invalid_label():
    _, checks = triage_reply("order_status")
    runs = [triage_reply("order_status"), ("not json", checks)]
    unstable = stability("triage", [case("tri-001", *runs)]).unstable[0]
    assert unstable.labels == {
        "category": ["order_status", "invalid"],
        "priority": ["normal", "invalid"],
    }


def test_cases_run_once_are_left_out():
    result = stability("rag", [case("rag-001", rag_run()), case("rag-002", rag_run(), rag_run())])
    assert result.repeated == 1


def test_no_repeated_case_means_no_stability_layer():
    assert stability("rag", [case("rag-001", rag_run())]) is None
    assert stability("rag", []) is None


def test_the_summary_and_the_baseline_carry_the_stable_share():
    cases = [
        case("rag-001", rag_run(), rag_run()),
        case("rag-002", rag_run(), rag_run(cites=False)),
        case("rag-003", rag_run()),
    ]
    summary = summarise("rag", cases)
    assert summary.stability.stable_share == 0.5
    assert [u.id for u in summary.stability.unstable] == ["rag-002"]
    results = FunctionResults(
        function="rag",
        version="v1",
        mode="replay",
        model="synthetic/system:free",
        prompt_sha256="p" * 64,
        dataset="rag.jsonl",
        dataset_sha256="0" * 64,
        repeats=2,
        summary=summary,
        cases=cases,
    )
    manifest = RunManifest(
        models={"system": "synthetic/system:free", "judge": "synthetic/judge:free"},
        prompt_versions={"rag": ("v1",)},
        repeats=2,
        datasets={"rag.jsonl": "0" * 64},
        recorded_from=TIME,
        recorded_to=TIME,
        planned_calls=5,
        recorded_calls=5,
        judge_repeats="first",
    )
    baseline = build_baseline([results], manifest)
    assert baseline.functions["rag"]["v1"].metrics.stable_share == 0.5


def test_a_summary_without_repeats_has_no_stability_layer():
    assert summarise("rag", [case("rag-001", rag_run())]).stability is None
