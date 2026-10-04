"""Pairwise comparison: two prompt versions' answers, asked twice with the order swapped.

Synthetic data: every pairwise verdict is a made-up JSON string served by an
httpx MockTransport through the real `ModelClient` (see `synthetic_judge`).
Cassettes are written only into `tmp_path`.
"""

import pytest

from llmeval.cassettes import CassetteStore
from llmeval.checks.judge import (
    OUTCOMES,
    RUBRIC_PATH,
    Judge,
    PairwiseVerdict,
    combine,
    compare_messages,
    load_rubric,
    pairwise_format,
    pairwise_schema,
    parse_verdict,
)
from llmeval.config import Mode
from tests.unit.synthetic_judge import (
    DOCS,
    MODELS,
    QUESTION,
    ROOT,
    SyntheticTransport,
    make_client,
    preference,
    user_turn,
)

RUBRIC = load_rubric(ROOT / RUBRIC_PATH)
JUDGE = MODELS.judge
V1 = "You have 30 days [kb-alpha]."
V2 = "Returns are accepted within 30 days of delivery [kb-alpha]. Contact support for help."


def compare(tmp_path, *replies, mode=Mode.RECORD):
    handler = SyntheticTransport(list(replies))
    with make_client(tmp_path, mode, handler) as client:
        result = Judge(client, JUDGE, RUBRIC).compare(
            QUESTION, DOCS, V1, V2, case="rag-001", versions=("v1", "v2")
        )
    return result, handler


def test_the_pairwise_schema_is_strict_compatible():
    schema = pairwise_schema()
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["preferred", "reasons"]
    assert schema["properties"]["preferred"]["enum"] == ["A", "B", "tie"]
    assert pairwise_format() == {
        "type": "json_schema",
        "json_schema": {"name": "pairwise_verdict", "strict": True, "schema": schema},
    }


@pytest.mark.parametrize("bad", ['{"preferred": "a", "reasons": "x"}', '{"preferred": "C"}'])
def test_a_pairwise_verdict_is_validated(bad):
    with pytest.raises(ValueError, match="preferred"):
        parse_verdict(bad, PairwiseVerdict)


def test_the_prompt_shows_both_answers_by_letter():
    system, user = compare_messages(RUBRIC, QUESTION, DOCS, V1, V2)
    assert system == {"role": "system", "content": RUBRIC.text}
    text = user["content"]
    assert f'<answer id="A">\n{V1}\n</answer>' in text
    assert f'<answer id="B">\n{V2}\n</answer>' in text
    assert text.index("<question>") < text.index('<answer id="A">') < text.index('<answer id="B">')
    assert "Comparing two answers" in text


def test_the_judge_is_asked_twice_with_the_order_swapped(tmp_path):
    _, handler = compare(tmp_path, preference("B"), preference("A"))
    first, second = handler.bodies
    assert f'<answer id="A">\n{V1}\n</answer>' in user_turn(first)
    assert f'<answer id="B">\n{V2}\n</answer>' in user_turn(first)
    assert f'<answer id="A">\n{V2}\n</answer>' in user_turn(second)
    assert f'<answer id="B">\n{V1}\n</answer>' in user_turn(second)
    for body in handler.bodies:
        assert body["model"] == JUDGE.model
        assert body["response_format"] == pairwise_format()
    tags = sorted(entry.tag.case for entry in CassetteStore(tmp_path))
    assert tags == ["rag-001:A=v1", "rag-001:A=v2"]
    assert {entry.tag.version for entry in CassetteStore(tmp_path)} == {"v1-v2"}
    assert (tmp_path / "pairwise-v1-v2.jsonl").is_file()


@pytest.mark.parametrize(
    ("first", "second", "outcome"),
    [
        ("A", "B", "v1"),  # v1 preferred in both orders
        ("B", "A", "v2"),  # v2 preferred in both orders
        ("tie", "tie", "tie"),
        ("A", "A", "inconsistent"),  # the judge followed position A
        ("B", "B", "inconsistent"),  # the judge followed position B
        ("A", "tie", "inconsistent"),
        ("tie", "A", "inconsistent"),
    ],
)
def test_the_two_orders_give_one_of_four_outcomes(tmp_path, first, second, outcome):
    result, handler = compare(tmp_path, preference(first), preference(second))
    assert result.outcome == outcome
    assert len(handler.bodies) == 2
    assert [order.first for order in result.orders] == ["v1", "v2"]
    assert [order.preferred for order in result.orders] == [first, second]


def test_an_inconsistent_pair_keeps_both_answers_of_the_judge(tmp_path):
    result, _ = compare(tmp_path, preference("A", "A is clearer."), preference("A", "A again."))
    assert result.outcome == "inconsistent"
    assert [order.winner for order in result.orders] == ["v1", "v2"]
    assert [order.verdict.reasons for order in result.orders] == ["A is clearer.", "A again."]


@pytest.mark.parametrize(
    "replies",
    [
        ('{"preferred": "A"}', preference("A")),
        (preference("B"), ""),
        ("A is better.", "B is better."),
    ],
    ids=["no reasons first", "empty second", "prose both"],
)
def test_an_invalid_verdict_makes_the_pair_invalid_never_a_winner(tmp_path, replies):
    result, handler = compare(tmp_path, *replies)
    assert result.outcome == "invalid"
    assert len(handler.bodies) == 2  # neither order is asked again
    assert any(not order.valid for order in result.orders)
    assert len(CassetteStore(tmp_path)) == 2


def test_the_comparison_replays_without_the_network(tmp_path):
    recorded, _ = compare(tmp_path, preference("B"), preference("A"))
    replayed, handler = compare(tmp_path, mode=Mode.REPLAY)
    assert replayed.outcome == recorded.outcome == "v2"
    assert [o.call.key for o in replayed.orders] == [o.call.key for o in recorded.orders]
    assert handler.bodies == []


def test_identical_answers_are_their_own_outcome(tmp_path):
    # Both orders are the same request, so one recorded verdict serves both and
    # says nothing about position or preference.
    handler = SyntheticTransport([preference("A")])
    with make_client(tmp_path, Mode.RECORD, handler) as client:
        result = Judge(client, JUDGE, RUBRIC).compare(
            QUESTION, DOCS, V1, V1, case="rag-001", versions=("v1", "v2")
        )
    assert result.outcome == "identical"
    assert len(handler.bodies) == 1 and len(CassetteStore(tmp_path)) == 1
    assert result.orders[0].call.key == result.orders[1].call.key
    assert [order.preferred for order in result.orders] == ["A", "A"]


def test_combine_is_the_whole_rule():
    assert OUTCOMES == ("v1", "v2", "tie", "inconsistent", "identical", "invalid")
    for winner in ("v1", "v2", "tie"):
        assert combine(winner, winner) == winner
    assert combine("v1", "v2") == combine("v2", "v1") == "inconsistent"
    assert combine("tie", "v2") == combine("v1", "tie") == "inconsistent"
    assert combine(None, "v1") == combine("v2", None) == combine(None, None) == "invalid"
