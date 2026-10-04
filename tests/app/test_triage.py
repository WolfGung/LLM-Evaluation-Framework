"""Ticket triage: the TriageResult schema, honest parsing, both output paths.

Uses the synthetic fake model from `fakes.py`; every reply below is a made-up
string written for the test.
"""

import json

import pytest
from pydantic import ValidationError

from app.prompting import load_prompt
from app.triage import (
    TRIAGE_FUNCTION,
    Category,
    Priority,
    TriageAnswer,
    TriageError,
    TriageResult,
    parse_triage,
    prepare,
    response_format_for,
    triage,
    triage_schema,
)
from llmeval.cassettes import CallTag
from llmeval.client import MissingRecording
from tests.app.fakes import STRUCTURED, UNSTRUCTURED, FakeModel

TICKET = "My order TS-104233 has not arrived and it is a week late."
GOOD = {
    "category": "shipping",
    "priority": "high",
    "order_id": "TS-104233",
    "summary": "Order TS-104233 is a week late.",
}
GOOD_JSON = json.dumps(GOOD)


def test_categories_and_priorities_are_fixed():
    assert [c.value for c in Category] == [
        "shipping",
        "returns",
        "payment",
        "warranty",
        "order_status",
        "product_question",
        "other",
    ]
    assert [p.value for p in Priority] == ["low", "normal", "high", "urgent"]


def test_a_valid_result():
    result = TriageResult(**GOOD)
    assert result.category is Category.SHIPPING
    assert result.priority is Priority.HIGH


@pytest.mark.parametrize("order_id", ["TS-000000", "TS-999999", None])
def test_order_id_accepts_the_shop_format_or_null(order_id):
    assert TriageResult(**{**GOOD, "order_id": order_id}).order_id == order_id


@pytest.mark.parametrize(
    "order_id", ["ts-104233", "TS-10423", "TS-1042334", " TS-104233", "104233"]
)
def test_order_id_refuses_anything_else(order_id):
    with pytest.raises(ValidationError):
        TriageResult(**{**GOOD, "order_id": order_id})


def test_order_id_must_be_present_even_when_null():
    without = {k: v for k, v in GOOD.items() if k != "order_id"}
    with pytest.raises(ValidationError):
        TriageResult(**without)


def test_summary_is_1_to_200_characters():
    assert TriageResult(**{**GOOD, "summary": "x" * 200})
    for summary in ("", "x" * 201):
        with pytest.raises(ValidationError):
            TriageResult(**{**GOOD, "summary": summary})


def test_extra_fields_are_refused():
    with pytest.raises(ValidationError):
        TriageResult(**GOOD, confidence=0.9)


def test_the_schema_requires_every_field_and_forbids_extras():
    schema = triage_schema()
    assert sorted(schema["required"]) == ["category", "order_id", "priority", "summary"]
    assert schema["additionalProperties"] is False


def test_parse_accepts_a_plain_json_object():
    assert parse_triage(GOOD_JSON) == TriageResult(**GOOD)


def test_parse_accepts_surrounding_whitespace():
    assert parse_triage(f"\n  {GOOD_JSON}  \n") == TriageResult(**GOOD)


@pytest.mark.parametrize(
    "fenced",
    [
        f"```json\n{GOOD_JSON}\n```",
        f"  ```json\n{GOOD_JSON}```\n",
        f"```json  \n{GOOD_JSON}\n  ```",
    ],
)
def test_parse_accepts_one_fenced_json_block(fenced):
    assert parse_triage(fenced) == TriageResult(**GOOD)


@pytest.mark.parametrize(
    "raw",
    [
        f"Here is the result: {GOOD_JSON}",
        f"{GOOD_JSON}\nHope this helps!",
        f"```\n{GOOD_JSON}\n```",
        f"```JSON\n{GOOD_JSON}\n```",
        f"Sure.\n```json\n{GOOD_JSON}\n```",
        "{'category': 'shipping'}",
        "{",
    ],
)
def test_parse_refuses_anything_more_lenient_as_invalid_json(raw):
    with pytest.raises(TriageError) as caught:
        parse_triage(raw)
    assert caught.value.kind == "invalid_json"
    assert caught.value.raw == raw


@pytest.mark.parametrize(
    "value",
    [
        {**GOOD, "category": "Shipping"},
        {**GOOD, "category": "delivery"},
        {**GOOD, "priority": 3},
        {**GOOD, "order_id": 104233},
        {**GOOD, "order_id": "TS-10423"},
        {**GOOD, "summary": "x" * 201},
        {**GOOD, "extra": True},
        {k: v for k, v in GOOD.items() if k != "summary"},
        [GOOD],
        "shipping",
    ],
)
def test_parse_refuses_json_that_breaks_the_schema(value):
    raw = json.dumps(value)
    with pytest.raises(TriageError) as caught:
        parse_triage(raw)
    assert caught.value.kind == "invalid_schema"
    assert caught.value.raw == raw


@pytest.mark.parametrize("raw", ["", "   \n"])
def test_parse_calls_an_empty_reply_empty(raw):
    with pytest.raises(TriageError) as caught:
        parse_triage(raw)
    assert caught.value.kind == "empty"
    assert caught.value.raw == raw


def test_prepare_for_an_unstructured_role_puts_the_schema_in_the_prompt_only():
    messages, response_format = prepare(TICKET, "v1", UNSTRUCTURED)
    assert response_format is None
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == TICKET
    assert json.dumps(triage_schema(), indent=2) in messages[0]["content"]
    assert messages[0]["content"].startswith(load_prompt("triage", "v1").split("{{schema}}")[0])


def test_prepare_for_a_structured_role_also_sends_the_schema():
    messages, response_format = prepare(TICKET, "v2", STRUCTURED)
    assert response_format == {
        "type": "json_schema",
        "json_schema": {"name": "triage_result", "strict": True, "schema": triage_schema()},
    }
    # The prompt is the same on both paths; only the enforcement differs.
    assert messages == prepare(TICKET, "v2", UNSTRUCTURED)[0]


def test_response_format_follows_the_role():
    assert response_format_for(UNSTRUCTURED) is None
    assert response_format_for(STRUCTURED)["type"] == "json_schema"


@pytest.mark.parametrize("text", ["", "  "])
def test_a_blank_ticket_is_refused(text):
    with pytest.raises(ValueError, match="ticket"):
        prepare(text, "v1", UNSTRUCTURED)


def test_triage_on_the_unstructured_path():
    model = FakeModel(reply=GOOD_JSON)
    outcome = triage(model, UNSTRUCTURED, TICKET, "v1", repeat=1, case="tri-003")
    assert isinstance(outcome, TriageAnswer)
    assert outcome.result == TriageResult(**GOOD)
    assert outcome.call.content == GOOD_JSON
    [call] = model.calls
    assert "response_format" not in call["body"]
    assert "provider" not in call["body"]
    assert call["repeat"] == 1
    assert call["tag"] == CallTag(function=TRIAGE_FUNCTION, case="tri-003", version="v1")


def test_triage_on_the_structured_path_routes_to_supporting_endpoints_only():
    model = FakeModel(reply=f"```json\n{GOOD_JSON}\n```")
    outcome = triage(model, STRUCTURED, TICKET, "v2")
    assert outcome.result.order_id == "TS-104233"
    [call] = model.calls
    assert call["body"]["response_format"]["json_schema"]["schema"] == triage_schema()
    assert call["body"]["provider"] == {"require_parameters": True}
    assert call["tag"] == CallTag(function="triage", case="adhoc", version="v2")


def test_an_invalid_reply_raises_with_the_raw_text_and_the_call():
    model = FakeModel(reply='{"category": "shipping"}')
    with pytest.raises(TriageError) as caught:
        triage(model, UNSTRUCTURED, TICKET, "v2")
    assert caught.value.raw == '{"category": "shipping"}'
    assert caught.value.kind == "invalid_schema"
    assert caught.value.call is not None
    assert caught.value.call.content == caught.value.raw


def test_an_empty_completion_raises_with_the_empty_text_and_the_reason():
    model = FakeModel(reply="", finish_reason="length")
    with pytest.raises(TriageError, match="length") as caught:
        triage(model, UNSTRUCTURED, TICKET, "v1")
    assert caught.value.raw == ""
    assert caught.value.kind == "empty"
    assert caught.value.call.empty_reason == "length"


def test_a_missing_recording_propagates():
    model = FakeModel(error=MissingRecording("k" * 64, 0, None))
    with pytest.raises(MissingRecording):
        triage(model, UNSTRUCTURED, TICKET, "v1")
