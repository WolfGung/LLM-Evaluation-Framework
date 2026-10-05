"""The request key and the JSONL cassette store.

Synthetic data: every request body and cassette entry here is made up for the
test. Nothing in this module is a recorded model answer.
"""

import copy
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from llmeval.cassettes import CallTag, CassetteEntry, CassetteError, CassetteStore, request_key

BODY = {
    "model": "vendor-a/small:free",
    "messages": [
        {"role": "system", "content": "You are a test double."},
        {"role": "user", "content": "Synthetic question?"},
    ],
    "temperature": 0.2,
    "seed": 7,
    "max_tokens": 100,
    "response_format": {
        "type": "json_schema",
        "json_schema": {"name": "t", "strict": True, "schema": {"type": "object"}},
    },
    "provider": {"require_parameters": True},
}


def changed(path: tuple, value):
    body = copy.deepcopy(BODY)
    target = body
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = value
    return body


def make_entry(key="k" * 64, tag=None, repeat=0) -> CassetteEntry:
    moment = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    return CassetteEntry(
        key=key,
        repeat=repeat,
        tag=tag,
        request=BODY,
        response={"id": "gen-synthetic", "model": BODY["model"], "content": "Synthetic answer."},
        usage={"prompt_tokens": 10, "completion_tokens": 3},
        cost_usd=0.0,
        cost_source="provider",
        prices=None,
        latency_ms=12.5,
        requested_at=moment,
        recorded_at=moment,
    )


def test_key_is_deterministic():
    first = request_key(BODY, repeat=0)

    assert first == request_key(copy.deepcopy(BODY), repeat=0)
    assert len(first) == 64
    assert int(first, 16) >= 0


def test_key_ignores_dict_order():
    reordered = dict(reversed(list(copy.deepcopy(BODY).items())))
    reordered["messages"] = [dict(reversed(list(m.items()))) for m in reordered["messages"]]
    reordered["response_format"] = {
        "json_schema": {"schema": {"type": "object"}, "strict": True, "name": "t"},
        "type": "json_schema",
    }

    assert request_key(reordered, repeat=0) == request_key(BODY, repeat=0)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model",), "vendor-b/other:free"),
        (("messages", 1, "content"), "Another synthetic question?"),
        (("messages", 0, "role"), "user"),
        (("temperature",), 0.0),
        (("seed",), 8),
        (("max_tokens",), 101),
        (("response_format",), None),
        (("response_format", "json_schema", "strict"), False),
        (("provider",), None),
    ],
)
def test_key_changes_with_every_request_field(path, value):
    assert request_key(changed(path, value), repeat=0) != request_key(BODY, repeat=0)


def test_key_changes_with_repeat():
    keys = {request_key(BODY, repeat=n) for n in range(3)}

    assert len(keys) == 3


def test_key_covers_the_reasoning_option():
    with_reasoning = {**BODY, "reasoning": {"effort": "low"}}

    assert request_key(with_reasoning, repeat=0) != request_key(BODY, repeat=0)
    assert request_key(with_reasoning, repeat=0) != request_key(
        {**BODY, "reasoning": {"effort": "high"}}, repeat=0
    )


def test_key_rejects_an_unknown_request_field():
    # A new request parameter must be added to the key on purpose; a field the
    # key does not know would otherwise be sent but not recorded.
    with pytest.raises(ValueError, match="stream"):
        request_key({**BODY, "stream": True}, repeat=0)


def test_key_ignores_none_values():
    assert request_key({**BODY, "seed": None}, repeat=0) == request_key(
        {k: v for k, v in BODY.items() if k != "seed"}, repeat=0
    )


def test_key_keeps_none_inside_the_request():
    # In a JSON schema, "default": null is a real value, unlike an unset option.
    schema_with_null = copy.deepcopy(BODY)
    schema_with_null["response_format"]["json_schema"]["schema"]["default"] = None

    assert request_key(schema_with_null, repeat=0) != request_key(BODY, repeat=0)


def test_tag_label_and_file_name():
    tag = CallTag(function="rag", case="rag-001", version="v2")

    assert tag.label(repeat=1) == "rag-001/v2/1"
    assert tag.file_stem == "rag-v2"


@pytest.mark.parametrize("bad", ["../escape", "a/b", "", "sp ace"])
def test_tag_rejects_unsafe_file_names(bad):
    with pytest.raises(ValueError):
        CallTag(function=bad, case="c", version="v1")


def test_append_then_reload(tmp_path):
    tag = CallTag(function="rag", case="rag-001", version="v1")
    store = CassetteStore(tmp_path)
    store.append(make_entry(key="a" * 64, tag=tag))
    store.append(make_entry(key="b" * 64, tag=tag, repeat=1))

    reloaded = CassetteStore(tmp_path)

    assert len(reloaded) == 2
    assert "a" * 64 in reloaded
    assert reloaded.get("b" * 64).repeat == 1
    assert reloaded.get("c" * 64) is None
    lines = (tmp_path / "rag-v1.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["key"] for line in lines] == ["a" * 64, "b" * 64]


def test_untagged_calls_go_to_their_own_file(tmp_path):
    CassetteStore(tmp_path).append(make_entry())

    assert (tmp_path / "untagged.jsonl").exists()


def test_store_creates_its_directory_on_first_write(tmp_path):
    store = CassetteStore(tmp_path / "cassettes")

    assert len(store) == 0
    store.append(make_entry())

    assert (tmp_path / "cassettes" / "untagged.jsonl").exists()


def test_a_key_is_recorded_once(tmp_path):
    store = CassetteStore(tmp_path)
    store.append(make_entry())

    with pytest.raises(CassetteError, match="already recorded"):
        store.append(make_entry())


def test_a_broken_line_names_the_file_and_line(tmp_path):
    CassetteStore(tmp_path).append(make_entry())
    with (tmp_path / "untagged.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")

    with pytest.raises(CassetteError, match=r"untagged\.jsonl:2"):
        CassetteStore(tmp_path)


def test_entries_have_no_room_for_headers():
    data = make_entry().model_dump()
    data["headers"] = {"x": "y"}

    with pytest.raises(ValidationError):
        CassetteEntry.model_validate(data)


# --- a torn last line (a write interrupted by a crash or a full disk) ----------


def torn_store(tmp_path):
    """One complete entry, then the first half of a second one with no newline."""
    store = CassetteStore(tmp_path)
    store.append(make_entry(key="a" * 64))
    second = make_entry(key="b" * 64).model_dump_json()
    with (tmp_path / "untagged.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(second[: len(second) // 2])
    return tmp_path / "untagged.jsonl"


def test_a_torn_last_line_is_ignored_with_a_notice(tmp_path):
    torn_store(tmp_path)
    store = CassetteStore(tmp_path)
    assert len(store) == 1 and "a" * 64 in store and "b" * 64 not in store
    assert store.notices == [
        "ignored an unfinished last line in untagged.jsonl; that call will be recorded again"
    ]


def test_the_next_append_replaces_the_torn_tail(tmp_path):
    path = torn_store(tmp_path)
    store = CassetteStore(tmp_path)
    store.append(make_entry(key="b" * 64))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["key"] for line in lines] == ["a" * 64, "b" * 64]
    reloaded = CassetteStore(tmp_path)
    assert len(reloaded) == 2 and reloaded.notices == []


def test_a_complete_last_line_without_a_newline_is_kept_and_the_next_starts_fresh(tmp_path):
    path = tmp_path / "untagged.jsonl"
    path.write_text(make_entry(key="a" * 64).model_dump_json(), encoding="utf-8")
    store = CassetteStore(tmp_path)
    assert len(store) == 1 and store.notices == []
    store.append(make_entry(key="b" * 64))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["key"] for line in lines] == ["a" * 64, "b" * 64]


def test_a_broken_last_line_that_ends_with_a_newline_still_fails(tmp_path):
    path = torn_store(tmp_path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(CassetteError, match=r"untagged\.jsonl:2"):
        CassetteStore(tmp_path)


def test_a_broken_line_before_the_last_still_fails(tmp_path):
    path = tmp_path / "untagged.jsonl"
    complete = make_entry(key="a" * 64).model_dump_json()
    path.write_text("{not json\n" + complete, encoding="utf-8")
    with pytest.raises(CassetteError, match=r"untagged\.jsonl:1"):
        CassetteStore(tmp_path)


def test_unfinished_lines_can_be_cut_at_once(tmp_path):
    path = torn_store(tmp_path)
    CassetteStore(tmp_path).cut_unfinished_lines()
    assert path.read_text(encoding="utf-8").endswith("\n")
    reloaded = CassetteStore(tmp_path)
    assert len(reloaded) == 1 and reloaded.notices == []
