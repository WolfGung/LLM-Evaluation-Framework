"""The labels file: one complete JSON line per labelled answer.

Synthetic data: every label, comment and cassette key below is made up for
the test. Each test writes its own file in pytest's `tmp_path`; nothing is
written to the repository's `labels/`.
"""

import json
import os
from datetime import UTC, datetime, timedelta, timezone

import pytest

from llmeval.labels import LABELER, HumanLabel, LabelError, append_label, load_labels

TIME = datetime(2026, 1, 2, 9, 30, 5, tzinfo=UTC)


def label(case="rag-001", version="v1", value="pass", comment="", key="a" * 64, **extra):
    fields = {
        "case": case,
        "version": version,
        "repeat": 0,
        "answer_key": key,
        "label": value,
        "comment": comment,
        "labeler": LABELER,
        "labeled_at": TIME,
    }
    return HumanLabel(**{**fields, **extra})


def test_appended_labels_are_one_json_line_each_and_load_back(tmp_path):
    path = tmp_path / "labels" / "human.jsonl"
    first = label(comment="Grounded and polite.")
    second = label(case="rag-002", value="fail", comment="Invents a fee — «€5».", key="b" * 64)
    append_label(path, first)
    append_label(path, second)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    assert len(lines) == 2 and all(line.endswith("}\n") for line in lines)
    assert list(json.loads(lines[0])) == [
        "case",
        "version",
        "repeat",
        "answer_key",
        "label",
        "comment",
        "labeler",
        "labeled_at",
    ]
    assert json.loads(lines[1])["labeled_at"] == "2026-01-02T09:30:05Z"
    assert json.loads(lines[1])["labeler"] == "Pavel Zhukov Atum"
    assert load_labels(path) == [first, second]


def test_no_labels_file_means_no_labels(tmp_path):
    assert load_labels(tmp_path / "human.jsonl") == []


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json}\n", "line 1: not a valid label"),
        ("\n", "line 1 is empty"),
        (label().model_dump_json(), "line 1 has no newline at the end"),
        (label().model_dump_json().replace(LABELER, "Someone Else") + "\n", "line 1: not a valid"),
        (label().model_dump_json().replace('"pass"', '"maybe"') + "\n", "line 1: not a valid"),
    ],
)
def test_a_broken_labels_file_is_refused_with_the_line(tmp_path, content, message):
    path = tmp_path / "human.jsonl"
    path.write_text(label(case="rag-009").model_dump_json() + "\n" + content, encoding="utf-8")
    with pytest.raises(LabelError, match=message.replace("line 1", "line 2")):
        load_labels(path)


@pytest.mark.parametrize(
    "when",
    [datetime(2026, 1, 2, 9, 30), datetime(2026, 1, 2, 9, 30, tzinfo=timezone(timedelta(hours=2)))],
)
def test_the_label_time_is_utc(when):
    with pytest.raises(ValueError, match="labeled_at must be in UTC"):
        label(labeled_at=when)


def test_two_labels_of_the_same_answer_are_refused(tmp_path):
    path = tmp_path / "human.jsonl"
    path.write_text(
        label().model_dump_json() + "\n" + label(value="fail").model_dump_json() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        LabelError, match="lines 1 and 2 label the same answer: rag-001 v1 repeat 0"
    ):
        load_labels(path)


def test_a_label_of_a_changed_answer_is_another_answer(tmp_path):
    path = tmp_path / "human.jsonl"
    append_label(path, label(key="a" * 64))
    append_label(path, label(key="c" * 64))
    assert [item.answer_key for item in load_labels(path)] == ["a" * 64, "c" * 64]


def test_nothing_is_appended_after_a_cut_off_line(tmp_path):
    path = tmp_path / "human.jsonl"
    path.write_text('{"case": "rag-0', encoding="utf-8")
    with pytest.raises(LabelError, match="does not end with a newline"):
        append_label(path, label())
    assert path.read_text(encoding="utf-8") == '{"case": "rag-0'


def test_an_interrupted_write_leaves_no_partial_line(tmp_path):
    path = tmp_path / "human.jsonl"
    append_label(path, label())
    before = path.read_bytes()

    def half_then_interrupt(fd, data):
        os.write(fd, data[: len(data) // 2])
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        append_label(path, label(case="rag-002"), write=half_then_interrupt)
    assert path.read_bytes() == before


def test_a_short_write_leaves_no_partial_line(tmp_path):
    path = tmp_path / "human.jsonl"

    def short(fd, data):
        return os.write(fd, data[:10])

    with pytest.raises(OSError, match="wrote 10 of"):
        append_label(path, label(), write=short)
    assert path.read_bytes() == b""
