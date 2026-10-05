"""The labelling tool: what it shows, what it asks and what it saves.

Synthetic data: the dataset rows, the answers (recorded through the real
client with a mock transport), the labels and comments are made up for the
test. Every file lives in pytest's `tmp_path`; nothing is written to the
repository's `labels/` or `cassettes/`.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from app import assistant
from app.retrieval import Hit
from llmeval.cassettes import CassetteStore
from llmeval.config import load_models_config
from llmeval.labels import (
    LABEL_QUESTION,
    LABELER,
    HumanLabel,
    LabelItem,
    label_session,
    load_labels,
    prepare_items,
    render_intro,
    render_item,
    unlabelled,
)
from tests.unit.synthetic_labels import ANSWERS, ROWS, record_answers

NOW = datetime(2026, 1, 2, 9, 30, 5, tzinfo=UTC)
QUESTIONS = {row["id"]: row["question"] for row in ROWS}
# Words that would tell the labeller what the judge or the checks said, or
# which prompt version and category the answer has.
NOT_SHOWN = ("judge", "score", "groundedness", "helpfulness", "verdict", "rule", "check")


@pytest.fixture
def recording(tmp_path):
    return record_answers(tmp_path)


def items_of(recording, *refs):
    role = load_models_config(recording.config).system
    store = CassetteStore(recording.cassettes)
    return prepare_items(recording.sample(*refs), QUESTIONS, store, role)


def words(text: str) -> str:
    return " ".join(text.split())


def test_each_item_shows_the_documents_of_the_prompt_that_wrote_the_answer(recording):
    items, problems = items_of(recording)
    assert problems == []
    store = CassetteStore(recording.cassettes)
    for position, item in enumerate(items, start=1):
        recorded = store.get(item.item.answer_key)
        system_turn = recorded.request["messages"][0]["content"]
        assert assistant.format_documents(item.documents) in system_turn
        assert item.documents  # every synthetic question retrieves something
        assert item.answer == ANSWERS[(item.item.case, item.item.version)]
        assert item.question == QUESTIONS[item.item.case]
        assert item.position == position


def test_an_answer_whose_prompt_changed_is_not_shown(recording):
    sample = recording.sample()
    changed = sample.items[1].model_copy(update={"answer_key": "f" * 64})
    sample = sample.model_copy(update={"items": [sample.items[0], changed, sample.items[2]]})
    role = load_models_config(recording.config).system
    items, problems = prepare_items(sample, QUESTIONS, CassetteStore(recording.cassettes), role)
    assert [item.item.ref for item in items] == [sample.items[0].ref, sample.items[2].ref]
    assert problems == [
        "rag-001 v2 repeat 0: the prompt now differs from the one that wrote the answer "
        "(prompt, documents, question or model config changed); the sample is stale"
    ]


def test_an_answer_missing_from_the_cassettes_or_the_dataset_is_named(recording, tmp_path):
    role = load_models_config(recording.config).system
    empty = CassetteStore(tmp_path / "no-cassettes")
    items, problems = prepare_items(recording.sample(("rag-001", "v1")), QUESTIONS, empty, role)
    assert items == []
    assert problems == ["rag-001 v1 repeat 0: the answer is not in the cassettes"]
    items, problems = prepare_items(
        recording.sample(("rag-029", "v2")),
        {"rag-001": QUESTIONS["rag-001"]},
        CassetteStore(recording.cassettes),
        role,
    )
    assert problems == ["rag-029 v2 repeat 0: the case is not in the dataset"]


def test_an_item_shows_the_counter_question_documents_and_answer_and_nothing_else(recording):
    items, _ = items_of(recording)
    item = items[1]
    text = render_item(item, total=30, width=72)
    lines = text.splitlines()
    assert "Item 2 of 30" in lines
    assert all(len(line) <= 72 for line in lines)
    assert words(item.question) in words(text)
    assert words(item.answer) in words(text)
    for number, hit in enumerate(item.documents, start=1):
        title, _, body = hit.text.partition("\n\n")
        assert f"  [{number}] {hit.doc_id}: {title}" in lines
        assert words(body) in words(text)
    # The tool's own lines (the shown material is indented by four spaces or more).
    own = "\n".join(line for line in lines if not line.startswith("    ")).lower()
    for word in NOT_SHOWN:
        assert word not in own
    assert "v1" not in own and "v2" not in own and "answerable" not in own
    assert item.item.case not in text


def test_list_lines_keep_their_hanging_indent_when_wrapped():
    item = LabelItem(
        item=None,
        position=1,
        question="Synthetic question?",
        documents=(
            Hit(
                "kb-synthetic",
                1.0,
                "Synthetic\n\nThese items cannot be returned:\n- items cut to length for you, "
                "such as chain, cable and hose, and anything else that is long;\n- gift cards.",
            ),
        ),
        answer="",
    )
    lines = render_item(item, total=1, width=48).splitlines()
    at = lines.index("      - items cut to length for you, such as")
    assert lines[at + 1].startswith("        chain, cable")
    assert "    (empty answer)" in lines


def test_the_intro_asks_the_labelling_question_and_says_how_to_stop():
    text = render_intro(
        total=30,
        labelled=4,
        todo=25,
        labels_path=Path("labels/human.jsonl"),
        problems=["rag-001 v2 repeat 0: the answer is not in the cassettes"],
        width=72,
    )
    assert all(len(line) <= 72 for line in text.splitlines())
    assert words(LABEL_QUESTION) in words(text)
    assert LABEL_QUESTION == (
        "Would you send this answer to the customer as is? Pass if it is grounded in the "
        "shown documents, answers the question (or says honestly that the documents do not "
        "cover it), and is polite. Fail otherwise."
    )
    assert "Labelled so far: 4 of 30. To label now: 25." in words(text)
    assert "p pass, f fail, s skip for now, q quit" in words(text)
    assert "saved at once to labels/human.jsonl" in words(text)
    assert "Ctrl-C" in text
    assert (
        "To change a saved label, delete its line in labels/human.jsonl and run make label "
        "again." in words(text)
    )
    assert "1 sample answer cannot be shown" in words(text)
    assert "rag-001 v2 repeat 0: the answer is not in the cassettes" in words(text)
    for word in NOT_SHOWN:
        assert word not in text.lower()


class Script:
    """Answers the tool's prompts from a list; an exception in the list is raised."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.asked: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.asked.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def run_session(items, path, script):
    shown: list[str] = []
    outcome = label_session(
        items, path, total=len(items), ask=script, echo=shown.append, now=lambda: NOW, width=80
    )
    return outcome, "\n".join(shown)


def test_a_session_saves_pass_and_fail_with_the_comment_labeler_and_time(recording, tmp_path):
    items, _ = items_of(recording)
    path = tmp_path / "labels" / "human.jsonl"
    script = Script("p", "Grounded.", " F ", "", "s")
    outcome, shown = run_session(items, path, script)
    assert (outcome.saved, outcome.reason) == (2, "done")
    labels = load_labels(path)
    assert labels == [
        HumanLabel(
            case="rag-001",
            version="v1",
            repeat=0,
            answer_key=recording.keys[("rag-001", "v1")],
            label="pass",
            comment="Grounded.",
            labeler=LABELER,
            labeled_at=NOW,
        ),
        HumanLabel(
            case="rag-001",
            version="v2",
            repeat=0,
            answer_key=recording.keys[("rag-001", "v2")],
            label="fail",
            comment="",
            labeler=LABELER,
            labeled_at=NOW,
        ),
    ]
    assert "Item 3 of 3" in shown  # the skipped item was shown, and nothing saved
    assert "Skipped" in shown


def test_the_comment_prompt_names_the_choice_and_offers_a_way_back(recording, tmp_path):
    items, _ = items_of(recording, ("rag-001", "v1"))
    script = Script("p", "b", "f", "No source for the fee.")
    outcome, shown = run_session(items, tmp_path / "human.jsonl", script)
    assert outcome.saved == 1
    saved = load_labels(tmp_path / "human.jsonl")
    assert [(label.label, label.comment) for label in saved] == [("fail", "No source for the fee.")]
    assert script.asked[1] == "Comment for PASS (optional; Enter saves, b goes back)"
    assert script.asked[3] == "Comment for FAIL (optional; Enter saves, b goes back)"
    assert script.asked[0] == script.asked[2]  # b asks for the same item's label again


@pytest.mark.parametrize("key", ["s", "q", "p", "f", "S", " Q ", "skip", "pass", "Fail"])
def test_a_label_key_typed_as_the_comment_is_refused(recording, tmp_path, key):
    items, _ = items_of(recording, ("rag-001", "v1"))
    outcome, shown = run_session(items, tmp_path / "human.jsonl", Script("p", key, ""))
    assert outcome.saved == 1
    assert load_labels(tmp_path / "human.jsonl")[0].comment == ""
    assert shown.count("That looks like a label key: type b to go back, or write a comment.") == 1


@pytest.mark.parametrize("typed", ["Grounde\x1b[Dd", "tab\there", "rub\x7fout", "bell\x07"])
def test_a_comment_with_control_characters_is_asked_again(recording, tmp_path, typed):
    items, _ = items_of(recording, ("rag-001", "v1"))
    script = Script("f", typed, "Invents a fee.")
    outcome, shown = run_session(items, tmp_path / "human.jsonl", script)
    assert outcome.saved == 1
    assert load_labels(tmp_path / "human.jsonl")[0].comment == "Invents a fee."
    assert (
        shown.count(
            "The comment has a control character (an arrow or another special key?): type it again."
        )
        == 1
    )


def test_an_unknown_key_is_asked_again(recording, tmp_path):
    items, _ = items_of(recording, ("rag-001", "v1"))
    script = Script("x", "", "pass", "f", "Invents a fee.")
    outcome, shown = run_session(items, tmp_path / "human.jsonl", script)
    assert outcome.saved == 1
    assert load_labels(tmp_path / "human.jsonl")[0].label == "fail"
    assert shown.count("Type p, f, s or q.") == 3


def test_quit_stops_and_keeps_what_was_saved(recording, tmp_path):
    items, _ = items_of(recording)
    outcome, _ = run_session(items, tmp_path / "human.jsonl", Script("p", "", "q"))
    assert (outcome.saved, outcome.reason) == (1, "quit")
    assert len(load_labels(tmp_path / "human.jsonl")) == 1


@pytest.mark.parametrize(
    "replies",
    [
        ("p", "", KeyboardInterrupt()),  # at the next label
        ("p", "", "f", KeyboardInterrupt()),  # at the comment: that item is not saved
        ("p", "", EOFError()),
    ],
)
def test_ctrl_c_keeps_every_complete_label_and_writes_no_partial_one(recording, tmp_path, replies):
    items, _ = items_of(recording)
    path = tmp_path / "human.jsonl"
    outcome, _ = run_session(items, path, Script(*replies))
    assert (outcome.saved, outcome.reason) == (1, "interrupted")
    assert path.read_text(encoding="utf-8").count("\n") == 1
    assert [label.case for label in load_labels(path)] == ["rag-001"]


def test_labelled_answers_are_not_shown_again_unless_the_answer_changed(recording):
    items, _ = items_of(recording)

    def label_of(item, key):
        return HumanLabel(
            case=item.item.case,
            version=item.item.version,
            repeat=0,
            answer_key=key,
            label="pass",
            comment="",
            labeler=LABELER,
            labeled_at=NOW,
        )

    current = label_of(items[0], items[0].item.answer_key)
    stale = label_of(items[1], "e" * 64)
    assert unlabelled(items, [current, stale]) == items[1:]
