"""The judge's verdict: schema, parsing, invalid verdicts, the call and the checks.

Synthetic data: every verdict is a made-up JSON string served by an httpx
MockTransport through the real `ModelClient` (see `synthetic_judge`).
Cassettes are written only into `tmp_path`; nothing goes to `cassettes/` or
`results/`.
"""

import json
from datetime import timedelta

import pytest

from llmeval.cassettes import CassetteStore, request_key
from llmeval.checks import judge as judge_module
from llmeval.checks.judge import (
    CRITERIA,
    PAIRWISE_DESCRIPTION,
    RUBRIC_PATH,
    VERDICT_DESCRIPTION,
    InvalidVerdict,
    Judge,
    JudgeVerdict,
    PairwiseVerdict,
    grade_messages,
    load_rubric,
    pairwise_format,
    parse_rubric,
    parse_verdict,
    verdict_format,
    verdict_schema,
)
from llmeval.client import build_role_request
from llmeval.config import Mode
from tests.unit.synthetic_judge import (
    ANSWER,
    DOCS,
    MODELS,
    NOW,
    QUESTION,
    ROOT,
    SyntheticTransport,
    make_client,
    user_turn,
    verdict,
)

RUBRIC = load_rubric(ROOT / RUBRIC_PATH)
JUDGE = MODELS.judge

# Keywords some providers' strict structured-output modes refuse.
STRICT_UNSUPPORTED = {
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "format",
}


def keywords(node):
    """Every key in a JSON schema, except the field names under `properties`."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "properties":
                for field in value.values():
                    yield from keywords(field)
            else:
                yield key
                yield from keywords(value)
    elif isinstance(node, list):
        for item in node:
            yield from keywords(item)


# --- the verdict --------------------------------------------------------------


def test_the_verdict_grades_exactly_the_rubric_criteria():
    scores = [name for name in JudgeVerdict.model_fields if name in CRITERIA]
    assert tuple(scores) == RUBRIC.criteria == CRITERIA


def test_the_schema_is_strict_compatible():
    schema = verdict_schema()
    assert not STRICT_UNSUPPORTED & set(keywords(schema))
    assert schema["additionalProperties"] is False
    assert list(schema["properties"]) == [*CRITERIA, "pass", "reasons"]
    assert schema["required"] == list(schema["properties"])
    for criterion in CRITERIA:
        assert schema["properties"][criterion]["enum"] == [1, 2, 3, 4, 5]
    assert schema["properties"]["pass"]["type"] == "boolean"


def test_the_response_format_asks_for_the_strict_schema():
    assert verdict_format() == {
        "type": "json_schema",
        "json_schema": {"name": "judge_verdict", "strict": True, "schema": verdict_schema()},
    }


def test_the_schema_describes_itself_for_the_model_not_with_the_docstring():
    # The schema is part of every judge request key; a docstring edit must not
    # change what the judge is sent.
    assert verdict_schema()["description"] == VERDICT_DESCRIPTION
    assert verdict_schema()["title"] == "JudgeVerdict"
    sent = json.dumps([verdict_format(), pairwise_format()])
    for model in (JudgeVerdict, PairwiseVerdict):
        assert model.__doc__.split("\n")[0] not in sent
    assert pairwise_format()["json_schema"]["schema"]["description"] == PAIRWISE_DESCRIPTION


def test_editing_a_docstring_does_not_change_the_judge_request(monkeypatch):
    def request_body():
        messages = grade_messages(RUBRIC, QUESTION, DOCS, ANSWER)
        return (
            build_role_request(messages, JUDGE, verdict_format()),
            build_role_request(messages, JUDGE, pairwise_format()),
        )

    before = request_body()

    class Renamed(JudgeVerdict):
        """An edited docstring that a developer wrote for other developers."""

    class RenamedPairwise(PairwiseVerdict):
        """Another edited docstring."""

    monkeypatch.setattr(judge_module, "JudgeVerdict", Renamed)
    monkeypatch.setattr(judge_module, "PairwiseVerdict", RenamedPairwise)
    assert request_body() == before


def test_a_valid_verdict_is_parsed():
    parsed = parse_verdict(verdict(5, 3, 4, True, "Supported by kb-alpha."))
    assert parsed.scores == {"groundedness": 5, "helpfulness": 3, "tone": 4}
    assert parsed.passed is True
    assert parsed.reasons == "Supported by kb-alpha."


def test_a_whole_reply_json_block_is_read_like_a_triage_reply():
    assert parse_verdict(f"```json\n{verdict()}\n```").scores["tone"] == 5


@pytest.mark.parametrize(
    ("raw", "kind", "detail"),
    [
        ("", "empty", "no text"),
        ("  \n", "empty", "no text"),
        ("Grounded: 5, helpful: 4", "invalid_json", "Expecting value"),
        (f"Here is my verdict: {verdict()}", "invalid_json", "Expecting value"),
        ("[5, 5, 5]", "invalid_schema", "an object"),
        (verdict(groundedness=6), "invalid_schema", "groundedness"),
        (verdict(helpfulness=0), "invalid_schema", "helpfulness"),
        (verdict(tone="4"), "invalid_schema", "whole number"),
        (verdict(tone=4.0), "invalid_schema", "whole number"),
        (verdict(tone=True), "invalid_schema", "whole number"),
        (verdict(passed="yes"), "invalid_schema", "pass"),
        (verdict(reasons="   "), "invalid_schema", "reasons"),
        (verdict(reasons=None), "invalid_schema", "reasons"),
        (json.dumps({"groundedness": 5, "helpfulness": 5, "tone": 5}), "invalid_schema", "pass"),
        (verdict().replace('"pass"', '"passed"'), "invalid_schema", "pass"),
        (verdict()[:-1] + ', "confidence": 0.9}', "invalid_schema", "confidence"),
    ],
    ids=[
        "empty",
        "whitespace",
        "prose",
        "prose around JSON",
        "an array",
        "a score above 5",
        "a score of 0",
        "a score as text",
        "a score as a float",
        "a score as true",
        "pass as text",
        "blank reasons",
        "no reasons",
        "no pass field",
        "pass under another name",
        "an extra field",
    ],
)
def test_an_invalid_verdict_says_what_is_wrong(raw, kind, detail):
    with pytest.raises(InvalidVerdict) as caught:
        parse_verdict(raw)
    assert caught.value.kind == kind
    assert detail in caught.value.detail


# --- the prompt ---------------------------------------------------------------


def test_the_prompt_holds_the_rubric_question_documents_and_answer():
    system, user = grade_messages(RUBRIC, QUESTION, DOCS, ANSWER)
    assert system == {"role": "system", "content": RUBRIC.text}
    assert user["role"] == "user"
    text = user["content"]
    assert f"<question>\n{QUESTION}\n</question>" in text
    assert '<document id="kb-alpha">' in text and "accepted within 30 days" in text
    assert '<document id="kb-beta">' in text
    assert f"<answer>\n{ANSWER}\n</answer>" in text
    assert text.index("<question>") < text.index("<documents>") < text.index("<answer>")


def test_no_documents_are_named_as_such():
    _, user = grade_messages(RUBRIC, QUESTION, (), ANSWER)
    assert "<documents>\nNo documents matched the question.\n</documents>" in user["content"]


def judge_key(question=QUESTION, docs=DOCS, answer=ANSWER, rubric=RUBRIC, repeat=0):
    messages = grade_messages(rubric, question, docs, answer)
    return request_key(build_role_request(messages, JUDGE, verdict_format()), repeat)


def test_the_request_key_is_deterministic_and_follows_every_input():
    base = judge_key()
    assert judge_key() == base
    other_rubric = parse_rubric(
        (ROOT / RUBRIC_PATH).read_text(encoding="utf-8") + "\nSynthetic extra line.\n",
        name="judge.md",
    )
    changed = [
        judge_key(question="How long do I have to send something back?"),
        judge_key(docs=DOCS[:1]),
        judge_key(answer="You have 30 days."),
        judge_key(rubric=other_rubric),
        judge_key(repeat=1),
    ]
    assert base not in changed
    assert len(set(changed)) == len(changed)


def test_the_key_does_not_depend_on_the_time_of_the_call(tmp_path):
    keys = []
    for offset in (0, 3):
        handler = SyntheticTransport(verdict())
        cassettes = tmp_path / f"run-{offset}"
        with make_client(cassettes, Mode.RECORD, handler, now=NOW + timedelta(days=offset)) as c:
            keys.append(Judge(c, JUDGE, RUBRIC).grade(QUESTION, DOCS, ANSWER).call.key)
    assert keys[0] == keys[1] == judge_key()


# --- the call -----------------------------------------------------------------


def test_the_judge_calls_the_judge_role_with_the_schema(tmp_path):
    handler = SyntheticTransport(verdict())
    with make_client(tmp_path, Mode.RECORD, handler) as client:
        Judge(client, JUDGE, RUBRIC).grade(QUESTION, DOCS, ANSWER, case="rag-007", version="v2")
    (body,) = handler.bodies
    assert body["model"] == JUDGE.model != MODELS.system.model
    assert body["temperature"] == 0.0
    assert body["seed"] == JUDGE.seed
    assert body["reasoning"] == {"effort": "low"}
    assert body["response_format"] == verdict_format()
    assert body["provider"] == {"require_parameters": True}
    assert body["messages"][0]["content"] == RUBRIC.text
    assert ANSWER in user_turn(body)
    (entry,) = CassetteStore(tmp_path)
    assert entry.tag.function == "judge"
    assert entry.tag.case == "rag-007:judge"
    assert entry.tag.version == "v2"
    assert (tmp_path / "judge-v2.jsonl").is_file()


def test_the_judge_refuses_a_role_without_structured_output(tmp_path):
    with (
        make_client(tmp_path, Mode.REPLAY) as client,
        pytest.raises(ValueError, match="structured_output"),
    ):
        Judge(client, MODELS.system, RUBRIC)


def test_a_valid_verdict_is_applied_with_the_rubric_rule(tmp_path):
    handler = SyntheticTransport(verdict(3, 5, 5, False, "One detail is not in kb-alpha."))
    with make_client(tmp_path, Mode.RECORD, handler) as client:
        judgement = Judge(client, JUDGE, RUBRIC).grade(QUESTION, DOCS, ANSWER)
    assert judgement.valid and judgement.error is None
    assert judgement.verdict.scores == {"groundedness": 3, "helpfulness": 5, "tone": 5}
    assert judgement.short_of == ("groundedness",)
    assert judgement.rule_pass is False
    assert judgement.agrees is True


def test_a_judge_that_misapplies_the_rule_is_noticed_not_trusted(tmp_path):
    # The judge says pass although groundedness 3 is below the minimum of 4.
    handler = SyntheticTransport(verdict(3, 5, 5, True))
    with make_client(tmp_path, Mode.RECORD, handler) as client:
        judgement = Judge(client, JUDGE, RUBRIC).grade(QUESTION, DOCS, ANSWER)
    assert judgement.rule_pass is False
    assert judgement.verdict.passed is True
    assert judgement.agrees is False


def test_an_invalid_verdict_is_recorded_counted_and_not_retried(tmp_path):
    handler = SyntheticTransport(verdict(groundedness=7))
    with make_client(tmp_path, Mode.RECORD, handler) as client:
        judgement = Judge(client, JUDGE, RUBRIC).grade(QUESTION, DOCS, ANSWER, case="rag-001")
    assert len(handler.bodies) == 1  # asked once, never again
    assert not judgement.valid
    assert judgement.verdict is None
    assert judgement.error == "invalid_schema"
    assert "groundedness" in judgement.detail
    assert judgement.raw == verdict(groundedness=7)
    assert judgement.rule_pass is None and judgement.agrees is None
    # The invalid reply is in the cassette like any other answer.
    (entry,) = CassetteStore(tmp_path)
    assert entry.response.content == verdict(groundedness=7)

    # Replay gives the same invalid judgement, without the network and without a retry.
    with make_client(tmp_path, Mode.REPLAY) as client:
        replayed = Judge(client, JUDGE, RUBRIC).grade(QUESTION, DOCS, ANSWER, case="rag-001")
    assert (replayed.error, replayed.raw, replayed.call.key) == (
        judgement.error,
        judgement.raw,
        judgement.call.key,
    )


def test_an_empty_verdict_is_an_invalid_judgement(tmp_path):
    # A reasoning judge can spend its whole budget thinking and write nothing.
    handler = SyntheticTransport("", finish_reason="length")
    with make_client(tmp_path, Mode.RECORD, handler) as client:
        judgement = Judge(client, JUDGE, RUBRIC).grade(QUESTION, DOCS, ANSWER)
    assert judgement.error == "empty"
    assert "length" in judgement.detail
    assert len(CassetteStore(tmp_path)) == 1


def test_a_valid_verdict_replays_without_the_network(tmp_path):
    with make_client(tmp_path, Mode.RECORD, SyntheticTransport(verdict(4, 4, 5))) as client:
        recorded = Judge(client, JUDGE, RUBRIC).grade(QUESTION, DOCS, ANSWER)
    with make_client(tmp_path, Mode.REPLAY) as client:
        replayed = Judge(client, JUDGE, RUBRIC).grade(QUESTION, DOCS, ANSWER)
    assert replayed.verdict == recorded.verdict
    assert replayed.call.key == recorded.call.key


# --- checks -------------------------------------------------------------------


def judged(tmp_path, raw):
    with make_client(tmp_path, Mode.RECORD, SyntheticTransport(raw)) as client:
        judge = Judge(client, JUDGE, RUBRIC)
        return judge, judge.grade(QUESTION, DOCS, ANSWER)


def test_a_passing_verdict_passes_every_judge_check(tmp_path):
    judge, judgement = judged(tmp_path, verdict(4, 3, 3, True))
    checks = judge.checks(judgement)
    assert [c.name for c in checks] == ["verdict_valid", *CRITERIA]
    assert all(c.passed for c in checks)
    assert checks[1].detail == "4/5 (pass needs 4 or more)"


def test_each_criterion_below_its_minimum_fails_its_own_check(tmp_path):
    judge, judgement = judged(tmp_path, verdict(2, 5, 2, False, "Invents a fee. Curt."))
    checks = {c.name: c for c in judge.checks(judgement)}
    assert checks["verdict_valid"].passed
    assert not checks["groundedness"].passed and not checks["tone"].passed
    assert checks["helpfulness"].passed
    assert checks["groundedness"].detail == (
        "2/5 (pass needs 4 or more); judge: Invents a fee. Curt."
    )


def test_the_layer_passes_exactly_when_the_rubric_rule_passes(tmp_path):
    for n, scores in enumerate([(4, 3, 3), (3, 3, 3), (5, 2, 5), (5, 5, 2), (5, 5, 5)]):
        judge, judgement = judged(tmp_path / str(n), verdict(*scores, passed=True))
        assert all(c.passed for c in judge.checks(judgement)) is judgement.rule_pass


def test_a_disagreeing_pass_is_reported_in_the_verdict_check(tmp_path):
    judge, judgement = judged(tmp_path, verdict(3, 5, 5, True))
    check = judge.checks(judgement)[0]
    assert check.passed
    assert "the judge's own pass (true) differs from the rubric rule (false)" in check.detail


def test_an_invalid_verdict_fails_one_check_that_names_the_judge(tmp_path):
    judge, judgement = judged(tmp_path, "Grounded, helpful, polite.")
    (check,) = judge.checks(judgement)
    assert check.name == "verdict_valid"
    assert not check.passed
    assert check.detail.startswith("invalid judgement (invalid_json): ")
