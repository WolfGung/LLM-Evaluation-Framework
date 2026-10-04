"""The judge rubric loader: criteria, anchors and one pass rule in prose and in code.

Synthetic data: every rubric below is a made-up text written for the test.
The repository's own rubric is checked in `tests/repo/test_rubric.py`.
"""

import hashlib

import pytest

from llmeval.checks.judge import CRITERIA, RubricError, load_rubric, parse_rubric

VALID = """\
---
criteria: [groundedness, helpfulness, tone]
pass_rule:
  groundedness: 4
  helpfulness: 3
  tone: 3
---
# Synthetic rubric

Do not reward length.

## Groundedness

- 5: synthetic anchor five.
- 3: synthetic anchor three.
- 1: synthetic anchor one.

## Helpfulness

- 5: synthetic anchor five.
- 3: synthetic anchor three.
- 1: synthetic anchor one.

## Tone

- 5: synthetic anchor five.
- 3: synthetic anchor three.
- 1: synthetic anchor one.

## Pass rule

`pass` is true when groundedness is at least 4, helpfulness is at least 3 and tone is at least 3.
"""


def test_a_valid_rubric_gives_criteria_rule_and_the_text_for_the_judge():
    rubric = parse_rubric(VALID, name="synthetic.md")
    assert rubric.criteria == CRITERIA == ("groundedness", "helpfulness", "tone")
    assert rubric.pass_rule == {"groundedness": 4, "helpfulness": 3, "tone": 3}
    # The judge reads the body only; the front matter is for the code.
    assert rubric.text.startswith("# Synthetic rubric")
    assert "pass_rule" not in rubric.text and "---" not in rubric.text
    # The hash covers what the judge reads: the body.
    assert rubric.sha256 == hashlib.sha256(rubric.text.encode("utf-8")).hexdigest()


def test_a_front_matter_comment_keeps_the_rubric_hash():
    # The judge never sees the front matter, so a comment there changes no
    # judge request and must not print a false "re-record" notice.
    commented = VALID.replace("criteria:", "# Synthetic comment for the code.\ncriteria:", 1)
    assert commented != VALID
    assert parse_rubric(commented, name="x.md").sha256 == parse_rubric(VALID, name="x.md").sha256


def test_a_body_edit_changes_the_rubric_hash():
    edited = VALID.replace("Do not reward length.", "Do not reward length or padding.")
    assert parse_rubric(edited, name="x.md").sha256 != parse_rubric(VALID, name="x.md").sha256


def test_line_endings_do_not_change_the_rubric_hash():
    crlf = VALID.replace("\n", "\r\n")
    assert parse_rubric(crlf, name="x.md").sha256 == parse_rubric(VALID, name="x.md").sha256


def test_the_rule_says_which_criteria_fall_short():
    rubric = parse_rubric(VALID, name="synthetic.md")
    assert rubric.short_of({"groundedness": 4, "helpfulness": 3, "tone": 3}) == ()
    assert rubric.short_of({"groundedness": 3, "helpfulness": 5, "tone": 2}) == (
        "groundedness",
        "tone",
    )
    assert rubric.passes({"groundedness": 5, "helpfulness": 3, "tone": 3})
    assert not rubric.passes({"groundedness": 3, "helpfulness": 5, "tone": 5})


def test_load_reads_a_file(tmp_path):
    path = tmp_path / "judge.md"
    path.write_text(VALID, encoding="utf-8")
    assert load_rubric(path) == parse_rubric(VALID, name="judge.md")


def test_a_missing_file_is_a_rubric_error(tmp_path):
    with pytest.raises(RubricError, match="cannot read"):
        load_rubric(tmp_path / "missing.md")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda t: t.replace("---\ncriteria", "criteria", 1), "front matter"),
        (lambda t: t.replace("criteria: [groundedness, helpfulness, tone]", "criteria: ["), "YAML"),
        (
            lambda t: t.replace("[groundedness, helpfulness, tone]", "[groundedness, tone]"),
            "criteria must be groundedness, helpfulness, tone",
        ),
        (
            lambda t: t.replace("[groundedness, helpfulness,", "[helpfulness, groundedness,"),
            "criteria must be groundedness, helpfulness, tone",
        ),
        (lambda t: t.replace("  tone: 3\n---", "---"), "pass_rule must give a minimum"),
        (lambda t: t.replace("  tone: 3\n---", "  tone: 6\n---"), "whole number from 1 to 5"),
        (lambda t: t.replace("  tone: 3\n---", "  tone: true\n---"), "whole number from 1 to 5"),
        (lambda t: t.replace("  tone: 3\n---", "  tone: 3\nextra: 1\n---"), "unknown front matter"),
    ],
    ids=[
        "no front matter",
        "broken YAML",
        "a criterion missing",
        "criteria out of order",
        "no minimum for a criterion",
        "a minimum above 5",
        "a minimum that is not a number",
        "an unknown key",
    ],
)
def test_a_broken_front_matter_is_refused(change, message):
    with pytest.raises(RubricError, match=message):
        parse_rubric(change(VALID), name="synthetic.md")


def test_every_criterion_needs_its_own_section():
    text = VALID.replace("## Tone", "## Manner")
    with pytest.raises(RubricError, match="no '## Tone' section"):
        parse_rubric(text, name="synthetic.md")


@pytest.mark.parametrize("anchor", ["5", "3", "1"])
def test_every_criterion_needs_anchors_for_1_3_and_5(anchor):
    # Remove the anchor from the helpfulness section only.
    head, rest = VALID.split("## Helpfulness", 1)
    section, tail = rest.split("## Tone", 1)
    section = section.replace(f"- {anchor}: synthetic anchor", "- synthetic anchor", 1)
    text = f"{head}## Helpfulness{section}## Tone{tail}"
    with pytest.raises(RubricError, match=f"Helpfulness has no anchor for score {anchor}"):
        parse_rubric(text, name="synthetic.md")


def test_the_prose_pass_rule_must_match_the_front_matter():
    # The judge reads the prose, the code applies the front matter: they must agree.
    text = VALID.replace("groundedness is at least 4", "groundedness is at least 3")
    with pytest.raises(
        RubricError, match="pass rule says groundedness is at least 3; the front matter says 4"
    ):
        parse_rubric(text, name="synthetic.md")


def test_the_prose_pass_rule_must_name_every_criterion():
    text = VALID.replace(" and tone is at least 3", "")
    with pytest.raises(RubricError, match="pass rule does not say 'tone is at least 3'"):
        parse_rubric(text, name="synthetic.md")


def test_a_pass_rule_section_is_required():
    text = VALID.split("## Pass rule", 1)[0]
    with pytest.raises(RubricError, match="no '## Pass rule' section"):
        parse_rubric(text, name="synthetic.md")


@pytest.mark.parametrize(
    "rule",
    [
        "groundedness is at least 4, helpfulness is at least 3 or tone is at least 3",
        "groundedness is at least 4 or helpfulness is at least 3 and tone is at least 3",
        "groundedness is at least 4, helpfulness is at least 3, tone is at least 3",
        "groundedness is at least 4. Helpfulness matters too: helpfulness is at least 3 "
        "and tone is at least 3",
    ],
    ids=["or at the end", "or at the start", "no and", "split across sentences"],
)
def test_the_prose_must_join_the_minimums_with_and(rule):
    text = VALID.replace(
        "groundedness is at least 4, helpfulness is at least 3 and tone is at least 3", rule
    )
    with pytest.raises(RubricError, match="join the minimums with 'and'"):
        parse_rubric(text, name="synthetic.md")


@pytest.mark.parametrize(
    "rule",
    [
        "groundedness is at least 4, helpfulness is at least 3, and tone is at least 3",
        "groundedness is at least 4 and helpfulness is at least 3 and tone is at least 3",
        "tone is at least 3, groundedness is at least 4 and helpfulness is at least 3",
    ],
    ids=["serial comma", "and twice", "another order"],
)
def test_minimums_joined_with_and_are_accepted(rule):
    text = VALID.replace(
        "groundedness is at least 4, helpfulness is at least 3 and tone is at least 3", rule
    )
    assert parse_rubric(text, name="synthetic.md").pass_rule["tone"] == 3
