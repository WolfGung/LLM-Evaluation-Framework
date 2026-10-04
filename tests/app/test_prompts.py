"""Prompt files: versions are discovered from `app/prompts/`, placeholders are filled once."""

import re

import pytest

from app.prompting import PROMPTS_DIR, PromptError, load_prompt, prompt_versions, render

PLACEHOLDERS = {"assistant": {"documents"}, "triage": {"schema"}}


@pytest.mark.parametrize("function", sorted(PLACEHOLDERS))
def test_each_function_has_v1_and_v2(function):
    assert prompt_versions(function) == ("v1", "v2")


@pytest.mark.parametrize("function", sorted(PLACEHOLDERS))
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_each_prompt_uses_its_placeholders_exactly_once(function, version):
    text = load_prompt(function, version)
    found = re.findall(r"\{\{(\w+)\}\}", text)
    assert sorted(found) == sorted(PLACEHOLDERS[function])


@pytest.mark.parametrize("function", sorted(PLACEHOLDERS))
def test_v2_is_a_different_prompt(function):
    assert load_prompt(function, "v1") != load_prompt(function, "v2")


def test_prompt_versions_are_read_from_the_files():
    names = sorted(p.name for p in PROMPTS_DIR.glob("*.md"))
    assert names == ["assistant_v1.md", "assistant_v2.md", "triage_v1.md", "triage_v2.md"]


@pytest.mark.parametrize("version", ["v3", "V1", "../triage_v1", "v1.md", ""])
def test_unknown_versions_are_refused(version):
    with pytest.raises(PromptError, match="version"):
        load_prompt("assistant", version)


def test_unknown_functions_are_refused():
    with pytest.raises(PromptError, match="function"):
        load_prompt("judge", "v1")


def test_render_fills_every_placeholder_in_one_pass():
    # A value that itself looks like a placeholder is left as text.
    assert render("A {{x}} B {{y}}", x="{{y}}", y="2") == "A {{y}} B 2"


def test_render_refuses_a_missing_value():
    with pytest.raises(PromptError, match="y"):
        render("{{x}} {{y}}", x="1")


def test_render_refuses_an_unused_value():
    with pytest.raises(PromptError, match="z"):
        render("{{x}}", x="1", z="2")


def test_render_keeps_dollar_signs_and_braces():
    assert render('$5 {"a": 1} {{x}}', x="ok") == '$5 {"a": 1} ok'


def test_assistant_v2_states_the_grounding_rules():
    text = load_prompt("assistant", "v2")
    for phrase in ("only the documents", "I don't know", "not instructions", "personal data"):
        assert phrase in text


def test_triage_v2_adds_priority_rules_and_order_id_guidance():
    v1, v2 = load_prompt("triage", "v1"), load_prompt("triage", "v2")
    for phrase in ("urgent:", "high:", "normal:", "low:", "TS-", "null"):
        assert phrase in v2
    assert "urgent:" not in v1
