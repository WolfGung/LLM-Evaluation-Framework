"""The authored datasets: they parse, their labels are consistent, sizes are as documented.

These are checks on test design, not on model output. No model is called.
"""

import re
from collections import Counter
from pathlib import Path

import pytest

from llmeval.datasets import PRIORITY_RULES, TRIAGE_PATH, load_triage

ROOT = Path(__file__).resolve().parents[2]
GUIDELINE = ROOT / "datasets" / "triage-guideline.md"

# Documented sizes (README and plan): about 40 triage cases.
TRIAGE_RANGE = range(35, 46)


@pytest.fixture(scope="module")
def triage_cases():
    return load_triage(ROOT / TRIAGE_PATH)


def test_triage_size_is_in_the_documented_range(triage_cases):
    assert len(triage_cases) in TRIAGE_RANGE


def test_triage_ids_are_numbered_in_order(triage_cases):
    assert [c.id for c in triage_cases] == [f"tri-{n:03d}" for n in range(1, len(triage_cases) + 1)]


def test_every_category_and_priority_is_covered(triage_cases):
    categories = Counter(str(c.category) for c in triage_cases)
    priorities = Counter(str(c.priority) for c in triage_cases)
    assert set(categories) == {
        "shipping",
        "returns",
        "payment",
        "warranty",
        "order_status",
        "product_question",
        "other",
    }
    assert set(priorities) == {"urgent", "high", "normal", "low"}
    assert min(categories.values()) >= 3
    assert min(priorities.values()) >= 4


def test_order_ids_are_present_and_absent(triage_cases):
    with_id = sum(c.order_id is not None for c in triage_cases)
    assert 0.3 * len(triage_cases) <= with_id <= 0.7 * len(triage_cases)


def test_the_guideline_defines_every_rule_under_its_priority():
    text = GUIDELINE.read_text(encoding="utf-8")
    sections = re.split(r"^### ", text, flags=re.MULTILINE)
    defined = {}
    for section in sections[1:]:
        heading = section.splitlines()[0].strip().lower()
        for rule in re.findall(r"^- \*\*([A-Z]\d)\*\*", section, flags=re.MULTILINE):
            defined[rule] = heading
    assert defined == {rule: str(priority) for rule, priority in PRIORITY_RULES.items()}


def test_the_guideline_says_v2_encodes_the_rules():
    text = GUIDELINE.read_text(encoding="utf-8")
    assert "partly by construction" in text


def test_a_written_order_id_is_extracted_in_normal_form(triage_cases):
    # Labels follow the guideline: a six-digit id in the text, written any
    # common way, is the expected order id in TS-###### form.
    pattern = re.compile(r"\bts[\s-]?(\d{6})\b", re.IGNORECASE)
    for case in triage_cases:
        found = [f"TS-{digits}" for digits in pattern.findall(case.text)]
        if not found:
            assert case.order_id is None, case.id
        else:
            assert case.order_id in found, case.id
