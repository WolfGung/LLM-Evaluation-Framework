"""Text normalisation used by the fact and "I don't know" checks.

Synthetic data: every string below is made up for the test.
"""

import pytest

from llmeval.checks.text import contains, normalise, specifics


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("30 days", "thirty days"),
        ("30 days", "30-day"),
        ("30 days", "Thirty Days."),
        ("1 year", "one-year"),
        ("25 percent", "twenty-five %"),
        ("25 percent", "twenty five percent"),
        ("$6.95", "6.95 dollars"),
        ("$6.95", "USD 6.95"),
        ("$6.95", "$ 6.95"),
        ("$6.95", "6.95 USD"),
        ("$75", "$75.00"),
        ("$1200", "$1,200"),
        ("3 to 5 business days", "3-5 business days"),
        ("3 to 5 business days", "3 – 5 business days"),
        ("40 to 60 percent", "40-60%"),
        ("+1 555 0199", "+1-555-0199"),
        ("+1 555 0199", "1 (555) 0199"),
        ("2 p.m.", "2 pm"),
        ("2 p.m.", "2PM"),
        ("don't accept", "don’t accept"),
        ("photo ID", "photo   id"),
        ("TS-104233", "ts 104233"),
        ("8:00 to 18:00", "8:00–18:00"),
    ],
)
def test_equal_meaning_normalises_to_equal_text(a, b):
    assert normalise(a) == normalise(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("$6.95", "$69.5"),
        ("30 days", "3 days"),
        ("8:00", "800"),
        ("1 year", "2 years"),
    ],
)
def test_different_facts_stay_different(a, b):
    assert normalise(a) != normalise(b)


def test_decimals_and_times_keep_their_separator():
    assert normalise("It costs $5.95. Open 8:00.") == "it costs $5.95 open 8:00"


def test_contains_matches_whole_tokens_only():
    text = normalise("Returns are accepted within 30 days, not 300.")
    assert contains(text, normalise("30 days"))
    assert not contains(text, normalise("0 days"))
    assert not contains(normalise("I know"), normalise("no"))


def test_contains_refuses_an_empty_needle():
    with pytest.raises(ValueError):
        contains("anything", "")


def test_specifics_are_numbers_and_calendar_words():
    found = specifics("Rent it for $25 a day, Monday to Friday in June, 10% off, call 8:30.")
    assert found == {"25", "monday", "friday", "june", "10", "8:30"}


def test_specifics_ignore_one_and_may():
    # "one" is usually a pronoun ("that one") and "may" a verb.
    assert specifics("You may want that one.") == set()


def test_specifics_compare_numbers_by_value():
    assert specifics("$75.00") == specifics("75 dollars") == {"75"}
    assert specifics("thirty") == {"30"}


def test_codes_with_digits_are_one_specific():
    # A tracking number or a discount code is one specific, not loose digits.
    assert specifics("Tracking 1Z999AA10123456784, code SAVE20.") == {
        "1z999aa10123456784",
        "save20",
    }
