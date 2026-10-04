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
        # Review round 1: ranges after am/pm, "between", weekdays, percent.
        ("9 am to 2 pm", "9 AM – 2 PM"),
        ("8 am to 6 pm", "8am–6pm"),
        ("monday to saturday", "Monday–Saturday"),
        ("3 to 5 business days", "between 3 and 5 business days"),
        ("40 to 60 percent", "between 40% and 60%"),
        ("40 to 60 percent", "40%-60%"),
        ("$20 per day", "$20/day"),
        ("$20 per day", "$20 a day"),
        # Contractions are expanded.
        ("we will match the price", "We'll match the price"),
        ("does not accept", "doesn't accept"),
        ("cannot be returned", "can't be returned"),
        ("cannot be returned", "can not be returned"),
        # One support redirect.
        ("contact support", "contact our support team"),
        ("contact support", "reach out to Toolshop customer support"),
        ("contact support", "get in touch with customer service"),
        # Units and ordinals written against the number.
        ("18 V", "18V"),
        ("2.0 Ah", "2.0Ah"),
        ("the 3 of June", "the 3rd of June"),
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


def test_a_phone_number_does_not_swallow_the_next_number():
    assert normalise("TS-104233 (2 items)") == "ts 104233 2 items"
    assert specifics("TS-104233 (2 items)") == {"104233", "2"}
    assert normalise("Call 555-0142 (2 lines)") == "call 5550142 2 lines"


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


def test_units_and_ordinals_are_not_codes():
    assert specifics("an 18V drill with 2.0Ah packs, on the 3rd") == {"18", "2", "3"}
    assert specifics("18 V drill with 2.0 Ah packs, 3 days") == {"18", "2", "3"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("100 2000", "100 2000"),
        ("ships 100 2000 units", "ships 100 2000 units"),
        ("2026-10-04", "2026 10 04"),
        ("on 2026-10-04 at 9", "on 2026 10 04 at 9"),
    ],
)
def test_plain_number_pairs_and_dates_stay_apart(text, expected):
    assert normalise(text) == expected


@pytest.mark.parametrize(
    "text", ["+1 555 0199", "+1-555-0199", "1 (555) 0199", "1 555 0199", "555-0199", "555.0199"]
)
def test_phone_shaped_numbers_still_collapse(text):
    assert normalise(text).endswith("5550199")
