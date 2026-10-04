"""Text normalisation for fact matching and the "invented specifics" check.

`normalise` maps equal meanings to equal text, so a required fact matches an
answer however the model formats it:

- case, curly quotes, dashes, apostrophes and punctuation;
- contractions ("doesn't" -> "does not", "can't" and "can not" -> "cannot",
  "we'll" -> "we will", "here's" -> "here is");
- number words ("thirty days" -> "30 day", "twenty-five" -> "25");
- currency ("6.95 dollars", "USD 6.95", "$ 6.95", "$6.95" -> "$6.95";
  "$75.00" -> "$75"; "$1,200" -> "$1200") and rates ("$20/day", "$20 a
  day" -> "$20 per day");
- ranges ("3-5", "3 – 5", "between 3 and 5" -> "3 to 5"; "9 AM – 2 PM" ->
  "9 am to 2 pm"; "Monday–Saturday" -> "monday to saturday";
  "40%-60%", "between 40% and 60%" -> "40 to 60 percent");
- phone numbers ("+1-555-0199", "1 (555) 0199" -> "15550199"; a plain
  pair such as "100 2000" stays as it is) and ISO dates ("2026-10-04" ->
  "2026 10 04", never a range);
- times ("2 p.m.", "2PM" -> "2 pm"; "8:00" keeps its colon);
- units and ordinals written against the number ("18V" -> "18 v",
  "2.0Ah" -> "2.0 ah", "3rd" -> "3");
- a request to contact support, in any of its usual forms ("contact our
  support team", "reach out to customer service") -> "contact support";
- units in the singular ("days" -> "day", "30-day" -> "30 day").

Limits: number words are converted up to ninety-nine; "a year" and "an
hour" are not "1 year" and "1 hour" (facts list them as alternatives);
24-hour and 12-hour times are not converted into each other ("14:00" vs
"2 pm"), and "working days" is not "business days", so facts list both forms
as alternatives.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}  # fmt: skip
_SINGLE_DIGIT = "|".join(name for name, value in _UNITS.items() if 1 <= value <= 9)
_NUMBER_WORD = re.compile(
    rf"\b(?:(?P<tens>{'|'.join(_TENS)})(?:[\s-](?P<unit>{_SINGLE_DIGIT}))?"
    rf"|(?P<small>{'|'.join(_UNITS)}))\b"
)
_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", "‑": "-", "‐": "-"})
_QUOTES = str.maketrans({"’": "'", "‘": "'", "`": "'", "“": '"', "”": '"'})
_CONTRACTIONS = (
    (re.compile(r"\bcan'?t\b"), "cannot"),
    (re.compile(r"\bcan not\b"), "cannot"),
    (re.compile(r"\bwon'?t\b"), "will not"),
    (re.compile(r"\bshan't\b"), "shall not"),
    (re.compile(r"(\w)n't\b"), r"\1 not"),
    (re.compile(r"\b(i|you|we|they|he|she|it|that|there)'ll\b"), r"\1 will"),
    (re.compile(r"\b(you|we|they)'re\b"), r"\1 are"),
    (re.compile(r"\b(i|you|we|they)'ve\b"), r"\1 have"),
    (re.compile(r"\bi'm\b"), "i am"),
    (re.compile(r"\b(here|there|that|what|it|who|where)'s\b"), r"\1 is"),
    (re.compile(r"\b(i|you|we|they|he|she)'d\b"), r"\1 would"),
)
_ORDINAL = re.compile(r"(\d)(?:st|nd|rd|th)\b")
_AM_PM = re.compile(r"(\d)\s*([ap])\.?\s?m\b\.?")
# A unit written against its number, as in "18V" or "2.0Ah".
_UNIT_SUFFIX = re.compile(r"(\d)(v|ah|mah|mm|cm|m|kg|g|lb|lbs|oz|w|kw|ft|h)\b")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_CURRENCY_BEFORE = re.compile(r"\b(?:usd|us\$)\s*(?=\d)")
_CURRENCY_AFTER = re.compile(r"(\d+(?:\.\d+)?)\s*(?:usd|us dollars|dollars|dollar|bucks)\b")
_DOLLAR_SPACE = re.compile(r"\$\s+(?=\d)")
_ZERO_CENTS = re.compile(r"(\$\d+)\.00\b")
_PER_UNIT = re.compile(r"(\d)\s*(?:/|\ba\b|\ban\b|\bper\b)\s*(day|week|month|hour|year)\b")
# A phone number: optional "+" and country code, a 3-digit group (optionally in
# brackets), 3-4 digits, and an optional 4-digit group. Shaped, so it never
# runs on into a neighbouring number ("TS-104233 (2 items)"). It is collapsed
# only with a sign of a phone (see `_collapse_phone`), so a plain pair such as
# "100 2000" stays two numbers.
_PHONE = re.compile(
    r"(?<![\w.$:])(?P<plus>\+)?(?:(?P<cc>\d{1,3})[ .-]?)?(?P<open>\()?\d{3}\)?"
    r"(?P<sep>[ .-]?)\d{3,4}(?:[ .-](?P<ext>\d{4}))?(?!\w|\.\d|:)"
)
# An ISO date keeps its parts apart instead of turning into a range.
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_QUALIFIED = r"\$?\d+(?:[.:]\d+)?(?: (?:am|pm|percent))?"
_BETWEEN = re.compile(rf"\bbetween ({_QUALIFIED}) and (?=\$?\d)")
_RANGE = re.compile(rf"({_QUALIFIED}) ?- ?(?=\$?\d)")
_PERCENT_RANGE = re.compile(r"(\d+(?:\.\d+)?) percent to (\d+(?:\.\d+)?) percent\b")
_LOOSE_SEPARATOR = re.compile(r"(?<!\d)[.:]|[.:](?!\d)")
_OTHER = re.compile(r"[^a-z0-9$.: ]+")
_PLURAL_UNIT = re.compile(r"\b(day|year|hour|minute|week|month)s\b")
_SPACES = re.compile(r"\s+")

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
# "may" is left out: in an answer it is almost always the verb.
_MONTHS = (
    "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december",
)  # fmt: skip
_CALENDAR = re.compile(rf"\b({'|'.join(_WEEKDAYS + _MONTHS)})s?\b")
_WEEKDAY_RANGE = re.compile(rf"\b({'|'.join(_WEEKDAYS)}) ?- ?({'|'.join(_WEEKDAYS)})\b")
# Every usual way to send a customer to support. Shared by the fact check
# ("contact support") and the "I don't know" check, which treats a redirect
# as a separate signal from a decline.
SUPPORT_REDIRECT = re.compile(
    r"\b(?:contact|contacting|reach out to|reaching out to|get in touch with|ask|call|"
    r"email|write to|message|talk to|speak to|speak with)"
    r"(?: (?:our|the|toolshop|toolshops|a|customer|friendly))*"
    r" (?:support|customer service|customer care|help desk)"
    r"(?: (?:team|staff|agent|agents|line|desk))?\b"
)
_NUMBER = re.compile(r"(?<![a-z0-9.:$])\$?\d+(?:[.:]\d+)*(?![a-z0-9])")
# Letters and digits in one token: tracking numbers, discount codes ("save20").
_CODE = re.compile(r"\b(?=[a-z0-9]*\d)(?=[a-z0-9]*[a-z])[a-z0-9]+\b")


def _words_to_digits(text: str, skip: frozenset[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        if match.group("small") is not None:
            word = match.group("small")
            return word if word in skip else str(_UNITS[word])
        value = _TENS[match.group("tens")]
        if unit := match.group("unit"):
            value += _UNITS[unit]
        return str(value)

    return _NUMBER_WORD.sub(replace, text)


def _collapse_phone(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    phone_like = (
        match.group("plus")
        or match.group("open")
        or match.group("cc")
        or match.group("ext")
        or match.group("sep") in ("-", ".")
    )
    return digits if phone_like and len(digits) >= 7 else match.group(0)


def normalise(text: str, *, keep_words: frozenset[str] = frozenset()) -> str:
    """Lower-case, canonical text; see the module docstring.

    `keep_words` are number words left as words (the invented-specifics check
    keeps "one", which is usually a pronoun).
    """
    text = unicodedata.normalize("NFKC", text).lower().translate(_DASHES).translate(_QUOTES)
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    text = _words_to_digits(text, keep_words)
    text = _ORDINAL.sub(r"\1", text)
    text = _AM_PM.sub(r"\1 \2m ", text)
    text = _UNIT_SUFFIX.sub(r"\1 \2", text)
    text = _THOUSANDS.sub("", text)
    text = _CURRENCY_BEFORE.sub("$", text)
    text = _CURRENCY_AFTER.sub(r"$\1", text)
    text = _DOLLAR_SPACE.sub("$", text)
    text = _ZERO_CENTS.sub(r"\1", text)
    text = _PER_UNIT.sub(r"\1 per \2", text)
    text = _SPACES.sub(" ", text.replace("%", " percent "))
    text = _ISO_DATE.sub(r"\1 \2 \3", text)
    text = _PHONE.sub(_collapse_phone, text)
    text = _BETWEEN.sub(r"\1 to ", text)
    text = _RANGE.sub(r"\1 to ", text)
    text = _WEEKDAY_RANGE.sub(r"\1 to \2", text)
    text = _PERCENT_RANGE.sub(r"\1 to \2 percent", text)
    text = text.replace("'", "")
    text = _LOOSE_SEPARATOR.sub(" ", text)
    text = _OTHER.sub(" ", text)
    text = _PLURAL_UNIT.sub(r"\1", text)
    text = _SPACES.sub(" ", text).strip()
    return SUPPORT_REDIRECT.sub("contact support", text)


def contains(haystack: str, needle: str) -> bool:
    """Whether normalised `needle` occurs in normalised `haystack` as whole tokens."""
    if not needle:
        raise ValueError("cannot look for an empty fact")
    return f" {needle} " in f" {haystack} "


def _canonical_number(token: str) -> str:
    token = token.lstrip("$")
    if ":" in token:
        return token
    try:
        return format(Decimal(token).normalize(), "f")
    except InvalidOperation:  # pragma: no cover - the regex only yields numbers
        return token


def specific_tokens(text: str) -> list[tuple[str, str]]:
    """(as written after normalisation, canonical value) for every specific in `text`.

    Specifics are numbers (amounts, counts, times, phone numbers; compared by
    value, so "$75.00" equals "75 dollars"), codes that mix letters and digits
    (tracking numbers, discount codes) and weekday or month names.
    """
    normal = normalise(text, keep_words=frozenset({"one"}))
    found: list[tuple[int, str, str]] = []
    for match in _NUMBER.finditer(normal):
        found.append((match.start(), match.group(0), _canonical_number(match.group(0))))
    for match in _CODE.finditer(normal):
        found.append((match.start(), match.group(0), match.group(0)))
    for match in _CALENDAR.finditer(normal):
        found.append((match.start(), match.group(0), match.group(1)))
    return [(shown, value) for _, shown, value in sorted(found)]


def specifics(text: str) -> set[str]:
    """The canonical specifics of `text` (see `specific_tokens`)."""
    return {value for _, value in specific_tokens(text)}
