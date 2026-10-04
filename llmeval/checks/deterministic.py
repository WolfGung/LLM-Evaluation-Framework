"""Deterministic checks: rules a program decides without a model.

Triage checks read the raw reply exactly as the parser does
(`app.triage.unwrap_reply`) and are independent of each other, so one reply
can fail `schema_valid` for a bad order id while `enums_valid` still passes.

RAG checks look at the answer text, the cited ids and the retrieved ids.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from app.triage import Category, Priority, TriageError, parse_triage, unwrap_reply
from llmeval.checks import CheckResult
from llmeval.checks.text import contains, normalise, specific_tokens, specifics

# Answers above this many words fail `within_length`. v2 asks for at most five
# sentences; 150 words is a generous ceiling for a support answer either way.
MAX_ANSWER_WORDS = 150

REQUIRED_TRIAGE_FIELDS = ("category", "priority", "order_id", "summary")
_CATEGORIES = frozenset(str(c) for c in Category)
_PRIORITIES = frozenset(str(p) for p in Priority)


# --- triage -----------------------------------------------------------------


def reply_object(raw: str) -> tuple[dict[str, Any] | None, str]:
    """The reply as a JSON object, or None and the reason it is not one."""
    if not raw.strip():
        return None, "empty reply"
    try:
        value = json.loads(unwrap_reply(raw))
    except json.JSONDecodeError as exc:
        return None, f"not valid JSON: {exc.msg} at line {exc.lineno}"
    if not isinstance(value, dict):
        return None, f"valid JSON but not a JSON object ({type(value).__name__})"
    return value, ""


def triage_json_valid(raw: str) -> CheckResult:
    obj, reason = reply_object(raw)
    return CheckResult("json_valid", obj is not None, reason or "one JSON object")


def triage_required_fields(raw: str) -> CheckResult:
    obj, reason = reply_object(raw)
    if obj is None:
        return CheckResult("required_fields", False, reason)
    missing = [name for name in REQUIRED_TRIAGE_FIELDS if name not in obj]
    if missing:
        return CheckResult("required_fields", False, f"missing {', '.join(missing)}")
    return CheckResult("required_fields", True, "all fields present")


def triage_enums_valid(raw: str) -> CheckResult:
    obj, reason = reply_object(raw)
    if obj is None:
        return CheckResult("enums_valid", False, reason)
    problems = []
    for name, allowed in (("category", _CATEGORIES), ("priority", _PRIORITIES)):
        if name not in obj:
            problems.append(f"{name} missing")
        elif obj[name] not in allowed:
            problems.append(f"{name} {obj[name]!r} is not one of {', '.join(sorted(allowed))}")
    if problems:
        return CheckResult("enums_valid", False, "; ".join(problems))
    return CheckResult("enums_valid", True, "category and priority are valid")


def triage_schema_valid(raw: str) -> CheckResult:
    try:
        parse_triage(raw)
    except TriageError as exc:
        return CheckResult("schema_valid", False, f"{exc.kind}: {exc.detail}")
    return CheckResult("schema_valid", True, "matches TriageResult")


def triage_checks(raw: str) -> list[CheckResult]:
    """Every deterministic triage check, in a fixed order."""
    return [
        triage_json_valid(raw),
        triage_required_fields(raw),
        triage_enums_valid(raw),
        triage_schema_valid(raw),
    ]


# --- RAG answers ------------------------------------------------------------


def has_text(text: str, empty_reason: str | None = None) -> CheckResult:
    if text.strip():
        return CheckResult("has_text", True, "the answer has text")
    return CheckResult("has_text", False, f"empty answer ({empty_reason or 'no_content'})")


def cites_retrieved(cited: Sequence[str], retrieved: Sequence[str]) -> CheckResult:
    """At least one cited id is a retrieved document."""
    good = [doc for doc in cited if doc in retrieved]
    if good:
        return CheckResult("cites_retrieved", True, f"cites {', '.join(good)}")
    if not retrieved:
        return CheckResult("cites_retrieved", False, "nothing was retrieved, so nothing to cite")
    shown = ", ".join(cited) if cited else "nothing"
    return CheckResult(
        "cites_retrieved",
        False,
        f"cites {shown}; retrieved {', '.join(retrieved)}",
    )


def no_unretrieved_citations(cited: Sequence[str], retrieved: Sequence[str]) -> CheckResult:
    """No cited id is outside the retrieved documents (made up or never shown)."""
    extra = [doc for doc in cited if doc not in retrieved]
    if extra:
        return CheckResult(
            "no_unretrieved_citations", False, f"cites documents not retrieved: {', '.join(extra)}"
        )
    return CheckResult("no_unretrieved_citations", True, "every citation was retrieved")


def no_forbidden(text: str, forbidden: Sequence[str]) -> CheckResult:
    """None of the forbidden phrases occurs, after normalisation."""
    normal = normalise(text)
    found = [phrase for phrase in forbidden if contains(normal, normalise(phrase))]
    if found:
        return CheckResult("no_forbidden", False, f"contains {', '.join(found)}")
    return CheckResult("no_forbidden", True, "no forbidden phrase")


def within_length(text: str, limit: int = MAX_ANSWER_WORDS) -> CheckResult:
    words = len(text.split())
    return CheckResult("within_length", words <= limit, f"{words} words (limit {limit})")


# --- "I don't know" ---------------------------------------------------------

# Matched against `normalise(text)`: lower case, apostrophes removed
# ("don't" -> "dont", "I'm" -> "im").
DECLINE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bi (?:dont|do not) know\b",
        r"\b(?:im|i am) not (?:sure|certain)\b",
        r"\bnot sure\b",
        r"\bi (?:cant|cannot|couldnt|could not|am unable to|was unable to) "
        r"(?:find|see|answer|confirm|say|tell)\b",
        r"\bunable to (?:find|answer|confirm)\b",
        r"\b(?:documents?|information|knowledge base|sources?|docs|details)"
        r"(?: i have| available| provided| here)? "
        r"(?:dont|do not|doesnt|does not) "
        r"(?:say|mention|cover|contain|include|answer|specify|state)\b",
        r"\b(?:i )?(?:dont|do not) have (?:any )?(?:information|details|info|data)\b",
        r"\bno information\b",
        r"\b(?:isnt|is not|arent|are not|not) (?:mentioned|covered|specified|stated|listed)\b",
        r"\b(?:contact|reach out to|get in touch with|ask|call) "
        r"(?:our |the |toolshop |toolshops )*(?:customer )?support\b",
    )
)


def declines(text: str) -> str | None:
    """The decline phrase found in `text`, or None.

    A decline is any of a set of phrasings that say the answer is not known
    or send the customer to support ("I don't know", "I'm not sure", "the
    documents don't say", "I can't find", "please contact support" ...).
    """
    normal = normalise(text)
    for pattern in DECLINE_PATTERNS:
        if match := pattern.search(normal):
            return match.group(0)
    return None


def invented_specifics(text: str, sources: Sequence[str]) -> list[str]:
    """Specifics in `text` that appear in none of `sources`, in order of appearance.

    Specifics are numbers (prices, counts, times, phone numbers, compared by
    value), codes that mix letters and digits, and weekday or month names.
    `sources` are the retrieved documents plus the question; for a time such
    as "8:00" in a source, "8" counts too.
    """
    allowed: set[str] = set()
    for source in sources:
        for value in specifics(source):
            allowed.add(value)
            if ":" in value:
                allowed.update(part.lstrip("0") or "0" for part in value.split(":"))
    invented: list[str] = []
    for shown, value in specific_tokens(text):
        if value not in allowed and shown not in invented:
            invented.append(shown)
    return invented


def dont_know(text: str, sources: Sequence[str]) -> CheckResult:
    """The answer declines and states no specifics the sources do not contain.

    This measures behaviour, not one wording: any phrasing in
    `DECLINE_PATTERNS` counts, so a prompt that tells the model to say exactly
    "I don't know" gets no advantage from that sentence alone.

    Known limits:
    - The phrase list is English and finite. A decline in other words ("that
      is outside what I can see") fails.
    - "Please contact support" counts as a decline, so an answer that invents
      a non-numeric claim and then sends the customer to support passes this
      check. Each case's `forbidden` phrases catch the claims the author
      expected; others are left to the judge layer.
    - Only numbers and calendar words are checked for invention. Invented
      policy terms without numbers ("we price match") are caught only by
      `forbidden`.
    - A number that appears anywhere in a retrieved document is allowed, even
      if the answer uses it for something else.
    - "one" is not treated as a number (it is usually a pronoun), and a phone
      number written without its country code does not match the full one.
    """
    if not text.strip():
        return CheckResult("dont_know", False, "empty answer")
    phrase = declines(text)
    invented = invented_specifics(text, sources)
    if phrase is None:
        detail = "no decline phrasing found"
        if invented:
            detail += f"; specifics not in the documents: {', '.join(invented)}"
        return CheckResult("dont_know", False, detail)
    if invented:
        return CheckResult(
            "dont_know",
            False,
            f"declines ({phrase!r}) but states specifics not in the documents: "
            f"{', '.join(invented)}",
        )
    return CheckResult("dont_know", True, f"declines ({phrase!r}) and invents no specifics")
