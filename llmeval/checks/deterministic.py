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


# Clause breaks of the raw text: sentence ends, a comma that is not a
# thousands separator, a dash (an em or en dash, or a hyphen with spaces
# around it), a contrasting conjunction ("I'm not sure about online orders,
# but we price match in store" holds a claim after "but") and "and" when a
# subject follows it ("I don't know the fee and you can rent one" holds a
# claim after "and"; "charged between 40% and 60%" does not break).
# Normalisation drops punctuation, so the text is split first and each clause
# is normalised on its own.
_CLAUSE_BREAK = re.compile(
    r"(?<=[.!?;…])\s+|\n+|,(?=\s|$)|[—–]|\s-\s"
    r"|\b(?:but|however|although|though|yet)\b"
    r"|\band(?= (?:we|you|they|it|there|our|toolshop)\b)",
    re.IGNORECASE,
)
# A clause opened by one of these words questions what follows ("I don't
# know if we sharpen blades"); it does not claim it.
_CLAUSE_OPENERS = frozenset({"if", "whether"})
# Within its own clause, a decline phrase or one of these hedges what follows
# ("I'm not sure we price match", "I do not think we price match").
_HEDGES = (re.compile(r"\bi do not (?:think|believe)\b"),)


def clauses(text: str) -> list[str]:
    """The raw text split at clause breaks (see `_CLAUSE_BREAK`), blank parts dropped."""
    return [part for part in _CLAUSE_BREAK.split(text) if part.strip()]


def _occurrences(tokens: list[str], wanted: list[str]) -> list[int]:
    return [
        start
        for start in range(len(tokens) - len(wanted) + 1)
        if tokens[start : start + len(wanted)] == wanted
    ]


def _claims(text: str, phrase: str) -> bool:
    """`phrase` (normalised) occurs in `text` at least once as a claim.

    An occurrence is hedged, not a claim, only by words earlier in its own
    clause: "if" or "whether", a decline phrase ("not sure", "I can't
    confirm", "the documents don't say" ...), or "I do not think / believe".
    A decline in another clause or sentence hedges nothing.
    """
    wanted = phrase.split()
    for clause in clauses(text):
        tokens = normalise(clause).split()
        for start in _occurrences(tokens, wanted):
            before = tokens[:start]
            prefix = " ".join(before)
            hedged = (
                bool(_CLAUSE_OPENERS & set(before))
                or any(p.search(prefix) for p in DECLINE_PATTERNS)
                or any(p.search(prefix) for p in _HEDGES)
            )
            if not hedged:
                return True
    return False


def no_forbidden(text: str, forbidden: Sequence[str]) -> CheckResult:
    """None of the forbidden phrases is claimed, after normalisation.

    An entry may list alternatives separated by `|`; any of them counts. A
    hedged mention is not a claim (see `_claims`): "I'm not sure we price
    match" and "I don't know if tools can be rented" pass; "If so, we price
    match" and "I'm not sure about online orders, but we price match in
    store" fail. Hedges outside the list ("I doubt we price match") count as
    claims.

    What this measures: listed invention. A claim in words the case's list
    does not foresee is not caught here; wording-level invention beyond the
    lists is the judge layer's job.

    Known limits:
    - Commas around an aside cut off a hedge: in "I'm not sure if tile
      cutters, or any other tools, can be rented" the last clause has no
      hedge of its own, so the honest answer fails. This errs on the
      cautious side.
    - A forbidden list matches its listed wording only. Invention that
      means the same in other words ("the shop refunds the difference") is
      the judge layer's job.
    - "anyway" and "still" do not break a clause. Without a comma, "I'm not
      sure about online orders anyway we price match in store" is one
      clause, so the hedge covers the claim and the answer passes; with a
      comma before "anyway" or "still" it fails as it should.
    - "and" breaks a clause only before a pronoun subject (we, you, they, it,
      there, our, Toolshop). A noun subject does not break: "I don't know the
      amount and students get a discount" keeps the claim under the hedge.
    """
    found = [
        entry
        for entry in forbidden
        if any(_claims(text, normalise(alt)) for alt in entry.split("|") if alt.strip())
    ]
    if found:
        return CheckResult("no_forbidden", False, f"contains {', '.join(found)}")
    return CheckResult("no_forbidden", True, "no forbidden phrase")


def within_length(text: str, limit: int = MAX_ANSWER_WORDS) -> CheckResult:
    words = len(text.split())
    return CheckResult("within_length", words <= limit, f"{words} words (limit {limit})")


# --- "I don't know" ---------------------------------------------------------

# Matched against `normalise(text)`: lower case, contractions expanded
# ("don't" -> "do not", "I'm" -> "i am", "can't" -> "cannot").
_SOURCE_NOUNS = (
    r"(?:documents?|documentation|information|knowledge base|sources?|docs|notes|records|"
    r"pages?|articles?)"
)
# "the documents", "our help pages", "the provided documentation" ...
_SOURCES = (
    r"(?:(?:the|my|our|these|this|those|any) )?"
    r"(?:(?:provided|available|given|help|support|retrieved|reference) )?" + _SOURCE_NOUNS
)
DECLINE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bi do not know\b",
        r"\bi am not (?:sure|certain)\b",
        r"\bnot sure\b",
        r"\bunsure\b",
        r"\bno idea\b",
        r"\bnot aware of\b",
        r"\bi (?:cannot|could not|am unable to|was unable to|am not able to) "
        r"(?:find|see|answer|confirm|say|tell)\b",
        # "wasn't" is normalised to "was not" before the match.
        r"\b(?:i|we) (?:was|were) not able to (?:find|see|confirm|say)\b",
        r"\bunable to (?:find|answer|confirm)\b",
        r"\bi do not see (?:anything|any information|any details|any mention)\b",
        rf"\b{_SOURCES}(?: i have| available| provided| here| i can see)? "
        r"(?:do not|does not|did not) "
        r"(?:say|mention|cover|contain|include|answer|specify|state|tell|explain|show)\b",
        rf"\b{_SOURCES}(?: i have| available| provided| here)? (?:do not|does not) seem to "
        r"(?:say|mention|cover|include|answer)\b",
        rf"\bnone of {_SOURCES} (?:mentions?|says?|covers?|contains?|includes?)\b",
        r"\bdo not have (?:(?:that|this|the|enough|any|specific|such|those) )?"
        r"(?:information|details?|info|data)\b",
        r"\bhave no (?:details|information|info|data)\b",
        r"\bno information\b",
        r"\bno mention of\b",
        r"\bno (?:details?|information|mention) (?:about|of|on|regarding)\b",
        r"\bno [a-z0-9 ]{1,40}? (?:is|are) (?:mentioned|listed|stated|described)\b",
        r"\bdo not have (?:pricing|prices|a price|price information) (?:for|on|about)\b",
        r"\bnot something i have (?:any )?(?:information|details|info) (?:about|on)\b",
        rf"\bcannot help with (?:that|this)\b[a-z0-9 ]{{0,40}}? based on {_SOURCES}\b",
        # A negation counts only with a subject that refers to the sources:
        # "not mentioned in the documents", never a bare "not covered".
        r"\bnot (?:mentioned|listed|stated|specified|covered|included|described|addressed) "
        rf"(?:in|by) {_SOURCES}\b",
        rf"\b(?:is|are) not in {_SOURCES}\b",
        rf"\bnothing (?:in|about (?:this|that) in) {_SOURCES}\b",
        rf"\bnothing (?:about|on|regarding) [a-z0-9$.: ]{{1,60}}? in {_SOURCES}\b",
        rf"\b{_SOURCES} (?:says?|contains?|mentions?) nothing\b",
        r"\b(?:information|info|details?) (?:is|are) (?:not available|unavailable)\b",
    )
)


def declines(text: str) -> str | None:
    """The decline phrase found in `text`, or None.

    A decline says the answer is not known or not in the sources ("I don't
    know", "I'm not sure", "I'm unsure", "I have no idea", "I wasn't able to
    find", "I'm not aware of", "the documents don't say", "our help pages
    don't say", "none of the documents mention", "there's no mention of", "I
    don't have that information", "not mentioned in the documents", "there is
    nothing about X in my documents", "that information is unavailable" ...).
    The list is wide on purpose: a prompt that tells the model to say "I
    don't know" must not win by that sentence.

    Sending the customer to support is not a decline on its own (see
    `redirects`), and a bare negation ("is not covered") is a policy
    statement, not a decline.

    Known limit: the list is finite. A decline in words it does not hold
    ("No info on that", "that is outside what our documents cover") is not
    recognised, and the honest answer fails `dont_know`.
    """
    normal = normalise(text)
    for pattern in DECLINE_PATTERNS:
        if match := pattern.search(normal):
            return match.group(0)
    return None


def redirects(text: str) -> bool:
    """Whether the answer sends the customer to support (`SUPPORT_REDIRECT`)."""
    return contains(normalise(text), "contact support")


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

    What this measures: a decline, and no numeric invention (numbers, codes,
    calendar words). Together with `no_forbidden` (listed invention) that is
    "declines, and no numeric or listed invention". Invention in other words
    is not caught by these checks; it is the judge layer's job.

    This measures behaviour, not one wording: any phrasing in
    `DECLINE_PATTERNS` counts, so a prompt that tells the model to say exactly
    "I don't know" gets no advantage from that sentence alone.

    A redirect to support is reported in the detail but is not a decline:
    "Yes, you can rent one, ask our support team" invents an answer and then
    redirects.

    Known limits:
    - The phrase list is English and finite. A decline in other words ("No
      info on that", "that is outside what our documents cover") fails.
    - A decline phrase anywhere passes this check, even next to an invented
      claim ("We sharpen blades for free. Not sure about chisels."). Each
      case's `forbidden` phrases catch the claims the author expected; others
      are left to the judge layer.
    - Only numbers, codes and calendar words are checked for invention.
      Invented policy terms without numbers ("we price match") are caught only
      by `forbidden`.
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
        if redirects(text):
            detail += " (redirects to support, which alone is not a decline)"
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
