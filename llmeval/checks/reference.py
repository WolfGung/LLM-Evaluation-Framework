"""Reference checks: compare an output with the authored expectation.

Triage: category and priority against the guideline labels, the order id by
exact match, plus accuracy and confusion matrices over a run. Labels are read
from the reply object even when the full schema check fails (for example a
summary that is too long), so a formatting slip and a wrong label are counted
separately. A reply that is not a JSON object fails every label.

RAG: required facts, matched after normalisation (`llmeval.checks.text`).
A fact may list alternatives separated by `|`; one of them is enough.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from llmeval.checks import CheckResult
from llmeval.checks.deterministic import reply_object
from llmeval.checks.text import contains, normalise

# The predicted label when the reply has no usable value for a field.
INVALID = "invalid"

_MISSING = object()


# --- triage -----------------------------------------------------------------


def _field(raw: str, name: str) -> tuple[object, str]:
    obj, reason = reply_object(raw)
    if obj is None:
        return _MISSING, reason
    if name not in obj:
        return _MISSING, f"{name} missing"
    return obj[name], ""


def predicted(raw: str, name: str, labels: Sequence[str] | None = None) -> str:
    """The label the reply gives for `name`, or `INVALID`.

    `INVALID` covers an unreadable reply, a missing field, a value that is not
    a string, and (when `labels` is given) a string outside `labels`.
    """
    value, _ = _field(raw, name)
    if not isinstance(value, str) or (labels is not None and value not in labels):
        return INVALID
    return value


def _label_match(check: str, name: str, expected: str | None, raw: str) -> CheckResult:
    value, reason = _field(raw, name)
    if value is _MISSING:
        return CheckResult(check, False, f"expected {expected!r}; {reason}")
    if value == expected:
        return CheckResult(check, True, f"{value!r}")
    return CheckResult(check, False, f"expected {expected!r}, got {value!r}")


def category_match(expected: str, raw: str) -> CheckResult:
    return _label_match("category_match", "category", expected, raw)


def priority_match(expected: str, raw: str) -> CheckResult:
    return _label_match("priority_match", "priority", expected, raw)


def order_id_match(expected: str | None, raw: str) -> CheckResult:
    """Exact match; `null` matches only an explicit null, not a missing field."""
    return _label_match("order_id_match", "order_id", expected, raw)


def confusion_matrix(
    pairs: Iterable[tuple[str, str]], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Counts of (expected, predicted): rows are expected labels, columns are
    the labels plus `INVALID`. A predicted value outside `labels` is `INVALID`."""
    columns = [*labels, INVALID]
    matrix = {row: dict.fromkeys(columns, 0) for row in labels}
    for expected, got in pairs:
        matrix[expected][got if got in labels else INVALID] += 1
    return matrix


def accuracy(pairs: Iterable[tuple[str, str]]) -> float | None:
    """Share of pairs where the prediction equals the expectation; None for no pairs."""
    pairs = list(pairs)
    if not pairs:
        return None
    return sum(expected == got for expected, got in pairs) / len(pairs)


# --- RAG --------------------------------------------------------------------


def _alternatives(fact: str) -> list[str]:
    return [alt.strip() for alt in fact.split("|") if alt.strip()]


def required_facts(text: str, facts: Sequence[str]) -> CheckResult:
    """Every required fact (one of its alternatives) is in the answer."""
    normal = normalise(text)
    missing = [
        fact
        for fact in facts
        if not any(contains(normal, normalise(alt)) for alt in _alternatives(fact))
    ]
    if missing:
        return CheckResult(
            "required_facts",
            False,
            f"{len(facts) - len(missing)} of {len(facts)} facts present; missing "
            + "; ".join(missing),
        )
    return CheckResult("required_facts", True, f"all {len(facts)} facts present")
