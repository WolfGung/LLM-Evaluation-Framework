"""Ticket triage: a support ticket in, a validated `TriageResult` out.

Two output paths, chosen by the model role:

- `structured_output: false` (the default system model): the JSON schema of
  `TriageResult` is in the prompt, and the reply is parsed and validated here.
- `structured_output: true`: the same prompt, plus `response_format` with the
  schema, so OpenRouter routes only to endpoints that enforce it.

Parsing is deliberately strict, because the evaluation measures how often the
model returns valid JSON. The only thing tolerated is the whole reply being
one fenced ```json block. A reply that does not validate raises `TriageError`
with the raw text, so the checks can see exactly what the model wrote.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.prompting import ChatModel, load_prompt, render
from llmeval.cassettes import CallTag
from llmeval.client import CallResult
from llmeval.config import RoleConfig

TRIAGE_FUNCTION = "triage"
ADHOC_CASE = "adhoc"
SCHEMA_NAME = "triage_result"

_FENCED_JSON = re.compile(r"\A```json[ \t]*\n(.*?)\s*```\Z", re.DOTALL)


class Category(StrEnum):
    SHIPPING = "shipping"
    RETURNS = "returns"
    PAYMENT = "payment"
    WARRANTY = "warranty"
    ORDER_STATUS = "order_status"
    PRODUCT_QUESTION = "product_question"
    OTHER = "other"


class Priority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TriageResult(BaseModel):
    """Every field is required; `order_id` is null when the ticket names no order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: Category
    priority: Priority
    order_id: str | None = Field(pattern=r"^TS-\d{6}$")
    summary: str = Field(min_length=1, max_length=200)


TriageErrorKind = Literal["empty", "invalid_json", "invalid_schema"]


class TriageError(ValueError):
    """The model's reply is not a valid `TriageResult`.

    `raw` is the reply exactly as the model wrote it; `kind` says what was
    wrong; `call` is the model call, when there was one.
    """

    def __init__(
        self, kind: TriageErrorKind, raw: str, detail: str, call: CallResult | None = None
    ) -> None:
        self.kind = kind
        self.raw = raw
        self.detail = detail
        self.call = call
        super().__init__(f"triage reply is {kind.replace('_', ' ')}: {detail}")

    def with_call(self, call: CallResult) -> TriageError:
        return TriageError(self.kind, self.raw, self.detail, call)


@dataclass(frozen=True)
class TriageAnswer:
    result: TriageResult
    call: CallResult


def triage_schema() -> dict[str, Any]:
    return TriageResult.model_json_schema()


def response_format_for(role: RoleConfig) -> dict[str, Any] | None:
    """`response_format` for roles that support structured output, else None."""
    if not role.structured_output:
        return None
    return {
        "type": "json_schema",
        "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": triage_schema()},
    }


def prepare(
    text: str, version: str, role: RoleConfig
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    """The exact messages and `response_format` for one ticket.

    The prompt is the same for both output paths, so the only difference
    between them is whether the provider enforces the schema.
    """
    if not text.strip():
        raise ValueError("ticket text is empty")
    template = load_prompt("triage", version)
    system = render(template, schema=json.dumps(triage_schema(), indent=2))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    return messages, response_format_for(role)


def parse_triage(raw: str) -> TriageResult:
    """Validate a reply. Raises `TriageError` with the raw text on any problem."""
    text = raw.strip()
    if not text:
        raise TriageError("empty", raw, "the reply has no text")
    if fenced := _FENCED_JSON.match(text):
        text = fenced.group(1)
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        raise TriageError("invalid_json", raw, f"{exc.msg} at line {exc.lineno}") from None
    try:
        return TriageResult.model_validate_json(text, strict=True)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'reply'}: {err['msg']}"
            for err in exc.errors(include_input=False, include_url=False)
        )
    raise TriageError("invalid_schema", raw, problems)


def triage(
    client: ChatModel,
    role: RoleConfig,
    text: str,
    version: str,
    *,
    repeat: int = 0,
    case: str | None = None,
) -> TriageAnswer:
    """Triage one ticket with prompt `version` in model role `role`."""
    messages, response_format = prepare(text, version, role)
    tag = CallTag(function=TRIAGE_FUNCTION, case=case or ADHOC_CASE, version=version)
    call = client.complete(
        messages, role=role, response_format=response_format, repeat=repeat, tag=tag
    )
    if call.empty_reason is not None:
        raise TriageError(
            "empty", call.content, f"the model returned no text ({call.empty_reason})", call
        )
    try:
        result = parse_triage(call.content)
    except TriageError as exc:
        error = exc.with_call(call)
    else:
        return TriageAnswer(result=result, call=call)
    raise error
