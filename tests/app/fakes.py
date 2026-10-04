"""A fake model for the service's own tests.

Synthetic data: every answer below is a made-up string chosen by the test.
Nothing here is a model output; nothing is written to `cassettes/` or
`results/`. The fake builds the request body with the real
`build_role_request`, so it refuses a schema for an unstructured role exactly
like the real client.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from llmeval.cassettes import CallTag, Usage, request_key
from llmeval.client import CallResult, build_role_request
from llmeval.config import ReasoningConfig, RoleConfig

SYNTHETIC_TIME = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

UNSTRUCTURED = RoleConfig(
    model="synthetic/system-model:free",
    temperature=0.2,
    max_tokens=300,
    reasoning=ReasoningConfig(effort="none"),
    structured_output=False,
)
STRUCTURED = RoleConfig(
    model="synthetic/structured-model:free",
    temperature=0,
    seed=7,
    max_tokens=300,
    structured_output=True,
)

Reply = str | Callable[[Sequence[Mapping[str, Any]]], str]


@dataclass
class FakeModel:
    """Answers every call with `reply` (a string, or a function of the messages)."""

    reply: Reply = ""
    finish_reason: str | None = "stop"
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        role: RoleConfig,
        response_format: Mapping[str, Any] | None = None,
        repeat: int = 0,
        tag: CallTag | None = None,
    ) -> CallResult:
        body = build_role_request(messages, role, response_format)
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "role": role,
                "response_format": response_format,
                "repeat": repeat,
                "tag": tag,
                "body": body,
            }
        )
        if self.error is not None:
            raise self.error
        content = self.reply(messages) if callable(self.reply) else self.reply
        return CallResult(
            content=content,
            model_requested=role.model,
            model_used=role.model,
            usage=Usage(prompt_tokens=10, completion_tokens=5),
            cost_usd=0.0,
            cost_source="provider",
            latency_ms=1.0,
            recorded_at=SYNTHETIC_TIME,
            key=request_key(body, repeat),
            repeat=repeat,
            finish_reason=self.finish_reason,
        )
