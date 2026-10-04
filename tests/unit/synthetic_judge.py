"""A synthetic judge transport for the judge tests.

Synthetic data: every verdict served here is a made-up JSON string chosen by
the test, never a model output. Calls go through the real `ModelClient` with
an httpx MockTransport, and cassettes are written only into pytest's
`tmp_path`, never into `cassettes/`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr

from app.retrieval import Hit
from llmeval.cassettes import CassetteStore
from llmeval.client import ModelClient
from llmeval.config import Config, Mode, Settings, load_models_config

ROOT = Path(__file__).resolve().parents[2]
# The repository's own roles, so the tests check the judge role from config.
MODELS = load_models_config(ROOT / "config" / "models.yaml")
NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

DOCS = (
    Hit("kb-alpha", 3.2, "Returns\n\nSynthetic shop: returns are accepted within 30 days."),
    Hit("kb-beta", 1.1, "Shipping\n\nSynthetic shop: standard shipping costs $5.95."),
)
QUESTION = "How many days do I have for returns?"
ANSWER = "You have 30 days to return an item [kb-alpha]."


def verdict(
    groundedness: Any = 5,
    helpfulness: Any = 5,
    tone: Any = 5,
    passed: Any = True,
    reasons: Any = "Synthetic reasons.",
) -> str:
    """A verdict as the judge would write it (JSON text)."""
    return json.dumps(
        {
            "groundedness": groundedness,
            "helpfulness": helpfulness,
            "tone": tone,
            "pass": passed,
            "reasons": reasons,
        }
    )


def preference(preferred: str, reasons: str = "Synthetic reasons.") -> str:
    """A pairwise verdict as the judge would write it (JSON text)."""
    return json.dumps({"preferred": preferred, "reasons": reasons})


Reply = str | list[str] | Callable[[dict[str, Any]], str]


class SyntheticTransport:
    """Mock OpenRouter: answers with `reply` (one text, a scripted list, or a
    function of the request body) and keeps every request body it saw."""

    def __init__(self, reply: Reply, finish_reason: str = "stop") -> None:
        self.reply = reply
        self.finish_reason = finish_reason
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if callable(self.reply):
            content = self.reply(body)
        elif isinstance(self.reply, list):
            content = self.reply[len(self.bodies) - 1]
        else:
            content = self.reply
        return httpx.Response(
            200,
            json={
                "id": f"gen-synthetic-{len(self.bodies)}",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": self.finish_reason,
                        "message": {"content": content},
                    }
                ],
                "usage": {"prompt_tokens": 400, "completion_tokens": 60, "cost": 0.0},
            },
        )


def refuse_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"replay must not send a request: {request.url}")


class NoWait:
    def acquire(self) -> float:
        return 0.0


def make_client(
    cassettes: Path,
    mode: Mode,
    handler: Callable[[httpx.Request], httpx.Response] = refuse_network,
    *,
    now: datetime = NOW,
) -> ModelClient:
    """A real client over `cassettes` with the repository's model roles."""
    key = None if mode is Mode.REPLAY else SecretStr("synthetic-key-123")
    config = Config(models=MODELS, settings=Settings(api_key=key))
    return ModelClient(
        mode,
        CassetteStore(cassettes),
        config,
        httpx.MockTransport(handler),
        limiter=NoWait(),
        now=lambda: now,
    )


def user_turn(body: dict[str, Any]) -> str:
    return body["messages"][-1]["content"]
