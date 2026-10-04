"""The model client: one `complete()` call, three modes.

- live:   call OpenRouter and return the answer; nothing is written.
- record: call OpenRouter and append the call to the cassettes. A call that is
          already recorded is returned from the cassette instead, so an
          interrupted recording resumes without paying twice.
- replay: answer from the cassettes only. The client builds no HTTP client
          at all in this mode, so it cannot open a socket, and it needs no key.

Every evaluation layer (system under test and judge) goes through this class,
so everything replays in CI for free.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from typing import Any, Literal

import httpx

from llmeval.cassettes import (
    CallTag,
    CassetteEntry,
    CassetteStore,
    StoredResponse,
    Usage,
    request_key,
    utc_now,
)
from llmeval.config import Config, Mode
from llmeval.openrouter import OpenRouterError, http_client, read_json, require_key, send
from llmeval.pricing import ModelPrice, fetch_prices
from llmeval.quota import RateLimiter, wait_or_stop

__all__ = ["CallResult", "MissingRecording", "ModelClient", "Usage", "build_request"]

Message = Mapping[str, Any]


class MissingRecording(LookupError):
    """Replay found no cassette entry for a call."""

    def __init__(self, key: str, repeat: int, tag: CallTag | None) -> None:
        self.key = key
        self.repeat = repeat
        self.tag = tag
        where = tag.label(repeat) if tag else f"key {key[:12]}/{repeat}"
        super().__init__(f"no recording for {where}: run make record")


@dataclass(frozen=True)
class CallResult:
    """One model answer, the same shape whether it came live or from a cassette."""

    content: str
    model_requested: str
    model_used: str
    usage: Usage
    cost_usd: float
    cost_source: Literal["provider", "published_prices"]
    latency_ms: float
    recorded_at: datetime
    key: str
    repeat: int

    @classmethod
    def from_entry(cls, entry: CassetteEntry) -> CallResult:
        return cls(
            content=entry.response.content,
            model_requested=entry.request["model"],
            model_used=entry.response.model,
            usage=entry.usage,
            cost_usd=entry.cost_usd,
            cost_source=entry.cost_source,
            latency_ms=entry.latency_ms,
            recorded_at=entry.recorded_at,
            key=entry.key,
            repeat=entry.repeat,
        )


def build_request(
    messages: Sequence[Message],
    *,
    model: str,
    temperature: float,
    seed: int | None,
    max_tokens: int,
    response_format: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The exact JSON body sent to `/chat/completions` (and stored in the cassette)."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        body["seed"] = seed
    if response_format is not None:
        body["response_format"] = dict(response_format)
        # Route only to endpoints that support structured output, instead of
        # letting a provider silently ignore the schema.
        body["provider"] = {"require_parameters": True}
    return body


class ModelClient:
    def __init__(
        self,
        mode: Mode | str,
        store: CassetteStore,
        config: Config,
        transport: httpx.BaseTransport | None = None,
        *,
        limiter: RateLimiter | None = None,
        prices: Mapping[str, ModelPrice] | None = None,
        now: Callable[[], datetime] = utc_now,
        timer: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.mode = Mode(mode)
        self.store = store
        self.config = config
        self._prices = dict(prices or {})
        self._now = now
        self._timer = timer
        self._sleep = sleep
        self._transport = transport
        self._http: httpx.Client | None = None
        self._api_key: str | None = None
        if self.mode is not Mode.REPLAY:
            self._api_key = require_key(config.settings.api_key)
            self._limiter = limiter or RateLimiter(config.models.rpm)
            self._http = http_client(transport)

    def __enter__(self) -> ModelClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._http is not None:
            self._http.close()

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        seed: int | None,
        max_tokens: int,
        response_format: Mapping[str, Any] | None = None,
        repeat: int = 0,
        tag: CallTag | None = None,
    ) -> CallResult:
        body = build_request(
            messages,
            model=model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        key = request_key(body, repeat)

        if self.mode is Mode.REPLAY:
            entry = self.store.get(key)
            if entry is None:
                raise MissingRecording(key, repeat, tag)
            return CallResult.from_entry(entry)

        if self.mode is Mode.RECORD and (entry := self.store.get(key)) is not None:
            return CallResult.from_entry(entry)

        entry = self._call(body, key, repeat, tag)
        if self.mode is Mode.RECORD:
            self.store.append(entry)
        return CallResult.from_entry(entry)

    def _call(
        self, body: dict[str, Any], key: str, repeat: int, tag: CallTag | None
    ) -> CassetteEntry:
        assert self._http is not None  # live and record modes always have one
        requested_at = self._now()
        for attempt in count():
            self._limiter.acquire()
            started = self._timer()
            response = send(
                self._http, "POST", "/chat/completions", api_key=self._api_key, json_body=body
            )
            latency_ms = (self._timer() - started) * 1000
            if response.status_code != 429:
                break
            self._sleep(wait_or_stop(response.headers, now=self._now(), attempt=attempt))

        data = read_json(response, api_key=self._api_key)
        answer = _parse_answer(data, body["model"])
        usage = _parse_usage(data)
        cost_usd, cost_source, prices = self._cost(data, usage, body["model"])
        return CassetteEntry(
            key=key,
            repeat=repeat,
            tag=tag,
            request=body,
            response=answer,
            usage=usage,
            cost_usd=cost_usd,
            cost_source=cost_source,
            prices=prices.snapshot() if prices else None,
            latency_ms=latency_ms,
            requested_at=requested_at,
            recorded_at=self._now(),
        )

    def _cost(
        self, data: Mapping[str, Any], usage: Usage, model_requested: str
    ) -> tuple[float, Literal["provider", "published_prices"], ModelPrice | None]:
        provider_cost = data["usage"].get("cost")
        if isinstance(provider_cost, int | float):
            return float(provider_cost), "provider", None
        # OpenRouter bills by the requested model id (a ":free" id costs 0),
        # so its published price is the fallback.
        if model_requested not in self._prices:
            self._prices.update(
                fetch_prices([model_requested], transport=self._transport, now=self._now)
            )
        price = self._prices[model_requested]
        return price.cost(usage), "published_prices", price


def _parse_answer(data: Mapping[str, Any], model_requested: str) -> StoredResponse:
    choices = data.get("choices")
    if not choices:
        error = data.get("error")
        detail = error.get("message") if isinstance(error, dict) else error
        raise OpenRouterError(f"response has no answer: {detail or 'no choices'}")
    choice = choices[0]
    return StoredResponse(
        id=data.get("id"),
        model=data.get("model") or model_requested,
        # A model can return no text (for example, only reasoning); that is a
        # result worth evaluating, not a transport error.
        content=(choice.get("message") or {}).get("content") or "",
        finish_reason=choice.get("finish_reason"),
    )


def _parse_usage(data: Mapping[str, Any]) -> Usage:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        raise OpenRouterError("response has no usage block")
    return Usage(
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
    )
