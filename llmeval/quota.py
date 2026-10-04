"""Stay inside OpenRouter's free limits: requests per minute and requests per day.

Free model variants allow about 20 requests per minute and 50 per day. A
recording therefore has to pace itself (`RateLimiter`), wait out a short
per-minute 429, and stop cleanly with a clear message when the daily quota is
gone (`QuotaExhausted`). Every finished call is already on disk, so the owner
simply reruns `make record` after the reset time.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from pydantic import SecretStr

from llmeval.openrouter import get_json, http_client, require_key

# A per-minute limit resets within a minute; anything later is the daily quota.
MAX_SHORT_WAIT_S = 65.0
MAX_RETRIES = 3


class RateLimiter:
    """Token bucket that holds one token and refills it every 60/rpm seconds.

    A one-token bucket spaces calls evenly, so no 60-second window ever holds
    more than `rpm` calls, even right after an idle period.
    """

    def __init__(
        self,
        rpm: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rpm < 1:
            raise ValueError("rpm must be at least 1")
        self.interval = 60.0 / rpm
        self._clock = clock
        self._sleep = sleep
        self._tokens = 1.0
        self._last = clock()

    def acquire(self) -> float:
        """Block until a call is allowed; return the seconds waited."""
        now = self._clock()
        self._tokens = min(1.0, self._tokens + (now - self._last) / self.interval)
        self._last = now
        wait = 0.0
        if self._tokens < 1.0:
            wait = (1.0 - self._tokens) * self.interval
            self._sleep(wait)
            self._last = now + wait
            self._tokens = 1.0
        self._tokens -= 1.0
        return wait


@dataclass(frozen=True)
class RateLimit:
    """What a rate-limited response says about when to try again."""

    reset_at: datetime | None
    remaining: int | None


class QuotaExhausted(RuntimeError):
    """The free quota is used up; the run stops and can be resumed after `reset_at`."""

    headline = "free daily quota reached"

    def __init__(
        self, reset_at: datetime | None, recorded: int | None = None, needed: int | None = None
    ) -> None:
        self.reset_at = reset_at
        self.recorded = recorded
        self.needed = needed
        super().__init__(self._message())

    def with_progress(self, recorded: int, needed: int) -> QuotaExhausted:
        """The same error with the run's progress, for the final message."""
        return type(self)(self.reset_at, recorded, needed)

    def _message(self) -> str:
        parts = [self.headline]
        if self.recorded is not None and self.needed is not None:
            parts.append(f"{self.recorded} of {self.needed} calls recorded")
        if self.reset_at is None:
            parts.append("rerun later (the API gave no reset time)")
        else:
            parts.append(f"rerun after {self.reset_at.astimezone(UTC):%Y-%m-%d %H:%M} UTC")
        return "; ".join(parts)


class RateLimitRetriesExhausted(QuotaExhausted):
    """Short per-minute waits did not help; the run stops the same way."""

    headline = f"rate limited after {MAX_RETRIES} retries"


def parse_rate_limit(headers: Mapping[str, str]) -> RateLimit:
    lowered = {name.lower(): value for name, value in headers.items()}
    return RateLimit(
        reset_at=_parse_reset(lowered.get("x-ratelimit-reset")),
        remaining=_parse_int(lowered.get("x-ratelimit-remaining")),
    )


def wait_or_stop(headers: Mapping[str, str], *, now: datetime, attempt: int) -> float:
    """Decide what to do after HTTP 429.

    Returns the seconds to wait before retrying when the limit resets soon
    (the per-minute limit). Raises `QuotaExhausted` when it resets later (the
    daily quota) or the reset time is unknown, and `RateLimitRetriesExhausted`
    after `MAX_RETRIES` short waits.
    """
    limit = parse_rate_limit(headers)
    if limit.reset_at is None:
        raise QuotaExhausted(None)
    wait = (limit.reset_at - now).total_seconds()
    if wait > MAX_SHORT_WAIT_S:
        raise QuotaExhausted(limit.reset_at)
    if attempt >= MAX_RETRIES:
        raise RateLimitRetriesExhausted(limit.reset_at)
    return max(wait, 1.0)


def free_daily_remaining(
    api_key: SecretStr | None, *, transport: httpx.BaseTransport | None = None
) -> int | None:
    """Free-model requests left today for this key, from `GET /api/v1/key`.

    Returns None when the API does not report the number.
    """
    key = require_key(api_key)
    with http_client(transport) as client:
        body = get_json(client, "/key", api_key=key)
    data = body.get("data", body)
    quota = data.get("free_model_daily_requests") if isinstance(data, dict) else None
    remaining = quota.get("remaining") if isinstance(quota, dict) else None
    return remaining if isinstance(remaining, int) else None


def _parse_reset(value: str | None) -> datetime | None:
    number = _parse_int(value)
    if number is None:
        return None
    # OpenRouter sends epoch milliseconds; accept seconds as well.
    seconds = number / 1000 if number > 10**11 else number
    return datetime.fromtimestamp(seconds, UTC)


def _parse_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None
