"""Requests-per-minute limiter, 429 handling and the free daily quota check.

Synthetic data: clocks are fake, rate-limit headers are made up, and the key
endpoint is served by an httpx MockTransport.
"""

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import SecretStr

from llmeval.quota import (
    MAX_SHORT_WAIT_S,
    QuotaExhausted,
    RateLimiter,
    free_daily_remaining,
    parse_rate_limit,
    wait_or_stop,
)

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
KEY = "synthetic-key"


class FakeClock:
    """A monotonic clock that only moves when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def reset_header(moment: datetime) -> dict[str, str]:
    return {"X-RateLimit-Reset": str(int(moment.timestamp() * 1000))}


def test_first_call_does_not_wait():
    clock = FakeClock()

    assert RateLimiter(18, clock=clock, sleep=clock.sleep).acquire() == 0
    assert clock.sleeps == []


def test_calls_are_spaced_evenly():
    clock = FakeClock()
    limiter = RateLimiter(18, clock=clock, sleep=clock.sleep)

    for _ in range(4):
        limiter.acquire()

    assert clock.sleeps == pytest.approx([60 / 18] * 3)


def test_no_minute_holds_more_than_rpm_calls():
    clock = FakeClock()
    limiter = RateLimiter(18, clock=clock, sleep=clock.sleep)
    moments = []
    for _ in range(60):
        limiter.acquire()
        moments.append(clock.now)

    # The tolerance absorbs float rounding: 18 steps of 60/18 s sum to 59.999... s.
    window = 60 - 1e-9
    busiest = max(sum(1 for m in moments if start <= m < start + window) for start in moments)

    assert busiest <= 18


def test_idle_time_does_not_allow_a_burst():
    clock = FakeClock()
    limiter = RateLimiter(18, clock=clock, sleep=clock.sleep)
    limiter.acquire()
    clock.now += 600

    limiter.acquire()
    limiter.acquire()

    assert clock.sleeps == pytest.approx([60 / 18])


def test_time_already_passed_counts():
    clock = FakeClock()
    limiter = RateLimiter(18, clock=clock, sleep=clock.sleep)
    limiter.acquire()
    clock.now += 2.0

    limiter.acquire()

    assert clock.sleeps == pytest.approx([60 / 18 - 2.0])


def test_rpm_must_be_positive():
    with pytest.raises(ValueError):
        RateLimiter(0)


def test_parse_reset_in_milliseconds_and_seconds():
    later = NOW + timedelta(minutes=5)

    assert parse_rate_limit(reset_header(later)).reset_at == later
    assert parse_rate_limit({"X-RateLimit-Reset": str(int(later.timestamp()))}).reset_at == later


def test_parse_remaining_and_missing_headers():
    limit = parse_rate_limit({"x-ratelimit-remaining": "0"})

    assert limit.remaining == 0
    assert limit.reset_at is None
    assert parse_rate_limit({"X-RateLimit-Reset": "soon"}).reset_at is None


def test_short_reset_means_wait_and_retry():
    wait = wait_or_stop(reset_header(NOW + timedelta(seconds=20)), now=NOW, attempt=0)

    assert wait == pytest.approx(20)


def test_long_reset_stops_the_run():
    tomorrow = NOW + timedelta(hours=12)

    with pytest.raises(QuotaExhausted) as caught:
        wait_or_stop(reset_header(tomorrow), now=NOW, attempt=0)

    assert caught.value.reset_at == tomorrow
    assert "free daily quota reached" in str(caught.value)
    assert "rerun after 2026-01-02 00:00 UTC" in str(caught.value)


def test_missing_reset_stops_the_run():
    with pytest.raises(QuotaExhausted, match="no reset time"):
        wait_or_stop({}, now=NOW, attempt=0)


def test_retries_are_limited_and_say_so():
    with pytest.raises(QuotaExhausted) as caught:
        wait_or_stop(reset_header(NOW + timedelta(seconds=5)), now=NOW, attempt=3)

    error = caught.value.with_progress(recorded=3, needed=9)
    assert str(error).startswith("rate limited after 3 retries; 3 of 9 calls recorded")
    assert "free daily quota" not in str(error)


def test_short_wait_limit_covers_one_minute_window():
    assert 60 <= MAX_SHORT_WAIT_S < 120


def test_progress_is_part_of_the_message():
    tomorrow = NOW + timedelta(hours=12)

    error = QuotaExhausted(tomorrow).with_progress(recorded=12, needed=40)

    assert str(error) == (
        "free daily quota reached; 12 of 40 calls recorded; rerun after 2026-01-02 00:00 UTC"
    )
    assert (error.recorded, error.needed) == (12, 40)


def test_free_daily_remaining_reads_the_key_endpoint():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = {"data": {"free_model_daily_requests": {"used": 8, "limit": 50, "remaining": 42}}}
        return httpx.Response(200, json=body)

    remaining = free_daily_remaining(SecretStr(KEY), transport=httpx.MockTransport(handler))

    assert remaining == 42
    assert seen[0].url.path == "/api/v1/key"
    assert seen[0].headers["authorization"] == f"Bearer {KEY}"


def test_free_daily_remaining_without_the_field_is_unknown():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": {}}))

    assert free_daily_remaining(SecretStr(KEY), transport=transport) is None
