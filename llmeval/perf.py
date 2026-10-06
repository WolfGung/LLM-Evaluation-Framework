"""Performance: latency, tokens and cost of the model calls, by function and version.

Every call record carries what the API reported when the call was recorded:
latency, token counts, and the cost from OpenRouter's `usage.cost` (or from
the published prices when that field was absent). Replay reads these values
from the cassettes, so the measures below are reproducible.

- Latency: p50 and p95 by the nearest-rank method (the smallest recorded
  value with at least that share of calls at or below it). With few calls
  p95 is close to the slowest call.
- Tokens: mean prompt tokens (in), completion tokens (out) and the reasoning
  tokens inside the completion.
- Cost, in USD: `total_usd` is the sum of the known costs. A call whose cost
  is unknown (`cost_source: unknown`) is counted in `unknown_calls`, never as
  zero, and then no mean is given, because it would understate the cost.
  `per_case_usd` divides by the cases, `per_run_usd` by the runs; a run (one
  repeat of one case, or one pairwise question) is one call.
- Tail: the calls at or above p95, slowest first, each next to the fastest
  other repeat of the same request (`TailCall`). Only for calls whose repeats
  send the same request, the system's answers: when a repeat of the same
  request wrote as much in far less time, the slow call's extra time went to
  waiting for the provider, not to the work. None for the judge's grades and
  the pairwise questions.

Two prompt versions that wrote the same answer share one grading recording;
each version's measures count it, so the sum over versions can exceed what
the recording cost. `llmeval status` counts distinct recordings.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

COST_PLACES = 8


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Latency(_Record):
    p50_ms: float | None
    p95_ms: float | None


class Cost(_Record):
    """USD; see the module docstring for what is summed and when a mean is given."""

    total_usd: float
    unknown_calls: int
    per_case_usd: float | None
    per_run_usd: float | None


class TailCall(_Record):
    """One call at or above p95: its case, repeat, latency and completion
    tokens, and the same of the fastest other repeat of the same request
    (None when the request ran once)."""

    case: str
    repeat: int
    latency_ms: float
    completion_tokens: int
    fastest_repeat: int | None
    fastest_repeat_latency_ms: float | None
    fastest_repeat_completion_tokens: int | None


class Performance(_Record):
    calls: int
    latency: Latency
    mean_prompt_tokens: float | None
    mean_completion_tokens: float | None
    mean_reasoning_tokens: float | None
    cost: Cost
    tail: list[TailCall] | None = None


class CallMeasures(Protocol):
    """What a call record holds about one call (see `llmeval.results.CallRecord`)."""

    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float | None


def percentile(values: Sequence[float], p: float) -> float | None:
    """The nearest-rank percentile `p` (0-100) of `values`; None for no values."""
    if not 0 <= p <= 100:
        raise ValueError("a percentile is between 0 and 100")
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def cost_sum(costs: Iterable[float | None]) -> tuple[float, int]:
    """(sum of the known costs, number of unknown costs). None is never counted as 0."""
    known = 0.0
    unknown = 0
    for cost in costs:
        if cost is None:
            unknown += 1
        else:
            known += cost
    return round(known, COST_PLACES), unknown


def _mean(values: Sequence[float], places: int = 2) -> float | None:
    return round(sum(values) / len(values), places) if values else None


def _per(total: float, unknown: int, count: int | None) -> float | None:
    if unknown or not count:
        return None
    return round(total / count, COST_PLACES)


def _ms(value: float) -> float:
    return round(value, 1)


def tail_calls(
    calls: Sequence[CallMeasures], requests: Sequence[tuple[str, int]]
) -> list[TailCall]:
    """The calls at or above the p95 latency of `calls`, slowest first, each
    with the fastest other repeat of its request. `requests` names each
    call's request and repeat, as (case, repeat): calls with the same case
    sent the same request. Latencies are compared as stored, to 0.1 ms."""
    if len(calls) != len(requests):
        raise ValueError("one request per call")
    timed = [
        (case, repeat, _ms(call.latency_ms), call.completion_tokens)
        for call, (case, repeat) in zip(calls, requests, strict=True)
    ]
    p95 = percentile([ms for _, _, ms, _ in timed], 95)
    if p95 is None:
        return []
    by_case: dict[str, list[tuple[str, int, float, int]]] = {}
    for item in timed:
        by_case.setdefault(item[0], []).append(item)
    tail = []
    for case, repeat, ms, tokens in sorted(timed, key=lambda item: (-item[2], item[0], item[1])):
        if ms < p95:
            break
        others = [item for item in by_case[case] if item[1] != repeat]
        fastest = min(others, key=lambda item: (item[2], item[1])) if others else None
        tail.append(
            TailCall(
                case=case,
                repeat=repeat,
                latency_ms=ms,
                completion_tokens=tokens,
                fastest_repeat=fastest[1] if fastest else None,
                fastest_repeat_latency_ms=fastest[2] if fastest else None,
                fastest_repeat_completion_tokens=fastest[3] if fastest else None,
            )
        )
    return tail


def performance(
    calls: Sequence[CallMeasures],
    *,
    cases: int | None = None,
    requests: Sequence[tuple[str, int]] | None = None,
) -> Performance:
    """The measures of `calls`. `cases` is how many cases they cover (None:
    unknown). `requests` names each call's request as (case, repeat) when the
    repeats of a case send the same request; only then is the tail measured."""
    latencies = [call.latency_ms for call in calls]
    p50, p95 = percentile(latencies, 50), percentile(latencies, 95)
    total, unknown = cost_sum(call.cost_usd for call in calls)
    return Performance(
        calls=len(calls),
        latency=Latency(
            p50_ms=None if p50 is None else round(p50, 1),
            p95_ms=None if p95 is None else round(p95, 1),
        ),
        mean_prompt_tokens=_mean([call.prompt_tokens for call in calls]),
        mean_completion_tokens=_mean([call.completion_tokens for call in calls]),
        mean_reasoning_tokens=_mean([call.reasoning_tokens for call in calls]),
        cost=Cost(
            total_usd=total,
            unknown_calls=unknown,
            per_case_usd=_per(total, unknown, cases),
            per_run_usd=_per(total, unknown, len(calls)),
        ),
        tail=None if requests is None else tail_calls(calls, requests),
    )
