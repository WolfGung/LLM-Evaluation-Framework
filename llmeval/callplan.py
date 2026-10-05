"""The call plan of a full recording, how much of it is recorded, and what the rest costs.

`llmeval estimate`, `llmeval status` and `llmeval record` share this module:

- `full_plan`: every call of the recording, from `config/models.yaml`
  (`repeats`, `stability_cases`, `judge_repeats`), the datasets, every prompt
  version and the judge rubric (see `llmeval.runner.plan_requests`).
- `count_plan`: the counts by function, version and repeat, and how many are
  recorded. Counts are of distinct cassette keys: identical requests share
  one recording. A judge call has a key only once the answers it grades are
  recorded; until then it counts as one call each, so the judge count is an
  upper bound (identical answers share a grading and need no pairwise call).
- `quota_lines`: the free-model arithmetic, from OpenRouter's published
  limits (`llmeval.quota.FREE_LIMITS`, with the date they were checked).
- `estimate_remaining_cost`: the cost of the calls still to record, per
  model role: `$0.00` for a `:free` model id; else the highest recorded
  `usage.cost` per call of that model, if any call is recorded; else the
  published prices, every call at its full token budget (an upper bound).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import httpx

from app.retrieval import BM25Index
from llmeval.cassettes import CassetteStore, utc_now
from llmeval.checks.judge import JUDGE_FUNCTION, PAIRWISE_FUNCTION, Rubric
from llmeval.config import ModelsConfig, RoleConfig
from llmeval.datasets import RagCase, TriageCase
from llmeval.perf import cost_sum
from llmeval.pricing import PlannedCall, estimate_cost, estimate_prompt_tokens, fetch_prices
from llmeval.quota import FREE_LIMITS
from llmeval.runner import EVAL_FUNCTIONS, PlannedRequest, plan_requests, recorded_answers

FREE_SUFFIX = ":free"
ROLES: tuple[Literal["system", "judge"], ...] = ("system", "judge")


def is_free(model: str) -> bool:
    """A free model variant: OpenRouter ids that end in `:free` cost nothing."""
    return model.endswith(FREE_SUFFIX)


@dataclass(frozen=True)
class PlanInputs:
    """What a recording covers: the model config, the cases, the prompt
    versions per function and the judge rubric."""

    models: ModelsConfig
    rag_cases: Sequence[RagCase]
    triage_cases: Sequence[TriageCase]
    rubric: Rubric
    versions: Mapping[str, Sequence[str]]
    index: BM25Index | None = None


def full_plan(inputs: PlanInputs, store: CassetteStore | None) -> list[PlannedRequest]:
    """Every call of the recording. With `store`, judge calls whose answers are
    recorded get their keys (see `recorded_answers`)."""
    models = inputs.models
    return plan_requests(
        models.system,
        rag_cases=inputs.rag_cases,
        triage_cases=inputs.triage_cases,
        versions=inputs.versions,
        repeats=models.repeats,
        stability_cases=models.stability_cases,
        index=inputs.index,
        judge=models.judge,
        rubric=inputs.rubric,
        answers=recorded_answers(store) if store is not None else None,
        judge_repeats=models.judge_repeats,
    )


def to_record(plan: Sequence[PlannedRequest], store: CassetteStore | None) -> list[PlannedRequest]:
    """The planned calls still to record: one per distinct key not in `store`,
    plus every judge call still waiting for its answers."""
    seen: set[str] = set()
    left = []
    for p in plan:
        if p.key is None:
            left.append(p)
        elif p.key not in seen and (store is None or p.key not in store):
            seen.add(p.key)
            left.append(p)
    return left


@dataclass(frozen=True)
class Row:
    """The calls of one evaluated function and version (or one version pair)."""

    label: str
    system: int
    by_repeat: Mapping[int, int]
    judge: int
    judge_by_repeat: Mapping[int, int]
    waiting: int
    recorded: int

    @property
    def total(self) -> int:
        return self.system + self.judge


@dataclass(frozen=True)
class PlanCounts:
    """Distinct calls of a plan and how many are recorded.

    `waiting` judge calls have no key yet; each counts as one, so `judge` and
    `total` are upper bounds while `waiting` is above zero (`exact` is False).
    `free_to_record` counts the calls still to record whose model is `:free`.
    """

    rows: tuple[Row, ...]
    system: int
    judge: int
    waiting: int
    recorded: int
    free_to_record: int

    @property
    def total(self) -> int:
        return self.system + self.judge

    @property
    def to_record(self) -> int:
        return self.total - self.recorded

    @property
    def exact(self) -> bool:
        return self.waiting == 0


def _repeats(planned: Sequence[PlannedRequest]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for p in planned:
        counts[p.repeat] = counts.get(p.repeat, 0) + 1
    return dict(sorted(counts.items()))


def _distinct(planned: Sequence[PlannedRequest]) -> int:
    return len({p.key for p in planned if p.key}) + sum(p.key is None for p in planned)


def _row(
    label: str,
    system: Sequence[PlannedRequest],
    judge: Sequence[PlannedRequest],
    store: CassetteStore | None,
) -> Row:
    keys = {p.key for p in (*system, *judge) if p.key}
    return Row(
        label=label,
        system=_distinct(system),
        by_repeat=_repeats(system),
        judge=_distinct(judge),
        judge_by_repeat=_repeats(judge),
        waiting=sum(p.key is None for p in judge),
        recorded=sum(key in store for key in keys) if store is not None else 0,
    )


def count_plan(
    plan: Sequence[PlannedRequest], store: CassetteStore | None, models: ModelsConfig
) -> PlanCounts:
    """Counts by function and version: a row per system function and version
    (with the gradings of its answers), and a row per pairwise version pair."""
    rows = []
    for function in EVAL_FUNCTIONS:
        versions = dict.fromkeys(p.version for p in plan if p.function == function)
        for version in versions:
            system = [p for p in plan if p.function == function and p.version == version]
            judge = [
                p
                for p in plan
                if function == "rag" and p.function == JUDGE_FUNCTION and p.version == version
            ]
            rows.append(_row(f"{function} {version}", system, judge, store))
    for pair in dict.fromkeys(p.version for p in plan if p.function == PAIRWISE_FUNCTION):
        pairwise = [p for p in plan if p.function == PAIRWISE_FUNCTION and p.version == pair]
        first, second = pair.split("-", 1)
        rows.append(_row(f"rag {first} vs {second}", (), pairwise, store))
    system = [p for p in plan if p.role == "system"]
    judge = [p for p in plan if p.role == "judge"]
    known = {p.key for p in plan if p.key}
    left = to_record(plan, store)
    return PlanCounts(
        rows=tuple(rows),
        system=_distinct(system),
        judge=_distinct(judge),
        waiting=sum(p.key is None for p in plan),
        recorded=sum(key in store for key in known) if store is not None else 0,
        free_to_record=sum(is_free(models.role(p.role).model) for p in left),
    )


def up_to(count: int, upper_bound: bool) -> str:
    return f"up to {count}" if upper_bound else str(count)


def _by_repeat(counts: Mapping[int, int]) -> str:
    return ", ".join(f"repeat {repeat}: {n}" for repeat, n in counts.items())


def plan_lines(counts: PlanCounts, models: ModelsConfig) -> list[str]:
    """The call plan as printed by `estimate` and `status`."""
    subset = ", ".join(models.stability_cases) if models.stability_cases else "every case"
    lines = [
        f"call plan: repeats {models.repeats}, judge_repeats {models.judge_repeats}, "
        f"stability_cases: {subset}"
    ]
    for row in counts.rows:
        parts = []
        if row.system:
            parts.append(f"{row.system} system calls ({_by_repeat(row.by_repeat)})")
        bound = row.waiting > 0
        if row.judge:
            kind = "pairwise judge" if " vs " in row.label else "judge"
            repeats = _by_repeat(row.judge_by_repeat)
            parts.append(f"{up_to(row.judge, bound)} {kind} calls ({repeats})")
        total = up_to(row.total, bound)
        lines.append(f"  {row.label}: {' + '.join(parts)} = {total}; recorded {row.recorded}")
    bound = not counts.exact
    lines.append(
        f"total: {counts.system} system calls + {up_to(counts.judge, bound)} judge calls = "
        f"{up_to(counts.total, bound)} distinct calls; recorded {counts.recorded}, "
        f"to record {up_to(counts.to_record, bound)}"
    )
    if bound:
        lines.append(
            f"{counts.waiting} judge calls are planned once the answers they grade are recorded; "
            "until then their count is an upper bound (identical answers share a grading "
            "and need no pairwise call)"
        )
    return lines


def quota_lines(counts: PlanCounts, models: ModelsConfig) -> list[str]:
    """The published free-model limits and the days the calls still to record take."""
    lines = [FREE_LIMITS.describe()]
    calls = counts.free_to_record
    if not calls:
        if any(is_free(models.role(name).model) for name in ROLES):
            lines.append("free-model calls to record: none")
        else:
            lines.append(
                f"free-model calls to record: none: the config names no {FREE_SUFFIX} model, "
                "so the free-model limits do not apply"
            )
        return lines
    at_low, at_high = FREE_LIMITS.days(calls)
    minutes = math.ceil(calls / models.rpm)
    lines.append(
        f"free-model calls to record: {up_to(calls, not counts.exact)}: "
        f"{_plural(at_low, 'day')} at {FREE_LIMITS.per_day} a day, "
        f"{_plural(at_high, 'day')} at {FREE_LIMITS.per_day_with_credits} a day; "
        f"about {_plural(minutes, 'minute')} of calls at rpm {models.rpm}"
    )
    return lines


def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}" if n == 1 else f"{n} {unit}s"


@dataclass(frozen=True)
class CostEstimate:
    """The estimated cost of the calls still to record, and what is already spent.

    `spent_usd` sums the known recorded costs of planned calls;
    `spent_unknown` counts recorded calls whose cost is unknown (never zero).
    """

    usd: float
    spent_usd: float
    spent_unknown: int
    lines: tuple[str, ...]

    def headline(self) -> str:
        amount = "$0.00" if self.usd == 0 else f"${self.usd:.4f}"
        return f"estimated cost of the calls still to record: {amount}"


def _prompt_tokens(planned: PlannedRequest, system: RoleConfig) -> int:
    tokens = estimate_prompt_tokens(planned.messages)
    if planned.key is None:
        # The answers are not recorded yet: leave room for each at its budget.
        tokens += len(planned.grades) * system.max_tokens
    return tokens


def estimate_remaining_cost(
    plan: Sequence[PlannedRequest],
    store: CassetteStore | None,
    models: ModelsConfig,
    *,
    transport: httpx.BaseTransport | None = None,
    now: Callable[[], datetime] = utc_now,
) -> CostEstimate:
    """Estimate the calls still to record (see the module docstring).

    Raises `PricingError` (or `OpenRouterError`) when a paid model has no
    recorded cost and no published price: then no honest estimate exists.
    """
    left = to_record(plan, store)
    entries = list(store) if store is not None else []
    lines: list[str] = []
    by_role: dict[str, list[PlannedRequest]] = {
        name: [p for p in left if p.role == name] for name in ROLES
    }
    history = {
        name: [
            entry.cost_usd
            for entry in entries
            if entry.request.get("model") == models.role(name).model
            and entry.cost_source == "provider"
            and entry.cost_usd is not None
        ]
        for name in ROLES
    }
    need_prices = [
        models.role(name).model
        for name in ROLES
        if by_role[name] and not is_free(models.role(name).model) and not history[name]
    ]
    prices = fetch_prices(need_prices, transport=transport, now=now) if need_prices else {}
    total = 0.0
    for name in ROLES:
        role: RoleConfig = models.role(name)
        calls = by_role[name]
        count = up_to(len(calls), any(p.key is None for p in calls))
        label = f"{name} ({role.model})"
        if not calls:
            lines.append(f"{label}: nothing left to record")
        elif is_free(role.model):
            lines.append(f"{label}: $0.00 for {count} calls (a {FREE_SUFFIX} model id)")
        elif history[name]:
            per_call = max(history[name])
            cost = per_call * len(calls)
            total += cost
            lines.append(
                f"{label}: ${cost:.4f} for {count} calls at ${per_call:.6f} each, "
                f"the highest usage.cost among {len(history[name])} recorded calls"
            )
        else:
            sized = [PlannedCall.for_role(role, _prompt_tokens(p, models.system)) for p in calls]
            cost = estimate_cost(sized, prices)
            total += cost
            lines.append(
                f"{label}: ${cost:.4f} for {count} calls from the published prices "
                "(GET /api/v1/models), each call at its full token budget"
            )
    planned_keys = {p.key for p in plan if p.key}
    spent, unknown = cost_sum(entry.cost_usd for entry in entries if entry.key in planned_keys)
    recorded = sum(entry.key in planned_keys for entry in entries)
    if unknown:
        lines.append(
            f"recorded so far: ${spent:.6f} over {recorded - unknown} calls with a known cost, "
            f"plus {unknown} calls with unknown cost"
        )
    elif recorded:
        lines.append(f"recorded so far: ${spent:.6f} over {recorded} calls")
    return CostEstimate(usd=total, spent_usd=spent, spent_unknown=unknown, lines=tuple(lines))
