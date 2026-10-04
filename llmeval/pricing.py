"""Published prices, cost of a call, cost estimate of a run, and the budget guard.

There are no hard-coded prices. Prices come from OpenRouter's public model list
(`GET /api/v1/models`, USD per token as strings) at the start of a live or
record run. A recorded call uses the cost OpenRouter reports in `usage.cost`;
published prices are the fallback when that field is absent.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation

import httpx

from llmeval.cassettes import StoredPrices, Usage, utc_now
from llmeval.openrouter import get_json, http_client

# English text averages about 4 characters per token. Assuming 3 makes the
# estimate an upper bound, which is the safe side for a spend limit.
CHARS_PER_TOKEN = 3
TOKENS_PER_MESSAGE = 4


class PricingError(RuntimeError):
    """A price is missing or unusable, so no honest cost can be given."""


class BudgetExceeded(RuntimeError):
    """The estimated cost of a run is above MAX_RUN_COST_USD."""

    def __init__(self, estimate: float, limit: float) -> None:
        super().__init__(
            f"estimated cost ${estimate:.4f} exceeds MAX_RUN_COST_USD=${limit:.2f}; "
            "nothing was sent. Lower the plan or raise the limit."
        )
        self.estimate = estimate
        self.limit = limit


@dataclass(frozen=True)
class ModelPrice:
    """USD per token for one model, as published at `fetched_at`."""

    model: str
    prompt: Decimal
    completion: Decimal
    fetched_at: datetime

    def cost(self, usage: Usage) -> float:
        total = usage.prompt_tokens * self.prompt + usage.completion_tokens * self.completion
        return float(total)

    def snapshot(self) -> StoredPrices:
        return StoredPrices(
            model=self.model,
            prompt=str(self.prompt),
            completion=str(self.completion),
            fetched_at=self.fetched_at,
        )


@dataclass(frozen=True)
class PlannedCall:
    """One call a run intends to make, sized for the estimate."""

    model: str
    prompt_tokens: int
    max_completion_tokens: int


def fetch_prices(
    models: Iterable[str],
    *,
    transport: httpx.BaseTransport | None = None,
    now: Callable[[], datetime] = utc_now,
) -> dict[str, ModelPrice]:
    """Read the published prices of `models`. Live and record runs only: this calls the API."""
    wanted = set(models)
    with http_client(transport) as client:
        listing = get_json(client, "/models")
    fetched_at = now()

    prices: dict[str, ModelPrice] = {}
    for item in listing.get("data", []):
        model = item.get("id")
        if model in wanted:
            prices[model] = _parse_price(model, item.get("pricing") or {}, fetched_at)

    missing = sorted(wanted - prices.keys())
    if missing:
        raise PricingError(f"no published price for: {', '.join(missing)}")
    return prices


def estimate_prompt_tokens(messages: Sequence[Mapping[str, object]]) -> int:
    chars = sum(len(str(message.get("content", ""))) for message in messages)
    return math.ceil(chars / CHARS_PER_TOKEN) + TOKENS_PER_MESSAGE * len(messages)


def estimate_cost(plan: Iterable[PlannedCall], prices: Mapping[str, ModelPrice]) -> float:
    """Upper-bound cost of a plan: every call is assumed to use its full max_tokens."""
    total = Decimal(0)
    for call in plan:
        price = prices.get(call.model)
        if price is None:
            raise PricingError(f"no price for {call.model}; fetch prices before estimating")
        total += call.prompt_tokens * price.prompt + call.max_completion_tokens * price.completion
    return float(total)


def check_budget(estimate: float, limit: float) -> None:
    if estimate > limit:
        raise BudgetExceeded(estimate, limit)


def _parse_price(model: str, pricing: Mapping[str, object], fetched_at: datetime) -> ModelPrice:
    try:
        prompt = Decimal(str(pricing["prompt"]))
        completion = Decimal(str(pricing["completion"]))
    except (KeyError, InvalidOperation):
        raise PricingError(f"no published price for: {model}") from None
    if prompt < 0 or completion < 0:
        # OpenRouter marks routers with a variable price as -1.
        raise PricingError(f"no published price for: {model} (price varies)")
    return ModelPrice(model, prompt, completion, fetched_at)

