"""Published prices, cost estimates and the budget guard.

Synthetic data: the model list and prices are made up and served by an httpx
MockTransport. They are not OpenRouter's real prices.
"""

from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest

from llmeval.cassettes import Usage
from llmeval.pricing import (
    BudgetExceeded,
    ModelPrice,
    PlannedCall,
    PricingError,
    check_budget,
    estimate_cost,
    estimate_prompt_tokens,
    fetch_prices,
)

MODELS = {
    "data": [
        {"id": "vendor-a/small:free", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "vendor-b/paid", "pricing": {"prompt": "0.000001", "completion": "0.000004"}},
        {"id": "vendor-c/router", "pricing": {"prompt": "-1", "completion": "-1"}},
    ]
}
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def models_transport(calls: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=MODELS)

    return httpx.MockTransport(handler)


def price(model="vendor-b/paid", prompt="0.000001", completion="0.000004") -> ModelPrice:
    return ModelPrice(model, Decimal(prompt), Decimal(completion), NOW)


def test_fetch_prices_reads_the_models_list():
    calls: list[httpx.Request] = []

    prices = fetch_prices(
        ["vendor-a/small:free", "vendor-b/paid"], transport=models_transport(calls), now=lambda: NOW
    )

    assert prices["vendor-b/paid"] == price()
    assert prices["vendor-a/small:free"].prompt == Decimal("0")
    assert [c.url.path for c in calls] == ["/api/v1/models"]
    assert "authorization" not in calls[0].headers


def test_fetch_prices_names_unknown_models():
    with pytest.raises(PricingError, match="vendor-x/missing"):
        fetch_prices(["vendor-x/missing"], transport=models_transport([]), now=lambda: NOW)


def test_fetch_prices_refuses_unpublished_prices():
    with pytest.raises(PricingError, match="vendor-c/router"):
        fetch_prices(["vendor-c/router"], transport=models_transport([]), now=lambda: NOW)


def test_cost_from_published_prices():
    usage = Usage(prompt_tokens=1000, completion_tokens=500)

    assert price().cost(usage) == pytest.approx(0.001 + 0.002)


def test_price_snapshot_keeps_the_original_strings():
    snapshot = price().snapshot()

    assert snapshot.prompt == "0.000001"
    assert snapshot.completion == "0.000004"
    assert snapshot.fetched_at == NOW


def test_free_models_estimate_to_zero():
    prices = {"vendor-a/small:free": price("vendor-a/small:free", "0", "0")}
    plan = [PlannedCall("vendor-a/small:free", prompt_tokens=5000, max_completion_tokens=600)] * 50

    assert estimate_cost(plan, prices) == 0.0


def test_paid_estimate_uses_max_completion_tokens():
    plan = [PlannedCall("vendor-b/paid", prompt_tokens=1000, max_completion_tokens=500)] * 2

    assert estimate_cost(plan, {"vendor-b/paid": price()}) == pytest.approx(0.006)


def test_estimate_needs_a_price_for_every_model():
    with pytest.raises(PricingError, match="vendor-b/paid"):
        estimate_cost([PlannedCall("vendor-b/paid", 1, 1)], {})


def test_prompt_token_estimate_is_conservative():
    messages = [{"role": "user", "content": "x" * 300}]

    # Real tokenisers give about 4 characters per token for English; the
    # estimate assumes 3 so the budget guard errs on the safe side.
    assert estimate_prompt_tokens(messages) >= 100


def test_budget_guard_allows_the_limit_itself():
    check_budget(1.00, limit=1.00)


def test_budget_guard_stops_above_the_limit():
    with pytest.raises(BudgetExceeded) as caught:
        check_budget(1.25, limit=1.00)

    assert "1.25" in str(caught.value)
    assert "MAX_RUN_COST_USD" in str(caught.value)
    assert caught.value.estimate == 1.25
