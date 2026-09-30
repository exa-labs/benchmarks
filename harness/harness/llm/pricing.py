"""Published list prices for the models and hosted tools the harness drives.

Prices are USD per million tokens, copied from each provider's public pricing
page. A model missing from the table still runs; its generations report
``cost_usd=None`` and run summaries mark the cost as unknown instead of
counting it as zero. Override or extend the table with ``register_price``.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.llm.types import Usage


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token rates. ``cache_creation`` applies to cache-write tokens."""

    input: float
    output: float
    cached_input: float
    cache_creation: float | None = None


_PRICES: dict[str, ModelPrice] = {
    # https://openai.com/api/pricing
    "openai/gpt-6-luna": ModelPrice(0.10, 0.50, 0.01, 0.125),
    "openai/gpt-5.6-sol": ModelPrice(5.00, 30.00, 0.50),
    "openai/gpt-5.6-terra": ModelPrice(2.00, 12.00, 0.20),
    "openai/gpt-5.6-luna": ModelPrice(0.20, 1.20, 0.02),
    "openai/gpt-5.5": ModelPrice(5.00, 30.00, 0.50),
    "openai/gpt-5.4": ModelPrice(2.50, 15.00, 0.25),
    "openai/gpt-5.4-mini": ModelPrice(0.75, 4.50, 0.075),
    "openai/gpt-5.4-nano": ModelPrice(0.20, 1.25, 0.02),
    # https://www.anthropic.com/pricing#api
    "anthropic/claude-opus-5": ModelPrice(5.00, 25.00, 0.50, 6.25),
    "anthropic/claude-sonnet-5": ModelPrice(2.00, 10.00, 0.20, 2.50),
    "anthropic/claude-opus-4-7": ModelPrice(5.00, 25.00, 0.50, 6.25),
    "anthropic/claude-sonnet-4-6": ModelPrice(3.00, 15.00, 0.30, 3.75),
    "anthropic/claude-haiku-4-5": ModelPrice(1.00, 5.00, 0.10, 1.25),
}

# Provider-hosted web search, USD per executed search call.
# OpenAI: $10 / 1k web_search calls; Anthropic: $10 / 1k web_search requests.
HOSTED_SEARCH_PRICE_PER_CALL: dict[str, float] = {
    "openai": 0.01,
    "anthropic": 0.01,
}


def register_price(model: str, price: ModelPrice) -> None:
    """Add or replace the price for a provider-prefixed model name."""
    _PRICES[model] = price


def model_price(model: str) -> ModelPrice | None:
    """Return the price for ``model``, matching dated snapshots by longest prefix."""
    if model in _PRICES:
        return _PRICES[model]
    candidates = [name for name in _PRICES if model.startswith(f"{name}-")]
    return _PRICES[max(candidates, key=len)] if candidates else None


def token_cost(model: str, usage: Usage, *, cached_included_in_input: bool) -> float | None:
    """Price one call's usage.

    OpenAI reports cached tokens as a subset of ``input_tokens``; Anthropic
    reports cache reads and writes separately from ``input_tokens``. The caller
    says which convention its usage follows.
    """
    price = model_price(model)
    if price is None:
        return None
    uncached = usage.input_tokens
    if cached_included_in_input:
        uncached = max(0, usage.input_tokens - usage.cached_input_tokens)
    cache_creation_rate = (
        price.cache_creation if price.cache_creation is not None else price.input * 1.25
    )
    return (
        uncached * price.input
        + usage.cached_input_tokens * price.cached_input
        + usage.cache_creation_tokens * cache_creation_rate
        + usage.output_tokens * price.output
    ) / 1_000_000


def hosted_search_cost(provider: str, num_searches: int) -> float:
    """Price provider-hosted web-search calls."""
    return HOSTED_SEARCH_PRICE_PER_CALL.get(provider, 0.0) * num_searches
