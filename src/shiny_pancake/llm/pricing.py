"""Rough cost estimation from token usage.

Prices are USD per million tokens (Anthropic first-party list prices, cached 2026-10).
They are an *estimate* for dashboards and budgets; always check the provider's pricing page.
Cache writes are billed at ~1.25x input (5-minute TTL), cache reads at ~0.1x input.
"""

from __future__ import annotations

from .base import Usage

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}


def price_for(model: str) -> tuple[float, float] | None:
    if model in PRICES_PER_MTOK:
        return PRICES_PER_MTOK[model]
    # tolerate provider prefixes such as "anthropic.claude-sonnet-5-5"
    for name, price in PRICES_PER_MTOK.items():
        if model.endswith(name):
            return price
    return None


def estimate_cost(model: str, usage: Usage) -> float:
    """Estimated USD cost of one call; 0.0 for unknown/local models."""
    price = price_for(model)
    if price is None:
        return 0.0
    inp, out = price
    total = (
        usage.input_tokens * inp
        + usage.cache_write_tokens * inp * 1.25
        + usage.cache_read_tokens * inp * 0.10
        + usage.output_tokens * out
    )
    return round(total / 1_000_000, 6)
