"""Per-model pricing table and cost calculation.

Why a separate module
---------------------
Pricing changes more often than tracing logic. Keeping the table here
means cost updates are a one-line edit, never a surgical patch through
the trace code. The :func:`cost_usd_for` helper is the only function
that other modules should call.

Authoritative source
--------------------
Anthropic publishes prices at https://www.anthropic.com/pricing.
Update the :data:`PRICING` dict when the published numbers change.

Unknown models fall back to :data:`DEFAULT_PRICE` (Sonnet-tier) so a
new model name produces a sensible *non-zero* cost rather than silently
making everything appear free in dashboards.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """Pricing for one model expressed in USD per 1,000,000 tokens."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens / 1_000_000.0) * self.input_per_mtok
            + (output_tokens / 1_000_000.0) * self.output_per_mtok
        )


# USD per 1M tokens. Snapshot of public Anthropic pricing — keep current.
PRICING: dict[str, ModelPrice] = {
    # Claude 4 family
    "claude-opus-4-0":            ModelPrice(15.00, 75.00),
    "claude-opus-4":              ModelPrice(15.00, 75.00),
    "claude-sonnet-4-0":          ModelPrice(3.00, 15.00),
    "claude-sonnet-4":            ModelPrice(3.00, 15.00),
    "claude-haiku-4-0":           ModelPrice(0.80,  4.00),
    "claude-haiku-4":             ModelPrice(0.80,  4.00),
    # Claude 3.5
    "claude-3-5-sonnet-20240620": ModelPrice(3.00, 15.00),
    "claude-3-5-sonnet-latest":   ModelPrice(3.00, 15.00),
    "claude-3-5-haiku-20241022":  ModelPrice(0.80,  4.00),
    "claude-3-5-haiku-latest":    ModelPrice(0.80,  4.00),
    # Claude 3
    "claude-3-opus-20240229":     ModelPrice(15.00, 75.00),
    "claude-3-sonnet-20240229":   ModelPrice(3.00, 15.00),
    "claude-3-haiku-20240307":    ModelPrice(0.25,  1.25),
}

# Used when the model name does not match any entry. Sonnet-tier is a
# safe overestimate — better to flag a too-high cost in monitoring than
# to silently report $0 for a model nobody priced yet.
DEFAULT_PRICE = ModelPrice(3.00, 15.00)


def get_price(model: str) -> ModelPrice:
    """Look up the price entry for ``model``.

    Tries exact match first, then prefix match in either direction so
    aliases like ``claude-3-5-sonnet`` (no date suffix) still resolve.
    Returns :data:`DEFAULT_PRICE` if nothing matches.
    """
    if not model:
        return DEFAULT_PRICE
    if model in PRICING:
        return PRICING[model]
    for key, price in PRICING.items():
        if model.startswith(key) or key.startswith(model):
            return price
    return DEFAULT_PRICE


def cost_usd_for(model: str, input_tokens: int, output_tokens: int) -> float:
    """Convenience: ``ModelPrice.cost`` for a known model.

    Always returns a non-negative float. Negative token counts (which
    shouldn't happen but have been observed in flaky API responses) are
    clamped to 0.
    """
    in_tok = max(0, int(input_tokens or 0))
    out_tok = max(0, int(output_tokens or 0))
    return get_price(model).cost(in_tok, out_tok)
