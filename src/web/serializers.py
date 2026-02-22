"""
Serializers for converting LangGraph state deltas into JSON-safe dicts for SSE.
"""

from __future__ import annotations

from typing import Any


def serialize_state_update(
    node_name: str,
    state_delta: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Convert a LangGraph state delta into a JSON-serializable dict for SSE.

    Args:
        node_name: The graph node that just completed (e.g. "fundamentals").
        state_delta: The partial state dict returned by that node.
        meta: Display metadata — label, stage number, total stages.

    Returns:
        A plain dict safe for ``json.dumps``.
    """
    result: dict[str, Any] = {
        "node": node_name,
        "label": meta.get("label", node_name),
        "stage": meta.get("stage", 0),
        "total_stages": meta.get("total", 6),
    }

    # --- Pydantic model lists → plain dicts ---
    if "signals" in state_delta:
        result["signals"] = [s.model_dump() for s in state_delta["signals"]]

    if "risk_assessments" in state_delta:
        result["risk_assessments"] = [
            r.model_dump() for r in state_delta["risk_assessments"]
        ]

    if "decisions" in state_delta:
        result["decisions"] = [d.model_dump() for d in state_delta["decisions"]]

    # For market data: send loaded flag + price data for chart rendering.
    if "market_data" in state_delta:
        result["market_data_loaded"] = True
        result["tickers_loaded"] = list(state_delta["market_data"].keys())
        # Include OHLCV prices for frontend chart rendering (~15KB per ticker)
        result["price_data"] = {
            ticker: data.get("prices", [])
            for ticker, data in state_delta["market_data"].items()
        }

    return result
