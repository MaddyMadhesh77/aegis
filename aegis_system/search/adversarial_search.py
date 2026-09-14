from __future__ import annotations

from typing import Any, Dict


def minimax_decision(values: Dict[str, float], maximizing: bool = True) -> str:
    if not values:
        raise ValueError("No values provided for minimax decision")
    if maximizing:
        best_key = max(values, key=lambda key: values[key])
    else:
        best_key = min(values, key=lambda key: values[key])
    return best_key
