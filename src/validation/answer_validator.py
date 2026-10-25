"""Check that numbers stated in the final answer come from tool outputs."""
from __future__ import annotations

import json
import re

NUM = re.compile(r"(?<![\w.])[-+]?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|(?<![\w.])[-+]?\$?\d+(?:\.\d+)?%?")


def _to_float(token: str) -> float | None:
    try:
        return float(token.replace("$", "").replace(",", "").replace("%", "").lstrip("+"))
    except ValueError:
        return None


def _collect_numbers(obj, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_numbers(v, out)
    elif isinstance(obj, str):
        out.extend(f for f in (_to_float(t) for t in NUM.findall(obj)) if f is not None)


def validate_numbers(answer: str, tool_outputs: list, rel_tol: float = 0.01) -> dict:
    """Flag decimal numbers in `answer` that don't match any tool output value.

    Only numbers with a decimal point are checked: indicator values and prices
    are decimals, while integers in prose (years, '14-day', counts) are not.
    """
    known: list[float] = []
    for o in tool_outputs:
        _collect_numbers(o if not isinstance(o, str) else _safe_json(o), known)

    checked, unverified = 0, []
    for tok in NUM.findall(answer):
        if "." not in tok:
            continue
        val = _to_float(tok)
        if val is None:
            continue
        checked += 1
        if not any(abs(val - k) <= max(rel_tol * abs(k), 0.011) for k in known):
            unverified.append(tok)
    return {"numbers_checked": checked, "unverified": unverified, "passed": not unverified}


def _safe_json(s: str):
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return s
