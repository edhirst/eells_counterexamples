"""Outward decimal rounding for the displayed constants (Table A.1) and small helpers."""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from math import floor, log10


def _quantum(x, sig):
    e = floor(log10(abs(x))) if x != 0 else 0
    return Decimal(1).scaleb(e - sig + 1)


def round_up(x, sig=4):
    """Decimal string >= the float x with `sig` significant digits (x is itself an upper bound)."""
    x = float(x)
    if x == 0:
        return "0"
    d = Decimal(x).quantize(_quantum(x, sig), rounding=ROUND_CEILING)
    return format(d.normalize(), "e") if abs(x) < 1e-3 or abs(x) >= 1e6 else format(d, "f")


def round_down(x, sig=4):
    """Decimal string <= the float x (x is itself a lower bound)."""
    x = float(x)
    if x == 0:
        return "0"
    d = Decimal(x).quantize(_quantum(x, sig), rounding=ROUND_FLOOR)
    return format(d.normalize(), "e") if abs(x) < 1e-3 or abs(x) >= 1e6 else format(d, "f")


def sci(s):
    """'1.234e-05' -> LaTeX  1.234\\cdot 10^{-5};  plain decimals unchanged."""
    s = str(s)
    if "e" in s or "E" in s:
        m, e = s.lower().split("e")
        return f"{m}\\cdot 10^{{{int(e)}}}"
    return s
