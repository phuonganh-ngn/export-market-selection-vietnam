from __future__ import annotations

PREFERENCE_SETS = {
    "W_narrow": (0.40, 0.60),
    "W_primary": (0.25, 0.75),
    "W_broad": (0.00, 1.00),
}
PREFERENCE_GRID = (0.00, 0.25, 0.50, 0.75, 1.00)


def vertices(interval: tuple[float, float]) -> tuple[float, float]:
    lo, hi = map(float, interval)
    if lo < 0 or hi > 1 or lo > hi:
        raise ValueError(f"Invalid preference interval: {interval}")
    return (lo, hi) if lo != hi else (lo,)


def scalarize(objective1, objective2, weight1: float):
    return weight1 * objective1 + (1.0 - weight1) * objective2
