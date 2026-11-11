"""Equal-weight (or weighted) rank fusion of component scores within decision pools.

Added in the 2026-11 revision so that every experiment uses one implementation.

Normalisation. A component rank r (1 = best) in a pool of n candidates maps to (n - r) / (n - 1),
which lies in [0, 1]. The archived pipeline used (n - r + 1) / (n - 1), which is a constant shift of
1 / (n - 1) for every candidate of the pool; it therefore gives exactly the same ordering and the same
ties, but the best candidate scored above 1.

Ties. Equal-weight averaging of two ranks often produces exact ties. The rule is configurable:
  iso3            - ascending destination code (archived behaviour; arbitrary but reproducible)
  first_component - the first score column (PPML in the paper) decides, then ISO3
  average         - fractional ("average") component ranks, then ISO3 for any remaining tie
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIE_RULES = ("iso3", "first_component", "average")


def _component_rank(g: pd.DataFrame, col: str, id_col: str, method: str) -> pd.Series:
    """Rank 1 = highest score. 'ordinal' breaks equal scores by id_col, 'average' keeps fractional ranks."""
    if method == "average":
        return g[col].rank(ascending=False, method="average")
    order = g.sort_values([col, id_col], ascending=[False, True], kind="mergesort").index
    return pd.Series(np.arange(1, len(g) + 1, dtype=float), index=order).reindex(g.index)


def fuse_pool(g: pd.DataFrame, score_cols: list[str], weights: list[float] | None = None,
              tie_rule: str = "iso3", id_col: str = "destination_iso3") -> pd.DataFrame:
    """Return the pool with columns fused_score (higher is better) and fused_rank (1 = best)."""
    if tie_rule not in TIE_RULES:
        raise ValueError(f"tie_rule must be one of {TIE_RULES}, got {tie_rule!r}")
    weights = list(weights) if weights is not None else [1.0 / len(score_cols)] * len(score_cols)
    if len(weights) != len(score_cols) or not np.isclose(sum(weights), 1.0):
        raise ValueError("weights must match score_cols and sum to one")
    g = g.copy()
    n = len(g)
    den = max(1, n - 1)
    method = "average" if tie_rule == "average" else "ordinal"
    fused = np.zeros(n)
    for w, col in zip(weights, score_cols):
        r = _component_rank(g, col, id_col, method).to_numpy(float)
        g[f"_rank_{col}"] = r
        fused += w * (n - r) / den
    g["fused_score"] = fused
    keys = ["fused_score"] + ([f"_rank_{score_cols[0]}"] if tie_rule == "first_component" else []) + [id_col]
    asc = [False] + ([True] if tie_rule == "first_component" else []) + [True]
    order = g.sort_values(keys, ascending=asc, kind="mergesort").index
    g["fused_rank"] = pd.Series(np.arange(1, n + 1), index=order).reindex(g.index).astype(int)
    return g


def fuse(frame: pd.DataFrame, pool_cols: list[str], score_cols: list[str], weights: list[float] | None = None,
         tie_rule: str = "iso3", id_col: str = "destination_iso3") -> pd.DataFrame:
    """Apply fuse_pool to every pool (e.g. pool_cols = ["task", "hs2", "outcome_year"])."""
    parts = [fuse_pool(g, score_cols, weights, tie_rule, id_col) for _, g in frame.groupby(pool_cols, sort=False)]
    return pd.concat(parts).loc[frame.index]
