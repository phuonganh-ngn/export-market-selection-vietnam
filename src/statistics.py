from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import CFG, path, write_csv


def forecast_metrics(actual, prediction) -> dict:
    y = np.asarray(actual, dtype=float)
    p = np.maximum(0.0, np.asarray(prediction, dtype=float))
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if not len(y):
        return {k: np.nan for k in ["mae_usd", "rmse_usd", "mae_log1p", "wape", "mae_zero_usd", "mae_positive_usd"]}
    err = np.abs(y-p)
    den = np.abs(y).sum()
    zero = y.eq(0) if isinstance(y, pd.Series) else y == 0
    return {"mae_usd": float(err.mean()), "rmse_usd": float(np.sqrt(np.mean((y-p)**2))),
            "mae_log1p": float(np.mean(np.abs(np.log1p(y)-np.log1p(p)))),
            "wape": float(err.sum()/den) if den > 0 else np.nan,
            "mae_zero_usd": float(err[zero].mean()) if zero.any() else np.nan,
            "mae_positive_usd": float(err[~zero].mean()) if (~zero).any() else np.nan,
            "n": int(len(y)), "zero_share": float(zero.mean())}


def precision_at_k(relevance: np.ndarray, k: int) -> float:
    if len(relevance) == 0:
        return np.nan
    top = np.asarray(relevance)[:k]
    return float(top.sum()/min(k, len(relevance)))


def ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    rel = np.asarray(relevance, dtype=float)[:k]
    if not len(rel) or np.max(rel, initial=0) <= 0:
        return np.nan
    discounts = 1 / np.log2(np.arange(2, len(rel)+2))
    dcg = np.sum((2**rel - 1)*discounts)
    ideal = np.sort(np.asarray(relevance, dtype=float))[::-1][:k]
    idcg = np.sum((2**ideal - 1)*discounts[:len(ideal)])
    return float(dcg/idcg) if idcg > 0 else np.nan


def paired_destination_bootstrap(preds: pd.DataFrame, model_a: str, model_b: str, metric: str = "mae_usd", reps: int | None = None, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    d = preds.loc[preds.model.isin([model_a, model_b])].pivot_table(index=["destination_iso3", "hs2", "evaluation_year"], columns="model", values=["actual_usd", "predicted_usd"], aggfunc="first")
    needed = [("actual_usd", model_a), ("predicted_usd", model_a), ("predicted_usd", model_b)]
    if any(k not in d.columns for k in needed) or d.empty:
        return {"model_a": model_a, "model_b": model_b, "metric": metric, "difference": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "n_blocks": 0, "status": "insufficient paired predictions"}
    y = d[("actual_usd", model_a)].to_numpy(float)
    pa, pb = d[("predicted_usd", model_a)].to_numpy(float), d[("predicted_usd", model_b)].to_numpy(float)
    block = d.index.get_level_values("destination_iso3").to_numpy()
    blocks = np.unique(block)
    reps = reps or CFG["study"]["bootstrap_replicates"]
    point = float(np.mean(np.abs(y-pa) - np.abs(y-pb)))
    boots = []
    for _ in range(reps):
        take = rng.choice(blocks, size=len(blocks), replace=True)
        idx = np.concatenate([np.flatnonzero(block == b) for b in take])
        boots.append(float(np.mean(np.abs(y[idx]-pa[idx]) - np.abs(y[idx]-pb[idx]))))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    pval = 2*min(np.mean(np.asarray(boots) <= 0), np.mean(np.asarray(boots) >= 0))
    return {"model_a": model_a, "model_b": model_b, "metric": metric, "difference": point,
            "ci_low": float(lo), "ci_high": float(hi), "p_value": float(min(1,pval)), "n_blocks": int(len(block)),
            "status": "destination-block bootstrap; clustered dependence preserved within destination"}


def holm_adjust(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    vals = p[ok]
    order = np.argsort(vals)
    adjusted = np.maximum.accumulate((len(vals)-np.arange(len(vals))) * vals[order])
    adjusted = np.minimum(adjusted, 1)
    restore = np.empty_like(adjusted)
    restore[order] = adjusted
    out[ok] = restore
    return out
