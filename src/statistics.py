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
    """NDCG@k (exponential gain) for relevance values listed in ranked order.

    Undefined (NaN) only when no item in the whole list is relevant, so the ideal DCG is zero.
    A ranking that puts no relevant item in its top k scores 0.

    Fix 2026-11: the previous version returned NaN whenever the top k contained no relevant
    item, which dropped every pool where the model missed all successes and inflated the
    averages reported by E2, E3, the clustering analysis and robustness check R14.
    """
    full = np.asarray(relevance, dtype=float)
    if not len(full):
        return np.nan
    n = min(k, len(full))
    discounts = 1 / np.log2(np.arange(2, n + 2))
    ideal = np.sort(full)[::-1][:n]
    idcg = np.sum((2**ideal - 1) * discounts)
    if idcg <= 0:
        return np.nan
    dcg = np.sum((2**full[:n] - 1) * discounts)
    return float(dcg / idcg)


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


# ---------------------------------------------------------------------------------------------
# Paired pool-level inference (added 2026-11). A "difference" is a per-pool metric difference
# between two ranking policies evaluated on the same pool (one HS2 chapter in one year).
# ---------------------------------------------------------------------------------------------

def hs2_block_inference(diff, blocks, reps: int = 10000, seed: int = 20260101) -> dict:
    """Mean paired difference with an HS2-block percentile bootstrap and a block sign-flip test.

    All years of a chapter are resampled (or sign-flipped) together, as in the frozen evaluation.
    """
    d = pd.DataFrame({"b": np.asarray(blocks), "x": np.asarray(diff, dtype=float)})
    d = d[np.isfinite(d.x)]
    if d.empty:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_signflip": np.nan, "n": 0, "blocks": 0}
    g = d.groupby("b").x.agg(["sum", "count"])
    s, c = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    est = float(d.x.mean())
    rng = np.random.default_rng(seed)
    nb = len(g)
    ix = rng.integers(0, nb, (reps, nb))
    boot = s[ix].sum(1) / c[ix].sum(1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, nb))
    null = (signs * s[None, :]).sum(1) / c.sum()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    p = float((1 + np.sum(np.abs(null) >= abs(est))) / (reps + 1))
    return {"estimate": est, "ci_low": float(lo), "ci_high": float(hi), "p_signflip": p, "n": int(len(d)), "blocks": int(nb)}


def twoway_cluster_t(diff, group1, group2) -> dict:
    """t-statistic of the mean difference with standard errors clustered two ways
    (Cameron-Gelbach-Miller), e.g. by HS2 chapter and by year."""
    import statsmodels.api as sm
    d = pd.DataFrame({"x": np.asarray(diff, dtype=float), "g1": np.asarray(group1), "g2": np.asarray(group2)})
    d = d[np.isfinite(d.x)].reset_index(drop=True)
    if len(d) < 3:
        return {"se_twoway": np.nan, "t_twoway": np.nan}
    groups = np.column_stack([pd.factorize(d.g1)[0], pd.factorize(d.g2)[0]])
    res = sm.OLS(d.x.to_numpy(), np.ones((len(d), 1))).fit(cov_type="cluster", cov_kwds={"groups": groups})
    se = float(res.bse[0])
    return {"se_twoway": se, "t_twoway": float(d.x.mean() / se) if se > 0 else np.nan}


def year_signflip_p(diff, years) -> dict:
    """Exact sign-flip test on the annual mean differences (all 2^T sign patterns).

    With T years the smallest attainable two-sided p-value is 2 / 2^T (0.016 for T = 7, 0.25 for T = 3),
    so a three-year window can never reach conventional significance once year-level dependence is respected.
    """
    d = pd.DataFrame({"x": np.asarray(diff, dtype=float), "y": np.asarray(years)})
    ym = d[np.isfinite(d.x)].groupby("y").x.mean().to_numpy()
    t = len(ym)
    if t == 0:
        return {"p_year_signflip": np.nan, "years_positive": 0, "n_years": 0, "p_floor": np.nan}
    flips = ((np.arange(2 ** t)[:, None] >> np.arange(t)) & 1) * 2 - 1
    stat = np.abs((flips * ym[None, :]).mean(1))
    p = float(np.mean(stat >= abs(ym.mean()) - 1e-12))
    return {"p_year_signflip": p, "years_positive": int((ym > 0).sum()), "n_years": int(t), "p_floor": 2.0 / 2 ** t}


def paired_contrast_inference(diff, blocks, years, reps: int = 10000, seed: int = 20260101) -> dict:
    """All three layers used in the revised paper for one contrast."""
    out = hs2_block_inference(diff, blocks, reps=reps, seed=seed)
    out.update(twoway_cluster_t(diff, blocks, years))
    out.update(year_signflip_p(diff, years))
    return out
