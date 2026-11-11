from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_at_k(relevance, k: int) -> float:
    rel = np.asarray(relevance, dtype=float)
    if k <= 0:
        raise ValueError("k must be positive")
    if len(rel) == 0:
        return np.nan
    if not np.isfinite(rel).all() or (rel < 0).any():
        raise ValueError("Relevance must be finite and nonnegative")
    denom = min(k, len(rel))
    return float(rel[:k].sum() / denom)


def dcg_from_gain(gain, k: int) -> float:
    gain = np.asarray(gain, dtype=float)[:k]
    if not np.isfinite(gain).all() or (gain < 0).any():
        raise ValueError("DCG gains must be finite and nonnegative")
    discount = 1.0 / np.log2(np.arange(2, len(gain) + 2))
    return float(np.sum(gain * discount))


def ndcg_from_gain(gain, ideal_gain, k: int) -> float:
    """Standard NDCG: zero when the ranking misses positives; NA only if ideal gain is zero."""
    gain = np.asarray(gain, dtype=float)
    ideal_gain = np.asarray(ideal_gain, dtype=float)
    ideal = dcg_from_gain(np.sort(ideal_gain)[::-1], k)
    if ideal <= 0:
        return np.nan
    return float(dcg_from_gain(gain, k) / ideal)


def ndcg_at_k(relevance, k: int, exponential_gain: bool = True) -> float:
    rel = np.asarray(relevance, dtype=float)
    if not np.isfinite(rel).all() or (rel < 0).any():
        raise ValueError("Relevance must be finite and nonnegative")
    gain = np.exp2(rel) - 1.0 if exponential_gain else rel
    return ndcg_from_gain(gain, gain, k)


def rank_pool(frame: pd.DataFrame, score: str, destination: str = "destination_iso3") -> pd.DataFrame:
    """Deterministic score-descending ranking with ISO3 ascending tie break."""
    return frame.sort_values([score, destination], ascending=[False, True], na_position="last", kind="mergesort").reset_index(drop=True)


def capped_export_grade(increment_usd, threshold: float = 50_000) -> np.ndarray:
    inc = np.asarray(increment_usd, dtype=float)
    inc = np.maximum(0.0, np.nan_to_num(inc, nan=0.0, posinf=0.0, neginf=0.0))
    return np.minimum(4, np.floor(np.log10(1.0 + inc / threshold))).astype(np.int8)


def discovery_pool_metrics(y, score, k: int = 10) -> dict:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    if len(y) != len(score) or not np.isfinite(y).all() or not np.isfinite(score).all():
        raise ValueError("Metric vectors must be equal length and observed/finite")
    if (y < 0).any() or (y > 1).any():
        raise ValueError("Binary relevance must lie in [0,1]")
    order = np.argsort(-score, kind="mergesort")
    ranked = y[order]
    positives = int(y.sum())
    return {
        "n_observed": len(y), "positive_events": positives, "prevalence": float(y.mean()) if len(y) else np.nan,
        "pr_auc": float(average_precision_score(y, score)) if positives and positives < len(y) else np.nan,
        "roc_auc": float(roc_auc_score(y, score)) if 0 < positives < len(y) else np.nan,
        "precision_at_5": precision_at_k(ranked, 5), "precision_at_10": precision_at_k(ranked, k),
        "recall_at_10": float(ranked[:k].sum() / positives) if positives else np.nan,
        "ndcg_at_10": ndcg_at_k(ranked, k), "entry_successes_at_10": int(ranked[:k].sum()),
    }


def ranking_pool_metrics(frame: pd.DataFrame, score_col: str, threshold: int = 50_000, k: int = 10) -> dict:
    """Known-label conditional ranking metrics; unknown labels are never imputed."""
    d = frame.loc[frame.label_observed].copy()
    if d.empty:
        return {"n_observed": 0, "n_missing_in_exante_pool": len(frame), "ndcg10_capped_grade": np.nan,
                "ndcg10_log1p_gain": np.nan, "ndcg10_raw_gain": np.nan, "precision10_expansion": np.nan,
                "known_incremental_usd_top10": np.nan, "missing_label_in_top10": np.nan}
    d = rank_pool(d, score_col)
    increment = d.realized_incremental_export_usd.to_numpy(float)
    grade = capped_export_grade(increment, threshold)
    raw_gain = np.maximum(0.0, increment)
    log_gain = np.log1p(raw_gain)
    binary = (increment >= threshold).astype(float)
    return {
        "n_observed": len(d), "n_missing_in_exante_pool": len(frame) - int(frame.label_observed.sum()),
        "ndcg10_capped_grade": ndcg_at_k(grade, k),
        "ndcg10_log1p_gain": ndcg_from_gain(log_gain, log_gain, k),
        "ndcg10_raw_gain": ndcg_from_gain(raw_gain, raw_gain, k),
        "precision10_expansion": precision_at_k(binary, k),
        "known_incremental_usd_top10": float(np.sum(increment[:k])),
        "missing_label_in_top10": int((~frame.label_observed.iloc[:k].astype(bool)).sum()) if score_col in frame else np.nan,
    }


def holm_adjust(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid])]
    m = len(order)
    adjusted_sorted = np.maximum.accumulate([(m - i) * p[idx] for i, idx in enumerate(order)])
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    out[order] = adjusted_sorted
    return out


def hs2_block_bootstrap(differences: pd.DataFrame, value: str, reps: int = 5000, seed: int = 773) -> tuple[float, float, float, int]:
    """Resample whole HS2 blocks, retaining all year pools within sampled chapters."""
    d = differences[["hs2", value]].dropna()
    cluster = d.groupby("hs2", sort=False)[value].mean()
    x = cluster.to_numpy(float)
    n = len(x)
    if n < 10:
        return np.nan, np.nan, np.nan, n
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n, size=(reps, n))
    boots = x[draw].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    # Cluster sign-flip randomization test; model seeds are averaged before this step.
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, n))
    null = (signs * x).mean(axis=1)
    observed = x.mean()
    p = (1 + np.sum(np.abs(null) >= abs(observed))) / (reps + 1)
    return float(lo), float(hi), float(p), int(n)


def destination_block_bootstrap(errors: pd.DataFrame, reps: int = 5000, seed: int = 991) -> tuple[float, float, float, int]:
    """Paired micro-MAE difference bootstrap by destination blocks."""
    d = errors.dropna(subset=["destination_iso3", "err_a", "err_b"])
    blocks = {c: g[["err_a", "err_b"]].to_numpy(float) for c, g in d.groupby("destination_iso3", sort=False)}
    countries = list(blocks)
    if len(countries) < 10:
        return np.nan, np.nan, np.nan, len(countries)
    all_diff = (d.err_a - d.err_b).mean()
    rng = np.random.default_rng(seed)
    boots = np.empty(reps)
    # Preserve all product-year observations of every sampled destination.
    for b in range(reps):
        chosen = rng.choice(countries, size=len(countries), replace=True)
        a = np.concatenate([blocks[c][:, 0] for c in chosen])
        bb = np.concatenate([blocks[c][:, 1] for c in chosen])
        boots[b] = np.mean(a - bb)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    dest_mean = d.assign(diff=d.err_a - d.err_b).groupby("destination_iso3")["diff"].mean()
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(dest_mean.to_numpy(), zero_method="zsplit", alternative="two-sided").pvalue)
        # Preserve a conservative numeric upper bound when floating-point underflow returns zero.
        p = max(p, 1e-16)
    except Exception:
        p = np.nan
    return float(lo), float(hi), p, len(countries)
