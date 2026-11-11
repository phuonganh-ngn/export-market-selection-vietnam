from __future__ import annotations

import numpy as np
import pandas as pd


def dcg(gains, k: int) -> float:
    x = np.asarray(gains, dtype=float)[:k]
    if k <= 0:
        raise ValueError("k must be positive")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Gains must be finite and nonnegative")
    return float(np.sum(x / np.log2(np.arange(2, len(x) + 2))))


def ndcg_from_gains(ranked_gains, all_gains, k: int) -> float:
    g = np.asarray(ranked_gains, dtype=float)
    ideal = np.sort(np.asarray(all_gains, dtype=float))[::-1]
    denom = dcg(ideal, k)
    if denom <= 0:
        return np.nan
    return dcg(g, k) / denom


def order_indices(scores, iso3):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Ranking scores must be finite")
    return np.lexsort((np.asarray(iso3, dtype=str), -scores))


def pool_metrics(frame: pd.DataFrame, score: str, task: str, cutoffs=None) -> dict:
    """Complete-case metric after ranking within the full ex-ante candidate pool."""
    all_n = len(frame)
    missing_top10 = 0
    full_order = frame.iloc[order_indices(frame[score], frame.destination_iso3)].copy()
    missing_top10 = int((~full_order.label_observed.astype(bool).iloc[:10]).sum())
    d = full_order.loc[full_order.label_observed.astype(bool)].copy()
    if d.empty:
        return {"n_exante": all_n, "n_observed": 0, "n_missing": all_n, "missing_in_top5": missing_top10,
                "ndcg_at_5": np.nan, "ndcg_at_10": np.nan, "precision_at_5": np.nan,
                "precision_at_10": np.nan, "recall_at_10": np.nan, "top5_total_gain": np.nan,
                "top10_total_gain": np.nan, "top5_positive_increment_usd": np.nan,
                "top10_positive_increment_usd": np.nan}
    if task == "Task_A_new_entry":
        rel = d.entry_success.astype(int).to_numpy()
        gains = rel.astype(float)
    else:
        inc = np.maximum(0.0, d.realized_incremental_export_usd.to_numpy(float))
        rel = np.floor(np.log10(1.0 + inc / 50_000.0)).clip(0, 4).astype(int)
        gains = np.exp2(rel) - 1.0
    all_gains = gains.copy()
    y = d.entry_success.astype(int).to_numpy() if task == "Task_A_new_entry" else (d.realized_incremental_export_usd.to_numpy(float) >= 50_000.0).astype(int)
    ranked_gains = gains
    n5, n10 = min(5, len(d)), min(10, len(d))
    p5 = float(y[:5].sum() / min(5, len(d)))
    p10 = float(y[:10].sum() / min(10, len(d)))
    positives = int(y.sum())
    out = {
        "n_exante": all_n, "n_observed": len(d), "n_missing": all_n - len(d),
        "missing_in_top5": int((~full_order.label_observed.astype(bool).iloc[:5]).sum()),
        "missing_in_top10": missing_top10,
        "ndcg_at_5": ndcg_from_gains(ranked_gains, all_gains, 5),
        "ndcg_at_10": ndcg_from_gains(ranked_gains, all_gains, 10),
        "precision_at_5": p5, "precision_at_10": p10,
        "recall_at_10": float(y[:10].sum() / positives) if positives else np.nan,
        "top5_total_gain": float(gains[:5].sum()), "top10_total_gain": float(gains[:10].sum()),
        "top5_positive_increment_usd": float(np.maximum(0.0, d.realized_incremental_export_usd.to_numpy(float)[:5]).sum()),
        "top10_positive_increment_usd": float(np.maximum(0.0, d.realized_incremental_export_usd.to_numpy(float)[:10]).sum()),
    }
    if task == "Task_B_existing_underexploited" and cutoffs is not None:
        qrel = np.zeros(len(d), dtype=np.int8)
        inc = d.realized_incremental_export_usd.to_numpy(float)
        pos = inc > 0
        qrel[pos] = 1 + np.searchsorted(np.asarray(cutoffs), inc[pos], side="left")
        qgain = np.exp2(qrel) - 1.0
        out["ndcg_at_10_training_cutoff"] = ndcg_from_gains(qgain, qgain, 10)
    return out

