from __future__ import annotations

import numpy as np
import pandas as pd

from .candidate_protocol import HERE
from .ranking_metrics import ndcg_from_gains

BOOTSTRAP_REPS = 5000
DESTINATION_BOOTSTRAP_REPS = 500


def _holm(p_values):
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if not len(idx):
        return out
    order = idx[np.argsort(p[idx])]
    vals = np.maximum.accumulate([(len(order) - i) * p[j] for i, j in enumerate(order)])
    out[order] = np.minimum(1.0, vals)
    return out


def _cluster_inference(pool_diff: pd.DataFrame, value: str, seed: int) -> dict:
    d = pool_diff[["hs2", value]].dropna()
    # Target the same pool-weighted macro estimand as the main table. Each
    # HS2 chapter is resampled as a block, but all its valid product-year pools
    # retain equal weight in the pooled mean.
    block_table = d.groupby("hs2", sort=False)[value].agg(["sum", "count"])
    block_sum = block_table["sum"].to_numpy(float)
    block_count = block_table["count"].to_numpy(float)
    if len(block_sum) < 10:
        estimate = float(block_sum.sum() / block_count.sum()) if block_count.sum() else np.nan
        return {"estimate": estimate, "ci95_low": np.nan, "ci95_high": np.nan,
                "p_raw": np.nan, "blocks": len(block_sum)}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(block_sum), size=(BOOTSTRAP_REPS, len(block_sum)))
    boot_sum = block_sum[indices].sum(axis=1)
    boot_count = block_count[indices].sum(axis=1)
    boot = np.divide(boot_sum, boot_count, out=np.full(BOOTSTRAP_REPS, np.nan), where=boot_count > 0)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(BOOTSTRAP_REPS, len(block_sum)))
    null = (block_sum[None, :] * signs).sum(axis=1) / block_count.sum()
    observed = float(block_sum.sum() / block_count.sum())
    p = float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (BOOTSTRAP_REPS + 1))
    boot = boot[np.isfinite(boot)]
    return {"estimate": observed, "ci95_low": float(np.quantile(boot, .025)),
            "ci95_high": float(np.quantile(boot, .975)), "p_raw": p, "blocks": len(block_sum)}


def align_paired_pool_metrics(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> pd.DataFrame:
    keys = ["hs2", "outcome_year"]
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise AssertionError("Model has duplicate product-year metrics")
    pair = a[keys + [metric]].merge(b[keys + [metric]], on=keys, suffixes=("_a", "_b"), validate="one_to_one").dropna()
    pair["difference"] = pair[f"{metric}_a"] - pair[f"{metric}_b"]
    return pair


def paired_model_comparisons(metrics: pd.DataFrame, regret: pd.DataFrame):
    selected = metrics.loc[
        (metrics.model.eq("LambdaMART") & metrics.seed.eq(-2))
        | (~metrics.model.eq("LambdaMART") & metrics.seed.eq(-1))
    ].copy()
    rows = []
    comparisons = {
        "Task_A_new_entry": [("LightGBM", "E6_LightGBM_classifier"),
                             ("Gravity_PPML", "E2_Gravity_PPML_score"),
                             ("TabM", "E2_TabM_classifier")],
        "Task_B_existing_underexploited": [("Gravity_PPML", "E3_Gravity_PPML_increment"),
                                           ("LightGBM", "E3_LightGBM_increment"),
                                           ("MLP", "E3_MLP_increment"),
                                           ("TabM", "E3_TabM_increment")],
    }
    for task, pairs in comparisons.items():
        metric_names = ["ndcg_at_10", "precision_at_10"] if task == "Task_A_new_entry" else ["ndcg_at_10"]
        base = selected.loc[selected.task.eq(task)]
        for metric in metric_names:
            for j, (model_b, label) in enumerate(pairs):
                a = base.loc[base.model.eq("LambdaMART"), ["hs2", "outcome_year", metric]]
                b = base.loc[base.model.eq(model_b), ["hs2", "outcome_year", metric]]
                pair = align_paired_pool_metrics(a, b, metric)
                inf = _cluster_inference(pair, "difference", seed=4700 + j + len(rows))
                rows.append({"family": f"{task}_{metric}", "task": task, "metric": metric,
                             "model_a": "LambdaMART", "model_b": model_b, "contrast": label,
                             "difference_a_minus_b": inf["estimate"], "ci95_low": inf["ci95_low"],
                             "ci95_high": inf["ci95_high"], "p_raw": inf["p_raw"],
                             "paired_pools": len(pair), "hs2_blocks": inf["blocks"],
                             "direction": "positive favors LambdaMART", "inference": "paired HS2-block bootstrap CI; cluster sign-flip p-value"})

    # Task B primary decision-loss comparison: mean realized-outcome opportunity regret@10.
    r = regret.loc[regret.task.eq("Task_B_existing_underexploited")
                   & regret.k.eq(10) & regret.gain_definition.eq("capped_grade_gain")].copy()
    r = r.loc[(r.model.eq("LambdaMART") & r.seed.eq(-2))
              | (~r.model.eq("LambdaMART") & r.seed.eq(-1))]
    for j, (model_b, label) in enumerate(comparisons["Task_B_existing_underexploited"]):
        a = r.loc[r.model.eq("LambdaMART"), ["hs2", "outcome_year", "regret"]]
        b = r.loc[r.model.eq(model_b), ["hs2", "outcome_year", "regret"]]
        pair = a.merge(b, on=["hs2", "outcome_year"], suffixes=("_a", "_b"), validate="one_to_one").dropna()
        pair["difference"] = pair.regret_a - pair.regret_b
        inf = _cluster_inference(pair, "difference", seed=6900 + j)
        rows.append({"family": "Task_B_regret10", "task": "Task_B_existing_underexploited", "metric": "mean_regret@10_capped_gain",
                     "model_a": "LambdaMART", "model_b": model_b, "contrast": label,
                     "difference_a_minus_b": inf["estimate"], "ci95_low": inf["ci95_low"],
                     "ci95_high": inf["ci95_high"], "p_raw": inf["p_raw"],
                     "paired_pools": len(pair), "hs2_blocks": inf["blocks"],
                     "direction": "negative favors LambdaMART", "inference": "paired HS2-block bootstrap CI; cluster sign-flip p-value"})

    result = pd.DataFrame(rows)
    result["p_holm"] = np.nan
    for _, ix in result.groupby("family").groups.items():
        result.loc[ix, "p_holm"] = _holm(result.loc[ix, "p_raw"].to_numpy())
    result.to_csv(HERE / "results/paired_model_comparisons.csv", index=False, na_rep="")
    result[["family", "task", "metric", "model_a", "model_b", "difference_a_minus_b",
            "ci95_low", "ci95_high", "paired_pools", "hs2_blocks"]].to_csv(
        HERE / "results/confidence_intervals.csv", index=False, na_rep="")
    result[["family", "task", "metric", "model_a", "model_b", "p_raw", "p_holm",
            "paired_pools", "hs2_blocks", "direction"]].to_csv(
        HERE / "results/holm_adjusted_tests.csv", index=False, na_rep="")
    return result


def destination_block_ndcg_sensitivity(ranked: pd.DataFrame, reps: int = DESTINATION_BOOTSTRAP_REPS) -> pd.DataFrame:
    """Paired destination-cluster bootstrap sensitivity for primary macro NDCG@10.

    A destination is drawn once per replicate and its multiplicity is shared
    across every HS2-year pool and model. Duplicate sampled destinations are
    represented as multiplicity weights in DCG and IDCG, preserving dependence
    across products and years without independently resampling candidates.
    This is a cross-sectional sensitivity, not a temporal generalization CI.
    """
    models_by_task = {
        "Task_A_new_entry": ["LightGBM", "Gravity_PPML", "TabM"],
        "Task_B_existing_underexploited": ["Gravity_PPML", "LightGBM", "MLP", "TabM"],
    }
    selected = ranked.loc[
        (ranked.model.eq("LambdaMART") & ranked.seed.eq(-2))
        | (~ranked.model.eq("LambdaMART") & ranked.seed.eq(-1))
    ].copy()
    destinations = sorted(selected.destination_iso3.dropna().astype(str).unique())
    did = {d: i for i, d in enumerate(destinations)}
    rng = np.random.default_rng(91827)
    draws = np.zeros((reps, len(destinations)), dtype=np.int16)
    for b in range(reps):
        sampled = rng.integers(0, len(destinations), size=len(destinations))
        draws[b] = np.bincount(sampled, minlength=len(destinations))

    def gains_for(pool: pd.DataFrame, task: str) -> np.ndarray:
        if task == "Task_A_new_entry":
            return pool.entry_success.astype(float).to_numpy()
        inc = np.maximum(0.0, pool.realized_incremental_export_usd.to_numpy(float))
        grade = np.minimum(4, np.floor(np.log10(1.0 + inc / 50_000.0)))
        return np.exp2(grade) - 1.0

    def ndcg_draws(sorted_frame: pd.DataFrame, task: str) -> np.ndarray:
        g = gains_for(sorted_frame, task)
        ids = sorted_frame.destination_iso3.astype(str).map(did).to_numpy(int)
        weights = draws[:, ids]
        cumulative = np.cumsum(weights, axis=1)
        k = 10
        positions = np.arange(1, k + 1, dtype=int)

        def dcg(order_gain: np.ndarray, order_ids: np.ndarray) -> np.ndarray:
            w = draws[:, order_ids]
            csum = np.cumsum(w, axis=1)
            selected_idx = (csum[:, :, None] >= positions[None, None, :]).argmax(axis=1)
            valid = csum[:, -1, None] >= positions[None, :]
            chosen_gain = order_gain[selected_idx]
            discount = 1.0 / np.log2(positions + 1.0)
            return np.sum(np.where(valid, chosen_gain * discount[None, :], 0.0), axis=1)

        pred_dcg = dcg(g, ids)
        ideal_order = np.argsort(-g, kind="mergesort")
        ideal_dcg = dcg(g[ideal_order], ids[ideal_order])
        out = np.full(reps, np.nan)
        np.divide(pred_dcg, ideal_dcg, out=out, where=ideal_dcg > 0)
        return out

    output = []
    for task, baselines in models_by_task.items():
        t = selected.loc[selected.task.eq(task) & selected.label_observed.astype(bool)].copy()
        # Cache each model's ranked gains and bootstrap NDCG by common pool.
        cache = {}
        for (model, hs2, year), q in t.groupby(["model", "hs2", "outcome_year"], sort=False):
            ordered = q.sort_values(["score", "destination_iso3"], ascending=[False, True], kind="mergesort")
            observed_gains = gains_for(ordered, task)
            cache[(model, str(hs2).zfill(2), int(year))] = (
                ndcg_draws(ordered, task), ndcg_from_gains(observed_gains, observed_gains, 10))
        for model_b in baselines:
            keys = sorted(k for k in cache if k[0] == "LambdaMART" and (model_b, k[1], k[2]) in cache)
            if not keys:
                continue
            delta_list = [cache[("LambdaMART", h, y)][0] - cache[(model_b, h, y)][0] for _, h, y in keys]
            stack = np.vstack(delta_list)
            valid = np.isfinite(stack)
            denom = valid.sum(axis=0)
            macro = np.divide(np.nansum(stack, axis=0), denom,
                              out=np.full(reps, np.nan), where=denom > 0)
            finite = macro[np.isfinite(macro)]
            point_diffs = [cache[("LambdaMART", h, y)][1] - cache[(model_b, h, y)][1]
                           for _, h, y in keys
                           if np.isfinite(cache[("LambdaMART", h, y)][1])
                           and np.isfinite(cache[(model_b, h, y)][1])]
            output.append({"task": task, "metric": "macro_NDCG@10", "model_a": "LambdaMART",
                           "model_b": model_b, "difference_a_minus_b": float(np.mean(point_diffs)) if point_diffs else np.nan,
                           "destination_block_ci95_low": float(np.quantile(finite, .025)) if len(finite) else np.nan,
                           "destination_block_ci95_high": float(np.quantile(finite, .975)) if len(finite) else np.nan,
                           "bootstrap_replicates": int(len(finite)), "paired_pools": len(keys),
                           "destination_blocks": len(destinations),
                           "resampling_unit": "destination; multiplicities shared across HS2-year queries",
                           "interpretation": "cross-sectional sensitivity interval; not future-year uncertainty"})
    out = pd.DataFrame(output)
    out.to_csv(HERE / "results/destination_block_sensitivity.csv", index=False, na_rep="")
    return out

