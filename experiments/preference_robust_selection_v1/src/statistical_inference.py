from __future__ import annotations

from pathlib import Path
import itertools
import time

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parents[1]
PAIR_KEYS = ["task", "hs2", "outcome_year", "k"]
BENCHMARKS = ["Standard_minimax_regret", "Relative_robust_Weber", "Gravity_PPML",
              "LightGBM", "LambdaMART", "Equal_weight_MCDM"]


def _cluster_bootstrap(d: pd.DataFrame, value: str, reps: int = 3000, seed: int = 20261012):
    clusters = sorted(d.hs2.astype(str).unique())
    if len(clusters) < 8:
        return {"ci_low": np.nan, "ci_high": np.nan, "p_bootstrap": np.nan,
                "n_clusters": len(clusters), "bootstrap_reps": 0}
    by_cluster = {h: d.loc[d.hs2.astype(str).eq(h), value].dropna().to_numpy(float) for h in clusters}
    rng = np.random.default_rng(seed)
    observed = float(d[value].mean())
    draws = np.empty(reps, dtype=float)
    for r in range(reps):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pooled = [by_cluster[h] for h in sampled if len(by_cluster[h])]
        draws[r] = float(np.concatenate(pooled).mean()) if pooled else np.nan
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return {"ci_low": np.nan, "ci_high": np.nan, "p_bootstrap": np.nan,
                "n_clusters": len(clusters), "bootstrap_reps": 0}
    # Center the cluster bootstrap distribution at the observed estimate to form a
    # two-sided null-reference test; the CI is the percentile interval.
    centered = draws - observed
    lower = int(np.sum(centered <= -abs(observed)))
    upper = int(np.sum(centered >= abs(observed)))
    # Plus-one Monte Carlo correction prevents a finite bootstrap from reporting
    # p=0 and keeps the stated test conservative at its simulation resolution.
    p = min(1.0, 2.0 * (min(lower, upper) + 1) / (len(draws) + 1))
    return {"ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
            "p_bootstrap": p, "n_clusters": len(clusters), "bootstrap_reps": len(draws)}


def run_inference() -> dict:
    start = time.perf_counter()
    regret = pd.read_csv(HERE / "results/R5_regret_at_5_10.csv")
    regret["hs2"] = regret.hs2.astype(str).str.zfill(2)
    regret["policy"] = regret.policy.astype(str)
    rows = []
    for (task, k), g in regret.groupby(["task", "k"], sort=True):
        wide_regret = g.pivot_table(index=PAIR_KEYS[1:], columns="policy",
                                    values="worst_preference_regret_primary", aggfunc="mean")
        wide_full = g.pivot_table(index=PAIR_KEYS[1:], columns="policy", values="fully_observed_topk", aggfunc="min")
        if "PR-EMS" not in wide_regret:
            continue
        for baseline in BENCHMARKS:
            if baseline not in wide_regret:
                continue
            common = wide_regret[["PR-EMS", baseline]].dropna().index
            d = wide_regret.loc[common, ["PR-EMS", baseline]].copy()
            d["difference"] = d["PR-EMS"] - d[baseline]
            if "PR-EMS" in wide_full and baseline in wide_full:
                complete_idx = wide_full.index[(wide_full["PR-EMS"].astype(bool)) & (wide_full[baseline].astype(bool))]
                dc = d.loc[d.index.intersection(complete_idx)].copy()
            else:
                dc = d.iloc[0:0]
            # Inference is restricted to paired pools where both policies have the
            # requested K observed outcomes. Comparing policy-specific observed-slot
            # counts would give different hindsight-oracle cardinalities and is not a
            # matched decision estimand. All-pool evidence is in separate partial-ID bounds.
            for population, z in [("common_full_k_observed", dc)]:
                if z.empty:
                    continue
                z = z.reset_index().rename(columns={"index": "hs2"})
                # reset_index preserves named multi-index columns.
                if "hs2" not in z.columns:
                    first = z.columns[0]; z = z.rename(columns={first: "hs2"})
                boot = _cluster_bootstrap(z, "difference", reps=3000,
                                          seed=20261012 + int(k) + sum(map(ord, baseline + task)))
                rows.append({"task": task, "k": int(k), "metric": "worst_preference_realized_regret",
                             "policy_a": "PR-EMS", "policy_b": baseline, "estimand": "mean(A-B); negative favors PR-EMS",
                             "population": population, "paired_pool_count": len(z),
                             "paired_hs2_blocks": z.hs2.nunique(), "difference": float(z.difference.mean()),
                             "difference_median": float(z.difference.median()), **boot,
                             "inference_note": "HS2 block bootstrap keeps all years within each sampled product chapter; only three inspected years; exploratory"})

    result = pd.DataFrame(rows)
    if not result.empty:
        result["holm_p"] = np.nan
        result["holm_family"] = "task × K × population × metric"
        for _, ix in result.groupby(["task", "k", "population", "metric"]).groups.items():
            valid = result.loc[ix, "p_bootstrap"].notna()
            if valid.any():
                loc = result.loc[ix].index[valid]
                result.loc[loc, "holm_p"] = multipletests(result.loc[loc, "p_bootstrap"], method="holm")[1]
    result.to_csv(HERE / "results/statistical_comparisons.csv", index=False, na_rep="")
    (result.loc[result.metric.eq("worst_preference_realized_regret")]
     .to_csv(HERE / "results/holm_tests.csv", index=False, na_rep=""))

    partial = pd.read_csv(HERE / "results/R4_partial_identification.csv")
    if not partial.empty:
        # Recalculate sign counts on paired per-pool bounds.
        sign_rows = []
        for keys, z in partial.groupby(["task", "k", "policy_b", "weight_objective1"], sort=True):
            sign_rows.append({"task": keys[0], "k": keys[1], "policy_b": keys[2], "weight_objective1": keys[3],
                              "paired_pools": len(z),
                              "mean_difference_lower": z.difference_a_minus_b_lower.mean(),
                              "mean_difference_upper": z.difference_a_minus_b_upper.mean(),
                              "all_completions_favor_pr": bool((z.difference_a_minus_b_upper <= 0).all()),
                              "all_completions_favor_baseline": bool((z.difference_a_minus_b_lower >= 0).all()),
                              "completion_bounds": "rectangular utility support; task A conservative relaxation"})
        pd.DataFrame(sign_rows).to_csv(HERE / "results/R4_partial_identification_summary.csv", index=False, na_rep="")

    # Pool denominators and complete-case selection coverage are explicit audit outputs.
    completeness = regret.groupby(["task", "k", "policy"], as_index=False).agg(
        pools=("hs2", "size"), complete_full_k_share=("fully_observed_topk", "mean"),
        mean_missing_selected=("selected_unknown_count", "mean"),
        mean_worst_preference_regret=("worst_preference_regret_primary", "mean"))
    completeness.to_csv(HERE / "results/inference_population_audit.csv", index=False, na_rep="")
    return {"paired_comparisons": len(result), "completed_missing_bound_summaries": len(partial),
            "runtime_seconds": time.perf_counter() - start}
