from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .candidate_protocol import HERE
from .decision_regret import _gain
from .ranking_metrics import pool_metrics


def _top_set(frame: pd.DataFrame, k: int) -> set[str]:
    x = frame.sort_values(["score", "destination_iso3"], ascending=[False, True], kind="mergesort").head(k)
    return set(x.destination_iso3.astype(str))


def _ndcg_task(frame: pd.DataFrame, task: str, score_col: str = "score") -> float:
    from .ranking_metrics import ndcg_from_gains
    sort_cols = [score_col, "destination_iso3"]
    ascending = [False, True]
    if score_col == "selection_frequency" and "mean_score" in frame.columns:
        sort_cols = ["selection_frequency", "mean_score", "destination_iso3"]
        ascending = [False, False, True]
    d = frame.loc[frame.label_observed.astype(bool)].sort_values(
        sort_cols, ascending=ascending, kind="mergesort")
    if task == "Task_A_new_entry":
        gain = d.entry_success.astype(float).to_numpy()
    else:
        inc = np.maximum(0.0, d.realized_incremental_export_usd.to_numpy(float))
        grade = np.minimum(4, np.floor(np.log10(1 + inc / 50_000.0)))
        gain = np.exp2(grade) - 1
    return ndcg_from_gains(gain, gain, 10)


def run_stability(ranked: pd.DataFrame, metrics: pd.DataFrame):
    pool = ["task", "hs2", "outcome_year"]
    details = []
    for (task, hs2, year), q in ranked.groupby(pool, sort=False):
        models = q.model.unique().tolist()
        # Seed-to-seed LambdaMART selection stability on the complete ex-ante candidate set.
        lam = q.loc[q.model.eq("LambdaMART") & q.seed.ge(11)]
        for k in [5, 10]:
            seed_sets = {int(seed): _top_set(g, k) for seed, g in lam.groupby("seed")}
            seed_vals = list(seed_sets)
            for i, a in enumerate(seed_vals):
                for b in seed_vals[i + 1:]:
                    union = seed_sets[a] | seed_sets[b]
                    details.append({"task": task, "hs2": hs2, "outcome_year": year, "k": k,
                                    "comparison": "LambdaMART_seed_pair", "model_a": "LambdaMART", "model_b": "LambdaMART",
                                    "seed_a": a, "seed_b": b,
                                    "jaccard": len(seed_sets[a] & seed_sets[b]) / len(union) if union else np.nan,
                                    "overlap": len(seed_sets[a] & seed_sets[b]), "universe_n": q.candidate_id.nunique()})
            mean_seed = q.loc[q.model.eq("LambdaMART") & q.seed.eq(-2)]
            pairs = [("LambdaMART", -2, name, -1) for name in models if name != "LambdaMART"]
            for model_a, seed_a, model_b, seed_b in pairs:
                a = q.loc[q.model.eq(model_a) & q.seed.eq(seed_a)]
                b = q.loc[q.model.eq(model_b) & q.seed.eq(seed_b)]
                if a.empty or b.empty:
                    continue
                sa, sb = _top_set(a, k), _top_set(b, k)
                union = sa | sb
                details.append({"task": task, "hs2": hs2, "outcome_year": year, "k": k,
                                "comparison": "LambdaMART_vs_baseline", "model_a": model_a, "model_b": model_b,
                                "seed_a": seed_a, "seed_b": seed_b,
                                "jaccard": len(sa & sb) / len(union) if union else np.nan,
                                "overlap": len(sa & sb), "universe_n": q.candidate_id.nunique()})
    detail = pd.DataFrame(details)
    if detail.empty:
        raise RuntimeError("No Top-K stability comparisons were generated")
    detail.to_csv(HERE / "results/E7_topk_stability.csv", index=False, na_rep="")
    summary = detail.groupby(["task", "k", "comparison", "model_a", "model_b"], as_index=False).agg(
        mean_jaccard=("jaccard", "mean"), median_jaccard=("jaccard", "median"),
        mean_common_markets=("overlap", "mean"), paired_pool_seed_cases=("jaccard", "count"))
    summary.to_csv(HERE / "results/E7_topk_stability_summary.csv", index=False, na_rep="")

    # Training-window stability: year-to-year comparisons only on the common destination universe.
    temporal_rows = []
    for (task, hs2), g in ranked.loc[ranked.seed.eq(-2) & ranked.model.eq("LambdaMART")].groupby(["task", "hs2"], sort=False):
        by_year = {int(y): x for y, x in g.groupby("outcome_year")}
        for y0, y1 in [(2022, 2023), (2023, 2024), (2022, 2024)]:
            if y0 not in by_year or y1 not in by_year:
                continue
            a, b = by_year[y0], by_year[y1]
            da, db = set(a.destination_iso3.astype(str)), set(b.destination_iso3.astype(str))
            common = sorted(da & db)
            for k in [5, 10]:
                ta, tb = _top_set(a, k), _top_set(b, k)
                union = ta | tb
                # Rank correlation is calculated over common destinations only.
                aa = a.set_index("destination_iso3").loc[common, "score"]
                bb = b.set_index("destination_iso3").loc[common, "score"]
                spr = float(spearmanr(aa, bb).statistic) if len(common) > 2 else np.nan
                kt = float(kendalltau(aa, bb).statistic) if len(common) > 2 else np.nan
                temporal_rows.append({"task": task, "hs2": hs2, "year_a": y0, "year_b": y1, "k": k,
                                      "common_destinations": len(common),
                                      "candidate_universe_jaccard": len(da & db) / len(da | db) if da | db else np.nan,
                                      "topk_jaccard_unrestricted": len(ta & tb) / len(union) if union else np.nan,
                                      "spearman_common_universe": spr, "kendall_common_universe": kt})
    temporal = pd.DataFrame(temporal_rows)
    temporal.to_csv(HERE / "results/E7_training_window_stability.csv", index=False, na_rep="")

    # Prespecified consensus: within each pool, rank by seed selection frequency, then mean score, then ISO3.
    robust_rows = []
    for (task, hs2, year), q in ranked.groupby(pool, sort=False):
        lam = q.loc[q.model.eq("LambdaMART") & q.seed.ge(11)].copy()
        if lam.empty:
            continue
        freq_frames = []
        for k in [10]:
            top = lam.loc[lam.rank_full_exante.le(k)].groupby("candidate_id").size().rename("selection_frequency")
            mean_score = lam.groupby("candidate_id").score.mean().rename("mean_score")
            consensus = lam.drop_duplicates("candidate_id").set_index("candidate_id").join(top).join(mean_score)
            consensus["selection_frequency"] = consensus.selection_frequency.fillna(0)
            consensus = consensus.reset_index().sort_values(
                ["selection_frequency", "mean_score", "destination_iso3"], ascending=[False, False, True], kind="mergesort")
            # Evaluate a consensus over the full pool. k-specific consensus is recorded separately.
            consensus["consensus_rank"] = np.arange(1, len(consensus) + 1)
            known = consensus.loc[consensus.label_observed.astype(bool)]
            n_metric = _ndcg_task(consensus, task, "mean_score")
            n_consensus = _ndcg_task(consensus, task, "selection_frequency")
            robust_rows.append({"task": task, "hs2": hs2, "outcome_year": year, "reference_k": k,
                                "selection_frequency_ndcg10": n_consensus, "mean_seed_score_ndcg10": n_metric,
                                "mean_seed_selection_frequency_topk": float(consensus.selection_frequency.head(k).mean()),
                                "topk_known_candidates": len(known), "policy": "frequency descending, mean score descending, ISO3 ascending"})
    robust = pd.DataFrame(robust_rows)
    robust.to_csv(HERE / "results/E7_robust_selection.csv", index=False, na_rep="")
    return detail, summary, temporal, robust


def run_candidate_eligibility_sensitivity(panel: pd.DataFrame, ranked: pd.DataFrame) -> pd.DataFrame:
    """Restrict candidates to above-median historical import markets.

    The HS2-specific threshold is computed only from rows available by y-2.
    This is a prespecified descriptive restriction, not the primary population.
    It does not replace the audit-compatible full ex-ante candidate universe.
    """
    model_policy = ranked.loc[
        (ranked.model.eq("LambdaMART") & ranked.seed.eq(-2))
        | (~ranked.model.eq("LambdaMART") & ranked.seed.eq(-1))
    ].copy()
    demand = panel[["candidate_id", "hs2", "outcome_year", "destination_import_demand",
                    "log_destination_import_demand", "eligible_ex_ante"]].drop_duplicates("candidate_id")
    model_policy = model_policy.merge(demand, on=["candidate_id", "hs2", "outcome_year"],
                                      how="left", validate="many_to_one")
    rows = []
    for (task, hs2, year, model, seed), q in model_policy.groupby(
        ["task", "hs2", "outcome_year", "model", "seed"], sort=False
    ):
        cutoff = int(year) - 2
        historical = panel.loc[
            panel.outcome_year.le(cutoff) & panel.eligible_ex_ante,
            ["hs2", "destination_import_demand"],
        ]
        thresholds = historical.groupby("hs2").destination_import_demand.median()
        threshold = thresholds.get(str(hs2).zfill(2), np.nan)
        if not np.isfinite(threshold):
            raise AssertionError(f"No training-period market-size threshold for HS2 {hs2}, year {year}")
        restricted = q.loc[q.destination_import_demand.ge(float(threshold))].copy()
        full_ndcg = _ndcg_task(q, task)
        restricted_ndcg = _ndcg_task(restricted, task) if len(restricted) else np.nan
        full_top = _top_set(q, 10)
        restricted_top = _top_set(restricted, 10)
        union = full_top | restricted_top
        observed = restricted.loc[restricted.label_observed.astype(bool)]
        if task == "Task_A_new_entry":
            gains = observed.entry_success.astype(float).to_numpy()
        else:
            inc = np.maximum(0.0, observed.realized_incremental_export_usd.to_numpy(float))
            grade = np.minimum(4, np.floor(np.log10(1.0 + inc / 50_000.0)))
            gains = np.exp2(grade) - 1.0
        k_eff = min(10, len(gains))
        ranked_observed = observed.sort_values(["score", "destination_iso3"], ascending=[False, True], kind="mergesort")
        if task == "Task_A_new_entry":
            selected_gain = float(ranked_observed.entry_success.astype(float).head(10).sum())
        else:
            inc_selected = np.maximum(0.0, ranked_observed.realized_incremental_export_usd.head(10).to_numpy(float))
            selected_gain = float(np.exp2(np.minimum(4, np.floor(np.log10(1.0 + inc_selected / 50_000.0)))).__sub__(1).sum())
        oracle_gain = float(np.sort(gains)[::-1][:10].sum())
        rows.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year),
                     "model": model, "seed": int(seed), "eligibility_rule": "historical y-2 import demand >= HS2 training-period median",
                     "threshold_usd": float(threshold), "full_candidates": len(q),
                     "restricted_candidates": len(restricted),
                     "restricted_observed_candidates": int(restricted.label_observed.sum()),
                     "restricted_unknown_candidates": int((~restricted.label_observed.astype(bool)).sum()),
                     "restricted_fraction": float(len(restricted) / len(q)) if len(q) else np.nan,
                     "full_ndcg10": full_ndcg, "restricted_ndcg10": restricted_ndcg,
                     "restricted_minus_full_ndcg10": restricted_ndcg - full_ndcg if np.isfinite(full_ndcg) and np.isfinite(restricted_ndcg) else np.nan,
                     "top10_jaccard_full_vs_restricted": len(full_top & restricted_top) / len(union) if union else np.nan,
                     "restricted_top10_selected_gain": selected_gain,
                     "restricted_oracle_top10_gain": oracle_gain,
                     "restricted_regret10": max(0.0, oracle_gain - selected_gain),
                     "k_effective_observed": k_eff,
                     "interpretation": "eligibility sensitivity; threshold uses historical features only; not same universe as primary"})
    out = pd.DataFrame(rows)
    if (out.restricted_regret10 < -1e-10).any():
        raise AssertionError("Restricted candidate regret became negative")
    out.to_csv(HERE / "results/E7_candidate_eligibility_sensitivity.csv", index=False, na_rep="")
    return out

