from __future__ import annotations

import numpy as np
import pandas as pd

from .candidate_protocol import HERE
from .ranking_metrics import ndcg_from_gains, pool_metrics
from .rank_baselines import combined_prediction_table
from .ranking_targets import capped_export_grade

POOL_KEYS = ["task", "hs2", "outcome_year"]


def _task_b_sensitivity(d: pd.DataFrame, score: str) -> dict:
    observed = d.loc[d.label_observed.astype(bool)].copy()
    observed = observed.sort_values([score, "destination_iso3"], ascending=[False, True], kind="mergesort")
    inc = np.maximum(0.0, observed.realized_incremental_export_usd.to_numpy(float))
    grade = capped_export_grade(inc)
    log_gain = np.log1p(inc)
    raw = inc
    return {
        "ndcg10_audit_capped": ndcg_from_gains(np.exp2(grade)-1, np.exp2(grade)-1, 10),
        "ndcg10_log1p_gain": ndcg_from_gains(log_gain, log_gain, 10),
        "ndcg10_raw_usd_gain": ndcg_from_gains(raw, raw, 10),
        "ndcg10_expansion_binary": ndcg_from_gains((inc >= 50_000).astype(float), (inc >= 50_000).astype(float), 10),
    }


def evaluate_all(test: pd.DataFrame, lambdamart: pd.DataFrame):
    scores = combined_prediction_table(test, lambdamart)
    ranked_parts, pool_rows = [], []
    for (task, model, seed, hs2, year), g in scores.groupby(["task", "model", "seed", "hs2", "outcome_year"], sort=False):
        g = g.sort_values(["score", "destination_iso3"], ascending=[False, True], kind="mergesort").copy()
        g["rank_full_exante"] = np.arange(1, len(g) + 1, dtype=np.int32)
        known = g.loc[g.label_observed.astype(bool)].copy()
        known["rank_known_outcome"] = np.arange(1, len(known) + 1, dtype=np.int32)
        g = g.merge(known[["candidate_id", "rank_known_outcome"]], on="candidate_id", how="left", validate="one_to_one")
        g["candidate_pool_id"] = f"{task}|{hs2}|{year}"
        g["eligibility_status"] = np.where(g.label_observed.astype(bool), "ex-ante eligible; observed outcome",
                                           "ex-ante eligible; outcome unknown remains NA")
        metric = pool_metrics(g, "score", task)
        if task == "Task_B_existing_underexploited":
            metric.update(_task_b_sensitivity(g, "score"))
        pool_rows.append({"task": task, "model": model, "seed": int(seed), "hs2": str(hs2).zfill(2),
                          "outcome_year": int(year), **metric})
        ranked_parts.append(g)
    ranked = pd.concat(ranked_parts, ignore_index=True)
    metrics = pd.DataFrame(pool_rows)

    # Only LambdaMART is newly refitted by seed in this extension. The archived
    # baseline scores, including MLP/TabM/tree scores, are seed-averaged already
    # and are represented at seed=-1.
    preferred = metrics.loc[
        (metrics.model.eq("LambdaMART") & metrics.seed.eq(-2))
        | (~metrics.model.eq("LambdaMART") & metrics.seed.eq(-1))
    ].copy()
    metrics_out = preferred.groupby(["task", "model"], as_index=False).agg(
        **{c: (c, "mean") for c in ["ndcg_at_5", "ndcg_at_10", "precision_at_5", "precision_at_10",
                                     "recall_at_10", "top5_total_gain", "top10_total_gain",
                                     "top5_positive_increment_usd", "top10_positive_increment_usd",
                                     "ndcg10_audit_capped", "ndcg10_log1p_gain", "ndcg10_raw_usd_gain",
                                     "ndcg10_expansion_binary"] if c in preferred.columns},
        total_pools=("outcome_year", "size"),
        mean_exante_candidates=("n_exante", "mean"), mean_observed_candidates=("n_observed", "mean"),
        missing_candidates_in_full_top10=("missing_in_top10", "mean"),
    )
    # Each metric retains its own validity denominator.
    valid_counts = preferred.groupby(["task", "model"], as_index=False).agg(
        valid_ndcg5=("ndcg_at_5", "count"), valid_ndcg10=("ndcg_at_10", "count"),
        valid_precision10=("precision_at_10", "count"), valid_top10_usd=("top10_positive_increment_usd", "count"))
    metrics_out = metrics_out.merge(valid_counts, on=["task", "model"], validate="one_to_one")
    by_hs2 = preferred.groupby(["task", "model", "hs2"], as_index=False).agg(
        ndcg_at_10=("ndcg_at_10", "mean"), precision_at_10=("precision_at_10", "mean"),
        top10_positive_increment_usd=("top10_positive_increment_usd", "mean"),
        ndcg10_audit_capped=("ndcg10_audit_capped", "mean"), ndcg10_log1p_gain=("ndcg10_log1p_gain", "mean"),
        valid_ndcg10=("ndcg_at_10", "count"), pools=("outcome_year", "nunique"))
    by_year = preferred.groupby(["task", "model", "outcome_year"], as_index=False).agg(
        ndcg_at_10=("ndcg_at_10", "mean"), precision_at_10=("precision_at_10", "mean"),
        top10_positive_increment_usd=("top10_positive_increment_usd", "mean"),
        ndcg10_audit_capped=("ndcg10_audit_capped", "mean"), ndcg10_log1p_gain=("ndcg10_log1p_gain", "mean"),
        valid_ndcg10=("ndcg_at_10", "count"), hs2_chapters=("hs2", "nunique"))
    for model, subset in (("LambdaMART", metrics.loc[metrics.model.eq("LambdaMART")]),):
        seed_perf = subset.groupby(["task", "seed"], as_index=False).agg(
            ndcg_at_10=("ndcg_at_10", "mean"), precision_at_10=("precision_at_10", "mean"),
            ndcg10_audit_capped=("ndcg10_audit_capped", "mean"), valid_pools=("ndcg_at_10", "count"))
        seed_perf.to_csv(HERE / "results/seed_stability.csv", index=False, na_rep="")

    ranked.to_parquet(HERE / "predictions/E6_ranking_predictions.parquet", index=False)
    metrics.to_csv(HERE / "results/E6_pool_metrics.csv", index=False, na_rep="")
    metrics_out.to_csv(HERE / "results/E6_model_comparison.csv", index=False, na_rep="")
    metrics.loc[metrics.model.eq("LambdaMART")].to_csv(HERE / "results/E6_lambdamart_performance.csv", index=False, na_rep="")
    by_hs2.to_csv(HERE / "results/E6_by_hs2.csv", index=False, na_rep="")
    by_year.to_csv(HERE / "results/E6_by_year.csv", index=False, na_rep="")
    return ranked, metrics, metrics_out, by_hs2, by_year

