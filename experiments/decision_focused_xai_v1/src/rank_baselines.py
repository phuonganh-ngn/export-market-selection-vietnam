from __future__ import annotations

import pandas as pd

from .candidate_protocol import HERE, load_baseline_predictions


def combined_prediction_table(test: pd.DataFrame, lambdamart: pd.DataFrame) -> pd.DataFrame:
    base = load_baseline_predictions(test)
    base = base.rename(columns={"baseline_score": "score"})
    key = ["destination_iso3", "hs2", "outcome_year"]
    candidate_keys = test[["candidate_id", *key]].drop_duplicates(key)
    base = base.merge(candidate_keys, on=key, how="left", validate="many_to_one")
    if base.candidate_id.isna().any():
        raise AssertionError("A saved baseline candidate could not be mapped to the frozen candidate ID")
    base["seed"] = -1
    base["score_source"] = "verified_saved_prediction_seed_mean"
    lm = lambdamart[["candidate_id", "destination_iso3", "hs2", "outcome_year", "task", "label_observed",
                     "entry_success", "export_value_usd", "realized_incremental_export_usd",
                     "historical_export_value", "historical_market_share", "model", "seed", "score"]].copy()
    lm["score_source"] = "LambdaMART_fitted_outer_seed"
    common = ["candidate_id", "destination_iso3", "hs2", "outcome_year", "label_observed",
              "entry_success", "export_value_usd", "realized_incremental_export_usd", "historical_export_value",
              "historical_market_share", "task", "model", "seed", "score", "score_source"]
    base = base.merge(test[["candidate_id", "label_observed", "entry_success", "export_value_usd",
                            "realized_incremental_export_usd", "historical_export_value", "historical_market_share"]],
                      on="candidate_id", how="left", validate="many_to_one", suffixes=("", "_candidate"))
    base["task"] = base.task.astype(str)
    for c in ["label_observed", "entry_success", "export_value_usd", "realized_incremental_export_usd",
              "historical_export_value", "historical_market_share"]:
        if c + "_candidate" in base:
            base[c] = base[c].where(base[c].notna(), base[c + "_candidate"])
    base = base[common]
    lm = lm[common]
    out = pd.concat([base, lm], ignore_index=True)
    # Mean-score rankings are the primary ensemble policies for all five-seed methods.
    # Only LambdaMART has per-seed fits in this extension. Archived baseline
    # predictions are already collapsed across their training seeds upstream.
    means = out.loc[out.model.eq("LambdaMART") & out.seed.ge(11)].groupby(
        ["candidate_id", "destination_iso3", "hs2", "outcome_year", "task", "model"], as_index=False
    ).agg(score=("score", "mean"), label_observed=("label_observed", "first"),
         entry_success=("entry_success", "first"), export_value_usd=("export_value_usd", "first"),
         realized_incremental_export_usd=("realized_incremental_export_usd", "first"),
         historical_export_value=("historical_export_value", "first"),
         historical_market_share=("historical_market_share", "first"))
    means["seed"] = -2
    means["score_source"] = "mean_seed_score"
    out = pd.concat([out, means[common]], ignore_index=True)
    # Any model must predict all candidates in its task pool; no model-specific row loss is allowed.
    for (task, model, seed), g in out.groupby(["task", "model", "seed"]):
        task_manifest = test.loc[
            (test.entry_candidate_ex_ante if task == "Task_A_new_entry" else test.task_b_eligible), "candidate_id"
        ]
        if set(g.candidate_id) != set(task_manifest):
            raise AssertionError(f"Candidate set mismatch in {task}/{model}/seed {seed}")
    path = HERE / "predictions/E6_ranking_predictions.parquet"
    out.to_parquet(path, index=False)
    return out

