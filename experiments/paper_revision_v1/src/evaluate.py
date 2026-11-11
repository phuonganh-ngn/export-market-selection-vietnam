"""Main evaluation for the revised paper: 2018-2024, PPML + re-fitted (bagged) LightGBM + hybrid + import-demand rule.

Inputs : stored PPML scores (explainable_reliability_gate_v1), predictions/lgbm_refit_scores.parquet.
Outputs: results/pool_metrics.csv, results/main_by_year.csv, results/main_contrasts.csv,
         results/stored_vs_refit_check.csv (agreement of the re-fit with the archived deterministic scores).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from revision_lib import (KEY, PRED_DIR, RES_DIR, add_rules_and_hybrid, all_pool_metrics, attach_next_year_value,
                          contrast_table, load_stored_candidates, zfill_hs2)


def main(reps: int = 10000) -> pd.DataFrame:
    stored = load_stored_candidates().rename(columns={"lightgbm_score": "lightgbm_score_archived"})
    refit = zfill_hs2(pd.read_parquet(PRED_DIR / "lgbm_refit_scores.parquet"))
    d = stored.merge(refit[KEY + ["lightgbm_score"]], on=KEY, how="left", validate="one_to_one")
    if d.lightgbm_score.isna().any():
        raise AssertionError(f"{int(d.lightgbm_score.isna().sum())} candidates have no re-fitted LightGBM score")
    # how far the re-fit moves away from the archived deterministic model
    chk = [{"task": t, "year": y, "spearman_refit_vs_archived": spearmanr(g.lightgbm_score, g.lightgbm_score_archived).correlation}
           for (t, y), g in d.groupby(["task", "outcome_year"])]
    pd.DataFrame(chk).to_csv(RES_DIR / "stored_vs_refit_check.csv", index=False)
    d = attach_next_year_value(add_rules_and_hybrid(d))
    pools = all_pool_metrics(d)
    pools.to_csv(RES_DIR / "pool_metrics.csv", index=False)
    by_year = (pools[(pools.K == 10) & ~pools.persistent].groupby(["task", "year", "model"]).ndcg
               .agg(["mean", "count"]).reset_index())
    by_year.to_csv(RES_DIR / "main_by_year.csv", index=False)
    ct = contrast_table(pools, reps=reps)
    ct.to_csv(RES_DIR / "main_contrasts.csv", index=False)
    main_rows = ct[(ct.period == "2018-2024") & (ct.K == 10) & (ct.metric == "ndcg") & ~ct.persistent]
    with pd.option_context("display.width", 220, "display.float_format", lambda x: f"{x:,.4f}"):
        print(main_rows[["task", "contrast", "mean_a", "mean_b", "estimate", "ci_low", "ci_high", "t_twoway",
                         "p_year_signflip", "years_positive", "p_holm_family3"]].to_string(index=False))
    return ct


if __name__ == "__main__":
    main()
