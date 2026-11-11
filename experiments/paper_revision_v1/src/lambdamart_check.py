"""Is LambdaMART a better component or benchmark? Frozen window 2022-2024, stored predictions only.

Uses decision_focused_xai_v1/predictions/E6_lambdamart_predictions.parquet (5 seeds, averaged). Compares
LambdaMART alone and a PPML + LambdaMART hybrid with the PPML + LightGBM hybrid (re-fitted LightGBM).
Caution: the stored LambdaMART fits report best_iteration = 2 (early stopping almost immediately).
Writes results/lambdamart_check.csv.
"""
from __future__ import annotations

import pandas as pd

from revision_lib import (KEY, PRED_DIR, PROJECT, RES_DIR, TA, TB, add_rules_and_hybrid, all_pool_metrics,
                          fusion, load_stored_candidates, st, zfill_hs2)

LM = PROJECT / "experiments/decision_focused_xai_v1/predictions/E6_lambdamart_predictions.parquet"


def main(reps: int = 10000) -> pd.DataFrame:
    lm = zfill_hs2(pd.read_parquet(LM, columns=["task", "destination_iso3", "hs2", "outcome_year", "seed", "score", "best_iteration"]))
    print("LambdaMART best_iteration (distribution):", lm.best_iteration.value_counts().head().to_dict())
    lm = lm.groupby(KEY, as_index=False).score.mean().rename(columns={"score": "lambdamart_score"})
    d = load_stored_candidates().drop(columns="lightgbm_score")
    d = d[d.outcome_year >= 2022]
    refit = zfill_hs2(pd.read_parquet(PRED_DIR / "lgbm_refit_scores.parquet"))[KEY + ["lightgbm_score"]]
    d = d.merge(refit, on=KEY, how="left", validate="one_to_one").merge(lm, on=KEY, how="left", validate="one_to_one")
    missing = int(d.lambdamart_score.isna().sum())
    d = add_rules_and_hybrid(d.dropna(subset=["lambdamart_score"]))
    h2 = fusion.fuse(d, ["task", "hs2", "outcome_year"], ["ppml_score", "lambdamart_score"])
    d["hybrid_lambdamart_score"] = -h2.fused_rank
    models = {"LightGBM": "lightgbm_score", "LambdaMART": "lambdamart_score", "Hybrid": "hybrid_score", "HybridLM": "hybrid_lambdamart_score", "PPML": "ppml_score"}
    pools = all_pool_metrics(d, models=models, ks=(10,), persistent_until=0)
    rows = []
    for task in (TA, TB):
        w = pools[(pools.task == task)].pivot_table(index=["hs2", "year"], columns="model", values="ndcg").reset_index()
        for a, b in (("LambdaMART", "LightGBM"), ("HybridLM", "Hybrid"), ("HybridLM", "PPML")):
            r = st.paired_contrast_inference((w[a] - w[b]).to_numpy(float), w.hs2.to_numpy(), w.year.to_numpy(), reps=reps)
            rows.append({"task": task, "contrast": f"{a} - {b}", "mean_a": w[a].mean(), "mean_b": w[b].mean(), **r})
    out = pd.DataFrame(rows)
    out["candidates_without_lambdamart_score"] = missing
    out.to_csv(RES_DIR / "lambdamart_check.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", lambda x: f"{x:,.4f}"):
        print(out[["task", "contrast", "mean_a", "mean_b", "estimate", "ci_low", "ci_high", "p_signflip", "years_positive"]].to_string(index=False))
    return out


if __name__ == "__main__":
    main()
