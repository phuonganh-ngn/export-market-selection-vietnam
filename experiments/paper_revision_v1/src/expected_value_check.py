"""Exploratory: two-part 'expected value' fusion, P(entry) x conditional scale (cf. Xiang, Ding & Huang 2026,
Front. Phys. 14:1962015), against the equal-weight rank fusion. Discovery only, 2018-2024.

EV score = LightGBM entry probability x expm1(PPML log-mean opportunity score). The LightGBM probabilities
use balanced class weights and are not calibrated, so EV is a ranking index, not an expected dollar value.
Writes results/expected_value_check.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from revision_lib import (KEY, PRED_DIR, RES_DIR, TA, add_rules_and_hybrid, all_pool_metrics, load_stored_candidates,
                          st, zfill_hs2)


def main(reps: int = 10000) -> pd.DataFrame:
    d = load_stored_candidates().drop(columns="lightgbm_score")
    d = d[d.task == TA]
    refit = zfill_hs2(pd.read_parquet(PRED_DIR / "lgbm_refit_scores.parquet"))[KEY + ["lightgbm_score"]]
    d = add_rules_and_hybrid(d.merge(refit, on=KEY, how="left", validate="one_to_one"))
    d["ev_score"] = d.lightgbm_score * np.expm1(d.ppml_score.clip(upper=50))
    models = {"Hybrid": "hybrid_score", "EV": "ev_score", "LightGBM": "lightgbm_score", "PPML": "ppml_score"}
    pools = all_pool_metrics(d, models=models, ks=(5, 10, 20), persistent_until=0)
    rows = []
    for k in (5, 10, 20):
        w = pools[(pools.K == k)].pivot_table(index=["hs2", "year"], columns="model", values="ndcg").reset_index()
        for a, b in (("EV", "Hybrid"), ("EV", "LightGBM"), ("EV", "PPML")):
            r = st.paired_contrast_inference((w[a] - w[b]).to_numpy(float), w.hs2.to_numpy(), w.year.to_numpy(), reps=reps)
            rows.append({"K": k, "contrast": f"{a} - {b}", "mean_a": w[a].mean(), "mean_b": w[b].mean(), **r})
    out = pd.DataFrame(rows)
    out.to_csv(RES_DIR / "expected_value_check.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", lambda x: f"{x:,.4f}"):
        print(out[["K", "contrast", "mean_a", "mean_b", "estimate", "ci_low", "ci_high", "p_signflip", "years_positive"]].to_string(index=False))
    return out


if __name__ == "__main__":
    main()
