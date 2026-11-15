"""Decision utility of the shortlists, 2018-2024 (main models).

An agency that screens the Top-K destinations of a pool pays a cost per screened market. We report, for each
model and K = 1..20: successes per pool and per 100 screened markets, the number of markets screened per success,
and capped realised value per pool. Discovery: a success is >= USD 50,000 after three zero years, value = outcome
exports capped at USD 85,000 (the development-period cap used in the paper). Expansion: a success is a positive
increment >= USD 50,000, value = increment capped at USD 1 million. Intervals: HS2-block bootstrap (2,000 draws).
These are realised outcomes in the selected markets, not causal effects of acting on the lists.
Writes results/utility/*.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from revision_lib import KEY, PRED_DIR, RES_DIR, TA, TB, add_rules_and_hybrid, load_stored_candidates, zfill_hs2

OUT = RES_DIR / "utility"; OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"Hybrid": "hybrid_score", "PPML": "ppml_score", "LightGBM": "lightgbm_score", "ImportDemand": "import_demand_score"}
CAP = {TA: 85_000, TB: 1_000_000}
KS = list(range(1, 21))


def main():
    d = load_stored_candidates().drop(columns="lightgbm_score")
    refit = zfill_hs2(pd.read_parquet(PRED_DIR / "lgbm_refit_scores.parquet"))[KEY + ["lightgbm_score"]]
    d = add_rules_and_hybrid(d.merge(refit, on=KEY, how="left", validate="one_to_one"))
    d = d[d.label_observed.astype(bool)].copy()
    inc = np.maximum(0, pd.to_numeric(d.realized_incremental_export_usd, errors="coerce").fillna(0).to_numpy(float))
    d["succ"] = np.where(d.task == TA, d.entry_success.astype(float), (inc >= 50_000).astype(float))
    d["val"] = np.where(d.task == TA, np.minimum(d.export_value_usd.fillna(0), CAP[TA]) * d.succ, np.minimum(inc, CAP[TB]))
    rows = []
    for (task, hs2, yr), g in d.groupby(["task", "hs2", "outcome_year"]):
        for m, c in MODELS.items():
            o = g.sort_values([c, "destination_iso3"], ascending=[False, True], kind="mergesort")
            cs, cv = np.cumsum(o.succ.to_numpy()), np.cumsum(o.val.to_numpy())
            for k in KS:
                kk = min(k, len(o))
                rows.append({"task": task, "hs2": hs2, "year": yr, "model": m, "K": k, "screened": kk,
                             "succ": cs[kk - 1], "val": cv[kk - 1]})
    p = pd.DataFrame(rows); p.to_csv(OUT / "pool_utility.csv", index=False)
    rng = np.random.default_rng(20261115)
    summ = []
    for (task, k), g in p.groupby(["task", "K"]):
        w = g.pivot_table(index=["hs2", "year"], columns="model", values=["succ", "val", "screened"])
        hs = w.index.get_level_values(0).to_numpy(); blocks = np.unique(hs)
        idx_by_block = {b: np.where(hs == b)[0] for b in blocks}
        boot = [np.concatenate([idx_by_block[b] for b in rng.choice(blocks, len(blocks))]) for _ in range(2000)]
        ref = "Hybrid" if task == TA else "PPML"
        for m in MODELS:
            s, v, n = w["succ"][m].to_numpy(), w["val"][m].to_numpy(), w["screened"][m].to_numpy()
            rec = {"task": task, "K": k, "model": m, "succ_per_pool": s.mean(), "succ_per_100_screened": 100 * s.sum() / n.sum(),
                   "screened_per_success": n.sum() / s.sum() if s.sum() else np.nan, "capped_value_per_pool": v.mean()}
            if m != ref:
                sr, vr = w["succ"][ref].to_numpy(), w["val"][ref].to_numpy()
                ds = np.array([(sr[b] - s[b]).mean() for b in boot]); dv = np.array([(vr[b] - v[b]).mean() for b in boot])
                dr = np.array([n[b].sum() / max(s[b].sum(), 1e-9) - n[b].sum() / max(sr[b].sum(), 1e-9) for b in boot])
                rec.update({"ref": ref, "ref_minus_model_succ": (sr - s).mean(), "succ_ci_low": np.quantile(ds, .025), "succ_ci_high": np.quantile(ds, .975),
                            "ref_minus_model_value": (vr - v).mean(), "value_ci_low": np.quantile(dv, .025), "value_ci_high": np.quantile(dv, .975),
                            "screenings_saved_per_success": n.sum() / s.sum() - n.sum() / sr.sum(),
                            "saved_ci_low": np.quantile(dr, .025), "saved_ci_high": np.quantile(dr, .975)})
            summ.append(rec)
    s = pd.DataFrame(summ); s.to_csv(OUT / "utility_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", lambda x: f"{x:,.3f}"):
        print(s[s.K.isin([5, 10, 20])].assign(task=lambda x: x.task.str[:6]).to_string(index=False))


if __name__ == "__main__":
    main()
