"""Robustness checks: success threshold at constant 2018 prices (re-labelled at evaluation) and 90% intervals for equivalence in expansion."""
import numpy as np, pandas as pd
import revision_lib as rl
from revision_lib import *  # noqa
import statistics as st

# US CPI-U, all items, annual average (BLS); thresholds are USD 50,000 at 2018 prices
CPI = {2018: 251.107, 2019: 255.657, 2020: 258.811, 2021: 270.970, 2022: 292.655, 2023: 304.702, 2024: 313.689}
d = load_stored_candidates().drop(columns="lightgbm_score")
refit = zfill_hs2(pd.read_parquet(PRED_DIR / "lgbm_refit_scores.parquet"))[KEY + ["lightgbm_score"]]
d = add_rules_and_hybrid(d.merge(refit, on=KEY, how="left", validate="one_to_one"))
thr = d.outcome_year.map(CPI) / CPI[2018] * 50_000
models = {"PPML": "ppml_score", "LightGBM": "lightgbm_score", "Hybrid": "hybrid_score", "Demand": "import_demand_score"}

def ndcg(g, col, y):
    r = g.sort_values([col, "destination_iso3"], ascending=[False, True], kind="mergesort")
    r = r[r.label_observed.astype(bool)]; yy = y.loc[r.index].to_numpy(float); n = min(10, len(r))
    if n == 0: return np.nan
    disc = 1 / np.log2(np.arange(2, n + 2)); idcg = (np.sort(yy)[::-1][:n] * disc).sum()
    return (yy[:n] * disc).sum() / idcg if idcg > 0 else np.nan

out = []
for task in (TA, TB):
    q = d[d.task == task]
    if task == TA:
        y = pd.Series(np.where(q.label_observed.astype(bool), (q.export_value_usd >= thr.loc[q.index]).astype(float), np.nan), index=q.index)
        n_succ = int(np.nansum(y)); 
    else:
        pos = np.maximum(0.0, pd.to_numeric(q.realized_incremental_export_usd, errors="coerce").to_numpy(float)); t = thr.loc[q.index].to_numpy(float)
        y = pd.Series(np.exp2(np.where(pos >= t, np.minimum(4, 1 + np.floor(np.log10(np.maximum(pos, t) / t))), 0)) - 1.0, index=q.index); n_succ = int((pos >= t).sum())
    recs = [{"hs2": h, "year": yr, **{m: ndcg(g, c, y) for m, c in models.items()}} for (h, yr), g in q.groupby(["hs2", "outcome_year"])]
    w = pd.DataFrame(recs).dropna(subset=["Hybrid"])
    pairs = [("Hybrid", "LightGBM"), ("Hybrid", "PPML"), ("Hybrid", "Demand")] if task == TA else [("PPML", "Hybrid"), ("PPML", "LightGBM"), ("PPML", "Demand")]
    for a, b in pairs:
        r = st.paired_contrast_inference((w[a] - w[b]).to_numpy(float), w.hs2.to_numpy(), w.year.to_numpy(), reps=10000)
        out.append({"task": task[:6], "contrast": f"{a} - {b}", "pools": len(w), "successes": n_succ, "mean_a": w[a].mean(), "mean_b": w[b].mean(),
                    **{k: r[k] for k in ("estimate", "ci_low", "ci_high", "years_positive", "n_years")}})
o = pd.DataFrame(out); o.to_csv(RES_DIR / "robustness" / "real_threshold.csv", index=False)
pd.set_option("display.width", 250); print(o.round(4).to_string(index=False))

# equivalence: 90% HS2-block bootstrap intervals for PPML against the two target-aligned LightGBM models
w = pd.read_csv(RES_DIR / "aligned" / "pool_ndcg10.csv"); w = w[w.task == TB].dropna(subset=["PPML"])
rng = np.random.default_rng(20261116); hs = w.hs2.to_numpy(); u = np.unique(hs); idx = {h: np.where(hs == h)[0] for h in u}
for b in ("E_pois", "E_rank", "Hybrid"):
    x = (w.PPML - w[b]).to_numpy(float); bs = []
    for _ in range(10000):
        ii = np.concatenate([idx[h] for h in rng.choice(u, len(u))]); bs.append(x[ii].mean())
    print("PPML -", b, round(x.mean(), 4), "90% CI", np.round(np.percentile(bs, [5, 95]), 4), "95% CI", np.round(np.percentile(bs, [2.5, 97.5]), 4))
