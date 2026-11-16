"""DA-1: does the discovery gain reflect complementary information or the stabilisation of a noisy classifier?
(a) gain by year against the size of the classifier's training set;
(b) fusions of each estimated model with simple rules, same rank fusion as the hybrid."""
import numpy as np, pandas as pd
import fusion, statistics as st
from revision_lib import *  # noqa
import aligned_baselines as ab

d = load_stored_candidates().drop(columns="lightgbm_score")
refit = zfill_hs2(pd.read_parquet(PRED_DIR / "lgbm_refit_scores.parquet"))[KEY + ["lightgbm_score"]]
d = add_rules_and_hybrid(d.merge(refit, on=KEY, how="left", validate="one_to_one"))
d = d[d.task == TA].copy()
print([c for c in d.columns if c.endswith("_score")])
rules = [c for c in ["import_demand_score", "gdp_score", "past_exports_score", "historical_export_score"] if c in d]
pairs = {"L+Demand": ["lightgbm_score", "import_demand_score"], "P+Demand": ["ppml_score", "import_demand_score"]}
if "gdp_score" in d:
    pairs["L+GDP"] = ["lightgbm_score", "gdp_score"]; pairs["Demand+GDP"] = ["import_demand_score", "gdp_score"]
for n, cols in pairs.items():
    f = fusion.fuse(d, ["task", "hs2", "outcome_year"], cols, tie_rule=CFG["study"].get("fusion_tie_rule", "iso3"))
    d[n] = -f["fused_rank"].to_numpy()
models = {"PPML": "ppml_score", "LightGBM": "lightgbm_score", "Hybrid": "hybrid_score", "Demand": "import_demand_score", **{n: n for n in pairs}}
if "gdp_score" in d: models["GDP"] = "gdp_score"
recs = []
for (hs2, yr), g in d.groupby(["hs2", "outcome_year"]):
    r = {"hs2": hs2, "year": yr}
    for m, c in models.items():
        pm = pool_metrics(g, c, TA, 10); r[m] = pm["ndcg"]
    recs.append(r)
w = pd.DataFrame(recs).dropna(subset=["Hybrid"])
print(len(w), w[list(models)].mean().round(4).to_string())
cons = [("L+Demand", "LightGBM"), ("Hybrid", "L+Demand"), ("P+Demand", "PPML"), ("Hybrid", "P+Demand"), ("Hybrid", "LightGBM")]
if "L+GDP" in w: cons += [("L+GDP", "LightGBM"), ("Hybrid", "L+GDP"), ("Hybrid", "Demand+GDP")]
out = []
for start in (2018, 2020):
    q = w[w.year >= start]
    for a, b in cons:
        x = (q[a] - q[b]).to_numpy(float)
        r = st.paired_contrast_inference(x, q.hs2.to_numpy(), q.year.to_numpy(), reps=10000)
        out.append({"window": f"{start}-2024", "contrast": f"{a} - {b}", "pools": len(q), "mean_a": q[a].mean(), "mean_b": q[b].mean(),
                    **{k: r[k] for k in ("estimate", "ci_low", "ci_high", "p_signflip", "years_positive", "n_years", "p_year_signflip")}})
o = pd.DataFrame(out); o.to_csv(RES_DIR / "robustness" / "mechanism_fusions.csv", index=False)
pd.set_option("display.width", 250); print(o.round(4).to_string(index=False))

# (a) training-set size by year
panel, _ = ab.load_and_freeze(write=False); panel = zfill_hs2(panel)
rows = []
for y in range(2018, 2025):
    tr = panel.loc[panel.outcome_observed.astype(bool) & panel.export_value_usd.notna() & panel.destination_import_demand.gt(0) & panel.outcome_year.le(y - 2)]
    tr = tr.loc[ab.entry_training_mask(tr) & tr.entry_success.notna()]
    q = w[w.year == y]
    rows.append({"year": y, "train_years": tr.outcome_year.nunique(), "train_rows": len(tr), "train_pos": int(tr.entry_success.sum()),
                 "pools": len(q), "LightGBM": q.LightGBM.mean(), "PPML": q.PPML.mean(), "Hybrid": q.Hybrid.mean(),
                 "H-L": (q.Hybrid - q.LightGBM).mean(), "H-P": (q.Hybrid - q.PPML).mean(), "L+Demand-L": (q["L+Demand"] - q.LightGBM).mean()})
a = pd.DataFrame(rows); a.to_csv(RES_DIR / "robustness" / "mechanism_by_training_size.csv", index=False)
print(a.round(4).to_string(index=False))
from scipy.stats import spearmanr
print("spearman(train_pos, H-L)", spearmanr(a.train_pos, a["H-L"]), "spearman(train_pos, LightGBM)", spearmanr(a.train_pos, a.LightGBM))
