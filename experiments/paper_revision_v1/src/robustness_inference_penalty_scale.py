"""Robustness checks: few-cluster p-values for the two-way t, sensitivity to the PPML penalty, and the economic scale of discovery entries."""
import time, warnings
import numpy as np, pandas as pd
from scipy import stats
from revision_lib import *  # noqa
import statistics as st, fusion
from experiments.explainable_reliability_gate_v1.src.task_audit import load_and_freeze
import gravity_ppml
warnings.filterwarnings("ignore")
T0 = time.time(); log = lambda m: print(f"[{time.time()-T0:5.0f}s] {m}", flush=True)

# 1. few-cluster reference distribution for the two-way t
mc = pd.read_csv(RES_DIR / "main_contrasts.csv"); print(mc.columns.tolist())
tcol = [c for c in mc.columns if "t_two" in c][0]
mc["p_t6"] = 2 * stats.t.sf(mc[tcol].abs(), 6)
print(mc[[c for c in mc.columns if c in ("task", "contrast", "k", "metric", "estimate", tcol, "p_t6", "n_years")]].round(4).to_string())

# 3. economic scale
d = load_stored_candidates().drop(columns="lightgbm_score")
refit = zfill_hs2(pd.read_parquet(PRED_DIR / "lgbm_refit_scores.parquet"))[KEY + ["lightgbm_score"]]
d = add_rules_and_hybrid(d.merge(refit, on=KEY, how="left", validate="one_to_one"))
a = d[d.task == TA].copy(); s = a[a.entry_success == 1]
print("entries", len(s), "total USD m", s.export_value_usd.sum() / 1e6, "median", s.export_value_usd.median(), "mean", s.export_value_usd.mean(),
      "p90", s.export_value_usd.quantile(.9), "share of value in top 10 entries", s.export_value_usd.nlargest(10).sum() / s.export_value_usd.sum())
print(s.groupby("outcome_year").export_value_usd.agg(["size", "sum"]).assign(sum=lambda x: x["sum"] / 1e6).round(1).to_string())
for m, c in (("Hybrid", "hybrid_score"), ("LightGBM", "lightgbm_score"), ("PPML", "ppml_score"), ("Demand", "import_demand_score")):
    a["_r"] = a.groupby(["hs2", "outcome_year"])[c].rank(ascending=False, method="first")
    top = a[(a._r <= 10) & (a.entry_success == 1)]
    print(m, "entries found", len(top), "share", round(len(top) / len(s), 3), "value USD m", round(top.export_value_usd.sum() / 1e6, 1), "value share", round(top.export_value_usd.sum() / s.export_value_usd.sum(), 3))
print("candidates per pool", a.groupby(["hs2", "outcome_year"]).size().describe().round(1).to_dict())

# 2. PPML penalty sensitivity
panel, _ = load_and_freeze(write=False); panel = zfill_hs2(panel)
ok = panel.destination_import_demand.gt(0); obs = panel.outcome_observed.astype(bool) & panel.export_value_usd.notna() & ok
res = []
for alpha in (0.1, 0.01, 0.001):
    CFG["compute"]["gravity_ridge_alpha"] = alpha; CFG["compute"]["gravity_max_iter"] = 20000
    parts, conv = [], []
    for y in range(2018, 2025):
        tr, te = panel.loc[obs & panel.outcome_year.le(y - 2)], panel.loc[ok & panel.outcome_year.eq(y)]
        pred, _, diag, _ = gravity_ppml.fit_ppml(tr, te)
        parts.append(te[["destination_iso3", "hs2", "outcome_year"]].assign(p=pred)); conv.append(diag["converged"])
    log(f"alpha {alpha}: converged {sum(conv)}/7")
    pp = pd.concat(parts)
    x = d.merge(pp, on=["destination_iso3", "hs2", "outcome_year"], how="left")
    f = fusion.fuse(x, ["task", "hs2", "outcome_year"], ["p", "lightgbm_score"], tie_rule=CFG["study"].get("fusion_tie_rule", "iso3"))
    x["h"] = -f["fused_rank"].to_numpy()
    recs = []
    for (task, hs2, yr), g in x.groupby(["task", "hs2", "outcome_year"]):
        recs.append({"task": task, "hs2": hs2, "year": yr, "PPMLa": pool_metrics(g, "p", task, 10)["ndcg"], "Ha": pool_metrics(g, "h", task, 10)["ndcg"],
                     "LightGBM": pool_metrics(g, "lightgbm_score", task, 10)["ndcg"], "PPML": pool_metrics(g, "ppml_score", task, 10)["ndcg"]})
    w = pd.DataFrame(recs).dropna(subset=["PPMLa"])
    for task, (aa, bb) in ((TA, ("Ha", "LightGBM")), (TA, ("Ha", "PPMLa")), (TB, ("PPMLa", "Ha")), (TB, ("PPMLa", "LightGBM"))):
        q = w[w.task == task]; r = st.paired_contrast_inference((q[aa] - q[bb]).to_numpy(float), q.hs2.to_numpy(), q.year.to_numpy(), reps=5000)
        res.append({"alpha": alpha, "converged": sum(conv), "task": task[:6], "contrast": f"{aa}-{bb}", "pools": len(q), "mean_a": q[aa].mean(), "mean_b": q[bb].mean(),
                    **{k: r[k] for k in ("estimate", "ci_low", "ci_high", "years_positive", "n_years")}})
o = pd.DataFrame(res); o.to_csv(RES_DIR / "robustness" / "ppml_penalty.csv", index=False)
pd.set_option("display.width", 250); print(o.round(4).to_string(index=False))
