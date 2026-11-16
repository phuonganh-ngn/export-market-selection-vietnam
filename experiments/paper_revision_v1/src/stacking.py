"""Stacking benchmark for discovery: the bagged LightGBM entry classifier with out-of-time PPML predictions as features.

For a row with outcome year s the PPML features come from a PPML fitted on outcomes up to s-2, the lag used in
deployment, so training and test rows carry predictions of the same vintage. Two features are added:
log(1 + PPML prediction) and its percentile rank within the HS2-year group. Hyperparameters, seeds and the
training population are those of the main classifier. A run without the extra features checks the code path."""
import time, warnings
import numpy as np, pandas as pd
from scipy import sparse
from revision_lib import CFG, KEY, PRED_DIR, RES_DIR, TA, add_rules_and_hybrid, load_stored_candidates, pool_metrics, zfill_hs2
import statistics as st, fusion
from experiments.explainable_reliability_gate_v1.src.task_audit import load_and_freeze
from market_discovery import entry_training_mask
from ml_models import build_estimator, feature_columns, make_preprocessor, tune_classifier
from gravity_ppml import fit_ppml
warnings.filterwarnings("ignore")
T0 = time.time(); log = lambda m: print(f"[{time.time()-T0:5.0f}s] {m}", flush=True)
CFG["compute"]["model_threads"] = 8
panel, _ = load_and_freeze(write=False); panel = zfill_hs2(panel)
ok = panel.destination_import_demand.gt(0)
obs = panel.outcome_observed.astype(bool) & panel.export_value_usd.notna() & ok

# out-of-time PPML predictions for outcome years 2016..2024
panel["ppml_oot"] = np.nan
for s in range(2016, 2025):
    tr, te = panel.loc[obs & panel.outcome_year.le(s - 2)], panel.loc[ok & panel.outcome_year.eq(s)]
    pred = fit_ppml(tr, te)[0]
    panel.loc[te.index, "ppml_oot"] = pred
    log(f"PPML for {s}: train {len(tr):,} rows")
panel["ppml_log"] = np.log1p(panel.ppml_oot)

def extra(frame):
    r = frame.groupby(["hs2", "outcome_year"]).ppml_oot.rank(pct=True)
    return np.column_stack([frame.ppml_log.to_numpy(float), r.to_numpy(float)])

def prob(et, test, params, seed, stack):
    nums, cats = feature_columns(); nums, cats = [c for c in nums if c in et.columns], [c for c in cats if c in et.columns]
    prep = make_preprocessor(et)
    x, xt = prep.fit_transform(et[nums + cats]), prep.transform(test[nums + cats])
    if stack:
        x, xt = sparse.hstack([x, sparse.csr_matrix(extra(et))], format="csr"), sparse.hstack([xt, sparse.csr_matrix(extra(test))], format="csr")
    m = build_estimator("LightGBM", dict(params), seed=seed, classification=True)
    m.fit(x.astype(np.float32), et.entry_success.astype(int).to_numpy())
    return m.predict_proba(xt.astype(np.float32))[:, 1]

seeds = [int(s) for s in CFG["study"]["seeds"]]
rows = []
for y in range(2018, 2025):
    train = panel.loc[obs & panel.outcome_year.le(y - 2)]
    test = panel.loc[panel.outcome_year.eq(y) & ok & panel.entry_candidate_ex_ante.astype(bool) & panel.strict_inactive_verified.astype(bool)]
    et = train.loc[entry_training_mask(train) & train.entry_success.notna()].copy()
    params = tune_classifier("LightGBM", et, "entry_success", lag=2, trials=CFG["study"]["tuning_trials"])[0]
    out = test[["destination_iso3", "hs2", "outcome_year"]].assign(task=TA, ppml_oot=test.ppml_oot.to_numpy())
    out["plain"] = np.mean([prob(et, test, params, s, False) for s in seeds], axis=0)
    out["stack"] = np.mean([prob(et, test, params, s, True) for s in seeds], axis=0)
    rows.append(out); log(f"{y}: {len(et):,} training rows, {int(et.entry_success.sum())} positives")
sc = pd.concat(rows, ignore_index=True)
sc.to_parquet(PRED_DIR / "stacking_scores.parquet", index=False)

d = load_stored_candidates().drop(columns="lightgbm_score")
refit = zfill_hs2(pd.read_parquet(PRED_DIR / "lgbm_refit_scores.parquet"))[KEY + ["lightgbm_score"]]
d = add_rules_and_hybrid(d.merge(refit, on=KEY, how="left", validate="one_to_one"))
d = d[d.task == TA].merge(sc, on=KEY, how="left", validate="one_to_one")
print("rank corr stored vs refit PPML:", d[["ppml_score", "ppml_oot"]].corr(method="spearman").iloc[0, 1].round(5),
      "| stored vs re-run LightGBM:", d[["lightgbm_score", "plain"]].corr(method="spearman").iloc[0, 1].round(5))
f = fusion.fuse(d, ["task", "hs2", "outcome_year"], ["ppml_score", "stack"], tie_rule=CFG["study"].get("fusion_tie_rule", "iso3"))
d["H_stack"] = -f["fused_rank"].to_numpy()
models = {"PPML": "ppml_score", "LightGBM": "lightgbm_score", "Hybrid": "hybrid_score", "Plain": "plain", "Stack": "stack", "H_stack": "H_stack"}
recs = []
for (hs2, yr), g in d.groupby(["hs2", "outcome_year"]):
    recs.append({"hs2": hs2, "year": yr, **{m: pool_metrics(g, c, TA, 10)["ndcg"] for m, c in models.items()}})
w = pd.DataFrame(recs).dropna(subset=["Hybrid"]); w.to_csv(RES_DIR / "robustness" / "stacking_pool_ndcg10.csv", index=False)
print(len(w), w[list(models)].mean().round(4).to_string()); print(w.groupby("year")[list(models)].mean().round(3).to_string())
out = []
for start in (2018, 2020):
    q = w[w.year >= start]
    for a, b in [("Stack", "LightGBM"), ("Hybrid", "Stack"), ("Stack", "PPML"), ("H_stack", "Hybrid")]:
        r = st.paired_contrast_inference((q[a] - q[b]).to_numpy(float), q.hs2.to_numpy(), q.year.to_numpy(), reps=10000)
        out.append({"window": f"{start}-2024", "contrast": f"{a} - {b}", "pools": len(q), "mean_a": q[a].mean(), "mean_b": q[b].mean(),
                    **{k: r[k] for k in ("estimate", "ci_low", "ci_high", "p_signflip", "years_positive", "n_years", "p_year_signflip")}})
o = pd.DataFrame(out); o.to_csv(RES_DIR / "robustness" / "stacking_contrasts.csv", index=False)
pd.set_option("display.width", 250); print(o.round(4).to_string(index=False))
