"""Group ablation, 2018-2024: re-fit the models without one feature group at a time and measure the loss in
shortlist quality. A check on the explanations that does not depend on SHAP or on synthetic inputs.

For every outcome year: LightGBM hyperparameters are tuned once on the full feature set; then, for each feature
group, the bagged five-seed LightGBM (discovery classifier, expansion regressor) and PPML are re-fitted without
that group. Reported: NDCG@10 of the recommended model per task (hybrid for discovery, PPML for expansion) and of
LightGBM alone, with the change against the full-feature models (HS2-block bootstrap intervals).
Writes results/ablation/*.csv.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from revision_lib import CFG, KEY, RES_DIR, TA, TB, add_rules_and_hybrid, load_stored_candidates, pool_metrics, zfill_hs2
import statistics as st
from experiments.explainable_reliability_gate_v1.src.task_audit import load_and_freeze
from gravity_ppml import fit_ppml
from market_discovery import entry_training_mask
from ml_models import (FittedModel, build_estimator, feature_columns, fit_regressor, make_preprocessor, monotone_params,
                       tune_classifier, tune_regressor)
from temporal_validation import FEATURE_GROUPS

warnings.filterwarnings("ignore")
YEARS = list(range(2018, 2025))
SEEDS = [int(s) for s in CFG["study"]["seeds"]]
OUT = RES_DIR / "ablation"; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def fit_classifier_drop(train, params, seed, drop):
    nums, cats = feature_columns(drop)
    nums, cats = [c for c in nums if c in train.columns], [c for c in cats if c in train.columns]
    prep = make_preprocessor(train, drop)
    x = prep.fit_transform(train[nums + cats]).astype(np.float32)
    est = build_estimator("LightGBM", monotone_params("LightGBM", dict(params), prep), seed=seed, classification=True)
    est.fit(x, train.entry_success.astype(int).to_numpy())
    return FittedModel(prep, est, nums, cats, 1.0, FEATURE_GROUPS)


def ppml_drop(train, test, drop):
    base = list(CFG["study"].get("excluded_features") or [])
    CFG["study"]["excluded_features"] = base + [f for g in (drop or []) for f in FEATURE_GROUPS[g]]
    try:
        return fit_ppml(train, test)[0]
    finally:
        CFG["study"]["excluded_features"] = base


def main():
    panel, _ = load_and_freeze(write=False); panel = zfill_hs2(panel)
    variants = [None] + [[g] for g in FEATURE_GROUPS]
    rows = []
    for year in YEARS:
        train = panel.loc[panel.outcome_observed.astype(bool) & panel.export_value_usd.notna() & panel.destination_import_demand.gt(0)
                          & panel.outcome_year.le(year - 2)]
        test = panel.loc[panel.outcome_year.eq(year) & panel.destination_import_demand.gt(0)]
        et = train.loc[entry_training_mask(train) & train.entry_success.notna()]
        cp, _ = tune_classifier("LightGBM", et, "entry_success", lag=2, trials=CFG["study"]["tuning_trials"])
        rp, _ = tune_regressor("LightGBM", train, lag=2, trials=CFG["study"]["tuning_trials"])
        ex = test.loc[test.entry_candidate_ex_ante.astype(bool) & test.strict_inactive_verified.astype(bool)]
        eb = test.loc[test.task_b_eligible.astype(bool)]
        hist = eb.historical_export_value.to_numpy(float)
        for drop in variants:
            label = "none" if drop is None else drop[0]
            pp = ppml_drop(train, test, drop)
            ppml = pd.Series(pp, index=test.index)
            probs = np.mean([m.estimator.predict_proba(m.preprocessor.transform(ex[m.numeric_features + m.categorical_features]).astype(np.float32))[:, 1]
                             for m in (fit_classifier_drop(et, cp, s, drop) for s in SEEDS)], axis=0)
            usd = np.mean([fit_regressor("LightGBM", train, rp, s, lag=2, drop_groups=drop).predict(eb, log_target=True) for s in SEEDS], axis=0)
            rows.append(ex[["destination_iso3", "hs2", "outcome_year"]].assign(task=TA, dropped=label, ppml_score=ppml.loc[ex.index].to_numpy(),
                                                                               lightgbm_score=probs))
            rows.append(eb[["destination_iso3", "hs2", "outcome_year"]].assign(task=TB, dropped=label,
                                                                               ppml_score=np.maximum(0, ppml.loc[eb.index].to_numpy() - hist),
                                                                               lightgbm_score=np.maximum(0, usd - hist)))
            print(f"[{time.time() - T0:5.0f}s] {year} drop={label}", flush=True)
    sc = pd.concat(rows, ignore_index=True)
    sc.to_parquet(OUT / "ablation_scores.parquet", index=False)
    evaluate(sc)


def evaluate(sc):
    lab = load_stored_candidates().drop(columns=["ppml_score", "lightgbm_score"])
    res = []
    for label, g in sc.groupby("dropped"):
        d = add_rules_and_hybrid(lab.merge(g.drop(columns="dropped"), on=KEY, how="inner", validate="one_to_one"))
        for (task, hs2, yr), p in d.groupby(["task", "hs2", "outcome_year"]):
            res.append({"dropped": label, "task": task, "hs2": hs2, "year": yr,
                        "Hybrid": pool_metrics(p, "hybrid_score", task, 10)["ndcg"], "PPML": pool_metrics(p, "ppml_score", task, 10)["ndcg"],
                        "LightGBM": pool_metrics(p, "lightgbm_score", task, 10)["ndcg"]})
    w = pd.DataFrame(res); w.to_csv(OUT / "pool_ndcg10.csv", index=False)
    full = w[w.dropped == "none"].set_index(["task", "hs2", "year"])
    out = []
    for label, g in w[w.dropped != "none"].groupby("dropped"):
        g = g.set_index(["task", "hs2", "year"])
        for task, model in ((TA, "Hybrid"), (TA, "LightGBM"), (TB, "PPML"), (TB, "LightGBM")):
            a = g.loc[task, model]; b = full.loc[task, model]
            diff = (a - b.reindex(a.index)).dropna()
            idx = diff.index
            r = st.paired_contrast_inference(diff.to_numpy(float), np.array([i[0] for i in idx]), np.array([i[1] for i in idx]), reps=2000)
            out.append({"dropped": label, "task": task, "model": model, "ndcg_full": float(b.mean()), "ndcg_dropped": float(a.mean()),
                        **{k: r[k] for k in ("estimate", "ci_low", "ci_high", "years_positive", "n_years")}})
    o = pd.DataFrame(out).sort_values(["task", "model", "estimate"]); o.to_csv(OUT / "ablation_summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.float_format", lambda x: f"{x:,.3f}"):
        print(o.assign(task=o.task.str[:6]).to_string(index=False))


if __name__ == "__main__":
    import sys
    if "--evaluate-only" in sys.argv:
        evaluate(pd.read_parquet(OUT / "ablation_scores.parquet"))
    else:
        main()
