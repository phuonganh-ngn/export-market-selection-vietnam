"""Objective-aligned LightGBM baselines, 2018-2024 (reply to the fairness critique).

Discovery (binary entry success):
  D_rank   LightGBM LambdaRank, groups = HS2 x outcome year, label = entry success
  D_pois   LightGBM Poisson on the outcome-year export value (USD m): same target family as PPML
Expansion (graded positive increment):
  E_rank   LightGBM LambdaRank, groups = HS2 x outcome year, label = graded gain r in {0,...,4}
  E_inc    LightGBM L2 regression on log(1 + Delta+)
  E_pinc   LightGBM Poisson on Delta+ (USD m): estimates E[(V_y - V_t)+ | X] directly
  E_pois   LightGBM Poisson on the export level V (USD m), score = max(0, V_hat - V_t): same target as PPML
Every baseline: same training windows, populations and features as the main models; hyperparameters chosen on a
chronological inner split (last training outcome year, NDCG@10 on that year's candidate pools) from a small grid;
five seeds averaged with row/column subsampling. Hybrids PPML + baseline use the same rank fusion as the paper.
Writes predictions/aligned_baseline_scores.parquet and results/aligned/*.csv.
"""
from __future__ import annotations

import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from revision_lib import expansion_grade, CFG, KEY, PRED_DIR, RES_DIR, TA, TB, add_rules_and_hybrid, gain, load_stored_candidates, pool_metrics, zfill_hs2
import fusion
import statistics as st
from experiments.explainable_reliability_gate_v1.src.task_audit import load_and_freeze
from market_discovery import entry_training_mask
from ml_models import lightgbm_bagging_params, make_preprocessor

warnings.filterwarnings("ignore")
import os
YEARS = [int(y) for y in os.environ.get("ALIGNED_YEARS", "2018,2019,2020,2021,2022,2023,2024").split(",")]
SEEDS = [int(s) for s in CFG["study"]["seeds"]]
OUT = RES_DIR / "aligned"; OUT.mkdir(parents=True, exist_ok=True)
GRID = [dict(num_leaves=15, n_estimators=120), dict(num_leaves=31, n_estimators=120),
        dict(num_leaves=31, n_estimators=300), dict(num_leaves=63, n_estimators=300)]
BASE = dict(learning_rate=0.05, min_child_samples=40, verbose=-1, n_jobs=-1, **lightgbm_bagging_params())
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)


def ndcg10(y, s, groups):
    out = []
    for g in np.unique(groups):
        m = groups == g
        yy, ss = y[m], s[m]
        if yy.max() <= 0:
            continue
        order = np.argsort(-ss, kind="mergesort")[:10]
        disc = 1 / np.log2(np.arange(2, len(order) + 2))
        dcg = (yy[order] * disc).sum()
        ideal = (np.sort(yy)[::-1][:len(order)] * disc).sum()
        out.append(dcg / ideal)
    return float(np.mean(out)) if out else np.nan


def graded(delta):
    return expansion_grade(delta)


class Spec:
    """One baseline: training rows, label, objective, and how a prediction becomes a score."""
    def __init__(self, name, task, kind, rows, label, score=None, val_rows=None, val_gain=None):
        self.name, self.task, self.kind, self.rows, self.label = name, task, kind, rows, label
        self.score = score or (lambda pred, frame: pred)
        self.val_rows, self.val_gain = val_rows, val_gain


def fit_predict(spec, train, test, params, seeds):
    tr = spec.rows(train).copy()
    tr = tr.sort_values(["outcome_year", "hs2"], kind="mergesort")
    prep = make_preprocessor(tr)
    x = prep.fit_transform(tr).astype(np.float32)
    xt = prep.transform(test).astype(np.float32)
    y = spec.label(tr)
    preds = []
    for s in seeds:
        if spec.kind == "rank":
            groups = tr.groupby(["outcome_year", "hs2"], sort=False).size().to_numpy()
            m = lgb.LGBMRanker(objective="lambdarank", random_state=s, lambdarank_truncation_level=20, **BASE, **params)
            m.fit(x, y.astype(int), group=groups)
        else:
            m = lgb.LGBMRegressor(objective=spec.kind, random_state=s, **BASE, **params)
            m.fit(x, y)
        preds.append(m.predict(xt))
    return spec.score(np.mean(preds, axis=0), test)


def tune(spec, train):
    """Chronological inner split: validate on the last training outcome year, train on years <= that - 2."""
    vy = int(train.outcome_year.max())
    inner = train[train.outcome_year <= vy - 2]
    val = spec.val_rows(train[train.outcome_year == vy])
    if spec.rows(inner).empty or val.empty or spec.label(spec.rows(inner)).sum() == 0:
        return GRID[1], np.nan
    best, best_v = GRID[1], -1
    for p in GRID:
        s = fit_predict(spec, inner, val, p, SEEDS[:1])
        v = ndcg10(spec.val_gain(val), np.asarray(s), (val.hs2 + "_" + val.outcome_year.astype(str)).to_numpy())
        if v > best_v:
            best, best_v = p, v
    return best, best_v


def specs():
    disc_rows = lambda f: f.loc[entry_training_mask(f) & f.entry_success.notna()]
    disc_val = lambda f: f.loc[f.entry_candidate_ex_ante.astype(bool) & f.strict_inactive_verified.astype(bool) & f.entry_success.notna()]
    exp_rows = lambda f: f.loc[f.task_b_eligible.astype(bool) & f.outcome_observed.astype(bool) & f.export_value_usd.notna()]
    delta = lambda f: (f.export_value_usd - f.historical_export_value).clip(lower=0).to_numpy(float)
    exp_gain = lambda f: (np.exp2(graded(delta(f))) - 1).astype(float)
    hist = lambda f: f.historical_export_value.to_numpy(float)
    return [
        Spec("D_rank", TA, "rank", disc_rows, lambda f: f.entry_success.to_numpy(int), val_rows=disc_val, val_gain=lambda f: f.entry_success.to_numpy(float)),
        Spec("D_pois", TA, "poisson", disc_rows, lambda f: f.export_value_usd.to_numpy(float) / 1e6, val_rows=disc_val,
             val_gain=lambda f: f.entry_success.to_numpy(float)),
        Spec("E_rank", TB, "rank", exp_rows, lambda f: graded(delta(f)), val_rows=exp_rows, val_gain=exp_gain),
        Spec("E_inc", TB, "regression", exp_rows, lambda f: np.log1p(delta(f)), val_rows=exp_rows, val_gain=exp_gain),
        Spec("E_pinc", TB, "poisson", exp_rows, lambda f: delta(f) / 1e6, val_rows=exp_rows, val_gain=exp_gain),
        Spec("E_pois", TB, "poisson", lambda f: f, lambda f: f.export_value_usd.to_numpy(float) / 1e6,
             score=lambda pred, f: np.maximum(0, pred * 1e6 - hist(f)), val_rows=exp_rows, val_gain=exp_gain),
    ]


def main():
    panel, _ = load_and_freeze(write=False); panel = zfill_hs2(panel)
    rows, tuned = [], []
    for year in YEARS:
        train = panel.loc[panel.outcome_observed.astype(bool) & panel.export_value_usd.notna() & panel.destination_import_demand.gt(0)
                          & panel.outcome_year.le(year - 2)]
        test = panel.loc[panel.outcome_year.eq(year) & panel.destination_import_demand.gt(0)]
        cand = {TA: test.loc[test.entry_candidate_ex_ante.astype(bool) & test.strict_inactive_verified.astype(bool)],
                TB: test.loc[test.task_b_eligible.astype(bool)]}
        for spec in specs():
            p, v = tune(spec, train)
            tuned.append({"year": year, "model": spec.name, **p, "inner_ndcg10": v})
            sc = fit_predict(spec, train, cand[spec.task], p, SEEDS)
            rows.append(cand[spec.task][["destination_iso3", "hs2", "outcome_year"]].assign(task=spec.task, model=spec.name, score=np.asarray(sc, float)))
            log(f"{year} {spec.name}: {p} inner NDCG {v:.3f}")
    scores = pd.concat(rows, ignore_index=True)
    scores.to_parquet(PRED_DIR / "aligned_baseline_scores.parquet", index=False)
    pd.DataFrame(tuned).to_csv(OUT / "tuning.csv", index=False)
    evaluate(scores)


def evaluate(scores):
    d = load_stored_candidates().drop(columns="lightgbm_score")
    refit = zfill_hs2(pd.read_parquet(PRED_DIR / "lgbm_refit_scores.parquet"))[KEY + ["lightgbm_score"]]
    d = add_rules_and_hybrid(d.merge(refit, on=KEY, how="left", validate="one_to_one"))
    wide = scores.pivot_table(index=KEY, columns="model", values="score").reset_index()
    d = d.merge(wide, on=KEY, how="left", validate="one_to_one")
    names = [c for c in ["D_rank", "D_pois", "E_rank", "E_inc", "E_pinc", "E_pois"] if c in d]
    for n in names:   # PPML + baseline hybrids, same fusion as the paper
        sub = d[d[n].notna()]
        fused = fusion.fuse(sub, ["task", "hs2", "outcome_year"], ["ppml_score", n], tie_rule=CFG["study"].get("fusion_tie_rule", "iso3"))
        d.loc[sub.index, f"H_{n}"] = -fused.loc[sub.index, "fused_rank"].to_numpy()
    models = {"PPML": "ppml_score", "LightGBM": "lightgbm_score", "Hybrid": "hybrid_score", "ImportDemand": "import_demand_score",
              **{n: n for n in names}, **{f"H_{n}": f"H_{n}" for n in names}}
    pools = []
    for (task, hs2, yr), g in d.groupby(["task", "hs2", "outcome_year"]):
        rec = {"task": task, "hs2": hs2, "year": yr}
        for m, c in models.items():
            if g[c].notna().all():
                rec[m] = pool_metrics(g, c, task, 10)["ndcg"]
        pools.append(rec)
    w = pd.DataFrame(pools)
    w.to_csv(OUT / "pool_ndcg10.csv", index=False)
    levels = w.groupby("task")[[c for c in w if c not in ("task", "hs2", "year")]].mean().T
    levels.to_csv(OUT / "ndcg10_levels.csv")
    fams = {TA: [("Hybrid", "D_rank"), ("Hybrid", "D_pois"), ("H_D_rank", "D_rank"), ("H_D_pois", "D_pois"), ("Hybrid", "H_D_rank")],
            TB: [("PPML", "E_rank"), ("PPML", "E_inc"), ("PPML", "E_pinc"), ("PPML", "E_pois"),
                 ("H_E_rank", "PPML"), ("H_E_inc", "PPML"), ("H_E_pinc", "PPML"), ("H_E_pois", "PPML")]}
    res = []
    for task, pairs in fams.items():
        q = w[w.task == task].dropna(subset=["Hybrid" if task == TA else "PPML"])
        fam = []
        for a, b in pairs:
            x = (q[a] - q[b]).to_numpy(float); ok = ~np.isnan(x)
            r = st.paired_contrast_inference(x[ok], q.hs2.to_numpy()[ok], q.year.to_numpy()[ok], reps=5000)
            fam.append({"task": task, "contrast": f"{a} - {b}", "mean_a": q[a].mean(), "mean_b": q[b].mean(), "pools": int(ok.sum()),
                        **{k: r[k] for k in ("estimate", "ci_low", "ci_high", "p_signflip", "t_twoway", "years_positive", "n_years", "p_year_signflip")}})
        for f, h in zip(fam, st.holm_adjust([f["p_signflip"] for f in fam])):
            f["p_holm"] = float(h)
        for f, h in zip(fam, st.holm_adjust([f["p_year_signflip"] for f in fam])):
            f["p_year_holm"] = float(h)
        res += fam
    r = pd.DataFrame(res); r.to_csv(OUT / "contrasts.csv", index=False)
    with pd.option_context("display.width", 240, "display.max_columns", 20, "display.float_format", lambda x: f"{x:,.3f}"):
        print(levels.round(3).to_string()); print(); print(r.assign(task=r.task.str[:6]).to_string(index=False))


if __name__ == "__main__":
    import sys
    if "--evaluate-only" in sys.argv:
        evaluate(pd.read_parquet(PRED_DIR / "aligned_baseline_scores.parquet"))
    else:
        main()
