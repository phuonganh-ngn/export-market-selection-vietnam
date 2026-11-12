"""Robustness of the discovery result to choices that require re-fitting LightGBM, 2018-2024.

Variants (PPML scores are those of the main analysis; the hybrid is recomputed for every variant):
  membership   decomposition of the Hybrid - LightGBM NDCG@10 gain into membership and ordering (bridge lists)
  threshold    success threshold USD 25,000 and 100,000, both re-labelled only at evaluation (stored scores)
               and with the classifier re-trained on the new label
  strict       classifier trained only on strictly inactive histories (the evaluation population)
  seeds        ten disjoint seed sets of five seeds each (tuned parameters held fixed per year)
Outputs in results/robustness/ (one CSV per variant, written as soon as it finishes).
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from revision_lib import CFG, KEY, PRED_DIR, RES_DIR, TA, TB, add_rules_and_hybrid, load_stored_candidates, pool_metrics, zfill_hs2
import statistics as st
from experiments.explainable_reliability_gate_v1.src.task_audit import load_and_freeze
from market_discovery import entry_training_mask
from ml_models import fit_classifier, tune_classifier

warnings.filterwarnings("ignore")
YEARS = list(range(2018, 2025))
OUT = RES_DIR / "robustness"; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def evaluate(frame: pd.DataFrame, label: str, reps: int = 2000) -> list[dict]:
    """NDCG@10 contrasts of the hybrid against its components and the demand rule, 2018-2024."""
    d = add_rules_and_hybrid(frame)
    rows = []
    for (hs2, yr), g in d.groupby(["hs2", "outcome_year"]):
        rec = {"hs2": hs2, "year": yr}
        for name, col in (("PPML", "ppml_score"), ("LightGBM", "lightgbm_score"), ("Hybrid", "hybrid_score"), ("ImportDemand", "import_demand_score")):
            rec[name] = pool_metrics(g, col, TA, 10)["ndcg"]
        rows.append(rec)
    w = pd.DataFrame(rows).dropna(subset=["Hybrid"])
    out = []
    for b in ("LightGBM", "PPML", "ImportDemand"):
        res = st.paired_contrast_inference((w.Hybrid - w[b]).to_numpy(float), w.hs2.to_numpy(), w.year.to_numpy(), reps=reps)
        out.append({"variant": label, "contrast": f"Hybrid - {b}", "pools": len(w), "mean_hybrid": w.Hybrid.mean(), "mean_b": w[b].mean(),
                    **{k: res[k] for k in ("estimate", "ci_low", "ci_high", "p_signflip", "t_twoway", "years_positive", "n_years", "p_year_signflip")}})
    return out


def membership(frame: pd.DataFrame) -> pd.DataFrame:
    d = add_rules_and_hybrid(frame)
    d = d[d.label_observed.astype(bool)].copy()
    d["y"] = d.entry_success.astype(float)
    rows = []
    for (hs2, yr), g in d.groupby(["hs2", "outcome_year"]):
        lg = g.sort_values(["lightgbm_score", "destination_iso3"], ascending=[False, True], kind="mergesort")
        hy = g.sort_values(["hybrid_score", "destination_iso3"], ascending=[False, True], kind="mergesort")
        L, H = lg.head(10), hy.head(10)
        idcg = float(np.sum(np.sort(g.y.to_numpy())[::-1][:10] / np.log2(np.arange(2, min(10, len(g)) + 2))))
        def ndcg(lst):
            y = lst.y.to_numpy()[:10]
            return float(np.sum(y / np.log2(np.arange(2, len(y) + 2)))) / idcg if idcg > 0 else np.nan
        bridge = lg[lg.destination_iso3.isin(H.destination_iso3)].head(10)      # hybrid members, LightGBM order
        rev = hy[hy.destination_iso3.isin(L.destination_iso3)].head(10)         # LightGBM members, hybrid order
        added, displaced = H[~H.destination_iso3.isin(L.destination_iso3)], L[~L.destination_iso3.isin(H.destination_iso3)]
        rows.append({"hs2": hs2, "year": yr, "ndcg_l": ndcg(L), "ndcg_h": ndcg(H), "ndcg_bridge": ndcg(bridge), "ndcg_rev": ndcg(rev),
                     "shared": len(H) - len(added), "added": len(added), "displaced": len(displaced),
                     "added_success": added.y.sum(), "displaced_success": displaced.y.sum()})
    m = pd.DataFrame(rows)
    m["membership_fwd"], m["ordering_fwd"] = m.ndcg_bridge - m.ndcg_l, m.ndcg_h - m.ndcg_bridge
    m["ordering_rev"], m["membership_rev"] = m.ndcg_rev - m.ndcg_l, m.ndcg_h - m.ndcg_rev
    return m


def main(parts):
    panel, _ = load_and_freeze(write=False); panel = zfill_hs2(panel)
    stored = load_stored_candidates(); stored = stored[stored.task == TA]
    refit = zfill_hs2(pd.read_parquet(PRED_DIR / "lgbm_refit_scores.parquet"))
    base = stored.drop(columns="lightgbm_score").merge(refit[KEY + ["lightgbm_score"]], on=KEY, validate="one_to_one")
    log(f"base candidates {len(base):,}")
    if "membership" in parts:
        m = membership(base); m.to_csv(OUT / "membership_by_pool.csv", index=False)
        v = m.dropna(subset=["ndcg_h"])
        tot = (v.ndcg_h - v.ndcg_l).mean()
        s = pd.DataFrame([{"pools_ndcg": len(v), "gain": tot, "membership_fwd": v.membership_fwd.mean(), "ordering_fwd": v.ordering_fwd.mean(),
                           "membership_rev": v.membership_rev.mean(), "ordering_rev": v.ordering_rev.mean(),
                           "share_membership_fwd": v.membership_fwd.mean() / tot, "share_membership_rev": v.membership_rev.mean() / tot,
                           "pools_all": len(m), "added": m.added.sum(), "added_success": m.added_success.sum(), "displaced_success": m.displaced_success.sum(),
                           "mean_shared": m.shared.mean(), "mean_jaccard": (m.shared / (20 - m.shared)).mean()}])
        s.to_csv(OUT / "membership_summary.csv", index=False); log("membership\n" + s.round(4).T.to_string())
        yr = v.groupby("year").apply(lambda g: pd.Series({"gain": (g.ndcg_h - g.ndcg_l).mean(), "share_membership_fwd": g.membership_fwd.mean() / (g.ndcg_h - g.ndcg_l).mean()}))
        yr.to_csv(OUT / "membership_by_year.csv"); log("\n" + yr.round(3).to_string())

    need_fit = [p for p in parts if p in ("threshold", "strict", "seeds")]
    if not need_fit:
        return
    data = {}
    for year in YEARS:
        train = panel.loc[panel.outcome_observed.astype(bool) & panel.export_value_usd.notna() & panel.destination_import_demand.gt(0)
                          & panel.outcome_year.le(year - 2)]
        test = panel.loc[panel.outcome_year.eq(year) & panel.destination_import_demand.gt(0) & panel.entry_candidate_ex_ante.astype(bool)
                         & panel.strict_inactive_verified.astype(bool)]
        data[year] = (train, test)

    def refit_scores(label_thr=None, population=None, seeds=None, fixed_params=None):
        old = CFG["study"].get("entry_training_population")
        if population:
            CFG["study"]["entry_training_population"] = population
        seeds = seeds or [int(s) for s in CFG["study"]["seeds"]]
        rows, params_used = [], {}
        try:
            for year, (train, test) in data.items():
                et = train.loc[entry_training_mask(train) & train.entry_success.notna()].copy()
                if label_thr:
                    et["entry_success"] = (et.export_value_usd >= label_thr).astype(int)
                params = fixed_params[year] if fixed_params else tune_classifier("LightGBM", et, "entry_success", lag=2, trials=CFG["study"]["tuning_trials"])[0]
                params_used[year] = params
                p = np.mean([_prob(et, test, params, s) for s in seeds], axis=0)
                rows.append(test[["destination_iso3", "hs2", "outcome_year"]].assign(task=TA, lightgbm_score=p))
        finally:
            CFG["study"]["entry_training_population"] = old
        return pd.concat(rows, ignore_index=True), params_used

    def _prob(et, test, params, s):
        m = fit_classifier("LightGBM", et, "entry_success", params, s)
        return m.estimator.predict_proba(m.preprocessor.transform(test[m.numeric_features + m.categorical_features]).astype(np.float32))[:, 1]

    def with_scores(frame, scores):
        return frame.drop(columns="lightgbm_score").merge(scores[KEY + ["lightgbm_score"]], on=KEY, how="left", validate="one_to_one")

    def relabel(frame, thr):
        f = frame.copy()
        f["entry_success"] = np.where(f.label_observed.astype(bool), (f.export_value_usd >= thr).astype(float), np.nan)
        return f

    if "threshold" in parts:
        res = evaluate(base, "main (USD 50k)")
        for thr in (25_000, 100_000):
            res += evaluate(relabel(base, thr), f"USD {thr // 1000}k, evaluation only")
            sc, _ = refit_scores(label_thr=thr)
            res += evaluate(relabel(with_scores(base, sc), thr), f"USD {thr // 1000}k, re-trained")
            log(f"threshold {thr} done")
        r = pd.DataFrame(res); r.to_csv(OUT / "threshold.csv", index=False); log("\n" + r.round(3).to_string(index=False))

    if "strict" in parts:
        sc, _ = refit_scores(population="strict_inactive")
        r = pd.DataFrame(evaluate(with_scores(base, sc), "strict-zero training")); r.to_csv(OUT / "strict_training.csv", index=False)
        log("\n" + r.round(3).to_string(index=False))

    if "seeds" in parts:
        fixed = {}
        for year, (train, _) in data.items():
            et = train.loc[entry_training_mask(train) & train.entry_success.notna()]
            fixed[year] = tune_classifier("LightGBM", et, "entry_success", lag=2, trials=CFG["study"]["tuning_trials"])[0]
        res = []
        for k in range(10):
            seeds = [1000 * (k + 1) + j for j in range(5)]
            sc, _ = refit_scores(seeds=seeds, fixed_params=fixed)
            res += [dict(r, seed_set=k + 1) for r in evaluate(with_scores(base, sc), f"seed set {k + 1}", reps=1000)]
            pd.DataFrame(res).to_csv(OUT / "seed_sets.csv", index=False)
            log(f"seed set {k + 1}: {res[-3]['estimate']:.4f}")
        r = pd.DataFrame(res)
        log("\n" + r.groupby("contrast").estimate.agg(["mean", "std", "min", "max"]).round(4).to_string()
            + "\nshare with CI above 0: " + r.assign(pos=r.ci_low > 0).groupby("contrast").pos.mean().round(2).to_string())


def seeds_expansion(panel, reps=1000):
    """Ten disjoint seed sets for the expansion regressor; tuned leaves held fixed per year."""
    from ml_models import fit_regressor, tune_regressor
    stored = load_stored_candidates(); stored = stored[stored.task == TB]
    refit = zfill_hs2(pd.read_parquet(PRED_DIR / "lgbm_refit_scores.parquet"))
    base = stored.drop(columns="lightgbm_score").merge(refit[KEY + ["lightgbm_score"]], on=KEY, validate="one_to_one")
    data, fixed = {}, {}
    for year in YEARS:
        train = panel.loc[panel.outcome_observed.astype(bool) & panel.export_value_usd.notna() & panel.destination_import_demand.gt(0)
                          & panel.outcome_year.le(year - 2)]
        test = panel.loc[panel.outcome_year.eq(year) & panel.destination_import_demand.gt(0) & panel.task_b_eligible.astype(bool)]
        data[year] = (train, test)
        fixed[year] = tune_regressor("LightGBM", train, lag=2, trials=CFG["study"]["tuning_trials"])[0]

    def ev(frame, label):
        d = add_rules_and_hybrid(frame)
        rows = []
        for (hs2, yr), g in d.groupby(["hs2", "outcome_year"]):
            rows.append({"hs2": hs2, "year": yr, **{n: pool_metrics(g, c, TB, 10)["ndcg"] for n, c in
                         (("PPML", "ppml_score"), ("LightGBM", "lightgbm_score"), ("Hybrid", "hybrid_score"), ("ImportDemand", "import_demand_score"))}})
        w = pd.DataFrame(rows).dropna(subset=["PPML"])
        out = []
        for b in ("Hybrid", "LightGBM", "ImportDemand"):
            r = st.paired_contrast_inference((w.PPML - w[b]).to_numpy(float), w.hs2.to_numpy(), w.year.to_numpy(), reps=reps)
            out.append({"variant": label, "contrast": f"PPML - {b}", "pools": len(w), "mean_ppml": w.PPML.mean(), "mean_b": w[b].mean(),
                        **{k: r[k] for k in ("estimate", "ci_low", "ci_high", "p_signflip", "years_positive", "n_years")}})
        return out

    res = ev(base, "main seeds")
    for k in range(10):
        seeds = [1000 * (k + 1) + j for j in range(5)]
        rows = []
        for year, (train, test) in data.items():
            usd = np.mean([fit_regressor("LightGBM", train, fixed[year], s, lag=2).predict(test, log_target=True) for s in seeds], axis=0)
            rows.append(test[["destination_iso3", "hs2", "outcome_year"]].assign(task=TB, lightgbm_score=np.maximum(0, usd - test.historical_export_value.to_numpy(float))))
        sc = pd.concat(rows, ignore_index=True)
        f = base.drop(columns="lightgbm_score").merge(sc[KEY + ["lightgbm_score"]], on=KEY, how="left", validate="one_to_one")
        res += [dict(r, seed_set=k + 1) for r in ev(f, f"seed set {k + 1}")]
        pd.DataFrame(res).to_csv(OUT / "seed_sets_expansion.csv", index=False)
        log(f"expansion seed set {k + 1}: PPML - Hybrid {res[-3]['estimate']:.4f}, PPML - LightGBM {res[-2]['estimate']:.4f}")
    r = pd.DataFrame(res); r = r[r.variant != "main seeds"]
    log("\n" + r.groupby("contrast").estimate.agg(["mean", "std", "min", "max"]).round(4).to_string()
        + "\nshare with CI above 0: " + r.assign(pos=r.ci_low > 0).groupby("contrast").pos.mean().round(2).to_string())


if __name__ == "__main__":
    import sys
    if "--expansion-seeds" in sys.argv:
        p, _ = load_and_freeze(write=False)
        seeds_expansion(zfill_hs2(p))
        raise SystemExit
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="membership,threshold,strict,seeds")
    main(ap.parse_args().parts.split(","))
