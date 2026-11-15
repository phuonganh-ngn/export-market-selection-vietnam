"""Decision-level explainability for expansion of existing markets, 2018-2024 (companion to xai_decision.py).

Scores: LightGBM = max(0, mean over seeds of predicted USD - historical exports); PPML = max(0, predicted USD -
historical exports) from the cached per-year PPML fits. SHAP explains the LightGBM log1p prediction (raw output)
with a same-year background of expansion candidates; history features vary here and are explained too.
Swaps compare the PPML Top-10 (the preferred expansion model) with the LightGBM Top-10. Faithfulness and theory
consistency as in xai_decision.py. Margins: increase in destination import demand that brings destinations at
PPML ranks 11-15 into the PPML Top-10. Outputs in results/xai_expansion/.
"""
from __future__ import annotations

import time
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from revision_lib import HERE, KEY, PRED_DIR, RES_DIR, TB, load_stored_candidates, zfill_hs2
from experiments.explainable_reliability_gate_v1.src.task_audit import load_and_freeze
from temporal_validation import FEATURE_GROUPS
from xai_decision import EXPECTED, SEEDS, YEARS, column_meta, dense, donor_values, pool_rank, ppml_design, ppml_logmean

warnings.filterwarnings("ignore")
OUT = RES_DIR / "xai_expansion"; OUT.mkdir(parents=True, exist_ok=True)
LGB_DIR, PPML_DIR = HERE / "models" / "lightgbm", HERE / "models" / "ppml"
GROUPS = list(FEATURE_GROUPS)


def lgbm_usd(models, frame):
    return np.mean([m.predict(frame, log_target=True) for m in models], axis=0)


def main():
    t0 = time.time()
    panel, _ = load_and_freeze(write=False); panel = zfill_hs2(panel)
    stored = load_stored_candidates(); stored = stored[stored.task == TB]
    refit = zfill_hs2(pd.read_parquet(PRED_DIR / "lgbm_refit_scores.parquet"))
    stored = stored.drop(columns="lightgbm_score").merge(refit[KEY + ["lightgbm_score"]], on=KEY, validate="one_to_one")
    feats = [c for c in panel.columns if c not in stored.columns or c in ("destination_iso3", "hs2", "outcome_year")]
    checks, imp_rows, swap_rows, cons_rows, faith_rows, margin_rows = [], [], [], [], [], []
    rng = np.random.default_rng(20261112)
    for year in YEARS:
        cand = stored[stored.outcome_year == year].merge(panel[feats], on=["destination_iso3", "hs2", "outcome_year"], how="left",
                                                         validate="one_to_one").reset_index(drop=True)
        hist = cand.historical_export_value.to_numpy(float)
        cand["inc"] = np.maximum(0, cand.realized_incremental_export_usd.astype(float))
        models = [joblib.load(LGB_DIR / f"expansion_{year}_seed{s}.joblib")["bundle"] for s in SEEDS]
        cand["lgb_rebuilt"] = np.maximum(0, lgbm_usd(models, cand) - hist)
        checks.append({"year": year, "check": "lgbm_rebuild_max_abs_diff_usd", "value": float(np.abs(cand.lgb_rebuilt - cand.lightgbm_score).max())})
        po = joblib.load(PPML_DIR / f"ppml_{year}.joblib"); pm, pobj = po["model"], po["objects"]
        cand["ppml_refit"] = np.maximum(0, np.exp(ppml_logmean(pm, pobj, cand)) * 1e6 - hist)
        checks.append({"year": year, "check": "ppml_refit_vs_stored_spearman", "value": float(spearmanr(cand.ppml_refit, cand.ppml_score).correlation)})
        cand["lgb_rank"] = 0; cand["ppml_rank"] = 0
        for _, g in cand.groupby("hs2"):
            iso = g.destination_iso3.to_numpy()
            cand.loc[g.index, "lgb_rank"] = pool_rank(g.lightgbm_score.to_numpy(), iso)
            cand.loc[g.index, "ppml_rank"] = pool_rank(g.ppml_score.to_numpy(), iso)
        expl = cand[(cand.lgb_rank <= 20) | (cand.ppml_rank <= 20)].copy()
        bg = cand.sample(min(100, len(cand)), random_state=year)
        meta, ssum = None, None
        for s, m in zip(SEEDS, models):
            cols = m.numeric_features + m.categorical_features
            # float64 inputs: rounding to float32 first breaks additivity for the regressor (tested), so it is not used here
            xe, xb = (dense(m.preprocessor.transform(f[cols])).astype(np.float64) for f in (expl, bg))
            ex = shap.TreeExplainer(m.estimator, data=xb, feature_perturbation="interventional", model_output="raw")
            sv = np.asarray(ex.shap_values(xe, check_additivity=False))
            err = np.abs(sv.sum(1) + np.atleast_1d(ex.expected_value)[-1] - m.estimator.predict(xe))
            checks.append({"year": year, "check": f"shap_additivity_share_rows_err_gt_0.01_seed{s}", "value": float((err > 0.01).mean())})
            checks.append({"year": year, "check": f"shap_additivity_share_gt_1e-6_seed{s}", "value": float((err > 1e-6).mean())})
            checks.append({"year": year, "check": f"shap_additivity_share_gt_1e-3_seed{s}", "value": float((err > 1e-3).mean())})
            checks.append({"year": year, "check": f"shap_additivity_max_abs_err_seed{s}", "value": float(err.max())})
            meta = meta if meta is not None else column_meta(m.preprocessor.get_feature_names_out())
            for gname, v in pd.DataFrame(np.abs(sv), columns=meta.group).T.groupby(level=0).sum().T.mean().items():
                imp_rows.append({"year": year, "seed": s, "group": gname, "mean_abs_shap": float(v)})
            ssum = sv if ssum is None else ssum + sv
        shap_mean = pd.DataFrame(ssum / len(models), columns=meta.column, index=expl.index)
        g_shap = shap_mean.T.groupby(meta.group.to_numpy()).sum().T
        f_shap = shap_mean.T.groupby(meta.feature.to_numpy()).sum().T
        xp, pnames = ppml_design(pobj, expl); pmeta = column_meta(pnames)
        p_group = pd.DataFrame(xp * pm.coef_, columns=pmeta.column, index=expl.index).T.groupby(pmeta.group.to_numpy()).sum().T
        # swaps: PPML Top-10 vs LightGBM Top-10
        for hs2, full in cand.groupby("hs2"):
            a_idx = full[(full.ppml_rank <= 10) & (full.lgb_rank > 10)].sort_values("ppml_rank").index
            b_idx = full[(full.lgb_rank <= 10) & (full.ppml_rank > 10)].sort_values("lgb_rank").index
            for a, b in zip(a_idx, b_idx):
                dl, dp = g_shap.loc[a] - g_shap.loc[b], p_group.loc[a] - p_group.loc[b]
                for gname in sorted(set(dl.index) | set(dp.index)):
                    swap_rows.append({"year": year, "hs2": hs2, "ppml_only": cand.at[a, "destination_iso3"], "lgbm_only": cand.at[b, "destination_iso3"],
                                      "group": gname, "lgbm_delta": float(dl.get(gname, 0.0)), "ppml_delta": float(dp.get(gname, 0.0)),
                                      "ppml_only_hit": cand.at[a, "inc"] >= 50_000, "lgbm_only_hit": cand.at[b, "inc"] >= 50_000,
                                      "ppml_only_hist": cand.at[a, "historical_export_value"], "lgbm_only_hist": cand.at[b, "historical_export_value"]})
        # theory consistency of LightGBM attributions in the PPML Top-10
        med = cand.groupby("hs2")[list(EXPECTED)].median()
        for idx in expl.index[expl.ppml_rank <= 10]:
            for f, sgn in EXPECTED.items():
                x = cand.at[idx, f]; sv = f_shap.at[idx, f] if f in f_shap.columns else np.nan
                dev = x - med.at[cand.at[idx, "hs2"], f]
                if pd.isna(x) or pd.isna(sv) or pd.isna(dev) or dev == 0 or sv == 0:
                    continue
                cons_rows.append({"year": year, "feature": f, "abs_shap": abs(sv), "consistent": bool(np.sign(sv) == sgn * np.sign(dev))})
        # faithfulness on the LightGBM Top-10
        for idx in expl.index[expl.lgb_rank <= 10]:
            row_g = g_shap.loc[idx, [g for g in GROUPS if g in g_shap.columns]]
            if (row_g <= 0).all():
                continue
            main_g = row_g.idxmax(); ctrl = rng.choice([g for g in GROUPS if g != main_g])
            pool = cand[cand.hs2 == cand.at[idx, "hs2"]]; others = pool.drop(index=idx)
            for kind, grp, how in (("attributed", main_g, "median"), ("random", ctrl, "median"),
                                   ("attributed_donor", main_g, "donor"), ("random_donor", ctrl, "donor")):
                mod = cand.loc[[idx]].copy()
                fs = [f for f in FEATURE_GROUPS[grp] if f in mod.columns]
                if how == "median":
                    for f in fs:
                        mod[f] = pool[f].median()
                else:
                    mod[fs] = donor_values(pool, idx, grp).to_numpy()
                new = max(0.0, float(lgbm_usd(models, mod)[0]) - cand.at[idx, "historical_export_value"])
                nr = 1 + int((others.lightgbm_score > new).sum()) + int(((others.lightgbm_score == new) & (others.destination_iso3 < cand.at[idx, "destination_iso3"])).sum())
                faith_rows.append({"year": year, "kind": kind, "group": grp, "rank_before": int(cand.at[idx, "lgb_rank"]), "rank_after": nr, "left_top10": nr > 10})
        # margins for PPML ranks 11-15
        grid = np.round(np.arange(0, 1.501, 0.05), 2)
        for hs2, pool in cand.groupby("hs2"):
            iso = pool.destination_iso3.to_numpy(); ref = pool.ppml_refit.to_numpy(); r0 = pool_rank(ref, iso)
            for k0 in np.where((r0 >= 11) & (r0 <= 15))[0]:
                idx = pool.index[k0]
                mod = pd.concat([cand.loc[[idx]]] * len(grid), ignore_index=True)
                mod["log_destination_import_demand"] = mod.log_destination_import_demand + grid
                newv = np.maximum(0, np.exp(ppml_logmean(pm, pobj, mod)) * 1e6 - cand.at[idx, "historical_export_value"])
                need = np.nan
                for k, d in enumerate(grid):
                    s2 = ref.copy(); s2[k0] = newv[k]
                    if pool_rank(s2, iso)[k0] <= 10:
                        need = d; break
                margin_rows.append({"year": year, "hs2": hs2, "ppml_rank": int(r0[k0]), "min_log_demand_increase": need,
                                    "min_demand_increase_pct": float(np.expm1(need) * 100) if not np.isnan(need) else np.nan})
        print(f"{year}: explained {len(expl)} rows | {time.time() - t0:.0f}s", flush=True)

    ck = pd.DataFrame(checks); ck.to_csv(OUT / "fidelity_checks.csv", index=False)
    imp = pd.DataFrame(imp_rows); imp.to_csv(OUT / "group_importance_by_year_seed.csv", index=False)
    stab = []
    for year, g in imp.groupby("year"):
        w = g.pivot(index="group", columns="seed", values="mean_abs_shap")
        rho = [spearmanr(w[a], w[b]).correlation for i, a in enumerate(w) for b in list(w)[i + 1:]]
        stab.append({"level": "seeds", "year": year, "mean_spearman": np.mean(rho), "min_spearman": np.min(rho), "top3": ", ".join(w.mean(1).nlargest(3).index)})
    yw = imp.groupby(["year", "group"]).mean_abs_shap.mean().unstack(0)
    rho = [spearmanr(yw[a], yw[b]).correlation for i, a in enumerate(yw) for b in list(yw)[i + 1:]]
    stab.append({"level": "years", "year": "all", "mean_spearman": np.mean(rho), "min_spearman": np.min(rho), "top3": ", ".join(yw.mean(1).nlargest(3).index)})
    st = pd.DataFrame(stab); st.to_csv(OUT / "stability.csv", index=False)
    sw = pd.DataFrame(swap_rows); sw.to_csv(OUT / "swap_group_deltas.csv", index=False)
    s = sw.groupby("group").agg(lgbm_mean_abs=("lgbm_delta", lambda x: x.abs().mean()), lgbm_favours_ppml_only=("lgbm_delta", lambda x: (x > 0).mean()),
                                ppml_mean_abs=("ppml_delta", lambda x: x.abs().mean()), ppml_favours_ppml_only=("ppml_delta", lambda x: (x > 0).mean()))
    s["lgbm_share"] = s.lgbm_mean_abs / s.lgbm_mean_abs.sum(); s["ppml_share"] = s.ppml_mean_abs / s.ppml_mean_abs.sum()
    s = s.sort_values("ppml_share", ascending=False); s.to_csv(OUT / "swap_group_summary.csv")
    pairs = sw.groupby(["year", "hs2", "ppml_only", "lgbm_only"]).first()[["ppml_only_hit", "lgbm_only_hit", "ppml_only_hist", "lgbm_only_hist"]]
    pairs.to_csv(OUT / "swap_pairs.csv")
    cons = pd.DataFrame(cons_rows)
    cs = cons.groupby("feature").apply(lambda g: pd.Series({"n": len(g), "share_consistent": g.consistent.mean(),
                                                            "weighted": (g.abs_shap * g.consistent).sum() / g.abs_shap.sum()}))
    cs.loc["ALL"] = [len(cons), cons.consistent.mean(), (cons.abs_shap * cons.consistent).sum() / cons.abs_shap.sum()]
    cs.to_csv(OUT / "theory_consistency_lgbm.csv")
    fa = pd.DataFrame(faith_rows); fa.to_csv(OUT / "faithfulness_rows.csv", index=False)
    fs = fa.groupby("kind").agg(n=("left_top10", "size"), share_left_top10=("left_top10", "mean"), mean_rank_after=("rank_after", "mean"))
    fs.to_csv(OUT / "faithfulness_summary.csv")
    mg = pd.DataFrame(margin_rows); mg.to_csv(OUT / "decision_margins.csv", index=False)
    ms = pd.DataFrame([{"n": len(mg), "within_10pct": (mg.min_demand_increase_pct <= 10).mean(), "within_50pct": (mg.min_demand_increase_pct <= 50).mean(),
                        "within_x4.5": mg.min_demand_increase_pct.notna().mean(), "median_pct": mg.min_demand_increase_pct.median()}])
    ms.to_csv(OUT / "decision_margins_summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.float_format", lambda x: f"{x:,.4f}"):
        print("\nFIDELITY\n", ck.groupby("check").value.max().to_string())
        print("\nSTABILITY\n", st.to_string(index=False))
        print("\nSWAPS (PPML-only vs LightGBM-only Top-10 destinations)\n", s.round(3).to_string())
        print("pairs", len(pairs), "| hit rate PPML-only", round(pairs.ppml_only_hit.mean(), 3), "LightGBM-only", round(pairs.lgbm_only_hit.mean(), 3),
              "| median history USD PPML-only", round(pairs.ppml_only_hist.median()), "LightGBM-only", round(pairs.lgbm_only_hist.median()))
        print("\nTHEORY CONSISTENCY\n", cs.round(3).to_string())
        print("\nFAITHFULNESS\n", fs.round(3).to_string())
        print("\nMARGINS\n", ms.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
