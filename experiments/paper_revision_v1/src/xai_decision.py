"""Decision-level explainability for new-market discovery, 2018-2024 (bagged LightGBM + PPML + hybrid).

Requires the saved seed models (refit_lightgbm.py --save-models). PPML is re-fitted per year and cached.

1. Fidelity checks: scores rebuilt from the saved models equal the scores used in evaluation; the re-fitted
   PPML reproduces the stored PPML ranking; SHAP values add up to each model's raw margin.
2. SHAP (interventional TreeSHAP, log-odds) with a background of candidates from the same year, so features
   that are constant across candidates (export history) get exactly zero attribution. Averaged over seeds and
   summed into feature groups.
3. Stability of group importance across seeds and across years.
4. Swaps: destinations the hybrid adds to / displaces from the LightGBM Top-10, with LightGBM SHAP and PPML
   log-mean contribution differences by group.
5. Theory consistency: share of attribution whose direction matches the expected sign of the feature
   (manuscript Table 'Variables, theoretical grounding and expected signs'); PPML coefficient signs.
6. Faithfulness: neutralise (set to the pool median) the feature group that SHAP names as the main reason
   for a Top-10 placement and check whether the destination leaves the Top-10, against a random group.
7. Decision margins: for destinations ranked 11-15 by the hybrid, the smallest increase in destination import
   demand that would bring them into the hybrid Top-10 (all else equal), using both component models.
Outputs in results/xai/.
"""
from __future__ import annotations

import time
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from revision_lib import HERE, KEY, PRED_DIR, RES_DIR, TA, load_stored_candidates, zfill_hs2
from experiments.explainable_reliability_gate_v1.src.task_audit import load_and_freeze
from gravity_ppml import fit_ppml
from temporal_validation import FEATURE_GROUPS

warnings.filterwarnings("ignore")
YEARS = list(range(2018, 2025))
SEEDS = [11, 23, 37, 51, 73]
OUT = RES_DIR / "xai"; OUT.mkdir(parents=True, exist_ok=True)
LGB_DIR, PPML_DIR = HERE / "models" / "lightgbm", HERE / "models" / "ppml"; PPML_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_TO_GROUP = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
# Expected signs (manuscript variable table); inflation and urbanisation have no clear sign and are left out.
EXPECTED = {"log_destination_import_demand": 1, "destination_import_demand_growth": 1, "destination_supplier_concentration": -1,
            "log_gdp": 1, "log_gdp_per_capita": 1, "log_population": 1, "log_distance": -1, "contiguous": 1,
            "colonial_link": 1, "landlocked": -1, "government_effectiveness": 1, "regulatory_quality": 1,
            "rule_of_law": 1, "political_stability": 1, "internet_usage": 1, "trade_openness": 1}
NEUTRALISABLE = [g for g in FEATURE_GROUPS if g != "historical_trade"]   # history is constant for discovery candidates


def donor_values(pool, idx, grp):
    """Values of feature group `grp` taken from a real candidate in the same pool: the one closest to `idx` on all
    other feature groups (z-scored within the pool, missing values at the mean). Avoids synthetic input combinations."""
    other = [f for g, fs in FEATURE_GROUPS.items() if g != grp for f in fs if f in pool.columns]
    z = pool[other].astype(float)
    sd = z.std(ddof=0).replace(0, np.nan)
    z = ((z - z.mean()) / sd).fillna(0.0)
    dist = ((z - z.loc[idx]) ** 2).sum(axis=1).drop(index=idx)
    return pool.loc[dist.idxmin(), [f for f in FEATURE_GROUPS[grp] if f in pool.columns]]


def column_meta(names) -> pd.DataFrame:
    rows = []
    for n in names:
        block, _, feat = n.partition("__")
        if block == "numeric" and feat.startswith("missingindicator_"):
            rows.append((n, feat.replace("missingindicator_", ""), "predictor_missingness"))
        elif block == "numeric":
            rows.append((n, feat, FEATURE_TO_GROUP.get(feat, "other")))
        elif feat.startswith("destination_iso3"):
            rows.append((n, "destination_iso3", "destination_identifier"))
        else:
            rows.append((n, "hs2", "product_indicator"))
    return pd.DataFrame(rows, columns=["column", "feature", "group"])


def dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def load_lgbm(year):
    return [joblib.load(LGB_DIR / f"discovery_{year}_seed{s}.joblib")["bundle"] for s in SEEDS]


def lgbm_prob(models, frame) -> np.ndarray:
    return np.mean([m.estimator.predict_proba(dense(m.preprocessor.transform(frame[m.numeric_features + m.categorical_features])).astype(np.float32))[:, 1]
                    for m in models], axis=0)


def ppml_design(objects, frame):
    nums, cats, imputer, scaler, enc = objects
    xnum = scaler.transform(imputer.transform(frame[nums])).astype(float)
    ind = [f"missingindicator_{nums[i]}" for i in imputer.indicator_.features_] if imputer.indicator_ is not None else []
    names = [f"numeric__{n}" for n in nums + ind] + [f"category__{c}" for c in enc.get_feature_names_out(cats)]
    return np.hstack([xnum, dense(enc.transform(frame[cats]))]), names


def ppml_logmean(model, objects, frame) -> np.ndarray:
    x, _ = ppml_design(objects, frame)
    return model.intercept_ + x @ model.coef_          # log of predicted USD millions


def pool_rank(scores: np.ndarray, iso: np.ndarray) -> np.ndarray:
    order = np.lexsort((iso, -scores))
    r = np.empty(len(scores), int); r[order] = np.arange(1, len(scores) + 1); return r


def hybrid_rank(ppml_s, lgb_s, iso) -> np.ndarray:
    n = len(iso); den = max(1, n - 1)
    fused = np.round(0.5 * (n - pool_rank(ppml_s, iso)) / den + 0.5 * (n - pool_rank(lgb_s, iso)) / den, 12)
    return pool_rank(fused, iso)


def main():
    t0 = time.time()
    panel, _ = load_and_freeze(write=False); panel = zfill_hs2(panel)
    stored = load_stored_candidates(); stored = stored[stored.task == TA]
    refit = zfill_hs2(pd.read_parquet(PRED_DIR / "lgbm_refit_scores.parquet"))
    stored = stored.drop(columns="lightgbm_score").merge(refit[KEY + ["lightgbm_score"]], on=KEY, validate="one_to_one")
    feats = [c for c in panel.columns if c not in stored.columns or c in ("destination_iso3", "hs2", "outcome_year")]
    checks, imp_rows, swap_rows, cons_rows, coef_rows, faith_rows, margin_rows = [], [], [], [], [], [], []
    rng = np.random.default_rng(20261112)
    for year in YEARS:
        cand = stored[stored.outcome_year == year].merge(panel[feats], on=["destination_iso3", "hs2", "outcome_year"], how="left", validate="one_to_one")
        cand = cand.reset_index(drop=True)
        models = load_lgbm(year)
        # 1a. rebuild LightGBM scores from saved models
        rebuilt = lgbm_prob(models, cand)
        checks.append({"year": year, "check": "lgbm_rebuild_max_abs_diff", "value": float(np.abs(rebuilt - cand.lightgbm_score).max())})
        # PPML: re-fit (cached) and compare with the stored ranking
        pf = PPML_DIR / f"ppml_{year}.joblib"
        if pf.exists():
            ppml_model, ppml_obj = (lambda o: (o["model"], o["objects"]))(joblib.load(pf))
        else:
            train = panel.loc[panel.outcome_observed.astype(bool) & panel.export_value_usd.notna() & panel.destination_import_demand.gt(0)
                              & panel.outcome_year.le(year - 2)]
            _, ppml_model, diag, ppml_obj = fit_ppml(train, cand)
            joblib.dump({"model": ppml_model, "objects": ppml_obj, "diagnostics": diag}, pf)
        cand["ppml_refit_logmean"] = ppml_logmean(ppml_model, ppml_obj, cand)
        checks.append({"year": year, "check": "ppml_refit_vs_stored_spearman",
                       "value": float(spearmanr(cand.ppml_refit_logmean, cand.ppml_score).correlation)})
        # ranks per pool
        cand["lgb_rank"] = 0; cand["hyb_rank"] = 0
        for _, g in cand.groupby("hs2"):
            iso = g.destination_iso3.to_numpy()
            cand.loc[g.index, "lgb_rank"] = pool_rank(g.lightgbm_score.to_numpy(), iso)
            cand.loc[g.index, "hyb_rank"] = hybrid_rank(g.ppml_score.to_numpy(), g.lightgbm_score.to_numpy(), iso)
        # 2. SHAP on the rows that matter for the decisions
        expl = cand[(cand.lgb_rank <= 20) | (cand.hyb_rank <= 20)].copy()
        bg = cand.sample(min(100, len(cand)), random_state=year)
        meta = None; shap_sum = None
        for s, m in zip(SEEDS, models):
            cols = m.numeric_features + m.categorical_features
            # LightGBM was trained on float32 inputs: round to float32 first so SHAP sees the same split decisions
            xe = dense(m.preprocessor.transform(expl[cols])).astype(np.float32).astype(np.float64)
            xb = dense(m.preprocessor.transform(bg[cols])).astype(np.float32).astype(np.float64)
            ex = shap.TreeExplainer(m.estimator, data=xb, feature_perturbation="interventional", model_output="raw")
            sv = ex.shap_values(xe, check_additivity=False)
            sv = sv[1] if isinstance(sv, list) else sv
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            ev = np.atleast_1d(ex.expected_value)[-1]
            margin = m.estimator.predict(xe, raw_score=True)
            err = np.abs(sv.sum(1) + ev - margin)
            checks.append({"year": year, "check": f"shap_additivity_max_abs_err_seed{s}", "value": float(err.max())})
            checks.append({"year": year, "check": f"shap_additivity_share_gt_1e-6_seed{s}", "value": float((err > 1e-6).mean())})
            checks.append({"year": year, "check": f"shap_additivity_share_gt_1e-3_seed{s}", "value": float((err > 1e-3).mean())})
            if meta is None:
                meta = column_meta(m.preprocessor.get_feature_names_out())
            grp = pd.DataFrame(np.abs(sv), columns=meta.group).T.groupby(level=0).sum().T.mean()
            for gname, v in grp.items():
                imp_rows.append({"year": year, "seed": s, "group": gname, "mean_abs_shap": float(v)})
            shap_sum = sv if shap_sum is None else shap_sum + sv
        shap_mean = pd.DataFrame(shap_sum / len(models), columns=meta.column, index=expl.index)
        g_shap = shap_mean.T.groupby(meta.group.to_numpy()).sum().T            # rows x groups
        f_shap = shap_mean.T.groupby(meta.feature.to_numpy()).sum().T          # rows x raw features (value + its missing indicator)
        # PPML log-mean contributions for the same rows
        xp, pnames = ppml_design(ppml_obj, expl)
        pmeta = column_meta(pnames)
        p_contrib = pd.DataFrame(xp * ppml_model.coef_, columns=pmeta.column, index=expl.index)
        p_group = p_contrib.T.groupby(pmeta.group.to_numpy()).sum().T
        for col, b in zip(pmeta.column, ppml_model.coef_):
            f = pmeta.set_index("column").loc[col, "feature"]
            if f in EXPECTED and not col.split("__")[1].startswith("missingindicator_"):
                coef_rows.append({"year": year, "feature": f, "coef_std_scale": float(b), "expected_sign": EXPECTED[f],
                                  "consistent": bool(np.sign(b) == EXPECTED[f]) if b != 0 else np.nan})
        # 4. swaps (pair added and displaced destinations in rank order)
        for hs2, g in expl.groupby("hs2"):
            full = cand[cand.hs2 == hs2]
            added = full[(full.hyb_rank <= 10) & (full.lgb_rank > 10)].sort_values("hyb_rank").index
            displaced = full[(full.lgb_rank <= 10) & (full.hyb_rank > 10)].sort_values("lgb_rank").index
            for a, b in zip(added, displaced):
                dl, dp = g_shap.loc[a] - g_shap.loc[b], p_group.loc[a] - p_group.loc[b]
                for gname in sorted(set(dl.index) | set(dp.index)):
                    swap_rows.append({"year": year, "hs2": hs2, "added": cand.at[a, "destination_iso3"], "displaced": cand.at[b, "destination_iso3"],
                                      "group": gname, "lgbm_delta": float(dl.get(gname, 0.0)), "ppml_delta": float(dp.get(gname, 0.0)),
                                      "added_success": cand.at[a, "entry_success"], "displaced_success": cand.at[b, "entry_success"]})
        # 5. theory consistency of LightGBM attributions for destinations in the hybrid Top-10
        top = expl[expl.hyb_rank <= 10]
        med = cand.groupby("hs2")[list(EXPECTED)].median()
        for idx in top.index:
            for f, sgn in EXPECTED.items():
                x, sv = cand.at[idx, f], f_shap.at[idx, f] if f in f_shap.columns else np.nan
                dev = x - med.at[cand.at[idx, "hs2"], f]
                if pd.isna(x) or pd.isna(sv) or pd.isna(dev) or dev == 0 or sv == 0:
                    continue
                cons_rows.append({"year": year, "feature": f, "abs_shap": abs(sv), "consistent": bool(np.sign(sv) == sgn * np.sign(dev))})
        # 6. faithfulness: neutralise the main positive group of each LightGBM Top-10 destination
        for idx in expl.index[expl.lgb_rank <= 10]:
            row_g = g_shap.loc[idx, [g for g in NEUTRALISABLE if g in g_shap.columns]]
            if (row_g <= 0).all():
                continue
            main = row_g.idxmax()
            ctrl = rng.choice([g for g in NEUTRALISABLE if g != main])
            pool = cand[cand.hs2 == cand.at[idx, "hs2"]]
            for kind, grp, how in (("attributed", main, "median"), ("random", ctrl, "median"),
                                   ("attributed_donor", main, "donor"), ("random_donor", ctrl, "donor")):
                mod = cand.loc[[idx]].copy()
                fs = [f for f in FEATURE_GROUPS[grp] if f in mod.columns]
                if how == "median":
                    for f in fs:
                        mod[f] = pool[f].median()
                else:
                    mod[fs] = donor_values(pool, idx, grp).to_numpy()
                new = lgbm_prob(models, mod)[0]
                others = pool.drop(index=idx)
                new_rank = 1 + int((others.lightgbm_score > new).sum()) + int(((others.lightgbm_score == new) & (others.destination_iso3 < cand.at[idx, "destination_iso3"])).sum())
                lg = pool.lightgbm_score.copy(); lg.loc[idx] = new
                hr = hybrid_rank(pool.ppml_score.to_numpy(), lg.to_numpy(), pool.destination_iso3.to_numpy())[list(pool.index).index(idx)]
                faith_rows.append({"year": year, "hs2": cand.at[idx, "hs2"], "destination": cand.at[idx, "destination_iso3"], "kind": kind, "group": grp,
                                   "rank_before": int(cand.at[idx, "lgb_rank"]), "rank_after": new_rank, "left_top10": new_rank > 10,
                                   "hyb_rank_before": int(cand.at[idx, "hyb_rank"]), "hyb_rank_after": int(hr)})
        # 7. decision margins for hybrid ranks 11-15 (both components re-scored)
        grid = np.round(np.arange(0, 1.501, 0.05), 2)
        for hs2, pool in cand.groupby("hs2"):
            iso = pool.destination_iso3.to_numpy(); pos = {i: k for k, i in enumerate(pool.index)}
            ppml_ref = pool.ppml_refit_logmean.to_numpy(); lgb_ref = pool.lightgbm_score.to_numpy()
            base_rank = hybrid_rank(ppml_ref, lgb_ref, iso)
            for idx in pool.index[(base_rank >= 11) & (base_rank <= 15)]:
                mod = pd.concat([cand.loc[[idx]]] * len(grid), ignore_index=True)
                mod["log_destination_import_demand"] = mod.log_destination_import_demand + grid
                p_new, l_new = ppml_logmean(ppml_model, ppml_obj, mod), lgbm_prob(models, mod)
                need = np.nan
                for k, delta in enumerate(grid):
                    ps, ls = ppml_ref.copy(), lgb_ref.copy(); ps[pos[idx]], ls[pos[idx]] = p_new[k], l_new[k]
                    if hybrid_rank(ps, ls, iso)[pos[idx]] <= 10:
                        need = delta; break
                margin_rows.append({"year": year, "hs2": hs2, "destination": cand.at[idx, "destination_iso3"], "hybrid_rank": int(base_rank[pos[idx]]),
                                    "min_log_demand_increase": need, "min_demand_increase_pct": float(np.expm1(need) * 100) if not np.isnan(need) else np.nan,
                                    "success_observed": cand.at[idx, "entry_success"]})
        print(f"{year}: explained {len(expl)} rows | {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ summaries
    pd.DataFrame(checks).to_csv(OUT / "fidelity_checks.csv", index=False)
    imp = pd.DataFrame(imp_rows); imp.to_csv(OUT / "group_importance_by_year_seed.csv", index=False)
    stab = []
    for year, g in imp.groupby("year"):
        w = g.pivot(index="group", columns="seed", values="mean_abs_shap")
        rho = [spearmanr(w[a], w[b]).correlation for i, a in enumerate(w) for b in list(w)[i + 1:]]
        top3 = w.mean(1).sort_values(ascending=False).index[:3].tolist()
        stab.append({"level": "seeds_within_year", "year": year, "mean_spearman": float(np.mean(rho)), "min_spearman": float(np.min(rho)), "top3_groups": ", ".join(top3)})
    yw = imp.groupby(["year", "group"]).mean_abs_shap.mean().unstack(0)
    rho_y = [spearmanr(yw[a], yw[b]).correlation for i, a in enumerate(yw) for b in list(yw)[i + 1:]]
    stab.append({"level": "across_years", "year": "all", "mean_spearman": float(np.mean(rho_y)), "min_spearman": float(np.min(rho_y)),
                 "top3_groups": ", ".join(yw.mean(1).sort_values(ascending=False).index[:3])})
    pd.DataFrame(stab).to_csv(OUT / "stability.csv", index=False)
    sw = pd.DataFrame(swap_rows); sw.to_csv(OUT / "swap_group_deltas.csv", index=False)
    if len(sw):
        s = sw.groupby("group").agg(lgbm_mean_abs=("lgbm_delta", lambda x: x.abs().mean()), lgbm_favours_added=("lgbm_delta", lambda x: (x > 0).mean()),
                                    ppml_mean_abs=("ppml_delta", lambda x: x.abs().mean()), ppml_favours_added=("ppml_delta", lambda x: (x > 0).mean()))
        s["lgbm_share"] = s.lgbm_mean_abs / s.lgbm_mean_abs.sum(); s["ppml_share"] = s.ppml_mean_abs / s.ppml_mean_abs.sum()
        s.sort_values("lgbm_share", ascending=False).to_csv(OUT / "swap_group_summary.csv")
        pairs = sw.groupby(["year", "hs2", "added", "displaced"]).agg(lgbm=("lgbm_delta", "sum"), ppml=("ppml_delta", "sum"),
                                                                      added_success=("added_success", "first"), displaced_success=("displaced_success", "first"))
        pairs.to_csv(OUT / "swap_pairs.csv")
    cons = pd.DataFrame(cons_rows)
    cs = cons.groupby("feature").apply(lambda g: pd.Series({"n": len(g), "share_consistent": g.consistent.mean(),
                                                            "attribution_weighted_consistent": (g.abs_shap * g.consistent).sum() / g.abs_shap.sum()}))
    overall = pd.DataFrame([{"feature": "ALL", "n": len(cons), "share_consistent": cons.consistent.mean(),
                             "attribution_weighted_consistent": (cons.abs_shap * cons.consistent).sum() / cons.abs_shap.sum()}]).set_index("feature")
    pd.concat([cs, overall]).to_csv(OUT / "theory_consistency_lgbm.csv")
    pd.DataFrame(coef_rows).to_csv(OUT / "theory_consistency_ppml_coef.csv", index=False)
    fa = pd.DataFrame(faith_rows); fa.to_csv(OUT / "faithfulness_rows.csv", index=False)
    fsum = fa.groupby("kind").agg(n=("left_top10", "size"), share_left_top10=("left_top10", "mean"),
                                  mean_rank_before=("rank_before", "mean"), mean_rank_after=("rank_after", "mean"), mean_hybrid_rank_after=("hyb_rank_after", "mean"))
    fsum.to_csv(OUT / "faithfulness_summary.csv")
    mg = pd.DataFrame(margin_rows); mg.to_csv(OUT / "decision_margins.csv", index=False)
    msum = pd.DataFrame([{"n": len(mg), "reachable_within_+10pct": (mg.min_demand_increase_pct <= 10).mean(),
                          "reachable_within_+50pct": (mg.min_demand_increase_pct <= 50).mean(),
                          "reachable_within_x4.5": mg.min_demand_increase_pct.notna().mean(),
                          "median_increase_pct_if_reachable": mg.min_demand_increase_pct.median()}])
    msum.to_csv(OUT / "decision_margins_summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.float_format", lambda x: f"{x:,.4f}"):
        print("\nFIDELITY\n", pd.DataFrame(checks).groupby("check").value.max().to_string())
        print("\nSTABILITY\n", pd.DataFrame(stab).to_string(index=False))
        if len(sw):
            print("\nSWAPS\n", s.sort_values("lgbm_share", ascending=False).round(3).to_string())
            print("pairs:", len(pairs), "| LightGBM prefers added in", round((pairs.lgbm > 0).mean(), 3), "| PPML prefers added in", round((pairs.ppml > 0).mean(), 3))
        print("\nTHEORY CONSISTENCY (LightGBM, hybrid Top-10)\n", pd.concat([cs, overall]).round(3).to_string())
        print("\nPPML coefficient sign consistency by feature\n", pd.DataFrame(coef_rows).groupby("feature").consistent.mean().round(2).to_string())
        print("\nFAITHFULNESS\n", fsum.round(3).to_string())
        print("\nDECISION MARGINS\n", msum.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
