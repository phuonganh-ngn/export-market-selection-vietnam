"""Deletion test at the level of the hybrid decision (discovery, 2018-2024).

For every destination in the hybrid Top-10, the feature group with the largest positive combined attribution is
replaced by its pool median, or by the values of the most similar real candidate, BOTH component scores are
recomputed (LightGBM probability and PPML log-mean), the hybrid rank is recomputed within the pool, and we record
whether the destination leaves the hybrid Top-10. A randomly chosen other group is the control.

Combined attribution of a group: the average of (i) the LightGBM SHAP value of the group divided by the pool
standard deviation of the LightGBM raw score and (ii) the PPML log-mean contribution of the group, measured from the
pool median, divided by the pool standard deviation of the PPML log-mean. Equal weights mirror the equal-weight
rank fusion. Export history is not perturbed because it is constant among discovery candidates."""
from __future__ import annotations
import time, warnings
import joblib, numpy as np, pandas as pd, shap
from revision_lib import KEY, PRED_DIR, RES_DIR, TA, load_stored_candidates, zfill_hs2
from experiments.explainable_reliability_gate_v1.src.task_audit import load_and_freeze
from temporal_validation import FEATURE_GROUPS
from xai_decision import (NEUTRALISABLE, PPML_DIR, SEEDS, YEARS, column_meta, dense, donor_values, hybrid_rank, lgbm_prob,
                          load_lgbm, ppml_design, ppml_logmean)
warnings.filterwarnings("ignore")
OUT = RES_DIR / "xai"
t0 = time.time()
panel, _ = load_and_freeze(write=False); panel = zfill_hs2(panel)
stored = load_stored_candidates(); stored = stored[stored.task == TA]
refit = zfill_hs2(pd.read_parquet(PRED_DIR / "lgbm_refit_scores.parquet"))
stored = stored.drop(columns="lightgbm_score").merge(refit[KEY + ["lightgbm_score"]], on=KEY, validate="one_to_one")
feats = [c for c in panel.columns if c not in stored.columns or c in ("destination_iso3", "hs2", "outcome_year")]
rng = np.random.default_rng(20261116)
rows = []
for year in YEARS:
    cand = stored[stored.outcome_year == year].merge(panel[feats], on=["destination_iso3", "hs2", "outcome_year"], how="left", validate="one_to_one").reset_index(drop=True)
    models = load_lgbm(year)
    o = joblib.load(PPML_DIR / f"ppml_{year}.joblib"); ppml_model, ppml_obj = o["model"], o["objects"]
    cand["ppml_lm"] = ppml_logmean(ppml_model, ppml_obj, cand)
    cand["lgb_raw"] = np.log(np.clip(cand.lightgbm_score, 1e-12, 1 - 1e-12) / (1 - np.clip(cand.lightgbm_score, 1e-12, 1 - 1e-12)))
    cand["hyb_rank"] = 0
    for _, g in cand.groupby("hs2"):
        cand.loc[g.index, "hyb_rank"] = hybrid_rank(g.ppml_lm.to_numpy(), g.lightgbm_score.to_numpy(), g.destination_iso3.to_numpy())
    expl = cand[cand.hyb_rank <= 10]
    bg = cand.sample(min(100, len(cand)), random_state=year)
    shap_sum, meta = None, None
    for m in models:
        cols = m.numeric_features + m.categorical_features
        xe = dense(m.preprocessor.transform(expl[cols])).astype(np.float32).astype(np.float64)
        xb = dense(m.preprocessor.transform(bg[cols])).astype(np.float32).astype(np.float64)
        sv = shap.TreeExplainer(m.estimator, data=xb, feature_perturbation="interventional", model_output="raw").shap_values(xe, check_additivity=False)
        sv = sv[1] if isinstance(sv, list) else sv
        if sv.ndim == 3: sv = sv[:, :, 1]
        if meta is None: meta = column_meta(m.preprocessor.get_feature_names_out())
        shap_sum = sv if shap_sum is None else shap_sum + sv
    g_shap = pd.DataFrame(shap_sum / len(models), columns=meta.column, index=expl.index).T.groupby(meta.group.to_numpy()).sum().T
    xp, pnames = ppml_design(ppml_obj, cand); pmeta = column_meta(pnames)
    p_group = pd.DataFrame(xp * ppml_model.coef_, columns=pmeta.column, index=cand.index).T.groupby(pmeta.group.to_numpy()).sum().T
    groups = [g for g in NEUTRALISABLE if g in g_shap.columns and g in p_group.columns]
    for hs2, pool in cand.groupby("hs2"):
        iso = pool.destination_iso3.to_numpy(); pos = {i: k for k, i in enumerate(pool.index)}
        sd_l, sd_p = pool.lgb_raw.std(ddof=0) or 1.0, pool.ppml_lm.std(ddof=0) or 1.0
        p_dev = p_group.loc[pool.index, groups] - p_group.loc[pool.index, groups].median()
        base_l, base_p = pool.lightgbm_score.to_numpy(), pool.ppml_lm.to_numpy()
        for idx in pool.index[pool.hyb_rank <= 10]:
            comb = 0.5 * g_shap.loc[idx, groups] / sd_l + 0.5 * p_dev.loc[idx] / sd_p
            if (comb <= 0).all():
                continue
            main = comb.idxmax(); ctrl = rng.choice([g for g in groups if g != main])
            for kind, grp, how in (("attributed", main, "median"), ("random", ctrl, "median"), ("attributed_donor", main, "donor"), ("random_donor", ctrl, "donor")):
                mod = cand.loc[[idx]].copy(); fs = [f for f in FEATURE_GROUPS[grp] if f in mod.columns]
                if how == "median":
                    for f in fs: mod[f] = pool[f].median()
                else:
                    mod[fs] = donor_values(pool, idx, grp).to_numpy()
                ls, ps = base_l.copy(), base_p.copy()
                ls[pos[idx]], ps[pos[idx]] = lgbm_prob(models, mod)[0], ppml_logmean(ppml_model, ppml_obj, mod)[0]
                # component ranks after the perturbation, for the record
                hr = hybrid_rank(ps, ls, iso)[pos[idx]]
                only_l = hybrid_rank(base_p, ls, iso)[pos[idx]]; only_p = hybrid_rank(ps, base_l, iso)[pos[idx]]
                rows.append({"year": year, "hs2": hs2, "destination": cand.at[idx, "destination_iso3"], "kind": kind, "group": grp,
                             "hyb_rank_before": int(cand.at[idx, "hyb_rank"]), "hyb_rank_after": int(hr), "left_top10": bool(hr > 10),
                             "left_if_only_lgbm_changes": bool(only_l > 10), "left_if_only_ppml_changes": bool(only_p > 10),
                             "success": cand.at[idx, "entry_success"]})
    print(f"{year}: {len(expl)} hybrid Top-10 rows | {time.time() - t0:.0f}s", flush=True)
fa = pd.DataFrame(rows); fa.to_csv(OUT / "hybrid_faithfulness_rows.csv", index=False)
s = fa.groupby("kind").agg(n=("left_top10", "size"), share_left_top10=("left_top10", "mean"), mean_rank_before=("hyb_rank_before", "mean"),
                           mean_rank_after=("hyb_rank_after", "mean"), only_lgbm=("left_if_only_lgbm_changes", "mean"), only_ppml=("left_if_only_ppml_changes", "mean"))
s.to_csv(OUT / "hybrid_faithfulness_summary.csv"); print(s.round(4).to_string())
print(fa[fa.kind == "attributed"].group.value_counts(normalize=True).round(3).to_string())
by = fa.pivot_table(index="year", columns="kind", values="left_top10", aggfunc="mean"); print(by.round(3).to_string())
# HS2-block bootstrap for the difference attributed - random (pool median and donor)
r = np.random.default_rng(1)
for a, b in (("attributed", "random"), ("attributed_donor", "random_donor")):
    w = fa[fa.kind.isin([a, b])].pivot_table(index=["year", "hs2", "destination"], columns="kind", values="left_top10").reset_index()
    d = (w[a].astype(float) - w[b].astype(float)).to_numpy(); hs = w.hs2.to_numpy(); u = np.unique(hs); ix = {h: np.where(hs == h)[0] for h in u}
    bs = [d[np.concatenate([ix[h] for h in r.choice(u, len(u))])].mean() for _ in range(5000)]
    print(a, "-", b, round(d.mean(), 4), np.round(np.percentile(bs, [2.5, 97.5]), 4))
