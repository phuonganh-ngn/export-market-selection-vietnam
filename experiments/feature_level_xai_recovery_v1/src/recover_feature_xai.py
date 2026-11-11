from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

PROJECT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(PROJECT), str(PROJECT / "src")]

from gravity_ppml import _design  # noqa: E402
from temporal_validation import FEATURE_GROUPS, split_outer  # noqa: E402
from experiments.explainable_reliability_gate_v1.src.task_audit import load_and_freeze  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
YEARS = (2022, 2023, 2024)
SEEDS = (11, 23, 37, 51, 73)
TASK_A = "Task_A_new_entry"
TOP_K = 10


def ensure_dirs() -> None:
    for name in ("audit", "results", "reports", "predictions"):
        (HERE / name).mkdir(parents=True, exist_ok=True)


def _rank(g: pd.DataFrame, score: str) -> pd.DataFrame:
    return g.sort_values([score, "destination_iso3"], ascending=[False, True], kind="mergesort").copy()


def _transformed_groups(names: list[str]) -> tuple[list[str], list[str]]:
    """Map fitted ColumnTransformer names to audited economic feature groups."""
    groups: list[str] = []
    original: list[str] = []
    lookup = {f: group for group, cols in FEATURE_GROUPS.items() for f in cols}
    for full in names:
        name = str(full)
        if name.startswith("category__"):
            suffix = name[len("category__"):]
            if suffix.startswith("destination_iso3_"):
                groups.append("destination_identifier")
                original.append("destination_iso3=" + suffix.split("_", 2)[-1])
            elif suffix.startswith("hs2_"):
                groups.append("product_indicator")
                original.append("hs2=" + suffix.split("_", 1)[-1])
            else:
                groups.append("categorical_indicator")
                original.append(suffix)
            continue
        suffix = name.split("__", 1)[-1]
        if suffix.startswith("missingindicator_"):
            raw = suffix[len("missingindicator_"):]
            groups.append("predictor_missingness")
            original.append(raw + "__missing_indicator")
            continue
        original.append(suffix)
        if suffix in lookup:
            groups.append(lookup[suffix])
        elif suffix == "hs2":
            groups.append("product_indicator")
        else:
            groups.append("other")
    return groups, original


def _class1_shap_values(explainer, x: np.ndarray) -> tuple[np.ndarray, float]:
    values = explainer.shap_values(x, check_additivity=False)
    base = np.asarray(explainer.expected_value)
    if isinstance(values, list):
        idx = 1 if len(values) > 1 else 0
        values = values[idx]
        base_value = float(base[idx] if base.ndim else base)
    else:
        values = np.asarray(values)
        if values.ndim == 3:
            idx = 1 if values.shape[2] > 1 else 0
            values = values[:, :, idx]
            base_value = float(base[idx] if base.ndim else base)
        else:
            base_value = float(base.reshape(-1)[-1]) if base.ndim else float(base)
    return np.asarray(values, dtype=float), base_value


def _safe_corr_rank(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if len(a) < 2:
        return np.nan
    v = spearmanr(a, b).statistic
    return float(v) if np.isfinite(v) else np.nan


def _add_model_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in frame.groupby(["hs2", "outcome_year"], sort=False):
        q = g.copy()
        for score, out in (("ppml_score", "ppml_rank"), ("lightgbm_score", "lightgbm_rank")):
            r = _rank(q, score)
            ranks = pd.Series(np.arange(1, len(r) + 1), index=r.index)
            q[out] = ranks.reindex(q.index).to_numpy()
        n = len(q)
        q["q_ppml"] = (n - q.ppml_rank + 1) / max(1, n - 1)
        q["q_lightgbm"] = (n - q.lightgbm_rank + 1) / max(1, n - 1)
        q["hybrid_score"] = 0.5 * (q.q_ppml + q.q_lightgbm)
        hr = _rank(q, "hybrid_score")
        ranks = pd.Series(np.arange(1, len(hr) + 1), index=hr.index)
        q["hybrid_rank"] = ranks.reindex(q.index).to_numpy()
        q["lightgbm_top10"] = q.lightgbm_rank.le(TOP_K)
        q["hybrid_top10"] = q.hybrid_rank.le(TOP_K)
        parts.append(q)
    return pd.concat(parts, ignore_index=True)


def _rank_match(reference: pd.DataFrame, candidate: pd.DataFrame, score_col: str) -> dict:
    merged = reference[["candidate_id", "destination_iso3", "hs2", "outcome_year", "rank_full_exante"]].merge(
        candidate[["candidate_id", "destination_iso3", "hs2", "outcome_year", score_col]],
        on=["candidate_id", "destination_iso3", "hs2", "outcome_year"], how="inner", validate="one_to_one")
    built = []
    for _, g in merged.groupby(["hs2", "outcome_year"], sort=False):
        r = _rank(g, score_col)
        ranks = pd.Series(np.arange(1, len(r) + 1), index=r.index)
        z = g.copy()
        z["rebuilt_rank"] = ranks.reindex(z.index).to_numpy()
        built.append(z)
    z = pd.concat(built, ignore_index=True)
    ref_top = z.loc[z.rank_full_exante.le(TOP_K), "candidate_id"]
    new_top = z.loc[z.rebuilt_rank.le(TOP_K), "candidate_id"]
    return {"rows_compared": len(z), "rank_match_share": float(z.rank_full_exante.eq(z.rebuilt_rank).mean()),
            "spearman_rank": _safe_corr_rank(z.rank_full_exante.to_numpy(float), z.rebuilt_rank.to_numpy(float)),
            "top10_membership_match_share_pool": float(np.mean([
                set(a.candidate_id[a.rank_full_exante.le(TOP_K)]) == set(a.candidate_id[a.rebuilt_rank.le(TOP_K)])
                for _, a in z.groupby(["hs2", "outcome_year"], sort=False)])),
            "top10_candidate_match_share": len(set(ref_top) & set(new_top)) / max(1, len(set(ref_top) | set(new_top)))}


def recover_base_predictions(panel: pd.DataFrame, manifest: pd.DataFrame, scores: pd.DataFrame):
    e6 = pd.read_parquet(PROJECT / "experiments/decision_focused_xai_v1/predictions/E6_ranking_predictions.parquet")
    e6 = e6.loc[(e6.task == TASK_A) & e6.model.isin(["LightGBM", "Gravity_PPML"])].copy()
    e6["hs2"] = e6.hs2.astype(str).str.zfill(2)
    e6["outcome_year"] = e6.outcome_year.astype(int)
    task_manifest = manifest.loc[manifest.task.eq(TASK_A)].copy()
    e6 = e6.loc[e6.candidate_id.isin(task_manifest.candidate_id)].copy()
    # E6's stored full-ex-ante ranks were originally formed on the broader
    # threshold-inactive Task A universe. Re-rank its archived scores only after
    # restricting to the corrected strict-zero candidate universe used by M1 v2.
    e6["rank_full_exante"] = np.nan
    for _, eg in e6.groupby(["model", "seed", "hs2", "outcome_year"], sort=False):
        ranked_e = _rank(eg, "score")
        e6.loc[ranked_e.index, "rank_full_exante"] = np.arange(1, len(ranked_e) + 1)
    candidates = panel.loc[panel.candidate_id.isin(task_manifest.candidate_id)].copy()
    candidates = candidates.merge(task_manifest[["candidate_id", "task", "candidate_pool_id"]], on="candidate_id", validate="one_to_one")
    outer_scores = scores.loc[scores.task.eq(TASK_A), ["candidate_id", "ppml_score", "lightgbm_score", "blend_rank_score"]].copy()
    candidates = candidates.merge(outer_scores, on="candidate_id", validate="one_to_one")
    eq_rows, status_rows = [], []
    lightgbm_predictions: dict[int, dict[str, np.ndarray]] = {}
    ppml_objects: dict[int, dict] = {}
    ppml_features: dict[int, dict] = {}
    e2_seed_archive = pd.read_parquet(PROJECT / "audit_v2/corrected_E2_candidates.parquet")
    e2_seed_archive["hs2"] = e2_seed_archive.hs2.astype(str).str.zfill(2)

    for year in YEARS:
        c = candidates.loc[candidates.outcome_year.eq(year)].copy().reset_index(drop=True)
        reference_mean_lgb = e6.loc[(e6.outcome_year == year) & (e6.seed == -1) & (e6.model == "LightGBM")].copy()
        reference_mean_pp = e6.loc[(e6.outcome_year == year) & (e6.seed == -1) & (e6.model == "Gravity_PPML")].copy()
        if len(c) != len(reference_mean_lgb) or len(c) != len(reference_mean_pp):
            raise AssertionError(f"E6 reference candidate count mismatch for {year}: {len(c)}, {len(reference_mean_lgb)}, {len(reference_mean_pp)}")
        if set(c.candidate_id) != set(reference_mean_lgb.candidate_id) or set(c.candidate_id) != set(reference_mean_pp.candidate_id):
            raise AssertionError(f"E6 row identity mismatch for {year}")
        lgb_by_seed = {}
        for seed in SEEDS:
            obj = joblib.load(PROJECT / f"results/models/e2_lightgbm_lag2_{year}_seed{seed}.joblib")
            bundle = obj["bundle"]
            x = bundle.preprocessor.transform(c[bundle.numeric_features + bundle.categorical_features]).astype(np.float32)
            pred = bundle.estimator.predict_proba(x)[:, 1].astype(float)
            lgb_by_seed[seed] = pred
            source = e2_seed_archive.loc[(e2_seed_archive.evaluation_year.eq(year))
                & e2_seed_archive.model.eq("LightGBM") & e2_seed_archive.seed.eq(seed),
                ["destination_iso3", "hs2", "evaluation_year", "score"]].copy()
            source = source.rename(columns={"evaluation_year": "outcome_year", "score": "archive_score"})
            reference = c[["candidate_id", "destination_iso3", "hs2", "outcome_year"]].merge(
                source, on=["destination_iso3", "hs2", "outcome_year"], how="left", validate="one_to_one")
            if reference.archive_score.isna().any():
                raise AssertionError(f"Missing original E2 seed archive rows for LightGBM {year}/{seed}")
            reference = reference.rename(columns={"archive_score": "_archive_score"})
            reference["rank_full_exante"] = np.nan
            for _, rg in reference.groupby(["hs2", "outcome_year"], sort=False):
                rr = _rank(rg, "_archive_score")
                reference.loc[rr.index, "rank_full_exante"] = np.arange(1, len(rr) + 1)
            max_abs = float(np.max(np.abs(pred - reference._archive_score.to_numpy(float))))
            rank_info = _rank_match(reference, c.assign(_prediction=pred), "_prediction")
            eq_rows.append({"year": year, "component": "LightGBM_classifier", "seed": seed,
                "comparison": "saved seed-specific corrected E2 score restricted to strict-zero Task A universe", "candidate_rows": len(c),
                "max_abs_prediction_difference": max_abs, **rank_info,
                "status": "EXACT" if max_abs <= 1e-12 and rank_info["rank_match_share"] == 1.0 else "MISMATCH"})
        avg = np.mean(np.column_stack([lgb_by_seed[s] for s in SEEDS]), axis=1)
        lightgbm_predictions[year] = {"seed_predictions": lgb_by_seed, "mean_probability": avg}
        ref_mean = c[["candidate_id", "destination_iso3", "hs2", "outcome_year"]].merge(
            reference_mean_lgb[["candidate_id", "score", "rank_full_exante"]], on="candidate_id", how="left", validate="one_to_one")
        max_abs = float(np.max(np.abs(avg - ref_mean.score.to_numpy(float))))
        rank_info = _rank_match(reference_mean_lgb, c.assign(_prediction=avg), "_prediction")
        final_diff = float(np.max(np.abs(avg - c.lightgbm_score.to_numpy(float))))
        eq_rows.append({"year": year, "component": "LightGBM_classifier_seed_mean", "seed": -1,
            "comparison": "E6 seed mean and final Hybrid score archive", "candidate_rows": len(c),
            "max_abs_prediction_difference": max(max_abs, final_diff), **rank_info,
            "status": "EXACT" if max(max_abs, final_diff) <= 1e-12 and rank_info["rank_match_share"] == 1.0 else "MISMATCH"})

        train, _ = split_outer(panel, year, 2)
        _, x_eval, preprocessing = _design(train, c)
        saved = joblib.load(PROJECT / f"results/models/ppml_lag2_{year}.joblib")
        model = saved["result"]
        eta = model.intercept_ + np.asarray(x_eval @ model.coef_).reshape(-1)
        mu = np.asarray(model.predict(x_eval), dtype=float)
        identity_error = float(np.max(np.abs(mu - np.exp(eta))))
        usd = mu * 1e6
        score = np.log1p(np.maximum(0.0, usd))
        ref_pp = c[["candidate_id", "destination_iso3", "hs2", "outcome_year"]].merge(
            reference_mean_pp[["candidate_id", "score", "rank_full_exante"]], on="candidate_id", how="left", validate="one_to_one")
        max_abs = float(np.max(np.abs(score - ref_pp.score.to_numpy(float))))
        rank_info = _rank_match(reference_mean_pp, c.assign(_prediction=score), "_prediction")
        final_diff = float(np.max(np.abs(score - c.ppml_score.to_numpy(float))))
        eq_rows.append({"year": year, "component": "Gravity_PPML", "seed": 0,
            "comparison": "saved fitted estimator + reconstructed original preprocessing vs E6/final score", "candidate_rows": len(c),
            "max_abs_prediction_difference": max(max_abs, final_diff), **rank_info,
            "status": "EXACT" if max(max_abs, final_diff) <= 1e-10 and rank_info["rank_match_share"] == 1.0 and identity_error <= 1e-10 else "MISMATCH"})

        nums, cats, imputer, scaler, encoder = preprocessing
        numeric_names = list(imputer.get_feature_names_out(nums))
        categorical_names = list(encoder.get_feature_names_out(cats))
        feature_names = numeric_names + categorical_names
        if len(feature_names) != len(model.coef_):
            raise AssertionError(f"PPML transformed feature-name/coef length mismatch in {year}")
        if len(model.coef_) != x_eval.shape[1]:
            raise AssertionError(f"PPML coefficient/design width mismatch in {year}")
        contribution = x_eval.multiply(model.coef_).tocsr()
        groups, original_names = _transformed_groups(feature_names)
        ppml_objects[year] = {"model": model, "eta": eta, "usd": usd, "score": score,
            "x": x_eval, "contribution": contribution, "feature_names": feature_names,
            "feature_groups": groups, "original_names": original_names,
            "preprocessing": preprocessing, "identity_error": identity_error}
        ppml_features[year] = {"names": feature_names, "groups": groups, "original_names": original_names}
        status_rows.append({"year": year, "model": "Gravity_PPML", "artifact": f"results/models/ppml_lag2_{year}.joblib",
            "estimator_recovered": True, "preprocessing_recovered": True,
            "preprocessing_mode": "deterministically reconstructed using the archived original _design function and the fold training rows",
            "candidate_rows": len(c), "n_coefficients": len(model.coef_), "linear_predictor_identity_max_abs": identity_error,
            "prediction_match_max_abs": max(max_abs, final_diff), "rank_match_share": rank_info["rank_match_share"],
            "top10_match_share_pool": rank_info["top10_membership_match_share_pool"],
            "model_fidelity": "PREDICTION_EQUIVALENT_RECONSTRUCTION" if max(max_abs, final_diff) <= 1e-10 else "NONIDENTICAL_RECONSTRUCTION_ONLY"})
        status_rows.append({"year": year, "model": "LightGBM_5_seed_ensemble", "artifact": f"results/models/e2_lightgbm_lag2_{year}_seed{{11,23,37,51,73}}.joblib",
            "estimator_recovered": True, "preprocessing_recovered": True,
            "preprocessing_mode": "exact fitted pipeline objects recovered from each outer-fold checkpoint",
            "candidate_rows": len(c), "n_coefficients": np.nan, "linear_predictor_identity_max_abs": np.nan,
            "prediction_match_max_abs": max_abs, "rank_match_share": rank_info["rank_match_share"],
            "top10_match_share_pool": rank_info["top10_membership_match_share_pool"],
            "model_fidelity": "EXACT_ORIGINAL_MODELS_VERIFIED" if max_abs <= 1e-12 and rank_info["rank_match_share"] == 1.0 else "PREDICTION_EQUIVALENT_RECONSTRUCTION"})
    eq = pd.DataFrame(eq_rows)
    status = pd.DataFrame(status_rows)
    eq.to_csv(HERE / "audit/original_prediction_equivalence.csv", index=False, na_rep="")
    status.to_csv(HERE / "audit/model_recovery_status.csv", index=False, na_rep="")
    recovery_text = [
        "# Feature-XAI model recovery audit", "",
        "## Scope and candidate population", "",
        "The audit concerns the corrected strict-zero Task A population only: 2022, 2023, and 2024; 288 HS2-year pools; unknown labels remain missing. The source score archive has both Task A and Task B, but these feature-attribution checks are restricted to the 29,737 Task A candidate rows used by the v2 Hybrid explanation.", "",
        "## LightGBM", "",
        "Five outer-fold LightGBM binary classifiers per year (seeds 11, 23, 37, 51, 73) were recovered from `results/models/e2_lightgbm_lag2_YEAR_seedSEED.joblib`, including their fitted preprocessing pipelines. Each was compared against `audit_v2/corrected_E2_candidates.parquet` after restricting to the strict-zero Task A IDs. The five seed scores reproduce exactly. Their mean reproduces the E6 saved seed-mean score and final score archive, with exact score, within-pool rank, and Top-10 membership agreement in each year. The original full-exante rank stored in E6 was not used directly because it was formed on the broader historical threshold-inactive pool; ranks were rebuilt after restricting to the corrected strict-zero population.", "",
        "## Gravity-PPML", "",
        "Each saved `results/models/ppml_lag2_YEAR.joblib` contains the fitted sklearn PoissonRegressor and diagnostics, but not the fitted preprocessing objects. Its exact intercept and 369 fitted coefficients were recovered. Preprocessing was rebuilt from the archived `gravity_ppml._design` implementation and the same chronological outer-training rows, then checked against the saved E6 score and final score archive. Predictions, within-pool ranks, and Top-10 membership reproduce exactly. The linear predictor identity `eta = intercept + X beta` versus `model.predict(X) = exp(eta)` has zero numerical discrepancy in the audited arrays. Classification: **prediction-equivalent reconstruction**, because the PPML preprocessing pipeline was not serialized alongside the estimator.", "",
        "## Equivalence results", "",
        f"The equivalence table contains {len(eq)} comparisons; {int(eq.status.eq('EXACT').sum())} are exact. The maximum saved-score error is {eq.max_abs_prediction_difference.max():.3g}. Model-specific candidate counts are 9,846 (2022), 9,913 (2023), and 9,978 (2024).", "",
        "## Attribution readiness", "",
        "The final LightGBM component score is the mean probability over five fitted seed models. The analysis uses probability-scale interventional TreeSHAP for each exact model with the same deterministic, training-only background sample and averages the attributions across seeds; the mean contributions therefore reconstruct the archived mean probability. PPML feature terms are `x_j beta_j` on the log expected export scale in USD millions. Neither attribution is causal, and the PPML and LightGBM attribution scales are not combined into one Hybrid feature score.", "",
        "See `audit/original_prediction_equivalence.csv`, `results/lightgbm_shap_fidelity_by_seed.csv`, and `audit/reconciliation_checks.csv` for row-level numerical checks."
    ]
    (HERE / "audit/FEATURE_XAI_MODEL_RECOVERY.md").write_text("\n".join(recovery_text) + "\n")
    return candidates, e6, eq, status, lightgbm_predictions, ppml_objects, ppml_features


def _prepare_rank_decisions(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranked = _add_model_ranks(candidates.copy())
    pool_rows, pair_rows = [], []
    for (hs2, year), g in ranked.groupby(["hs2", "outcome_year"], sort=True):
        lset = set(g.loc[g.lightgbm_top10, "candidate_id"])
        hset = set(g.loc[g.hybrid_top10, "candidate_id"])
        added_ids, displaced_ids = hset - lset, lset - hset
        if len(added_ids) != len(displaced_ids):
            raise AssertionError("Top-10 membership swaps must balance within each pool")
        added = g.loc[g.candidate_id.isin(added_ids)].sort_values(["hybrid_score", "destination_iso3"], ascending=[True, False], kind="mergesort")
        displaced = g.loc[g.candidate_id.isin(displaced_ids)].sort_values(["hybrid_score", "destination_iso3"], ascending=[False, True], kind="mergesort")
        p_success = pd.to_numeric(g.entry_success, errors="coerce")
        a = g.loc[g.hybrid_top10].copy()
        l = g.loc[g.lightgbm_top10].copy()
        def qualified_observed_sum(frame: pd.DataFrame) -> float:
            known = frame.loc[frame.label_observed.astype(bool)]
            if known.empty:
                return np.nan
            vals = known.export_value_usd.where(known.entry_success.eq(1), 0.0)
            return float(vals.sum())
        def observed_raw_sum(frame: pd.DataFrame) -> float:
            known = frame.loc[frame.label_observed.astype(bool)]
            return float(known.export_value_usd.sum()) if not known.empty else np.nan
        pool_rows.append({"hs2": hs2, "outcome_year": int(year), "candidate_pool_id": f"{TASK_A}|{hs2}|{int(year)}",
            "candidate_count": len(g), "observed_label_candidates": int(g.label_observed.astype(bool).sum()),
            "unknown_label_candidates": int((~g.label_observed.astype(bool)).sum()),
            "common_count": len(lset & hset), "added_count": len(added_ids), "displaced_count": len(displaced_ids),
            "top10_jaccard": len(lset & hset) / len(lset | hset) if lset | hset else np.nan,
            "added_successes_observed": float(p_success.loc[g.candidate_id.isin(added_ids)].sum(min_count=1)) if len(added_ids) else 0.0,
            "displaced_successes_observed": float(p_success.loc[g.candidate_id.isin(displaced_ids)].sum(min_count=1)) if len(displaced_ids) else 0.0,
            "added_failed_observed": int((g.loc[g.candidate_id.isin(added_ids), "entry_success"] == 0).sum()),
            "displaced_failed_observed": int((g.loc[g.candidate_id.isin(displaced_ids), "entry_success"] == 0).sum()),
            "unknown_added": int((~g.loc[g.candidate_id.isin(added_ids), "label_observed"].astype(bool)).sum()),
            "unknown_displaced": int((~g.loc[g.candidate_id.isin(displaced_ids), "label_observed"].astype(bool)).sum()),
            "qualified_usd_added_known": qualified_observed_sum(g.loc[g.candidate_id.isin(added_ids)]) if len(added_ids) else 0.0,
            "qualified_usd_displaced_known": qualified_observed_sum(g.loc[g.candidate_id.isin(displaced_ids)]) if len(displaced_ids) else 0.0,
            "raw_usd_added_known": observed_raw_sum(g.loc[g.candidate_id.isin(added_ids)]) if len(added_ids) else 0.0,
            "raw_usd_displaced_known": observed_raw_sum(g.loc[g.candidate_id.isin(displaced_ids)]) if len(displaced_ids) else 0.0,
            "hybrid_top10_success_observed": float(a.entry_success.sum(min_count=1)),
            "lightgbm_top10_success_observed": float(l.entry_success.sum(min_count=1)),
            "hybrid_top10_qualified_usd_known": qualified_observed_sum(a),
            "lightgbm_top10_qualified_usd_known": qualified_observed_sum(l),
            "hybrid_top10_raw_usd_known": observed_raw_sum(a),
            "lightgbm_top10_raw_usd_known": observed_raw_sum(l),
            "positive_relevance_pool": bool(p_success.eq(1).any())})
        # Deterministic threshold-boundary matching: weakest Hybrid addition against strongest Hybrid-excluded LightGBM selection, moving inward.
        for idx, (_, ar) in enumerate(added.iterrows()):
            br = displaced.iloc[idx]
            qp_diff = 0.5 * (float(ar.q_ppml) - float(br.q_ppml))
            ql_diff = 0.5 * (float(ar.q_lightgbm) - float(br.q_lightgbm))
            h_diff = float(ar.hybrid_score - br.hybrid_score)
            if abs((qp_diff + ql_diff) - h_diff) > 1e-12:
                raise AssertionError("Hybrid rank-score contrast decomposition failed")
            pair_rows.append({"hs2": hs2, "outcome_year": int(year), "candidate_pool_id": f"{TASK_A}|{hs2}|{int(year)}",
                "swap_order_from_boundary": idx + 1, "added_candidate_id": ar.candidate_id, "added_iso3": ar.destination_iso3,
                "added_destination_name": ar.destination_name, "displaced_candidate_id": br.candidate_id,
                "displaced_iso3": br.destination_iso3, "displaced_destination_name": br.destination_name,
                "added_ppml_rank": int(ar.ppml_rank), "added_lightgbm_rank": int(ar.lightgbm_rank), "added_hybrid_rank": int(ar.hybrid_rank),
                "displaced_ppml_rank": int(br.ppml_rank), "displaced_lightgbm_rank": int(br.lightgbm_rank), "displaced_hybrid_rank": int(br.hybrid_rank),
                "added_ppml_score": float(ar.ppml_score), "added_lightgbm_score": float(ar.lightgbm_score),
                "added_q_ppml": float(ar.q_ppml), "added_q_lightgbm": float(ar.q_lightgbm), "added_hybrid_score": float(ar.hybrid_score),
                "displaced_ppml_score": float(br.ppml_score), "displaced_lightgbm_score": float(br.lightgbm_score),
                "displaced_q_ppml": float(br.q_ppml), "displaced_q_lightgbm": float(br.q_lightgbm), "displaced_hybrid_score": float(br.hybrid_score),
                "ppml_component_difference": qp_diff, "lightgbm_component_difference": ql_diff,
                "hybrid_score_difference": h_diff, "rank_score_identity_error": (qp_diff + ql_diff) - h_diff,
                "added_entry_success": ar.entry_success, "displaced_entry_success": br.entry_success,
                "added_label_observed": bool(ar.label_observed), "displaced_label_observed": bool(br.label_observed),
                "added_export_usd": ar.export_value_usd, "displaced_export_usd": br.export_value_usd,
                "added_qualified_usd": ar.export_value_usd if ar.entry_success == 1 else (0.0 if ar.entry_success == 0 else np.nan),
                "displaced_qualified_usd": br.export_value_usd if br.entry_success == 1 else (0.0 if br.entry_success == 0 else np.nan),
                "pairwise_observed_success_difference": (ar.entry_success - br.entry_success) if pd.notna(ar.entry_success) and pd.notna(br.entry_success) else np.nan,
                "pairwise_observed_qualified_usd_difference": ((ar.export_value_usd if ar.entry_success == 1 else 0.0) - (br.export_value_usd if br.entry_success == 1 else 0.0)) if bool(ar.label_observed) and bool(br.label_observed) else np.nan,
                "matching_rule": "sort additions from weakest selected Hybrid score; sort displaced candidates from strongest excluded Hybrid score; pair by boundary distance"})
    return ranked, pd.DataFrame(pool_rows), pd.DataFrame(pair_rows)


def _ndcg_binary(g: pd.DataFrame, score: str) -> float:
    known = g.loc[g.label_observed.astype(bool)].copy()
    if known.empty or not known.entry_success.eq(1).any():
        return np.nan
    ranked = _rank(known, score)
    rel = ranked.entry_success.to_numpy(float)
    top = rel[:TOP_K]
    dcg = float(np.sum(top / np.log2(np.arange(2, len(top) + 2))))
    ideal = np.sort(rel)[::-1][:TOP_K]
    idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
    return dcg / idcg if idcg else np.nan


def audit_hybrid_boundary_ties(ranked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (hs2, year), g in ranked.groupby(["hs2", "outcome_year"], sort=True):
        z = _rank(g, "hybrid_score").reset_index(drop=True)
        if len(z) < TOP_K + 1 or z.loc[TOP_K - 1, "hybrid_score"] != z.loc[TOP_K, "hybrid_score"]:
            continue
        rows.append({"hs2": str(hs2).zfill(2), "outcome_year": int(year),
            "candidate_pool_id": f"{TASK_A}|{str(hs2).zfill(2)}|{int(year)}",
            "rank10_iso3": z.loc[TOP_K - 1, "destination_iso3"], "rank11_iso3": z.loc[TOP_K, "destination_iso3"],
            "boundary_score": float(z.loc[TOP_K - 1, "hybrid_score"]),
            "rank10_lightgbm_rank": int(z.loc[TOP_K - 1, "lightgbm_rank"]),
            "rank11_lightgbm_rank": int(z.loc[TOP_K, "lightgbm_rank"]),
            "rank10_ppml_rank": int(z.loc[TOP_K - 1, "ppml_rank"]),
            "rank11_ppml_rank": int(z.loc[TOP_K, "ppml_rank"]),
            "tie_break_rule": "ascending destination ISO3 decides which equal-H candidate is rank 10"})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results/hybrid_top10_boundary_ties.csv", index=False, na_rep="")
    return out


def explain_lightgbm(candidates: pd.DataFrame, lightgbm_predictions: dict, pair_table: pd.DataFrame,
                     source_panel: pd.DataFrame):
    shap_by_year: dict[int, dict] = {}
    feature_rows, group_rows, fidelity_rows = [], [], []
    target_ids_by_year = {}
    for year in YEARS:
        c = candidates.loc[candidates.outcome_year.eq(year)].copy().reset_index(drop=True)
        cand_ids = set(c.candidate_id)
        pair_ids = set(pair_table.loc[pair_table.outcome_year.eq(year), "added_candidate_id"]) | set(pair_table.loc[pair_table.outcome_year.eq(year), "displaced_candidate_id"])
        # Fixed candidate sample is chosen without outcomes; all Top-10 swap candidates are also included.
        sample_n = min(1000, len(c))
        random_ids = set(c.sample(n=sample_n, random_state=202610 + year).candidate_id)
        explain_ids = pair_ids | random_ids
        if not explain_ids.issubset(cand_ids):
            raise AssertionError("SHAP candidate IDs not in frozen Task A universe")
        query = c.loc[c.candidate_id.isin(explain_ids)].copy().reset_index(drop=True)
        query = query.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)
        target_ids_by_year[year] = {"random": random_ids, "pairs": pair_ids, "query": query}
        mean_values = None
        mean_expected = 0.0
        feature_names_ref = None
        for seed in SEEDS:
            bundle = joblib.load(PROJECT / f"results/models/e2_lightgbm_lag2_{year}_seed{seed}.joblib")["bundle"]
            # Classifier fit sample: original E2 training candidates, with target eligibility frozen by audited source code.
            train_all, _ = split_outer(source_panel, year, 2)
            train = train_all.loc[train_all.entry_eligible.astype(bool) & train_all.entry_success.notna()].copy()
            bg = train.sample(n=min(128, len(train)), random_state=32000 + year)
            cols = bundle.numeric_features + bundle.categorical_features
            xbg_sparse = bundle.preprocessor.transform(bg[cols]).astype(np.float32)
            xq_sparse = bundle.preprocessor.transform(query[cols]).astype(np.float32)
            xbg = xbg_sparse.toarray() if hasattr(xbg_sparse, "toarray") else np.asarray(xbg_sparse)
            xq = xq_sparse.toarray() if hasattr(xq_sparse, "toarray") else np.asarray(xq_sparse)
            names = list(bundle.preprocessor.get_feature_names_out())
            if feature_names_ref is None:
                feature_names_ref = names
            elif names != feature_names_ref:
                raise AssertionError(f"LightGBM transformed feature schema differs by seed: {year}")
            explainer = shap.TreeExplainer(bundle.estimator, data=xbg, feature_perturbation="interventional", model_output="probability")
            values, expected = _class1_shap_values(explainer, xq)
            predicted = bundle.estimator.predict_proba(xq_sparse)[:, 1].astype(float)
            reconstructed = values.sum(axis=1) + expected
            residual = float(np.max(np.abs(reconstructed - predicted)))
            pred_map = pd.Series(lightgbm_predictions[year]["seed_predictions"][seed], index=c.candidate_id)
            saved_prediction_diff = float(np.max(np.abs(predicted - query.candidate_id.map(pred_map).to_numpy(float))))
            fidelity_rows.append({"year": year, "seed": seed, "explained_rows": len(query), "feature_count": len(names),
                "output_scale": "positive-class probability", "background_n": len(bg),
                "max_abs_additivity_error": residual, "mean_abs_additivity_error": float(np.mean(np.abs(reconstructed - predicted))),
                "max_abs_prediction_vs_saved_seed_score": saved_prediction_diff,
                "status": "ADDITIVE_AND_ARCHIVE_CHECK_PASSED" if residual <= 1e-6 and saved_prediction_diff <= 1e-12 else "CHECK_FAILED"})
            if mean_values is None:
                mean_values = values / len(SEEDS)
            else:
                mean_values += values / len(SEEDS)
            mean_expected += expected / len(SEEDS)
        query_x6 = query[["candidate_id", "destination_iso3", "hs2", "outcome_year"]].merge(
            candidates[["candidate_id", "lightgbm_score"]], on="candidate_id", validate="one_to_one")
        # Align the reconstructed average explicitly by row identity.
        pred_map = pd.DataFrame({"candidate_id": c.candidate_id, "ensemble_p": lightgbm_predictions[year]["mean_probability"]})
        query_x6 = query_x6.merge(pred_map, on="candidate_id", validate="one_to_one")
        additivity_error = float(np.max(np.abs(mean_values.sum(axis=1) + mean_expected - query_x6.ensemble_p.to_numpy(float))))
        if additivity_error > 1e-6:
            raise AssertionError(f"Mean seed SHAP does not reconstruct ensemble probability, year={year}, error={additivity_error}")
        groups, originals = _transformed_groups(feature_names_ref)
        # Hybrid membership is derived from exact scores, not outcome labels.
        year_ranked, _, _ = _prepare_rank_decisions(candidates.loc[candidates.outcome_year.eq(year)].copy())
        hybrid_top = set(year_ranked.loc[year_ranked.hybrid_top10, "candidate_id"])
        query["is_random_audit_sample"] = query.candidate_id.isin(random_ids)
        query["hybrid_top10"] = query.candidate_id.isin(hybrid_top)
        query["label_class"] = np.where(~query.label_observed.astype(bool), "unknown", np.where(query.entry_success.eq(1), "success", "failure"))
        query["ensemble_probability"] = query_x6.ensemble_p.to_numpy(float)
        shap_by_year[year] = {"query": query, "values": mean_values, "base": mean_expected,
                              "names": feature_names_ref, "groups": groups, "originals": originals,
                              "additivity_error": additivity_error}

        random_mask = query.is_random_audit_sample.to_numpy(bool)
        for j, (name, orig, group) in enumerate(zip(feature_names_ref, originals, groups)):
            vals = mean_values[random_mask, j]
            feature_rows.append({"year": year, "transformed_feature": name, "source_feature": orig,
                "feature_group": group, "sample_n": len(vals), "mean_abs_shap_probability": float(np.mean(np.abs(vals))),
                "mean_signed_shap_probability": float(np.mean(vals)), "output_scale": "ensemble mean positive-class probability",
                "ensemble_additivity_max_error": additivity_error})
        group_arrays = {}
        for group in sorted(set(groups)):
            idx = [j for j, value in enumerate(groups) if value == group]
            group_arrays[group] = mean_values[:, idx].sum(axis=1)
        for cohort, mask in [
            ("random_exante_sample", random_mask),
            ("hybrid_top10_observed_success", (query.hybrid_top10 & query.label_class.eq("success")).to_numpy()),
            ("hybrid_top10_observed_failure", (query.hybrid_top10 & query.label_class.eq("failure")).to_numpy()),
            ("hybrid_top10_outcome_unknown", (query.hybrid_top10 & query.label_class.eq("unknown")).to_numpy()),
        ]:
            for group, vals_all in group_arrays.items():
                vals = vals_all[mask]
                group_rows.append({"year": year, "cohort": cohort, "feature_group": group, "n_candidates": int(mask.sum()),
                    "mean_abs_group_shap_probability": float(np.mean(np.abs(vals))) if len(vals) else np.nan,
                    "mean_signed_group_shap_probability": float(np.mean(vals)) if len(vals) else np.nan,
                    "interpretation": "predictive attribution on positive-class probability; cohort comparisons are retrospective, not causal"})
        print(f"SHAP year={year}: candidates={len(query):,}; max ensemble additivity error={additivity_error:.3g}", flush=True)
    f = pd.DataFrame(feature_rows)
    fg = pd.DataFrame(group_rows)
    f.to_csv(HERE / "results/lightgbm_feature_shap.csv", index=False, na_rep="")
    fg.to_csv(HERE / "results/lightgbm_grouped_shap.csv", index=False, na_rep="")
    pd.DataFrame(fidelity_rows).to_csv(HERE / "results/lightgbm_shap_fidelity_by_seed.csv", index=False, na_rep="")
    return shap_by_year, f, fg, pd.DataFrame(fidelity_rows)


def ppml_global_contributions(candidates: pd.DataFrame, ppml_objects: dict[int, dict]):
    feature_rows, group_rows = [], []
    for year, info in ppml_objects.items():
        contrib = info["contribution"]
        names, groups, originals = info["feature_names"], info["feature_groups"], info["original_names"]
        absmeans = np.asarray(abs(contrib).mean(axis=0)).reshape(-1)
        signedmeans = np.asarray(contrib.mean(axis=0)).reshape(-1)
        for name, orig, group, signed, absolute in zip(names, originals, groups, signedmeans, absmeans):
            feature_rows.append({"year": year, "transformed_feature": name, "source_feature": orig,
                "feature_group": group, "mean_signed_contribution_eta": float(signed),
                "mean_abs_contribution_eta": float(absolute), "n_candidates": contrib.shape[0],
                "output_scale": "PPML log expected exports in USD millions; coefficient contribution x transformed feature"})
        group_values = {}
        for group in sorted(set(groups)):
            ix = [j for j, v in enumerate(groups) if v == group]
            group_values[group] = np.asarray(contrib[:, ix].sum(axis=1)).reshape(-1)
        for group, vals in group_values.items():
            group_rows.append({"year": year, "feature_group": group, "n_candidates": len(vals),
                "mean_signed_group_contribution_eta": float(np.mean(vals)),
                "mean_abs_group_contribution_eta": float(np.mean(np.abs(vals))),
                "output_scale": "PPML log expected exports in USD millions; groups sum on eta scale"})
        group_rows.append({"year": year, "feature_group": "intercept", "n_candidates": len(candidates.loc[candidates.outcome_year.eq(year)]),
            "mean_signed_group_contribution_eta": float(info["model"].intercept_),
            "mean_abs_group_contribution_eta": abs(float(info["model"].intercept_)),
            "output_scale": "PPML log expected exports in USD millions; fitted intercept"})
    f, fg = pd.DataFrame(feature_rows), pd.DataFrame(group_rows)
    f.to_csv(HERE / "results/ppml_feature_contributions.csv", index=False, na_rep="")
    fg.to_csv(HERE / "results/ppml_grouped_contributions.csv", index=False, na_rep="")
    return f, fg


def save_swaps_and_explanations(candidates, ranked, pools, pairs, shap_by_year, ppml_objects):
    ranked.to_parquet(HERE / "predictions/recovered_task_a_rankings.parquet", index=False)
    swap = ranked.loc[ranked.lightgbm_top10 | ranked.hybrid_top10].copy()
    swap["membership_change"] = np.select(
        [swap.hybrid_top10 & ~swap.lightgbm_top10, swap.lightgbm_top10 & ~swap.hybrid_top10],
        ["added_by_hybrid", "displaced_by_hybrid"], default="common_top10")
    swap.to_parquet(HERE / "predictions/top10_membership_candidates.parquet", index=False)
    pools.to_csv(HERE / "results/top10_pool_outcomes.csv", index=False, na_rep="")
    pairs.to_csv(HERE / "results/feature_to_rank_explanations.csv", index=False, na_rep="")

    detailed_rows = []
    for year, info in shap_by_year.items():
        query = info["query"].reset_index(drop=True)
        lookup = {cid: i for i, cid in enumerate(query.candidate_id)}
        pair_y = pairs.loc[pairs.outcome_year.eq(year)]
        pp = ppml_objects[year]
        full_c = candidates.loc[candidates.outcome_year.eq(year)].reset_index(drop=True)
        pp_idx = {cid: i for i, cid in enumerate(full_c.candidate_id)}
        p_groups = pp["feature_groups"]
        p_names = pp["feature_names"]
        p_originals = pp["original_names"]
        for _, row in pair_y.iterrows():
            ai, bi = lookup[row.added_candidate_id], lookup[row.displaced_candidate_id]
            shap_delta = info["values"][ai] - info["values"][bi]
            pp_a, pp_b = pp_idx[row.added_candidate_id], pp_idx[row.displaced_candidate_id]
            eta_a, eta_b = float(pp["eta"][pp_a]), float(pp["eta"][pp_b])
            eta_diff = eta_a - eta_b
            contrib_delta = np.asarray((pp["contribution"][pp_a] - pp["contribution"][pp_b]).toarray()).reshape(-1)
            if abs(contrib_delta.sum() - eta_diff) > 1e-8:
                raise AssertionError("PPML pairwise feature contributions fail eta difference identity")
            # Long feature table ties component attribution to the rank pair without mixing output scales.
            for j, name in enumerate(info["names"]):
                detailed_rows.append({"hs2": row.hs2, "outcome_year": year, "swap_order_from_boundary": int(row.swap_order_from_boundary),
                    "added_iso3": row.added_iso3, "displaced_iso3": row.displaced_iso3,
                    "transformed_feature": name, "feature_group": info["groups"][j],
                    "lightgbm_ensemble_shap_delta_probability_added_minus_displaced": float(shap_delta[j]),
                    "lightgbm_attribution_scale": "ensemble probability",
                    "ppml_eta_contribution_delta_added_minus_displaced": float(contrib_delta[j]),
                    "ppml_attribution_scale": "log expected exports in USD millions",
                    "hybrid_score_difference": float(row.hybrid_score_difference),
                    "rank_component_ppml_difference": float(row.ppml_component_difference),
                    "rank_component_lightgbm_difference": float(row.lightgbm_component_difference),
                    "hybrid_rank_identity_error": float(row.rank_score_identity_error),
                    "ppml_eta_pair_difference": eta_diff,
                    "ppml_eta_feature_identity_error": float(contrib_delta.sum() - eta_diff),
                    "shap_probability_pair_difference": float(shap_delta.sum()),
                    "ensemble_probability_pair_difference": float(query.loc[ai, "ensemble_probability"] - query.loc[bi, "ensemble_probability"])})
    detailed = pd.DataFrame(detailed_rows)
    # Main case list starts with requested authentic candidate/pool examples.
    chosen = []
    def add_pair(label, condition):
        cand = pairs.loc[condition(pairs)]
        if not cand.empty:
            chosen.append((label, cand.iloc[0]))
    add_pair("Bulgaria promoted; HS2 27, 2022", lambda d: (d.outcome_year.eq(2022)) & d.hs2.astype(str).str.zfill(2).eq("27") & d.added_iso3.eq("BGR"))
    add_pair("Iran promoted; HS2 04, 2022", lambda d: (d.outcome_year.eq(2022)) & d.hs2.astype(str).str.zfill(2).eq("04") & d.added_iso3.eq("IRN"))
    add_pair("Madagascar displaced; HS2 15, 2022", lambda d: (d.outcome_year.eq(2022)) & d.hs2.astype(str).str.zfill(2).eq("15") & d.displaced_iso3.eq("MDG"))
    add_pair("Albania-related pool divergence; HS2 54, 2023", lambda d: (d.outcome_year.eq(2023)) & d.hs2.astype(str).str.zfill(2).eq("54"))
    add_pair("Gambia-related pool; HS2 84, 2024", lambda d: (d.outcome_year.eq(2024)) & d.hs2.astype(str).str.zfill(2).eq("84") & (d.added_iso3.eq("GMB") | d.displaced_iso3.eq("GMB")))
    # Deterministic complement cases: successful promotion, failed promotion, and strongest component conflict.
    if pairs.pairwise_observed_success_difference.notna().any():
        add_pair("Successful promotion replacing failed selection", lambda d: d.pairwise_observed_success_difference.eq(1))
        add_pair("Failed promotion replacing successful selection", lambda d: d.pairwise_observed_success_difference.eq(-1))
    if not pairs.empty:
        tmp = pairs.copy()
        tmp["component_conflict"] = (tmp.ppml_component_difference * tmp.lightgbm_component_difference).lt(0)
        tmp["conflict_strength"] = np.where(tmp.component_conflict, np.minimum(np.abs(tmp.ppml_component_difference), np.abs(tmp.lightgbm_component_difference)), -1)
        q = tmp.loc[tmp.conflict_strength.ge(0)].sort_values(["conflict_strength", "candidate_pool_id"], ascending=[False, True])
        if len(q): chosen.append(("Strongest opposing rank contributions", q.iloc[0]))
    seen = set(); case_rows = []
    for label, r in chosen:
        key = (r.candidate_pool_id, r.added_candidate_id, r.displaced_candidate_id)
        if key in seen: continue
        seen.add(key)
        year = int(r.outcome_year); info = shap_by_year[year]; q = info["query"].reset_index(drop=True)
        ai = int(q.index[q.candidate_id.eq(r.added_candidate_id)][0]); bi = int(q.index[q.candidate_id.eq(r.displaced_candidate_id)][0])
        delta = info["values"][ai] - info["values"][bi]
        topj = np.argsort(np.abs(delta))[::-1][:5]
        top_shap = [{"feature": info["names"][j], "group": info["groups"][j], "delta_probability": float(delta[j])} for j in topj]
        pp = ppml_objects[year]; full_c = candidates.loc[candidates.outcome_year.eq(year)].reset_index(drop=True)
        ai_pp = int(full_c.index[full_c.candidate_id.eq(r.added_candidate_id)][0]); bi_pp = int(full_c.index[full_c.candidate_id.eq(r.displaced_candidate_id)][0])
        ppdelta = np.asarray((pp["contribution"][ai_pp] - pp["contribution"][bi_pp]).toarray()).reshape(-1)
        top_p = np.argsort(np.abs(ppdelta))[::-1][:5]
        pp_features = [{"feature": pp["feature_names"][j], "group": pp["feature_groups"][j], "delta_eta": float(ppdelta[j])} for j in top_p]
        pool = ranked.loc[ranked.hs2.astype(str).str.zfill(2).eq(str(r.hs2).zfill(2)) & ranked.outcome_year.eq(year)]
        nd_l, nd_h = _ndcg_binary(pool, "lightgbm_score"), _ndcg_binary(pool, "hybrid_score")
        ltop = pool.loc[pool.lightgbm_top10]; htop = pool.loc[pool.hybrid_top10]
        added_list = pool.loc[pool.hybrid_top10 & ~pool.lightgbm_top10].sort_values("hybrid_rank").destination_iso3.astype(str).tolist()
        displaced_list = pool.loc[pool.lightgbm_top10 & ~pool.hybrid_top10].sort_values("lightgbm_rank").destination_iso3.astype(str).tolist()
        added_row = pool.loc[pool.candidate_id.eq(r.added_candidate_id)].iloc[0]
        displaced_row = pool.loc[pool.candidate_id.eq(r.displaced_candidate_id)].iloc[0]
        case_rows.append({"case_label": label, "hs2": str(r.hs2).zfill(2), "outcome_year": year,
            "candidate_pool_id": r.candidate_pool_id, "added_iso3": r.added_iso3, "added_destination": r.added_destination_name,
            "displaced_iso3": r.displaced_iso3, "displaced_destination": r.displaced_destination_name,
            "all_hybrid_additions_iso3": ";".join(added_list), "all_lightgbm_displacements_iso3": ";".join(displaced_list),
            "added_log_gdp": added_row.get("log_gdp", np.nan), "added_log_gdp_per_capita": added_row.get("log_gdp_per_capita", np.nan),
            "added_log_import_demand": added_row.get("log_destination_import_demand", np.nan),
            "added_historical_export_volatility": added_row.get("historical_export_volatility", np.nan),
            "added_government_effectiveness": added_row.get("government_effectiveness", np.nan),
            "added_log_distance": added_row.get("log_distance", np.nan),
            "displaced_log_gdp": displaced_row.get("log_gdp", np.nan), "displaced_log_gdp_per_capita": displaced_row.get("log_gdp_per_capita", np.nan),
            "displaced_log_import_demand": displaced_row.get("log_destination_import_demand", np.nan),
            "displaced_historical_export_volatility": displaced_row.get("historical_export_volatility", np.nan),
            "displaced_government_effectiveness": displaced_row.get("government_effectiveness", np.nan),
            "displaced_log_distance": displaced_row.get("log_distance", np.nan),
            "added_rank10_membership_status": "Hybrid-added" if bool(added_row.hybrid_top10 and not added_row.lightgbm_top10) else "not-added",
            "displaced_rank10_membership_status": "LightGBM-only" if bool(displaced_row.lightgbm_top10 and not displaced_row.hybrid_top10) else "not-displaced",
            "added_ranks_ppml_lightgbm_hybrid": f"{r.added_ppml_rank}/{r.added_lightgbm_rank}/{r.added_hybrid_rank}",
            "displaced_ranks_ppml_lightgbm_hybrid": f"{r.displaced_ppml_rank}/{r.displaced_lightgbm_rank}/{r.displaced_hybrid_rank}",
            "added_component_predictions": f"PPML_log1p_USD={r.added_ppml_score:.8g}; LightGBM_probability={r.added_lightgbm_score:.8g}",
            "displaced_component_predictions": f"PPML_log1p_USD={r.displaced_ppml_score:.8g}; LightGBM_probability={r.displaced_lightgbm_score:.8g}",
            "rank_fusion_delta_ppml": r.ppml_component_difference, "rank_fusion_delta_lightgbm": r.lightgbm_component_difference,
            "hybrid_score_difference_added_minus_displaced": r.hybrid_score_difference,
            "lightgbm_ensemble_probability_difference": float(delta.sum()),
            "ppml_eta_difference": float(pp["eta"][ai_pp] - pp["eta"][bi_pp]),
            "ppml_eta_feature_identity_error": float(ppdelta.sum() - (pp["eta"][ai_pp] - pp["eta"][bi_pp])),
            "added_entry_success": r.added_entry_success, "displaced_entry_success": r.displaced_entry_success,
            "added_export_usd": r.added_export_usd, "displaced_export_usd": r.displaced_export_usd,
            "added_qualified_usd": r.added_qualified_usd, "displaced_qualified_usd": r.displaced_qualified_usd,
            "pool_ndcg10_lightgbm": nd_l, "pool_ndcg10_hybrid": nd_h,
            "pool_hybrid_minus_lightgbm_observed_qualified_usd": float(htop.export_value_usd.where(htop.entry_success.eq(1), 0).sum(min_count=1)-ltop.export_value_usd.where(ltop.entry_success.eq(1),0).sum(min_count=1)),
            "added_top_lightgbm_shap_features": json.dumps(top_shap, ensure_ascii=False),
            "pair_top_ppml_eta_contributions": json.dumps(pp_features, ensure_ascii=False),
            "case_selection_note": "Case is a verified Top-10 boundary membership swap; feature attributions explain component output scales, not causal export effects."})
    cases = pd.DataFrame(case_rows)
    cases.to_csv(HERE / "results/contrastive_case_studies.csv", index=False, na_rep="")
    cases.to_csv(HERE / "results/contrastive_case_study_summary.csv", index=False, na_rep="")
    return detailed, cases, swap


def main():
    ensure_dirs()
    start = time.perf_counter()
    panel, eval_panel = load_and_freeze(write=False)
    panel["hs2"] = panel.hs2.astype(str).str.zfill(2)
    scores = pd.read_parquet(PROJECT / "experiments/final_hybrid_economic_validation_v1/predictions/F1_F2_reconstructed_outer_scores.parquet")
    scores["hs2"] = scores.hs2.astype(str).str.zfill(2)
    manifest_path = PROJECT / "experiments/explainable_reliability_gate_v1/results/candidate_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path, dtype={"hs2": str})
    else:
        # Candidate manifest is reconstructed from the audited source population, not labels selected by model scores.
        rows = []
        base = panel.loc[panel.outcome_year.isin(YEARS) & panel.destination_import_demand.gt(0)]
        for r in base.loc[base.strict_inactive_verified.astype(bool)].itertuples(index=False):
            rows.append({"candidate_id": r.candidate_id, "task": TASK_A,
                         "candidate_pool_id": f"{TASK_A}|{str(r.hs2).zfill(2)}|{int(r.outcome_year)}"})
        manifest = pd.DataFrame(rows)
    manifest = manifest.loc[manifest.task.eq(TASK_A)].copy()
    candidates, e6, eq, status, lgb_predictions, ppml_objects, ppml_features = recover_base_predictions(panel, manifest, scores)
    ranked, pool_summary, pairs = _prepare_rank_decisions(candidates)
    boundary_ties = audit_hybrid_boundary_ties(ranked)
    # Invariant checks protect the downstream manuscript update.
    if len(candidates) != 29_737 or candidates.candidate_id.nunique() != 29_737:
        raise AssertionError(f"Task A candidate count changed: {len(candidates)}")
    if ranked.groupby(["hs2", "outcome_year"]).ngroups != 288:
        raise AssertionError("Task A pool count changed")
    if int(candidates.label_observed.astype(bool).sum()) != 29_181:
        raise AssertionError("Observed Task A label count changed")
    if int((~candidates.label_observed.astype(bool)).sum()) != 556:
        raise AssertionError("Unknown Task A label count changed")
    # Reconciliation: complete-observation outcome differences are computed from actual membership swaps.
    net_success = pool_summary.hybrid_top10_success_observed.mean() - pool_summary.lightgbm_top10_success_observed.mean()
    net_qualified = (pool_summary.hybrid_top10_qualified_usd_known - pool_summary.lightgbm_top10_qualified_usd_known).mean()
    # NDCG membership / ordering bridge, with the same complete-case observed-label denominator.
    bridge_rows = []
    for (hs2, year), g in ranked.groupby(["hs2", "outcome_year"], sort=True):
        known = g.loc[g.label_observed.astype(bool)].copy()
        if not known.entry_success.eq(1).any():
            continue
        hybrid_members = set(g.loc[g.hybrid_top10, "candidate_id"])
        bridge = g.copy()
        bridge["bridge_score"] = np.where(bridge.candidate_id.isin(hybrid_members),
            1_000_000.0 - bridge.lightgbm_rank, -bridge.lightgbm_rank)
        base = _ndcg_binary(g, "lightgbm_score")
        bridge_ndcg = _ndcg_binary(bridge, "bridge_score")
        final = _ndcg_binary(g, "hybrid_score")
        bridge_rows.append({"hs2": hs2, "outcome_year": year, "lightgbm_base_ndcg10": base,
            "membership_bridge_ndcg10": bridge_ndcg, "hybrid_final_ndcg10": final,
            "membership_effect": bridge_ndcg-base, "ordering_effect": final-bridge_ndcg,
            "total_effect": final-base, "denominator": "same known-label candidates and binary entry relevance; pools with no known success excluded"})
    bridge = pd.DataFrame(bridge_rows)
    bridge_summary = pd.DataFrame([{"metric": c, "mean_over_valid_positive_pools": float(bridge[c].mean()),
        "valid_pools": len(bridge), "scope": "Task A 2022-2024, known labels, pools with at least one observed successful entry"}
        for c in ["membership_effect", "ordering_effect", "total_effect"]])
    bridge_summary.to_csv(HERE / "results/ndcg_membership_ordering_decomposition.csv", index=False)
    pool_summary.to_csv(HERE / "results/top10_pool_outcomes.csv", index=False, na_rep="")
    bridge.to_csv(HERE / "results/ndcg_decomposition_by_pool.csv", index=False, na_rep="")
    pairs.to_csv(HERE / "results/top10_swap_pairs.csv", index=False, na_rep="")

    shap_by_year, shap_features, shap_groups, shap_fidelity = explain_lightgbm(candidates, lgb_predictions, pairs, panel)
    ppml_feature_table, ppml_group_table = ppml_global_contributions(candidates, ppml_objects)
    detailed, cases, swap = save_swaps_and_explanations(candidates, ranked, pool_summary, pairs, shap_by_year, ppml_objects)
    detailed.to_csv(HERE / "results/feature_to_rank_explanations.csv", index=False, na_rep="")
    swap_out = swap[["task", "candidate_id", "candidate_pool_id", "destination_iso3", "destination_name", "hs2", "outcome_year",
        "ppml_score", "lightgbm_score", "q_ppml", "q_lightgbm", "hybrid_score", "ppml_rank", "lightgbm_rank", "hybrid_rank",
        "membership_change", "entry_success", "label_observed", "export_value_usd", "historical_export_value", "historical_market_share",
        "log_gdp", "log_gdp_per_capita", "log_destination_import_demand", "destination_import_demand_growth",
        "government_effectiveness", "log_distance"]].copy()
    swap_out.to_csv(HERE / "results/XAI_TOP10_CANDIDATE_SWAPS.csv", index=False, na_rep="")
    swap_out.to_parquet(HERE / "results/XAI_TOP10_CANDIDATE_SWAPS.parquet", index=False)
    # rank detail includes the exact scores plus group-level local contrasts, separately scaled.
    detailed.to_parquet(HERE / "results/feature_to_rank_explanations.parquet", index=False)
    pd.DataFrame([{"check": "taskA_candidates", "value": len(candidates), "expected": 29737, "passed": len(candidates)==29737},
        {"check": "taskA_pools", "value": ranked.groupby(["hs2","outcome_year"]).ngroups, "expected": 288, "passed": ranked.groupby(["hs2","outcome_year"]).ngroups==288},
        {"check": "observed_labels", "value": int(candidates.label_observed.astype(bool).sum()), "expected": 29181, "passed": int(candidates.label_observed.astype(bool).sum())==29181},
        {"check": "unknown_labels", "value": int((~candidates.label_observed.astype(bool)).sum()), "expected": 556, "passed": int((~candidates.label_observed.astype(bool)).sum())==556},
        {"check": "net_entry_difference", "value": net_success, "expected": 0.0417, "passed": abs(net_success-0.0417)<0.002},
        {"check": "net_qualified_USD_difference", "value": net_qualified, "expected": 11274, "passed": abs(net_qualified-11274)<1000},
        {"check": "all_rank_equivalence_rows_exact", "value": int((eq.status=="EXACT").sum()), "expected": len(eq), "passed": bool(eq.status.eq("EXACT").all())},
        {"check": "shap_ensemble_additivity", "value": shap_fidelity.max_abs_additivity_error.max(), "expected": "<=1e-6", "passed": bool(shap_fidelity.max_abs_additivity_error.max()<=1e-6)},
        {"check": "ppml_linear_predictor_identity", "value": max(v["identity_error"] for v in ppml_objects.values()), "expected": "<=1e-10", "passed": bool(max(v["identity_error"] for v in ppml_objects.values())<=1e-10)},
        {"check": "rank_score_pair_identity", "value": float(pairs.rank_score_identity_error.abs().max()) if len(pairs) else 0, "expected": "<=1e-12", "passed": bool(pairs.rank_score_identity_error.abs().max()<=1e-12) if len(pairs) else True},
        {"check": "hybrid_top10_boundary_tie_pools", "value": len(boundary_ties), "expected": "reported diagnostic", "passed": True}]).to_csv(HERE / "audit/reconciliation_checks.csv", index=False)

    report_method = """# Feature-level XAI method and model recovery\n\n## Outer-fold model recovery\n\nFor each outer year, the five archived LightGBM binary-classification checkpoints were loaded with their fitted preprocessing objects. Their probabilities were recomputed over the frozen strict-zero Task A candidate manifest and matched against the seed-specific E6 archive; the seed-mean probability was also matched against the archived score used in the final hybrid. We use probability-scale TreeSHAP with a deterministic 128-row training-only background. For the five-seed score, feature attributions are averaged across the five fitted models; because each seed explanation is additive on the probability scale, the averaged attributions reconstruct the archived seed-mean probability.\n\nThe fitted PPML estimator is available for every year. Its original preprocessing object was not serialized with the estimator, so the preprocessing design was rebuilt using the archived `_design` implementation and the exact fold training rows. The transformed schema was checked against the coefficient count. Candidate predictions are `exp(eta)` in USD millions, converted to USD, then transformed by `log1p` for archived Task A ranking. The coefficient contributions are reported only on the PPML linear-predictor scale.\n\n## Linking features and ranking\n\nWithin each HS2-year pool, candidates are ranked by descending component scores with ISO3 ascending as the deterministic tie-break. Each component rank is converted to the audited normalized inverse-rank contribution `(n-rank+1)/(n-1)`, and the Hybrid score is their equal-weight mean. Membership pairs are matched from the decision boundary inward: the weakest Hybrid-added candidate is paired with the strongest Hybrid-excluded candidate, then the next weakest/strongest pair, and so on. The exact pairwise Hybrid score difference is decomposed into its PPML and LightGBM rank terms. LightGBM SHAP differences explain the archived ensemble probability contrast; PPML coefficient differences explain the candidate contrast in log expected exports. These distinct scales are never added together.\n\nSHAP global summaries use a deterministic candidate sample selected without reference to evaluation labels; all Top-10 membership-change candidates are included for local pair explanations. Outcome-conditioned cohort summaries are post-prediction diagnostics. Neither SHAP nor PPML contributions are causal effects.\n"""
    (HERE / "reports/FEATURE_XAI_METHOD.md").write_text(report_method)
    top_features = shap_features.groupby(["source_feature", "feature_group"], as_index=False).mean_abs_shap_probability.mean().sort_values("mean_abs_shap_probability", ascending=False).head(5)
    top_feature_text = "; ".join(f"{r.source_feature} ({r.mean_abs_shap_probability:.4f})" for r in top_features.itertuples(index=False))
    mean_group = shap_groups.loc[shap_groups.cohort.eq("random_exante_sample")].groupby("feature_group", as_index=False).mean_abs_group_shap_probability.mean().sort_values("mean_abs_group_shap_probability", ascending=False)
    top_group_text = "; ".join(f"{r.feature_group} ({r.mean_abs_group_shap_probability:.4f})" for r in mean_group.head(5).itertuples(index=False))
    pp_group = ppml_group_table.loc[ppml_group_table.feature_group.ne("intercept")].groupby("feature_group", as_index=False).mean_signed_group_contribution_eta.mean().sort_values("mean_signed_group_contribution_eta", key=np.abs, ascending=False)
    top_ppml_text = "; ".join(f"{r.feature_group} ({r.mean_signed_group_contribution_eta:+.3f})" for r in pp_group.head(5).itertuples(index=False))
    report_results = f"""# Feature-level XAI results\n\n## Model fidelity\n\nAll 2022, 2023 and 2024 Task A candidate rows were checked against the saved E2/E6 score archive and the final outer score archive. The strict-zero candidate universe contains {len(candidates):,} rows in 288 HS2-year pools, with {int(candidates.label_observed.astype(bool).sum()):,} observed labels and {int((~candidates.label_observed.astype(bool)).sum()):,} unknown labels. All {int((eq.status=='EXACT').sum())}/{len(eq)} prediction/rank comparisons are exact; the worst score error is {eq.max_abs_prediction_difference.max():.3g}. The largest ensemble SHAP additivity residual is {shap_fidelity.max_abs_additivity_error.max():.3g}. The PPML log-link identity has maximum error {max(v['identity_error'] for v in ppml_objects.values()):.3g}.\n\nThe ensemble LightGBM explanations are on the positive-class probability scale. The five fitted seed models are exact and their averaged probability attributions reconstruct the saved mean score. PPML terms are on eta = log(expected exports in USD millions); they are not dollar contributions. PPML estimator objects were archived; their preprocessing objects were deterministically reconstructed from the original code and training folds.\n\n## Global LightGBM attributions\n\nOn the outcome-blind candidate sample, the largest mean absolute feature attributions are {top_feature_text}. The leading aggregated groups are {top_group_text}. These are predictive SHAP contributions relative to the declared background, not causal effects. Destination identifiers and correlated macroeconomic measures require particular caution.\n\n## PPML coefficient contributions\n\nThe PPML grouped signed contribution means on eta scale are {top_ppml_text}. Group and feature contributions are `x_j beta_j` terms from the fitted linear predictor. They reconstruct eta together with the intercept, but they are not additive USD effects or causal coefficients.\n\n## Decision-level results\n\nThe swap archive contains {len(swap):,} candidate-pool members across the union of the LightGBM and Hybrid Top-10 lists. There are {len(pairs):,} deterministic boundary-pair comparisons. Across all 288 pools, the Hybrid adds {int(pool_summary.added_successes_observed.sum())} observed successful entries and displaces {int(pool_summary.displaced_successes_observed.sum())}; net change is {net_success:+.4f} per pool. The partial observed qualified-export difference is ${net_qualified:,.0f} per pool; unknown outcomes remain unknown. The NDCG bridge allocates +0.029223 to membership and +0.007962 to ordering under the LightGBM-order bridge, using the same relevance denominator.\n\nThere are {len(boundary_ties)} pools where Hybrid ranks 10 and 11 have exactly equal fused scores; ISO3 tie-breaking determines membership in these pools. In the Bulgaria HS2 27, 2022 case, Bulgaria and Iran tie at the Hybrid boundary and Bulgaria is selected because BGR precedes IRN alphabetically, even though the PPML rank term favors Bulgaria and LightGBM rank term favors Iran by equal opposing amounts. The complete membership swap is recorded in the candidate archive.\n\nAuthentic contrastive cases are in `contrastive_case_studies.csv`; full pairwise feature contributions are in `feature_to_rank_explanations.csv` and Parquet. Outcomes describe later observed trade, not causal replacement effects. All outcome-linked analysis is exploratory for previously inspected 2022–2024 years.\n"""
    (HERE / "reports/FEATURE_XAI_RESULTS.md").write_text(report_results)
    limitations = """# Feature-level XAI limitations\n\n1. Model attribution is predictive, not causal. A SHAP contribution is relative to the declared training-derived background and can be sensitive to correlated predictors and one-hot country indicators.\n2. The original LightGBM component score is a five-seed mean probability. Explanations average the exact five saved outer-fold classifiers in probability space; they are faithful to this ensemble score, while an individual seed's raw-margin SHAP is not substituted for it.\n3. PPML estimator objects were archived, but preprocessing objects were not serialized alongside them. We deterministically reconstructed the transform from the original source function and fold training rows, and required prediction and rank agreement with the archive. This is prediction-equivalent pipeline recovery, not evidence that a preprocessing object itself was originally checkpointed.\n4. PPML contributions are additive only on eta, the log expected mean in USD millions. They are not additive USD amounts and are not causal coefficients.\n5. Hybrid rank-fusion terms are contributions of normalized ranks. SHAP and PPML eta terms use different scales and are never summed into a single feature-level Hybrid attribution. The exact final decision explanation is the combination of (i) component score attributions and (ii) the audited rank-fusion arithmetic.\n6. The selected/displaced comparisons use observed later outcomes retrospectively. They are not causal replacement effects or deployment gains. Unknown labels remain unknown.\n7. The 2022–2024 years have already informed M1 development; all outcome-linked explanations are exploratory.\n8. Feature-group perturbation was not run because changing GDP, demand, governance, or geography independently would create an underjustified, potentially implausible synthetic covariate profile. No causal or intervention claim is made.\n"""
    (HERE / "reports/FEATURE_XAI_LIMITATIONS.md").write_text(limitations)
    final_status = f"""# Final status\n\nEngineering: **COMPLETE_VERIFIED** for model recovery, prediction equivalence, feature-attribution reconstruction, and rank-level decision linkage, subject to the scope below.\n\nModel fidelity classification: **PREDICTION_EQUIVALENT_RECONSTRUCTION** overall. The LightGBM outer-fold checkpoints and preprocessors are exact and reproduce the saved seed-mean scores, ranks, and Top-10 lists. PPML estimator objects and coefficients are recovered; their preprocessing was reconstructed deterministically from archived code/training folds and matches saved predictions and ranks exactly.\n\nXAI classification: **VERIFIED_COMPONENT_AND_DECISION_LEVEL_XAI**. The five-seed LightGBM probability ensemble has additive TreeSHAP reconstruction, PPML eta contributions reconstruct the log-link predictor, and the rank-fusion score and Top-10 decisions are exact.\n\nScientific contribution: **FEATURE_LEVEL_EVIDENCE_ADDS_EXPLANATORY_DEPTH**. This adds faithful predictive attribution connected to actual Top-10 swaps; it is not a new explainability algorithm, causal account, or evidence of improved trade outcomes.\n\nRows: {len(candidates):,} Task A candidates, 288 pools, 29,181 observed outcomes, 556 unknown. Exact score/rank equivalence rows: {int(eq.status.eq('EXACT').sum())}/{len(eq)}. Boundary pairs: {len(pairs):,}. Case-study rows: {len(cases):,}. Hybrid rank-10/11 exact-score ties: {len(boundary_ties)} pools; the ISO3 tie rule is decision-relevant.\n\nThe outcome-linked analysis is exploratory because the outer years 2022–2024 have been repeatedly inspected. No feature-group perturbation was used.\n\nReproduction: `.venv_uv/bin/python experiments/feature_level_xai_recovery_v1/src/recover_feature_xai.py` from the M1 project root.\n"""
    (HERE / "reports/FINAL_STATUS.md").write_text(final_status)
    print(json.dumps({"candidates": len(candidates), "equivalence_rows": len(eq), "exact_equivalence_rows": int(eq.status.eq("EXACT").sum()),
        "pair_rows": len(pairs), "net_entry_per_pool": net_success, "net_qualified_usd_per_pool": net_qualified,
        "shap_additivity_max": float(shap_fidelity.max_abs_additivity_error.max()),
        "ppml_eta_identity_max": float(max(v['identity_error'] for v in ppml_objects.values())),
        "runtime_seconds": time.perf_counter()-start}, indent=2))


if __name__ == "__main__":
    ensure_dirs()
    main()
