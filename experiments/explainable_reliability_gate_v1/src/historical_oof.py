from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from .task_audit import HERE, PROJECT

sys.path.insert(0, str(PROJECT / "src"))
from gravity_ppml import fit_ppml  # noqa: E402
from ml_models import fit_classifier, fit_regressor, tune_classifier, tune_regressor  # noqa: E402

OOF_YEARS = (2018, 2019, 2020, 2021)
SEED = 11


def _pool_id(task: str, hs2: str, year: int) -> str:
    return f"{task}|{str(hs2).zfill(2)}|{int(year)}"


def _base_parameters() -> dict:
    return {"n_estimators": 120, "num_leaves": 31, "learning_rate": 0.05,
            "max_depth": -1, "min_child_samples": 40, "n_jobs": 4, "verbosity": -1}


def _trainable(panel: pd.DataFrame, outcome_cutoff: int) -> pd.DataFrame:
    return panel.loc[panel.outcome_observed.astype(bool) & panel.export_value_usd.notna()
                     & panel.destination_import_demand.gt(0)
                     & panel.outcome_year.le(int(outcome_cutoff))].copy()


def reconstruct_historical_oof(panel: pd.DataFrame, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct historical rolling-origin predictions for the two fixed experts.

    Each historical outcome year is predicted from models whose outcomes end at y-2.
    These are training data for the later pool-level gate, never in-sample predictions.
    """
    outpath = HERE / "predictions/historical_oof_expert_predictions.parquet"
    diagpath = HERE / "results/historical_oof_model_diagnostics.csv"
    if outpath.exists() and diagpath.exists() and not force:
        return pd.read_parquet(outpath), pd.read_csv(diagpath)
    outputs, diagnostics = [], []
    panel = panel.copy()
    panel["hs2"] = panel.hs2.astype(str).str.zfill(2)
    for year in OOF_YEARS:
        fold_start = time.perf_counter()
        cutoff = year - 2
        train = _trainable(panel, cutoff)
        test = panel.loc[panel.outcome_year.eq(year) & panel.destination_import_demand.gt(0)].copy()
        if train.empty or test.empty:
            raise RuntimeError(f"Empty historical fold {year}")
        if int(train.outcome_year.max()) > cutoff or (train.feature_year > train.outcome_year - 2).any():
            raise AssertionError(f"Historical training leakage in fold {year}")
        if (test.feature_year > test.outcome_year - 2).any():
            raise AssertionError(f"Historical test feature leakage in fold {year}")
        test["candidate_id"] = [
            __import__("hashlib").sha256(f"{a}|{b}|{int(c)}".encode()).hexdigest()[:20]
            for a, b, c in test[["destination_iso3", "hs2", "outcome_year"]].itertuples(index=False, name=None)]
        # Gravity and LightGBM regression predict export levels. Entry ranking uses
        # log1p(PPML prediction), the same monotone transform as the saved E2 baseline.
        ppml_prediction, ppml_model, ppml_diag, ppml_objects = fit_ppml(train, test)
        joblib.dump({"model": ppml_model, "preprocessing": ppml_objects,
                     "prediction_year": year, "train_outcome_max": int(train.outcome_year.max()),
                     "diagnostics": ppml_diag}, HERE / f"models/ppml_oof_{year}.joblib")
        diagnostics.append({"year": year, "model": "Gravity_PPML", "train_n": len(train),
            "train_outcome_max": int(train.outcome_year.max()), "prediction_n": len(test),
            "converged": ppml_diag["converged"], "iterations": ppml_diag["iterations"],
            "parameters": ppml_diag["n_parameters"], "runtime_seconds": time.perf_counter()-fold_start,
            "specification": "M1 PoissonRegressor log-link with L2; original USD predictions"})
        test_base = test[["candidate_id", "destination_iso3", "destination_name", "hs2", "outcome_year", "label_observed",
                          "export_value_usd", "realized_incremental_export_usd", "entry_success",
                          "historical_export_value", "historical_market_share", "log_gdp",
                          "log_gdp_per_capita", "log_population", "log_destination_import_demand",
                          "destination_import_demand_growth", "historical_export_volatility",
                          "internet_usage", "government_effectiveness", "digitalization_missing_indicator",
                          "destination_import_demand", "historical_export_presence", "destination_supplier_concentration",
                          "lagged_export_growth", "inflation", "trade_openness", "log_distance", "contiguous",
                          "common_official_language", "colonial_link", "common_colonizer", "landlocked",
                          "regulatory_quality", "rule_of_law", "political_stability", "continent"]].copy()
        for task, mask in (("Task_A_new_entry", test.strict_inactive_verified.astype(bool)),
                           ("Task_B_existing_underexploited", test.task_b_eligible.astype(bool))):
            part = test_base.loc[mask].copy()
            if part.empty:
                continue
            part["task"] = task
            part["candidate_pool_id"] = [_pool_id(task, h, y) for h, y in zip(part.hs2, part.outcome_year)]
            if task == "Task_A_new_entry":
                part["ppml_score"] = np.log1p(np.maximum(0.0, ppml_prediction[mask.to_numpy()]))
            else:
                part["ppml_score"] = np.maximum(0.0, ppml_prediction[mask.to_numpy()] - part.historical_export_value.to_numpy(float))
            part["ppml_prediction_usd"] = ppml_prediction[mask.to_numpy()]
            part["lightgbm_score"] = np.nan
            part["lightgbm_prediction_usd"] = np.nan
            part["prediction_provenance"] = "rolling-origin historical OOF; model outcomes end at outcome_year-2"
            outputs.append(part)

        # Regression expert. Tuning/scaling uses the project chronological inner split.
        reg_params, reg_validation = tune_regressor("LightGBM", train, lag=2, trials=2)
        reg_bundle = fit_regressor("LightGBM", train, reg_params, SEED, lag=2)
        joblib.dump({"bundle": reg_bundle, "params": reg_params, "prediction_year": year,
                     "train_outcome_max": int(train.outcome_year.max())}, HERE / f"models/lightgbm_reg_oof_{year}.joblib")
        pred_usd = reg_bundle.predict(test, log_target=True)
        diagnostics.append({"year": year, "model": "LightGBM_regression", "train_n": len(train),
            "train_outcome_max": int(train.outcome_year.max()), "prediction_n": len(test),
            "converged": True, "iterations": reg_params.get("n_estimators"), "parameters": len(reg_params),
            "validation_score": reg_validation, "runtime_seconds": time.perf_counter()-fold_start,
            "specification": "M1 log1p regression; chronological inner selection and development-only USD scaling"})
        for task, mask in (("Task_B_existing_underexploited", test.task_b_eligible.astype(bool)),):
            idx = np.flatnonzero(mask.to_numpy())
            if not len(idx):
                continue
            part = test_base.iloc[idx].copy()
            part["task"] = task
            part["candidate_pool_id"] = [_pool_id(task, h, y) for h, y in zip(part.hs2, part.outcome_year)]
            part["lightgbm_score"] = np.maximum(0.0, pred_usd[idx] - part.historical_export_value.to_numpy(float))
            part["lightgbm_prediction_usd"] = pred_usd[idx]
            part["ppml_score"] = np.nan
            part["ppml_prediction_usd"] = np.nan
            part["prediction_provenance"] = "rolling-origin historical OOF; model outcomes end at outcome_year-2"
            outputs.append(part)

        # Entry expert: reproduce the E2 LightGBM probability model on the historical
        # below-threshold candidate training population. Only strict-zero candidates
        # are scored in the primary evaluation pool.
        entry_train = train.loc[train.entry_eligible.astype(bool) & train.entry_success.notna()].copy()
        entry_test_mask = test.entry_candidate_ex_ante.astype(bool)
        if entry_train.entry_success.nunique() < 2:
            raise RuntimeError(f"Insufficient entry classes for historical LightGBM fold {year}")
        clf_params, clf_validation = tune_classifier("LightGBM", entry_train, "entry_success", lag=2, trials=2)
        clf_bundle = fit_classifier("LightGBM", entry_train, "entry_success", clf_params, SEED)
        joblib.dump({"bundle": clf_bundle, "params": clf_params, "prediction_year": year,
                     "train_outcome_max": int(entry_train.outcome_year.max())}, HERE / f"models/lightgbm_clf_oof_{year}.joblib")
        entry_idx = np.flatnonzero(entry_test_mask.to_numpy())
        test_entry = test.iloc[entry_idx]
        x = clf_bundle.preprocessor.transform(test_entry[clf_bundle.numeric_features+clf_bundle.categorical_features]).astype(np.float32)
        probs = clf_bundle.estimator.predict_proba(x)[:, 1]
        diagnostics.append({"year": year, "model": "LightGBM_classifier", "train_n": len(entry_train),
            "train_outcome_max": int(entry_train.outcome_year.max()), "prediction_n": len(test_entry),
            "converged": True, "iterations": clf_params.get("n_estimators"), "parameters": len(clf_params),
            "validation_score": clf_validation, "runtime_seconds": time.perf_counter()-fold_start,
            "specification": "M1 entry classifier fitted only on historically eligible thresholds; selected params on chronological validation"})
        strict_idx_mask = test_entry.strict_inactive_verified.astype(bool).to_numpy()
        strict_rows = test_entry.loc[strict_idx_mask]
        strict_probs = probs[strict_idx_mask]
        if len(strict_rows):
            part = test_base.loc[strict_rows.index].copy()
            part["task"] = "Task_A_new_entry"
            part["candidate_pool_id"] = [_pool_id("Task_A_new_entry", h, y) for h, y in zip(part.hs2, part.outcome_year)]
            part["lightgbm_score"] = strict_probs
            part["lightgbm_prediction_usd"] = np.nan
            part["ppml_score"] = np.nan
            part["ppml_prediction_usd"] = np.nan
            part["prediction_provenance"] = "rolling-origin historical OOF entry probability; model outcomes end at outcome_year-2"
            outputs.append(part)

        # The code is restartable by fold; save after every completed year.
        combined = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
        # Keep one row per candidate-task; score columns are populated by the expert
        # appropriate to that task in a subsequent consolidation step.
        combined = _consolidate(combined)
        combined.to_parquet(outpath, index=False)
        pd.DataFrame(diagnostics).to_csv(diagpath, index=False)
        print(f"OOF year {year} complete: PPML/LGBM fits; {len(combined):,} candidate-task predictions")
    return pd.read_parquet(outpath), pd.read_csv(diagpath)


def _consolidate(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    key = ["task", "candidate_id", "outcome_year"]
    # Regression/PPML and classifier/regression rows can be produced independently.
    # Combine only scores for the same candidate-task key; outcomes/features are invariant.
    invariant = [c for c in rows.columns if c not in {"ppml_score", "ppml_prediction_usd",
        "lightgbm_score", "lightgbm_prediction_usd", "prediction_provenance"}]
    out = rows.groupby(key, as_index=False, sort=False).agg(
        **{c: (c, "first") for c in invariant if c not in key},
        ppml_score=("ppml_score", "mean"), ppml_prediction_usd=("ppml_prediction_usd", "mean"),
        lightgbm_score=("lightgbm_score", "mean"), lightgbm_prediction_usd=("lightgbm_prediction_usd", "mean"),
        prediction_provenance=("prediction_provenance", lambda x: " | ".join(sorted(set(x.astype(str))))))
    return out

