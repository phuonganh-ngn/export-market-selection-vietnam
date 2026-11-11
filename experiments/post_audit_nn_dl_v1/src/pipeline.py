from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
import joblib
import torch

from .data import (HERE, PROJECT, YEARS, LAG, ENTRY_THRESHOLD_USD, SEEDS, KEY,
                   load_frozen_population, load_baselines, write_input_audit,
                   feature_columns)
from .metrics import (discovery_pool_metrics, rank_pool, ndcg_at_k,
                      ndcg_from_gain, capped_export_grade, precision_at_k,
                      hs2_block_bootstrap, destination_block_bootstrap, holm_adjust)
from .models import (MLP_CONFIGS, TABM_CONFIGS, FoldEncoder, fit_with_validation,
                     fit_fixed_epochs, predict_task, seed_everything)

warnings.filterwarnings("ignore", category=FutureWarning)
OUT = HERE
KEYS = ["destination_iso3", "hs2", "outcome_year"]
RUNS = OUT / "predictions/_runs"
TUNING = OUT / "checkpoints/tuning"
MODEL_DIR = OUT / "checkpoints/models"
for directory in [RUNS, TUNING, MODEL_DIR, OUT / "results", OUT / "reports", OUT / "logs"]:
    directory.mkdir(parents=True, exist_ok=True)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def temporal_outer(panel: pd.DataFrame, eval_year: int, task: str, targeted: bool = False):
    train = panel.loc[panel.outcome_observed.astype(bool) & panel.export_value_usd.notna() &
                      panel.outcome_year.le(eval_year - LAG)].copy()
    test = panel.loc[panel.outcome_year.eq(eval_year)].copy()
    if task == "classification":
        train = train.loc[train.entry_candidate_ex_ante & train.entry_success.notna()].copy()
        test = test.loc[test.entry_candidate_ex_ante].copy()
    if train.empty or test.empty:
        raise ValueError(f"Empty {task} outer split for {eval_year}")
    if (train.outcome_year > eval_year - LAG).any():
        raise AssertionError("Outer training outcomes exceed the two-year information cutoff")
    for d in [train, test]:
        if (d.feature_year > d.outcome_year - LAG).any():
            raise AssertionError("Feature year violates two-year lag")
    if test.outcome_year.nunique() != 1 or int(test.outcome_year.iloc[0]) != eval_year:
        raise AssertionError("Outer test year mismatch")
    return train, test


def temporal_inner(train: pd.DataFrame, task: str):
    years = sorted(train.outcome_year.astype(int).unique())
    if len(years) < 3:
        raise ValueError("Insufficient calendar years for chronological inner validation")
    candidate_years = [int(y) for y in years[-2:] if int(y) - LAG >= int(years[0])]
    if candidate_years:
        val_year = candidate_years[-1]
        inner_train = train.loc[train.outcome_year.le(val_year - LAG)].copy()
        inner_val = train.loc[train.outcome_year.eq(val_year)].copy()
    else:
        val_year = years[-1]
        inner_train = train.loc[train.outcome_year.lt(val_year)].copy()
        inner_val = train.loc[train.outcome_year.eq(val_year)].copy()
    if task == "classification":
        inner_train = inner_train.loc[inner_train.entry_candidate_ex_ante & inner_train.entry_success.notna()].copy()
        inner_val = inner_val.loc[inner_val.entry_candidate_ex_ante & inner_val.entry_success.notna()].copy()
    if len(inner_train) < 100 or len(inner_val) < 30:
        raise ValueError(f"Insufficient inner split: train={len(inner_train)}, validation={len(inner_val)}")
    if inner_train.outcome_year.max() > val_year - LAG:
        raise AssertionError("Inner training outcomes violate the lag cutoff")
    return inner_train, inner_val, int(val_year)


def configurations(model_name: str):
    return MLP_CONFIGS if model_name == "MLP" else TABM_CONFIGS


def fit_neural_fold(panel: pd.DataFrame, eval_year: int, model_name: str, task: str,
                    seeds: list[int], feature_set: str = "F4_full", force: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Tune on an earlier calendar fold, then refit on all available outer training outcomes."""
    target_all, test = temporal_outer(panel, eval_year, task)
    target_col = "entry_success" if task == "classification" else "export_value_usd"
    # Fit the chronological tuning protocol once; seed variation is only in final model training.
    tune_path = TUNING / f"{task}_{model_name}_{eval_year}_{feature_set}.json"
    if tune_path.exists() and not force:
        tune_record = json.loads(tune_path.read_text())
    else:
        itrain, ival, val_year = temporal_inner(target_all, task)
        encoder_inner = FoldEncoder.fit(itrain, model_name, feature_set)
        xit, cit = encoder_inner.transform(itrain)
        xiv, civ = encoder_inner.transform(ival)
        yit = itrain[target_col].astype(float).to_numpy()
        yiv = ival[target_col].astype(float).to_numpy()
        if task == "regression":
            yit = np.log1p(yit)
            yiv = np.log1p(yiv)
        cfg_results = []
        for config_idx, config in enumerate(configurations(model_name), start=1):
            fit = fit_with_validation(model_name, task, encoder_inner, xit, cit, yit, xiv, civ, yiv,
                                      config, seed=11, max_epochs=100, patience=12)
            cfg_results.append({"configuration_id": config_idx, "config": config,
                                "validation_score": fit["validation_score"], "best_epoch": fit["best_epoch"],
                                "calibration_factor": fit.get("calibration_factor", np.nan),
                                "epochs_run": fit["epochs_run"], "runtime_seconds": fit["runtime_seconds"],
                                "parameter_count": fit["parameter_count"]})
        if task == "classification":
            best = max(cfg_results, key=lambda x: x["validation_score"])
        else:
            best = min(cfg_results, key=lambda x: x["validation_score"])
        tune_record = {"model": model_name, "task": task, "evaluation_year": int(eval_year),
                       "feature_set": feature_set, "inner_validation_year": val_year,
                       "inner_train_n": len(itrain), "inner_validation_n": len(ival),
                       "best_config": best["config"], "best_epoch": int(best["best_epoch"]),
                       "validation_score": float(best["validation_score"]),
                       "calibration_factor": best.get("calibration_factor", 1.0),
                       "configurations": cfg_results, "max_epochs": 100, "early_stopping_patience": 12}
        tune_path.write_text(json.dumps(tune_record, indent=2, default=float) + "\n")

    best_config = tune_record["best_config"]
    epochs = int(tune_record["best_epoch"])
    calibration = float(tune_record.get("calibration_factor", 1.0) or 1.0)
    encoder = FoldEncoder.fit(target_all, model_name, feature_set)
    xt, ct = encoder.transform(target_all)
    xv, cv = encoder.transform(test)
    y = target_all[target_col].astype(float).to_numpy()
    if task == "regression":
        y = np.log1p(y)
    run_predictions, costs = [], []
    for seed in seeds:
        run_path = RUNS / f"{task}_{model_name}_{eval_year}_{feature_set}_seed{seed}.parquet"
        meta_path = MODEL_DIR / f"{task}_{model_name}_{eval_year}_{feature_set}_seed{seed}.json"
        state_path = MODEL_DIR / f"{task}_{model_name}_{eval_year}_{feature_set}_seed{seed}.pt"
        prep_path = MODEL_DIR / f"{task}_{model_name}_{eval_year}_{feature_set}_seed{seed}_preprocess.joblib"
        if run_path.exists() and not force:
            saved = pd.read_parquet(run_path)
            if len(saved) == len(test) and set(saved.candidate_id) == set(test.candidate_id):
                run_predictions.append(saved)
                if meta_path.exists():
                    costs.append(json.loads(meta_path.read_text()))
                continue
        fit = fit_fixed_epochs(model_name, task, encoder, xt, ct, y, best_config, int(seed), epochs)
        t0 = time.perf_counter()
        prediction, raw = predict_task(fit["model"], model_name, task, xv, cv, calibration)
        inference_seconds = time.perf_counter() - t0
        if task == "classification":
            pred = test[["candidate_id", *KEYS, "entry_success", "label_observed", "entry_candidate_ex_ante"]].copy()
            pred["predicted_probability"] = prediction
            pred["score"] = prediction
        else:
            pred = test[["candidate_id", *KEYS, "export_value_usd", "label_observed", "historical_export_value",
                         "realized_incremental_export_usd", "entry_candidate_ex_ante", "historical_market_share",
                         "destination_import_demand", "log_gdp", "log_destination_import_demand", "lagged_export_growth"]].copy()
            pred["predicted_export_value_usd"] = prediction
            pred["predicted_log1p_export_raw"] = raw
            pred["predicted_export_uncalibrated_usd"] = np.maximum(0, np.expm1(np.clip(raw, 0, 35)))
        pred["model"] = model_name
        pred["task"] = task
        pred["feature_set"] = feature_set
        pred["seed"] = int(seed)
        pred["evaluation_year"] = int(eval_year)
        pred["training_rows"] = len(target_all)
        pred["training_outcome_max"] = int(target_all.outcome_year.max())
        write_parquet(pred, run_path)
        torch.save(fit["model"].state_dict(), state_path)
        joblib.dump(encoder.get_state(), prep_path, compress=3)
        meta = {"model": model_name, "task": task, "seed": seed, "evaluation_year": eval_year,
                "feature_set": feature_set, "config": best_config, "epochs": epochs,
                "training_rows": len(target_all), "training_outcome_max": int(target_all.outcome_year.max()),
                "parameter_count": fit["parameter_count"], "training_runtime_seconds": fit["runtime_seconds"],
                "inference_runtime_seconds": inference_seconds, "calibration_factor": calibration,
                "class_weight": fit.get("class_weight"), "torch_version": torch.__version__}
        meta_path.write_text(json.dumps(meta, indent=2, default=float) + "\n")
        run_predictions.append(pred)
        costs.append(meta)
        del fit
    return pd.concat(run_predictions, ignore_index=True), costs


def fit_logistic(train: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    from .models import FoldEncoder
    train = train.loc[train.entry_candidate_ex_ante & train.entry_success.notna()].copy()
    if train.entry_success.nunique() < 2:
        raise ValueError("Logistic benchmark has fewer than two classes")
    encoder = FoldEncoder.fit(train, "MLP", "F4_full")
    xtr, _ = encoder.transform(train)
    xte, _ = encoder.transform(candidates)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1200, solver="lbfgs", n_jobs=1)
    model.fit(xtr, train.entry_success.astype(int))
    out = candidates[["candidate_id", *KEYS, "entry_success", "label_observed", "entry_candidate_ex_ante"]].copy()
    out["score"] = model.predict_proba(xte)[:, 1]
    out["predicted_probability"] = out.score
    out["model"] = "LogisticRegression"
    out["task"] = "classification"
    out["seed"] = 0
    out["feature_set"] = "F4_full"
    out["training_rows"] = len(train)
    out["training_outcome_max"] = int(train.outcome_year.max())
    return out


def build_e2_predictions(panel: pd.DataFrame, candidates: pd.DataFrame, forecast_base: pd.DataFrame,
                          entry_base: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    files = sorted(RUNS.glob("classification_MLP_*.parquet")) + sorted(RUNS.glob("classification_TabM_*.parquet"))
    neural = [pd.read_parquet(p) for p in files if "_F4_full_" in p.name]
    if len(neural) != 30:
        raise RuntimeError(f"Expected 30 primary E2 neural runs (2 architectures × 3 years × 5 seeds), found {len(neural)}")
    blocks = []
    entry_candidates = candidates.loc[candidates.entry_candidate_ex_ante].copy()
    for year in YEARS:
        cand = entry_candidates.loc[entry_candidates.outcome_year.eq(year)].copy()
        train = panel.loc[panel.outcome_observed.astype(bool) & panel.export_value_usd.notna() &
                          panel.outcome_year.le(year - LAG)].copy()
        train = train.loc[train.entry_candidate_ex_ante & train.entry_success.notna()]
        blocks.append(fit_logistic(train, cand))
        prior = train.groupby("hs2").entry_success.mean()
        hist = cand[["candidate_id", *KEYS, "entry_success", "label_observed", "entry_candidate_ex_ante"]].copy()
        hist["score"] = hist.hs2.map(prior).fillna(float(train.entry_success.mean()))
        hist["predicted_probability"] = hist.score
        hist["model"] = "Historical_entry_frequency"
        hist["task"] = "classification"
        hist["seed"] = 0
        hist["feature_set"] = "historical_only"
        hist["training_rows"] = len(train)
        hist["training_outcome_max"] = int(train.outcome_year.max())
        blocks.append(hist)
        demand = cand[["candidate_id", *KEYS, "entry_success", "label_observed", "entry_candidate_ex_ante", "log_destination_import_demand"]].copy()
        demand["score"] = demand.log_destination_import_demand
        demand["predicted_probability"] = np.nan
        demand["model"] = "World_import_demand"
        demand["task"] = "ranking_score"
        demand["seed"] = 0
        demand["feature_set"] = "single_indicator"
        blocks.append(demand)
        ppml = forecast_base.loc[forecast_base.model.eq("Gravity_PPML") & forecast_base.outcome_year.eq(year),
                                 [*KEYS, "predicted_usd"]].copy()
        ppml = cand.merge(ppml, on=KEYS, validate="one_to_one")
        ppml["score"] = np.log1p(ppml.predicted_usd.clip(lower=0))
        ppml["predicted_probability"] = np.nan
        ppml["model"] = "Gravity_PPML"
        ppml["task"] = "ranking_score"
        ppml["seed"] = 0
        ppml["feature_set"] = "gravity_export_mean"
        blocks.append(ppml)
    # The audited classifiers are used as saved, seed-specific prediction baselines.
    e2 = entry_base.rename(columns={"score": "predicted_probability"}).copy()
    e2["model"] = e2.model.replace({"RandomForest": "RandomForest", "XGBoost": "XGBoost", "LightGBM": "LightGBM"})
    e2["task"] = "classification"
    e2["feature_set"] = "audit_v2_saved"
    e2 = e2.merge(candidates[[*KEYS, "candidate_id", "label_observed", "entry_candidate_ex_ante"]],
                  on=KEYS, how="inner", validate="many_to_one", suffixes=("", "_current"))
    e2["score"] = e2.predicted_probability
    e2["entry_success"] = e2.entry_success.fillna(e2.get("entry_success_current", np.nan))
    blocks.append(e2)
    for p in neural:
        p["task"] = "classification"
        blocks.append(p)
    result = pd.concat(blocks, ignore_index=True, sort=False)
    result = result.loc[result.entry_candidate_ex_ante.astype(bool)].copy()
    # A model-seed-candidate record is unique; any failure to predict is an error, not a silent drop.
    key = ["model", "seed", *KEYS]
    if result.duplicated(key).any():
        dups = result.loc[result.duplicated(key, keep=False), key].head().to_dict("records")
        raise AssertionError(f"Duplicate E2 prediction keys: {dups}")
    expected = set(map(tuple, entry_candidates[KEYS].itertuples(index=False, name=None)))
    for (model, seed, year), g in result.groupby(["model", "seed", "outcome_year"]):
        keys = set(map(tuple, g[KEYS].itertuples(index=False, name=None)))
        expected_year = set(map(tuple, entry_candidates.loc[entry_candidates.outcome_year.eq(year), KEYS].itertuples(index=False, name=None)))
        if keys != expected_year:
            raise AssertionError(f"E2 population mismatch {model}/{seed}/{year}: {len(keys)} vs {len(expected_year)}")
    write_parquet(result, OUT / "predictions/E2_neural_predictions.parquet")
    return result


def e2_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    observed = predictions.loc[predictions.label_observed & predictions.entry_success.notna()].copy()
    exante_counts = predictions.groupby(["model", "seed", "outcome_year", "hs2"]).candidate_id.nunique()
    for (model, seed, year, hs2), pool in observed.groupby(["model", "seed", "outcome_year", "hs2"], sort=False):
        # Stable deterministic ISO3 tie order for Top-K metrics.
        pool = pool.sort_values("destination_iso3", kind="mergesort")
        metrics = discovery_pool_metrics(pool.entry_success.astype(int), pool.score.astype(float), 10)
        rows.append({"model": model, "seed": int(seed), "evaluation_year": int(year), "hs2": hs2,
                     **metrics, "eligible_candidate_count_exante": int(exante_counts.loc[(model, seed, year, hs2)])})
    pool_metrics = pd.DataFrame(rows)
    metric_cols = ["pr_auc", "roc_auc", "precision_at_5", "precision_at_10", "recall_at_10", "ndcg_at_10", "prevalence"]
    overall = pool_metrics.groupby("model", as_index=False).agg(
        **{m: (m, "mean") for m in metric_cols},
        **{f"{m}_pool_sd": (m, "std") for m in metric_cols},
        mean_observed_candidates=("n_observed", "mean"), seeds=("seed", "nunique"),
        valid_pr_auc_pools=("pr_auc", "count"), valid_roc_auc_pools=("roc_auc", "count"),
        valid_ndcg_at_10_pools=("ndcg_at_10", "count"), valid_precision_at_10_pools=("precision_at_10", "count"))
    unique_pool_counts = pool_metrics.drop_duplicates(["model", "evaluation_year", "hs2"]).groupby("model").size()
    overall["valid_pools"] = overall.model.map(unique_pool_counts).astype(int)
    unique_metric = pool_metrics.drop_duplicates(["model", "evaluation_year", "hs2"])
    for metric, column in [("pr_auc", "valid_pr_auc_pools"), ("roc_auc", "valid_roc_auc_pools"),
                           ("ndcg_at_10", "valid_ndcg_at_10_pools"), ("precision_at_10", "valid_precision_at_10_pools")]:
        counts = unique_metric.loc[unique_metric[metric].notna()].groupby("model").size()
        overall[column] = overall.model.map(counts).fillna(0).astype(int)
    unique_events = observed.drop_duplicates(["model", "outcome_year", "hs2", "destination_iso3"]).groupby("model").entry_success.sum()
    overall["positive_events"] = overall.model.map(unique_events).fillna(0).astype(int)
    unique_candidate_n = observed.drop_duplicates(["model", "outcome_year", "hs2", "destination_iso3"]).groupby("model").size()
    overall["observed_candidates"] = overall.model.map(unique_candidate_n).fillna(0).astype(int)
    overall["pooled_prevalence"] = overall.positive_events / overall.observed_candidates.replace(0, np.nan)
    overall = overall.rename(columns={"prevalence": "macro_pool_prevalence"})
    by_year = pool_metrics.groupby(["model", "evaluation_year"], as_index=False).agg(
        **{m: (m, "mean") for m in metric_cols}, mean_observed_candidates_per_pool=("n_observed", "mean"),
        seeds=("seed", "nunique"), valid_pr_auc_pools=("pr_auc", "count"),
        valid_roc_auc_pools=("roc_auc", "count"), valid_ndcg_at_10_pools=("ndcg_at_10", "count"),
        valid_precision_at_10_pools=("precision_at_10", "count"))
    year_pool_counts = pool_metrics.drop_duplicates(["model", "evaluation_year", "hs2"]).groupby(["model", "evaluation_year"]).size()
    by_year["pools"] = [int(year_pool_counts.loc[(r.model, r.evaluation_year)]) for r in by_year.itertuples(index=False)]
    unique_year_metric = pool_metrics.drop_duplicates(["model", "evaluation_year", "hs2"])
    for metric, column in [("pr_auc", "valid_pr_auc_pools"), ("roc_auc", "valid_roc_auc_pools"),
                           ("ndcg_at_10", "valid_ndcg_at_10_pools"), ("precision_at_10", "valid_precision_at_10_pools")]:
        counts = unique_year_metric.loc[unique_year_metric[metric].notna()].groupby(["model", "evaluation_year"]).size()
        by_year[column] = [int(counts.get((r.model, r.evaluation_year), 0)) for r in by_year.itertuples(index=False)]
    unique_by_year = observed.drop_duplicates(["model", "outcome_year", "hs2", "destination_iso3"]).groupby(["model", "outcome_year"])
    observed_count = unique_by_year.size()
    event_count = unique_by_year.entry_success.sum()
    by_year["observed_candidates"] = [int(observed_count.loc[(r.model, r.evaluation_year)]) for r in by_year.itertuples(index=False)]
    by_year["positive_events"] = [int(event_count.loc[(r.model, r.evaluation_year)]) for r in by_year.itertuples(index=False)]
    by_year["pooled_prevalence"] = by_year.positive_events / by_year.observed_candidates.replace(0, np.nan)
    by_year = by_year.rename(columns={"prevalence": "macro_pool_prevalence"})
    by_hs2 = pool_metrics.groupby(["model", "hs2"], as_index=False).agg(
        **{m: (m, "mean") for m in metric_cols}, pools=("evaluation_year", "nunique"), seeds=("seed", "nunique"),
        valid_pr_auc_pools=("pr_auc", "count"), valid_roc_auc_pools=("roc_auc", "count"),
        valid_ndcg_at_10_pools=("ndcg_at_10", "count"), valid_precision_at_10_pools=("precision_at_10", "count"))
    unique_hs_metric = pool_metrics.drop_duplicates(["model", "evaluation_year", "hs2"])
    for metric, column in [("pr_auc", "valid_pr_auc_pools"), ("roc_auc", "valid_roc_auc_pools"),
                           ("ndcg_at_10", "valid_ndcg_at_10_pools"), ("precision_at_10", "valid_precision_at_10_pools")]:
        counts = unique_hs_metric.loc[unique_hs_metric[metric].notna()].groupby(["model", "hs2"]).size()
        by_hs2[column] = [int(counts.get((r.model, r.hs2), 0)) for r in by_hs2.itertuples(index=False)]
    unique_by_hs2 = observed.drop_duplicates(["model", "outcome_year", "hs2", "destination_iso3"]).groupby(["model", "hs2"])
    hs_count, hs_events = unique_by_hs2.size(), unique_by_hs2.entry_success.sum()
    by_hs2["observed_candidates"] = [int(hs_count.loc[(r.model, r.hs2)]) for r in by_hs2.itertuples(index=False)]
    by_hs2["positive_events"] = [int(hs_events.loc[(r.model, r.hs2)]) for r in by_hs2.itertuples(index=False)]
    by_hs2["pooled_prevalence"] = by_hs2.positive_events / by_hs2.observed_candidates.replace(0, np.nan)
    by_hs2 = by_hs2.rename(columns={"prevalence": "macro_pool_prevalence"})
    seed_stability = pool_metrics.groupby(["model", "seed"], as_index=False).agg(
        **{m: (m, "mean") for m in metric_cols}, valid_pools=("hs2", "size"))
    stability_rows = []
    for model, d in seed_stability.groupby("model"):
        for metric in metric_cols:
            values = d[metric].dropna().to_numpy(float)
            stability_rows.append({"task": "E2_discovery", "model": model, "metric": metric,
                                   "mean_across_seeds": float(np.mean(values)) if len(values) else np.nan,
                                   "sd_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                                   "min_seed": float(np.min(values)) if len(values) else np.nan,
                                   "max_seed": float(np.max(values)) if len(values) else np.nan,
                                   "n_seed_values": len(values)})
    for name, frame in [("E2_neural_discovery.csv", overall), ("E2_discovery_by_hs2.csv", by_hs2),
                        ("E2_discovery_by_year.csv", by_year)]:
        frame.to_csv(OUT / "results" / name, index=False, na_rep="")
    pool_metrics.to_csv(OUT / "results/E2_discovery_pool_metrics.csv", index=False, na_rep="")
    seed_stability.to_csv(OUT / "results/E2_discovery_seed_metrics.csv", index=False, na_rep="")
    pd.DataFrame(stability_rows).to_csv(OUT / "results/seed_stability_discovery.csv", index=False, na_rep="")
    return overall, by_hs2, by_year, pool_metrics


def paired_pool_tests(pool_metrics: pd.DataFrame, experiment: str, metric: str, pairs: list[tuple[str, str]],
                      reps: int = 5000) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        av = pool_metrics.loc[pool_metrics.model.eq(a), ["hs2", "evaluation_year", "seed", metric]].copy()
        bv = pool_metrics.loc[pool_metrics.model.eq(b), ["hs2", "evaluation_year", "seed", metric]].copy()
        av = av.groupby(["hs2", "evaluation_year"], as_index=False)[metric].mean().rename(columns={metric: "a"})
        bv = bv.groupby(["hs2", "evaluation_year"], as_index=False)[metric].mean().rename(columns={metric: "b"})
        d = av.merge(bv, on=["hs2", "evaluation_year"], how="inner").dropna()
        if d.empty:
            rows.append({"experiment": experiment, "metric": metric, "model_a": a, "model_b": b,
                         "difference_a_minus_b": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
                         "p_raw": np.nan, "valid_hs2_blocks": 0, "valid_paired_pools": 0})
            continue
        d["difference"] = d.a - d.b
        lo, hi, p, blocks = hs2_block_bootstrap(d.rename(columns={"difference": "value"}), "value", reps=reps,
                                                seed=1000 + len(rows))
        rows.append({"experiment": experiment, "metric": metric, "model_a": a, "model_b": b,
                     "difference_a_minus_b": float(d.difference.mean()), "ci95_low": lo, "ci95_high": hi,
                     "p_raw": p, "valid_hs2_blocks": blocks, "valid_paired_pools": len(d),
                     "direction": f"positive favors {a}; negative favors {b}",
                     "inference": "HS2-block bootstrap CI; paired cluster sign-flip p-value; model seeds averaged within pool"})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out.p_raw.to_numpy())
    return out


def run_discovery(panel: pd.DataFrame, candidates: pd.DataFrame, forecast_base: pd.DataFrame,
                  entry_base: pd.DataFrame, years=YEARS, seeds=SEEDS, force=False):
    for year in years:
        for model in ["MLP", "TabM"]:
            fit_neural_fold(panel, int(year), model, "classification", list(seeds), force=force)
    pred_path = OUT / "predictions/E2_neural_predictions.parquet"
    if pred_path.exists() and not force:
        pred = pd.read_parquet(pred_path)
        required_models = {"MLP", "TabM", "LightGBM", "RandomForest", "XGBoost", "Gravity_PPML",
                           "LogisticRegression", "Historical_entry_frequency", "World_import_demand"}
        if not required_models.issubset(set(pred.model.unique())):
            pred = build_e2_predictions(panel, candidates, forecast_base, entry_base, force)
        else:
            expected_keys = set(map(tuple, candidates.loc[candidates.entry_candidate_ex_ante, KEYS].itertuples(index=False, name=None)))
            for (model_name, seed, year), group in pred.groupby(["model", "seed", "outcome_year"]):
                want = set(map(tuple, candidates.loc[candidates.entry_candidate_ex_ante & candidates.outcome_year.eq(year), KEYS].itertuples(index=False, name=None)))
                have = set(map(tuple, group[KEYS].itertuples(index=False, name=None)))
                if have != want:
                    raise AssertionError(f"Cached E2 predictions are incomplete for {model_name}/{seed}/{year}")
    else:
        pred = build_e2_predictions(panel, candidates, forecast_base, entry_base, force)
    overall, by_hs2, by_year, pool = e2_metrics(pred)
    pairs = [("MLP", "LightGBM"), ("MLP", "Gravity_PPML"), ("TabM", "LightGBM"),
             ("TabM", "Gravity_PPML"), ("TabM", "MLP")]
    tests = []
    for metric in ["pr_auc", "precision_at_10", "ndcg_at_10"]:
        tests.append(paired_pool_tests(pool, "E2", metric, pairs))
    test_df = pd.concat(tests, ignore_index=True)
    test_df.to_csv(OUT / "results/statistical_comparisons_E2.csv", index=False, na_rep="")
    return pred, overall, by_hs2, by_year, pool, test_df


def run_regression(panel: pd.DataFrame, years=YEARS, seeds=SEEDS, force=False):
    for year in years:
        for model in ["MLP", "TabM"]:
            fit_neural_fold(panel, int(year), model, "regression", list(seeds), force=force)
    return load_neural_runs("regression", "F4_full")


def load_neural_runs(task: str, feature_set: str = "F4_full") -> pd.DataFrame:
    files = sorted(RUNS.glob(f"{task}_*_*.parquet"))
    frames = [pd.read_parquet(p) for p in files if f"_{feature_set}_seed" in p.name]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_forecast_predictions(neural_regression: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_parquet(PROJECT / "results/E1_predictions.parquet")
    source["hs2"] = source.hs2.astype(str).str.zfill(2)
    source["evaluation_year"] = source.evaluation_year.astype(int)
    source = source.rename(columns={"evaluation_year": "outcome_year", "actual_usd": "export_value_usd",
                                    "predicted_usd": "predicted_export_value_usd"})
    panel, candidates, _ = load_frozen_population(write=False)
    labels = candidates[[*KEYS, "candidate_id", "label_observed", "export_value_usd", "historical_export_value"]]
    source = source.merge(labels, on=KEYS, how="left", validate="many_to_one", suffixes=("", "_manifest"))
    # Saved outcome values are verified against the frozen processed panel, not taken on trust.
    mismatch = source.label_observed & ~np.isclose(source.export_value_usd, source.export_value_usd_manifest,
                                                   rtol=1e-12, atol=1e-6, equal_nan=True)
    if mismatch.any():
        raise AssertionError(f"Saved baseline actuals mismatch frozen panel: {int(mismatch.sum())}")
    source["model"] = source.model.astype(str)
    source["task"] = "regression"
    source["predicted_log1p_export"] = np.log1p(source.predicted_export_value_usd.clip(lower=0))
    source["feature_set"] = "audit_saved_baseline"
    # Normalize the source actual and history fields to the rebuilt manifest values.
    source["export_value_usd"] = source.export_value_usd_manifest
    source = source.drop(columns=["export_value_usd_manifest"])
    neural = neural_regression.copy()
    if not neural.empty:
        neural["model"] = neural.model.astype(str)
        neural["task"] = "regression"
        neural["seed"] = neural.seed.astype(int)
        # Retain the raw model output and also provide log1p of the final calibrated USD forecast.
        neural["predicted_log1p_export"] = np.log1p(neural.predicted_export_value_usd.clip(lower=0))
        neural["actual_usd"] = neural.export_value_usd
        neural["predicted_usd"] = neural.predicted_export_value_usd
        neural["historical_export_value_usd"] = neural.historical_export_value
    source["actual_usd"] = source.export_value_usd
    source["predicted_usd"] = source.predicted_export_value_usd
    source["historical_export_value_usd"] = source.historical_export_value
    combined = pd.concat([source, neural], ignore_index=True, sort=False)
    # Every baseline prediction must be present for all known candidate keys and all saved seeds.
    observed_keys = set(map(tuple, candidates.loc[candidates.label_observed, KEYS].itertuples(index=False, name=None)))
    for (model, seed, year), group in source.groupby(["model", "seed", "outcome_year"]):
        keys = set(map(tuple, group[KEYS].itertuples(index=False, name=None)))
        expected = set(map(tuple, candidates.loc[candidates.label_observed & candidates.outcome_year.eq(year), KEYS].itertuples(index=False, name=None)))
        if keys != expected:
            raise AssertionError(f"Baseline E1 population differs for {model}/{seed}/{year}: {len(keys)} vs {len(expected)}")
    if not neural.empty:
        for (model, seed, year), group in neural.groupby(["model", "seed", "outcome_year"]):
            expected = set(map(tuple, candidates.loc[candidates.outcome_year.eq(year), KEYS].itertuples(index=False, name=None)))
            keys = set(map(tuple, group[KEYS].itertuples(index=False, name=None)))
            if keys != expected:
                raise AssertionError(f"Neural E1 predictions lost ex-ante candidates: {model}/{seed}/{year}")
    write_parquet(combined, OUT / "predictions/E1_neural_predictions.parquet")
    return combined


def _forecast_metrics(d: pd.DataFrame) -> dict:
    d = d.loc[d.label_observed & d.export_value_usd.notna() & d.predicted_export_value_usd.notna()].copy()
    y = d.export_value_usd.to_numpy(float)
    p = d.predicted_export_value_usd.clip(lower=0).to_numpy(float)
    err = p - y
    denom = float(np.sum(np.abs(y)))
    zero = y == 0
    positive = y > 0
    return {
        "n": len(d), "mae_usd": float(np.mean(np.abs(err))) if len(d) else np.nan,
        "rmse_usd": float(np.sqrt(np.mean(err ** 2))) if len(d) else np.nan,
        "mae_log1p": float(np.mean(np.abs(np.log1p(p) - np.log1p(y)))) if len(d) else np.nan,
        "wape": float(np.sum(np.abs(err)) / denom) if denom > 0 else np.nan,
        "mae_zero_flow_usd": float(np.mean(np.abs(err[zero]))) if zero.any() else np.nan,
        "mae_positive_flow_usd": float(np.mean(np.abs(err[positive]))) if positive.any() else np.nan,
    }


def evaluate_forecasting(predictions: pd.DataFrame):
    known = predictions.loc[predictions.label_observed & predictions.export_value_usd.notna()].copy()
    if known.empty:
        raise RuntimeError("No matched observed outcomes in forecasting table")
    rows = []
    for (model, seed), d in known.groupby(["model", "seed"], sort=False):
        rows.append({"model": model, "seed": int(seed), **_forecast_metrics(d)})
    seed_metrics = pd.DataFrame(rows)
    metrics = ["mae_usd", "rmse_usd", "mae_log1p", "wape", "mae_zero_flow_usd", "mae_positive_flow_usd"]
    overall = seed_metrics.groupby("model", as_index=False).agg(
        **{m: (m, "mean") for m in metrics}, **{f"{m}_seed_sd": (m, "std") for m in metrics},
        n=("n", "max"), seeds=("seed", "nunique"))
    by_year_rows = []
    for (model, seed, year), d in known.groupby(["model", "seed", "outcome_year"], sort=False):
        by_year_rows.append({"model": model, "seed": seed, "evaluation_year": year, **_forecast_metrics(d)})
    by_year_seed = pd.DataFrame(by_year_rows)
    by_year = by_year_seed.groupby(["model", "evaluation_year"], as_index=False).agg(
        **{m: (m, "mean") for m in metrics}, **{f"{m}_seed_sd": (m, "std") for m in metrics},
        n=("n", "max"), seeds=("seed", "nunique"))
    hs_rows = []
    for (model, seed, hs2), d in known.groupby(["model", "seed", "hs2"], sort=False):
        hs_rows.append({"model": model, "seed": seed, "hs2": hs2, **_forecast_metrics(d)})
    hs_seed = pd.DataFrame(hs_rows)
    by_hs2 = hs_seed.groupby(["model", "hs2"], as_index=False).agg(
        **{m: (m, "mean") for m in metrics}, n=("n", "max"), seeds=("seed", "nunique"))
    hs2year_rows = []
    for (model, seed, year, hs2), d in known.groupby(["model", "seed", "outcome_year", "hs2"], sort=False):
        hs2year_rows.append({"model": model, "seed": seed, "evaluation_year": year, "hs2": hs2, **_forecast_metrics(d)})
    hs2_year = pd.DataFrame(hs2year_rows)
    macro = hs2_year.groupby("model", as_index=False).mae_usd.mean().rename(columns={"mae_usd": "macro_hs2_year_mae_usd"})
    overall = overall.merge(macro, on="model", how="left", validate="one_to_one")
    overall.to_csv(OUT / "results/E1_neural_forecasting.csv", index=False, na_rep="")
    by_year.to_csv(OUT / "results/E1_forecasting_by_year.csv", index=False, na_rep="")
    by_hs2.to_csv(OUT / "results/E1_forecasting_by_hs2.csv", index=False, na_rep="")
    hs2_year.to_csv(OUT / "results/E1_forecasting_hs2_year.csv", index=False, na_rep="")
    seed_metrics.to_csv(OUT / "results/seed_stability_forecasting.csv", index=False, na_rep="")
    # Paired destination-block inference: model A minus model B, negative favors A.
    comparisons = []
    for a, b in [("MLP", "Persistence"), ("TabM", "Persistence"),
                 ("MLP", "Gravity_PPML"), ("TabM", "Gravity_PPML")]:
        a_rows = known.loc[known.model.eq(a)].copy()
        b_rows = known.loc[known.model.eq(b)].copy()
        a_rows["absolute_error"] = np.abs(a_rows.predicted_export_value_usd - a_rows.export_value_usd)
        b_rows["absolute_error"] = np.abs(b_rows.predicted_export_value_usd - b_rows.export_value_usd)
        pa = a_rows.groupby(KEYS, as_index=False).absolute_error.mean().rename(columns={"absolute_error": "err_a"})
        pb = b_rows.groupby(KEYS, as_index=False).absolute_error.mean().rename(columns={"absolute_error": "err_b"})
        pair = pa.merge(pb, on=KEYS, validate="one_to_one")
        lo, hi, p, n_dest = destination_block_bootstrap(pair.rename(columns={"destination_iso3": "destination_iso3"}), reps=5000)
        comparisons.append({"comparison": "E1_MAE_USD", "model_a": a, "model_b": b,
                            "difference_a_minus_b": float((pair.err_a - pair.err_b).mean()),
                            "ci95_low": lo, "ci95_high": hi, "p_raw": p, "destination_blocks": n_dest,
                            "p_value_floor_applied": bool(np.isfinite(p) and p <= 1e-16),
                            "paired_observations": len(pair), "direction": f"negative favors {a}"})
    comp = pd.DataFrame(comparisons)
    comp["p_holm"] = holm_adjust(comp.p_raw.to_numpy())
    comp.to_csv(OUT / "results/statistical_comparisons_E1.csv", index=False, na_rep="")
    return overall, by_hs2, by_year, hs2_year, seed_metrics, comp


def _make_task_populations(panel: pd.DataFrame, eval_year: int,
                           history_panel: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    test = panel.loc[panel.outcome_year.eq(eval_year)].copy()
    inactive = test.loc[test.entry_candidate_ex_ante].copy()
    cutoff = eval_year - LAG
    history_source = panel if history_panel is None else history_panel
    history = history_source.loc[history_source.outcome_year.le(cutoff) & history_source.outcome_observed &
                                 history_source.historical_export_value.gt(0) &
                                 history_source.historical_market_share.notna()].copy()
    medians = history.groupby("hs2").historical_market_share.median()
    test["training_median_market_share"] = test.hs2.map(medians)
    existing_under = test.loc[test.historical_export_value.gt(0) & test.training_median_market_share.notna() &
                              test.historical_market_share.le(test.training_median_market_share)].copy()
    existing_all = test.loc[test.historical_export_value.gt(0)].copy()
    return {"Task_A_new_entry": inactive, "Task_B_existing_underexploited": existing_under,
            "Task_B_all_existing_sensitivity": existing_all}


def _impute_indicator_score(panel: pd.DataFrame, pool: pd.DataFrame, year: int, model: str) -> tuple[pd.Series, int]:
    """Fill an unavailable simple-indicator score with a training-only median."""
    score_col = "log_gdp" if model == "GDP" else "log_destination_import_demand"
    missing = pool[score_col].isna()
    if not missing.any():
        return pd.to_numeric(pool[score_col], errors="coerce"), 0
    train = panel.loc[panel.outcome_year.le(year - LAG)]
    if model == "GDP":
        historical = train.drop_duplicates(["destination_iso3", "outcome_year"])[score_col]
    else:
        hs2 = str(pool.hs2.iloc[0]).zfill(2)
        historical = train.loc[train.hs2.astype(str).str.zfill(2).eq(hs2), score_col]
    fallback = pd.to_numeric(historical, errors="coerce").median()
    if not np.isfinite(fallback):
        raise ValueError(f"No training-period median available for {model} in {year}")
    return pd.to_numeric(pool[score_col], errors="coerce").fillna(float(fallback)), int(missing.sum())


def _ndcg_missing_bounds(ordered_full: pd.DataFrame, gain_known: np.ndarray, k: int = 10, max_gain: float = 15.0):
    """Conservative lower bound for capped-gain NDCG; upper bound is one."""
    known_flags = ordered_full.label_observed.astype(bool).to_numpy()
    discounts = 1 / np.log2(np.arange(2, min(k, len(ordered_full)) + 2))
    known_top_gains = np.zeros(len(discounts), dtype=float)
    known_positions = np.flatnonzero(known_flags)[:len(discounts)]
    known_values = gain_known
    # gain_known is aligned to rows with observed labels; align by the ordered subset.
    obs_ordered = ordered_full.loc[known_flags]
    obs_gain = dict(zip(obs_ordered.candidate_id.astype(str), known_values))
    for pos in range(len(discounts)):
        if known_flags[pos]:
            known_top_gains[pos] = obs_gain[str(ordered_full.candidate_id.iloc[pos])]
    dcg_lower = float(np.sum(known_top_gains * discounts))
    known_all = np.asarray(list(obs_gain.values()), dtype=float)
    n_unknown = int((~known_flags).sum())
    ideal_max = np.sort(np.r_[known_all, np.full(n_unknown, max_gain)])[::-1]
    idcg_max = float(np.sum(ideal_max[:k] / np.log2(np.arange(2, min(k, len(ideal_max)) + 2))))
    if idcg_max <= 0:
        return np.nan, np.nan
    lower = dcg_lower / idcg_max
    # Any NDCG in a correctly normalized pool is in [0,1]; keep this valid conservative upper bound.
    return float(lower), 1.0 if n_unknown else float(ndcg_from_gain(
        np.asarray([obs_gain.get(str(cid), 0.0) for cid in ordered_full.candidate_id]),
        np.asarray([obs_gain.get(str(cid), 0.0) for cid in ordered_full.candidate_id]), k))


def _ranking_strategies(panel: pd.DataFrame, forecast_base: pd.DataFrame, neural_regression: pd.DataFrame,
                        e2_predictions: pd.DataFrame) -> pd.DataFrame:
    candidates = panel.loc[panel.outcome_year.isin(YEARS)].copy()
    candidates["candidate_id"] = candidates.apply(lambda r: __import__("hashlib").sha256(
        f"{r.destination_iso3}|{str(r.hs2).zfill(2)}|{int(r.outcome_year)}".encode()).hexdigest()[:20], axis=1)
    candidates["label_observed"] = candidates.outcome_observed.astype(bool) & candidates.export_value_usd.notna()
    candidates["predicted_export_value_usd"] = np.nan
    # Reconstructed audited baselines cover every ex-ante key. Forecast predictions are averaged over seeds in audit_v2.
    base = forecast_base.copy()
    base = base.rename(columns={"predicted_usd": "predicted_export_value_usd"})
    for model in ["Persistence", "Gravity_PPML", "RandomForest", "XGBoost", "LightGBM"]:
        d = base.loc[base.model.eq(model), KEYS + ["predicted_export_value_usd"]]
        candidates = candidates.merge(d.rename(columns={"predicted_export_value_usd": f"pred_{model}"}),
                                      on=KEYS, how="left", validate="one_to_one")
    if neural_regression.empty:
        raise RuntimeError("No neural regression predictions are available for E3")
    for model in ["MLP", "TabM"]:
        d = neural_regression.loc[neural_regression.model.eq(model), KEYS + ["seed", "predicted_export_value_usd"]]
        for seed, g in d.groupby("seed"):
            candidates = candidates.merge(g[KEYS + ["predicted_export_value_usd"]].rename(
                columns={"predicted_export_value_usd": f"pred_{model}_seed{int(seed)}"}), on=KEYS, how="left", validate="one_to_one")
    # Entry probability ranks are kept distinct from regression opportunity scores.
    if not e2_predictions.empty:
        for model in ["MLP", "TabM", "LightGBM", "RandomForest", "XGBoost", "LogisticRegression"]:
            q = e2_predictions.loc[e2_predictions.model.eq(model) & e2_predictions.predicted_probability.notna(),
                                   KEYS + ["seed", "predicted_probability"]]
            for seed, g in q.groupby("seed"):
                candidates = candidates.merge(g[KEYS + ["predicted_probability"]].rename(
                    columns={"predicted_probability": f"prob_{model}_seed{int(seed)}"}), on=KEYS, how="left", validate="one_to_one")
    return candidates


def _ranking_stability_tables(ranking_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.stats import spearmanr

    keys = ["task", "evaluation_year", "hs2", "ranking_model", "seed"]
    top = ranking_frame.loc[ranking_frame.rank_full_exante.le(10)].groupby(keys).destination_iso3.apply(set).to_dict()
    pairs = [("MLP", "Gravity_PPML"), ("TabM", "Gravity_PPML"), ("MLP", "LightGBM"),
             ("TabM", "LightGBM"), ("TabM", "MLP")]
    overlap_rows = []
    pools = ranking_frame[["task", "evaluation_year", "hs2"]].drop_duplicates()
    for task, year, hs2 in pools.itertuples(index=False, name=None):
        for model_a, model_b in pairs:
            seeds_a = SEEDS if model_a in {"MLP", "TabM"} else [0]
            seeds_b = SEEDS if model_b in {"MLP", "TabM"} else [0]
            if model_a == model_b:
                seeds_b = seeds_a
            for i, seed_a in enumerate(seeds_a):
                seed_b = seeds_b[i] if len(seeds_a) == len(seeds_b) else seeds_b[0]
                a = top.get((task, year, hs2, model_a, seed_a), set())
                b = top.get((task, year, hs2, model_b, seed_b), set())
                union = a | b
                overlap_rows.append({"task": task, "evaluation_year": year, "hs2": hs2,
                                     "model_a": model_a, "model_b": model_b, "seed_a": seed_a, "seed_b": seed_b,
                                     "top10_jaccard": len(a & b) / len(union) if union else np.nan,
                                     "top10_common_destinations": len(a & b)})
    overlap = pd.DataFrame(overlap_rows)
    overlap_summary = overlap.groupby(["task", "model_a", "model_b"], as_index=False).agg(
        mean_top10_jaccard=("top10_jaccard", "mean"), median_top10_jaccard=("top10_jaccard", "median"),
        seed_pool_comparisons=("hs2", "size"))
    unique_overlap_pools = overlap.drop_duplicates(["task", "evaluation_year", "hs2", "model_a", "model_b"])
    unique_overlap_counts = unique_overlap_pools.groupby(["task", "model_a", "model_b"]).size()
    overlap_summary["paired_product_year_pools"] = [int(unique_overlap_counts.loc[(r.task, r.model_a, r.model_b)])
                                                      for r in overlap_summary.itertuples(index=False)]
    overlap.to_csv(OUT / "results/E3_top10_overlap_by_pool.csv", index=False, na_rep="")
    overlap_summary.to_csv(OUT / "results/E3_top10_overlap.csv", index=False, na_rep="")

    stability_rows = []
    rank_keys = ["task", "ranking_model", "seed", "hs2", "evaluation_year"]
    year_pairs = [(2022, 2023), (2023, 2024), (2022, 2024)]
    grouped = {key: g for key, g in ranking_frame.groupby(rank_keys, sort=False)}
    models = ranking_frame.ranking_model.unique()
    tasks = ranking_frame.task.unique()
    hs2s = ranking_frame.hs2.unique()
    seeds = ranking_frame.seed.unique()
    for task in tasks:
        for model in models:
            for seed in seeds:
                if not ((ranking_frame.task.eq(task) & ranking_frame.ranking_model.eq(model) & ranking_frame.seed.eq(seed)).any()):
                    continue
                for hs2 in hs2s:
                    for year_a, year_b in year_pairs:
                        a = grouped.get((task, model, seed, hs2, year_a))
                        b = grouped.get((task, model, seed, hs2, year_b))
                        if a is None or b is None:
                            continue
                        common = a[["destination_iso3", "rank_full_exante"]].merge(
                            b[["destination_iso3", "rank_full_exante"]], on="destination_iso3", suffixes=("_a", "_b"))
                        rho = float(spearmanr(common.rank_full_exante_a, common.rank_full_exante_b).statistic) if len(common) >= 10 else np.nan
                        stability_rows.append({"task": task, "ranking_model": model, "seed": seed, "hs2": hs2,
                                               "year_a": year_a, "year_b": year_b, "common_destinations": len(common),
                                               "spearman_rank_stability": rho})
    stability = pd.DataFrame(stability_rows)
    if not stability.empty:
        summary = stability.groupby(["task", "ranking_model", "year_a", "year_b"], as_index=False).agg(
            mean_spearman_rank_stability=("spearman_rank_stability", "mean"),
            valid_hs2_pools=("spearman_rank_stability", "count"), mean_common_destinations=("common_destinations", "mean"),
            seeds=("seed", "nunique"))
    else:
        summary = pd.DataFrame(columns=["task", "ranking_model", "year_a", "year_b", "mean_spearman_rank_stability",
                                       "valid_hs2_pools", "mean_common_destinations", "seeds"])
    stability.to_csv(OUT / "results/E3_year_stability_by_pool.csv", index=False, na_rep="")
    summary.to_csv(OUT / "results/E3_year_stability.csv", index=False, na_rep="")
    return overlap_summary, summary


def _matched_forecast_ranking(metrics_frame: pd.DataFrame, ranking_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare USD forecasting and decision ranking on identical task-specific candidates."""
    from scipy.stats import spearmanr

    forecast_models = ["Persistence", "Gravity_PPML", "RandomForest", "XGBoost", "LightGBM", "MLP", "TabM"]
    rows, tests = [], []
    pairs = [("MLP", "Persistence"), ("TabM", "Persistence"),
             ("MLP", "Gravity_PPML"), ("TabM", "Gravity_PPML")]
    for task in ["Task_A_new_entry", "Task_B_existing_underexploited"]:
        d = ranking_frame.loc[ranking_frame.task.eq(task) & ranking_frame.ranking_model.isin(forecast_models)
                              & ranking_frame.label_observed.astype(bool)
                              & ranking_frame.predicted_export_value_usd.notna()].copy()
        d["absolute_error_usd"] = np.abs(d.predicted_export_value_usd - d.observed_future_export_value_usd)
        seed_mae = d.groupby(["ranking_model", "seed"], as_index=False).absolute_error_usd.mean()
        pool_mae = d.groupby(["ranking_model", "seed", "hs2", "evaluation_year"], as_index=False).absolute_error_usd.mean()
        macro_seed_mae = pool_mae.groupby(["ranking_model", "seed"], as_index=False).absolute_error_usd.mean().rename(
            columns={"absolute_error_usd": "macro_hs2_year_mae_usd"})
        metric = "ndcg10_entry_binary" if task == "Task_A_new_entry" else "ndcg10_capped_grade"
        rank = metrics_frame.loc[metrics_frame.task.eq(task) & metrics_frame.ranking_model.isin(forecast_models)]
        rank_seed = rank.groupby(["ranking_model", "seed"], as_index=False).agg(
            ndcg10=(metric, "mean"), valid_rank_pools=(metric, "count"))
        forecast_summary = seed_mae.groupby("ranking_model", as_index=False).agg(
            forecast_mae_usd=("absolute_error_usd", "mean"), forecast_mae_seed_sd=("absolute_error_usd", "std"),
            seeds=("seed", "nunique"))
        macro_summary = macro_seed_mae.groupby("ranking_model", as_index=False).agg(
            forecast_macro_hs2_year_mae_usd=("macro_hs2_year_mae_usd", "mean"),
            forecast_macro_seed_sd=("macro_hs2_year_mae_usd", "std"))
        forecast_summary = forecast_summary.merge(macro_summary, on="ranking_model", validate="one_to_one")
        rank_summary = rank_seed.groupby("ranking_model", as_index=False).agg(
            ndcg10=("ndcg10", "mean"), ndcg10_seed_sd=("ndcg10", "std"),
            valid_rank_pools=("valid_rank_pools", "max"), ranking_seeds=("seed", "nunique"))
        summary = forecast_summary.merge(rank_summary, on="ranking_model", how="inner", validate="one_to_one")
        summary["task"] = task
        summary["n_observed_candidates"] = int(d.drop_duplicates(["destination_iso3", "hs2", "evaluation_year"]).shape[0])
        summary["forecast_metric"] = "MAE USD; lower is better"
        summary["ranking_metric"] = f"{metric}; higher is better"
        if len(summary) >= 3:
            summary["across_model_spearman_mae_vs_ndcg"] = float(spearmanr(
                summary.forecast_mae_usd, summary.ndcg10).statistic)
        else:
            summary["across_model_spearman_mae_vs_ndcg"] = np.nan
        rows.extend(summary.to_dict("records"))

        for model_a, model_b in pairs:
            a = d.loc[d.ranking_model.eq(model_a)].groupby(
                ["destination_iso3", "hs2", "evaluation_year"], as_index=False).absolute_error_usd.mean().rename(columns={"absolute_error_usd": "err_a"})
            b = d.loc[d.ranking_model.eq(model_b)].groupby(
                ["destination_iso3", "hs2", "evaluation_year"], as_index=False).absolute_error_usd.mean().rename(columns={"absolute_error_usd": "err_b"})
            pair = a.merge(b, on=["destination_iso3", "hs2", "evaluation_year"], validate="one_to_one")
            lo, hi, p, blocks = destination_block_bootstrap(pair, reps=5000, seed=2400 + len(tests))
            tests.append({"task": task, "metric": "MAE_USD", "model_a": model_a, "model_b": model_b,
                          "difference_a_minus_b": float((pair.err_a - pair.err_b).mean()),
                          "ci95_low": lo, "ci95_high": hi, "p_raw": p,
                          "p_value_floor_applied": bool(np.isfinite(p) and p <= 1e-16),
                          "destination_blocks": blocks, "paired_observations": len(pair),
                          "direction": f"negative favors {model_a}",
                          "inference": "same task-specific candidate population; destination-block bootstrap; paired destination Wilcoxon"})
    summary_frame = pd.DataFrame(rows)
    test_frame = pd.DataFrame(tests)
    if not test_frame.empty:
        test_frame["p_holm"] = np.nan
        for _, ix in test_frame.groupby("task").groups.items():
            test_frame.loc[ix, "p_holm"] = holm_adjust(test_frame.loc[ix, "p_raw"].to_numpy())
    summary_frame.to_csv(OUT / "results/E3_matched_forecast_ranking.csv", index=False, na_rep="")
    test_frame.to_csv(OUT / "results/statistical_comparisons_E3_matched_mae.csv", index=False, na_rep="")
    return summary_frame, test_frame


def evaluate_rankings(panel: pd.DataFrame, forecast_base: pd.DataFrame, neural_regression: pd.DataFrame,
                      e2_predictions: pd.DataFrame):
    candidates = _ranking_strategies(panel, forecast_base, neural_regression, e2_predictions)
    ranked_rows, metric_rows = [], []
    strategy_defs = [
        ("Persistence", "pred_Persistence", 0, True), ("Gravity_PPML", "pred_Gravity_PPML", 0, True),
        ("RandomForest", "pred_RandomForest", 0, True), ("XGBoost", "pred_XGBoost", 0, True),
        ("LightGBM", "pred_LightGBM", 0, True), ("GDP", "log_gdp", 0, False),
        ("Historical_import_demand", "log_destination_import_demand", 0, False),
        ("Historical_export_trend", "lagged_export_growth", 0, False),
    ]
    strategy_defs += [(m, f"pred_{m}_seed{s}", s, True) for m in ["MLP", "TabM"] for s in SEEDS]
    strategy_defs += [(f"{m}_entry_probability", f"prob_{m}_seed{s}", s, False)
                      for m in ["MLP", "TabM"] for s in SEEDS]
    for year in YEARS:
        pops = _make_task_populations(candidates, year, history_panel=panel)
        for task, task_pool in pops.items():
            # Ranking and Top-K metrics are defined within each HS2 × year pool.
            for _, pool_base in task_pool.groupby("hs2", sort=False):
                if pool_base.empty:
                    continue
                for output_model, score_col, seed, is_export_prediction in strategy_defs:
                        if output_model.endswith("_entry_probability") and task != "Task_A_new_entry":
                            continue
                        if score_col not in pool_base:
                            continue
                        pool = pool_base.copy()
                        score_imputed = 0
                        if output_model in {"GDP", "Historical_import_demand"}:
                            pool["score"], score_imputed = _impute_indicator_score(panel, pool, year, output_model)
                        else:
                            pool["score"] = pd.to_numeric(pool[score_col], errors="coerce")
                        if output_model == "Historical_export_trend":
                            # Undefined growth for a zero/near-zero history is a neutral no-trend score.
                            pool["score"] = pool.score.fillna(0.0)
                        elif pool.score.isna().any():
                            raise AssertionError(f"Model/indicator {output_model} has {int(pool.score.isna().sum())} missing candidate scores in {task}/{year}")
                        if output_model == "Persistence":
                            # Persistence predicts no incremental growth; score ties are ordered by ISO3.
                            pool["opportunity_score"] = 0.0
                        elif is_export_prediction:
                            pool["opportunity_score"] = np.maximum(0.0, pool.score - pool.historical_export_value.clip(lower=0))
                        else:
                            pool["opportunity_score"] = pool.score
                        ranked_full = rank_pool(pool, "opportunity_score")
                        ranked_full["rank_full"] = np.arange(1, len(ranked_full) + 1)
                        ranked_known = rank_pool(pool.loc[pool.label_observed], "opportunity_score")
                        ranked_known["rank_known_outcome"] = np.arange(1, len(ranked_known) + 1)
                        increments = ranked_known.realized_incremental_export_usd.to_numpy(float)
                        grade = capped_export_grade(increments, ENTRY_THRESHOLD_USD)
                        log_gain = np.log1p(np.maximum(0, increments))
                        raw_gain = np.maximum(0, increments)
                        exp_binary = (increments >= ENTRY_THRESHOLD_USD).astype(float)
                        entry_binary = ranked_known.entry_success.astype(float).to_numpy()
                        top_full = ranked_full.head(10)
                        kfull = min(10, len(ranked_full))
                        known_success_top = int(top_full.loc[top_full.label_observed, "entry_success"].fillna(0).sum())
                        missing_top = int((~top_full.label_observed.astype(bool)).sum())
                        known_inc_top = float(top_full.loc[top_full.label_observed, "realized_incremental_export_usd"].sum())
                        bounds_low, bounds_high = _ndcg_missing_bounds(ranked_full, grade, 10, 15.0)
                        bounds_entry_low, bounds_entry_high = _ndcg_missing_bounds(ranked_full, entry_binary, 10, 1.0)
                        entry_top_lower = known_success_top / kfull if kfull else np.nan
                        entry_top_upper = (known_success_top + missing_top) / kfull if kfull else np.nan
                        dmetrics = {
                            "n_exante": len(pool), "n_observed": int(pool.label_observed.sum()),
                            "n_missing": int((~pool.label_observed).sum()),
                            "score_imputed_candidates": int(score_imputed),
                            "ndcg10_capped_grade": ndcg_at_k(grade, 10),
                            "ndcg5_capped_grade": ndcg_at_k(grade, 5),
                            "ndcg10_log1p_gain": ndcg_from_gain(log_gain, log_gain, 10),
                            "ndcg5_log1p_gain": ndcg_from_gain(log_gain, log_gain, 5),
                            "ndcg10_raw_gain": ndcg_from_gain(raw_gain, raw_gain, 10),
                            "ndcg5_raw_gain": ndcg_from_gain(raw_gain, raw_gain, 5),
                            "ndcg10_expansion_binary": ndcg_at_k(exp_binary, 10),
                            "ndcg5_expansion_binary": ndcg_at_k(exp_binary, 5),
                            "ndcg10_entry_binary": ndcg_at_k(entry_binary, 10) if task == "Task_A_new_entry" else np.nan,
                            "ndcg5_entry_binary": ndcg_at_k(entry_binary, 5) if task == "Task_A_new_entry" else np.nan,
                            "precision5_expansion": precision_at_k(exp_binary, 5),
                            "precision10_expansion": precision_at_k(exp_binary, 10),
                            "precision5_entry": precision_at_k(entry_binary, 5) if task == "Task_A_new_entry" else np.nan,
                            "precision10_entry": precision_at_k(entry_binary, 10) if task == "Task_A_new_entry" else np.nan,
                            "observed_incremental_usd_top10_full_rank_partial": known_inc_top,
                            "known_entry_success_top10_full_rank": known_success_top,
                            "entry_success_rate_top10_full_rank_lower_bound": entry_top_lower,
                            "entry_success_rate_top10_full_rank_upper_bound": entry_top_upper,
                            "unknown_outcomes_top10": missing_top,
                            "full_population_ndcg10_lower_bound_capped_gain": bounds_low,
                            "full_population_ndcg10_upper_bound_capped_gain": bounds_high,
                            "full_population_ndcg10_lower_bound_entry_binary": bounds_entry_low,
                            "full_population_ndcg10_upper_bound_entry_binary": bounds_entry_high,
                        }
                        metric_rows.append({"ranking_model": output_model, "seed": seed, "task": task,
                                            "evaluation_year": year, "hs2": str(pool.hs2.iloc[0]), **dmetrics})
                        # Save all ex-ante ranks plus actual rank in the observed-label intersection.
                        known_rank_lookup = ranked_known.set_index("candidate_id").rank_known_outcome.to_dict()
                        for r in ranked_full.itertuples(index=False):
                            ranked_rows.append({"ranking_model": output_model, "seed": seed, "task": task,
                                "evaluation_year": year, "hs2": r.hs2, "destination_iso3": r.destination_iso3,
                                "candidate_id": r.candidate_id, "rank_full_exante": r.rank_full,
                                "rank_known_outcome_intersection": known_rank_lookup.get(r.candidate_id, np.nan),
                                "opportunity_score": r.opportunity_score,
                                "predicted_export_value_usd": getattr(r, score_col) if is_export_prediction else np.nan,
                                "historical_export_value_usd": r.historical_export_value,
                                "historical_import_demand_usd": r.destination_import_demand,
                                "historical_vietnam_market_share": r.historical_market_share,
                                "observed_future_export_value_usd": r.export_value_usd,
                                "realized_incremental_export_usd": r.realized_incremental_export_usd,
                                "entry_success": r.entry_success, "label_observed": r.label_observed,
                                "eligibility_status": "ex-ante eligible; outcome unknown remains NA"})
    ranking_frame = pd.DataFrame(ranked_rows)
    metrics_frame = pd.DataFrame(metric_rows)
    forecast_rankers = ["Persistence", "Gravity_PPML", "RandomForest", "XGBoost", "LightGBM", "MLP", "TabM"]
    for task in ["Task_A_new_entry", "Task_B_existing_underexploited"]:
        block = ranking_frame.loc[ranking_frame.task.eq(task) & ranking_frame.ranking_model.isin(forecast_rankers)]
        reference = None
        for (model, model_seed), group in block.groupby(["ranking_model", "seed"]):
            keys = set(group.candidate_id.astype(str))
            if reference is None:
                reference = keys
            elif keys != reference:
                raise AssertionError(f"Forecast/ranking model candidate sets differ in {task}: {model}/{model_seed}")
            if group.predicted_export_value_usd.isna().any():
                raise AssertionError(f"Forecast/ranking model {model}/{model_seed} lacks candidate predictions in {task}")
    write_parquet(ranking_frame, OUT / "predictions/E3_neural_rankings.parquet")
    _matched_forecast_ranking(metrics_frame, ranking_frame)
    _ranking_stability_tables(ranking_frame)
    # Macro estimates average within candidate pools and years; avoid pooled cross-product ranking.
    metric_names = ["ndcg10_capped_grade", "ndcg5_capped_grade", "ndcg10_log1p_gain", "ndcg5_log1p_gain",
                    "ndcg10_raw_gain", "ndcg5_raw_gain", "ndcg10_expansion_binary", "ndcg5_expansion_binary",
                    "ndcg10_entry_binary", "ndcg5_entry_binary", "precision5_expansion", "precision10_expansion",
                    "precision5_entry", "precision10_entry",
                    "observed_incremental_usd_top10_full_rank_partial",
                    "entry_success_rate_top10_full_rank_lower_bound", "entry_success_rate_top10_full_rank_upper_bound",
                    "full_population_ndcg10_lower_bound_capped_gain", "full_population_ndcg10_upper_bound_capped_gain",
                    "full_population_ndcg10_lower_bound_entry_binary", "full_population_ndcg10_upper_bound_entry_binary"]
    overall = metrics_frame.groupby(["ranking_model", "task"], as_index=False).agg(
        **{m: (m, "mean") for m in metric_names},
        mean_exante_candidates=("n_exante", "mean"), mean_observed_candidates=("n_observed", "mean"),
        mean_score_imputed_candidates=("score_imputed_candidates", "mean"),
        unknown_outcomes_top10_model_seed_total=("unknown_outcomes_top10", "sum"), seeds=("seed", "nunique"),
        valid_ndcg10_capped_grade_pools=("ndcg10_capped_grade", "count"),
        valid_ndcg10_entry_binary_pools=("ndcg10_entry_binary", "count"),
        valid_ndcg10_log1p_gain_pools=("ndcg10_log1p_gain", "count"),
        valid_ndcg10_raw_gain_pools=("ndcg10_raw_gain", "count"),
        valid_ndcg5_capped_grade_pools=("ndcg5_capped_grade", "count"),
        valid_ndcg5_entry_binary_pools=("ndcg5_entry_binary", "count"),
        valid_ndcg5_log1p_gain_pools=("ndcg5_log1p_gain", "count"),
        valid_ndcg5_raw_gain_pools=("ndcg5_raw_gain", "count"),
        valid_ndcg5_expansion_binary_pools=("ndcg5_expansion_binary", "count"))
    unique_task_pools = metrics_frame.drop_duplicates(["ranking_model", "task", "evaluation_year", "hs2"])
    for metric, column in [("ndcg10_capped_grade", "valid_ndcg10_capped_grade_pools"),
                           ("ndcg10_entry_binary", "valid_ndcg10_entry_binary_pools"),
                           ("ndcg10_log1p_gain", "valid_ndcg10_log1p_gain_pools"),
                           ("ndcg10_raw_gain", "valid_ndcg10_raw_gain_pools"),
                           ("ndcg5_capped_grade", "valid_ndcg5_capped_grade_pools"),
                           ("ndcg5_entry_binary", "valid_ndcg5_entry_binary_pools"),
                           ("ndcg5_log1p_gain", "valid_ndcg5_log1p_gain_pools"),
                           ("ndcg5_raw_gain", "valid_ndcg5_raw_gain_pools"),
                           ("ndcg5_expansion_binary", "valid_ndcg5_expansion_binary_pools")]:
        counts = unique_task_pools.loc[unique_task_pools[metric].notna()].groupby(["ranking_model", "task"]).size()
        overall[column] = [int(counts.get((r.ranking_model, r.task), 0)) for r in overall.itertuples(index=False)]
    pool_counts = unique_task_pools.groupby(["ranking_model", "task"]).size()
    unknown_counts = unique_task_pools.groupby(["ranking_model", "task"]).n_missing.sum()
    overall["valid_product_year_pools"] = [int(pool_counts.loc[(r.ranking_model, r.task)]) for r in overall.itertuples(index=False)]
    overall["unique_unobserved_candidate_labels"] = [int(unknown_counts.loc[(r.ranking_model, r.task)]) for r in overall.itertuples(index=False)]
    by_year = metrics_frame.groupby(["ranking_model", "task", "evaluation_year"], as_index=False).agg(
        **{m: (m, "mean") for m in metric_names},
        mean_exante_candidates=("n_exante", "mean"), mean_score_imputed_candidates=("score_imputed_candidates", "mean"),
        unknown_outcomes_top10_model_seed_total=("unknown_outcomes_top10", "sum"),
        seeds=("seed", "nunique"), valid_ndcg10_capped_grade_pools=("ndcg10_capped_grade", "count"),
        valid_ndcg10_entry_binary_pools=("ndcg10_entry_binary", "count"),
        valid_ndcg10_log1p_gain_pools=("ndcg10_log1p_gain", "count"), valid_ndcg10_raw_gain_pools=("ndcg10_raw_gain", "count"),
        valid_ndcg5_capped_grade_pools=("ndcg5_capped_grade", "count"), valid_ndcg5_entry_binary_pools=("ndcg5_entry_binary", "count"),
        valid_ndcg5_log1p_gain_pools=("ndcg5_log1p_gain", "count"), valid_ndcg5_raw_gain_pools=("ndcg5_raw_gain", "count"),
        valid_ndcg5_expansion_binary_pools=("ndcg5_expansion_binary", "count"))
    unique_year_pools = metrics_frame.drop_duplicates(["ranking_model", "task", "evaluation_year", "hs2"])
    for metric, column in [("ndcg10_capped_grade", "valid_ndcg10_capped_grade_pools"),
                           ("ndcg10_entry_binary", "valid_ndcg10_entry_binary_pools"),
                           ("ndcg10_log1p_gain", "valid_ndcg10_log1p_gain_pools"),
                           ("ndcg10_raw_gain", "valid_ndcg10_raw_gain_pools"),
                           ("ndcg5_capped_grade", "valid_ndcg5_capped_grade_pools"),
                           ("ndcg5_entry_binary", "valid_ndcg5_entry_binary_pools"),
                           ("ndcg5_log1p_gain", "valid_ndcg5_log1p_gain_pools"),
                           ("ndcg5_raw_gain", "valid_ndcg5_raw_gain_pools"),
                           ("ndcg5_expansion_binary", "valid_ndcg5_expansion_binary_pools")]:
        counts = unique_year_pools.loc[unique_year_pools[metric].notna()].groupby(
            ["ranking_model", "task", "evaluation_year"]).size()
        by_year[column] = [int(counts.get((r.ranking_model, r.task, r.evaluation_year), 0))
                           for r in by_year.itertuples(index=False)]
    year_pool_n = unique_year_pools.groupby(["ranking_model", "task", "evaluation_year"]).size()
    year_unknown = unique_year_pools.groupby(["ranking_model", "task", "evaluation_year"]).n_missing.sum()
    by_year["valid_hs2_pools"] = [int(year_pool_n.loc[(r.ranking_model, r.task, r.evaluation_year)]) for r in by_year.itertuples(index=False)]
    by_year["unique_unobserved_candidate_labels"] = [int(year_unknown.loc[(r.ranking_model, r.task, r.evaluation_year)]) for r in by_year.itertuples(index=False)]
    by_hs2 = metrics_frame.groupby(["ranking_model", "task", "hs2"], as_index=False).agg(
        **{m: (m, "mean") for m in metric_names}, pools=("evaluation_year", "nunique"),
        mean_exante_candidates=("n_exante", "mean"), mean_score_imputed_candidates=("score_imputed_candidates", "mean"),
        seeds=("seed", "nunique"),
        valid_ndcg10_capped_grade_pools=("ndcg10_capped_grade", "count"),
        valid_ndcg10_entry_binary_pools=("ndcg10_entry_binary", "count"),
        valid_ndcg10_log1p_gain_pools=("ndcg10_log1p_gain", "count"), valid_ndcg10_raw_gain_pools=("ndcg10_raw_gain", "count"),
        valid_ndcg5_capped_grade_pools=("ndcg5_capped_grade", "count"), valid_ndcg5_entry_binary_pools=("ndcg5_entry_binary", "count"),
        valid_ndcg5_log1p_gain_pools=("ndcg5_log1p_gain", "count"), valid_ndcg5_raw_gain_pools=("ndcg5_raw_gain", "count"),
        valid_ndcg5_expansion_binary_pools=("ndcg5_expansion_binary", "count"))
    for metric, column in [("ndcg10_capped_grade", "valid_ndcg10_capped_grade_pools"),
                           ("ndcg10_entry_binary", "valid_ndcg10_entry_binary_pools"),
                           ("ndcg10_log1p_gain", "valid_ndcg10_log1p_gain_pools"),
                           ("ndcg10_raw_gain", "valid_ndcg10_raw_gain_pools"),
                           ("ndcg5_capped_grade", "valid_ndcg5_capped_grade_pools"),
                           ("ndcg5_entry_binary", "valid_ndcg5_entry_binary_pools"),
                           ("ndcg5_log1p_gain", "valid_ndcg5_log1p_gain_pools"),
                           ("ndcg5_raw_gain", "valid_ndcg5_raw_gain_pools"),
                           ("ndcg5_expansion_binary", "valid_ndcg5_expansion_binary_pools")]:
        counts = unique_task_pools.loc[unique_task_pools[metric].notna()].groupby(
            ["ranking_model", "task", "hs2"]).size()
        by_hs2[column] = [int(counts.get((r.ranking_model, r.task, r.hs2), 0))
                          for r in by_hs2.itertuples(index=False)]
    overall.to_csv(OUT / "results/E3_neural_ranking.csv", index=False, na_rep="")
    by_hs2.to_csv(OUT / "results/E3_ranking_by_hs2.csv", index=False, na_rep="")
    by_year.to_csv(OUT / "results/E3_ranking_by_year.csv", index=False, na_rep="")
    metrics_frame.to_csv(OUT / "results/E3_ranking_pool_metrics.csv", index=False, na_rep="")
    # Prespecified Family B contrasts on the same product-year pools.
    tests_all = []
    pairs = [("TabM", "Gravity_PPML"), ("TabM", "LightGBM"), ("MLP", "Gravity_PPML"),
             ("MLP", "LightGBM"), ("TabM", "MLP")]
    for task in ["Task_A_new_entry", "Task_B_existing_underexploited"]:
        d = metrics_frame.loc[metrics_frame.task.eq(task)].rename(columns={"ranking_model": "model"})
        primary_metric = "ndcg10_entry_binary" if task == "Task_A_new_entry" else "ndcg10_capped_grade"
        tests_all.append(paired_pool_tests(d, f"E3_{task}", primary_metric, pairs))
    e3tests = pd.concat(tests_all, ignore_index=True)
    e3tests.to_csv(OUT / "results/statistical_comparisons_E3.csv", index=False, na_rep="")
    return overall, by_hs2, by_year, metrics_frame, ranking_frame, e3tests


