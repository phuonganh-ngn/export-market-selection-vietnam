from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .candidate_protocol import (CAT_FEATURES, FEATURES, HERE, LAG, YEARS, assert_outer_timing,
                                 training_rows, validation_rows)
from .ranking_metrics import ndcg_from_gains
from .ranking_targets import (assert_query_groups, query_sort,
                              task_b_quantile_grade, training_increment_cutoffs)

SEEDS = (11, 23, 37, 51, 73)
QUANTILES = (0.25, 0.50, 0.75)
LABEL_GAIN_BINARY = [0, 1]
LABEL_GAIN_GRADED = [0, 1, 3, 7, 15]
CONFIGS = [
    {"num_leaves": 15, "max_depth": 4, "learning_rate": 0.05, "n_estimators": 500,
     "min_child_samples": 30, "colsample_bytree": 0.90, "reg_alpha": 0.0,
     "reg_lambda": 1.0, "lambdarank_truncation_level": 10},
    {"num_leaves": 31, "max_depth": -1, "learning_rate": 0.03, "n_estimators": 700,
     "min_child_samples": 20, "colsample_bytree": 1.00, "reg_alpha": 0.0,
     "reg_lambda": 5.0, "lambdarank_truncation_level": 15},
    {"num_leaves": 15, "max_depth": 5, "learning_rate": 0.08, "n_estimators": 350,
     "min_child_samples": 50, "colsample_bytree": 0.80, "reg_alpha": 0.5,
     "reg_lambda": 5.0, "lambdarank_truncation_level": 10},
]


def feature_frame(frame: pd.DataFrame, categories: dict[str, list[str]] | None = None,
                  features: list[str] | None = None, cat_features: list[str] | None = None):
    features = list(FEATURES if features is None else features)
    cat_features = list(CAT_FEATURES if cat_features is None else cat_features)
    x = frame[features].copy()
    for c in cat_features:
        values = x[c].fillna("__MISSING__").astype(str)
        if categories is None:
            levels = sorted(values.unique().tolist())
        else:
            levels = categories[c]
        x[c] = pd.Categorical(values.where(values.isin(levels), np.nan), categories=levels)
    for c in [c for c in features if c not in cat_features]:
        x[c] = pd.to_numeric(x[c], errors="coerce").astype(np.float32)
    return x


def category_levels(frame: pd.DataFrame, cat_features: list[str] | None = None) -> dict[str, list[str]]:
    cat_features = list(CAT_FEATURES if cat_features is None else cat_features)
    return {c: sorted(frame[c].fillna("__MISSING__").astype(str).unique().tolist()) for c in cat_features}


def _relevance(frame: pd.DataFrame, task: str, cuts=None) -> np.ndarray:
    if task == "Task_A_new_entry":
        y = frame.entry_success.to_numpy(float)
        if not np.isfinite(y).all() or not np.isin(y, [0.0, 1.0]).all():
            raise AssertionError("Task A LambdaMART labels must be observed 0/1 integers")
        return y.astype(np.int32)
    if cuts is None:
        raise ValueError("Task B relevance requires training-only increment cutoffs")
    labels = task_b_quantile_grade(frame.realized_incremental_export_usd, cuts).astype(np.int32)
    return labels


def _query_arrays(frame: pd.DataFrame, task: str, features: list[str], cuts=None):
    relevance = _relevance(frame, task, cuts)
    d, x, y, groups = query_sort(frame, relevance, features)
    assert_query_groups(d, groups, y)
    return d, x, y, groups


def _macro_eval_ndcg(y_true, y_pred, weight=None, group=None):
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=float)
    if group is None:
        raise ValueError("LightGBM did not pass validation query groups to the custom metric")
    sizes = np.asarray(group, dtype=int)
    if sizes.sum() != len(y):
        raise AssertionError("Validation group sizes do not sum to validation labels")
    vals = []
    start = 0
    for size in sizes:
        end = start + int(size)
        yy, pp = y[start:end], pred[start:end]
        order = np.lexsort((np.arange(len(pp)), -pp))
        gain = np.exp2(yy.astype(float)) - 1.0
        value = ndcg_from_gains(gain[order], gain, 10)
        if np.isfinite(value):
            vals.append(value)
        start = end
    return "macro_ndcg10", float(np.mean(vals)) if vals else 0.0, True


def params_for(config: dict, seed: int, label_gain: list[int]) -> dict:
    out = dict(config)
    out.update({
        "objective": "lambdarank", "metric": "None", "label_gain": label_gain,
        "lambdarank_norm": True, "lambdarank_bias_regularization": 0.0,
        "random_state": int(seed), "n_jobs": 4, "verbosity": -1,
        "deterministic": True, "force_col_wise": True,
        "feature_fraction_seed": int(seed), "bagging_seed": int(seed), "data_random_seed": int(seed),
    })
    return out


def fit_validation_candidate(train: pd.DataFrame, valid: pd.DataFrame, task: str,
                             config: dict, seed: int = 11) -> dict:
    cuts = training_increment_cutoffs(train, QUANTILES) if task == "Task_B_existing_underexploited" else None
    train_s, xtrain_raw, ytrain, gtrain = _query_arrays(train, task, FEATURES, cuts)
    valid_s, xvalid_raw, yvalid, gvalid = _query_arrays(valid, task, FEATURES, cuts)
    categories = category_levels(train_s)
    xtrain = feature_frame(train_s, categories)
    xvalid = feature_frame(valid_s, categories)
    label_gain = LABEL_GAIN_BINARY if task == "Task_A_new_entry" else LABEL_GAIN_GRADED
    model = lgb.LGBMRanker(**params_for(config, seed, label_gain))
    start = time.perf_counter()
    model.fit(
        xtrain, ytrain, group=gtrain,
        eval_X=xvalid, eval_y=yvalid, eval_group=[gvalid], eval_at=[5, 10],
        eval_metric=_macro_eval_ndcg,
        categorical_feature=CAT_FEATURES,
        callbacks=[lgb.early_stopping(35, first_metric_only=True, verbose=False), lgb.log_evaluation(0)],
    )
    elapsed = time.perf_counter() - start
    best_iter = int(model.best_iteration_ or model.n_estimators_)
    best_score = float(model.best_score_["valid_0"].get("macro_ndcg10", np.nan))
    return {"model": model, "config": dict(config), "best_iteration": best_iter,
            "validation_score": best_score, "runtime_seconds": elapsed,
            "train_rows": len(train_s), "validation_rows": len(valid_s),
            "train_groups": len(gtrain), "validation_groups": len(gvalid),
            "categories": categories, "training_cutoffs": cuts}


def fit_final(train: pd.DataFrame, task: str, config: dict, best_iteration: int,
              seed: int, eval_year: int, objective: str = "lambdarank",
              drop_features: list[str] | None = None):
    features = [c for c in FEATURES if c not in set(drop_features or [])]
    cat_features = [c for c in CAT_FEATURES if c in features]
    cuts = training_increment_cutoffs(train, QUANTILES) if task == "Task_B_existing_underexploited" else None
    train_s, _, ytrain, groups = _query_arrays(train, task, features, cuts)
    categories = category_levels(train_s, cat_features)
    xtrain = feature_frame(train_s, categories, features, cat_features)
    label_gain = LABEL_GAIN_BINARY if task == "Task_A_new_entry" else LABEL_GAIN_GRADED
    final_config = dict(config)
    final_config["n_estimators"] = max(1, int(best_iteration))
    params = params_for(final_config, seed, label_gain)
    params["objective"] = objective
    model = lgb.LGBMRanker(**params)
    start = time.perf_counter()
    model.fit(xtrain, ytrain, group=groups, categorical_feature=cat_features)
    elapsed = time.perf_counter() - start
    return model, categories, cuts, elapsed, train_s, ytrain, groups, features, cat_features


def _cache_paths(task: str, year: int, seed: int):
    stem = f"lambdamart_{task}_{year}_seed{seed}"
    return HERE / "models" / f"{stem}.txt", HERE / "models" / f"{stem}.json"


def tune_outer_fold(panel: pd.DataFrame, task: str, eval_year: int) -> dict:
    tuning_path = HERE / "results" / f"tuning_{task}_{eval_year}.csv"
    if tuning_path.exists():
        saved = pd.read_csv(tuning_path)
        if len(saved):
            best_ix = saved.validation_macro_ndcg10.fillna(-np.inf).idxmax()
            row = saved.loc[best_ix]
            return {"config": json.loads(row.config), "best_iteration": int(row.best_iteration),
                    "validation_score": float(row.validation_macro_ndcg10)}
    validation_year = int(eval_year) - LAG
    train = training_rows(panel, task, eval_year, outcome_cutoff=validation_year - LAG)
    valid = validation_rows(panel, task, validation_year)
    if train.empty or valid.empty:
        raise ValueError(f"Empty chronological development/validation data for {task} {eval_year}")
    results = []
    fits = []
    for trial, config in enumerate(CONFIGS, start=1):
        fit = fit_validation_candidate(train, valid, task, config, seed=11)
        fits.append(fit)
        results.append({"task": task, "evaluation_year": eval_year, "validation_year": validation_year,
                        "trial": trial, "config": json.dumps(config, sort_keys=True),
                        "validation_macro_ndcg10": fit["validation_score"],
                        "best_iteration": fit["best_iteration"], "runtime_seconds": fit["runtime_seconds"],
                        "train_rows": fit["train_rows"], "train_groups": fit["train_groups"],
                        "validation_rows": fit["validation_rows"], "validation_groups": fit["validation_groups"],
                        "train_outcome_max": int(train.outcome_year.max()),
                        "validation_outcome_year": validation_year,
                        "training_increment_cutoffs": "" if fit["training_cutoffs"] is None else json.dumps(fit["training_cutoffs"].tolist())})
    table = pd.DataFrame(results)
    # Criterion is external macro-NDCG@10 over validation queries, not outer-test output.
    best_ix = table.validation_macro_ndcg10.fillna(-np.inf).idxmax()
    best = fits[int(best_ix)]
    table.to_csv(tuning_path, index=False)
    return best


def train_outer(panel: pd.DataFrame, test_manifest: pd.DataFrame, tasks: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = tasks or ["Task_A_new_entry", "Task_B_existing_underexploited"]
    prediction_frames, cost_rows, importance_rows = [], [], []
    for task in tasks:
        for year in YEARS:
            train = training_rows(panel, task, year)
            test = test_manifest.loc[test_manifest.task.eq(task) & test_manifest.outcome_year.eq(year)].copy()
            # The frozen manifest intentionally contains protocol fields only.
            # Join the already processed, lagged predictor frame by immutable ID.
            feature_cols = [c for c in FEATURES if c in panel.columns and c not in test.columns]
            available = set(feature_cols) | set(test.columns)
            missing_features = sorted(set(FEATURES) - available)
            if missing_features:
                raise AssertionError(f"Processed panel lacks required features: {missing_features}")
            predictor_view = panel[["candidate_id", "feature_year", *feature_cols]].drop_duplicates("candidate_id")
            test = test.merge(predictor_view, on="candidate_id", how="left", validate="one_to_one")
            if train.empty or test.empty:
                raise ValueError(f"Empty outer data for {task}/{year}")
            if int(train.outcome_year.max()) > year - LAG:
                raise AssertionError("Outer training outcomes exceed audited cutoff")
            assert_outer_timing(train, test, year)
            if int(test.historical_information_cutoff.max()) != year - LAG:
                raise AssertionError("Feature cutoff is not y-2")
            selected = tune_outer_fold(panel, task, year)
            for seed in SEEDS:
                model_path, meta_path = _cache_paths(task, year, seed)
                key_df = test[["candidate_id", "destination_iso3", "hs2", "outcome_year", "task", "label_observed",
                               "entry_success", "export_value_usd", "realized_incremental_export_usd",
                               "historical_export_value", "historical_market_share", "training_median_market_share"]].copy()
                if model_path.exists() and meta_path.exists():
                    cached_model = True
                    model = lgb.Booster(model_file=str(model_path))
                    meta = json.loads(meta_path.read_text())
                    cats = meta["categories"]
                    used_features = meta.get("feature_names", FEATURES)
                    used_cats = meta.get("categorical_features", CAT_FEATURES)
                    cutoffs = None if meta.get("training_cutoffs") is None else np.asarray(meta["training_cutoffs"], dtype=float)
                    x = feature_frame(test, cats, used_features, used_cats)
                    infer_start = time.perf_counter()
                    scores = model.predict(x, num_threads=4)
                    inference_runtime = time.perf_counter() - infer_start
                    fit_runtime = float(meta.get("training_runtime_seconds", np.nan))
                    best_iteration = int(meta["best_iteration"])
                    config = meta["config"]
                else:
                    cached_model = False
                    model, cats, cutoffs, runtime, train_sorted, ytrain, groups, used_features, used_cats = fit_final(
                        train, task, selected["config"], selected["best_iteration"], seed, year)
                    fit_runtime = runtime
                    x = feature_frame(test, cats, used_features, used_cats)
                    infer_start = time.perf_counter()
                    scores = model.predict(x, num_threads=4)
                    inference_runtime = time.perf_counter() - infer_start
                    model.booster_.save_model(str(model_path))
                    best_iteration = int(model.best_iteration_ or model.n_estimators_)
                    config = selected["config"]
                    meta = {"task": task, "evaluation_year": year, "seed": seed,
                            "training_outcome_cutoff": year - LAG,
                            "training_outcome_max": int(train.outcome_year.max()),
                            "feature_cutoff": year - LAG, "best_iteration": best_iteration,
                            "config": config, "categories": cats,
                            "training_cutoffs": None if cutoffs is None else cutoffs.tolist(),
                            "training_rows": len(train_sorted), "training_groups": len(groups),
                            "training_runtime_seconds": runtime, "feature_names": used_features,
                            "categorical_features": used_cats,
                            "objective": "lambdarank", "lightgbm_version": lgb.__version__}
                    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
                    importance = model.booster_.feature_importance(importance_type="gain")
                    importance_rows.extend({"task": task, "evaluation_year": year, "seed": seed,
                                            "feature": name, "gain_importance": float(value)}
                                           for name, value in zip(used_features, importance))
                key_df["model"] = "LambdaMART"
                key_df["seed"] = seed
                key_df["score"] = np.asarray(scores, dtype=float)
                key_df["training_rows"] = len(train)
                key_df["training_groups"] = int(train[["hs2", "outcome_year"]].drop_duplicates().shape[0])
                key_df["training_outcome_max"] = int(train.outcome_year.max())
                key_df["best_iteration"] = best_iteration
                key_df["config"] = json.dumps(config, sort_keys=True)
                prediction_frames.append(key_df)
                cost_rows.append({"task": task, "evaluation_year": year, "seed": seed,
                                  "fit_runtime_seconds": fit_runtime, "inference_runtime_seconds": inference_runtime,
                                  "train_rows": len(train), "train_queries": int(train[["hs2", "outcome_year"]].drop_duplicates().shape[0]),
                                  "test_rows": len(test), "best_iteration": best_iteration,
                                  "parameter_count": np.nan, "tree_count": int(model.num_trees()),
                                  "feature_count": len(used_features), "config": json.dumps(config, sort_keys=True),
                                  "cached_model": cached_model, "threads": 4})
    pred = pd.concat(prediction_frames, ignore_index=True)
    # Keep the fitted ranker outputs separate from the final merged baseline
    # comparison table written by the evaluation stage.
    out_path = HERE / "predictions/E6_lambdamart_predictions.parquet"
    pred.to_parquet(out_path, index=False)
    final_cost = pd.DataFrame(cost_rows)
    final_cost["stage"] = "outer_final_fit"
    final_cost["trial"] = np.nan
    tuning_cost_rows = []
    for task in tasks:
        for year in YEARS:
            tuning_path = HERE / "results" / f"tuning_{task}_{year}.csv"
            if not tuning_path.exists():
                raise FileNotFoundError(f"Missing chronological tuning record: {tuning_path.name}")
            tuning = pd.read_csv(tuning_path)
            for row in tuning.itertuples(index=False):
                tuning_cost_rows.append({"task": task, "evaluation_year": int(year), "seed": 11,
                                         "fit_runtime_seconds": float(row.runtime_seconds),
                                         "inference_runtime_seconds": np.nan,
                                         "train_rows": int(row.train_rows), "train_queries": int(row.train_groups),
                                         "test_rows": int(row.validation_rows), "best_iteration": int(row.best_iteration),
                                         "parameter_count": np.nan, "tree_count": int(row.best_iteration),
                                         "feature_count": len(FEATURES), "config": row.config,
                                         "cached_model": False, "threads": 4,
                                         "stage": "inner_chronological_tuning", "trial": int(row.trial)})
    cost = pd.concat([final_cost, pd.DataFrame(tuning_cost_rows)], ignore_index=True, sort=False)
    cost.to_csv(HERE / "results/computational_cost.csv", index=False)
    if importance_rows:
        pd.DataFrame(importance_rows).to_csv(HERE / "results/lambdamart_gain_importance.csv", index=False)
    return pred, cost

