from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .candidate_protocol import CAT_FEATURES, FEATURES, HERE, LAG, YEARS, assert_outer_timing, training_rows
from .lambdamart import SEEDS, _cache_paths, feature_frame, fit_final, tune_outer_fold


def train_rank_xendcg(panel: pd.DataFrame, manifest: pd.DataFrame):
    """Objective sensitivity using LambdaMART-selected historical settings.

    This is exploratory: the objective changes while the chronological
    validation-selected configuration/iteration count are held fixed.
    """
    predictions, costs = [], []
    for task in ["Task_A_new_entry", "Task_B_existing_underexploited"]:
        for year in YEARS:
            train = training_rows(panel, task, year)
            test = manifest.loc[manifest.task.eq(task) & manifest.outcome_year.eq(year)].copy()
            features_available = [c for c in FEATURES if c in panel.columns and c not in test.columns]
            predictor_view = panel[["candidate_id", "feature_year", *features_available]].drop_duplicates("candidate_id")
            test = test.merge(predictor_view, on="candidate_id", how="left", validate="one_to_one")
            assert_outer_timing(train, test, year)
            selected = tune_outer_fold(panel, task, year)
            for seed in SEEDS:
                model_path, meta_path = _cache_paths(task, year, seed)
                model_path = model_path.with_name(model_path.name.replace("lambdamart_", "rank_xendcg_"))
                meta_path = meta_path.with_name(meta_path.name.replace("lambdamart_", "rank_xendcg_"))
                if model_path.exists() and meta_path.exists():
                    model = lgb.Booster(model_file=str(model_path))
                    meta = json.loads(meta_path.read_text())
                    categories = meta["categories"]
                    used_features = meta["feature_names"]
                    used_categories = meta["categorical_features"]
                    fit_runtime = float(meta["training_runtime_seconds"])
                    cached = True
                else:
                    model, categories, cuts, fit_runtime, train_sorted, labels, groups, used_features, used_categories = fit_final(
                        train, task, selected["config"], selected["best_iteration"], seed, year,
                        objective="rank_xendcg")
                    model.booster_.save_model(str(model_path))
                    meta = {"task": task, "evaluation_year": int(year), "seed": int(seed),
                            "objective": "rank_xendcg", "selection_note": "exploratory fixed-settings sensitivity",
                            "config_source": "chronological lambdarank validation selection",
                            "training_outcome_cutoff": int(year - LAG),
                            "training_outcome_max": int(train.outcome_year.max()),
                            "feature_cutoff": int(year - LAG), "best_iteration": int(selected["best_iteration"]),
                            "config": selected["config"], "categories": categories,
                            "feature_names": used_features, "categorical_features": used_categories,
                            "training_cutoffs": None if cuts is None else cuts.tolist(),
                            "training_rows": int(len(train_sorted)), "training_groups": int(len(groups)),
                            "training_runtime_seconds": float(fit_runtime), "lightgbm_version": lgb.__version__}
                    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
                    cached = False
                x = feature_frame(test, categories, used_features, used_categories)
                start = time.perf_counter()
                score = np.asarray(model.predict(x, num_threads=4), dtype=float)
                inference_runtime = time.perf_counter() - start
                booster = model.booster_ if hasattr(model, "booster_") else model
                out = test[["candidate_id", "destination_iso3", "destination_name", "hs2", "outcome_year", "task",
                            "label_observed", "entry_success", "export_value_usd", "realized_incremental_export_usd",
                            "historical_export_value", "historical_market_share"]].copy()
                out["model"] = "rank_xendcg_sensitivity"
                out["seed"] = int(seed)
                out["score"] = score
                out["objective"] = "rank_xendcg"
                predictions.append(out)
                costs.append({"task": task, "evaluation_year": int(year), "seed": int(seed),
                              "stage": "outer_rank_xendcg_sensitivity", "trial": np.nan,
                              "fit_runtime_seconds": fit_runtime, "inference_runtime_seconds": inference_runtime,
                              "train_rows": len(train), "train_queries": int(train[["hs2", "outcome_year"]].drop_duplicates().shape[0]),
                              "test_rows": len(test), "best_iteration": int(meta["best_iteration"]),
                              "parameter_count": np.nan, "tree_count": int(booster.num_trees()),
                              "feature_count": len(used_features), "config": json.dumps(meta["config"], sort_keys=True),
                              "cached_model": cached, "threads": 4})
    result = pd.concat(predictions, ignore_index=True)
    result.to_parquet(HERE / "predictions/E6_rank_xendcg_predictions.parquet", index=False)
    cost = pd.DataFrame(costs)
    cost.to_csv(HERE / "results/E6_rank_xendcg_cost.csv", index=False, na_rep="")
    return result, cost


def evaluate_rank_xendcg(predictions: pd.DataFrame):
    pool_rows = []
    for (task, seed, hs2, year), q in predictions.groupby(["task", "seed", "hs2", "outcome_year"], sort=False):
        q = q.sort_values(["score", "destination_iso3"], ascending=[False, True], kind="mergesort")
        from .ranking_metrics import pool_metrics
        metric = pool_metrics(q, "score", task)
        pool_rows.append({"task": task, "model": "rank_xendcg_sensitivity", "seed": int(seed),
                          "hs2": str(hs2).zfill(2), "outcome_year": int(year), **metric})
    pool = pd.DataFrame(pool_rows)
    seed_summary = pool.groupby(["task", "seed"], as_index=False).agg(
        ndcg10=("ndcg_at_10", "mean"), precision10=("precision_at_10", "mean"),
        valid_ndcg10=("ndcg_at_10", "count"), candidate_pools=("outcome_year", "size"))
    mean_score = predictions.groupby(["candidate_id", "task", "hs2", "outcome_year"], as_index=False).agg(
        score=("score", "mean"), destination_iso3=("destination_iso3", "first"),
        label_observed=("label_observed", "first"), entry_success=("entry_success", "first"),
        realized_incremental_export_usd=("realized_incremental_export_usd", "first"))
    mean_pool_rows = []
    for (task, hs2, year), q in mean_score.groupby(["task", "hs2", "outcome_year"], sort=False):
        metric = pool_metrics(q, "score", task)
        mean_pool_rows.append({"task": task, "model": "rank_xendcg_seed_mean", "seed": -2,
                               "hs2": str(hs2).zfill(2), "outcome_year": int(year), **metric})
    mean_pool = pd.DataFrame(mean_pool_rows)
    mean_summary = mean_pool.groupby(["task", "model", "seed"], as_index=False).agg(
        ndcg10=("ndcg_at_10", "mean"), precision10=("precision_at_10", "mean"),
        valid_ndcg10=("ndcg_at_10", "count"), candidate_pools=("outcome_year", "size"))
    pool.to_csv(HERE / "results/E6_rank_xendcg_pool_metrics.csv", index=False, na_rep="")
    summary = pd.concat([seed_summary.assign(model="rank_xendcg_sensitivity"), mean_summary], ignore_index=True)
    summary.to_csv(HERE / "results/E6_rank_xendcg_sensitivity.csv", index=False, na_rep="")
    return pool, summary
