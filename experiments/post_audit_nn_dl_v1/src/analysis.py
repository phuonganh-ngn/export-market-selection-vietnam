from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import (HERE, PROJECT, YEARS, LAG, ENTRY_THRESHOLD_USD, SEEDS, KEY,
                   FEATURE_SETS, load_frozen_population)
from .metrics import (discovery_pool_metrics, rank_pool, capped_export_grade,
                      ndcg_at_k, ndcg_from_gain, precision_at_k)
from .models import FoldEncoder, fit_fixed_epochs, predict_task
from .pipeline import (OUT, RUNS, TUNING, MODEL_DIR, temporal_outer,
                       _make_task_populations, write_parquet)

KEYS = KEY


def run_feature_ablation_2024(panel: pd.DataFrame) -> pd.DataFrame:
    """Exploratory final-year ablation; full-feature hyperparameters are held fixed."""
    year, seed = 2024, 11
    rows = []
    e2_frames, e3_frames = [], []
    all_feature_sets = list(FEATURE_SETS)
    for task in ["classification", "regression"]:
        target_col = "entry_success" if task == "classification" else "export_value_usd"
        train, test = temporal_outer(panel, year, task)
        train_y = train[target_col].astype(float).to_numpy()
        if task == "regression":
            train_y = np.log1p(train_y)
        for model in ["MLP", "TabM"]:
            full_tune = json.loads((TUNING / f"{task}_{model}_{year}_F4_full.json").read_text())
            config, epochs = full_tune["best_config"], int(full_tune["best_epoch"])
            for feature_set in all_feature_sets:
                run_path = RUNS / f"ablation_{task}_{model}_{year}_{feature_set}_seed{seed}.parquet"
                if feature_set == "F4_full":
                    primary_path = RUNS / f"{task}_{model}_{year}_F4_full_seed{seed}.parquet"
                    pred = pd.read_parquet(primary_path)
                elif run_path.exists():
                    pred = pd.read_parquet(run_path)
                else:
                    encoder = FoldEncoder.fit(train, model, feature_set)
                    xtr, ctr = encoder.transform(train)
                    xte, cte = encoder.transform(test)
                    fit = fit_fixed_epochs(model, task, encoder, xtr, ctr, train_y, config, seed, epochs)
                    inference_start = time.perf_counter()
                    output, raw = predict_task(fit["model"], model, task, xte, cte,
                                               float(full_tune.get("calibration_factor", 1.0) or 1.0))
                    inference_seconds = time.perf_counter() - inference_start
                    if task == "classification":
                        pred = test[["candidate_id", "destination_iso3", "hs2", "outcome_year", "entry_success",
                                     "label_observed", "entry_candidate_ex_ante"]].copy()
                        pred["predicted_probability"] = output
                        pred["score"] = output
                    else:
                        pred = test[["candidate_id", "destination_iso3", "hs2", "outcome_year", "export_value_usd",
                                     "label_observed", "historical_export_value", "realized_incremental_export_usd",
                                     "entry_candidate_ex_ante", "historical_market_share", "destination_import_demand",
                                     "log_gdp", "log_destination_import_demand", "lagged_export_growth"]].copy()
                        pred["predicted_export_value_usd"] = output
                        pred["predicted_log1p_export_raw"] = raw
                        pred["realized_incremental_export_usd"] = pred.realized_incremental_export_usd
                    pred["model"] = model
                    pred["task"] = task
                    pred["feature_set"] = feature_set
                    pred["seed"] = seed
                    pred["evaluation_year"] = year
                    pred["training_rows"] = len(train)
                    pred["training_outcome_max"] = int(train.outcome_year.max())
                    write_parquet(pred, run_path)
                    torch_path = MODEL_DIR / f"ablation_{task}_{model}_{year}_{feature_set}_seed{seed}.pt"
                    import torch
                    torch.save(fit["model"].state_dict(), torch_path)
                    joblib.dump(encoder.get_state(), MODEL_DIR / f"ablation_{task}_{model}_{year}_{feature_set}_seed{seed}_preprocess.joblib", compress=3)
                    meta_path = MODEL_DIR / f"ablation_{task}_{model}_{year}_{feature_set}_seed{seed}.json"
                    meta_path.write_text(json.dumps({"model": model, "task": task, "seed": seed,
                        "evaluation_year": year, "feature_set": feature_set, "config": config,
                        "epochs": epochs, "training_rows": len(train),
                        "training_outcome_max": int(train.outcome_year.max()),
                        "parameter_count": fit["parameter_count"],
                        "training_runtime_seconds": fit["runtime_seconds"],
                        "inference_runtime_seconds": inference_seconds,
                        "calibration_factor": float(full_tune.get("calibration_factor", 1.0) or 1.0),
                        "exploratory_fixed_hyperparameters": True}, indent=2, default=float) + "\n")
                if task == "classification":
                    pred = pred.copy()
                    pred["feature_set"] = feature_set
                    e2_frames.append(pred)
                else:
                    pred = pred.copy()
                    pred["feature_set"] = feature_set
                    e3_frames.append(pred)

    e2 = pd.concat(e2_frames, ignore_index=True)
    e3 = pd.concat(e3_frames, ignore_index=True)
    write_parquet(e2, OUT / "predictions/E2_feature_ablation.parquet")
    write_parquet(e3, OUT / "predictions/E3_feature_ablation.parquet")
    e2_known = e2.loc[e2.label_observed & e2.entry_success.notna()].copy()
    for (model, feature_set, hs2), g in e2_known.groupby(["model", "feature_set", "hs2"]):
        m = discovery_pool_metrics(g.entry_success.astype(int), g.score.astype(float), 10)
        rows.append({"experiment": "E4_ablation", "task": "discovery", "model": model,
                     "feature_set": feature_set, "evaluation_year": year, "seed": seed, "hs2": hs2,
                     "metric": "PR_AUC", "value": m["pr_auc"], "n_observed": m["n_observed"],
                     "positive_events": m["positive_events"], "scope": "2024 outer fold; one seed; F4 development-selected hyperparameters fixed"})
        for metric, key in [("Precision@10", "precision_at_10"), ("NDCG@10", "ndcg_at_10")]:
            rows.append({"experiment": "E4_ablation", "task": "discovery", "model": model,
                         "feature_set": feature_set, "evaluation_year": year, "seed": seed, "hs2": hs2,
                         "metric": metric, "value": m[key], "n_observed": m["n_observed"],
                         "positive_events": m["positive_events"], "scope": "2024 outer fold; one seed; F4 development-selected hyperparameters fixed"})
    pools = _make_task_populations(panel, year)
    for task, poolbase in pools.items():
        if task == "Task_B_all_existing_sensitivity":
            continue
        for (model, feature_set), pred in e3.groupby(["model", "feature_set"]):
            d = poolbase.merge(pred[["candidate_id", "predicted_export_value_usd"]], on="candidate_id",
                               how="left", validate="one_to_one")
            d = d.loc[d.label_observed].copy()
            for hs2, g in d.groupby("hs2"):
                g["opportunity_score"] = np.maximum(0.0, g.predicted_export_value_usd - g.historical_export_value)
                ranked = rank_pool(g, "opportunity_score")
                inc = np.maximum(0.0, ranked.realized_incremental_export_usd.to_numpy(float))
                grade = capped_export_grade(inc, ENTRY_THRESHOLD_USD)
                vals = {"NDCG@10_capped": ndcg_at_k(grade, 10),
                        "NDCG@10_log1p_gain": ndcg_from_gain(np.log1p(inc), np.log1p(inc), 10),
                        "NDCG@10_raw_gain": ndcg_from_gain(inc, inc, 10),
                        "Top10_known_incremental_USD": float(inc[:10].sum())}
                if task == "Task_A_new_entry":
                    vals["NDCG@10_entry_binary"] = ndcg_at_k(ranked.entry_success.astype(int).to_numpy(), 10)
                for metric, value in vals.items():
                    rows.append({"experiment": "E4_ablation", "task": task, "model": model,
                                 "feature_set": feature_set, "evaluation_year": year, "seed": seed, "hs2": hs2,
                                 "metric": metric, "value": value, "n_observed": len(g),
                                 "positive_events": int(np.sum(inc >= ENTRY_THRESHOLD_USD)),
                                 "scope": "2024 outer fold; one seed; F4 development-selected hyperparameters fixed"})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "results/feature_ablation.csv", index=False, na_rep="")
    summary = result.groupby(["task", "model", "feature_set", "metric"], as_index=False).agg(
        mean_across_hs2=("value", "mean"), sd_across_hs2=("value", "std"),
        valid_hs2_chapters=("hs2", "nunique"), observed_cases=("n_observed", "sum"), positive_events=("positive_events", "sum"))
    summary.to_csv(OUT / "results/feature_ablation_summary.csv", index=False, na_rep="")
    return result


def _section_from_hs2(value) -> str:
    code = int(str(value).zfill(2))
    sections = [
        (1, 5, "I Live animals and products"), (6, 14, "II Vegetable products"),
        (15, 15, "III Animal or vegetable fats"), (16, 24, "IV Prepared food, beverages, tobacco"),
        (25, 27, "V Mineral products"), (28, 38, "VI Chemicals"), (39, 40, "VII Plastics and rubber"),
        (41, 43, "VIII Hides and leather"), (44, 46, "IX Wood and articles"), (47, 49, "X Pulp and paper"),
        (50, 63, "XI Textiles"), (64, 67, "XII Footwear and headgear"), (68, 70, "XIII Stone, glass, ceramics"),
        (71, 71, "XIV Precious stones and metals"), (72, 83, "XV Base metals"), (84, 85, "XVI Machinery and electronics"),
        (86, 89, "XVII Transport equipment"), (90, 92, "XVIII Optical and musical instruments"),
        (93, 93, "XIX Arms and ammunition"), (94, 96, "XX Miscellaneous manufactures"),
        (97, 97, "XXI Works of art"),
    ]
    for low, high, name in sections:
        if low <= code <= high:
            return name
    return "Unmapped/reserved"


def _resample_ranked_gains(destination_ids: np.ndarray, gains: np.ndarray,
                           destination_multiplicity: np.ndarray) -> np.ndarray:
    """Replicate each destination's relevance in this model's own score order."""
    weights = destination_multiplicity[destination_ids]
    keep = weights > 0
    return np.repeat(gains[keep], weights[keep])


def _weighted_ndcg_draws(destination_ids: np.ndarray, relevance: np.ndarray,
                         multiplicity_draws: np.ndarray, k: int = 10) -> np.ndarray:
    """Exact NDCG for destination-block bootstrap draws without expanding duplicates."""
    relevance_gain = np.exp2(np.asarray(relevance, dtype=float)) - 1.0
    out = np.full(multiplicity_draws.shape[0], np.nan, dtype=float)
    if not len(destination_ids) or not np.any(relevance_gain > 0):
        return out
    positions = np.arange(1, k + 1, dtype=np.int64)
    discounts = 1.0 / np.log2(positions + 1.0)

    def dcg(gain: np.ndarray, ids: np.ndarray) -> np.ndarray:
        weights = multiplicity_draws[:, ids]
        cumulative = np.cumsum(weights, axis=1)
        hit = cumulative[:, :, None] >= positions[None, None, :]
        selected = hit.argmax(axis=1)
        valid_positions = cumulative[:, -1, None] >= positions[None, :]
        picked_gain = gain[selected]
        return np.sum(np.where(valid_positions, picked_gain * discounts[None, :], 0.0), axis=1)

    predicted_dcg = dcg(relevance_gain, destination_ids)
    ideal_order = np.argsort(-relevance_gain, kind="mergesort")
    ideal_dcg = dcg(relevance_gain[ideal_order], destination_ids[ideal_order])
    np.divide(predicted_dcg, ideal_dcg, out=out, where=ideal_dcg > 0)
    return out


def run_heterogeneity(panel: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Descriptive subgroup analysis; all size/share cutoffs use training-period data."""
    e2 = pd.read_parquet(OUT / "predictions/E2_neural_predictions.parquet")
    e1 = pd.read_parquet(OUT / "predictions/E1_neural_predictions.parquet")
    for frame in (e2, e1):
        if "evaluation_year" in frame:
            frame["evaluation_year"] = pd.to_numeric(frame.evaluation_year, errors="coerce")
        else:
            frame["evaluation_year"] = np.nan
        frame["evaluation_year"] = frame.evaluation_year.fillna(frame.outcome_year).astype(int)
    cframes = []
    for year in YEARS:
        test = panel.loc[panel.outcome_year.eq(year)].copy()
        tr = panel.loc[panel.outcome_observed & panel.outcome_year.le(year - LAG)]
        demand_cut = float(tr.log_destination_import_demand.median())
        active = tr.loc[tr.historical_export_value.gt(0) & tr.historical_market_share.notna()]
        share_cut = float(active.historical_market_share.median()) if len(active) else np.nan
        test["market_size_group"] = np.where(test.log_destination_import_demand.le(demand_cut), "below_or_equal_training_median", "above_training_median")
        test["market_share_group"] = np.select(
            [test.historical_market_share.isna(), test.historical_market_share.le(share_cut)],
            ["share_missing", "below_or_equal_active_training_median"], default="above_active_training_median")
        test["relationship_status"] = np.where(test.historical_export_value.gt(0),
                                                "existing_export_relationship", "previously_inactive_at_reference")
        test["hs_section"] = test.hs2.map(_section_from_hs2)
        for dim in ["relationship_status", "market_size_group", "market_share_group", "continent", "hs_section"]:
            for label, group in test.groupby(dim, dropna=False):
                z = group[[*KEYS, "candidate_id"]].copy()
                z["subgroup_dimension"] = dim
                z["subgroup"] = str(label)
                cframes.append(z)
    sub = pd.concat(cframes, ignore_index=True).drop_duplicates(["candidate_id", "subgroup_dimension", "subgroup"])
    # Discovery: rank only within each HS2 × year subgroup and retain common observed labels.
    e2 = e2.loc[e2.model.isin(["MLP", "TabM", "LightGBM", "Gravity_PPML"])].merge(
        sub[["candidate_id", "subgroup_dimension", "subgroup"]], on="candidate_id", how="inner", validate="many_to_many")
    discovery_rows = []
    e2known = e2.loc[e2.label_observed & e2.entry_success.notna()].copy()
    for keys, g in e2known.groupby(["model", "seed", "evaluation_year", "hs2", "subgroup_dimension", "subgroup"]):
        model, seed, year, hs2, dimension, subgroup = keys
        m = discovery_pool_metrics(g.entry_success.astype(int), g.score.astype(float), 10)
        discovery_rows.append({"experiment": "E5_heterogeneity", "task": "discovery", "model": model,
                               "seed": seed, "evaluation_year": year, "hs2": hs2,
                               "subgroup_dimension": dimension, "subgroup": subgroup,
                               "n_observed": m["n_observed"], "positive_events": m["positive_events"],
                               "pr_auc": m["pr_auc"], "precision_at_10": m["precision_at_10"],
                               "ndcg_at_10": m["ndcg_at_10"]})
    # Forecasting: average seed-specific absolute errors within identical country-product-year observations.
    e1 = e1.loc[e1.model.isin(["Persistence", "Gravity_PPML", "LightGBM", "MLP", "TabM"]) & e1.label_observed].copy()
    e1 = e1.merge(sub, on=[*KEYS, "candidate_id"], how="inner", validate="many_to_many")
    e1["abs_error_usd"] = np.abs(e1.predicted_export_value_usd - e1.export_value_usd)
    e1agg = e1.groupby(["model", "seed", "evaluation_year", "hs2", "subgroup_dimension", "subgroup"], as_index=False).agg(
        n_observed=("abs_error_usd", "size"), mae_usd=("abs_error_usd", "mean"))
    for r in e1agg.itertuples(index=False):
        discovery_rows.append({"experiment": "E5_heterogeneity", "task": "forecasting", "model": r.model,
                               "seed": r.seed, "evaluation_year": r.evaluation_year, "hs2": r.hs2,
                               "subgroup_dimension": r.subgroup_dimension, "subgroup": r.subgroup,
                               "n_observed": r.n_observed, "positive_events": np.nan,
                               "mae_usd": r.mae_usd})
    result = pd.DataFrame(discovery_rows)
    result.to_csv(OUT / "results/heterogeneity.csv", index=False, na_rep="")
    comparison_rows = []
    group_keys = ["evaluation_year", "hs2", "subgroup_dimension", "subgroup"]
    discovery = result.loc[result.task.eq("discovery")].copy()
    if not discovery.empty:
        dmean = discovery.groupby([*group_keys, "model"], as_index=False).agg(
            n_observed=("n_observed", "mean"), positive_events=("positive_events", "mean"),
            pr_auc=("pr_auc", "mean"), precision_at_10=("precision_at_10", "mean"),
            ndcg_at_10=("ndcg_at_10", "mean"))
        for neural in ["MLP", "TabM"]:
            for baseline in ["LightGBM", "Gravity_PPML"]:
                a = dmean.loc[dmean.model.eq(neural)].drop(columns="model")
                b = dmean.loc[dmean.model.eq(baseline)].drop(columns="model")
                pair = a.merge(b, on=group_keys, suffixes=("_neural", "_baseline"), validate="one_to_one")
                for r in pair.itertuples(index=False):
                    for metric in ["pr_auc", "precision_at_10", "ndcg_at_10"]:
                        comparison_rows.append({"task": "discovery", "model": neural, "baseline_model": baseline,
                            "evaluation_year": r.evaluation_year, "hs2": r.hs2,
                            "subgroup_dimension": r.subgroup_dimension, "subgroup": r.subgroup,
                            "metric": metric, "neural_value": getattr(r, f"{metric}_neural"),
                            "baseline_value": getattr(r, f"{metric}_baseline"),
                            "difference_neural_minus_baseline": getattr(r, f"{metric}_neural") - getattr(r, f"{metric}_baseline"),
                            "n_observed": min(r.n_observed_neural, r.n_observed_baseline),
                            "positive_events": r.positive_events_neural})
    forecast = result.loc[result.task.eq("forecasting")].copy()
    if not forecast.empty:
        fmean = forecast.groupby([*group_keys, "model"], as_index=False).agg(mae_usd=("mae_usd", "mean"), n_observed=("n_observed", "mean"))
        for neural in ["MLP", "TabM"]:
            for baseline in ["Persistence", "Gravity_PPML", "LightGBM"]:
                a = fmean.loc[fmean.model.eq(neural)].drop(columns="model")
                b = fmean.loc[fmean.model.eq(baseline)].drop(columns="model")
                pair = a.merge(b, on=group_keys, suffixes=("_neural", "_baseline"), validate="one_to_one")
                for r in pair.itertuples(index=False):
                    comparison_rows.append({"task": "forecasting", "model": neural, "baseline_model": baseline,
                        "evaluation_year": r.evaluation_year, "hs2": r.hs2,
                        "subgroup_dimension": r.subgroup_dimension, "subgroup": r.subgroup,
                        "metric": "mae_usd", "neural_value": r.mae_usd_neural,
                        "baseline_value": r.mae_usd_baseline,
                        "difference_neural_minus_baseline": r.mae_usd_neural - r.mae_usd_baseline,
                        "n_observed": min(r.n_observed_neural, r.n_observed_baseline), "positive_events": np.nan})
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(OUT / "results/heterogeneity_comparisons.csv", index=False, na_rep="")
    return result


def run_missing_label_sensitivity(panel: pd.DataFrame, candidates: pd.DataFrame,
                                  forecast_base: pd.DataFrame) -> pd.DataFrame:
    """Label bounds plus an explicitly empirical, training-maximum forecast stress scenario."""
    unknown = candidates.loc[~candidates.label_observed].copy()
    observed = candidates.loc[candidates.label_observed].copy()
    entries = candidates.loc[candidates.entry_candidate_ex_ante].copy()
    known_entries = entries.loc[entries.label_observed]
    n_events = int(known_entries.entry_success.sum())
    n_unknown_entries = int((~entries.label_observed).sum())
    n_all_entries = len(entries)
    rows = [{"analysis": "new_entry_prevalence", "scope": "all ex-ante inactive candidates",
             "n_observed": len(known_entries), "n_missing": n_unknown_entries, "n_total_ex_ante": n_all_entries,
             "known_events": n_events, "estimate": float(known_entries.entry_success.mean()),
             "lower_bound": n_events / n_all_entries,
             "upper_bound": (n_events + n_unknown_entries) / n_all_entries,
             "assumption": "Lower bound treats every unknown label as failure; upper bound treats every unknown label as success."}]
    # Descriptive covariate composition of all unknown labels relative to observed outcomes.
    descriptor_cols = ["log_gdp", "log_destination_import_demand", "historical_export_value", "historical_market_share"]
    for year, group in candidates.groupby("outcome_year"):
        for status, d in [("observed", group.loc[group.label_observed]), ("missing", group.loc[~group.label_observed])]:
            for col in descriptor_cols:
                rows.append({"analysis": "outcome_availability_composition", "scope": f"year={year}; status={status}; variable={col}",
                             "n_observed": int(d.label_observed.sum()), "n_missing": int((~d.label_observed).sum()),
                             "n_total_ex_ante": len(d), "known_events": np.nan,
                             "estimate": float(d[col].mean()) if len(d) else np.nan,
                             "lower_bound": np.nan, "upper_bound": np.nan,
                             "assumption": "Descriptive only; missing outcome labels remain NA."})
        train_history = panel.loc[panel.outcome_observed.astype(bool) & panel.outcome_year.le(year - LAG)]
        dest_gdp = train_history[["destination_iso3", "log_gdp"]].drop_duplicates("destination_iso3").log_gdp.dropna()
        gdp_cut = float(dest_gdp.median()) if len(dest_gdp) else np.nan
        demand_cut = float(train_history.log_destination_import_demand.median())
        for variable, column, cutoff in [("destination_gdp", "log_gdp", gdp_cut),
                                         ("historical_import_demand", "log_destination_import_demand", demand_cut)]:
            if not np.isfinite(cutoff):
                continue
            labels = np.where(group[column].le(cutoff), "at_or_below_training_median", "above_training_median")
            temp = group.assign(size_group=labels)
            for size_group, d in temp.groupby("size_group"):
                n_missing = int((~d.label_observed).sum())
                rows.append({"analysis": "missingness_by_training_size_group",
                             "scope": f"year={year}; size_variable={variable}; group={size_group}; cutoff={cutoff}",
                             "n_observed": int(d.label_observed.sum()), "n_missing": n_missing,
                             "n_total_ex_ante": len(d), "known_events": np.nan,
                             "estimate": n_missing / len(d) if len(d) else np.nan,
                             "lower_bound": np.nan, "upper_bound": np.nan,
                             "assumption": "Size split uses only training-period covariates; reported missing-label rate is descriptive."})
    # Top-10 entry performance is partially identified on the complete ex-ante pools.
    e2 = pd.read_parquet(OUT / "predictions/E2_neural_predictions.parquet")
    for (model, seed, year, hs2), pool in e2.groupby(["model", "seed", "outcome_year", "hs2"], sort=False):
        ranked = rank_pool(pool, "score")
        top = ranked.head(10)
        denom = min(10, len(ranked))
        known_success = int(top.loc[top.label_observed, "entry_success"].fillna(0).sum())
        missing_top = int((~top.label_observed).sum())
        rows.append({"analysis": "entry_precision_at_10_full_universe_bounds",
                     "scope": f"model={model}; seed={seed}; year={year}; hs2={hs2}",
                     "n_observed": int(pool.label_observed.sum()), "n_missing": int((~pool.label_observed).sum()),
                     "n_total_ex_ante": len(pool), "known_events": int(pool.loc[pool.label_observed, "entry_success"].sum()),
                     "estimate": known_success / denom if denom else np.nan,
                     "lower_bound": known_success / denom if denom else np.nan,
                     "upper_bound": (known_success + missing_top) / denom if denom else np.nan,
                     "assumption": "Full ex-ante Top-10 ranking; selected unknown outcomes may be all failures or all successes."})
    # Stress the forecast MAE difference for outcomes in [0, historical training maximum by HS2].
    neural = pd.read_parquet(OUT / "predictions/E1_neural_predictions.parquet")
    for year in YEARS:
        miss = candidates.loc[candidates.outcome_year.eq(year) & ~candidates.label_observed].copy()
        if miss.empty:
            continue
        train = panel.loc[panel.outcome_observed & panel.outcome_year.le(year - LAG)]
        caps = train.groupby("hs2").export_value_usd.max()
        miss["stress_cap_usd"] = miss.hs2.map(caps).fillna(0).clip(lower=0)
        actual_known = candidates.loc[candidates.outcome_year.eq(year) & candidates.label_observed,
                                      [*KEYS, "export_value_usd"]]
        base = forecast_base.loc[forecast_base.outcome_year.eq(year)]
        for model_a in ["MLP", "TabM"]:
            preds_a = neural.loc[neural.model.eq(model_a) & neural.outcome_year.eq(year),
                                 [*KEYS, "seed", "predicted_export_value_usd"]]
            for model_b in ["Persistence", "Gravity_PPML"]:
                preds_b = base.loc[base.model.eq(model_b), [*KEYS, "predicted_usd"]]
                for seed, ap in preds_a.groupby("seed"):
                    pair_known = ap.merge(preds_b, on=KEYS, validate="one_to_one").merge(actual_known, on=KEYS, validate="one_to_one")
                    known_diff = np.abs(pair_known.predicted_export_value_usd - pair_known.export_value_usd) - np.abs(pair_known.predicted_usd - pair_known.export_value_usd)
                    mp = miss.merge(ap[[*KEYS, "predicted_export_value_usd"]], on=KEYS, validate="one_to_one").merge(
                        preds_b, on=KEYS, validate="one_to_one")
                    lower_total, upper_total = float(known_diff.sum()), float(known_diff.sum())
                    for row in mp.itertuples(index=False):
                        cap = float(row.stress_cap_usd)
                        candidates_y = np.unique([0.0, cap, min(max(float(row.predicted_export_value_usd), 0), cap),
                                                  min(max(float(row.predicted_usd), 0), cap)])
                        differences = [abs(float(row.predicted_export_value_usd) - y) - abs(float(row.predicted_usd) - y)
                                       for y in candidates_y]
                        lower_total += min(differences)
                        upper_total += max(differences)
                    total_n = len(candidates.loc[candidates.outcome_year.eq(year)])
                    rows.append({"analysis": "forecast_mae_difference_training_max_stress",
                                 "scope": f"year={year}; {model_a} seed={seed} minus {model_b}",
                                 "n_observed": len(actual_known), "n_missing": len(miss), "n_total_ex_ante": total_n,
                                 "known_events": np.nan,
                                 "estimate": float(known_diff.mean()),
                                 "lower_bound": lower_total / total_n, "upper_bound": upper_total / total_n,
                                 "assumption": "Stress interval only: each missing USD outcome ranges from zero to the maximum observed training export for its HS2 chapter. This is not a guaranteed economic bound."})
    # Bounded NDCG sensitivity comes from full-pool lower and upper bounds already saved.
    e3path = OUT / "results/E3_ranking_pool_metrics.csv"
    if e3path.exists():
        e3 = pd.read_csv(e3path)
        for (model, task), d in e3.groupby(["ranking_model", "task"]):
            if task == "Task_A_new_entry":
                low_col, high_col = ("full_population_ndcg10_lower_bound_entry_binary",
                                     "full_population_ndcg10_upper_bound_entry_binary")
                analysis_name = "ranking_ndcg_entry_binary_full_universe_bound"
                bound_note = "Pool-wise binary-entry NDCG bounds; missing outcomes are retained in the ex-ante ranking and the upper bound is conservative [0,1]."
            else:
                low_col, high_col = ("full_population_ndcg10_lower_bound_capped_gain",
                                     "full_population_ndcg10_upper_bound_capped_gain")
                analysis_name = "ranking_ndcg_capped_gain_full_universe_bound"
                bound_note = "Pool-wise capped-gain NDCG bounds; missing outcomes are retained in the ex-ante ranking and the upper bound is conservative [0,1]."
            rows.append({"analysis": analysis_name,
                         "scope": f"model={model}; task={task}", "n_observed": int(d.n_observed.sum()),
                         "n_missing": int(d.n_missing.sum()), "n_total_ex_ante": int(d.n_exante.sum()),
                         "known_events": np.nan, "estimate": np.nan,
                         "lower_bound": float(d[low_col].mean()),
                         "upper_bound": float(d[high_col].mean()),
                         "assumption": bound_note})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "results/missing_label_sensitivity.csv", index=False, na_rep="")
    report = ["# Missing-label sensitivity", "", "**Status: COMPLETED.** No missing label is recoded to zero in the primary evaluation.", "",
              f"The frozen ex-ante universe contains {len(candidates):,} forecast candidates, of which {int(candidates.label_observed.sum()):,} have observed outcomes and {len(unknown):,} do not. Among {n_all_entries:,} ex-ante inactive-entry candidates, {len(known_entries):,} outcomes are known, {n_unknown_entries:,} remain unknown, and {n_events:,} observed entries occurred.", "",
              f"Observed entry prevalence among labeled candidates is {known_entries.entry_success.mean():.3%}. The assumption-free binary prevalence interval is [{n_events/n_all_entries:.3%}, {(n_events+n_unknown_entries)/n_all_entries:.3%}] if every unknown is respectively a failure or a success.", "",
              "The table provides annual observed-versus-missing covariate composition, full-population Top-10 entry-rate bounds for every model-seed-pool, capped-gain NDCG bounds, and a forecast-error stress interval. The latter caps an unknown outcome at the historical training maximum for its HS2 chapter; it is an empirical stress scenario, not a guaranteed upper bound. Without a defensible finite outcome cap, full-population USD MAE contrasts are not finitely identified.", "",
              "Complete-case rankings remain conditional on observed labels. WDI/WGI revised historical vintages and the unresolved distinction between true zero imports and unavailable reporting remain limitations."]
    (OUT / "reports/MISSING_LABEL_SENSITIVITY.md").write_text("\n".join(report) + "\n")
    return result


def destination_block_ranking_bootstrap(ranking_frame: pd.DataFrame, reps: int = 1000,
                                        seed: int = 1907) -> pd.DataFrame:
    """Two-way dependence sensitivity: resample destination blocks, then re-rank within each product-year pool.

    Bootstrap draws preserve each sampled destination's rows across all HS2 chapters and years.
    Repeated destination draws are represented as integer multiplicities in each pool.
    This estimates sensitivity to shared-destination dependence; it does not resample individual
    candidates independently and does not create an outer-year generalization interval.
    """
    rng = np.random.default_rng(seed)
    tasks = ["Task_A_new_entry", "Task_B_existing_underexploited"]
    pairs = [("TabM", "Gravity_PPML"), ("TabM", "LightGBM"),
             ("MLP", "Gravity_PPML"), ("MLP", "LightGBM"), ("TabM", "MLP")]
    data = ranking_frame.loc[ranking_frame.task.isin(tasks) & ranking_frame.label_observed.astype(bool)].copy()
    data = data.dropna(subset=["destination_iso3", "opportunity_score", "realized_incremental_export_usd"])
    destinations = np.asarray(sorted(data.destination_iso3.astype(str).unique()))
    dest_index = {c: i for i, c in enumerate(destinations)}
    data["destination_iso3"] = data.destination_iso3.astype(str)
    data["destination_index"] = data.destination_iso3.map(dest_index).astype(int)
    data["gain"] = capped_export_grade(data.realized_incremental_export_usd.to_numpy(float), ENTRY_THRESHOLD_USD)

    # Cache score ordering and relevance for each model/seed/pool. Repeated draws are weights.
    cache = {}
    for (task, year, hs2, model, model_seed), g in data.groupby(
            ["task", "evaluation_year", "hs2", "ranking_model", "seed"], sort=False):
        g = g.sort_values(["opportunity_score", "destination_iso3"], ascending=[False, True], kind="mergesort")
        gains = g.entry_success.astype(int).to_numpy(np.int8) if task == "Task_A_new_entry" else g.gain.to_numpy(np.int8)
        cache[(task, int(year), str(hs2), model, int(model_seed))] = (
            g.destination_index.to_numpy(np.int32), gains)

    def observed_ndcg(dest_ids: np.ndarray, gain: np.ndarray) -> float:
        return ndcg_at_k(gain, 10)

    out = []
    bootstrap_draws = np.stack([
        np.bincount(rng.integers(0, len(destinations), size=len(destinations)), minlength=len(destinations))
        for _ in range(reps)
    ]).astype(np.int16, copy=False)
    bootstrap_ndcg = {}
    for key, (destination_ids, relevance) in cache.items():
        if key[3] in {"MLP", "TabM", "Gravity_PPML", "LightGBM"}:
            bootstrap_ndcg[key] = _weighted_ndcg_draws(destination_ids, relevance, bootstrap_draws, 10)
    for task in tasks:
        for model_a, model_b in pairs:
            seeds_a = SEEDS if model_a in {"MLP", "TabM"} else [0]
            seeds_b = SEEDS if model_b in {"MLP", "TabM"} else [0]
            paired_seeds = list(zip(seeds_a, seeds_b)) if len(seeds_a) == len(seeds_b) else [(s, seeds_b[0]) for s in seeds_a]
            pool_keys = sorted({k[:3] for k in cache if k[0] == task and k[3] == model_a} &
                               {k[:3] for k in cache if k[0] == task and k[3] == model_b})
            # Average seed-specific metric differences within each observed product-year pool.
            base_pool_diffs = {}
            for pool in pool_keys:
                differences = []
                for sa, sb in paired_seeds:
                    a = cache.get((*pool, model_a, int(sa)))
                    b = cache.get((*pool, model_b, int(sb)))
                    if a is None or b is None:
                        continue
                    # The same outcome universe is required for every paired metric.
                    if not (np.array_equal(np.sort(a[0]), np.sort(b[0])) and np.array_equal(
                            a[1][np.argsort(a[0], kind="mergesort")], b[1][np.argsort(b[0], kind="mergesort")])):
                        raise AssertionError(f"Ranking bootstrap found different candidate/relevance sets: {task}/{pool}/{model_a}/{model_b}")
                    ndcg_a, ndcg_b = observed_ndcg(*a), observed_ndcg(*b)
                    if np.isfinite(ndcg_a) and np.isfinite(ndcg_b):
                        differences.append(ndcg_a - ndcg_b)
                if differences:
                    base_pool_diffs[pool] = float(np.mean(differences))
            if not base_pool_diffs:
                out.append({"task": task, "model_a": model_a, "model_b": model_b,
                            "difference_a_minus_b": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
                            "valid_paired_pools": 0, "destination_blocks": len(destinations), "bootstrap_reps": reps})
                continue
            point = float(np.mean(list(base_pool_diffs.values())))
            pool_draws = []
            for pool in base_pool_diffs:
                seed_draws = []
                for sa, sb in paired_seeds:
                    va = bootstrap_ndcg.get((*pool, model_a, int(sa)))
                    vb = bootstrap_ndcg.get((*pool, model_b, int(sb)))
                    if va is None or vb is None:
                        continue
                    paired = np.isfinite(va) & np.isfinite(vb)
                    seed_draws.append(np.where(paired, va - vb, np.nan))
                if seed_draws:
                    seed_matrix = np.stack(seed_draws)
                    n_valid = np.isfinite(seed_matrix).sum(axis=0)
                    pool_draws.append(np.divide(np.nansum(seed_matrix, axis=0), n_valid,
                                                out=np.full(reps, np.nan), where=n_valid > 0))
            if pool_draws:
                pool_matrix = np.stack(pool_draws)
                n_valid = np.isfinite(pool_matrix).sum(axis=0)
                draws = np.divide(np.nansum(pool_matrix, axis=0), n_valid,
                                  out=np.full(reps, np.nan), where=n_valid > 0)
            else:
                draws = np.full(reps, np.nan)
            draws = draws[np.isfinite(draws)]
            lo, hi = np.quantile(draws, [0.025, 0.975]) if len(draws) else (np.nan, np.nan)
            out.append({"task": task, "model_a": model_a, "model_b": model_b,
                        "difference_a_minus_b": point, "ci95_low": float(lo), "ci95_high": float(hi),
                        "valid_paired_pools": len(base_pool_diffs), "destination_blocks": len(destinations),
                        "bootstrap_reps_requested": reps, "bootstrap_reps_valid": len(draws),
                        "method": "destination-cluster bootstrap; resample destinations jointly across product-year pools; re-rank weighted candidates; model seeds paired then averaged",
                        "interpretation": "dependence sensitivity interval only; not a time-generalization interval"})
    result = pd.DataFrame(out)
    result.to_csv(OUT / "results/statistical_comparisons_E3_destination_bootstrap.csv", index=False, na_rep="")
    return result
