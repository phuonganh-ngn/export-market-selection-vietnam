from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .candidate_protocol import (CAT_FEATURES, FEATURES, HERE, LAG, PROJECT,
                                 training_rows)
from .lambdamart import feature_frame, _cache_paths, fit_final
from .ranking_metrics import pool_metrics

GROUPS = {
    "historical_trade": ["log_lagged_export_value", "historical_export_presence", "historical_market_share",
                         "lagged_export_growth", "historical_export_volatility"],
    "destination_import_demand": ["log_destination_import_demand", "destination_import_demand_growth",
                                  "destination_supplier_concentration"],
    "macroeconomic_size": ["log_gdp", "gdp_growth", "log_population", "urban_population_share", "inflation", "trade_openness"],
    "purchasing_power": ["log_gdp_per_capita"],
    "geography": ["log_distance", "contiguous", "common_official_language", "colonial_link", "common_colonizer", "landlocked"],
    "institutional_quality": ["government_effectiveness", "regulatory_quality", "rule_of_law", "political_stability"],
    "digitalization": ["internet_usage", "digitalization_missing_indicator"],
    "product_identity": ["hs2"],
    "destination_identity": ["destination_iso3"],
}


def pairwise_contribution_difference(contrib_a, contrib_b, score_a: float, score_b: float):
    difference = np.asarray(contrib_a, dtype=float) - np.asarray(contrib_b, dtype=float)
    score_difference = float(score_a - score_b)
    reconstructed = float(difference.sum())
    return difference, score_difference, reconstructed, abs(reconstructed - score_difference)


def _feature_group(feature: str) -> str:
    for group, cols in GROUPS.items():
        if feature in cols:
            return group
    return "other"


def _metrics_with_score(frame: pd.DataFrame, score: np.ndarray, task: str) -> dict:
    d = frame.copy()
    d["_score"] = score
    return pool_metrics(d, "_score", task)


def run_explanations(panel: pd.DataFrame, test: pd.DataFrame, manifest: pd.DataFrame):
    pair_rows, group_rows, fidelity_rows, stability_raw = [], [], [], []
    # One numerical feature-group perturbation per fitted 2024/2023/2022 model; all are fixed-model probes.
    perturb_rows = []
    for task in ["Task_A_new_entry", "Task_B_existing_underexploited"]:
        for year in [2022, 2023, 2024]:
            candidates = manifest.loc[manifest.task.eq(task) & manifest.outcome_year.eq(year), "candidate_id"].unique()
            raw = panel.loc[panel.candidate_id.isin(candidates)].copy()
            raw = raw.sort_values(["hs2", "destination_iso3"], kind="mergesort").reset_index(drop=True)
            if raw.candidate_id.nunique() != len(candidates):
                raise AssertionError("Explainability candidate rows differ from frozen task manifest")
            train = training_rows(panel, task, year)
            for seed in [11, 23, 37, 51, 73]:
                model_path, meta_path = _cache_paths(task, year, seed)
                if not (model_path.exists() and meta_path.exists()):
                    raise FileNotFoundError(f"Missing saved LambdaMART outer model {task}/{year}/seed{seed}")
                model = lgb.Booster(model_file=str(model_path))
                meta = json.loads(meta_path.read_text())
                x = feature_frame(raw, meta["categories"])
                scores = np.asarray(model.predict(x, num_threads=4), dtype=float)
                contrib = np.asarray(model.predict(x, pred_contrib=True, num_threads=4), dtype=float)
                if contrib.shape != (len(raw), len(FEATURES) + 1):
                    raise AssertionError(f"Unexpected pred_contrib shape: {contrib.shape}")
                reconstruction = contrib.sum(axis=1)
                err = np.abs(reconstruction - scores)
                fidelity_rows.append({"task": task, "evaluation_year": year, "seed": seed,
                                      "n_candidates": len(raw), "max_score_reconstruction_error": float(err.max()),
                                      "mean_score_reconstruction_error": float(err.mean()),
                                      "pairwise_ordering_fidelity": 1.0 if err.max() < 1e-6 else np.nan,
                                      "output_scale": "raw LambdaMART ranking score; additive tree contribution; final column is expected value"})

                feat_contrib = pd.DataFrame(contrib[:, :-1], columns=FEATURES)
                for group, columns in GROUPS.items():
                    existing = [c for c in columns if c in feat_contrib.columns]
                    if not existing:
                        continue
                    values = feat_contrib[existing].sum(axis=1).to_numpy(float)
                    group_rows.append({"task": task, "evaluation_year": year, "seed": seed,
                                       "feature_group": group, "features": ";".join(existing),
                                       "mean_absolute_group_contribution": float(np.mean(np.abs(values))),
                                       "mean_signed_group_contribution": float(np.mean(values)),
                                       "score_scale": "raw LambdaMART score", "interpretation": "predictive attribution, not causal"})
                    stability_raw.append({"task": task, "evaluation_year": year, "seed": seed,
                                          "feature_group": group,
                                          "mean_absolute_group_contribution": float(np.mean(np.abs(values)))})

                if seed == 11:
                    # Contrastive pair: the top destination versus rank 11 in each product-year query.
                    score_frame = raw[["candidate_id", "destination_iso3", "destination_name", "hs2", "outcome_year"]].copy()
                    score_frame["score"] = scores
                    score_frame["label_observed"] = raw.label_observed.to_numpy(bool)
                    for c in ["entry_success", "realized_incremental_export_usd"]:
                        score_frame[c] = raw[c].to_numpy()
                    for (hs2, qyear), q in score_frame.groupby(["hs2", "outcome_year"], sort=False):
                        q = q.sort_values(["score", "destination_iso3"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
                        if len(q) < 2:
                            continue
                        ia, ib = 0, min(10, len(q) - 1)
                        ra, rb = q.iloc[ia], q.iloc[ib]
                        ca, cb = raw.index[raw.candidate_id.eq(ra.candidate_id)][0], raw.index[raw.candidate_id.eq(rb.candidate_id)][0]
                        phi, score_diff, contribution_sum, recon_error = pairwise_contribution_difference(
                            contrib[ca, :-1], contrib[cb, :-1], ra.score, rb.score)
                        for feature, value in zip(FEATURES, phi):
                            pair_rows.append({"task": task, "evaluation_year": year, "hs2": str(hs2).zfill(2),
                                              "rank_a": 1, "destination_a": ra.destination_iso3,
                                              "destination_a_name": ra.destination_name,
                                              "rank_b": int(ib + 1), "destination_b": rb.destination_iso3,
                                              "destination_b_name": rb.destination_name,
                                              "score_a": float(ra.score), "score_b": float(rb.score),
                                              "score_difference": score_diff, "feature": feature,
                                              "feature_group": _feature_group(feature),
                                              "contribution_difference_a_minus_b": float(value),
                                              "pairwise_contribution_sum": contribution_sum,
                                              "pairwise_reconstruction_error": recon_error,
                                              "pairwise_order_preserved": bool(np.sign(contribution_sum) == np.sign(score_diff))})

                    # Fixed-model group perturbation: replace numeric inputs by outer-training medians.
                    base_scores = scores
                    base_ranks = raw[["candidate_id", "hs2", "outcome_year", "destination_iso3"]].copy()
                    base_ranks["score"] = base_scores
                    for group, columns in GROUPS.items():
                        numeric = [c for c in columns if c in FEATURES and c not in CAT_FEATURES]
                        if not numeric:
                            perturb_rows.append({"task": task, "evaluation_year": year, "seed": seed,
                                                 "feature_group": group, "status": "categorical identity; score contributions only; no off-manifold perturbation"})
                            continue
                        med = train[numeric].apply(pd.to_numeric, errors="coerce").median()
                        changed = raw.copy()
                        for c in numeric:
                            if np.isfinite(med.get(c, np.nan)):
                                changed[c] = float(med[c])
                        x_changed = feature_frame(changed, meta["categories"])
                        changed_scores = np.asarray(model.predict(x_changed, num_threads=4), dtype=float)
                        pert = raw[["candidate_id", "hs2", "outcome_year", "destination_iso3", "label_observed",
                                    "entry_success", "realized_incremental_export_usd"]].copy()
                        pert["score"] = changed_scores
                        for (phs, py), g in base_ranks.groupby(["hs2", "outcome_year"], sort=False):
                            gp = pert.loc[pert.hs2.eq(phs) & pert.outcome_year.eq(py)].copy()
                            gb = g.sort_values(["score", "destination_iso3"], ascending=[False, True], kind="mergesort")
                            gp = gp.sort_values(["score", "destination_iso3"], ascending=[False, True], kind="mergesort")
                            base_top5, base_top10 = set(gb.head(5).destination_iso3), set(gb.head(10).destination_iso3)
                            top5, top10 = set(gp.head(5).destination_iso3), set(gp.head(10).destination_iso3)
                            # Map average absolute rank displacement for a common candidate set.
                            base_pos = {c: i + 1 for i, c in enumerate(gb.destination_iso3)}
                            pert_pos = {c: i + 1 for i, c in enumerate(gp.destination_iso3)}
                            displacement = np.mean([abs(base_pos[c] - pert_pos[c]) for c in base_pos]) if base_pos else np.nan
                            base_pool = raw.loc[raw.hs2.eq(phs) & raw.outcome_year.eq(py),
                                                ["candidate_id", "destination_iso3", "label_observed", "entry_success",
                                                 "realized_incremental_export_usd"]].copy()
                            base_pool["score"] = base_scores[raw.index[raw.hs2.eq(phs) & raw.outcome_year.eq(py)]]
                            met_base = pool_metrics(base_pool, "score", task)
                            met = pool_metrics(gp, "score", task)
                            perturb_rows.append({"task": task, "evaluation_year": year, "hs2": str(phs).zfill(2),
                                                 "seed": seed, "feature_group": group, "features_perturbed": ";".join(numeric),
                                                 "status": "fixed_model_training_median_replacement",
                                                 "top5_jaccard": len(base_top5 & top5) / len(base_top5 | top5) if base_top5 | top5 else np.nan,
                                                 "top10_jaccard": len(base_top10 & top10) / len(base_top10 | top10) if base_top10 | top10 else np.nan,
                                                 "mean_absolute_rank_displacement": float(displacement),
                                                 "baseline_ndcg10": met_base.get("ndcg_at_10"),
                                                 "perturbed_ndcg10": met.get("ndcg_at_10"),
                                                 "interpretation": "fixed-model input sensitivity; median replacement may be off-manifold, not causal"})

    pair = pd.DataFrame(pair_rows)
    groups = pd.DataFrame(group_rows)
    fidelity = pd.DataFrame(fidelity_rows)
    perturb = pd.DataFrame(perturb_rows)
    pair.to_csv(HERE / "results/E8_pairwise_explanations.csv", index=False, na_rep="")
    groups.to_csv(HERE / "results/E8_feature_group_rank_effects.csv", index=False, na_rep="")
    fidelity.to_csv(HERE / "results/E8_explanation_fidelity.csv", index=False, na_rep="")

    # Seed stability of feature-group attribution ranks, plus temporal stability of fold-level profiles.
    stab_rows = []
    stab = pd.DataFrame(stability_raw)
    for (task, year), g in stab.groupby(["task", "evaluation_year"], sort=False):
        pivot = g.pivot(index="feature_group", columns="seed", values="mean_absolute_group_contribution")
        seeds = list(pivot.columns)
        for i, a in enumerate(seeds):
            ra = pivot[a].rank(ascending=False, method="average")
            for b in seeds[i + 1:]:
                rb = pivot[b].rank(ascending=False, method="average")
                kt = kendalltau(ra, rb).statistic if len(pivot) > 2 else np.nan
                stab_rows.append({"task": task, "comparison_type": "seed_pair", "year_a": year, "year_b": year,
                                  "seed_a": a, "seed_b": b, "group_rank_kendall_tau": float(kt),
                                  "feature_groups": len(pivot)})
    year_profiles = stab.groupby(["task", "evaluation_year", "feature_group"], as_index=False).agg(
        mean_absolute_group_contribution=("mean_absolute_group_contribution", "mean"))
    for task, g in year_profiles.groupby("task"):
        pivot = g.pivot(index="feature_group", columns="evaluation_year", values="mean_absolute_group_contribution")
        for ya, yb in [(2022, 2023), (2023, 2024), (2022, 2024)]:
            if ya in pivot and yb in pivot:
                kt = kendalltau(pivot[ya].rank(ascending=False), pivot[yb].rank(ascending=False)).statistic
                stab_rows.append({"task": task, "comparison_type": "outer_year_pair", "year_a": ya, "year_b": yb,
                                  "seed_a": np.nan, "seed_b": np.nan, "group_rank_kendall_tau": float(kt),
                                  "feature_groups": len(pivot)})
    stab_out = pd.DataFrame(stab_rows)
    stab_out.to_csv(HERE / "results/E8_feature_attribution_stability.csv", index=False, na_rep="")
    perturb.to_csv(HERE / "results/E8_fixed_model_group_perturbation.csv", index=False, na_rep="")
    topk_perturb = perturb.loc[perturb.status.eq("fixed_model_training_median_replacement")].copy()
    topk_summary = topk_perturb.groupby(
        ["task", "evaluation_year", "feature_group"], as_index=False
    ).agg(mean_top5_jaccard=("top5_jaccard", "mean"), mean_top10_jaccard=("top10_jaccard", "mean"),
         mean_rank_displacement=("mean_absolute_rank_displacement", "mean"),
         mean_baseline_ndcg10=("baseline_ndcg10", "mean"),
         mean_perturbed_ndcg10=("perturbed_ndcg10", "mean"), pools=("top10_jaccard", "count"))
    topk_summary.to_csv(HERE / "results/E8_topk_explanation_stability.csv", index=False, na_rep="")
    return pair, groups, fidelity, stab_out, perturb


def run_retrained_ablation(panel: pd.DataFrame, manifest: pd.DataFrame, ranked: pd.DataFrame):
    """One-seed, fixed-development-config retrained feature-group ablation."""
    output, prediction_frames = [], []
    for task in ["Task_A_new_entry", "Task_B_existing_underexploited"]:
        for year in [2022, 2023, 2024]:
            train = training_rows(panel, task, year)
            test_ids = manifest.loc[manifest.task.eq(task) & manifest.outcome_year.eq(year), "candidate_id"].unique()
            test = panel.loc[panel.candidate_id.isin(test_ids)].copy()
            full_model_path, full_meta_path = _cache_paths(task, year, 11)
            full_meta = json.loads(full_meta_path.read_text())
            config = full_meta["config"]
            best_iteration = int(full_meta["best_iteration"])
            base = ranked.loc[ranked.task.eq(task) & ranked.model.eq("LambdaMART")
                              & ranked.seed.eq(11) & ranked.outcome_year.eq(year),
                              ["candidate_id", "destination_iso3", "hs2", "outcome_year", "score",
                               "label_observed", "entry_success", "realized_incremental_export_usd"]].copy()
            groups = list(GROUPS)
            for group in groups:
                path = HERE / "models" / f"lambdamart_ablation_{task}_{year}_{group}_seed11.txt"
                meta_path = HERE / "models" / f"lambdamart_ablation_{task}_{year}_{group}_seed11.json"
                drop = GROUPS[group]
                if path.exists() and meta_path.exists():
                    model = lgb.Booster(model_file=str(path))
                    meta = json.loads(meta_path.read_text())
                    features = meta["feature_names"]
                    cats = meta["categorical_features"]
                    levels = meta["categories"]
                    runtime = 0.0
                else:
                    model, levels, cutoffs, runtime, tr_sorted, labels, query_groups, features, cats = fit_final(
                        train, task, config, best_iteration, 11, year, drop_features=drop)
                    model.booster_.save_model(str(path))
                    meta = {"task": task, "evaluation_year": year, "seed": 11, "dropped_group": group,
                            "dropped_features": drop, "feature_names": features, "categorical_features": cats,
                            "categories": levels, "training_rows": len(tr_sorted), "training_groups": len(query_groups),
                            "training_outcome_max": int(train.outcome_year.max()), "best_iteration_reused_from_full_model": best_iteration,
                            "training_cutoffs": None if cutoffs is None else cutoffs.tolist(),
                            "training_runtime_seconds": runtime,
                            "selection_note": "fixed configuration and iteration count chosen on full-feature chronological development fold; ablation is exploratory"}
                    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
                scores = np.asarray(model.predict(feature_frame(test, levels, features, cats), num_threads=4), dtype=float)
                pred = test[["candidate_id", "destination_iso3", "hs2", "outcome_year", "label_observed",
                             "entry_success", "realized_incremental_export_usd"]].copy()
                pred["task"] = task
                pred["feature_group_removed"] = group
                pred["seed"] = 11
                pred["score"] = scores
                prediction_frames.append(pred)
                compare = base.merge(pred[["candidate_id", "score"]], on="candidate_id", suffixes=("_full", "_ablated"), validate="one_to_one")
                for (hs2, qyear), q in compare.groupby(["hs2", "outcome_year"], sort=False):
                    full_order = q.sort_values(["score_full", "destination_iso3"], ascending=[False, True], kind="mergesort")
                    abl_order = q.sort_values(["score_ablated", "destination_iso3"], ascending=[False, True], kind="mergesort")
                    full_pos = {d: i + 1 for i, d in enumerate(full_order.destination_iso3.astype(str))}
                    abl_pos = {d: i + 1 for i, d in enumerate(abl_order.destination_iso3.astype(str))}
                    displacement = float(np.mean([abs(full_pos[d] - abl_pos[d]) for d in full_pos])) if full_pos else np.nan
                    for k in [5, 10]:
                        a = set(full_order.head(k).destination_iso3)
                        b = set(abl_order.head(k).destination_iso3)
                        output.append({"task": task, "evaluation_year": year, "hs2": str(hs2).zfill(2),
                                       "feature_group_removed": group, "seed": 11, "k": k,
                                       "n_candidates": len(q), "topk_jaccard_vs_full": len(a & b) / len(a | b) if a | b else np.nan,
                                       "topk_overlap_vs_full": len(a & b),
                                       "mean_absolute_rank_displacement": displacement,
                                       "ndcg10_full": _pool_ndcg(full_order, "score_full", task),
                                       "ndcg10_retrained_without_group": _pool_ndcg(abl_order, "score_ablated", task),
                                       "fit_runtime_seconds": runtime,
                                       "interpretation": "retrained feature ablation; fixed full-model development configuration, exploratory and noncausal"})
    preds = pd.concat(prediction_frames, ignore_index=True)
    table = pd.DataFrame(output)
    preds.to_parquet(HERE / "predictions/E8_retrained_ablation_predictions.parquet", index=False)
    table.to_csv(HERE / "results/E8_retrained_feature_ablation.csv", index=False, na_rep="")
    return table, preds


def _pool_ndcg(frame: pd.DataFrame, score: str, task: str) -> float:
    x = frame.sort_values([score, "destination_iso3"], ascending=[False, True], kind="mergesort")
    x = x.loc[x.label_observed.astype(bool)]
    if x.empty:
        return np.nan
    if task == "Task_A_new_entry":
        gains = x.entry_success.astype(float).to_numpy()
    else:
        inc = np.maximum(0.0, x.realized_incremental_export_usd.to_numpy(float))
        grade = np.minimum(4, np.floor(np.log10(1 + inc / 50_000.0)))
        gains = np.exp2(grade) - 1.0
    from .ranking_metrics import ndcg_from_gains
    return ndcg_from_gains(gains, gains, 10)
