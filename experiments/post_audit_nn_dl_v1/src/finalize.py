from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import HERE, YEARS, SEEDS

OUT = HERE
RESULTS = OUT / "results"
REPORTS = OUT / "reports"
CHECKPOINTS = OUT / "checkpoints"


def _read(name: str) -> pd.DataFrame:
    path = RESULTS / name
    return pd.read_csv(path, dtype={"hs2": str}) if path.exists() else pd.DataFrame()


def _fmt(value, digits=4):
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        if abs(value) >= 1e6:
            return f"{value:,.0f}"
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(frame: pd.DataFrame, columns=None, digits=4) -> str:
    if frame.empty:
        return "*(No computed rows.)*"
    d = frame[columns].copy() if columns else frame.copy()
    for c in d:
        if pd.api.types.is_numeric_dtype(d[c]):
            if c.lower().startswith("p_") or c.lower() == "p_holm":
                d[c] = d[c].map(lambda x: "<0.0001" if pd.notna(x) and 0 <= x < 0.0001 else _fmt(x, digits))
            else:
                d[c] = d[c].map(lambda x: _fmt(x, digits))
    return d.to_markdown(index=False)


def build_compute_cost() -> pd.DataFrame:
    rows = []
    model_dir = CHECKPOINTS / "models"
    for path in sorted(model_dir.glob("*.json")):
        try:
            x = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not {"model", "task", "seed", "evaluation_year"}.issubset(x):
            continue
        rows.append({"record_type": "outer_refit", "model": x["model"], "task": x["task"],
                     "evaluation_year": int(x["evaluation_year"]), "seed": int(x["seed"]),
                     "training_runtime_seconds": float(x.get("training_runtime_seconds", np.nan)),
                     "inference_runtime_seconds": float(x.get("inference_runtime_seconds", np.nan)),
                     "epochs": int(x.get("epochs", np.nan)), "parameter_count": int(x.get("parameter_count", np.nan)),
                     "training_rows": int(x.get("training_rows", np.nan)),
                     "training_outcome_max": int(x.get("training_outcome_max", np.nan)),
                     "config": json.dumps(x.get("config", {}), sort_keys=True),
                     "torch_version": x.get("torch_version", ""), "device": "cpu", "torch_threads": 4})
    tune_dir = CHECKPOINTS / "tuning"
    for path in sorted(tune_dir.glob("*.json")):
        try:
            x = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        configs = x.get("configurations", [])
        rows.append({"record_type": "chronological_tuning", "model": x.get("model", ""),
                     "task": x.get("task", ""), "evaluation_year": int(x.get("evaluation_year", -1)),
                     "seed": 11, "training_runtime_seconds": float(sum(c.get("runtime_seconds", 0) for c in configs)),
                     "inference_runtime_seconds": np.nan, "epochs": int(x.get("best_epoch", 0)),
                     "parameter_count": int(next((c.get("parameter_count") for c in configs
                                                   if c.get("config") == x.get("best_config")), 0)),
                     "training_rows": int(x.get("inner_train_n", 0)),
                     "training_outcome_max": int(x.get("inner_validation_year", -1)) - 2,
                     "config": json.dumps(x.get("best_config", {}), sort_keys=True),
                     "tuning_configurations": len(configs),
                     "validation_year": int(x.get("inner_validation_year", -1)),
                     "validation_score": float(x.get("validation_score", np.nan)),
                     "device": "cpu", "torch_threads": 4})
    cost = pd.DataFrame(rows)
    if not cost.empty:
        memory_path = OUT / "logs/regression_process_memory.csv"
        peak_rss_mb = np.nan
        if memory_path.exists():
            memory = pd.read_csv(memory_path)
            if "high_water_rss_kb" in memory and len(memory):
                peak_rss_mb = float(memory.high_water_rss_kb.max()) / 1024.0
        cost["peak_process_rss_mb"] = peak_rss_mb
        cost["peak_memory_scope"] = "whole sequential regression stage; process high-water RSS"
        cost.to_csv(RESULTS / "compute_cost.csv", index=False, na_rep="")
    return cost


def build_seed_stability() -> pd.DataFrame:
    rows = []
    e2 = _read("E2_discovery_seed_metrics.csv")
    if not e2.empty:
        for r in e2.itertuples(index=False):
            for metric in ["pr_auc", "roc_auc", "precision_at_5", "precision_at_10", "recall_at_10", "ndcg_at_10"]:
                rows.append({"task": "E2_discovery", "model": r.model, "seed": r.seed,
                             "metric": metric, "value": getattr(r, metric)})
    e1 = _read("seed_stability_forecasting.csv")
    if not e1.empty:
        for r in e1.itertuples(index=False):
            for metric in ["mae_usd", "rmse_usd", "mae_log1p", "wape", "mae_zero_flow_usd", "mae_positive_flow_usd"]:
                rows.append({"task": "E1_forecasting", "model": r.model, "seed": r.seed,
                             "metric": metric, "value": getattr(r, metric)})
    e3 = _read("E3_ranking_pool_metrics.csv")
    if not e3.empty:
        sm = e3.groupby(["ranking_model", "task", "seed"], as_index=False).agg(
            ndcg10_entry_binary=("ndcg10_entry_binary", "mean"),
            ndcg10_capped_grade=("ndcg10_capped_grade", "mean"),
            ndcg10_log1p_gain=("ndcg10_log1p_gain", "mean"),
            top10_incremental_usd=("observed_incremental_usd_top10_full_rank_partial", "mean"))
        for r in sm.itertuples(index=False):
            for metric in ["ndcg10_entry_binary", "ndcg10_capped_grade", "ndcg10_log1p_gain", "top10_incremental_usd"]:
                rows.append({"task": f"E3_{r.task}", "model": r.ranking_model, "seed": r.seed,
                             "metric": metric, "value": getattr(r, metric)})
    detail = pd.DataFrame(rows)
    if not detail.empty:
        detail.to_csv(RESULTS / "seed_stability.csv", index=False, na_rep="")
        summary = detail.groupby(["task", "model", "metric"], as_index=False).agg(
            mean_across_seeds=("value", "mean"), sd_across_seeds=("value", "std"),
            minimum=("value", "min"), maximum=("value", "max"), seed_values=("seed", "nunique"))
        summary.to_csv(RESULTS / "seed_stability_summary.csv", index=False, na_rep="")
    return detail


def combine_statistical_comparisons() -> pd.DataFrame:
    frames = []
    for file, label in [("statistical_comparisons_E2.csv", "E2_discovery"),
                        ("statistical_comparisons_E3.csv", "E3_ranking"),
                        ("statistical_comparisons_E3_matched_mae.csv", "E3_matched_forecast_ranking"),
                        ("statistical_comparisons_E1.csv", "E1_forecasting")]:
        d = _read(file)
        if not d.empty:
            d.insert(0, "task_family", label)
            frames.append(d)
    result = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not result.empty:
        result.to_csv(RESULTS / "statistical_comparisons.csv", index=False, na_rep="")
    return result


def _write_e2_report():
    e2, tests, by_year = _read("E2_neural_discovery.csv"), _read("statistical_comparisons_E2.csv"), _read("E2_discovery_by_year.csv")
    if e2.empty:
        return
    primary = e2.sort_values("pr_auc", ascending=False)
    table_cols = [c for c in ["model", "pr_auc", "precision_at_10", "ndcg_at_10", "pooled_prevalence",
                              "observed_candidates", "positive_events", "seeds", "valid_pools",
                              "valid_pr_auc_pools", "valid_ndcg_at_10_pools"] if c in primary]
    text = ["# E2 — New-market discovery", "", "**Status: COMPLETED.** All classifiers use the frozen two-year-lag ex-ante entry universe. Unknown outcomes remain NA; metrics are complete-case and conditional on observed labels.", "",
            "## Primary outcomes", "", "The primary target is a previously inactive destination–HS2 relationship reaching USD 50,000 in the outcome year after the audited three-year prior inactivity rule. The E2 macro averages are within HS2 × year pools; valid pool denominators are shown beside each metric. NDCG uses binary entry relevance. Score ties are resolved by ascending ISO3. Neural and logistic class-weighted scores are ranking scores; probability calibration is not claimed.", "",
            _markdown(primary, table_cols), "", "## Paired comparisons", "",
            "Differences are model A minus model B. The prespecified uncertainty procedure resamples whole HS2 blocks, retaining year pools; paired HS2-cluster sign-flip tests are Holm-adjusted within each metric-specific five-comparison family. Seeds are averaged within pool, never treated as economic observations.", "",
            _markdown(tests), "", "## Annual results", "",
            _markdown(by_year.loc[by_year.model.isin(["Gravity_PPML", "LightGBM", "MLP", "TabM"])].sort_values(["model", "evaluation_year"])), "",
            "## Interpretation", "", f"The observed best PR-AUC model is {primary.iloc[0].model} ({primary.iloc[0].pr_auc:.4f}). Conclusions are limited to this retrospective panel, three test years, and labeled candidates. See missing-label sensitivity and the research integrity report."]
    (REPORTS / "E2_DISCOVERY_RESULTS.md").write_text("\n".join(text) + "\n")


def _write_e1_report():
    e1, by_year, tests = _read("E1_neural_forecasting.csv"), _read("E1_forecasting_by_year.csv"), _read("statistical_comparisons_E1.csv")
    if e1.empty:
        return
    ordered = e1.sort_values("mae_usd")
    cols = [c for c in ["model", "mae_usd", "rmse_usd", "mae_log1p", "wape", "mae_zero_flow_usd",
                        "mae_positive_flow_usd", "macro_hs2_year_mae_usd", "n", "seeds", "mae_usd_seed_sd"] if c in ordered]
    text = ["# E1 — Export-value forecasting", "", "**Primary criterion: USD MAE.** The log1p metric is reported separately and is not substituted for the prespecified USD metric. Baseline models and neural models are evaluated on the same observed destination–HS2–year labels; the 559 missing outcome labels remain excluded from primary MAE.", "",
            "## Overall results", "", _markdown(ordered, cols), "", "## Annual results", "",
            _markdown(by_year.sort_values(["evaluation_year", "mae_usd"])), "", "## Paired inference", "",
            "Differences are model A minus model B; negative values favor A. Destination-cluster resampling retains every product–year observation within a sampled destination. Wilcoxon destination-level sign-rank p-values are Holm-adjusted over the four prespecified neural-versus-Persistence/PPML comparisons.", "",
            _markdown(tests), "", "## Interpretation", "",
            f"The lowest observed USD MAE is achieved by {ordered.iloc[0].model} ({ordered.iloc[0].mae_usd:,.0f} USD). Macro-HS2-year performance is reported separately because USD micro-MAE is dominated by high-value products."]
    (REPORTS / "E1_FORECASTING_RESULTS.md").write_text("\n".join(text) + "\n")


def _write_e3_report():
    e3, by_year, tests = _read("E3_neural_ranking.csv"), _read("E3_ranking_by_year.csv"), _read("statistical_comparisons_E3.csv")
    dest = _read("statistical_comparisons_E3_destination_bootstrap.csv")
    matched = _read("E3_matched_forecast_ranking.csv")
    matched_tests = _read("statistical_comparisons_E3_matched_mae.csv")
    if e3.empty:
        return
    text = ["# E3 — Export-market ranking", "", "**Status: COMPLETED on matched ex-ante candidate pools.** Unknown outcome labels are retained in full-pool ranks but are never assigned relevance; primary ranking metrics use the observed-label intersection and are conditional on observed outcomes.", "",
            "Two decision tasks are reported separately. Task A ranks previously inactive markets and uses binary entry success (USD 50,000) for its primary NDCG@10. Task B ranks existing but underexploited markets, defined by training-period HS2 median market share, and uses capped graded positive incremental exports for primary NDCG@10. Log1p and raw-dollar gain definitions are reported as sensitivity metrics. Precision@10 uses the denominator min(10, observed candidate count). Persistence predicts zero incremental opportunity; ties are ordered deterministically by ISO3, so its ranking is a tie-rule benchmark, not a substantive destination preference. For simple GDP/import-demand indicator baselines, missing scores are filled with the relevant training-period median (GDP across destination-years; import demand within HS2); imputation counts are reported. Undefined historical export-trend scores receive the prespecified neutral zero score.", "",
            "Two earlier E3 runs were invalidated during independent validation: the first pooled different HS2 chapters, and the second formed the underexploited-market cutoff from a truncated evaluation-only panel. Both are preserved under `results/invalidated_first_pass/` and excluded from all final metrics. The reported E3 results use corrected HS2 × year pools and full historical information only through each fold's training cutoff; see `E3_POOL_CORRECTION.md`.", ""]
    for task, label, metric in [("Task_A_new_entry", "Task A — new market entry", "ndcg10_entry_binary"),
                               ("Task_B_existing_underexploited", "Task B — existing underexploited markets", "ndcg10_capped_grade")]:
        part = e3.loc[e3.task.eq(task)].sort_values(metric, ascending=False)
        text += [f"## {label}", "", _markdown(part, [c for c in ["ranking_model", metric, "ndcg10_log1p_gain", "ndcg10_raw_gain",
                   "precision10_entry", "precision10_expansion", "observed_incremental_usd_top10_full_rank_partial",
                   "valid_product_year_pools", f"valid_{metric}_pools", "mean_exante_candidates",
                   "unique_unobserved_candidate_labels", "seeds"] if c in part]), ""]
    text += ["## Paired comparisons", "", "For Task A the inferential metric is binary entry NDCG@10; for Task B it is capped-gain NDCG@10. Product-year pools are matched and seed predictions are averaged within a pool. Confidence intervals and paired sign-flip p-values use 96 HS2 chapter blocks; Holm correction is applied within each task-specific five-comparison family. A destination-block bootstrap re-ranks weighted candidates jointly across product-year pools as a dependence sensitivity; it is not a time-generalization interval.", "", _markdown(tests), "", "### Destination-block bootstrap sensitivity", "", _markdown(dest), "", "## Year-specific results", ""]
    for task in ["Task_A_new_entry", "Task_B_existing_underexploited"]:
        met = "ndcg10_entry_binary" if task == "Task_A_new_entry" else "ndcg10_capped_grade"
        part = by_year.loc[by_year.task.eq(task)]
        text += [f"### {task}", "", _markdown(part.sort_values(["evaluation_year", met], ascending=[True, False])), ""]
    text += ["## Interpretation", "", "Top-K observed incremental values are the sum over known outcomes among markets ranked in the full ex-ante top ten; they are partial when a selected label is unknown. They are retrospective realized outcomes, not causal economic gains attributable to recommendations."]
    (REPORTS / "E3_RANKING_RESULTS.md").write_text("\n".join(text) + "\n")
    matched_text = ["# Matched forecasting versus market selection", "", "Forecast MAE and ranking NDCG are recomputed on identical known-label candidate sets for each decision task. New-market-entry and existing-underexploited populations are kept separate. Lower USD MAE is better; higher task-specific NDCG is better. The across-model Spearman coefficient is descriptive over the seven comparable prediction strategies, not an inferential result.", ""]
    for task in ["Task_A_new_entry", "Task_B_existing_underexploited"]:
        part = matched.loc[matched.task.eq(task)].sort_values("forecast_mae_usd") if "task" in matched else pd.DataFrame()
        test_part = matched_tests.loc[matched_tests.task.eq(task)] if "task" in matched_tests else pd.DataFrame()
        matched_text += [f"## {task}", "", _markdown(part), "", "### Paired destination-block MAE comparisons", "",
                         _markdown(test_part), ""]
    (REPORTS / "MATCHED_FORECAST_RANKING.md").write_text("\n".join(matched_text) + "\n")


def _write_split_manifest_and_report():
    rows = []
    for year in YEARS:
        rows.append({"outcome_year": year, "outer_training_outcome_max": year - 2,
                     "test_feature_year": year - 2, "inner_validation_year": year - 2,
                     "inner_training_outcome_max": year - 4,
                     "information_lag_years": 2,
                     "interpretation": "retrospective lag-based backtest; revised WDI/WGI vintages"})
    pd.DataFrame(rows).to_csv(RESULTS / "split_manifest.csv", index=False)
    feat = []
    from .data import feature_columns
    from temporal_validation import FEATURE_GROUPS
    cols, cats = feature_columns("F4_full")
    geography_static = set(FEATURE_GROUPS.get("geography", []))
    for c in cols:
        invariant = c in geography_static
        feat.append({"feature": c, "type": "time_invariant_geographic" if invariant else "time_varying_numeric",
                     "timing_rule": "pre-existing geographic control" if invariant else "outcome_year - 2 or earlier"})
    for c in cats:
        feat.append({"feature": c, "type": "categorical identity", "timing_rule": "training-fitted category map; unknown category supported"})
    pd.DataFrame(feat).to_csv(RESULTS / "feature_timing_audit.csv", index=False)
    pop = _read("evaluation_population.csv")
    one_year = _read("one_year_population_audit.csv")
    text = ["# Temporal validation", "", "**Primary specification: two-year lag, retrospective rolling-origin.** For outcome year y, test covariates use the panel's y−2 information fields and training outcomes stop at y−2. Inner chronological validation uses y−2 as the validation outcome year and inner training outcomes through y−4. Encoders, missing-value handling, scaling, and category vocabularies are fitted from training rows only. Hyperparameters, epochs, class weighting, and the regression calibration factor are selected only on the inner validation fold.", "", "## Frozen outer populations", "", _markdown(pop), "", "## One-year population audit", "", _markdown(one_year), "", "One-year-lag performance was not run: the ex-ante populations differ, so two-year predictions cannot serve as a matched one-year sensitivity. No one-year performance claim is made.", "", "WDI/WGI values are revised historical vintages and BACI reference-year observations are not proven to have been available at prediction time. This is not a deployment-grade point-in-time backtest. Only three outer years are available."]
    (REPORTS / "TEMPORAL_VALIDATION.md").write_text("\n".join(text) + "\n")


def build_reports_and_workbook(test_summary: str = "" ) -> str:
    RESULTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    cost = build_compute_cost()
    seed = build_seed_stability()
    stats = combine_statistical_comparisons()
    _write_e2_report(); _write_e1_report(); _write_e3_report(); _write_split_manifest_and_report()

    e1, e2, e3 = _read("E1_neural_forecasting.csv"), _read("E2_neural_discovery.csv"), _read("E3_neural_ranking.csv")
    ablation = _read("feature_ablation_summary.csv")
    hetero = _read("heterogeneity.csv")
    missing = _read("missing_label_sensitivity.csv")
    baseline_reconciliation = _read("baseline_reconciliation.csv")
    e2tests, e3tests, e1tests = _read("statistical_comparisons_E2.csv"), _read("statistical_comparisons_E3.csv"), _read("statistical_comparisons_E1.csv")
    desttests = _read("statistical_comparisons_E3_destination_bootstrap.csv")
    matched = _read("E3_matched_forecast_ranking.csv")
    matched_tests = _read("statistical_comparisons_E3_matched_mae.csv")
    e1best = e1.sort_values("mae_usd").iloc[0] if not e1.empty else None
    e2best = e2.sort_values("pr_auc", ascending=False).iloc[0] if not e2.empty else None
    e1baseline = e1.loc[~e1.model.isin(["MLP", "TabM"])].sort_values("mae_usd").iloc[0] if not e1.empty else None

    e3_summaries = {}
    if not e3.empty:
        for task, metric in [("Task_A_new_entry", "ndcg10_entry_binary"), ("Task_B_existing_underexploited", "ndcg10_capped_grade")]:
            p = e3.loc[e3.task.eq(task)].sort_values(metric, ascending=False)
            e3_summaries[task] = p.iloc[0] if len(p) else None

    e3_baselines = {}
    if not e3.empty:
        non_neural = {"MLP", "TabM", "MLP_entry_probability", "TabM_entry_probability"}
        for task, metric in [("Task_A_new_entry", "ndcg10_entry_binary"), ("Task_B_existing_underexploited", "ndcg10_capped_grade")]:
            p = e3.loc[e3.task.eq(task) & ~e3.ranking_model.isin(non_neural)].sort_values(metric, ascending=False)
            e3_baselines[task] = p.iloc[0] if len(p) else None

    entries = _read("evaluation_population.csv")
    total_candidates = int(entries.ex_ante_candidates.sum()) if "ex_ante_candidates" in entries else 0
    total_observed = int(entries.observed_labels.sum()) if "observed_labels" in entries else 0
    total_missing = int(entries.missing_labels.sum()) if "missing_labels" in entries else 0
    status = "COMPLETE_REDUCED_SPECIFICATION"
    missing_primary = []
    if e1.empty or e2.empty or e3.empty:
        status = "PARTIAL_RESULTS"
        missing_primary.append("one or more primary experiments lacks an output table")
    e2_runs = list((OUT / "predictions/_runs").glob("classification_MLP_*_F4_full_seed*.parquet")) + list((OUT / "predictions/_runs").glob("classification_TabM_*_F4_full_seed*.parquet"))
    e1_runs = list((OUT / "predictions/_runs").glob("regression_MLP_*_F4_full_seed*.parquet")) + list((OUT / "predictions/_runs").glob("regression_TabM_*_F4_full_seed*.parquet"))
    if len(e2_runs) != 30 or len(e1_runs) != 30:
        status = "PARTIAL_RESULTS"
        missing_primary.append(f"expected 30 E2 and 30 E1 neural outer fits; found {len(e2_runs)} and {len(e1_runs)}")
    if not test_summary or "passed" not in test_summary.lower():
        status = "PARTIAL_RESULTS"
        missing_primary.append("test-suite completion was not verified")
    required_artifacts = ["predictions/E2_neural_predictions.parquet", "predictions/E1_neural_predictions.parquet",
                          "predictions/E3_neural_rankings.parquet", "results/statistical_comparisons_E1.csv",
                          "results/statistical_comparisons_E2.csv", "results/statistical_comparisons_E3.csv",
                          "results/E3_matched_forecast_ranking.csv", "results/statistical_comparisons_E3_matched_mae.csv",
                          "results/missing_label_sensitivity.csv", "results/feature_ablation.csv",
                          "results/heterogeneity.csv", "results/heterogeneity_comparisons.csv",
                          "results/statistical_comparisons_E3_destination_bootstrap.csv",
                          "results/seed_stability.csv", "results/compute_cost.csv",
                          "results/baseline_reconciliation.csv"]
    absent = [x for x in required_artifacts if not (OUT / x).exists()]
    if absent:
        status = "PARTIAL_RESULTS"
        missing_primary.append("required artifacts missing: " + ", ".join(absent))
    if not baseline_reconciliation.empty and not baseline_reconciliation.verification.eq("PASS").all():
        status = "PARTIAL_RESULTS"
        missing_primary.append("saved E1 baseline prediction reconciliation failed")

    def line(metric_row, col):
        return _fmt(metric_row[col]) if metric_row is not None and col in metric_row.index else "NA"
    primary_conclusion = ""
    if e1best is not None and e2best is not None:
        primary_conclusion = (f"USD-MAE winner: {e1best.model} ({e1best.mae_usd:,.0f} USD). Discovery PR-AUC winner: {e2best.model} ({e2best.pr_auc:.4f}).")
    e3summary_text = ""
    for task, r in e3_summaries.items():
        if r is not None:
            metric = "ndcg10_entry_binary" if task == "Task_A_new_entry" else "ndcg10_capped_grade"
            e3summary_text += f"{task}: {r.ranking_model} {metric}={r[metric]:.4f}. "

    def inference_sentence(frame: pd.DataFrame, label: str) -> str:
        if frame.empty or not {"model_a", "model_b", "p_holm", "difference_a_minus_b"}.issubset(frame.columns):
            return f"{label}: no valid paired inference table."
        parts = []
        for r in frame.itertuples(index=False):
            p = getattr(r, "p_holm", np.nan)
            diff = getattr(r, "difference_a_minus_b", np.nan)
            if pd.isna(p):
                continue
            if p < 0.05:
                verdict = f"statistically distinguishable after Holm; {'favors ' + r.model_a if diff > 0 else 'favors ' + r.model_b}"
            else:
                verdict = "not statistically distinguishable after Holm"
            parts.append(f"{r.model_a} vs {r.model_b}: Δ={_fmt(diff)}, pHolm={p:.3f}, {verdict}")
        return label + ": " + ("; ".join(parts) if parts else "no finite adjusted comparisons.")

    e2_primary_tests = e2tests.loc[e2tests.metric.eq("pr_auc")] if "metric" in e2tests else pd.DataFrame()
    e3_task_a_tests = e3tests.loc[e3tests.experiment.eq("E3_Task_A_new_entry")] if "experiment" in e3tests else pd.DataFrame()
    e3_task_b_tests = e3tests.loc[e3tests.experiment.eq("E3_Task_B_existing_underexploited")] if "experiment" in e3tests else pd.DataFrame()
    matched_mae_tests = _read("statistical_comparisons_E3_matched_mae.csv")

    stat_report = ["# Statistical inference", "", "All reported comparisons are paired on the same observed-label units or candidate pools. Model seeds are averaged within an economic unit and are not treated as independent observations.", "",
                   "## E2 discovery", "", "Primary metrics are compared within metric-specific five-contrast families. Whole HS2 chapters are resampled with all years retained; a paired chapter sign-flip test supplies raw p-values, followed by Holm adjustment. There are only three outer years, so these intervals do not estimate temporal generalization.", "", _markdown(e2tests), "",
                   "## E3 ranking", "", "Task A uses binary entry NDCG@10; Task B uses capped-gain NDCG@10. Each task has a five-contrast Holm family. Pairs share the same product-year candidate pools. The primary HS2-block bootstrap resamples chapters and retains the year pools; a destination-block bootstrap sensitivity resamples destinations jointly across all product-year pools and recomputes rankings. Both are cross-sectional procedures and neither resolves uncertainty across three years.", "", _markdown(e3tests), "", "### Destination-block ranking sensitivity", "", _markdown(desttests), "",
                   "## Matched E3 task-specific forecast errors", "", "For each ranking task, USD MAE is recomputed on exactly the observed-label candidates used in that task's ranking population. These are secondary paired error contrasts, not the overall E1 primary metric.", "", _markdown(matched_tests), "",
                   "## E1 forecasting", "", "The primary scale is USD MAE. Paired absolute errors are averaged over neural seeds within the destination–HS2–year key, then resampled by destination, preserving each selected destination's product-year records. Wilcoxon tests use destination-level differences and Holm correction is applied across the four prespecified neural-versus-Persistence/PPML contrasts.", "", _markdown(e1tests), "",
                   "## Dependence and interpretation", "", "HS2 bootstrap treats HS2 chapters as independent blocks, while destination bootstrap treats destinations as blocks. Both leave residual dependence across years and shared trade shocks; the three evaluation years are too few for reliable year-cluster inference. Destination-level Wilcoxon p-values that numerically underflow to zero are conservatively floored at 1e-16 before Holm adjustment; p-values below 0.0001 are displayed as <0.0001. P-values are finite-sample resampling/sign-rank evidence under these assumptions, not proof of economic importance or out-of-time stability."]
    (REPORTS / "STATISTICAL_INFERENCE.md").write_text("\n".join(stat_report) + "\n")

    findings = ["# Research findings", "", f"**Status: {status}.**", "", "## H1 — Neural market discovery", "",
                f"Observed macro PR-AUC winner: {e2best.model} at {e2best.pr_auc:.4f}; MLP and TabM are reported beside LightGBM and PPML in `results/E2_neural_discovery.csv`. Paired metric-specific inference is in `results/statistical_comparisons_E2.csv`; interpret adjusted intervals and p-values, not winner labels alone.", "", _markdown(e2.sort_values("pr_auc", ascending=False), [c for c in ["model", "pr_auc", "precision_at_10", "ndcg_at_10", "valid_pools", "valid_pr_auc_pools", "valid_ndcg_at_10_pools"] if c in e2]), "",
                "## H2 — Neural market ranking", "", e3summary_text,
                "Task A (new entry) uses binary entry NDCG; Task B (existing underexploited markets) uses capped-grade incremental-export NDCG. Log1p/raw-gain metrics and destination-block sensitivity are also reported. Pool-specific sample sizes and no-relevance NA rules are explicit.", "",
                _markdown(e3, [c for c in ["ranking_model", "task", "ndcg10_entry_binary", "ndcg10_capped_grade", "ndcg10_log1p_gain", "observed_incremental_usd_top10_full_rank_partial", "valid_product_year_pools", "valid_ndcg10_entry_binary_pools", "valid_ndcg10_capped_grade_pools"] if c in e3]), "",
                "## H3 — Export value forecasting", "", primary_conclusion, "", _markdown(e1.sort_values("mae_usd")), "",
                "USD MAE is the primary criterion. Log1p MAE is a separate scale and does not supersede USD performance. See matched observed-label comparisons and paired destination-block inference.", "",
                "## H4 — Task-specific neural value", "", "E2, E3 Task A, E3 Task B, and E1 are separate estimands. Differences between a forecasting winner and a ranking winner alone do not prove a lack of association; matched-population MAE/NDCG values and paired task-specific MAE intervals are in `results/E3_matched_forecast_ranking.csv` and `results/statistical_comparisons_E3_matched_mae.csv`.", "", _markdown(matched), "",
                "## H5 — Information value and heterogeneity", "", f"Feature ablation rows: {len(ablation):,}; heterogeneity rows: {len(hetero):,}. The feature ablation is an exploratory 2024 single-seed analysis with F4-selected hyperparameters held fixed; it does not establish causal effects or optimal performance for every feature set. Heterogeneity estimates are descriptive and small-cell uncertainty is limited.", "",
                "## Paired comparative conclusions", "", inference_sentence(e2_primary_tests, "E2 PR-AUC"), "",
                inference_sentence(e3_task_a_tests, "E3 Task A binary-entry NDCG@10"), "",
                inference_sentence(e3_task_b_tests, "E3 Task B capped-gain NDCG@10"), "",
                inference_sentence(e1tests, "E1 all-candidate USD MAE"), "",
                inference_sentence(matched_mae_tests, "E3 task-matched USD MAE"), "",
                "## Inference and limits", "", f"The frozen ex-ante universe has {total_candidates:,} rows, {total_observed:,} known outcomes and {total_missing:,} unknown labels. Complete-case performance is conditional on label availability. Current WDI/WGI vintages are revised, the one-year-lag model sensitivity was not run because the population changes, and three evaluation years do not support precise temporal generalization.", "",
                "## Feature information and market groups", "", _markdown(ablation, [c for c in ["task", "model", "feature_set", "metric", "mean_across_hs2", "valid_hs2_chapters"] if c in ablation]), "",
                "Prespecified subgroup estimates and paired neural-minus-baseline differences are in `results/heterogeneity.csv` and `results/heterogeneity_comparisons.csv`. These are descriptive, with no causal interpretation.", "",
                "No observational ranking result is a causal export gain or firm-level recommendation. Statistical and economic importance should be judged separately; see compute costs and missing-label bounds."]
    (REPORTS / "RESEARCH_FINDINGS.md").write_text("\n".join(findings) + "\n")

    integrity = ["# Research integrity", "", f"**Final status: {status}.**", "", "## Data and timing", "",
                 "The extension reads the processed HS2 two-year-lag panel and the saved audited baseline prediction files; no raw data were modified or downloaded. Candidate eligibility is based on lagged positive destination import demand and verified historical inactivity. Outcome-year absent BACI records are not recoded to zero; the 559 unknown labels remain NA.", "",
                 "Annual WDI and WGI historical values come from revised current vintages; BACI reference-year release availability is not verified. The exercise is retrospective lag-based, not a real-time forecast evaluation. The one-year lag panel yields a different ex-ante universe and its neural performance sensitivity is not run.", "",
                 "Saved baseline E1 predictions were reconciled against `audit_v2/corrected_exante_forecast_predictions.parquet` over all 62,238 known outcome keys per model. Persistence, PPML, XGBoost, and LightGBM matched exactly; Random Forest's maximum seed-mean difference was 0.00005 USD (floating-point serialization scale). See `results/baseline_reconciliation.csv`.", "",
                 "## Models and evaluation", "", "MLP uses training-fitted median imputation/missing indicators, training-only scaling, and train-fitted one-hot destination/HS2 identity. TabM uses the official `tabm` distribution, k=8, train-fitted categorical cardinalities with unknown index 0; classification aggregates member probabilities after sigmoid, regression averages member log-scale outputs before inversion. Both train on identical declared features and chronological folds.", "",
                 "Hyperparameters and epochs are selected on inner chronological validation only. Neural regression is trained on log1p USD and uses a validation-selected multiplicative calibration for the USD target. Predictions are saved on log and USD scales; the calibration is not claimed to recover an unbiased conditional mean.", "",
                 "## Statistical assumptions", "", "Seeds describe algorithmic variability and are averaged within economic evaluation units. E2 and E3 macro comparisons use HS2-block bootstrap/sign-flip procedures; E1 uses destination-block bootstrap and destination-level sign-rank tests. E3 also reports a destination-block bootstrap that re-ranks weighted candidates jointly across pools. These procedures address cross-sectional dependence imperfectly and do not provide precise uncertainty across only three years.", "",
                 "## Scope and interpretation", "", "E3 Task A is binary entry; Task B is existing-market expansion. Two earlier E3 implementations were invalidated and are preserved outside the final-result tables; only the corrected per-HS2-year pools are reported. Missing labels make complete-case metrics conditional and bounds may be wide. Feature ablation uses one year and one seed with full-feature-selected hyperparameters; it is exploratory. No analysis is causal. Market rankings are country–product historical priorities, not firm-level recommendations, because firm capacity, costs, tariffs, and product quality are not modeled.", "",
                 "## Completed and incomplete work", "", "Primary two-year-lag E2, E3, and E1 run completion is determined by saved predictions and the final run manifest. The one-year-lag performance sensitivity is not completed. Any failed seed, missing model, or test status is listed in `FINAL_STATUS.md`; no unfinished analysis is labeled passed."]
    (REPORTS / "RESEARCH_INTEGRITY.md").write_text("\n".join(integrity) + "\n")

    paper = ["# Paper-ready empirical results (post-audit neural extension)", "", "## Data", "",
             f"The analysis uses Vietnam bilateral HS12 BACI V202601 trade aggregated to HS2 and the existing audited destination–product panel. The 2022–2024 primary evaluation universe contains {total_candidates:,} ex-ante eligible destination–HS2–year observations; {total_observed:,} outcome labels are observed and {total_missing:,} are unobserved. Unknown trade outcomes are retained as missing rather than coded as zero. The available macroeconomic and governance observations are retrospective current vintages.", "",
             "## Research design and empirical strategy", "", "The primary protocol imposes a two-year information lag and chronological outer folds for 2022, 2023, and 2024. Hyperparameter selection, preprocessing, categorical encoding, imputation, scaling, and regression calibration use only earlier training/validation data. Five fixed training seeds are used for stochastic MLP/TabM fits. The MLP is a feed-forward multilayer network; TabM uses the official parameter-efficient ensemble implementation (k=8). Baselines include persistence, PPML, and saved tree models.", "",
             "The discovery target is a verified previously inactive relationship reaching USD 50,000 in the outcome year. Ranking separates new entry from existing underexploited-market expansion. Forecast accuracy uses USD MAE as the primary metric and reports log1p metrics separately. Ranking metrics are macro-averaged within product-year candidate pools; paired comparisons use the same pools.", "",
             "## Forecasting results", "", _markdown(e1.sort_values("mae_usd")) if not e1.empty else "No completed table.", "", _markdown(e1tests), "",
             "USD MAE and log1p MAE answer different loss-scale questions. A neural model is not considered superior solely because its log-scale error is smaller.", "",
             "## Market discovery", "", _markdown(e2.sort_values("pr_auc", ascending=False)) if not e2.empty else "No completed table.", "", _markdown(e2tests), "",
             "Discovery PR-AUC, precision, and binary NDCG are reported separately. Inference is paired by HS2 blocks and Holm-adjusted within metric families.", "",
             "## Market ranking", "", _markdown(e3) if not e3.empty else "No completed table.", "", _markdown(e3tests), "",
             "New-market entry and existing-market expansion use distinct eligibility and relevance definitions. Destination-block re-ranking bootstrap results are a cross-sectional dependence sensitivity. Matched task-specific forecasting MAE and ranking NDCG are reported separately:", "", _markdown(matched), "",
             "## Explainability, ablation, and segmentation", "", "This neural extension evaluates information value by feature-group ablation and prespecified market/product subgroups. The ablation is exploratory and single-year/single-seed; no causal interpretation is warranted. The extension does not re-estimate SHAP; prior audit SHAP values remain tied to their original prediction task and are not attributed to MLP or TabM.", "", _markdown(ablation, [c for c in ["task", "model", "feature_set", "metric", "mean_across_hs2", "valid_hs2_chapters"] if c in ablation]), "",
             "## Robustness, discussion, and limitations", "", "The one-year-lag population was audited and found to differ from the primary universe, but one-year neural refits are not run. Complete-case estimates are conditional on verified labels; missing-label bounds and stress cases are in `MISSING_LABEL_SENSITIVITY.md`. Only three outer years and revised WDI/WGI vintages limit temporal and real-time claims. Country-level historical rankings do not imply firm-level profitability or causal export gains.", ""]
    (REPORTS / "PAPER_READY_RESULTS.md").write_text("\n".join(paper) + "\n")

    # Workbook is tables-only: no charts or embedded figures.
    workbook = OUT / "results/NN_DL_FULL_COMPARISON.xlsx"
    audit_sheet = _read("evaluation_population.csv")
    tabs = {
        "Data Audit": audit_sheet,
        "Discovery": e2,
        "Ranking": e3,
        "Forecasting": e1,
        "Matched Forecast Ranking": matched,
        "HS2 Heterogeneity": _read("heterogeneity_comparisons.csv"),
        "Feature Ablation": ablation,
        "Statistical Tests": stats,
        "Seed Stability": _read("seed_stability_summary.csv"),
        "Compute Cost": cost,
        "Missing Label Sensitivity": missing,
    }
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for sheet, frame in tabs.items():
            (frame if not frame.empty else pd.DataFrame({"status": ["No computed rows"]})).to_excel(
                writer, sheet_name=sheet, index=False, freeze_panes=(1, 0))

    def test_summary_text(frame, conditions):
        if frame.empty:
            return "Chưa có kiểm định hợp lệ."
        subset = frame.copy()
        for column, value in conditions.items():
            if column in subset:
                subset = subset.loc[subset[column].eq(value)]
        if subset.empty:
            return "Không đủ cặp mẫu/metric để kiểm định."
        items = []
        for r in subset.itertuples(index=False):
            p = getattr(r, "p_holm", np.nan)
            diff = getattr(r, "difference_a_minus_b", np.nan)
            ci_low, ci_high = getattr(r, "ci95_low", np.nan), getattr(r, "ci95_high", np.nan)
            if pd.notna(p):
                sig = "có khác biệt sau Holm" if p < 0.05 else "chưa phân biệt được sau Holm"
                p_text = "<0.001" if p < 0.001 else f"{p:.3f}"
                items.append(f"{r.model_a}−{r.model_b}: Δ={_fmt(diff)}, CI95% [{_fmt(ci_low)}, {_fmt(ci_high)}], pHolm={p_text} ({sig})")
        return "; ".join(items) if items else "Không có p-value xác định."

    summary_vi = ["# Tóm tắt kết quả mở rộng MLP + TabM", "", f"**Trạng thái: {status}.**", "",
                  "| Nhiệm vụ | Baseline tốt nhất | MLP | TabM | Kết quả quan sát | Kết luận thống kê |", "|---|---|---:|---:|---|---|"]
    if e2best is not None:
        mlp=e2.loc[e2.model.eq("MLP")].iloc[0]; tabm=e2.loc[e2.model.eq("TabM")].iloc[0]
        light=e2.loc[e2.model.eq("LightGBM")]
        stat_text=test_summary_text(e2tests, {"metric": "pr_auc"})
        summary_vi.append(f"| Phát hiện thị trường | LightGBM PR-AUC {_fmt(light.iloc[0].pr_auc)} | {_fmt(mlp.pr_auc)} | {_fmt(tabm.pr_auc)} | {e2best.model} đứng đầu PR-AUC | {stat_text} |")
    if e3_summaries:
        for task, metric, label in [("Task_A_new_entry", "ndcg10_entry_binary", "Xếp hạng — thị trường mới"),
                                    ("Task_B_existing_underexploited", "ndcg10_capped_grade", "Xếp hạng — mở rộng hiện hữu")]:
            part=e3.loc[e3.task.eq(task)]
            if part.empty: continue
            vals={m: part.loc[part.ranking_model.eq(m),metric] for m in ["MLP","TabM","Gravity_PPML","LightGBM"]}
            b=e3_baselines.get(task)
            best_base=f"{b.ranking_model} {_fmt(b[metric])}" if b is not None else "NA"
            stat_text=test_summary_text(e3tests.loc[e3tests.experiment.eq(f"E3_{task}")], {}) if not e3tests.empty else "NA"
            summary_vi.append(f"| {label} | {best_base} | {_fmt(vals['MLP'].iloc[0]) if len(vals['MLP']) else 'NA'} | {_fmt(vals['TabM'].iloc[0]) if len(vals['TabM']) else 'NA'} | {part.sort_values(metric,ascending=False).iloc[0].ranking_model} đứng đầu | {stat_text} |")
    if e1best is not None:
        m=e1.loc[e1.model.eq("MLP")]; t=e1.loc[e1.model.eq("TabM")]
        best_base=f"{e1baseline.model} {_fmt(e1baseline.mae_usd)}" if e1baseline is not None else "NA"
        stat_text=test_summary_text(e1tests, {})
        summary_vi.append(f"| Dự báo giá trị xuất khẩu (MAE USD) | {best_base} | {_fmt(m.iloc[0].mae_usd) if len(m) else 'NA'} | {_fmt(t.iloc[0].mae_usd) if len(t) else 'NA'} | {e1best.model} đứng đầu MAE | {stat_text} |")
    summary_vi += ["", f"Mẫu ex-ante gồm {total_candidates:,} quan sát; nhãn đã biết {total_observed:,}; thiếu {total_missing:,}. Kết quả chính là complete-case và chưa có độ nhạy neural một năm do tập ứng viên thay đổi.", "",
                   "Năm đánh giá chỉ có ba; WDI/WGI là vintage lịch sử đã được hiệu chỉnh. Không diễn giải xếp hạng quan sát như lợi ích xuất khẩu nhân quả hay khuyến nghị ở cấp doanh nghiệp.", "",
                   "Tệp chi tiết: `E1_FORECASTING_RESULTS.md`, `E2_DISCOVERY_RESULTS.md`, `E3_RANKING_RESULTS.md`, `STATISTICAL_INFERENCE.md`, `MISSING_LABEL_SENSITIVITY.md`."]
    (REPORTS / "EXECUTIVE_SUMMARY_VI.md").write_text("\n".join(summary_vi) + "\n")

    status_lines = [f"# Final status: {status}", "", f"Primary population: {total_candidates:,} ex-ante candidates; {total_observed:,} observed labels; {total_missing:,} missing labels.",
                    f"Neural final outer fit files: E2={len(e2_runs)}/30; E1={len(e1_runs)}/30.",
                    "Primary E2/E3/E1 metrics are complete-case and use the same observed-label candidate intersections. See saved prediction files and run logs.",
                    "Feature ablation is exploratory (2024, seed 11, F4-selected settings held fixed). The one-year-lag population is audited but its model performance is not run.",
                    "Test verification: " + (test_summary.strip() or "not supplied to finalizer"), "",
                    "Outstanding limitations: revised macro-data vintages; unavailable-versus-true-zero ambiguity for missing BACI outcomes; three outer test years; one-year sensitivity not executed; feature ablation limited to one year/seed."]
    if missing_primary:
        status_lines += ["", "Incomplete primary items:", ""] + [f"- {x}" for x in missing_primary]
    (REPORTS / "FINAL_STATUS.md").write_text("\n".join(status_lines) + "\n")
    return status


def main():
    test_file = OUT / "logs/test_summary.txt"
    test_summary = test_file.read_text() if test_file.exists() else ""
    status = build_reports_and_workbook(test_summary)
    print(f"Finalized reports and workbook; status={status}")


if __name__ == "__main__":
    main()
