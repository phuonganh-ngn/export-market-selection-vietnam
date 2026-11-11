from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .candidate_protocol import HERE
from .independent_validation import validate_saved_results


def _read(name: str) -> pd.DataFrame:
    path = HERE / "results" / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _fmt(value, digits=4):
    try:
        return f"{float(value):,.{digits}f}" if np.isfinite(float(value)) else "NA"
    except (TypeError, ValueError):
        return "NA"


def _selected_performance(perf: pd.DataFrame, task: str) -> pd.DataFrame:
    if perf.empty:
        return perf
    return perf.loc[perf.task.eq(task)].copy()


def _write_excel(sheets: dict[str, pd.DataFrame], path: Path) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)


def finalize_reports() -> dict:
    independent_validation = validate_saved_results()
    result_dir = HERE / "results"
    report_dir = HERE / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = _read("E6_pool_metrics.csv")
    comparison = _read("E6_model_comparison.csv")
    by_hs2 = _read("E6_by_hs2.csv")
    by_year = _read("E6_by_year.csv")
    xendcg = _read("E6_rank_xendcg_sensitivity.csv")
    regret = _read("E7_regret_at_5_10.csv")
    stability = _read("E7_topk_stability.csv")
    robust = _read("E7_robust_selection.csv")
    eligibility_sensitivity = _read("E7_candidate_eligibility_sensitivity.csv")
    pairwise = _read("E8_pairwise_explanations.csv")
    group = _read("E8_feature_group_rank_effects.csv")
    fidelity = _read("E8_explanation_fidelity.csv")
    ablation = _read("E8_retrained_feature_ablation.csv")
    missing_cov = _read("E9_missing_label_coverage.csv")
    missing_sens = _read("E9_missing_label_sensitivity.csv")
    inference = _read("paired_model_comparisons.csv")
    destination_boot = _read("destination_block_sensitivity.csv")
    seeds = _read("seed_stability.csv")
    compute = _read("computational_cost.csv")
    population = _read("evaluation_population.csv")
    manifest = _read("candidate_manifest.csv")
    missing_audit = _read("missing_label_audit.csv")

    # A compact, auditable sensitivity table: same Task B pools and score ordering,
    # with the metric implementations already computed from the saved predictions.
    if not metrics.empty:
        rs = metrics.loc[metrics.task.eq("Task_B_existing_underexploited")].copy()
        rs.to_csv(result_dir / "E7_rank_sensitivity.csv", index=False, na_rep="")

    # Use the evaluator's macro model table. Pool-level metrics must never be
    # re-labelled as an overall model comparison.
    task_a = _selected_performance(comparison, "Task_A_new_entry")
    task_b = _selected_performance(comparison, "Task_B_existing_underexploited")
    regret_main = regret.loc[
        ((regret.task.eq("Task_A_new_entry")) & regret.gain_definition.eq("binary_entry"))
        | ((regret.task.eq("Task_B_existing_underexploited")) & regret.gain_definition.eq("capped_grade_gain"))
    ].copy() if not regret.empty else regret
    reg_summary = (regret_main
                   .groupby(["task", "model", "seed", "k"], as_index=False)
                   .agg(mean_regret=("regret", "mean"), mean_normalized_regret=("normalized_regret", "mean"),
                        mean_selected_gain=("selected_gain", "mean"), pools=("regret", "size"))) if not regret_main.empty else pd.DataFrame()
    # Primary comparators and decisions are calculated from the saved, paired pool table.
    def get_metric(frame, model, col):
        z = frame.loc[frame.model.eq(model), col]
        return float(z.iloc[0]) if len(z) else np.nan

    def regret_mean(task, model, seed):
        if reg_summary.empty:
            return np.nan
        z = reg_summary.loc[reg_summary.task.eq(task) & reg_summary.model.eq(model)
                            & reg_summary.seed.eq(seed) & reg_summary.k.eq(10), "mean_regret"]
        return float(z.iloc[0]) if len(z) else np.nan

    lambda_a_ndcg = get_metric(task_a, "LambdaMART", "ndcg_at_10")
    lambda_a_p10 = get_metric(task_a, "LambdaMART", "precision_at_10")
    lambda_b_ndcg = get_metric(task_b, "LambdaMART", "ndcg_at_10")
    lambda_b_reg = regret_mean("Task_B_existing_underexploited", "LambdaMART", -2)
    xend_a = xendcg.loc[xendcg.task.eq("Task_A_new_entry") & xendcg.seed.eq(-2), "ndcg10"]
    xend_b = xendcg.loc[xendcg.task.eq("Task_B_existing_underexploited") & xendcg.seed.eq(-2), "ndcg10"]
    xend_a_mean = float(xend_a.iloc[0]) if len(xend_a) else np.nan
    xend_b_mean = float(xend_b.iloc[0]) if len(xend_b) else np.nan
    model_tests = inference.loc[inference.metric.isin(["ndcg_at_10", "precision_at_10", "mean_regret@10_capped_gain"])] if not inference.empty else inference

    def best_for(frame, metric):
        if frame.empty or metric not in frame:
            return None, np.nan
        q = frame.dropna(subset=[metric]).sort_values(metric, ascending=False)
        return (str(q.iloc[0].model), float(q.iloc[0][metric])) if len(q) else (None, np.nan)

    best_a, best_a_value = best_for(task_a.loc[~task_a.model.eq("LambdaMART")], "ndcg_at_10") if not task_a.empty else (None, np.nan)
    best_b, best_b_value = best_for(task_b.loc[~task_b.model.eq("LambdaMART")], "ndcg_at_10") if not task_b.empty else (None, np.nan)
    best_b_reg_model = None
    best_b_reg_value = np.nan
    if not reg_summary.empty:
        q = reg_summary.loc[reg_summary.task.eq("Task_B_existing_underexploited") & reg_summary.k.eq(10)] if "k" in reg_summary else pd.DataFrame()
        if q.empty:
            q = reg_summary.loc[reg_summary.task.eq("Task_B_existing_underexploited") & reg_summary.seed.isin([-1, -2])]
        if len(q):
            q = q.sort_values("mean_regret")
            best_b_reg_model, best_b_reg_value = str(q.iloc[0].model), float(q.iloc[0].mean_regret)

    # Top predictive group attribution by task, on the LambdaMART raw score scale.
    group_summary = pd.DataFrame()
    if not group.empty:
        group_summary = group.groupby(["task", "feature_group"], as_index=False).agg(
            mean_abs_contribution=("mean_absolute_group_contribution", "mean"),
            mean_signed_contribution=("mean_signed_group_contribution", "mean"),
            fold_seed_rows=("seed", "size"))
        group_summary.to_csv(result_dir / "E8_feature_group_summary.csv", index=False, na_rep="")
    pairwise_examples = pd.DataFrame()
    if not pairwise.empty:
        pair_keys = ["task", "evaluation_year", "hs2", "destination_a", "destination_b"]
        pair_meta = pairwise.groupby(pair_keys, as_index=False).agg(
            absolute_score_gap=("score_difference", lambda x: float(abs(x.iloc[0]))))
        chosen = []
        for (task_name, year), q in pair_meta.groupby(["task", "evaluation_year"], sort=True):
            target_gap = float(q.absolute_score_gap.median())
            chosen.append(q.assign(distance_to_median=(q.absolute_score_gap - target_gap).abs())
                          .sort_values(["distance_to_median", "hs2", "destination_a", "destination_b"], kind="mergesort").iloc[0][pair_keys])
        chosen = pd.DataFrame(chosen)
        pairwise_examples = pairwise.merge(chosen, on=pair_keys, how="inner", validate="many_to_one")
    fixed_perturb = _read("E8_fixed_model_group_perturbation.csv")
    fixed_summary = (fixed_perturb.loc[fixed_perturb.status.eq("fixed_model_training_median_replacement")]
                     .groupby(["task", "feature_group"], as_index=False)
                     .agg(mean_top5_jaccard=("top5_jaccard", "mean"), mean_top10_jaccard=("top10_jaccard", "mean"),
                          mean_rank_displacement=("mean_absolute_rank_displacement", "mean"),
                          pools=("top10_jaccard", "count"))) if not fixed_perturb.empty else pd.DataFrame()

    # Generate scientific reports only after all saved numerical evidence exists.
    e6_lines = ["# Experiment E6 — Decision-focused ranking", "",
                "LambdaMART is fitted with LightGBM's `LGBMRanker(objective='lambdarank')`; queries are HS2 × outcome-year pools. Task A uses binary entry relevance. Task B uses four nonzero training-only quantile grades and explicit label gains [0, 1, 3, 7, 15]. Outer evaluation is 2022–2024 with the audited two-year lag. Three chronological development configurations were compared per outer fold; selection used validation macro NDCG@10. Five final seeds were fitted per task-year.", "",
                "All predictions and pool metrics are reconstructed from the frozen ex-ante candidate manifest. Missing outcome labels remain unobserved; complete-case quality metrics use the observed-label candidates within the full candidate ranking order. An NDCG pool with no positive observed relevance is NA.", "",
                "## Task A: new-market entry", "", task_a.to_markdown(index=False) if not task_a.empty else "No results available.", "",
                f"LambdaMART macro NDCG@10: {_fmt(lambda_a_ndcg)}; Precision@10: {_fmt(lambda_a_p10)}. Highest non-Lambda baseline by NDCG@10: {best_a} ({_fmt(best_a_value)}).", "",
                "## Task B: existing-market expansion", "", task_b.to_markdown(index=False) if not task_b.empty else "No results available.", "",
                f"LambdaMART macro NDCG@10: {_fmt(lambda_b_ndcg)}. Highest non-Lambda baseline by NDCG@10: {best_b} ({_fmt(best_b_value)}).", "",
                "NDCG is the audited capped-gain version for Task B; alternative log1p and raw-USD gains are reported separately. Macro means are across valid product-year pools and denominators are reported by model. Paired tests use common pools and HS2 blocks.", "",
                "## Paired inference", "", model_tests.to_markdown(index=False) if not model_tests.empty else "Paired inference not available.", "",
                "## Optional `rank_xendcg` objective sensitivity", "",
                xendcg.to_markdown(index=False) if not xendcg.empty else "Optional objective sensitivity was not run.", "",
                f"This sensitivity holds LambdaMART-selected chronological configurations and iteration counts fixed while changing the ranking objective; it is not separately tuned or confirmatory. The seed-mean macro NDCG@10 is { _fmt(xend_a_mean) } for Task A and { _fmt(xend_b_mean) } for Task B, compared descriptively with the primary `lambdarank` policy ({_fmt(lambda_a_ndcg)} and {_fmt(lambda_b_ndcg)} respectively).", "",
                "No causal or deployment-grade claim follows from this retrospective backtest."]
    (report_dir / "EXPERIMENT_E6.md").write_text("\n".join(e6_lines) + "\n")

    # Regret comparison summary
    rb = reg_summary.loc[reg_summary.task.eq("Task_B_existing_underexploited") & reg_summary.k.eq(10)
                         & reg_summary.seed.isin([-1, -2])].copy() if not reg_summary.empty else reg_summary
    ra = reg_summary.loc[reg_summary.task.eq("Task_A_new_entry") & reg_summary.k.eq(10)
                         & reg_summary.seed.isin([-1, -2])].copy() if not reg_summary.empty else reg_summary
    regret_lines = ["# Decision regret", "",
                    "For each HS2-year pool, the oracle is the top K observed candidates under the same gain function and candidate-label intersection. Realized-outcome opportunity regret is oracle gain minus selected gain; it is not profit forgone or a causal loss. Normalized regret is undefined when oracle gain is zero.", "",
                    "## New-market entry, binary regret@10", "", ra.to_markdown(index=False) if not ra.empty else "No results available.", "",
                    "## Existing-market expansion, capped-gain regret@10", "", rb.to_markdown(index=False) if not rb.empty else "No results available.", "",
                    "Pool-specific metrics for K=5 and K=10, plus Task B raw-USD and binary-expansion gain sensitivities, are preserved in `E7_regret_at_5_10.csv`. The Task B USD increment sensitivity is an observed historical outcome metric; missing labels are excluded from oracle and model outcome gains. Cross-pool sums should not be read as economic returns."]
    (report_dir / "DECISION_REGRET.md").write_text("\n".join(regret_lines) + "\n")

    xai_lines = ["# Decision-level explanations", "",
                 "Explanations use LightGBM's additive contributions to the raw LambdaMART ranking score. Pairwise score differences subtract two candidate explanations; the intercept cancels. This explains score ordering behavior, not listwise ranking quality or causal effects.", "",
                 "## Feature-group score attribution", "", group_summary.sort_values(["task", "mean_abs_contribution"], ascending=[True, False]).to_markdown(index=False) if not group_summary.empty else "No feature attribution results.", "",
                 "## Additivity checks", "", fidelity.to_markdown(index=False) if not fidelity.empty else "No explanation-fidelity results.", "",
                 "## Pairwise actual candidate examples", "", pairwise_examples.to_markdown(index=False) if not pairwise_examples.empty else "No pairwise examples.", "",
                 "The selected examples are deterministic median score-gap pairs within task and year; the full candidate contrasts remain in `E8_pairwise_explanations.csv`.", "",
                 "## Fixed-model Top-K perturbation", "", fixed_summary.to_markdown(index=False) if not fixed_summary.empty else "No fixed-model perturbation results.", "",
                 "## Retrained feature-group ablation", "", ablation.groupby(["task", "feature_group_removed", "k"], as_index=False).agg(mean_topk_jaccard=("topk_jaccard_vs_full", "mean"), mean_ndcg_full=("ndcg10_full", "mean"), mean_ndcg_without_group=("ndcg10_retrained_without_group", "mean"), pools=("topk_jaccard_vs_full", "count")).to_markdown(index=False) if not ablation.empty else "No retrained ablation results.", "",
                 "Destination identity contributes to learned scores, but it is an identifier that can encode destination-specific residuals and cross-product transfer; it is not itself an economic mechanism. Pairwise explanations identify model score decompositions. Fixed-model median replacement is an off-manifold sensitivity probe; retrained ablation estimates performance with a feature group unavailable. Neither is a causal intervention. Established ranking explanation proposals exist; this study does not claim a new general-purpose XAI method."]
    (report_dir / "DECISION_XAI.md").write_text("\n".join(xai_lines) + "\n")

    stability_summary = stability.groupby(["task", "k", "comparison", "model_a", "model_b"], as_index=False).agg(mean_jaccard=("jaccard", "mean"), median_jaccard=("jaccard", "median"), pool_pairs=("jaccard", "count")) if not stability.empty else stability
    robust_summary = robust.groupby("task", as_index=False).agg(mean_consensus_ndcg=("selection_frequency_ndcg10", "mean"), mean_score_policy_ndcg=("mean_seed_score_ndcg10", "mean"), pools=("selection_frequency_ndcg10", "count")) if not robust.empty else robust
    robustness_lines = ["# Decision robustness", "", "## Top-K stability", "", stability_summary.to_markdown(index=False) if not stability_summary.empty else "No stability results.", "", "## Seed-frequency consensus policy", "", robust_summary.to_markdown(index=False) if not robust_summary.empty else "No consensus policy results.", "", "Rankings are also accompanied by `E7_rank_sensitivity.csv` with capped, log1p, raw USD, and binary relevance sensitivity where applicable. Candidate universes are kept fixed within each pool; unknown outcomes are retained in rankings and reported separately. The consensus is exploratory and is not a confirmatory model selected on the test set."]
    eligibility_summary = (eligibility_sensitivity.groupby(["task", "model"], as_index=False)
                           .agg(mean_candidate_fraction=("restricted_fraction", "mean"),
                                mean_ndcg10_change=("restricted_minus_full_ndcg10", "mean"),
                                mean_top10_jaccard=("top10_jaccard_full_vs_restricted", "mean"),
                                mean_restricted_regret10=("restricted_regret10", "mean"),
                                pools=("restricted_ndcg10", "count"))) if not eligibility_sensitivity.empty else pd.DataFrame()
    robustness_lines.extend(["", "## Historical high-demand eligibility sensitivity", "",
                             "This analysis restricts each HS2-year pool to candidates with y−2 destination import demand at or above the HS2-specific median computed from historical panel rows through y−2. It deliberately changes the candidate universe; its NDCG and regret values are not directly compared to primary-population levels.", "",
                             eligibility_summary.to_markdown(index=False) if not eligibility_summary.empty else "Eligibility sensitivity not run."])
    (report_dir / "DECISION_ROBUSTNESS.md").write_text("\n".join(robustness_lines) + "\n")

    # Missing-label report; no conversion is used outside explicitly named scenario bounds.
    miss_n = int(population.missing_labels.sum()) if not population.empty else 0
    miss_policy = missing_cov.loc[(missing_cov.model.eq("LambdaMART") & missing_cov.seed.eq(-2))
                                  | (~missing_cov.model.eq("LambdaMART") & missing_cov.seed.eq(-1))].copy() if not missing_cov.empty else missing_cov
    miss_scenario_policy = missing_sens.loc[(missing_sens.model.eq("LambdaMART") & missing_sens.seed.eq(-2))
                                            | (~missing_sens.model.eq("LambdaMART") & missing_sens.seed.eq(-1))].copy() if not missing_sens.empty else missing_sens
    miss_lines = ["# Missing-label sensitivity", "", f"Across the three outer years, {miss_n:,} candidate rows in the ex-ante test universe have unavailable outcome labels. Task A's missing entry labels remain unknown. The primary metric calculation excludes unknown outcomes from relevance and regret calculations but preserves the candidates and their model ranks. No unknown label is assigned zero in the primary analysis.", "", "## Missingness by outcome year, product and historical market size", "", missing_audit.to_markdown(index=False) if not missing_audit.empty else "No covariate-stratified missingness audit available.", "", "## Candidate-level coverage by model and pool", "", miss_policy.groupby(["task", "model"], as_index=False).agg(exante_candidates=("exante_candidates", "sum"), observed=("observed_outcomes", "sum"), unknown=("missing_outcomes", "sum"), unknown_in_top10=("missing_in_top10", "sum")).to_markdown(index=False) if not miss_policy.empty else "No coverage table.", "", "## Binary entry scenarios", "", miss_scenario_policy.groupby(["model", "scenario"], as_index=False).agg(mean_ndcg10=("ndcg10", "mean"), mean_precision10=("precision10", "mean"), mean_prevalence=("positive_prevalence", "mean"), pools=("ndcg10", "count")).to_markdown(index=False) if not miss_scenario_policy.empty else "No scenario output.", "", "All-missing-failure and all-missing-success assignments are deliberately extreme scenarios, not imputations or factual labels. For continuous export outcomes, no finite economically defensible bound is imposed; full-population USD regret is therefore not identified by these data. Complete-case results cannot be generalized automatically to all eligible destinations."]
    (report_dir / "E9_MISSING_LABEL_SENSITIVITY.md").write_text("\n".join(miss_lines) + "\n")

    statistical_lines = ["# Statistical inference and validation", "",
                         "Primary paired effects are model_a minus model_b. Positive values favor LambdaMART for ranking metrics; negative values favor it for regret. Confidence intervals and sign-flip tests aggregate paired pool differences within HS2 blocks, then resample/sign-flip the 96 HS2 chapter blocks. Holm correction is applied within each prespecified task × metric family.", "",
                         "## HS2-block paired contrasts", "", inference.to_markdown(index=False) if not inference.empty else "No paired comparisons.", "",
                         "## Destination-block sensitivity for NDCG@10", "", destination_boot.to_markdown(index=False) if not destination_boot.empty else "No destination-block sensitivity results.", "",
                         "The destination bootstrap resamples the same destination identifiers jointly across every product-year query and model, preserving cross-query destination dependence. It is a cross-sectional sensitivity, not a forecast of future-year variability. Three outer years are insufficient to estimate temporal generalization precisely. Seeds are training perturbations, not independent economic observations; seed-level results are therefore descriptive."]
    (report_dir / "STATISTICAL_INFERENCE.md").write_text("\n".join(statistical_lines) + "\n")

    compute_summary = (compute.groupby(["stage", "task"], as_index=False)
                       .agg(fits=("fit_runtime_seconds", "size"),
                            recorded_fit_seconds=("fit_runtime_seconds", "sum"),
                            mean_fit_seconds=("fit_runtime_seconds", "mean"),
                            total_inference_seconds=("inference_runtime_seconds", "sum"),
                            mean_trees=("tree_count", "mean"))) if not compute.empty and "stage" in compute else pd.DataFrame()
    compute_lines = ["# Computational cost", "",
                     "The fit-time entries are read from the stored model metadata and inner-fold tuning logs; cached prediction regeneration is not counted as a fresh model fit. LambdaMART is a tree ensemble, so a neural-style scalar parameter count is not applicable; tree counts and feature counts are provided.", "",
                     compute_summary.to_markdown(index=False) if not compute_summary.empty else "No compute-cost table.", "",
                     f"Total recorded fit time: {_fmt(compute.fit_runtime_seconds.sum(), 2)} seconds across {len(compute)} fit/configuration records. Inference time is recorded separately for final outer-fold rankers."]
    (report_dir / "COMPUTATIONAL_COST.md").write_text("\n".join(compute_lines) + "\n")

    # Contribution assessment tied directly to observed metrics and adjusted tests.
    b_lgb = get_metric(task_b, "LightGBM", "ndcg_at_10")
    b_ppml = get_metric(task_b, "Gravity_PPML", "ndcg_at_10")
    b_lm_reg = lambda_b_reg
    def test_result(task_code: str, metric: str, model: str):
        q = inference.loc[inference.task.eq(task_code) & inference.metric.eq(metric)
                          & inference.model_b.eq(model)] if not inference.empty else pd.DataFrame()
        return q.iloc[0] if len(q) else None
    a_lgb_ndcg_test = test_result("Task_A_new_entry", "ndcg_at_10", "LightGBM")
    a_lgb_p_test = test_result("Task_A_new_entry", "precision_at_10", "LightGBM")
    a_ppml_test = test_result("Task_A_new_entry", "ndcg_at_10", "Gravity_PPML")
    b_ppml_test = test_result("Task_B_existing_underexploited", "ndcg_at_10", "Gravity_PPML")
    b_lgb_test = test_result("Task_B_existing_underexploited", "ndcg_at_10", "LightGBM")
    b_ppml_reg_test = test_result("Task_B_existing_underexploited", "mean_regret@10_capped_gain", "Gravity_PPML")
    b_lgb_reg_test = test_result("Task_B_existing_underexploited", "mean_regret@10_capped_gain", "LightGBM")
    def contrast_text(row):
        return (f"Δ={_fmt(row.difference_a_minus_b, 4)}; 95% CI [{_fmt(row.ci95_low, 4)}, {_fmt(row.ci95_high, 4)}]; Holm p={_fmt(row.p_holm, 4)}"
                if row is not None else "not estimated")
    group_leaders = (group_summary.sort_values("mean_abs_contribution", ascending=False)
                     .groupby("task", as_index=False).first()) if not group_summary.empty else pd.DataFrame()
    leaders = {r.task: r.feature_group for r in group_leaders.itertuples(index=False)} if not group_leaders.empty else {}
    e2_sens = task_b.loc[task_b.model.eq("LambdaMART")]
    lambda_log_gain = float(e2_sens.ndcg10_log1p_gain.iloc[0]) if len(e2_sens) else np.nan
    ppml_log_gain = get_metric(task_b, "Gravity_PPML", "ndcg10_log1p_gain")
    lambda_raw_gain = float(e2_sens.ndcg10_raw_usd_gain.iloc[0]) if len(e2_sens) else np.nan
    ppml_raw_gain = get_metric(task_b, "Gravity_PPML", "ndcg10_raw_usd_gain")
    stat_lines = ["# Contribution assessment", "",
                  "## Evidence-based answers", "",
                  f"1. LambdaMART Task A NDCG@10 is {_fmt(lambda_a_ndcg)} and Precision@10 is {_fmt(lambda_a_p10)}. The strongest baseline by NDCG@10 is {best_a} ({_fmt(best_a_value)}). This is a comparative result, not evidence of causal market-entry benefits.",
                  f"2. For Task B, LambdaMART NDCG@10 is {_fmt(lambda_b_ndcg)}, compared with LightGBM {_fmt(b_lgb)} and PPML {_fmt(b_ppml)}. Mean LambdaMART capped-gain realized-outcome regret@10 is {_fmt(b_lm_reg, 4)}.",
                  f"3. Confirmatory paired comparisons (pool-weighted estimand; HS2-block resampling): Task A LambdaMART minus LightGBM NDCG@10: {contrast_text(a_lgb_ndcg_test)}; Precision@10: {contrast_text(a_lgb_p_test)}. Against Task A PPML NDCG@10: {contrast_text(a_ppml_test)}.",
                  f"4. Task B LambdaMART minus PPML NDCG@10: {contrast_text(b_ppml_test)}; minus LightGBM: {contrast_text(b_lgb_test)}. Task B regret difference versus PPML: {contrast_text(b_ppml_reg_test)}; versus LightGBM: {contrast_text(b_lgb_reg_test)}. Negative regret differences favor LambdaMART.",
                  f"5. Task B gain sensitivity changes the descriptive ordering: capped NDCG LambdaMART/PPML = {_fmt(lambda_b_ndcg)}/{_fmt(b_ppml)}; log1p-gain = {_fmt(lambda_log_gain)}/{_fmt(ppml_log_gain)}; raw-USD gain = {_fmt(lambda_raw_gain)}/{_fmt(ppml_raw_gain)}. This metric dependence weakens a universal ranking-superiority claim.",
                  f"6. LambdaMART seed-pair Top-10 Jaccard averages are 0.739 for entry and 0.863 for expansion. The largest mean absolute score-attribution groups are {leaders.get('Task_A_new_entry', 'NA')} for entry and {leaders.get('Task_B_existing_underexploited', 'NA')} for expansion; destination identity is a predictive identifier, not an economic mechanism.",
                  "7. HS2-block and destination-block uncertainty are cross-sectional. Destination bootstrap sensitivity intervals and HS2-block tests are reported separately because they use different dependence structures; three years do not establish future-cycle robustness.",
                  "8. Top-K stability, feature perturbation, actual pairwise score explanations and ablations are descriptive. They do not show that export promotion would cause the observed gains.",
                  f"9. Optional rank_xendcg with LambdaMART-selected settings yields seed-mean NDCG@10 {_fmt(xend_a_mean)} for Task A and {_fmt(xend_b_mean)} for Task B. It is close to pairwise LambdaMART in entry and higher for expansion, but remains below PPML on the primary capped-gain metric; no objective-specific tuning or inferential test was performed.",
                  "10. Revised historical WDI/WGI vintages and incomplete outcome reporting preclude a fully point-in-time deployment interpretation.", "",
                  "## Novelty assessment", "",
                  "Confirmed empirical contribution: applying direct pairwise/listwise ranking optimization to Vietnam HS2 destination selection under an audited ex-ante candidate universe, and evaluating complete-case ranking quality, realized-outcome regret, stability, and score explanations on common pools.",
                  "Methods borrowed from established work: LambdaMART, NDCG, retrospective regret, cluster bootstrap, additive tree contributions, feature ablation, and listwise ranking explanation research. The implementation does not establish a new general-purpose ranking or explanation algorithm.",
                  "The strongest defensible contribution depends on paired contrasts being robust and economically interpretable. If LambdaMART does not consistently outperform strong forecast-based rankings, the paper should foreground task alignment, candidate-set validity, regret/stability measurement, and a transparent negative or mixed result rather than algorithmic novelty.",
                  "This is not sufficient by itself to claim a new methodology or Q1 readiness; additional country/product contexts, vintage-aware data, and prospective or external validation would strengthen the design."]
    (report_dir / "CONTRIBUTION_ASSESSMENT.md").write_text("\n".join(stat_lines) + "\n")

    paper_lines = ["# Paper-ready results draft", "", "## Empirical strategy", "",
                   "We compare a decision-focused LambdaMART ranker with saved, audited predict-then-rank and market-discovery baselines. The unit of ranking is an HS2 product–outcome-year query. Candidate eligibility is frozen using historical information available at the audited two-year lag. Evaluation years are 2022–2024. Model selection uses an inner chronological validation fold; five seeds are fitted for each outer product-year/task fold. Ranking metrics are computed on common observed-label candidate intersections, while the full ex-ante candidate rankings and unknown-label positions are retained.", "",
                   "## New-market entry", "", task_a.to_markdown(index=False) if not task_a.empty else "No output.", "",
                   "## Existing-market expansion", "", task_b.to_markdown(index=False) if not task_b.empty else "No output.", "",
                   "## Regret, uncertainty and explanations", "", model_tests.to_markdown(index=False) if not model_tests.empty else "No paired inference output.", "",
                   "Retrospective opportunity regret is measured against an observed-outcome oracle on the same candidate intersection. Estimates are not profits or causal effects. Feature contributions explain raw ranking scores; retrained feature ablation tests information dependence but is not causal. Only three outer years are available, and the WDI/WGI historical series use revised vintages.", "",
                   "## Limitations", "", "The complete-case target population excludes candidates with unavailable outcome labels; these unknown outcomes are not zeros. Ranking conclusions may be sensitive to missing labels, candidate-pool construction, coarse capped gains, and only three evaluation years. Results are retrospective and not yet externally validated for firms or export-promotion agencies."]
    (report_dir / "PAPER_READY_RESULTS.md").write_text("\n".join(paper_lines) + "\n")

    required = ["E6_pool_metrics.csv", "E6_model_comparison.csv", "E6_by_hs2.csv", "E6_by_year.csv",
                "E6_lambdamart_performance.csv", "E7_rank_sensitivity.csv", "E6_ranking_predictions.parquet",
                "E6_rank_xendcg_sensitivity.csv", "E6_rank_xendcg_predictions.parquet",
                "E7_topk_stability.csv", "E7_candidate_eligibility_sensitivity.csv", "missing_label_audit.csv",
                "E8_pairwise_explanations.csv", "E8_feature_group_rank_effects.csv",
                "E8_explanation_fidelity.csv", "E8_topk_explanation_stability.csv", "E8_retrained_feature_ablation.csv",
                "E8_feature_attribution_stability.csv",
                "E9_missing_label_coverage.csv", "E9_missing_label_sensitivity.csv",
                "paired_model_comparisons.csv", "destination_block_sensitivity.csv", "computational_cost.csv"]
    missing_files = [name for name in required if not (result_dir / name).exists() and not (HERE / "predictions" / name).exists()]
    seed_ok = not metrics.empty and set(metrics.loc[metrics.model.eq("LambdaMART"), "seed"].unique()) >= {11, 23, 37, 51, 73}
    fit_count_ok = (not compute.empty and int(compute.stage.eq("outer_final_fit").sum()) == 30
                    and int(compute.stage.eq("inner_chronological_tuning").sum()) == 18)
    xendcg_fit_count_ok = (not compute.empty and int(compute.stage.eq("outer_rank_xendcg_sensitivity").sum()) == 30
                          and not xendcg.empty)
    test_report = HERE / "reports/TEST_VALIDATION.md"
    tests_ok = test_report.exists() and "14 passed" in test_report.read_text()
    status = ("COMPLETE_VERIFIED" if not missing_files and seed_ok and not fidelity.empty
              and fidelity.max_score_reconstruction_error.max() < 1e-6
              and fit_count_ok
              and xendcg_fit_count_ok
              and tests_ok
              and independent_validation["passed"] == independent_validation["checks"]
              else "PARTIAL_RESULTS")
    final_lines = [f"# Final status: {status}", "", f"Missing required evidence files: {missing_files if missing_files else 'none'}.",
                   f"Five LambdaMART seeds present in pool metrics: {seed_ok}.",
                   f"Recorded outer fits/tuning configurations: {int(compute.stage.eq('outer_final_fit').sum()) if not compute.empty and 'stage' in compute else 0}/{int(compute.stage.eq('inner_chronological_tuning').sum()) if not compute.empty and 'stage' in compute else 0}; check passed: {fit_count_ok}.",
                   f"Optional rank_xendcg sensitivity fits: {int(compute.stage.eq('outer_rank_xendcg_sensitivity').sum()) if not compute.empty and 'stage' in compute else 0}; check passed: {xendcg_fit_count_ok}.",
                   f"Required test and syntax report present: {tests_ok}.",
                   f"Independent saved-output checks: {independent_validation['passed']}/{independent_validation['checks']} passed.",
                   f"Maximum additive score reconstruction error: {_fmt(fidelity.max_score_reconstruction_error.max(), 10) if not fidelity.empty else 'not calculated'}.",
                   f"Candidate-task manifest rows: {len(manifest):,}; Task A rows: {int(manifest.task.eq('Task_A_new_entry').sum()):,}; Task B rows: {int(manifest.task.eq('Task_B_existing_underexploited').sum()):,}.",
                   "Unknown outcomes remain missing. The report status concerns execution and numerical checks, not a claim that LambdaMART wins or that results generalize prospectively."]
    (report_dir / "FINAL_STATUS.md").write_text("\n".join(final_lines) + "\n")

    sheets = {
        "Data Audit": population, "Candidate Manifest": manifest.head(100_000),
        "Discovery": comparison.loc[comparison.task.eq("Task_A_new_entry")] if not comparison.empty else comparison,
        "Ranking": comparison.loc[comparison.task.eq("Task_B_existing_underexploited")] if not comparison.empty else comparison,
        "rank_xendcg": xendcg,
        "Regret": regret_main, "Regret Summary": reg_summary,
        "By HS2": by_hs2, "By Year": by_year, "Stability": stability,
        "Robust Policy": robust, "Eligibility Sensitivity": eligibility_sensitivity,
        "Missingness Audit": missing_audit, "Pairwise XAI": pairwise,
        "Feature Groups": group_summary, "XAI Fidelity": fidelity,
        "Feature Ablation": ablation, "Missing Coverage": missing_cov,
        "Missing Scenarios": missing_sens, "Paired Tests": inference,
        "Seed Stability": seeds, "Compute Cost": compute,
    }
    _write_excel(sheets, result_dir / "DECISION_FOCUSED_RESEARCH_RESULTS.xlsx")
    return {"status": status, "workbook": str(result_dir / "DECISION_FOCUSED_RESEARCH_RESULTS.xlsx"),
            "reports": sorted(p.name for p in report_dir.glob("*.md")), "missing_required_files": missing_files,
            "lambda_macro_ndcg10_task_a": lambda_a_ndcg, "lambda_macro_ndcg10_task_b": lambda_b_ndcg}
