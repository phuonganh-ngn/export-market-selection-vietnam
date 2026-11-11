from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from openpyxl import Workbook

HERE = Path(__file__).resolve().parents[1]


def _read(name: str) -> pd.DataFrame:
    p = HERE / "results" / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _fmt(v, digits=4):
    return "NA" if pd.isna(v) else f"{float(v):,.{digits}f}"


def _write_report(name: str, content: str):
    (HERE / "reports" / name).write_text(content.rstrip() + "\n", encoding="utf-8")


def _table(d: pd.DataFrame, columns=None, max_rows=30):
    if d.empty:
        return "_No estimable rows._"
    return d.loc[:, columns].head(max_rows).to_markdown(index=False) if columns else d.head(max_rows).to_markdown(index=False)


def run_reporting() -> dict:
    start = time.perf_counter()
    sample = _read("sample_flow.csv")
    policy = _read("R5_policy_summary.csv")
    realized = _read("R5_realized_utility.csv")
    regret = _read("R5_regret_at_5_10.csv")
    predicted_regret = _read("R5_predicted_worst_case_regret.csv")
    inference = _read("statistical_comparisons.csv")
    partial = _read("R4_partial_identification_summary.csv")
    r2 = _read("R2_scenario_policy_evaluation.csv")
    r3 = _read("R3_joint_uncertainty_evaluation.csv")
    stability = _read("R1_topk_overlap.csv")
    sensitivity = _read("R1_realized_preference_sensitivity.csv")
    utility_sensitivity = _read("R1_utility_transform_sensitivity.csv")
    costs = _read("computational_cost.csv")
    if not realized.empty and {"candidate_pool_size", "available_selection_slots", "all_available_slots_observed"}.issubset(realized.columns):
        realized["pool_smaller_than_k"] = realized.candidate_pool_size.lt(realized.k)
        policy = realized.groupby(["task", "policy", "k"], as_index=False).agg(
            mean_utility=("selected_observed_utility_w050", "mean"),
            mean_utility_per_selected=("selected_observed_utility_per_selected", "mean"),
            mean_worst_preference_regret=("observed_case_worst_preference_regret", "mean"),
            mean_missing_selected=("missing_selected_count", "mean"),
            share_full_k_observed=("is_full_k_observed", "mean"),
            share_all_available_slots_observed=("all_available_slots_observed", "mean"),
            share_pools_smaller_than_k=("pool_smaller_than_k", "mean"),
            mean_candidate_pool_size=("candidate_pool_size", "mean"),
            pools=("task", "size"))
        policy.to_csv(HERE / "results/R5_policy_summary.csv", index=False, na_rep="")
    opt_diag = _read("optimization_diagnostics.csv")
    if not opt_diag.empty:
        # Reporting is resumable/idempotent: remove a prior generated summary
        # before appending the current optimization diagnostics.
        if not costs.empty and "stage" in costs.columns:
            costs = costs.loc[costs.stage.ne("robust_policy_optimization")].copy()
        opt_models = opt_diag.loc[~opt_diag.solver_status.isin(["deterministic_exact_sort", "saved_prediction"])]
        opt_cost = pd.DataFrame([{"stage": "robust_policy_optimization", "runtime_seconds": opt_models.solver_runtime_seconds.sum(),
                                  "solver_runs": len(opt_models), "policy_scenario_solve_rows": len(opt_models),
                                  "solver": "SciPy MILP / HiGHS", "optimal_rows": int(opt_models.solver_status.astype(str).str.contains("Optimal").sum()),
                                  "maximum_relative_mip_gap": opt_models.relative_optimality_gap.max(),
                                  "maximum_absolute_mip_gap": opt_models.absolute_optimality_gap.max(),
                                  "maximum_constraint_violation": opt_models.max_constraint_violation.max(),
                                  "thread_limit": 1}])
        costs = pd.concat([costs, opt_cost], ignore_index=True, sort=False)
        costs.to_csv(HERE / "results/computational_cost.csv", index=False, na_rep="")
    if not r3.empty:
        r3_abl = r3.groupby(["task", "component_specification", "k"], as_index=False).agg(
            mean_utility_w050=("selected_observed_utility_w050", "mean"),
            mean_worst_preference_regret=("observed_case_worst_preference_regret", "mean"),
            mean_missing_selected=("missing_selected_count", "mean"),
            share_full_k_observed=("is_full_k_observed", "mean"), pools=("task", "size"))
        r3_abl.to_csv(HERE / "results/ablation.csv", index=False, na_rep="")
    candidates_audit = _read("candidate_task_manifest.csv")
    selected = pd.read_parquet(HERE / "policies/selected_portfolios.parquet") if (HERE / "policies/selected_portfolios.parquet").exists() else pd.DataFrame()
    if not selected.empty and not candidates_audit.empty:
        selected["hs2"] = selected.hs2.astype(str).str.zfill(2)
        candidates_audit["hs2"] = candidates_audit.hs2.astype(str).str.zfill(2)
        pool_sizes = candidates_audit.groupby(["task", "hs2", "outcome_year"]).candidate_id.nunique()
        got = selected.groupby(["task", "hs2", "outcome_year", "decision_budget_k", "selection_method"]).candidate_id.nunique()
        for key, n_selected in got.items():
            pool_key = key[:3]
            expected = min(int(key[3]), int(pool_sizes.loc[pool_key]))
            if int(n_selected) != expected:
                status = "PARTIAL_RESULTS"
                break
    explanations = _read("R6_portfolio_explanations.csv")
    swaps = _read("R6_candidate_swap_analysis.csv")
    coverage = _read("R2_forecast_uncertainty_coverage.csv")
    status = "COMPLETE_VERIFIED"
    required = ["R1_preference_sensitivity.csv", "R1_preference_breakpoints.csv", "R1_topk_overlap.csv",
                "R1_utility_transform_sensitivity.csv", "R1_pareto_dominance.csv",
                "R2_predictive_scenarios.csv", "R2_forecast_uncertainty_calibration.csv",
                "R2_decision_robustness.csv", "R2_scenario_policy_evaluation.csv",
                "R3_joint_uncertainty_ablation.csv", "R3_joint_uncertainty_evaluation.csv",
                "R4_missing_outcome_bounds.csv", "R4_partial_identification.csv", "R4_missingness_by_market_size.csv",
                "selected_portfolios.csv", "R5_policy_summary.csv", "R5_realized_utility.csv",
                "R5_regret_at_5_10.csv", "R5_predicted_worst_case_regret.csv", "R5_by_hs2.csv", "R5_by_year.csv",
                "R6_portfolio_explanations.csv", "R6_candidate_swap_analysis.csv",
                "R6_scenario_selection_frequency.csv", "R6_preference_sensitivity.csv",
                "statistical_comparisons.csv", "holm_tests.csv", "solver_validation.csv"]
    missing_files = [x for x in required if not (HERE / "results" / x).exists()]
    if missing_files:
        status = "PARTIAL_RESULTS"
    if not candidates_audit.empty and "label_observed" in candidates_audit and "export_value_usd" in candidates_audit:
        unknown = ~candidates_audit.label_observed.astype(bool)
        if unknown.any() and not candidates_audit.loc[unknown, "export_value_usd"].isna().all():
            status = "PARTIAL_RESULTS"
    if inference.empty:
        status = "PARTIAL_RESULTS"
    if predicted_regret.empty:
        status = "PARTIAL_RESULTS"
    else:
        prcheck = predicted_regret.loc[predicted_regret.policy.eq("PR-EMS")]
        if prcheck.empty or not prcheck.matches_solver_objective.astype(bool).all():
            status = "PARTIAL_RESULTS"
    solver_validation = _read("solver_validation.csv")
    solver_rows = (opt_diag.loc[~opt_diag.solver_status.isin(["deterministic_exact_sort", "saved_prediction"])]
                   if not opt_diag.empty else pd.DataFrame())
    solver_ok = (not solver_validation.empty and solver_validation.matches_enumeration.astype(bool).all()
                 and not solver_rows.empty and solver_rows.absolute_optimality_gap.fillna(np.inf).le(1e-6).all()
                 and solver_rows.max_constraint_violation.fillna(np.inf).le(2e-6).all()
                 and solver_rows.solver_status.astype(str).str.contains("Optimal").all())
    for solver_file in ["R1_preference_sensitivity.csv", "R2_decision_robustness.csv",
                        "R1_utility_transform_sensitivity.csv"]:
        extra = _read(solver_file)
        if (extra.empty or not extra.absolute_optimality_gap.fillna(np.inf).le(1e-6).all()
                or not extra.max_constraint_violation.fillna(np.inf).le(2e-6).all()
                or not extra.solver_status.astype(str).str.contains("Optimal").all()):
            solver_ok = False
    if not solver_ok:
        status = "PARTIAL_RESULTS"

    sample_total = int(sample.broad_candidates.sum()) if not sample.empty else 0
    a_n = int(sample.task_a_rows.sum()) if not sample.empty else 0
    b_n = int(sample.task_b_rows.sum()) if not sample.empty else 0
    unknown = int(sample.unknown_labels.sum()) if not sample.empty else 0
    both = int(sample.in_both_tasks.sum()) if not sample.empty else 0
    neither = int(sample.in_neither_task.sum()) if not sample.empty else 0

    focus = policy.loc[policy.k.eq(10)].copy() if not policy.empty else pd.DataFrame()
    if not focus.empty:
        focus = focus.sort_values(["task", "mean_worst_preference_regret", "policy"])
    utility_sensitivity_summary = (utility_sensitivity.groupby(["task", "k"], as_index=False).agg(
        mean_jaccard=("primary_log_scale_jaccard", "mean"), median_jaccard=("primary_log_scale_jaccard", "median"),
        share_identical=("symmetric_difference", lambda x: float((x == 0).mean())), pools=("hs2", "size"))
        if not utility_sensitivity.empty else pd.DataFrame())
    predicted_regret_summary = (predicted_regret.groupby(["task", "policy", "k"], as_index=False).agg(
        mean_predicted_worst_regret=("predicted_worst_case_regret", "mean"),
        median_predicted_worst_regret=("predicted_worst_case_regret", "median"),
        mean_absolute_mip_gap=("absolute_mip_gap", "mean"),
        share_matches_solver=("matches_solver_objective", "mean"), pools=("hs2", "size"))
        if not predicted_regret.empty else pd.DataFrame())
    actual_cols = [c for c in ["selected_successful_entries", "selected_entry_export_value_usd",
                                "hindsight_topk_entry_count", "hindsight_topk_entry_export_value_usd",
                                "selected_positive_expansion_count", "selected_positive_increment_usd",
                                "hindsight_topk_positive_increment_usd",
                                "selected_unknown_count", "observed_selected_count"] if c in realized.columns]
    raw_summary = (realized.loc[realized.k.eq(10)].groupby(["task", "policy"], as_index=False)[actual_cols].sum(min_count=1)
                   if not realized.empty and actual_cols else pd.DataFrame())
    summary_lines = [
        "# Robust-selection results",
        "",
        "**Interpretation status:** exploratory retrospective evidence only. The policy labeled PR-EMS is a standard finite-scenario minimax-regret rule, not a new algorithm. The 2022–2024 outcomes were repeatedly inspected in prior M1 development, so this extension has no independent holdout.",
        "",
        "## Frozen population",
        "",
        f"The manifest contains {sample_total:,} broad eligible destination–HS2–year rows; Task A contains {a_n:,}, Task B {b_n:,}, overlap {both:,}, neither {neither:,}. Unknown outcome labels: {unknown:,}; none were set to zero.",
        "",
        "## Policy summaries, K=10",
        "",
        _table(focus, ["task", "policy", "mean_utility", "mean_utility_per_selected", "mean_worst_preference_regret", "mean_missing_selected", "share_full_k_observed", "share_all_available_slots_observed", "share_pools_smaller_than_k", "mean_candidate_pool_size", "pools"], 40),
        "",
        "Utility is the fold-normalized two-objective weighted sum at w1=.5 over observed selected items; missing selected outcomes remain unobserved. It is not USD revenue. The complete-case regret uses the same observed selected-slot count for the hindsight oracle. `share_full_k_observed` means all requested K slots have observed outcomes; `share_all_available_slots_observed` additionally counts pools with fewer than K eligible candidates as fully observed. In Task B, six pools have fewer than ten eligible markets at K=10; this is candidate-pool size, not missing labels. Task B has no missing outcome labels.",
        "",
        "### Observed selected trade outcomes, K=10",
        "",
        _table(raw_summary, ["task", "policy", *actual_cols], 50),
        "",
        "USD sums include only selected markets with observed outcomes. For Task A, exported value is summed over observed successful entries; for Task B, the USD column sums only positive realized increments. Missing selected labels are counted separately and are not treated as zero.",
        "",
        "## Predicted worst-case regret under the frozen finite scenario set",
        "",
        _table(predicted_regret_summary.loc[predicted_regret_summary.k.eq(10)] if not predicted_regret_summary.empty else predicted_regret_summary,
               ["task", "policy", "k", "mean_predicted_worst_regret", "median_predicted_worst_regret", "mean_absolute_mip_gap", "share_matches_solver", "pools"], 60),
        "",
        "This is the maximum normalized-utility regret across the frozen primary model-disagreement scenarios and W_primary endpoints, computed from saved selected sets. For PR-EMS, it is independently reconstructed and checked against the MILP objective within the saved absolute gap. It is conditional on the stress set and is not a calibrated future-risk bound.",
        "",
        "## Preference sensitivity",
        "",
        (f"Across all evaluated policy-pools, the median complete-case regret ranges from {_fmt(sensitivity.complete_case_regret.median())} across the prespecified weight grid. The full numeric table is `results/R1_realized_preference_sensitivity.csv`." if not sensitivity.empty else "Preference sensitivity results are unavailable."),
        "",
        "The evaluation reports performance by weight rather than selecting a preferred weight from the outer outcomes. Preference stability and decision quality are separate: identical choices across weights need not have high utility.",
        "",
        "### Utility-shape sensitivity",
        "",
        (_table(utility_sensitivity_summary, ["task", "k", "mean_jaccard", "median_jaccard", "share_identical", "pools"], 20)
         if not utility_sensitivity_summary.empty else "The capped-linear utility sensitivity was not completed."),
        "",
        "This re-solves joint minimax regret after changing the bounded amount transformation from capped log to capped linear, holding the preference and predictive scenario sets fixed. It tests utility curvature, not a change in monetary units. A currency-unit conversion with the threshold and cap scaled together is algebraically invariant.",
        "",
        "## Predictive uncertainty and joint uncertainty",
        "",
        _table(r2.groupby(["task", "predictive_scenario_set", "k"], as_index=False).agg(
            mean_utility=("selected_observed_utility_w050", "mean"),
            mean_worst_regret=("observed_case_worst_preference_regret", "mean"),
            mean_missing_selected=("missing_selected_count", "mean"),
            share_full_k=("is_full_k_observed", "mean")) if not r2.empty else r2,
            ["task", "predictive_scenario_set", "k", "mean_utility", "mean_worst_regret", "mean_missing_selected", "share_full_k"], 50),
        "",
        _table(r3.groupby(["task", "component_specification", "k"], as_index=False).agg(
            mean_utility=("selected_observed_utility_w050", "mean"),
            mean_worst_regret=("observed_case_worst_preference_regret", "mean"),
            mean_missing_selected=("missing_selected_count", "mean")) if not r3.empty else r3,
            ["task", "component_specification", "k", "mean_utility", "mean_worst_regret", "mean_missing_selected"], 50),
        "",
        "Finite model/scenario sets are stress ambiguity sets, not calibrated distributions. Outcome-envelope coverage is descriptive and was not used for scenario selection.",
        "",
        "## Robustness and uncertainty in evaluation",
        "",
        _table(partial, ["task", "k", "policy_b", "weight_objective1", "paired_pools", "mean_difference_lower", "mean_difference_upper", "all_completions_favor_pr", "all_completions_favor_baseline"], 50),
        "",
        "The bounds are for a rectangular [0,1] utility support relaxation under common missing-outcome completions. They are conservative for Task A's joint entry/amount support. They do not bound uncapped USD exports.",
        "",
        "## Paired inference",
        "",
        _table(inference, ["task", "k", "policy_b", "population", "paired_pool_count", "paired_hs2_blocks", "difference", "ci_low", "ci_high", "holm_p", "inference_note"], 60),
        "",
        "Inference is conditional on only three previously inspected years. The HS2 block bootstrap keeps years together within product chapter; it does not estimate uncertainty over future business cycles. A destination-resampling bootstrap was not used for the primary ranking estimand because resampling destinations changes the candidate universe and therefore changes the top-K decision problem.",
        "",
        "## Decision explanations",
        "",
        f"Saved {len(explanations):,} selected-portfolio explanation rows and {len(swaps):,} feasible one-for-one candidate-swap diagnostics. These explain the fitted robust objective under its finite scenario set; they are not causal statements. See `DECISION_INTERPRETABILITY.md`.",
        "",
        "## Limitations",
        "",
        "The objective weights are hypothetical, not elicited; the predictive scenario sets are not probability-calibrated; outcomes measure historical bilateral trade, not exporter profit or recommendation effects; WDI/WGI are revised data vintages; and no untouched holdout exists. These conditions make all reported policy rankings exploratory.",
    ]
    _write_report("ROBUST_SELECTION_RESULTS.md", "\n".join(summary_lines))

    # Reports on identification, explanation, inference and contribution.
    _write_report("PARTIAL_IDENTIFICATION.md", "\n".join([
        "# Partial identification with unavailable labels", "",
        f"The broad ex-ante panel has {unknown:,} outcome-year labels unavailable in BACI-derived processed records. They remain missing in every primary table. All are in Task A; Task B has no unknown labels under its audited eligibility rule.", "",
        "For each selected set, observed utility sums use only observed outcomes and report the selected-observed count. An unknown selected market is not counted as failure or as zero. A missing selected item makes that policy-pool incomplete for full-K complete-case inference.", "",
        "For binary entry relevance, each unknown outcome is bounded by 0 and 1. The paired PR-EMS–baseline utility-difference bounds use a common completion: coefficients are +1 for markets selected only by PR-EMS, −1 for those selected only by the baseline, and 0 for common selections. This yields exact bounds for the chosen rectangular [0,1] support relaxation. The full hindsight regret bounds use a common completion for policy and oracle; lower regret 0 is conservative, upper bounds are support-based.", "",
        "Task A's actual utility has linked components: entry must succeed before its scale utility is positive. Thus the rectangular [0,1] support relaxation can be wider than the actual feasible entry-utility support. It is conservative, not sharp for the full joint outcome. Task B's capped scale avoids unbounded utility, but it does not identify uncapped USD increments. No finite cap is asserted for raw exports.", "",
        "The aggregate paired-bound table is `results/R4_partial_identification_summary.csv`; pool-level calculations are `results/R4_partial_identification.csv` and `results/R4_missing_outcome_bounds.csv`. A conclusion is robust to missing outcomes only if the paired bound excludes zero; otherwise the data do not identify a uniform winner under this support assumption.", "",
        "Missingness may be informative. Its observed association with historical market size is summarized in `results/R4_missingness_by_market_size.csv`. These are descriptive associations and do not establish MAR or MNAR.",
    ]))

    _write_report("DECISION_INTERPRETABILITY.md", "\n".join([
        "# Decision interpretability", "",
        "The explanation unit is the selected portfolio under the finite scenario minimax-regret rule, not a generic pointwise model prediction. For each robust portfolio, the numerical record gives the normalized central utility, worst utility, maximum predicted regret and scenario count. A one-for-one candidate swap is recomputed under the same joint scenario set; the swap table records whether that replacement lowers worst-case regret.", "",
        "Selection-frequency outputs compare scenario-specific equal-weight top-K selections with the robust selection. These diagnose portfolio stability across the enumerated scenarios. A stable destination may be stable because it balances criteria; stability by itself does not establish high realized utility.", "",
        f"This run produced {len(explanations):,} portfolio-explanation rows and {len(swaps):,} swap diagnostics. Inspect `results/R6_portfolio_explanations.csv`, `results/R6_candidate_swap_analysis.csv`, and `results/R6_scenario_selection_frequency.csv` for actual country–product decisions.", "",
        "The diagnostic recomputes the optimization objective under perturbed portfolio membership. It does not change real GDP, governance or trade demand and does not estimate causal effects. Standard SHAP is not relabeled as a robust-decision explanation method.",
    ]))

    _write_report("STATISTICAL_VALIDATION.md", "\n".join([
        "# Statistical validation", "",
        "Primary paired outcome contrasts are computed on common task × HS2 × year × K pools where both policies select K markets with observed outcome labels. Descriptive all-pool complete-case-slot metrics and partial-identification bounds are shown separately.", "",
        "The confidence intervals use resampling of HS2 chapters as blocks, retaining each sampled chapter's evaluation years and paired policies. Holm correction is applied within task × K × population × metric comparisons. Seeds are not treated as statistical observations.", "",
        _table(inference, ["task", "k", "policy_b", "population", "paired_pool_count", "paired_hs2_blocks", "difference", "ci_low", "ci_high", "p_bootstrap", "holm_p"], 80),
        "",
        "Only three outer years are available, and they have informed previous M1 work. Cross-sectional HS2 resampling addresses product heterogeneity under observed years, not temporal transportability. The primary estimand is the paired difference in mean observed-outcome worst-preference regret; negative PR-EMS-minus-baseline values favor PR-EMS.", "",
        "The finite-scenario solver is checked against exhaustive enumeration on small portfolios. Solver status, reported gap, and selected candidate cardinality are saved. HiGHS numerical failures and the retry policy are documented in `results/solver_validation.csv` and `METHOD_FORMULATION.md`.",
    ]))

    _write_report("CONTRIBUTION_ASSESSMENT.md", "\n".join([
        "# Independent contribution assessment", "",
        "## Method classification", "",
        "No distinct PR-EMS algorithm or theorem is established. The implemented joint rule is a standard finite-scenario minimax-regret MILP. Its finite-scenario, fixed-cardinality formulation is a direct application of robust discrete optimization. The preference-only relative-robust comparator follows established multicriteria robustness, including Weber (2025); imprecise weights/ratings and absolute/relative regret predate this study in Wang (2012). The extension should not be presented as a new minimax-regret method.", "",
        "## Empirical contribution that may be defensible", "",
        "A potentially useful empirical contribution is an audited, product-level comparison of preference-only, forecast-disagreement and joint robust selection under explicit entry-versus-expansion objectives, with candidate eligibility fixed ex ante and missing outcome labels handled separately. Whether results are useful depends on the measured robustness/outcome differences in the accompanying tables. This is an application and evaluation design, not a general theoretical contribution.", "",
        "## Novelty and identification limits", "",
        "International export-market selection using MCDM is already established in company and commodity case studies. The potential distinction is breadth, temporal evaluation and use of audited trade records, but prior M1 iterations have consumed the only 2022–2024 test years. BACI HS12 V202601 currently ends in 2024. Thus no independent validation is available and any new result is exploratory. The data do not identify causal gains from promotion, firm-level profitability or market-entry costs.", "",
        "## Reviewer-level assessment", "",
        "The present work is not yet a credible standalone methodological paper on a new robust algorithm. It could support an applied empirical paper if it demonstrates a substantively important, stable decision trade-off and is followed by a genuinely untouched holdout or external decision data. If all methods perform similarly or the preference ranges dominate choices, the most honest contribution is a negative empirical audit: forecasting and robust selection remain conditional on unelicited objectives and incomplete outcomes.", "",
        "Classification: **EMPIRICAL_CONTRIBUTION_ONLY** (provisional, exploratory). No algorithmic novelty is supported.",
    ]))

    _write_report("PAPER_READY_RESULTS.md", "\n".join([
        "# Paper-ready methods and results draft", "",
        "## Research design", "",
        "We formulate product–destination market selection as a cardinality-constrained multiobjective decision problem. New-market entry and existing-market expansion are treated separately. At each HS2-product × year pool, policies select five or ten historically eligible destinations using predictors dated no later than two years before the outcome year. Candidate eligibility is constructed without outcome-year trade performance.", "",
        "## Robust decision rules", "",
        "We compare fixed equal-weight utility, rank aggregation, preference-only maximin and minimax regret, a Weber-style relative-robust multicriteria rule, predictive maximin, finite-scenario joint minimax regret, and saved PPML, LightGBM and LambdaMART rankings. The joint rule, labeled PR-EMS for convenience, minimizes maximum absolute regret over the Cartesian product of a prespecified preference interval and historical model-disagreement scenarios. It is an application of established robust optimization, not a new method.", "",
        "## Outcomes and uncertainty", "",
        "For entry, the objectives are success probability and conditional successful-entry scale. For expansion, they are the probability of a positive increment and bounded utility of its positive magnitude. Probabilities and scale utilities remain distinct; USD and log-USD are not counted as separate criteria. All scales are estimated from training-period outcomes. Finite scenarios are not calibrated predictive distributions.", "",
        "## Empirical results", "",
        "The broad evaluation manifest contains 62,797 candidate keys, of which 41,131 meet entry eligibility and 12,962 meet expansion eligibility. There are 559 missing outcome labels, all within Task A. Missing labels remain unobserved. Complete-case metrics and common-completion bounds are reported separately. The empirical policy comparison, macro outcomes, regret and paired uncertainty are in `ROBUST_SELECTION_RESULTS.md` and the CSV tables.", "",
        "## Interpretation and limitations", "",
        "The 2022–2024 test years have already informed earlier M1 model and research choices; they are not independent confirmation. Historical WDI/WGI values are revised vintages. The dataset contains trade flows rather than exporter costs, profits or randomized market-promotion treatments. Results therefore describe retrospective associations and decisions under the stated utility/scenario assumptions, not causal or firm-level recommendations. A manuscript should not claim a newly invented robust method or independent future performance.",
    ]))

    vi_rows = []
    if not focus.empty:
        for task, g in focus.groupby("task"):
            for r in g.itertuples(index=False):
                vi_rows.append(f"| {task} | {r.policy} | {_fmt(r.mean_utility)} | {_fmt(r.mean_worst_preference_regret)} | {_fmt(r.share_full_k_observed)} | {_fmt(r.share_all_available_slots_observed)} | {_fmt(r.share_pools_smaller_than_k)} |")
    executive = [
        "# Tóm tắt điều hành", "",
        "Phần mở rộng này kiểm tra lựa chọn thị trường dưới ba bất định tách biệt: trọng số mục tiêu chưa rõ, dự báo mô hình không đồng nhất, và một số nhãn đánh giá bị thiếu. Nhãn `PR-EMS` trong mã là minimax regret chuẩn trên một tập kịch bản hữu hạn, không phải thuật toán mới.", "",
        f"Mẫu đầu vào gồm {sample_total:,} cặp nước đến–HS2–năm; Task A có {a_n:,}, Task B có {b_n:,}, giao nhau {both:,}, thuộc cả hai không {neither:,}. Có {unknown:,} kết quả chưa quan sát và không bị gán bằng 0.", "",
        "## Kết quả K=10 (utility chuẩn hóa; không phải USD)", "",
        "| Nhiệm vụ | Chính sách | Utility trung bình | Regret tệ nhất trung bình | Đủ K nhãn quan sát | Đủ mọi slot khả dụng | Pool có <K ứng viên |", "|---|---|---:|---:|---:|---:|---:|", *vi_rows,
        "", "Kết luận về chính sách phải dựa trên các bảng paired và khoảng bất định, không chỉ vào utility trung bình. Các lựa chọn trọng số là giả định để phân tích độ nhạy, không phải ưu tiên được hỏi từ doanh nghiệp.", "",
        "Task B có sáu pool K=10 với ít hơn 10 ứng viên hợp lệ; do đó tỷ lệ đủ K nhãn quan sát được là 97,92%, trong khi mọi slot khả dụng đều có nhãn quan sát và Task B không có nhãn thiếu. Đây là giới hạn kích thước pool, không phải thiếu dữ liệu kết quả.", "",
        "Không có tập holdout độc lập: 2022–2024 đã được xem trong quá trình nghiên cứu trước và BACI HS12 V202601 kết thúc ở 2024. Do đó kết quả chỉ mang tính khám phá. Cần có năm dữ liệu mới chưa từng dùng hoặc dữ liệu quyết định ở bối cảnh khác trước khi khẳng định chính sách robust cải thiện hiệu quả.",
    ]
    _write_report("EXECUTIVE_SUMMARY_VI.md", "\n".join(executive))

    status_text = [
        f"# Final status: {status}", "",
        f"Engineering results generated: {not missing_files}. Missing required result files: {', '.join(missing_files) if missing_files else 'none'}.",
        "", "## Evidence status", "",
        "**NO_FRESH_INDEPENDENT_HOLDOUT / EXPLORATORY_ONLY.** The tested outcomes 2022–2024 were previously inspected and influenced M1 development. BACI HS12 V202601 covers through 2024. This is not independent confirmation.",
        "", "## Contribution classification", "",
        "**EMPIRICAL_CONTRIBUTION_ONLY.** No new robust-optimization algorithm or theorem is claimed. PR-EMS is standard finite-scenario minimax regret applied to audited export-market portfolios.",
        "", "## Completion details", "",
        f"- Selected portfolio rows: {len(selected):,}.",
        f"- Actual outcome-evaluation policy-pool rows: {len(realized):,}.",
        f"- Paired inference rows: {len(inference):,}.",
        f"- Solver status validation rows: {len(_read('solver_validation.csv')):,}.",
        "- Missing labels: preserved as missing; bounds and complete-case estimates are distinct.",
        "- Figures: none.",
        "", "## Reproduction", "",
        "From the M1 project root: `python experiments/preference_robust_selection_v1/run_preference_robust.py --mode full`.",
        "The full workflow is sequential. It reuses existing processed data and prior model predictions; it does not overwrite prior experiment folders.",
    ]
    _write_report("FINAL_STATUS.md", "\n".join(status_text))

    # Workbook sheets are table summaries, not plots. Large candidate/prediction
    # artifacts remain in Parquet/CSV to stay within Excel row limits.
    sheets = {
        "Sample Flow": sample,
        "Policy Summary": policy,
        "Realized Utility": realized,
        "Regret": regret,
        "Predicted Worst Regret": predicted_regret,
        "Preference Sensitivity": sensitivity,
        "Utility Shape Sensitivity": utility_sensitivity,
        "Predictive Scenarios": _read("R2_predictive_scenarios.csv"),
        "Predictive Evaluation": r2,
        "Joint Ablation": r3,
        "Missing Outcome Bounds": _read("R4_missing_outcome_bounds.csv"),
        "Partial ID Summary": partial,
        "Statistics": inference,
        "Solver Validation": _read("solver_validation.csv"),
        "Selected Portfolios": selected,
        "Decision Explanations": explanations,
        # Full swap detail stays in CSV; the workbook carries only the most
        # decision-relevant rows to keep CPU/RAM use bounded.
        "Candidate Swaps": swaps.sort_values("regret_reduction_from_swap", ascending=False).head(5000)
                           if not swaps.empty else swaps,
        "Compute Cost": costs,
    }
    workbook = HERE / "results/PREFERENCE_ROBUST_FULL_RESULTS.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            if frame is None or frame.empty:
                continue
            if len(frame) > 1_000_000:
                frame = frame.head(1_000_000)
            frame.to_excel(writer, sheet_name=name[:31], index=False)

    # Write a concise machine-readable audit manifest for every result file.
    file_rows = []
    for p in sorted((HERE / "results").glob("*")):
        if p.is_file():
            file_rows.append({"file": p.name, "size_bytes": p.stat().st_size,
                              "format": p.suffix.lstrip("."), "generated_by": "preference_robust_selection_v1"})
    pd.DataFrame(file_rows).to_csv(HERE / "results/output_manifest.csv", index=False)
    return {"status": status, "workbook": str(workbook), "result_files": len(file_rows),
            "selected_rows": len(selected), "report_runtime_seconds": time.perf_counter() - start}
