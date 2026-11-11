from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]


def _read(name: str) -> pd.DataFrame:
    p = HERE / "results" / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _fmt(x, n=3):
    if pd.isna(x): return "NA"
    if abs(float(x)) >= 1e6: return f"{float(x)/1e6:.{n}f} million"
    return f"{float(x):.{n}f}"


def _best_text(df: pd.DataFrame, metric: str, lower: bool = True) -> str:
    if df.empty or metric not in df: return "NA"
    x=df.dropna(subset=[metric])
    if x.empty: return "NA"
    r=x.loc[x[metric].idxmin() if lower else x[metric].idxmax()]
    return f"{r.get('policy',r.get('model',''))}: {_fmt(r[metric])}"


def generate_reports() -> None:
    reports=HERE/"reports"; reports.mkdir(exist_ok=True)
    results=HERE/"results"
    sample=_read("sample_flow.csv")
    static=_read("T1_static_task_routing.csv")
    blend=_read("T2_static_blending.csv")
    route=_read("T3_adaptive_routing.csv")
    freq=_read("T3_expert_selection_frequency.csv")
    coverage=_read("T5_coverage_risk.csv")
    matched=_read("T5_matched_coverage_comparison.csv")
    decisions=_read("T5_pool_acceptance_decisions.csv")
    inference=_read("inference.csv")
    calibration=_read("T4_risk_calibration.csv")
    highrisk=_read("T5_high_regret_detection.csv")
    dossiers=_read("T7_review_dossiers.csv")
    explanations=_read("T6_routing_explanations.csv")
    fidelity=_read("T7_explanation_fidelity.csv")
    if not explanations.empty:
        route_diff=pd.to_numeric(explanations.clipped_predicted_regret_difference_lgb_minus_ppml,errors="coerce")
        expected_route=np.where(route_diff.ge(-1e-12),"Gravity_PPML","LightGBM")
        fidelity=pd.DataFrame({
            "task":explanations.task,"outcome_year":explanations.outcome_year,"hs2":explanations.hs2,
            "candidate_pool_id":explanations.candidate_pool_id,
            "raw_gate_score_difference_error":pd.to_numeric(explanations.route_reconstruction_error,errors="coerce"),
            "raw_gate_score_reconstructed":pd.to_numeric(explanations.route_reconstruction_error,errors="coerce").lt(1e-8),
            "route_decision_reproduced":explanations.selected_expert.eq(expected_route),
            "tie_after_nonnegative_clipping":explanations.both_regret_scores_clipped_to_zero_tie.astype(bool)})
        fidelity.to_csv(results/"T7_explanation_fidelity.csv",index=False,na_rep="")
    # Rebuild explainability artifacts from saved score and pool records so the
    # report-only command reproduces the same review dossier and frequency table.
    candidate_path=HERE/"predictions/ergm_candidate_predictions.parquet"
    pool_path=HERE/"predictions/ergm_pool_predictions.parquet"
    if candidate_path.exists() and pool_path.exists() and not route.empty:
        from .ergm_pipeline import _review_dossiers
        dossiers,reasons,_=_review_dossiers(pd.read_parquet(candidate_path),pd.read_parquet(pool_path),route,coverage)
        dossiers.to_csv(results/"T7_review_dossiers.csv",index=False,na_rep="")
        reasons.to_csv(results/"T7_deferral_reason_codes.csv",index=False,na_rep="")
    deferred_hs2=pd.DataFrame()
    if not decisions.empty:
        d80=decisions.loc[decisions.policy.eq("ERGM_FULL") &
            decisions.coverage_mode.eq("development_threshold") & decisions.requested_coverage.eq(.8)].copy()
        if not d80.empty:
            deferred_hs2=d80.groupby(["task","hs2"],as_index=False).agg(
                evaluated_pools=("candidate_pool_id","nunique"),deferred_pools=("accepted",lambda x:int((~x.astype(bool)).sum())),
                pools_without_evaluable_regret=("regret@10",lambda x:int(x.isna().sum())))
            deferred_hs2["deferral_rate"] = deferred_hs2.deferred_pools / deferred_hs2.evaluated_pools
            deferred_hs2.to_csv(results/"T5_deferred_hs2_frequency.csv",index=False,na_rep="")
    missing=_read("T10_missingness_by_task.csv")
    missbounds=_read("T10_missing_outcome_sensitivity.csv")
    ablation=_read("T9_ablation.csv")
    cost=_read("computational_cost.csv")
    temporal=_read("temporal_validation.csv")
    model_diag=_read("historical_oof_model_diagnostics.csv")

    disagreement_diag=pd.DataFrame()
    if not route.empty:
        adaptive=route.loc[route.policy.eq("ADAPTIVE_ROUTING")].copy()
        adaptive["hs2"]=adaptive.hs2.astype(str).str.zfill(2)
        pool_pred_path=HERE/"predictions/ergm_pool_predictions.parquet"
        if pool_pred_path.exists():
            pool_pred=pd.read_parquet(pool_pred_path)
            pool_pred["hs2"]=pool_pred.hs2.astype(str).str.zfill(2)
            adaptive=adaptive.merge(pool_pred[["task","outcome_year","hs2","rank_displacement_norm"]],
                on=["task","outcome_year","hs2"],how="left",validate="one_to_one")
        pp=route.loc[route.policy.eq("STATIC_PPML"),["task","outcome_year","hs2","realized_regret10"]].rename(columns={"realized_regret10":"ppml_regret10"})
        ll=route.loc[route.policy.eq("STATIC_LIGHTGBM"),["task","outcome_year","hs2","realized_regret10"]].rename(columns={"realized_regret10":"lightgbm_regret10"})
        pp["hs2"]=pp.hs2.astype(str).str.zfill(2); ll["hs2"]=ll.hs2.astype(str).str.zfill(2)
        q=adaptive.merge(pp,on=["task","outcome_year","hs2"],validate="one_to_one").merge(
            ll,on=["task","outcome_year","hs2"],validate="one_to_one")
        q["expert_disagreement"] = 1-pd.to_numeric(q.top10_jaccard,errors="coerce") + pd.to_numeric(q.rank_displacement_norm,errors="coerce")
        q["best_expert_regret10_hindsight"] = q[["ppml_regret10","lightgbm_regret10"]].min(axis=1)
        dd=[]
        groups=[((t,"all_years"),d) for t,d in q.groupby("task")]
        groups += [((t,int(y)),d) for (t,y),d in q.groupby(["task","outcome_year"])]
        for keys,g in groups:
            hindsight_match=((g.selected_expert.eq("Gravity_PPML") & g.ppml_regret10.le(g.lightgbm_regret10)) |
                             (g.selected_expert.eq("LightGBM") & g.lightgbm_regret10.lt(g.ppml_regret10)))
            dd.append({"task":keys[0],"evaluation_year":keys[1],"pools":len(g),
                "spearman_disagreement_vs_adaptive_regret10":g.expert_disagreement.corr(g.realized_regret10,method="spearman"),
                "spearman_disagreement_vs_best_expert_regret10":g.expert_disagreement.corr(g.best_expert_regret10_hindsight,method="spearman"),
                "mean_adaptive_regret_low_disagreement_quartile":g.loc[g.expert_disagreement<=g.expert_disagreement.quantile(.25),"realized_regret10"].mean(),
                "mean_adaptive_regret_high_disagreement_quartile":g.loc[g.expert_disagreement>=g.expert_disagreement.quantile(.75),"realized_regret10"].mean(),
                "adaptive_router_matches_hindsight_better_expert_share":float(hindsight_match.mean()),
                "interpretation":"exploratory retrospective association; no causal interpretation"})
        disagreement_diag=pd.DataFrame(dd)
        disagreement_diag.to_csv(results/"T4_disagreement_diagnostics.csv",index=False,na_rep="")

    seed_stability=pd.DataFrame([
        {"component":"Ridge routing/reliability gate","seed_count":1,"seeds":"deterministic","status":"not applicable","note":"Fixed preprocessing and Ridge(alpha=10); no stochastic training."},
        {"component":"Historical rolling-origin LightGBM experts","seed_count":1,"seeds":"11","status":"NOT_ESTIMATED","note":"OOF predictions reconstructed with one fixed seed; seed sensitivity was not run."},
        {"component":"Outer historical LightGBM/PPML predictions","seed_count":1,"seeds":"archived seed mean","status":"REUSED_ARCHIVE","note":"Existing M1 saved predictions reused; original per-seed predictions are not refit here."}])
    seed_stability.to_csv(results/"seed_stability.csv",index=False,na_rep="")

    # Summarize raw pool decisions directly so pooled figures are candidate-pool
    # weighted, with observed-label denominators stated explicitly.
    def pool_summary(d: pd.DataFrame, mode_col: str = "coverage_mode") -> pd.DataFrame:
        rows=[]
        if d.empty: return pd.DataFrame()
        for keys,g in d.groupby(["task","policy",mode_col,"requested_coverage"],dropna=False):
            accepted=g.loc[g.accepted.astype(bool)]
            valid=accepted.loc[accepted["regret@10"].notna()]
            rows.append({"task":keys[0],"policy":keys[1],"coverage_mode":keys[2],
                "requested_coverage":keys[3],"automation_coverage":float(g.accepted.mean()),
                "all_pools":int(g.candidate_pool_id.nunique()),"accepted_pools":int(g.accepted.sum()),
                "accepted_pools_with_observed_metrics":int(len(valid)),
                "mean_regret10_accepted":float(valid["regret@10"].mean()) if len(valid) else np.nan,
                "mean_ndcg10_accepted":float(valid["ndcg@10"].mean()) if len(valid) else np.nan,
                "mean_precision10_accepted":float(valid["precision@10"].mean()) if len(valid) else np.nan,
                "observed_positive_usd_top10":float(valid["selected_positive_usd@10"].mean()) if len(valid) else np.nan,
                "mean_unknown_candidates_in_full_top10":float(accepted["missing_in_full_score_top10"].mean()) if len(accepted) else np.nan})
        return pd.DataFrame(rows)

    econ=pool_summary(decisions)
    if not matched.empty:
        random=matched.loc[matched.policy.eq("RANDOM_DEFERRAL_ADAPTIVE_ROUTE")].groupby(
            ["task","policy","coverage_mode","requested_coverage"],as_index=False).agg(
                automation_coverage=("automation_coverage","mean"),all_pools=("all_pools","sum"),
                accepted_pools=("accepted_pools","sum"),accepted_pools_with_observed_metrics=("accepted_pools_with_observed_metrics","mean"),
                mean_regret10_accepted=("mean_regret10_accepted","mean"),mean_ndcg10_accepted=("mean_ndcg10_accepted","mean"),
                mean_precision10_accepted=("mean_precision10_accepted","mean"),observed_positive_usd_top10=("mean_observed_positive_usd_top10","mean"))
        if not random.empty: econ=pd.concat([econ,random],ignore_index=True,sort=False)
    econ.to_csv(results/"T8_economic_opportunity.csv",index=False,na_rep="")
    econ.to_csv(results/"T8_export_value_usd.csv",index=False,na_rep="")
    deferred_econ=pd.DataFrame()
    if not decisions.empty:
        deferred=decisions.loc[~decisions.accepted.astype(bool)].copy()
        rows=[]
        for keys,g in deferred.groupby(["task","policy","coverage_mode","requested_coverage"]):
            valid=g.loc[g["regret@10"].notna()]
            rows.append({"task":keys[0],"policy":keys[1],"coverage_mode":keys[2],"requested_coverage":keys[3],
                "deferred_pools":int(g.candidate_pool_id.nunique()),"pools_with_observed_labels":int(len(valid)),
                "mean_observed_usd_opportunity_per_deferred_pool":float(valid["selected_positive_usd@10"].mean()) if len(valid) else np.nan,
                "total_observed_usd_opportunity_in_deferred_lists":float(valid["selected_positive_usd@10"].sum()) if len(valid) else np.nan,
                "mean_regret10_deferred_diagnostic":float(valid["regret@10"].mean()) if len(valid) else np.nan,
                "unknown_candidates_in_full_top10":int(g["missing_in_full_score_top10"].sum()),
                "interpretation":"retrospective observed opportunity left undecided; not economic loss or human-review outcome"})
        deferred_econ=pd.DataFrame(rows)
        deferred_econ.to_csv(results/"T8_deferred_pool_opportunity.csv",index=False,na_rep="")
    matched_summary=pool_summary(decisions.loc[decisions.coverage_mode.eq("matched_test_coverage") &
        decisions.requested_coverage.eq(.8)])
    if not matched.empty:
        random=matched.loc[matched.policy.eq("RANDOM_DEFERRAL_ADAPTIVE_ROUTE") & matched.requested_coverage.eq(.8)].groupby(
            ["task","policy","coverage_mode","requested_coverage"],as_index=False).agg(
                automation_coverage=("automation_coverage","mean"),all_pools=("all_pools","sum"),
                accepted_pools=("accepted_pools","sum"),accepted_pools_with_observed_metrics=("accepted_pools_with_observed_metrics","mean"),
                mean_regret10_accepted=("mean_regret10_accepted","mean"),mean_ndcg10_accepted=("mean_ndcg10_accepted","mean"),
                mean_precision10_accepted=("mean_precision10_accepted","mean"),observed_positive_usd_top10=("mean_observed_positive_usd_top10","mean"))
        if not random.empty: matched_summary=pd.concat([matched_summary,random],ignore_index=True,sort=False)
    matched_summary.to_csv(results/"T5_matched_coverage_summary.csv",index=False,na_rep="")

    task_results=[]
    for task in ["Task_A_new_entry","Task_B_existing_underexploited"]:
        s=static.loc[static.task.eq(task)] if not static.empty else pd.DataFrame()
        b=blend.loc[blend.task.eq(task)] if not blend.empty else pd.DataFrame()
        r=route.loc[route.task.eq(task)&route.policy.eq("ADAPTIVE_ROUTING")] if not route.empty else pd.DataFrame()
        m=matched.loc[matched.task.eq(task)&matched.requested_coverage.eq(.8)&matched.coverage_mode.eq("matched_test_coverage")] if not matched.empty else pd.DataFrame()
        c=coverage.loc[coverage.task.eq(task)&coverage.requested_coverage.eq(.8)&coverage.coverage_mode.eq("development_threshold")] if not coverage.empty else pd.DataFrame()
        q=m.loc[m.policy.eq("ERGM_FULL")] if not m.empty else pd.DataFrame()
        task_results.append({"task":task,"static_expert":"LightGBM" if task=="Task_A_new_entry" else "Gravity_PPML",
            "static_ndcg10":float(s.loc[s.policy.eq("STATIC_TASK_ROUTING"),"ndcg10"].mean()) if not s.empty else np.nan,
            "static_regret10":float(s.loc[s.policy.eq("STATIC_TASK_ROUTING"),"regret10"].mean()) if not s.empty else np.nan,
            "adaptive_ndcg10":float(r.ndcg10.mean()) if not r.empty else np.nan,
            "adaptive_regret10":float(r.realized_regret10.mean()) if not r.empty else np.nan,
            "adaptive_lgb_share":float(freq.loc[freq.task.eq(task)&freq.policy.eq("ADAPTIVE_ROUTING"),"selected_lightgbm_share"].mean()) if not freq.empty else np.nan,
            "blend_ndcg10":float(b.ndcg10.mean()) if not b.empty else np.nan,
            "ergm_matched80_regret":float(q.mean_regret10_accepted.mean()) if not q.empty else np.nan,
            "ergm_matched80_ndcg":float(q.mean_ndcg10_accepted.mean()) if not q.empty else np.nan,
            "ergm_operational80_coverage":float(c.loc[c.policy.eq("ERGM_FULL"),"automation_coverage"].mean()) if not c.empty else np.nan,
            "ergm_operational80_regret":float(c.loc[c.policy.eq("ERGM_FULL"),"mean_regret10_accepted"].mean()) if not c.empty else np.nan})
    task_df=pd.DataFrame(task_results)
    task_df.to_csv(results/"T8_task_summary.csv",index=False,na_rep="")

    method = [
        "# ERGM method and evaluation", "",
        "ERGM is evaluated as a pool-level two-stage system. A task × HS2 × outcome-year pool contains one complete destination candidate set. The first stage selects PPML or LightGBM; the second stage accepts the whole Top-10 recommendation or marks the pool `DEFER_TO_EXPERT`.", "",
        "## Estimands", "",
        "Task A primary candidate eligibility is zero exports in each of the three reporting-verified origin years; the success label is observed exports ≥ USD 50,000 in the outcome year. Task B requires positive historical exports and historical share no larger than the HS2 median based on data through the feature cutoff. The corrected sets are mutually exclusive.", "",
        "For each expert e and pool q, the learning target is retrospective complete-case Top-10 regret: oracle gain among observed-label candidates minus the gain selected by e, bounded below by zero. Task A gain is binary entry success. Task B gain is the audited capped grade: `grade=min(4,floor(log10(1+max(0,increment)/50,000)))`, with DCG gain `2^grade−1`; pool regret uses the difference in undiscounted Top-10 total capped gain. NDCG discounts position by `log2(rank+1)`.", "",
        "The gate uses a fixed Ridge model with alpha 10 to predict both experts' task-specific regret from pre-origin pool summaries, model-score disagreement and HS2 identity. Prior-expert regret features include only outcomes through y−2 for a y decision. The expert with lower predicted regret is selected. The predicted regret is a ranking score, not a calibrated probability.", "",
        "For operational coverage thresholds, risk quantiles are derived from chronological held-out historical pool predictions: a pool in historical year t is scored by a gate trained only on outcomes through t−2. This calibration history is short, especially for the 2022 fold, and is not a formal risk guarantee. Matched-coverage estimates select the requested fraction by test-time risk scores only (no test labels), solely to standardize workloads across methods. They are not deployable frozen thresholds.", "",
        "Human review labels, review cost and post-review decisions do not exist. Deferral is therefore a review-workload flag, not validated human-AI collaboration.", ""]
    (reports/"ERGM_METHOD.md").write_text("\n".join(method))

    # Report adaptive versus fixed routing from actual saved test-pool rows.
    route_summary=[]
    for (task,policy),g in static.groupby(["task","policy"]):
        route_summary.append({"task":task,"policy":policy,"pools":g.candidate_pool_id.nunique(),
            "mean_ndcg10":g.ndcg10.mean(),"mean_precision10":g.precision10.mean(),
            "mean_regret10":g.regret10.mean(),"mean_observed_positive_usd_top10":g.selected_positive_usd_top10.mean()})
    route_summary=pd.DataFrame(route_summary)
    route_summary.to_csv(results/"T1_routing_summary.csv",index=False,na_rep="")
    frequency_table=freq.loc[freq.policy.eq("ADAPTIVE_ROUTING")] if not freq.empty else pd.DataFrame()
    (reports/"ROUTING_RESULTS.md").write_text("\n".join([
        "# Expert routing results", "",
        "The primary fixed rule chooses LightGBM for strict new-entry pools and PPML for existing-market expansion pools, reflecting prior M1 development evidence. This is exploratory, not an untouched protocol. The adaptive ridge gate predicts each model's future pool regret using historical rolling-origin pool outcomes and pre-origin characteristics.", "",
        route_summary.to_markdown(index=False) if not route_summary.empty else "No routing table was produced.", "",
        frequency_table.to_markdown(index=False) if not frequency_table.empty else "No route frequency table was produced.", "",
        "`T3_expert_selection_frequency.csv` reports the routed expert share. The rank-average blend is formed from within-pool percentiles, avoiding direct averaging of raw scores on different scales. Neither adaptive routing nor blending is causal evidence.", "",
        "The gate's hindsight agreement share and the association between model disagreement and subsequent regret are reported in `T4_disagreement_diagnostics.csv`. Hindsight agreement is descriptive and not available at the decision origin.", "",
        f"Historical OOF model fits completed: {len(model_diag):,} model-folds across {model_diag.year.nunique() if not model_diag.empty else 0} outcome years. See `results/historical_oof_model_diagnostics.csv`.", ""]))

    # Selective automation / coverage-risk.
    primary=matched_summary
    matched_table=primary[[c for c in ["task","policy","automation_coverage","accepted_pools","all_pools",
        "accepted_pools_with_observed_metrics","mean_regret10_accepted","mean_ndcg10_accepted",
        "mean_precision10_accepted","observed_positive_usd_top10","mean_observed_positive_usd_top10"] if c in primary.columns]]
    (reports/"SELECTIVE_AUTOMATION.md").write_text("\n".join([
        "# Selective automation and review workload", "",
        "The primary comparison standardizes selective methods to a matched 80% automation fraction using model risk scores and fixed tie rules without test labels. A random-acceptance reference retains adaptive rankings and averages 500 random pool selections at the same workload. The development-threshold results are reported separately in `T5_coverage_risk.csv`; actual realized coverage can differ from the requested value.", "",
        matched_table.to_markdown(index=False) if not matched_table.empty else "No matched-coverage table was produced.", "",
        "Risk is measured among accepted pools with observed outcome labels. Deferred-pool regret is shown only as a hindsight diagnostic. A lower selective risk can result from deferring more; the matched-coverage table is the main comparison. No human-review result exists, so the system does not produce a deployed overall portfolio or human-AI accuracy estimate.", "",
        "The most frequently deferred product chapters under the frozen development-threshold 80% request are listed in `T5_deferred_hs2_frequency.csv`; actual coverage varies by year. Review dossier market lists preserve each model's ranking order and include observed historical candidate context.", "",
        disagreement_diag.to_markdown(index=False) if not disagreement_diag.empty else "Disagreement diagnostics unavailable.", "",
        "Risk calibration is descriptive. Reported risk scores are not formal probabilities or high-confidence guarantees. The number and composition of deferred pools appear in `T5_deferred_pool_characteristics.csv` and review dossiers.", ""]))

    # Economic outcomes.
    eco=_read("T8_economic_opportunity.csv")
    (reports/"ECONOMIC_VALUE.md").write_text("\n".join([
        "# Economic opportunity and retrospective regret", "",
        "The recorded USD amounts are trade values observed in BACI among known-outcome candidates selected by a Top-K ranking. Task A reports outcome-year export value for the selected strict-zero relationships; Task B reports positive incremental exports relative to the historical reference. Values are not profits, causal uplift, or returns to an export-promotion intervention.", "",
        "Complete-case ranking metrics retain model-score order among observed-label candidates and calculate Top-K on that observed subset; unknown labels in the full predicted Top-10 are counted separately. These are conditional-on-observation metrics, not realized utilities for every member of the originally selected full list. For a deferred pool, accepted-list value remains undecided; the retrospective value of the alternative list is not a human-review outcome.", "",
        eco.to_markdown(index=False) if not eco.empty else "No economic summary table was produced.", "",
        deferred_econ.loc[deferred_econ.coverage_mode.eq("matched_test_coverage") & deferred_econ.requested_coverage.eq(.8)].to_markdown(index=False) if not deferred_econ.empty else "No deferred-pool opportunity table was produced.", "",
        "The deferred-list USD amounts in the table are retrospective trade volume that would remain undecided pending review; they are not lost value, realized post-review value, or causal benefits. No observed review costs exist. `T8_review_cost_sensitivity.csv` contains purely hypothetical cost scenarios and must not be interpreted as empirical ROI.", ""]))

    # Explanations.
    topcoef=pd.DataFrame()
    if (results/"T3_routing_feature_effects.csv").exists():
        topcoef=_read("T3_routing_feature_effects.csv")
    if not topcoef.empty:
        topcoef=topcoef.assign(abs_coef=topcoef.coefficient_lightgbm_minus_ppml_regret.abs())
        leaders=topcoef.sort_values("abs_coef",ascending=False).groupby(["task","feature"],as_index=False).first()
        leaders=leaders.sort_values(["task","abs_coef"],ascending=[True,False]).groupby("task").head(10)
    else: leaders=topcoef
    fid=fidelity
    fidelity_summary=(f"Raw gate score-difference reconstruction passed for {int(fid.raw_gate_score_reconstructed.sum()):,}/{len(fid):,} pools; the selected expert was reproduced for {int(fid.route_decision_reproduced.sum()):,}/{len(fid):,} pools." if not fid.empty else "Fidelity table unavailable.")
    (reports/"EXPLAINABILITY.md").write_text("\n".join([
        "# Explainable routing and deferral", "",
        "The fitted Ridge gate is additive in standardized numeric and HS2 one-hot features. For each pool, `T6_routing_explanations.csv` lists the selected expert, alternative, both estimated regret scores and leading raw score-difference contributions. Nonnegative clipping can make both estimated regrets tie at zero; those ties are resolved in favor of PPML and flagged. Counterfactual rows locate a one-feature change that ties the untruncated Ridge outputs; clipping can prevent that tie from changing the final route, so this is a model-score diagnostic rather than a verified decision-switch counterfactual.", "",
        leaders.drop(columns=["abs_coef"],errors="ignore").to_markdown(index=False) if not leaders.empty else "No feature effect table was produced.", "",
        fidelity_summary,
        "The deferred review dossier gives both expert Top-10 lists, their overlap, differing destinations, predicted regret score and rule-based deferral reason. Explanations do not use human judgments and are not causal. The method does not claim a new XAI algorithm.", ""]))

    # Missingness and robustness.
    (reports/"MISSINGNESS_AND_ROBUSTNESS.md").write_text("\n".join([
        "# Missingness and robustness", "",
        "Unknown outcome labels remain NA. There are 556 unknown labels in the corrected strict-entry population and none in the existing-expansion population (the other three unknowns are outside both task populations). No unknown is recoded to zero. Complete-case performance is conditional on outcome observability.", "",
        missing.to_markdown(index=False) if not missing.empty else "Missingness table unavailable.", "",
        "Binary Task A regret bounds use exact enumeration of feasible success counts among missing selected and unselected candidates. The same joint completion is used for the selected portfolio and hindsight oracle, so bounds are sharp under the binary support and fixed Top-10 policy. This count-based procedure covers large missing pools without enumerating every label vector. Dollar outcomes remain unbounded from available evidence; no finite USD bounds are asserted.", "",
        f"Exact binary-bound rows: {int(missbounds.entry_bound_status.astype(str).str.contains('exact',case=False,na=False).sum()) if 'entry_bound_status' in missbounds else 0:,}; any non-estimated rows remain explicitly marked. Missingness and decision uncertainty are separate.", "",
        "The dataset has 2022–2024 outer outcomes, which prior M1 work has already inspected. WDI/WGI are revised historical vintages. No fresh independent holdout is available in the local BACI V202601 panel, whose observed outcomes end in 2024. All policy evidence is exploratory retrospective evidence.", ""]))

    # Inference and contribution.
    (reports/"STATISTICAL_VALIDATION.md").write_text("\n".join([
        "# Statistical validation", "",
        "The main uncertainty comparison is the difference in mean observed-label Top-10 regret among policies' accepted pools at an equal 80% realized automation fraction. Intervals resample HS2 product blocks, retaining each selected block's repeated years. Sign-flip tests act on paired HS2 mean differences; Holm adjustment is applied to the reported comparison family.", "",
        inference.to_markdown(index=False) if not inference.empty else "No inferential contrast was estimable; descriptive comparisons are reported instead.", "",
        seed_stability.to_markdown(index=False), "",
        "The estimand is conditional selective risk and policies can accept different product-year pools; it is not a paired same-case treatment effect. HS2 block resampling may not capture cross-product destination dependence, and three previously inspected evaluation years cannot establish robustness across future cycles. Model seeds are not treated as economic units.", "",
        "The expected-regret scores are not calibrated failure probabilities. Any performance differences must be interpreted alongside accepted coverage, review workload and missing-label bounds.", ""]))

    # Contribution evaluation based on computed tables, not narrative expectations.
    full_eval=matched.loc[matched.policy.eq("ERGM_FULL")&matched.coverage_mode.eq("matched_test_coverage")&matched.requested_coverage.eq(.8)] if not matched.empty else pd.DataFrame()
    simple=matched.loc[matched.policy.isin(["STATIC_PLUS_RISK","DISAGREEMENT_ONLY","STATIC_CONFIDENCE","LIGHTGBM_CONFIDENCE","PPML_CONFIDENCE"])
        &matched.coverage_mode.eq("matched_test_coverage")&matched.requested_coverage.eq(.8)] if not matched.empty else pd.DataFrame()
    contribution=("The evidence supports an exploratory empirical application of pool-level routing and selective automation. It does not support a new algorithm, a formal reliability guarantee, or validated human-AI collaboration." if not full_eval.empty else "The empirical system did not complete sufficiently to assess contribution.")
    (reports/"CONTRIBUTION_ASSESSMENT.md").write_text("\n".join([
        "# Contribution assessment", "",
        "## Findings", "",
        f"- Adaptive routing vs fixed task routing: see `T3_adaptive_routing.csv` and `T1_static_task_routing.csv`; results are actual historical Top-10 regret and NDCG on common corrected task pools.",
        f"- Reliability score: see `T4_risk_calibration.csv` and `T5_high_regret_detection.csv`. It is a predicted regret score; no probability calibration guarantee is claimed.",
        f"- Matched-coverage deferral: ERGM has {len(full_eval):,} task-year matched 80% summary rows; simple gates have {len(simple):,}. See paired estimates in `inference.csv`.",
        "- Expert review value is unidentified because no human recommendations or post-review outcomes are available.",
        "- The selected method components—expert routing, selective prediction, risk–coverage, confidence thresholding and abstention—are established in prior work; see `LITERATURE_AND_NOVELTY_AUDIT.md`.", "",
        "## Contribution classification", "",
        contribution,
        "The specific domain adaptation is a pool-level gate trained to predict Top-10 regret between gravity and ML experts. This is an empirical decision-unit adaptation, not a new general-purpose learning-to-defer method. The results use previously inspected 2022–2024 years and cannot support independent validation. Any potentially useful evidence must be confirmed on a genuinely new trade vintage and with expert review outcomes if human-AI claims are intended.", ""]))

    # Manuscript-oriented compact results section.
    (reports/"PAPER_READY_RESULTS.md").write_text("\n".join([
        "# Paper-ready methods and results (exploratory version)", "",
        "## Data and evaluation", "",
        "The extension uses the existing BACI HS12 V202601 HS2 panel and saved out-of-time baseline predictions. The primary evaluation uses the audited two-year information lag and decision pools formed by product chapter and outcome year. It covers 2022–2024, an interval already examined in prior M1 development. Revised WDI/WGI values make the backtest retrospective rather than archival point-in-time validation.", "",
        "Entry pools require verified zero Vietnam exports in each of the three origin years; the success threshold is USD 50,000. Existing expansion pools require positive historical exports and a share at or below the HS2 median available at the origin. This strict entry definition resolves the previous 6,511-key overlap created when ‘inactive’ meant only sub-threshold trade.", "",
        "## Routing and reliability", "",
        "Gravity–PPML and LightGBM scores are reused for the outer years. Historical gate targets are constructed from new rolling-origin OOF predictions for 2018–2021, with model outcomes ending no later than y−2. A separate regularized linear model predicts each expert's observed-label Top-10 regret for every task. The lower predicted-regret expert is selected. A second threshold compares that pool-level regret score with a development quantile; rejected recommendations are marked for review and are not assigned a hindsight-oracle replacement.", "",
        "## Results", "",
        task_df.to_markdown(index=False) if not task_df.empty else "Task summary unavailable.", "",
        "At matched coverage, selective-risk results are conditional on pools with observed outcomes; deferred pools remain undecided. The 559 unknown outcomes in the broad evaluation universe remain missing, including 556 corrected strict-entry candidates. No test-year outcome is used to construct candidate eligibility or fit the outer gate.", "",
        "## Interpretation and limitations", "",
        "The empirical system is a two-stage application of existing expert-routing and selective-prediction ideas. It does not establish human-AI collaboration, causal export gains, a calibrated decision-failure probability, or independent temporal validity. Evidence is exploratory because the test years were previously inspected, the number of product-year units is limited, and revised macroeconomic vintages are not point-in-time. The extension's principal contribution is the corrected evaluation of how a pool-level regret gate behaves against static and confidence/disagreement baselines in export-market ranking.", ""]))

    # Hypothetical review cost scenarios are never presented as observed costs.
    cost_sensitivity=[]
    for task in ["Task_A_new_entry","Task_B_existing_underexploited"]:
        for cov in [.5,.6,.7,.8,.9,1.0]:
            deferrals=1-cov
            for assumed_cost in [0,100,500,1000]:
                cost_sensitivity.append({"task":task,"assumed_review_cost_usd_per_pool":assumed_cost,
                    "assumed_coverage":cov,"hypothetical_review_cost_per_100_pools":100*deferrals*assumed_cost,
                    "status":"hypothetical sensitivity only; no observed review cost or ROI"})
    pd.DataFrame(cost_sensitivity).to_csv(results/"T8_review_cost_sensitivity.csv",index=False)

    executive = [
        "# Tóm tắt điều hành ERGM", "",
        "## Hệ thống làm gì", "",
        "ERGM chọn giữa Gravity–PPML và LightGBM cho từng nhóm sản phẩm HS2 × năm. Sau đó hệ thống xếp hạng toàn bộ danh sách thị trường Top-10 và có thể chuyển cả nhóm sang trạng thái cần chuyên gia xem xét nếu điểm rủi ro dự báo cao.", "",
        "## Dữ liệu và sửa định nghĩa", "",
        "Phân tích dùng dữ liệu BACI, panel HS2 và dự báo đã lưu của M1. Trong định nghĩa cũ, 6.511 khóa cùng lúc thỏa điều kiện ‘chưa đạt ngưỡng xuất khẩu USD 50.000’ và ‘đã có xuất khẩu dương’. Mở rộng lần này định nghĩa thị trường mới là không có xuất khẩu được xác minh trong cả ba năm trước; vì vậy Task A và Task B không còn giao nhau. Có 29.737 quan sát Task A và 12.962 quan sát Task B; 556 nhãn Task A vẫn không quan sát được và không được thay bằng 0.", "",
        "## Kết quả thực nghiệm", "",
        task_df.to_markdown(index=False) if not task_df.empty else "Bảng kết quả chưa có.", "",
        "Các bảng coverage–risk và so sánh cùng mức coverage 80% nằm trong `results/T5_coverage_risk.csv` và `results/T5_matched_coverage_comparison.csv`. Giá trị USD là cơ hội thương mại quan sát được trong dữ liệu lịch sử, không phải lợi nhuận hay tác động do khuyến nghị.", "",
        "## Giới hạn", "",
        "Không có quyết định hoặc kết quả sau thẩm định của chuyên gia con người. Vì thế đây là tự động hóa có chọn lọc và phân luồng hồ sơ xem xét, chưa phải nghiên cứu hợp tác người–AI đã được xác nhận. Các năm 2022–2024 đã được dùng trong phát triển M1; kết quả chỉ mang tính hồi cứu thăm dò, không có holdout độc lập. Không có thuật toán mới hoặc bảo đảm xác suất an toàn được tuyên bố.", ""]
    (reports/"EXECUTIVE_SUMMARY_VI.md").write_text("\n".join(executive))

    # Workbook contains computed tables only; no charts.
    sheets={
        "Data Audit":sample,"Static Routing":_read("T1_static_task_routing.csv"),
        "Static Blending":blend,"Adaptive Routing":route,"Expert Frequency":freq,
        "Risk Calibration":calibration,"Coverage Risk":coverage,"Matched Coverage":matched,
        "Deferred Pools":_read("T5_deferred_pool_characteristics.csv"),
        "Deferred HS2":deferred_hs2,
        "Matched Summary":matched_summary,
        "Random Deferral":_read("T5_random_deferral.csv"),
        "Disagreement":disagreement_diag,
        "Seed Stability":seed_stability,
        "Routing Explanations":_read("T6_routing_explanations.csv"),
        "Deferral Dossiers":dossiers,"Explanation Fidelity":fidelity,
        "Economic Value":eco,"Regret":_read("T8_regret.csv"),"Ablation":ablation,
        "Deferred Opportunity":deferred_econ,
        "Missing Outcomes":missbounds,"Missingness":missing,"Inference":inference,
        "Temporal Validation":temporal,"Compute Cost":cost}
    workbook=results/"ERGM_FULL_RESEARCH_RESULTS.xlsx"
    with pd.ExcelWriter(workbook,engine="openpyxl") as writer:
        for sheet,df in sheets.items():
            (df if not df.empty else pd.DataFrame({"status":["not estimable / no rows"]})).to_excel(writer,sheet_name=sheet[:31],index=False)

    complete_files=["T1_static_task_routing.csv","T2_static_blending.csv","T3_adaptive_routing.csv",
        "T3_expert_selection_frequency.csv","T3_routing_feature_effects.csv","T4_reliability_scores.csv",
        "T4_risk_calibration.csv","T5_coverage_risk.csv","T5_matched_coverage_comparison.csv",
        "T6_routing_explanations.csv","T6_counterfactual_switching.csv","T7_review_dossiers.csv",
        "T7_deferral_reason_codes.csv","T7_explanation_fidelity.csv","T8_economic_opportunity.csv",
        "T8_export_value_usd.csv","T8_regret.csv","T9_ablation.csv","T10_missing_outcome_sensitivity.csv",
        "T10_missingness_by_task.csv","inference.csv","holm_adjusted_tests.csv","temporal_validation.csv",
        "computational_cost.csv","seed_stability.csv","T4_disagreement_diagnostics.csv",
        "T5_matched_coverage_summary.csv","T5_deferred_hs2_frequency.csv",
        "T5_random_deferral.csv","T5_pool_acceptance_decisions.csv",
        "T8_deferred_pool_opportunity.csv","T8_review_cost_sensitivity.csv"]
    missing_required=[x for x in complete_files if not (results/x).exists()]
    evidence="EXPLORATORY_ONLY"
    eng="COMPLETE_REDUCED_SPECIFICATION" if not missing_required else "PARTIAL_RESULTS"
    contrib="EMPIRICAL_CONTRIBUTION_ONLY" if not route.empty and not matched.empty else "NO_DEFENSIBLE_NEW_CONTRIBUTION"
    (reports/"FINAL_STATUS.md").write_text("\n".join([
        f"# Final status: {eng}", "",
        f"Research evidence status: **{evidence}**. The 2022–2024 outcomes were previously inspected. The local BACI V202601 outcomes end in 2024; no fresh independent holdout was found.", "",
        f"Contribution classification: **{contrib}**. Expert routing and selective automation are established methods. This work is a pool-level empirical application; human-review performance is not observed.", "",
        f"Missing required outputs: {', '.join(missing_required) if missing_required else 'none'}.", "",
        f"Task A strict-zero candidate rows: {int(sample.strict_zero_entry.sum()) if 'strict_zero_entry' in sample else 'see candidate_manifest.csv'}; Task B rows: {int(sample.existing_underexploited.sum()) if 'existing_underexploited' in sample else 'see candidate_manifest.csv'}. Unknown labels are retained as NA.", "",
        "Historical LightGBM OOF predictions use seed 11 only; a multi-seed stability experiment was not run. The Ridge gate is deterministic. This is one reason for the reduced-specification status.", "",
        f"Excel workbook: `results/{workbook.name}`.", "",
        "Reproduction: `.venv_uv/bin/python -m experiments.explainable_reliability_gate_v1.run_ergm --stage full`, followed by `.venv_uv/bin/python -m pytest experiments/explainable_reliability_gate_v1/tests -q`.", "",
        "There are no figures. Human-AI review efficacy, true prospective reliability, future-cycle performance, and causal economic returns remain unverified.", ""]))
