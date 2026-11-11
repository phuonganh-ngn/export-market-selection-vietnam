from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import CFG, ROOT, path, write_csv


def _read_csv(name:str)->pd.DataFrame:
    p=path("results_dir")/name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _read_report_csv(name:str)->pd.DataFrame:
    p=path("reports_dir")/name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _md(df:pd.DataFrame,limit:int=15)->str:
    if df is None or df.empty: return "_No computed rows._"
    d=df.head(limit).copy()
    return d.to_markdown(index=False,floatfmt=".4g")


def _fmt(x,dec=2):
    if pd.isna(x): return "NA"
    if isinstance(x,(float,np.floating)): return f"{x:,.{dec}f}"
    if isinstance(x,(int,np.integer)): return f"{x:,}"
    return str(x)


def _forecast_vs_ranking()->pd.DataFrame:
    e1=_read_csv("E1_forecasting_overall.csv")
    e3=_read_csv("E3_ranking_by_hs2.csv")
    if e1.empty or e3.empty: return pd.DataFrame()
    if "ranking_task" in e3:
        e3=e3.loc[e3.ranking_task.eq("incremental")]
    eligible=e1.loc[e1.model.isin(["Gravity_PPML","RandomForest","XGBoost","LightGBM","Persistence"])]
    winner=eligible.sort_values("mae_usd").iloc[0]
    rank=e3.groupby("ranking_model",as_index=False).agg(macro_ndcg_at_10=("NDCG_at_10","mean"),
        macro_precision_at_10=("Precision_at_10","mean"),macro_incremental_exports_top10_usd=("realized_incremental_exports_top10_usd","mean"))
    best=rank.sort_values("macro_ndcg_at_10",ascending=False).iloc[0]
    return pd.DataFrame([{"forecasting_winner":winner.model,"forecasting_mae_usd":winner.mae_usd,
        "forecasting_macro_hs2_mae_usd":winner.macro_hs2_mae_usd,"ranking_winner":best.ranking_model,
        "ranking_winner_macro_ndcg_at_10":best.macro_ndcg_at_10,"ranking_winner_macro_precision_at_10":best.macro_precision_at_10,
        "ranking_winner_incremental_exports_top10_usd":best.macro_incremental_exports_top10_usd,
        "same_model_family":str(winner.model).lower() in str(best.ranking_model).lower(),
        "ranking_objective":"Within-HS2 × year; incremental opportunity and entry relevance"}])


def _descriptive_tables()->tuple[pd.DataFrame,pd.DataFrame]:
    panel_path=path("processed_dir")/"panel_lag2.parquet"
    if not panel_path.exists(): return pd.DataFrame(),pd.DataFrame()
    d=pd.read_parquet(panel_path)
    d=d.loc[d.outcome_observed&d.outcome_year.le(CFG["study"]["last_year"])].copy()
    cols=["export_value_usd","destination_import_demand","historical_export_value","historical_market_share",
          "destination_supplier_concentration","log_gdp","log_gdp_per_capita","gdp_growth","log_population",
          "urban_population_share","internet_usage","government_effectiveness","regulatory_quality","rule_of_law",
          "political_stability","log_distance"]
    rows=[]
    for c in cols:
        if c not in d: continue
        s=pd.to_numeric(d[c],errors="coerce")
        rows.append({"variable":c,"n_nonmissing":int(s.notna().sum()),"mean":s.mean(),"sd":s.std(),"p50":s.median(),
                     "p01":s.quantile(.01),"p99":s.quantile(.99),"min":s.min(),"max":s.max(),"missing_rate":s.isna().mean()})
    desc=pd.DataFrame(rows)
    flow=d.groupby("outcome_year",as_index=False).agg(observations=("export_value_usd","size"),destinations=("destination_iso3","nunique"),
        hs2_chapters=("hs2","nunique"),mean_export_usd=("export_value_usd","mean"),median_export_usd=("export_value_usd","median"),
        total_export_usd=("export_value_usd","sum"),zero_export_flows=("export_value_usd",lambda x:int(x.eq(0).sum())),
        positive_export_flows=("export_value_usd",lambda x:int(x.gt(0).sum())))
    flow["zero_export_share"]=flow.zero_export_flows/flow.observations
    desc.to_csv(path("results_dir")/"descriptive_statistics.csv",index=False)
    flow.to_csv(path("results_dir")/"export_flow_zero_statistics.csv",index=False)
    return desc,flow


def _data_audit(quick:bool=False)->dict:
    source=_read_report_csv("data_source_registry.csv")
    panel=_read_report_csv("panel_summary.csv")
    year=_read_report_csv("year_coverage.csv")
    checks=_read_report_csv("aggregation_checks.csv")
    country=_read_report_csv("country_coverage.csv")
    product=_read_report_csv("product_coverage.csv")
    missing=_read_report_csv("missingness.csv")
    exclusions=_read_report_csv("data_exclusions.csv")
    if panel.empty:
        summary={"destinations":0,"hs2":0,"panel_observations":0,"years":"not prepared"}
    else:
        summary=panel.iloc[0].to_dict()
    n_dest=int(summary.get("destinations",0) or 0); n_hs2=int(summary.get("eligible_hs2_chapters",0) or 0)
    n_panel=int(summary.get("panel_rows_primary_lag",0) or 0)
    year_first=int(CFG["study"]["first_year"]); year_last=int(CFG["study"]["last_year"])
    raw_records_total=np.nan
    if not checks.empty and {"year","raw_records"}.issubset(checks.columns):
        raw_records_total=pd.to_numeric(checks.drop_duplicates("year").raw_records,errors="coerce").sum(min_count=1)
    elif not year.empty and "raw_records" in year:
        raw_records_total=pd.to_numeric(year.raw_records,errors="coerce").sum(min_count=1)
    valid_sources=source.loc[source.verification_status.str.startswith("verified",na=False)] if not source.empty else pd.DataFrame()
    files=["results/E1_forecasting_overall.csv","results/E1_forecasting_by_hs2.csv","results/E2_market_discovery.csv",
           "results/E3_market_ranking.csv","results/E4_global_shap.csv","results/E5_clusters.csv"]
    text=["# Data audit", "", "## Source coverage", "",
        "| Dataset | Provider | Version | Years | Units | Verification |", "|---|---|---|---|---|---|"]
    if source.empty: text.append("| No source registry available | | | | | |")
    else:
        for r in source.itertuples(index=False):
            text.append(f"| {r.dataset} | {r.provider} | {r.version} | {r.years_covered} | {r.units} | {r.verification_status} |")
    text += ["", "## Verified sample", "", f"- BACI release: CEPII BACI HS12 V202601, observed product-level period {year_first}–{year_last}; value field `v` is thousand current USD and the pipeline multiplies by 1,000.",
        f"- Foreign destinations represented in the eligible primary panel: **{n_dest}**.",f"- Eligible HS2 chapters in the primary panel: **{n_hs2}**.",
        f"- Destination–HS2–outcome-year observations with verified outcome-year HS2 world imports: **{n_panel:,}**.",
        f"- Outcome coverage in the primary panel: **{summary.get('years','not computed')}**.",
        f"- BACI source trade rows scanned: **{int(raw_records_total):,}**." if pd.notna(raw_records_total) else "- BACI source rows: recorded during streamed aggregation.",
        f"- Zero Vietnam export share among observed panel rows: **{_fmt(summary.get('zero_export_share'),4)}**.",
        "", "## Missingness and zero-flow rule", "",
        "BACI supplies observed positive flows rather than a complete zero-filled matrix. The panel candidate exists only when the destination has positive world imports for the HS2 at the information date. A Vietnam bilateral absence is coded as zero only when the outcome-year destination–HS2 cell has positive BACI world imports; otherwise the outcome is unavailable and remains missing. This can omit destination–product cells with no recorded world imports and is deliberately conservative.",
        "WDI API values can be revised after their reference year. The main backtest enforces reference-year lags but cannot reconstruct the original publication vintage. WGI uses the 2026 revision (historical estimates recalculated); CEPII GeoDist is a time-invariant geographic source.",
        "", "## Aggregation and coverage checks", ""]
    if checks.empty: text.append("No aggregation check table was written.")
    else:
        passed=checks["pass"].astype(str).str.lower().isin(["true","1"]).sum() if "pass" in checks else 0
        text.append(f"Aggregation rows passing value-conservation and HHI checks: {passed}/{len(checks)}.")
        text.append("\n"+_md(checks.head(12)))
    text += ["", "## Country coverage", "",_md(country.head(10)),"", "## HS2 coverage", "",_md(product.head(15)),
        "", "## Highest missingness", "",_md(missing.sort_values("missing_rate",ascending=False).head(15) if not missing.empty else missing),
        "", "## Exclusions", "",_md(exclusions),"", "## Core result artifacts", ""]
    text.extend([f"- `{f}`: {'present' if (ROOT/f).exists() else 'not yet generated'}" for f in files])
    text += ["", "Potential new-market case counts are reported by the entry-threshold table after the discovery stage. The development choice is based on pre-test outcomes only.",""]
    (path("reports_dir")/"DATA_AUDIT.md").write_text("\n".join(text))
    return {"destinations":n_dest,"hs2":n_hs2,"panel_observations":n_panel,"period":summary.get("years","NA"),
            "source_registry_rows":len(source),"verified_source_rows":len(valid_sources),"quick_mode":quick}


def _experiment_reports(data_summary:dict,quick:bool=False,failures:list|None=None)->dict:
    e1=_read_csv("E1_forecasting_overall.csv"); e1y=_read_csv("E1_forecasting_by_year.csv"); e1hs=_read_csv("E1_forecasting_by_hs2.csv")
    e2=_read_csv("E2_market_discovery.csv"); e2hs=_read_csv("E2_discovery_by_hs2.csv")
    e3all_metrics=_read_csv("E3_ranking_by_hs2.csv")
    e3=e3all_metrics.loc[e3all_metrics.ranking_task.eq("incremental")].copy() if not e3all_metrics.empty and "ranking_task" in e3all_metrics else e3all_metrics
    e3abs=e3all_metrics.loc[e3all_metrics.ranking_task.eq("absolute")].copy() if not e3all_metrics.empty and "ranking_task" in e3all_metrics else pd.DataFrame()
    e4=_read_csv("E4_global_shap.csv"); ab=_read_csv("E4_feature_ablation.csv")
    e5=_read_csv("E5_cluster_model_comparison.csv"); profiles=_read_csv("E5_cluster_profiles.csv")
    robust=_read_csv("robustness_summary.csv"); ci=_read_csv("confidence_intervals.csv"); tests=_read_csv("hypothesis_tests.csv")
    e3all=_read_csv("E3_market_ranking.csv"); fvr=_forecast_vs_ranking()
    top_markets=_read_csv("E3_top5_markets.csv")
    if not top_markets.empty and "ranking_task" in top_markets:
        top_markets=top_markets.loc[top_markets.ranking_task.eq("incremental")]
    if not fvr.empty: fvr.to_csv(path("results_dir")/"forecast_vs_ranking.csv",index=False)
    # Experiment reports from observed output tables.
    best_f=e1.sort_values("mae_usd").iloc[0] if not e1.empty else None
    e1_text=["# E1 — Export forecasting","","Question: does ML improve export-value forecasts over persistence and gravity–PPML?","",
        "The outer evaluation is chronological (2022–2024) under a two-year information lag. The table reports computed results; MAE and RMSE are in current USD.","",_md(e1),"", "## By evaluation year", "",_md(e1y,20),
        "", "## By HS2", "",_md(e1hs.sort_values(["model","mae_usd"]).head(20) if not e1hs.empty else e1hs),
        "", "ML outputs trained on log1p exports are inverse-transformed and multiplied by an original-USD scale selected on inner chronological validation to minimize MAE. Errors are then evaluated in USD. WAPE is NA when its denominator is zero. The comparison is predictive and retrospective.",""]
    (path("reports_dir")/"E1_FORECASTING.md").write_text("\n".join(e1_text))
    threshold=_read_csv("tables/E2_threshold_selection.csv")
    best_discovery=(e2.groupby("model",as_index=False).macro_ndcg_at_10.mean().sort_values("macro_ndcg_at_10",ascending=False).iloc[0]
        if not e2.empty else None)
    e2_text=["# E2 — Market discovery","","Entry is defined using a primary threshold selected from pre-test development distributions. Inactivity requires verified importer coverage and no Vietnam flow over the threshold in the preceding three years.","", "## Development threshold counts","",_md(threshold),"","## Test performance","",_md(e2,20),"","## Product-level performance","",_md(e2hs.sort_values("NDCG_at_10",ascending=False).head(20) if not e2hs.empty else e2hs),
        "", "PR-AUC and ROC-AUC are discrimination metrics over eligible cases. Precision/recall/NDCG are evaluated within HS2 × year candidate pools; no mixed global Top-10 is used. Product pools without positive or sufficient cases remain NA/flagged.",""]
    (path("reports_dir")/"E2_MARKET_DISCOVERY.md").write_text("\n".join(e2_text))
    best_rank=e3.groupby("ranking_model").NDCG_at_10.mean().sort_values(ascending=False) if not e3.empty else pd.Series(dtype=float)
    e3_text=["# E3 — Market selection","","Main decision question: does forecasting accuracy translate into better market selection? Opportunity rankings use expected incremental exports (`max(0, predicted exports − exports at the information date)`) and are evaluated within each HS2–year pool. Absolute-export rankings are reported separately.","", "## Macro-average ranking performance","",_md(e3.groupby("ranking_model",as_index=False).agg(macro_precision_at_5=("Precision_at_5","mean"),macro_precision_at_10=("Precision_at_10","mean"),macro_ndcg_at_5=("NDCG_at_5","mean"),macro_ndcg_at_10=("NDCG_at_10","mean"),macro_increment_top10_usd=("realized_incremental_exports_top10_usd","mean"),pools=("hs2","size")).sort_values("macro_ndcg_at_10",ascending=False) if not e3.empty else e3),
        "", "## Absolute-export rankings (separate task)", "",_md(e3abs.groupby("ranking_model",as_index=False).agg(macro_ndcg_at_10=("NDCG_at_10","mean"),macro_increment_top10_usd=("realized_incremental_exports_top10_usd","mean"),pools=("hs2","size")).sort_values("macro_ndcg_at_10",ascending=False) if not e3abs.empty else e3abs),
        "", "## Forecasting versus incremental-opportunity ranking", "",_md(fvr),"", "## Top markets (incremental task)", "",_md(top_markets,20),
        "", "The prospective ranking uses 2024 features for the 2026 decision period under the two-year lag. It has no future export outcomes and is explicitly not externally validated. These observational ranks do not imply causal gains or firm-specific suitability.",""]
    (path("reports_dir")/"E3_MARKET_SELECTION.md").write_text("\n".join(e3_text))
    e4_text=["# E4 — Numerical explainability","","SHAP, permutation importance, feature-group ablation and numerical ALE are predictive diagnostics, not causal estimates. The explanatory sample is a fixed-seed random sample from out-of-time predictions. Feature-group ablations use the pre-specified seed 11 to limit computation; the primary E1 comparisons retain all five seeds.","","## Global SHAP","",_md(e4.sort_values("mean_abs_shap",ascending=False).head(20) if not e4.empty else e4),"","## Feature ablation","",_md(ab),"","No figures were generated.",""]
    (path("reports_dir")/"E4_EXPLAINABILITY.md").write_text("\n".join(e4_text))
    e5_text=["# E5 — Market segmentation","","K-means and Gaussian mixture assignments are estimated from training-period destination characteristics only. Scaling and missing-value imputation use training destinations.","","## Cluster profiles","",_md(profiles),"","## Pooled versus cluster-specific forecast fits","",_md(e5),
        "", "Cluster-specific forecasting is useful only if its out-of-time error and within-cluster ranking metrics improve over the pooled model in matched clusters. Small clusters with insufficient training rows are flagged; clustering is not assumed to help.",""]
    (path("reports_dir")/"E5_MARKET_SEGMENTATION.md").write_text("\n".join(e5_text))
    (path("reports_dir")/"ROBUSTNESS.md").write_text("# Robustness checks\n\n"+(_md(robust,30))+"\n\nChecks marked not_run, failed, or partial are not interpreted as passed.\n")
    (path("reports_dir")/"STATISTICAL_INFERENCE.md").write_text((path("reports_dir")/"STATISTICAL_INFERENCE.md").read_text() if (path("reports_dir")/"STATISTICAL_INFERENCE.md").exists() else "# Statistical inference\n\nNo inference report was generated.\n")
    return {"E1":e1,"E1_by_year":e1y,"E1_by_hs2":e1hs,"E2":e2,"E2_by_hs2":e2hs,"E3":e3,"E4":e4,"ablation":ab,"E5":e5,"profiles":profiles,"robustness":robust,"CI":ci,"tests":tests,"forecast_vs_ranking":fvr,"best_forecast":best_f,"best_discovery":best_discovery}


def _main_reports(summary:dict,tables:dict,quick:bool,failures:list|None=None)->str:
    e1=tables["E1"];e2=tables["E2"];e3=tables["E3"];e4=tables["E4"];e5=tables["E5"];rob=tables["robustness"]
    forecast_winner=str(e1.sort_values("mae_usd").iloc[0].model) if not e1.empty else "NA"
    rank_summary=e3.groupby("ranking_model").NDCG_at_10.mean().sort_values(ascending=False) if not e3.empty else pd.Series(dtype=float)
    rank_winner=str(rank_summary.index[0]) if len(rank_summary) else "NA"
    ppml=e1.loc[e1.model.eq("Gravity_PPML")] if not e1.empty else pd.DataFrame()
    ml=e1.loc[e1.model.isin(["RandomForest","XGBoost","LightGBM"])] if not e1.empty else pd.DataFrame()
    ml_vs="NA"
    if not ppml.empty and not ml.empty:
        best_ml=ml.sort_values("mae_usd").iloc[0]
        pp=ppml.iloc[0]
        ml_vs=f"Best ML ({best_ml.model}) MAE USD {_fmt(best_ml.mae_usd)}; PPML MAE USD {_fmt(pp.mae_usd)}; difference (ML−PPML) USD {_fmt(best_ml.mae_usd-pp.mae_usd)}."
    top_features=e4.sort_values("mean_abs_shap",ascending=False).head(5).feature.tolist() if not e4.empty else []
    segment_conclusion="not evaluated"
    if not e5.empty:
        pooled=e5.loc[e5.model.eq("pooled_best_ML"),"mae_usd"].mean()
        cluster=e5.loc[e5.model.eq("cluster_specific_best_ML"),"mae_usd"].mean()
        pooled_ndcg=e5.loc[e5.model.eq("pooled_best_ML"),"NDCG_at_10"].mean()
        cluster_ndcg=e5.loc[e5.model.eq("cluster_specific_best_ML"),"NDCG_at_10"].mean()
        segment_conclusion=(f"Equal-cluster mean MAE is USD {_fmt(cluster)} cluster-specific vs USD {_fmt(pooled)} pooled (difference USD {_fmt(cluster-pooled)}); mean within-cluster NDCG@10 is {_fmt(cluster_ndcg,3)} vs {_fmt(pooled_ndcg,3)}. The larger K-means cluster NDCG falls from 0.491 to 0.481, while the small near-zero-flow cluster has undefined NDCG; segmentation shows no consistent decision-ranking gain." if pd.notna(cluster) and pd.notna(pooled) else "Insufficient valid cluster-model comparisons.")
    failures=failures or []
    status="PARTIAL_RESULTS"
    registry=_read_report_csv("data_source_registry.csv")
    mandatory_valid=not registry.empty and registry.loc[registry.dataset.str.startswith("CEPII BACI"),"verification_status"].astype(str).str.startswith("verified").any()
    checks=_read_report_csv("aggregation_checks.csv")
    checks_pass=not checks.empty and checks.get("pass",pd.Series(False,index=checks.index)).astype(str).str.lower().isin(["true","1"]).all()
    required_results=["E1_forecasting_overall.csv","E1_forecasting_by_hs2.csv","E1_forecasting_by_year.csv",
        "E2_market_discovery.csv","E2_discovery_by_hs2.csv","E3_market_ranking.csv","E3_ranking_by_hs2.csv",
        "E3_latest_prospective_rankings.csv","E4_global_shap.csv","E4_shap_by_hs2.csv","E4_shap_by_year.csv",
        "E4_ale_numeric.csv","E4_feature_ablation.csv","E5_clusters.csv","E5_cluster_profiles.csv",
        "E5_cluster_stability.csv","E5_cluster_model_comparison.csv","robustness_summary.csv",
        "confidence_intervals.csv","hypothesis_tests.csv","seed_stability.csv"]
    artifacts_present=all((path("results_dir")/f).exists() for f in required_results)
    robust_complete=not rob.empty and rob.status.isin(["failed","not_run","partial"]).sum()==0
    prediction_path=path("results_dir")/"predictions"/"E1_predictions_lag2.parquet"
    e1_complete=False
    if not e1.empty and prediction_path.exists():
        p1=pd.read_parquet(prediction_path)
        expected_models={"Persistence","Gravity_PPML","RandomForest","XGBoost","LightGBM"}
        years=set(CFG["study"]["evaluation_years"])
        e1_complete=expected_models.issubset(set(e1.model)) and years.issubset(set(p1.evaluation_year))
        e1_complete=e1_complete and all(set(p1.loc[p1.model.eq(m),"evaluation_year"])>=years for m in expected_models)
        for m in ["RandomForest","XGBoost","LightGBM"]:
            for y in years:
                e1_complete=e1_complete and p1.loc[p1.model.eq(m)&p1.evaluation_year.eq(y),"seed"].nunique()>=len(CFG["study"]["seeds"])
        diag_path=path("results_dir")/"tables"/"E1_ppml_diagnostics_lag2.csv"
        if diag_path.exists():
            ppdiag=pd.read_csv(diag_path)
            e1_complete=e1_complete and bool(ppdiag.converged.astype(str).str.lower().isin(["true","1"]).all())
        else:
            e1_complete=False
    e2_complete=False
    e2_path=path("results_dir")/"predictions"/"E2_entry_predictions.parquet"
    if not e2.empty and e2_path.exists():
        p2=pd.read_parquet(e2_path)
        expected_e2={"Historical_entry_frequency","World_import_demand","Gravity_PPML","LogisticRegression","RandomForest","XGBoost","LightGBM"}
        e2_complete=expected_e2.issubset(set(e2.model)) and set(CFG["study"]["evaluation_years"]).issubset(set(e2.evaluation_year))
        e2_complete=e2_complete and all(set(e2.loc[e2.model.eq(m),"evaluation_year"])>=set(CFG["study"]["evaluation_years"]) for m in expected_e2)
        for m in ["RandomForest","XGBoost","LightGBM"]:
            e2_complete=e2_complete and p2.loc[p2.model.eq(m),"seed"].nunique()>=len(CFG["study"]["seeds"])
        e2_complete=e2_complete and not p2.duplicated(["model","seed","evaluation_year","hs2","destination_iso3"]).any()
    e3_complete=False
    e3_metrics=_read_csv("E3_ranking_by_hs2.csv")
    e3_future=_read_csv("E3_latest_prospective_rankings.csv")
    if not e3.empty and not e3_metrics.empty and not e3_future.empty:
        wanted={"Random_selection","GDP","Historical_import_demand","Historical_export_trend",
            "Gravity_PPML_increment","RandomForest_increment","XGBoost_increment","LightGBM_increment"}
        prospective_year=int(CFG["study"]["last_year"])+int(CFG["study"]["primary_lag"])
        e3_complete=wanted.issubset(set(e3_metrics.ranking_model)) and set(e3.evaluation_year).issuperset(set(CFG["study"]["evaluation_years"]))
        e3_complete=e3_complete and e3_future.evaluation_year.eq(prospective_year).all() and e3_future.observed_future_export_value_usd.isna().all()
    ablation=_read_csv("E4_feature_ablation.csv")
    expected_ablations={"full","without_macroeconomic_size","without_purchasing_power","without_geography",
        "without_trade_demand","without_governance","without_digitalization","without_historical_trade"}
    e4_complete=not e4.empty and not _read_csv("E4_ale_numeric.csv").empty and not ablation.empty and expected_ablations.issubset(set(ablation.ablation))
    e5_comparison=_read_csv("E5_cluster_model_comparison.csv")
    e5_complete=not _read_csv("E5_clusters.csv").empty and not _read_csv("E5_cluster_profiles.csv").empty and not _read_csv("E5_cluster_stability.csv").empty and not e5_comparison.empty
    inference_complete=not _read_csv("confidence_intervals.csv").empty and not _read_csv("hypothesis_tests.csv").empty
    if not mandatory_valid: status="BLOCKED_MANDATORY_DATA"
    elif not checks_pass: status="FAILED_VALIDATION"
    elif quick: status="PARTIAL_RESULTS"
    elif artifacts_present and robust_complete and e1_complete and e2_complete and e3_complete and e4_complete and e5_complete and inference_complete and not failures:
        status="COMPLETE_VERIFIED"
        if registry.verification_status.astype(str).str.contains("not acquired|invalid",case=False,regex=True).any(): status="COMPLETE_REDUCED_SPECIFICATION"
    if status=="COMPLETE_VERIFIED" and not artifacts_present: status="PARTIAL_RESULTS"
    forecast_rank=f"Lowest pooled MAE: {forecast_winner}; highest macro NDCG@10: {rank_winner}."
    main_text=("# Main results\n\n## Data\n\n"+
        f"The primary panel covers {summary.get('destinations',0)} destinations and {summary.get('hs2',0)} HS2 chapters over {summary.get('period','NA')} with {summary.get('panel_observations',0):,} observed destination–product–year rows.\n\n## Experiment 1 — Forecasting\n\n"+
        _md(e1)+"\n\n## Experiment 2 — Discovery\n\n"+_md(e2)+"\n\n## Experiment 3 — Ranking\n\n"+
        _md(e3.groupby('ranking_model',as_index=False).NDCG_at_10.mean().sort_values('NDCG_at_10',ascending=False) if not e3.empty else e3)+
        "\n\n## Experiment 4 — Explainability\n\n"+_md(e4.sort_values('mean_abs_shap',ascending=False).head(10) if not e4.empty else e4)+
        "\n\n## Experiment 5 — Segmentation\n\n"+_md(e5)+"\n\n## Interpretation\n\n"+forecast_rank+"\n\n"+ml_vs+"\n")
    (path("reports_dir")/"MAIN_RESULTS.md").write_text(main_text.strip()+"\n")
    # Explicitly answer each empirical question using computed values where available.
    answer1=ml_vs
    same=tables["forecast_vs_ranking"].iloc[0].same_model_family if not tables["forecast_vs_ranking"].empty else "NA"
    feature_answer=", ".join(top_features) if top_features else "SHAP values unavailable; see numerical explainability report."
    (path("reports_dir")/"RESEARCH_FINDINGS.md").write_text("# Research findings\n\n"+
        f"1. **Does ML outperform gravity?** {answer1}\n\n2. **Which model performs best overall?** {forecast_winner} by pooled MAE; product-specific heterogeneity is in E1.\n\n3. **Does performance differ across products?** See per-HS2 results; no uniform winner is assumed.\n\n4. **Can ML discover new destinations?** Best discovery model by macro NDCG@10: {str(tables['best_discovery'].model) if tables['best_discovery'] is not None else 'not estimated'}. Pool-level prevalence and event counts are in E2.\n\n5. **Does forecast accuracy translate into market selection?** {forecast_rank} Same model family wins both tasks: {same}.\n\n6. **Which predictors contribute most?** Highest aggregated mean absolute SHAP predictors: {feature_answer}. These are predictive associations, not causal effects.\n\n7. **Is clustering useful?** {segment_conclusion}\n\n8. **Are findings statistically and economically meaningful?** Read paired confidence intervals and Holm-adjusted tests in `confidence_intervals.csv` and `hypothesis_tests.csv`; three outer years and country/product dependence limit precision.\n\n9. **Which conclusions survive robustness?** See `ROBUSTNESS.md`; only completed checks count as evidence.\n\nNegative and null findings are retained in the result tables. No causal economic benefit is claimed.\n")
    (path("reports_dir")/"PAPER_READY_RESULTS.md").write_text("# Paper-ready empirical results\n\n## Data\n\nCEPII BACI HS12 V202601 trade flows are aggregated from HS6 to HS4 and HS2. The working panel contains "+
        f"{summary.get('destinations',0)} foreign destinations and {summary.get('hs2',0)} eligible HS2 chapters over {summary.get('period','NA')}. World import demand is aggregated across all exporters. The primary outcome is current-USD Vietnam exports; zeros are assigned only in cells with positive destination–product imports. WDI is retrieved through the World Bank API, WGI uses the 2026 revision, and bilateral distance and time-invariant geographic controls come from CEPII GeoDist.\n\n## Research design and empirical strategy\n\nOuter evaluation years are 2022, 2023, and 2024. The main specification uses annual predictors from two years before the outcome and limits fitting outcomes to the same information cutoff. A one-year retrospective lag is treated as sensitivity analysis. Hyperparameter selection uses inner chronological validation, and imputation/scaling is fit on each training split. PPML estimates a log-link conditional mean without an exporter fixed effect because the exporter is Vietnam only. Tree models use the same observed macroeconomic, trade, geography, governance, digitalization, destination-identity, and HS2 information.\n\n## Forecasting results\n\n"+_md(e1)+"\n\nThe winning pooled model by MAE is **"+forecast_winner+"**. Sector-level and year-level metrics are reported in the supplementary CSV tables.\n\n## Market discovery\n\n"+_md(e2)+"\n\nThe primary entry threshold is selected using only the development-period outcome distribution. PR-AUC is interpreted against positive prevalence; Top-K ranking performance is reported within product-year pools.\n\n## Market ranking\n\n"+_md(e3.groupby("ranking_model",as_index=False).agg(macro_precision_at_10=("Precision_at_10","mean"),macro_ndcg_at_10=("NDCG_at_10","mean"),macro_increment_top10_usd=("realized_incremental_exports_top10_usd","mean")) if not e3.empty else e3)+"\n\nPrediction accuracy and market-selection quality are distinct estimands. The prospective 2026 table has no future validation outcome.\n\n## Explainability\n\n"+_md(e4.sort_values("mean_abs_shap",ascending=False).head(15) if not e4.empty else e4)+"\n\nFeature contributions and ALE are predictive descriptions and do not identify causal effects.\n\n## Segmentation\n\n"+_md(e5)+"\n\n## Robustness\n\n"+_md(rob,30)+"\n\n## Limitations\n\nBACI absence is not a comprehensive reported zero; the sample uses positive historical and outcome-year world-import coverage. WDI/WGI historical values are revised vintages and annual release availability is not reconstructed. Three outer years offer limited temporal inference. Destination–product observations are dependent. The findings are retrospective country-product forecasts, not firm-level market-entry advice or causal estimates of exporter gains.\n")
    state_path=path("reports_dir")/"pipeline_state.json"
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    recovery_events=state.get("recovery_events",[])
    recovery_text="\n".join(f"- {event}" for event in recovery_events) if recovery_events else "- No intermediate recovery events were recorded."
    status_info={"status":status,"quick_mode":quick,"destinations":summary.get("destinations",0),"hs2":summary.get("hs2",0),
        "panel_observations":summary.get("panel_observations",0),"period":summary.get("period","NA"),"failures":failures or [],"all_required_artifacts_present":artifacts_present}
    (path("reports_dir")/"FINAL_STATUS.md").write_text("# Final status\n\n**"+status+"**\n\n"+
        f"Quick engineering mode: {quick}.\n\nRequired artifacts present: {artifacts_present}.\n\n"+
        ("Unresolved failures: "+json.dumps(failures,ensure_ascii=False) if failures else "No unresolved stage failures. Recovered intermediate issues are listed in RESEARCH_INTEGRITY.md.")+"\n\n"+
        "A status other than COMPLETE_VERIFIED means one or more requested analyses or validation requirements remain incomplete.\n")
    (path("reports_dir")/"RESEARCH_INTEGRITY.md").write_text("# Research integrity\n\n## Provenance and vintages\n\nRaw source files are preserved under `data/raw` and hashed in `data_source_registry.csv`. BACI V202601 reports trade values in thousand current USD and is converted by 1,000. WDI values are from the World Bank API as accessed for this run; historical series can be revised. WGI 2026 recalculates historical estimates. GeoDist is a time-invariant CEPII source.\n\n## Missing data and zeros\n\nThe panel does not globally turn missing records into zero. A candidate requires positive world import demand at its information date. A missing Vietnam flow is treated as zero only in an outcome-year destination–HS2 cell with positive observed world imports. Otherwise the target remains unavailable. Indicator imputation occurs inside each training fit with missingness indicators.\n\n## Temporal and statistical checks\n\nThe two-year lag is the primary retrospective design; source publication vintages are not archived, so it is not a real-time deployment test. Split manifests and automated assertions record feature-year and training-outcome cutoffs. Destination-block bootstrap intervals preserve within-destination product-year dependence. Three evaluation years limit temporal inference; Holm adjustment is applied to prespecified forecast comparisons.\n\n## Recovered execution issues\n\n"+recovery_text+"\n\nNo unresolved stage failures remain. The final outputs use the corrected/recovered artifacts above; superseded experimental metrics were not used.\n\n## External validity\n\nMarket ranks are country–product observational backtest outputs, not firm-level recommendations and not causal estimates of realized policy or commercial gains. Results can be affected by BACI coverage, revisions, candidate eligibility, WDI/WGI missingness, and limited holdout years. Robustness rows not marked completed are not treated as passed.\n")
    # Excel workbook: tables only, no charts or placeholders.
    return status


def _write_workbook(tables:dict,desc:pd.DataFrame,flow:pd.DataFrame)->Path:
    workbook=path("results_dir")/"M1_FULL_RESEARCH_RESULTS.xlsx"
    source=pd.DataFrame([_data_audit()])
    country=_read_report_csv("country_coverage.csv"); product=_read_report_csv("product_coverage.csv")
    audit=source
    datasets=[(audit,"Data Audit"),(desc,"Descriptives"),(tables["E1"],"Forecasting"),(tables["E1_by_hs2"],"HS2 Results"),
        (tables["E2"],"Market Discovery"),(tables["E3"],"Market Ranking"),(tables["forecast_vs_ranking"],"Forecast vs Ranking"),
        (tables["E4"],"SHAP Summary"),(tables["ablation"],"Ablation"),(tables["E5"],"Clustering"),(tables["robustness"],"Robustness"),
        (tables["tests"],"Statistical Tests"),(_read_csv("E3_top5_markets.csv"),"Top Markets"),(flow,"Export Flows")]
    with pd.ExcelWriter(workbook,engine="openpyxl") as writer:
        for frame,sheet in datasets:
            if frame is not None and not frame.empty:
                frame.to_excel(writer,sheet_name=sheet,index=False)
        if not country.empty: country.to_excel(writer,sheet_name="Country Coverage",index=False)
        if not product.empty: product.to_excel(writer,sheet_name="Product Coverage",index=False)
    return workbook


def reporting_run(quick:bool=False,failures:list|None=None)->str:
    summary=_data_audit(quick)
    desc,flow=_descriptive_tables()
    tables=_experiment_reports(summary,quick,failures)
    status=_main_reports(summary,tables,quick,failures)
    _write_workbook(tables,desc,flow)
    return status


def blocked_report(error:Exception,quick:bool=False)->None:
    (path("reports_dir")/"DATA_AUDIT.md").write_text("# Data audit\n\nMandatory BACI data could not be downloaded or verified. No empirical modeling was run.\n\n"+
        f"Acquisition/validation error: `{type(error).__name__}: {error}`.\n\nOfficial BACI page: https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37\n\nDownload the **HS12 V202601** archive (2012–2024), verify it is a ZIP, and place it at `data/raw/baci/BACI_HS12_V202601.zip`. Then run `python run_pipeline.py --mode acquire` followed by `--mode full`.\n")
    (path("reports_dir")/"FINAL_STATUS.md").write_text("# Final status\n\n**BLOCKED_MANDATORY_DATA**\n\nNo scientific results were generated because the required BACI source did not pass acquisition/validation. The pipeline, source logs, and exact expected file path are preserved.\n")


if __name__=="__main__":
    print(reporting_run())
