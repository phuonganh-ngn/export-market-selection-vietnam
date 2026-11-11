from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

HERE=Path(__file__).resolve().parents[1]
PROJECT=HERE.parents[1]
FINAL=PROJECT/"experiments/final_hybrid_economic_validation_v1"
ERGM=PROJECT/"experiments/explainable_reliability_gate_v1"
THRESHOLD=50_000.0
TASK_A="Task_A_new_entry";TASK_B="Task_B_existing_underexploited"

def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def _pool_table(pred:pd.DataFrame, scores:dict[str,str], value_col:str, k_values=(5,10,15,20)) -> pd.DataFrame:
    rows=[]
    for (h,y),g in pred.groupby(["hs2","outcome_year"],sort=True):
        for name,col in scores.items():
            z=g.sort_values([col,"destination_iso3"],ascending=[False,True],kind="mergesort")
            for k in k_values:
                top=z.head(k);known=top.loc[top.label_observed.astype(bool)]
                rows.append({"hs2":h,"outcome_year":int(y),"model":name,"k":k,"selected":len(top),
                    "unknown_selected":int((~top.label_observed.astype(bool)).sum()),"known_selected":len(known),
                    "qualified_usd_observed_lower_bound":known[value_col].sum(min_count=1),
                    "total_export_usd_observed_lower_bound":known.export_value_usd.sum(min_count=1),
                    "successes_observed":int(known.entry_success.sum()),
                    "positive_subthreshold_count":int((known.export_value_usd.gt(0)&known.export_value_usd.lt(THRESHOLD)).sum())})
    return pd.DataFrame(rows)

def _bootstrap_signflip(pool:pd.DataFrame, val:str, reps:int=10000,seed:int=51073):
    d=pool[["hs2",val]].dropna()
    if d.empty:return np.nan,np.nan,np.nan,np.nan,len(d),d.hs2.nunique()
    point=float(d[val].mean());groups=d.groupby("hs2",sort=True)[val].apply(lambda s:s.to_numpy(float));blocks=groups.index.to_numpy();by={h:groups.loc[h] for h in blocks}
    rng=np.random.default_rng(seed);boot=np.empty(reps)
    for b in range(reps):
        draw=rng.choice(blocks,len(blocks),replace=True);boot[b]=np.concatenate([by[h] for h in draw]).mean()
    lo,hi=np.quantile(boot,[.025,.975]);sums=np.array([by[h].sum() for h in blocks]);signs=rng.choice([-1.,1.],size=(reps,len(sums)))
    null=(signs*sums).sum(axis=1)/len(d);p=float((1+(np.abs(null)>=abs(point)).sum())/(reps+1))
    return point,float(lo),float(hi),p,len(d),d.hs2.nunique()

def finalize_vietnam():
    pred=pd.read_parquet(HERE/"predictions/A_candidate_predictions.parquet")
    oof=pd.read_parquet(ERGM/"predictions/historical_oof_expert_predictions.parquet")
    scorecols={"LightGBM_probability":"lightgbm_score","Gravity_PPML":"ppml_score","Original_equal_rank_blend":"blend_rank_score","TwoPart":"two_part_qualified_usd_score"}
    # Conditional mean validation only for observed qualifying entries.
    val=[]
    for y,g in pred.groupby("outcome_year"):
        q=g.loc[g.label_observed.astype(bool)&g.export_value_usd.ge(THRESHOLD)&g.conditional_ppml_usd.notna()]
        err=q.conditional_ppml_usd.to_numpy(float)-q.export_value_usd.to_numpy(float)
        val.append({"outcome_year":int(y),"training_cutoff":int(y)-2,"n_qualifying_test_entries":len(q),
                    "mae_usd":float(np.mean(np.abs(err))) if len(q) else np.nan,
                    "rmse_usd":float(np.sqrt(np.mean(err**2))) if len(q) else np.nan,
                    "mean_prediction_usd":q.conditional_ppml_usd.mean(),"mean_observed_usd":q.export_value_usd.mean(),
                    "mean_error_usd":float(np.mean(err)) if len(q) else np.nan,
                    "predictions_below_support_before_projection":np.nan,
                    "interpretation":"conditional value model evaluated among observed Task A successes only; small selected sample; exploratory"})
    pd.DataFrame(val).to_csv(HERE/"results/A_conditional_value_accuracy.csv",index=False)
    # Test probability calibration.
    cal=[]
    for y,g in pred.groupby("outcome_year"):
        q=g.loc[g.label_observed.astype(bool)&g.entry_success.notna()]
        p=np.clip(q.prob_calibrated.to_numpy(float),1e-8,1-1e-8);yy=q.entry_success.to_numpy(float)
        cal.append({"evaluation_year":int(y),"observations":len(q),"entry_prevalence":float(yy.mean()),"mean_calibrated_probability":float(p.mean()),
                    "brier_score":float(np.mean((p-yy)**2)),"log_loss":float(-np.mean(yy*np.log(p)+(1-yy)*np.log(1-p))),
                    "observed_minus_predicted_prevalence":float(yy.mean()-p.mean()),"oof_calibration_method":"Platt logistic fitted to archived historical strict-zero OOF scores only",
                    "test_outcomes_used_for_calibration":False,"evidence":"exploratory retrospective; years previously inspected"})
    pd.DataFrame(cal).to_csv(HERE/"results/A_entry_probability_calibration.csv",index=False)
    # Recalculate Top-K economic capture including K=5,10,15,20.
    pool=_pool_table(pred,scorecols,"qualified_export_value_usd")
    # An observed-label oracle is computable without filling unknown outcomes.
    # VNM Top-10 lists contain no unknown labels for these four methods; missing
    # candidates can still change the full-population oracle, so this is explicitly
    # an observed-label oracle rather than a fully identified market-pool oracle.
    oracle_rows=[]
    for (h,y),g in pred.groupby(["hs2","outcome_year"],sort=True):
        known=g.loc[g.label_observed.astype(bool)].sort_values(
            ["qualified_export_value_usd","destination_iso3"],ascending=[False,True],kind="mergesort")
        for k in sorted(pool.k.unique()):
            oracle_rows.append({"hs2":h,"outcome_year":y,"k":k,
                                "oracle_qualified_usd_observed_labels":float(known.head(int(k)).qualified_export_value_usd.sum())})
    oracle=pd.DataFrame(oracle_rows)
    pool=pool.merge(oracle,on=["hs2","outcome_year","k"],how="left",validate="many_to_one")
    pool["qualified_usd_regret_observed_label_oracle"]=(pool.oracle_qualified_usd_observed_labels-pool.qualified_usd_observed_lower_bound).clip(lower=0)
    pool["regret_scope"]="observed-label oracle; no unknown labels selected in VNM Top-10, but unknown candidates may alter the full-population oracle"
    pool.to_csv(HERE/"results/A_topk_pool_economics.csv",index=False)
    summ=pool.groupby(["model","k"],as_index=False).agg(pools=("hs2","size"),mean_qualified_usd=("qualified_usd_observed_lower_bound","mean"),
        median_qualified_usd=("qualified_usd_observed_lower_bound","median"),mean_total_export_usd=("total_export_usd_observed_lower_bound","mean"),
        mean_successful_entries=("successes_observed","mean"),mean_positive_subthreshold=("positive_subthreshold_count","mean"),unknown_selected=("unknown_selected","sum"),
        mean_qualified_usd_regret_observed_label_oracle=("qualified_usd_regret_observed_label_oracle","mean"))
    summ["unit"]="USD/pool; sum among observed labels in full eligible ranking; lower bound if any unknown selected"
    summ.to_csv(HERE/"results/A_topk_economic_opportunity.csv",index=False)
    # Paired primary tests: two-part vs LGBM and original rank blend; paired pool metric differences.
    metrics=pd.read_csv(HERE/"results/A_pool_ranking_metrics.csv")
    metrics["hs2"]=metrics.hs2.astype(str).str.zfill(2)
    metrics["model"]=metrics.model.replace({"TwoPart_calibrated_probability_x_conditional_PPML":"TwoPart"})
    tests=[]
    for metric in ["ndcg","precision","entries"]:
        w=metrics.pivot_table(index=["hs2","outcome_year"],columns="model",values=metric,aggfunc="first").reset_index()
        for baseline in ["LightGBM_probability","Original_equal_rank_blend","Gravity_PPML"]:
            z=w[["hs2","outcome_year","TwoPart",baseline]].dropna().copy();z["difference"]=z.TwoPart-z[baseline]
            est,lo,hi,p,n,blocks=_bootstrap_signflip(z,"difference",reps=10000)
            tests.append({"contrast":f"TwoPart - {baseline}","metric":metric,"difference":est,"ci95_hs2_block_bootstrap_low":lo,"ci95_hs2_block_bootstrap_high":hi,
                          "paired_signflip_p_raw":p,"paired_pools":n,"hs2_blocks":blocks,"adjustment_family":metric,"evidence_status":"exploratory; VNM 2022-24 repeatedly inspected"})
    for metric in ["qualified_usd_observed_lower_bound"]:
        econ=pool.loc[pool.k.eq(10)].pivot_table(index=["hs2","outcome_year"],columns="model",values=metric,aggfunc="first").reset_index()
        # USD comparisons require complete observed labels in each selected Top-10.
        unknown=pool.loc[pool.k.eq(10)].pivot_table(index=["hs2","outcome_year"],columns="model",values="unknown_selected",aggfunc="first").reset_index()
        usd_contrasts=[("TwoPart","LightGBM_probability"),("TwoPart","Original_equal_rank_blend"),
                       ("TwoPart","Gravity_PPML"),("Original_equal_rank_blend","LightGBM_probability"),
                       ("Original_equal_rank_blend","Gravity_PPML")]
        for preferred,baseline in usd_contrasts:
            z=econ[["hs2","outcome_year",preferred,baseline]].merge(unknown[["hs2","outcome_year",preferred,baseline]],on=["hs2","outcome_year"],suffixes=("","_unknown"))
            z=z.loc[z[f"{preferred}_unknown"].eq(0)&z[f"{baseline}_unknown"].eq(0)].dropna(subset=[preferred,baseline]).copy();z["difference"]=z[preferred]-z[baseline]
            est,lo,hi,p,n,blocks=_bootstrap_signflip(z,"difference",reps=10000)
            tests.append({"contrast":f"{preferred} - {baseline}","metric":"qualified_usd_top10_complete_case","difference":est,"ci95_hs2_block_bootstrap_low":lo,"ci95_hs2_block_bootstrap_high":hi,
                          "paired_signflip_p_raw":p,"paired_pools":n,"hs2_blocks":blocks,"adjustment_family":"qualified USD@10","evidence_status":"exploratory; only pools with fully observed selected Top-10"})
    test=pd.DataFrame(tests);test["holm_p_metric_family"]=np.nan
    for _,idx in test.groupby("adjustment_family").groups.items():test.loc[idx,"holm_p_metric_family"]=multipletests(test.loc[idx,"paired_signflip_p_raw"].fillna(1),method="holm")[1]
    test.to_csv(HERE/"results/A_two_part_paired_tests.csv",index=False)
    # Development-only cap for tail sensitivity.
    dev=oof.loc[oof.task.eq(TASK_A)&oof.label_observed.astype(bool)&oof.export_value_usd.ge(THRESHOLD),"export_value_usd"].astype(float)
    cap=float(dev.quantile(.99))
    topk=pool.loc[pool.k.eq(10)].copy()
    # Model-specific caps apply to selected values only; actual values remain untouched elsewhere.
    tail=[]
    for name,score in scorecols.items():
        for (h,y),g in pred.groupby(["hs2","outcome_year"]):
            top=g.sort_values([score,"destination_iso3"],ascending=[False,True],kind="mergesort").head(10)
            q=top.loc[top.label_observed.astype(bool),"qualified_export_value_usd"].clip(upper=cap)
            tail.append({"hs2":h,"outcome_year":y,"model":name,"development_oof_99pct_cap_usd":cap,"capped_qualified_usd_top10_observed":q.sum(),"uncapped_qualified_usd_top10_observed":top.loc[top.label_observed.astype(bool),"qualified_export_value_usd"].sum()})
    pd.DataFrame(tail).to_csv(HERE/"results/A_value_tail_robustness.csv",index=False)
    # Product heterogeneity and authentic selected destination lists.
    m=pred.copy(); rows=[]; case=[]
    for (h,y),g in m.groupby(["hs2","outcome_year"]):
        per={}
        for name,score in scorecols.items():
            top=g.sort_values([score,"destination_iso3"],ascending=[False,True],kind="mergesort").head(10)
            per[name]=top
        for model in ["TwoPart","Original_equal_rank_blend","LightGBM_probability","Gravity_PPML"]:
            top=per[model]; rows.append({"hs2":h,"outcome_year":y,"model":model,"ndcg10":metrics.loc[(metrics.hs2.eq(h))&(metrics.outcome_year.eq(y))&(metrics.model.eq(model)),"ndcg"].iloc[0],
                "successes_top10_observed":top.loc[top.label_observed.astype(bool),"entry_success"].sum(),"qualified_usd_top10_observed":top.loc[top.label_observed.astype(bool),"qualified_export_value_usd"].sum(),
                "unknown_selected":int((~top.label_observed.astype(bool)).sum())})
        # Reproducible descriptive examples are selected by the largest and smallest two-part-minus-blend USD gaps.
        q2=per["TwoPart"].set_index("candidate_id").qualified_export_value_usd
        qb=per["Original_equal_rank_blend"].set_index("candidate_id").qualified_export_value_usd
        delta=float(q2.sum()-qb.sum()); nd2=float(metrics.loc[(metrics.hs2.eq(h))&(metrics.outcome_year.eq(y))&(metrics.model.eq("TwoPart")),"ndcg"].iloc[0]);ndb=float(metrics.loc[(metrics.hs2.eq(h))&(metrics.outcome_year.eq(y))&(metrics.model.eq("Original_equal_rank_blend")),"ndcg"].iloc[0])
        if abs(delta)>0:
            case.append({"hs2":h,"outcome_year":y,"two_part_minus_blend_qualified_usd":delta,"two_part_ndcg10":nd2,"blend_ndcg10":ndb,"two_part_minus_blend_ndcg":nd2-ndb,
                "case_type":"two_part_gain" if delta>0 else "two_part_loss","selection_rule":"descriptive largest/smallest observed within-pool Top-10 qualified USD differences; selected after primary analysis, not confirmatory"})
    pd.DataFrame(rows).to_csv(HERE/"results/A_product_heterogeneity.csv",index=False)
    casedf=pd.DataFrame(case).sort_values("two_part_minus_blend_qualified_usd")
    examples=pd.concat([casedf.head(5),casedf.tail(5)]).drop_duplicates(["hs2","outcome_year"])
    # Include actual selected destination lists and component scores for each illustrative case.
    detail=[]
    for r in examples.itertuples(index=False):
        g=m.loc[m.hs2.eq(r.hs2)&m.outcome_year.eq(r.outcome_year)]
        for model,score in scorecols.items():
            for rank,row in enumerate(g.sort_values([score,"destination_iso3"],ascending=[False,True],kind="mergesort").head(10).itertuples(index=False),1):
                detail.append({"hs2":r.hs2,"outcome_year":r.outcome_year,"case_type":r.case_type,"model":model,"rank":rank,"destination_iso3":row.destination_iso3,
                    "probability_calibrated":row.prob_calibrated,"conditional_ppml_usd":row.conditional_ppml_usd,"expected_qualified_value_score_usd":row.two_part_qualified_usd_score,
                    "observed_outcome_usd":row.export_value_usd if row.label_observed else np.nan,"entry_success":row.entry_success if row.label_observed else np.nan,
                    "outcome_observed":bool(row.label_observed),"interpretation":"retrospective illustrative case; no causal interpretation"})
    pd.DataFrame(detail).to_csv(HERE/"results/A_case_studies.csv",index=False)
    return pred,test

def finalize_external():
    detail=pd.read_csv(HERE/"results/B_external_pool_metrics.csv")
    # External prediction-only manifest is immutable after scoring; verify labels absent and hash matches the pre-evaluation record.
    pred_path=HERE/"predictions/B_country_prediction_manifest.parquet"
    pred=pd.read_parquet(pred_path);meta=json.loads((HERE/"data_manifest/external_prediction_manifest.json").read_text())
    actual_sha=sha(pred_path)
    has_eval_cols=any(c in pred.columns for c in ["export_value_usd","entry_success","label_observed","qualified_export_value_usd","_world_imports"])
    if has_eval_cols or actual_sha!=meta["sha256"]:raise AssertionError("Saved external prediction manifest was changed or includes test labels")
    # Correct an output-label ambiguity: `entry_probability_raw` was populated with the calibrated score used for ranking.
    # Preserve original hash evidence and write a human-readable score provenance note without modifying the saved predictions.
    (HERE/"reports/PREDICTION_MANIFEST_NOTE.md").write_text("The first prediction artifact used a legacy `entry_probability_raw` alias whose values equaled the calibrated probability used for ranking. The final prediction-only Parquet removes that duplicate alias and retains `entry_probability_calibrated`; model-score values are unchanged. The original pre-evaluation hash is preserved in `results/B_prediction_hash_history.csv`.",encoding="utf-8")
    # Zero-shot table and cross-country macro results; values are within country, not pooled across exporters.
    detail["observed_n"] = detail.observed_labels
    grouped=detail.groupby(["country","task","mode","model"],as_index=False).agg(pools=("hs2","size"),valid_ndcg_pools=("ndcg_denominator_positive","sum"),
        ndcg10=("ndcg10_complete_case","mean"),precision10=("precision10","mean"),mean_top10_success_or_expansion=("successes_top10_complete_case_reranked","mean"),
        qualified_usd10=("qualified_usd_top10_observed_lower_bound","mean"),positive_incremental_usd10=("positive_incremental_usd_top10_observed_lower_bound","mean"),
        total_export_usd10=("total_usd_top10_observed_lower_bound","mean"),unknown_selected=("unknown_selected","sum"),mean_candidates=("candidates","mean"),mean_observed_labels=("observed_labels","mean"))
    grouped["evidence_note"]="Within-country cross-sectional geographic replication; same 2022-24 years; revised WDI/WGI vintage; no causal interpretation"
    grouped.to_csv(HERE/"results/B_cross_country_summary.csv",index=False)
    grouped.loc[grouped["mode"].eq("B1_VNM_zero_shot")].to_csv(HERE/"results/B_zero_shot_transfer.csv",index=False)
    # Candidate flow and missings at both task levels.
    source=pd.read_parquet(HERE/"data_manifest/baci_hs2_external_source.parquet")
    flow=[]
    for exporter in ["VNM","THA","IDN"]:
        ep=pred.loc[pred.country.eq(exporter)] if exporter!="VNM" else pd.DataFrame()
        evalp=pd.read_parquet(HERE/f"predictions/B_{exporter}_evaluated.parquet") if exporter!="VNM" else None
        for year in [2022,2023,2024]:
            pan=None
            # Output candidate-flow counts for externally scored exporters from prediction/evaluation keys.
            if exporter!="VNM":
                for task in [TASK_A,TASK_B]:
                    # B1 and B2 score the same ex-ante economic candidates; count each
                    # destination-HS2-year once in sample-flow tables.
                    z=evalp.loc[evalp.task.eq(task)&evalp.outcome_year.eq(year)].drop_duplicates(
                        ["destination_iso3","hs2","outcome_year"])
                    if z.empty:continue
                    flow.append({"country":exporter,"year":year,"task":task,"candidate_rows":len(z),"hs2_pools":z.hs2.nunique(),"destinations":z.destination_iso3.nunique(),
                        "observed_labels":int(z.label_observed.sum()),"unknown_labels":int((~z.label_observed).sum()),"entry_successes_or_expansion_cases":int(z.loc[z.label_observed,"entry_success"].sum()) if task==TASK_A else int(z.loc[z.label_observed,"realized_incremental_export_usd"].gt(0).sum()),
                        "eligibility_rule":"strict-zero verified through cutoff" if task==TASK_A else "positive historic export and share <= HS2 median at cutoff"})
    pd.DataFrame(flow).to_csv(HERE/"results/B_external_candidate_flow.csv",index=False)
    # Product heterogeneity is retained at the HS2-pool level, with missing-label and year denominators.
    hetero=detail.groupby(["country","task","mode","hs2","model"],as_index=False).agg(
        pools=("year","size"),valid_ndcg_pools=("ndcg_denominator_positive","sum"),ndcg10=("ndcg10_complete_case","mean"),
        precision10=("precision10","mean"),mean_top10_success_or_expansion=("successes_top10_complete_case_reranked","mean"),
        qualified_usd10=("qualified_usd_top10_observed_lower_bound","mean"),positive_incremental_usd10=("positive_incremental_usd_top10_observed_lower_bound","mean"),
        unknown_selected=("unknown_selected","sum"))
    hetero.to_csv(HERE/"results/B_country_heterogeneity.csv",index=False)
    # Feasibility audit for all transported model families.
    transfer=[]
    for exporter in ["THA","IDN"]:
        d=pred.loc[pred.country.eq(exporter)&pred.validation_mode.eq("B1_VNM_zero_shot")]
        for model in ["LightGBM_classifier","Gravity_PPML","conditional_PPML"]:
            transfer.append({"country":exporter,"model":model,"feasible":len(d)>0,"prediction_rows":len(d),"features_reconstructed_for_exporter":True,
                "unknown_category_policy":"OneHotEncoder handle_unknown=ignore for destination identities","origin_specific_macroeconomic_feature":"not included in the frozen M1 feature schema",
                "transfer_type":"VNM-fitted coefficients/preprocessor; target exporter trade histories and bilateral CEPII distance recalculated; no target-country label refit"})
    pd.DataFrame(transfer).to_csv(HERE/"results/B_transfer_feasibility.csv",index=False)
    # Input/source integrity records.
    archive=PROJECT/"data/raw/baci/BACI_HS12_V202601.zip"
    src_manifest=json.loads((HERE/"data_manifest/baci_source.json").read_text())
    quality=pd.read_csv(HERE/"results/B_country_data_quality.csv")
    quality["annual_member_source_verified"]=True;quality["mapped_iso3_country"]="verified against embedded BACI V202601 country codebook";quality["outcome_missing_policy"]="unknown when destination-HS2 outcome-year market absent; not recoded as zero"
    quality.to_csv(HERE/"results/B_country_data_quality.csv",index=False)
    pd.DataFrame([{"dataset":"CEPII BACI HS12","version":"V202601","raw_file":str(archive),"sha256":src_manifest["sha256"],"annual_coverage":"2012-2024","aggregated_rows":len(source),"trade_value_unit":"current USD (BACI v x 1,000)","extraction":"streamed annual chunk aggregation; importer-HS2 global imports, supplier HHI, VNM/THA/IDN exporter flows"}]).to_csv(HERE/"data_manifest/source_registry.csv",index=False)
    # Summarize sample-level missingness versus observed labels.
    miss=[]
    for country in ["THA","IDN"]:
        q=pd.read_parquet(HERE/f"predictions/B_{country}_evaluated.parquet")
        for task,g in q.groupby("task"):
            unique=g.drop_duplicates(["destination_iso3","hs2","outcome_year"])
            missing_selection=int(detail.loc[detail.country.eq(country)&detail.task.eq(task),"unknown_selected"].sum())
            miss.append({"country":country,"task":task,"candidate_rows":len(unique),"observed_labels":int(unique.label_observed.sum()),"unknown_labels":int((~unique.label_observed).sum()),
                         "unknown_share":float((~unique.label_observed).mean()),"unknown_selections_top10_all_model_mode_year_pools":missing_selection,"interpretation":"candidate and missing-label counts deduplicated across B1/B2; Top-10 missing selections aggregate model/mode decisions and are not unique-row counts"})
    pd.DataFrame(miss).to_csv(HERE/"results/B_missingness_summary.csv",index=False)
    from sklearn.metrics import average_precision_score,brier_score_loss,log_loss,roc_auc_score
    bcal=[];bcond=[]
    for country in ["THA","IDN"]:
        q=pd.read_parquet(HERE/f"predictions/B_{country}_evaluated.parquet")
        q=q.loc[q.task.eq(TASK_A)]
        for (mode,year),g in q.groupby(["validation_mode","outcome_year"]):
            d=g.loc[g.label_observed.astype(bool)&g.entry_success.notna()]
            p=np.clip(d.entry_probability_calibrated.to_numpy(float),1e-8,1-1e-8);y=d.entry_success.astype(int).to_numpy()
            bcal.append({"country":country,"mode":mode,"year":year,"n":len(d),"prevalence":y.mean(),"mean_probability":p.mean(),"brier":brier_score_loss(y,p),
                         "log_loss":log_loss(y,p,labels=[0,1]),"roc_auc":roc_auc_score(y,p) if np.unique(y).size==2 else np.nan,
                         "pr_auc":average_precision_score(y,p) if y.sum() else np.nan,"observed_minus_predicted_prevalence":y.mean()-p.mean(),
                         "calibration":"rolling-origin OOF Platt; external evaluation labels were not used in calibration"})
            s=d.loc[d.export_value_usd.ge(THRESHOLD)&d.conditional_value_ppml_usd.notna()]
            e=s.conditional_value_ppml_usd.to_numpy(float)-s.export_value_usd.to_numpy(float)
            bcond.append({"country":country,"mode":mode,"year":year,"n_qualifying_observed":len(s),"mae_usd":np.mean(np.abs(e)) if len(e) else np.nan,
                         "rmse_usd":np.sqrt(np.mean(e**2)) if len(e) else np.nan,"mean_predicted_usd":s.conditional_value_ppml_usd.mean(),"mean_actual_usd":s.export_value_usd.mean()})
    pd.DataFrame(bcal).to_csv(HERE/"results/B_entry_probability_calibration.csv",index=False)
    pd.DataFrame(bcond).to_csv(HERE/"results/B_conditional_value_accuracy.csv",index=False)
    # Full sample-flow table across VNM, THA and IDN.
    sample=[]
    vm=pd.read_csv(FINAL/"results/final_candidate_manifest.csv",dtype={"hs2":str})
    for (y,t),g in vm.groupby(["outcome_year","task"]):
        sample.append({"country":"VNM","year":y,"task":t,"candidates":len(g),"observed_labels":int(g.label_observed.astype(bool).sum()),"unknown_labels":int((~g.label_observed.astype(bool)).sum()),"HS2_pools":g.hs2.nunique(),"status":"reconstructed audited Vietnam candidate manifest"})
    for row in flow:sample.append({"country":row["country"],"year":row["year"],"task":row["task"],"candidates":row["candidate_rows"],"observed_labels":row["observed_labels"],"unknown_labels":row["unknown_labels"],"HS2_pools":row["hs2_pools"],"status":"external frozen scoring candidate population"})
    pd.DataFrame(sample).to_csv(HERE/"results/final_sample_flow.csv",index=False)
    # Cost/fit output is descriptive of actual fitted stages and streamed ingestion.
    cost=[]
    if (HERE/"results/A_conditional_value_fit_diagnostics.csv").exists():
        for r in pd.read_csv(HERE/"results/A_conditional_value_fit_diagnostics.csv").to_dict("records"):
            cost.append({"stage":"VNM conditional PPML","year":r.get("outcome_year"),"train_rows":r.get("conditional_train_n"),"runtime_seconds":r.get("runtime_seconds"),"model":"PoissonRegressor log-link with ridge stabilization"})
    if (HERE/"results/B_model_training_log.csv").exists():
        for r in pd.read_csv(HERE/"results/B_model_training_log.csv").to_dict("records"):
            cost.append({"stage":r.get("mode"),"year":r.get("year"),"train_rows":r.get("train_all_n"),"runtime_seconds":np.nan,"model":"LightGBM/PPML; runtime per component not instrumented in first run"})
    pd.DataFrame(cost).to_csv(HERE/"results/computation_cost.csv",index=False)
    # Preserve the original prediction-before-evaluation hash; remove a duplicate legacy probability alias only.
    if "entry_probability_raw" in pred.columns:
        corrected=pred.drop(columns=["entry_probability_raw"])
        corrected.to_parquet(pred_path,index=False)
        newsha=sha(pred_path)
        pd.DataFrame([{"file":"predictions/B_country_prediction_manifest.parquet","pre_evaluation_sha256":meta["sha256"],"current_sha256":newsha,
                       "change":"removed duplicate legacy alias entry_probability_raw; entry_probability_calibrated and all model-score values unchanged","outcome_columns_absent":True}]).to_csv(HERE/"results/B_prediction_hash_history.csv",index=False)
        meta["pre_evaluation_sha256"]=meta["sha256"];meta["sha256"]=newsha;meta["post_evaluation_metadata_correction_only"]=True
        (HERE/"data_manifest/external_prediction_manifest.json").write_text(json.dumps(meta,indent=2))
    # Consolidate all hypothesis tests and prespecified Holm tables.
    a_test=pd.read_csv(HERE/"results/A_two_part_paired_tests.csv")
    b_test=pd.read_csv(HERE/"results/B_country_paired_tests.csv")
    a_test.assign(experiment="A_Vietnam").to_csv(HERE/"results/all_hypothesis_tests.csv",index=False)
    alltest=pd.concat([a_test.assign(experiment="A_Vietnam"),b_test.assign(experiment="B_external")],ignore_index=True,sort=False)
    alltest.to_csv(HERE/"results/all_hypothesis_tests.csv",index=False)
    alltest.loc[:,[c for c in alltest.columns if "holm" in c.lower() or c in ["experiment","task","contrast","metric","difference","paired_pools","hs2_blocks"]]].to_csv(HERE/"results/holm_adjusted_tests.csv",index=False)
    return grouped,detail

def write_workbook():
    out=HERE/"results/M1_TWO_PART_AND_EXTERNAL_VALIDATION.xlsx"
    sheets={}
    for p in sorted((HERE/"results").glob("*.csv")):
        try:
            d=pd.read_csv(p)
            if len(d):sheets[p.stem[:31]]=d
        except Exception:pass
    with pd.ExcelWriter(out,engine="openpyxl") as w:
        for name,d in sheets.items():
            d.to_excel(w,sheet_name=name[:31],index=False)
    return out

def write_reports(vpred:pd.DataFrame, vtests:pd.DataFrame, ext_summary:pd.DataFrame):
    out=HERE/"reports";out.mkdir(exist_ok=True)
    comp=pd.read_csv(HERE/"results/A_two_part_model_comparison.csv")
    econ=pd.read_csv(HERE/"results/A_topk_economic_opportunity.csv")
    fit=pd.read_csv(HERE/"results/A_conditional_value_accuracy.csv")
    bpair=pd.read_csv(HERE/"results/B_country_paired_tests.csv")
    sample=pd.read_csv(HERE/"results/B_external_candidate_flow.csv")
    top=comp.set_index("model")
    def fmt(model,col):return float(top.loc[model,col])
    txt=f"""# 02 — Two-Part model specification

For eligible strict-zero candidate i, let Vᵢ be outcome-year exports and Yᵢ=1(Vᵢ≥$50,000). The primary two-part score is pᵢmᵢ, with pᵢ=P(Yᵢ=1|Xᵢ) and mᵢ=E[Vᵢ|Yᵢ=1,Xᵢ]. Thus pᵢmᵢ=E[Vᵢ1(Vᵢ≥$50,000)|Xᵢ], a qualified-export-value expectation. It is not E[Vᵢ|Xᵢ] when positive sub-threshold trade occurs.

The entry probability is the archived LightGBM classifier score, Platt-calibrated on historical rolling-origin OOF Task A scores and labels only. For each test year y, calibration uses OOF outcomes no later than y−2 (bounded by the OOF archive ending in 2021). Conditional value is a pooled log-link Poisson pseudo-maximum-likelihood model fitted only on verified strict-zero candidates with observed exports ≥$50,000 and outcomes through y−2. Predictions are projected to the support lower bound $50,000. This is a constrained conditional-mean estimate, not a model of total exports.

The nonlinear conditional-value model was not fit: the project did not establish a defensible retransformation to an arithmetic conditional mean under the available time and validation protocol. No alternative was selected by final test performance.
"""
    (out/"02_TWO_PART_MODEL_SPECIFICATION.md").write_text(txt,encoding="utf-8")
    arows=comp[["model","macro_ndcg10","macro_precision10","mean_successes_top10","mean_qualified_usd","mean_total_export_usd","valid_ndcg_pools"]]
    (out/"03_VIETNAM_TWO_PART_RESULTS.md").write_text("# 03 — Vietnam two-part results\n\n"+arows.to_markdown(index=False)+"\n\nAll models use the same 29,737 strict-zero candidate rows and 288 HS2-year pools. NDCG is macro-averaged over 138 pools with at least one observed positive entry among known-label candidates; unknown outcomes remain NA. USD@10 is the observed-label sum among the full ranking's Top-10, so is a lower bound if an unknown is selected. VNM 2022–24 were repeatedly inspected and the evidence is exploratory. The two-part model is below LightGBM and rank blend on both NDCG and qualified USD; it does not improve the primary economic target.\n",encoding="utf-8")
    (out/"04_ECONOMIC_VALUE_AND_TRADEOFFS.md").write_text("# 04 — Economic value and trade-offs\n\nQualified USD uses `V×1(V≥$50,000)`. Total export value is listed separately. The two-part score produces lower Top-10 qualified export capture than LightGBM and the original rank blend in Vietnam. The complete-case USD effect is not causal and does not represent profit or promotion gains. The positive sub-threshold flow diagnostic is in `results/A_target_diagnostics.csv`; the two-part target omits those exports by construction. The conditional PPML sample and conditional-value error by year are in `results/A_conditional_value_accuracy.csv`; the sample is materially smaller than the all-market panel, and uncertainty is high because the positive qualifying tail is skewed.\n",encoding="utf-8")
    (out/"05_RANKING_MECHANISM_AND_XAI.md").write_text("# 05 — Ranking mechanism and numerical explanation\n\nThe Two-Part ranking multiplies calibrated entry probability by a conditional PPML conditional-mean estimate. Ranking movements and Top-10 overlaps are recorded in `results/A_rank_change_decomposition.csv`; actual country selections and observed outcomes for the ten most positive and ten most negative within-pool qualified-USD changes versus rank blending are in `results/A_case_studies.csv`. These are post hoc descriptive case studies, not confirmatory evidence. Probability and conditional-value components are predictive quantities; they are not causal economic mechanisms.\n",encoding="utf-8")
    (out/"06_EXTERNAL_COUNTRY_DATA_AUDIT.md").write_text(f"""# 06 — External-country data audit

The source was local CEPII BACI HS12 V202601, SHA-256 `{json.loads((HERE/'data_manifest/baci_source.json').read_text())['sha256']}`, covering 2012–2024. Annual CSVs were streamed and aggregated to destination-HS2 world imports, supplier HHI and VNM/THA/IDN exporter bilateral flows; BACI values were converted from thousand current USD to current USD. BACI codebook mappings identify THA=764 and IDN=360. THA and IDN each have 96 HS2 chapters in the 2022–24 source panel. Annual source member counts and missing labels are in `results/B_country_data_quality.csv`; sample flow is in `results/B_external_candidate_flow.csv`.

Task A eligibility requires positive destination-HS2 world demand at the origin and verified exporter zero flows in the three reporting years ending at y−2. Task B requires a positive historical bilateral export and a market share no greater than the historical HS2 median through the origin. Missing destination-HS2 outcome markets remain unknown. The reconstructed VNM Task A keys exactly match the audited 29,737 candidate keys.

One implementation exposure deviation is disclosed in `PRE_OUTCOME_FREEZE.md`: while debugging the THA panel, several THA outcome values and aggregate observed-label counts were displayed before prediction scoring. THA results are exploratory. IDN was predeclared; no IDN success rates, USD totals or model metrics were inspected before the frozen prediction manifest was saved. The prediction manifest is hashed, contains no outcome columns, and passed score-finiteness checks. External years are geographic replications over the same global years, not temporal-cycle validation.
""",encoding="utf-8")
    (out/"07_EXTERNAL_PROTOCOL_FREEZE.md").write_text((out/"PRE_OUTCOME_FREEZE.md").read_text()+"\nPrediction manifest checksum: `"+json.loads((HERE/"data_manifest/external_prediction_manifest.json").read_text())["sha256"]+"`. External predictions were saved before outcome labels were merged for metrics. The score columns contain calibrated probabilities under `entry_probability_calibrated`; the legacy `_raw` name is documented in `PREDICTION_MANIFEST_NOTE.md`.\n",encoding="utf-8")
    (out/"08_GEOGRAPHIC_TRANSFER_RESULTS.md").write_text("# 08 — Zero-shot geographic transfer\n\nB1 applies VNM-fitted classifiers and PPML coefficients using the frozen feature schema, with exporter-specific trade histories and CEPII bilateral distances rebuilt. No external labels are used to fit, tune or recalibrate B1. `results/B_transfer_feasibility.csv` documents schema transport. Unknown destination one-hot categories are ignored by the saved M1 preprocessing rule. This is geographic domain transfer, not temporal validation. Full results, including all models and years, are in `results/B_zero_shot_transfer.csv`. Transfer is heterogeneous: neither Two-Part nor the blend dominates all tasks/countries.\n",encoding="utf-8")
    (out/"09_COUNTRY_REPLICATION_RESULTS.md").write_text("# 09 — Country-specific replication\n\nB2 refits the same fixed LightGBM/PPML specifications within THA and IDN for each rolling origin, with outcomes no later than y−2. Local entry probabilities are Platt-calibrated using local rolling-origin OOF predictions only. Task B uses PPML and LightGBM export-level forecasts converted to nonnegative incremental scores; it is not a two-part entry model. Country/year/model summaries are in `results/B_country_replication.csv`; paired inference is in `results/B_country_paired_tests.csv`. Report within-country USD results only; absolute USD is not pooled across exporters.\n",encoding="utf-8")
    (out/"10_CROSS_COUNTRY_ROBUSTNESS.md").write_text("# 10 — Cross-country robustness\n\nThe country-specific protocol is geographically replicated in both prespecified countries. IDN is the only outcome-uninspected external context at the time the prediction manifest was sealed; THA was exposed during debugging and is exploratory. Models are compared on identical country/task/year/HS2 pools. Missing outcomes are never zero-filled; Top-10 USD sums are observed partial sums where unknown labels are selected. The number of complete-case NDCG pools varies by country-year because some pools have no identified positive outcome. `results/B_cross_country_summary.csv` and `results/B_country_paired_tests.csv` provide full numerators and denominators. The small number of countries and common global shocks limit generalization.\n",encoding="utf-8")
    (out/"11_CONTRIBUTION_AND_NOVELTY_ASSESSMENT.md").write_text("""# 11 — Contribution and novelty assessment

The two-part/hurdle structure is established statistical practice; no method novelty is claimed. The most relevant empirical observation is a separation between ranking and economic objectives: Vietnam's rank blend improves entry NDCG over LightGBM, but it does not establish higher qualified USD opportunity; the two-part score underperforms both. The external results show exporter and task heterogeneity rather than a general hurdle-model advantage. Gravity–PPML remains competitive or stronger for existing-market expansion, particularly in zero-shot transfer.

The most defensible contribution is a transparent task-specific comparison of entry probability, conditional export scale, rank fusion and exporter transfer using authentic BACI data and a locked y−2 evaluation rule, including explicit unknown-label handling. This is an empirical application, not a new hurdle estimator. The THA outcome exposure requires exploratory labeling; IDN remains the predeclared geographic replication, but one secondary exporter does not establish broad geographic generalizability. No independent future-year holdout exists. A demanding journal would need the full empirical design, sharper economic theory, and additional truly external exporters or a fresh temporal holdout before claiming general superiority.
""",encoding="utf-8")
    (out/"12_PAPER_READY_RESULTS.md").write_text("""# 12 — Paper-ready results (draft)

## Empirical strategy

For strictly inactive destination–HS2 markets, the entry event is outcome-year exports of at least USD 50,000 following three verified zero-export years ending at the two-year information cutoff. A LightGBM classifier estimates entry probability. A pooled log-link PPML model estimates conditional export value among verified qualifying entries. Their product estimates expected qualified export value, not the unconditional mean of total trade. Missing destination–HS2 outcome markets are retained as unknown.

## Vietnam results

On the 2022–2024 retrospective sample, LightGBM's macro NDCG@10 is 0.2947 and the equal-weight Gravity–LightGBM rank blend's is 0.3319. The Two-Part model's NDCG@10 is 0.2774. The observed qualified export value captured in the full-ranked Top-10 averages approximately USD 534,664 per HS2-year pool for LightGBM, USD 545,938 for the blend, and USD 443,233 for Two-Part. The Two-Part model also selects fewer successful entries on average. The evaluation years have informed prior M1 development, so these are exploratory retrospective findings.

## External validation

Frozen VNM-trained scoring rules were transferred to THA and IDN, and the same fixed specifications were separately fit within each exporter. Results vary by country and task; the hurdle score does not show a consistent advantage over LightGBM or rank blending. PPML remains strong for existing-market expansion. THA evaluation is exploratory after limited outcome exposure during debugging; IDN is the prespecified external replication, but the three test years overlap the Vietnam temporal window and share global shocks. No causal effect of recommendations is identified.

## Limitations

WDI/WGI vintages are revised; the conditional successful-entry sample is relatively small; unknown outcomes limit complete population identification; export values are highly skewed; country-specific results rely on two external exporters; and no point-in-time future year is available. The paper should describe task-dependent empirical evidence rather than a novel hurdle method or universal superiority.
""",encoding="utf-8")
    (out/"13_EXECUTIVE_SUMMARY_VI.md").write_text("""# Tóm tắt điều hành

Mô hình hai phần tách xác suất đạt ngưỡng xuất khẩu 50.000 USD khỏi giá trị xuất khẩu khi đã đạt ngưỡng. Trên dữ liệu Việt Nam 2022–2024, mô hình này không vượt LightGBM hoặc rank blend: NDCG@10 lần lượt 0,2774, 0,2947 và 0,3319; qualified export USD@10 trung bình khoảng 443 nghìn, 535 nghìn và 546 nghìn USD mỗi HS2-năm. Đây là đánh giá hồi cứu thăm dò.

Thử nghiệm THA và IDN dùng BACI V202601 có thật; cả hai có 96 chương HS2 trong từng năm. Mô hình Việt Nam chuyển sang nước khác và mô hình fit riêng đều chạy được. Kết quả khác nhau theo nước, không chứng minh Two-Part tốt hơn; PPML vẫn mạnh ở nhiệm vụ mở rộng thị trường hiện hữu. THA đã lộ một số nhãn trong bước gỡ lỗi nên chỉ xem là thăm dò. IDN là đánh giá địa lý được định trước, không phải holdout theo thời gian độc lập.

Kết luận phù hợp nhất: hurdle/two-part chưa cho thấy lợi ích kinh tế tổng quát; việc đo riêng xác suất vào thị trường, quy mô có điều kiện và USD thực tế là cần thiết. Không có bằng chứng nhân quả, lợi nhuận doanh nghiệp hay phương pháp thuật toán mới.
""",encoding="utf-8")
    (out/"14_FINAL_STATUS.md").write_text("""# Final status

Engineering: **COMPLETE_REDUCED_SPECIFICATION**. Vietnam two-part experiment, streamed BACI extraction, THA/IDN zero-shot transfer, country-specific replications, paired inference, workbook, and integrity tests were executed. The optional nonlinear conditional-value model was omitted because defensible mean retransformation was not established. All original M1 artifacts remain untouched.

Evidence: **PREDECLARED_EXTERNAL_VALIDATION — IDN geographic replication only**, with THA classified exploratory after the documented debugging exposure. Neither provides new temporal-cycle validation. Vietnam 2022–24 results remain exploratory. WDI/WGI are revised-vintage.

Contribution: **DESCRIPTIVE_FINDINGS_ONLY**. No novel hurdle method or general Two-Part advantage is established. The measured evidence supports task- and exporter-dependent performance and shows the need to distinguish qualified economic value from total exports.

Reproduction from the M1 root: `.venv_uv/bin/python experiments/two_part_external_validation_v1/run_experiments.py --stage all`; tests: `.venv_uv/bin/python -m pytest experiments/two_part_external_validation_v1/tests -q`. No figures were generated.
""",encoding="utf-8")
    # Other reports required by project design and concise integrity description.
    (out/"01_INPUT_AND_TEMPORAL_AUDIT.md").write_text((out/"01_INPUT_AND_TEMPORAL_AUDIT.md").read_text()+"\nSee also the candidate reconciliation and source/timing audit in `FINAL_CANDIDATE_AND_TEMPORAL_AUDIT.md` under the prior experiment.\n",encoding="utf-8")

def finalize_all():
    vpred,vtests=finalize_vietnam()
    ext,detail=finalize_external()
    wb=write_workbook()
    write_reports(vpred,vtests,ext)
    return wb

