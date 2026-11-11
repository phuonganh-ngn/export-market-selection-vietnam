from __future__ import annotations

import traceback

import joblib
import numpy as np
import pandas as pd

from common import CFG, path
from experiment1 import run as run_e1
from hs4_robustness import run as run_hs4
from market_discovery import run as run_e2
from ml_models import MODEL_NAMES, fit_regressor
from statistics import forecast_metrics, ndcg_at_k, precision_at_k
from temporal_validation import split_outer


def _status(check, status, detail, artifact=""):
    return {"check":check,"status":status,"detail":detail,"artifact":artifact}


def _hs2_specific(panel:pd.DataFrame,e1:pd.DataFrame,best:str,lag:int,quick:bool)->pd.DataFrame:
    trainall,testall=split_outer(panel,2024,lag)
    params=joblib.load(path("results_dir")/"models"/f"{best.lower()}_lag{lag}_2024_seed{CFG['study']['seeds'][0]}.joblib")["params"]
    pooled=e1.loc[e1.evaluation_year.eq(2024)&e1.model.eq(best)].groupby(["hs2","destination_iso3"],as_index=False).predicted_usd.mean()
    pooled=pooled.merge(testall[["hs2","destination_iso3","export_value_usd"]],on=["hs2","destination_iso3"],how="inner",validate="one_to_one")
    seed=CFG["study"]["seeds"][0]
    rows=[]
    for hs2code,dtest in testall.groupby("hs2",sort=False):
        dtrain=trainall.loc[trainall.hs2.eq(hs2code)]
        common=pooled.loc[pooled.hs2.eq(hs2code)]
        if len(dtrain)<300 or len(dtest)<10:
            rows.append({"hs2":hs2code,"model":best,"pooled_n":len(dtest),"hs2_specific_n":len(dtrain),"status":"insufficient observations"})
            continue
        try:
            bundle=fit_regressor(best,dtrain,params,seed,lag)
            pred=bundle.predict(dtest)
            pooled_common=dtest[["destination_iso3","export_value_usd"]].merge(common[["destination_iso3","predicted_usd"]],on="destination_iso3",how="inner",validate="one_to_one")
            pooled_m=forecast_metrics(pooled_common.export_value_usd,pooled_common.predicted_usd)
            specific_m=forecast_metrics(dtest.export_value_usd,pred)
            rows.extend([{"hs2":hs2code,"model":best,"estimator":"pooled","n_train":len(trainall),"n_test":len(pooled_common),**pooled_m,"status":"completed"},
                         {"hs2":hs2code,"model":best,"estimator":"HS2-specific","n_train":len(dtrain),"n_test":len(dtest),**specific_m,"status":"completed"}])
        except Exception as exc:
            rows.append({"hs2":hs2code,"model":best,"status":f"failed: {type(exc).__name__}: {exc}"})
    result=pd.DataFrame(rows)
    result.to_csv(path("results_dir")/"tables"/"R2_pooled_vs_hs2_specific.csv",index=False)
    return result


def _country_groups(panel:pd.DataFrame,preds:pd.DataFrame)->pd.DataFrame:
    geo=panel[["destination_iso3","continent"]].drop_duplicates("destination_iso3") if "continent" in panel else pd.DataFrame()
    if geo.empty:
        return pd.DataFrame()
    d=preds.loc[preds.evaluation_year.isin(CFG["study"]["evaluation_years"])].merge(geo,on="destination_iso3",how="left",validate="many_to_one")
    rows=[]
    for (model,continent),g in d.groupby(["model","continent"],dropna=False):
        rows.append({"model":model,"country_group":continent,**forecast_metrics(g.actual_usd,g.predicted_usd)})
    result=pd.DataFrame(rows)
    result.to_csv(path("results_dir")/"tables"/"R9_country_group_performance.csv",index=False)
    return result


def _exclude_dominant(panel:pd.DataFrame,preds:pd.DataFrame,lag:int)->pd.DataFrame:
    rows=[]
    for year in CFG["study"]["evaluation_years"]:
        train,test=split_outer(panel,int(year),lag)
        dominant=set(train.groupby("destination_iso3").historical_export_value.sum().nlargest(5).index)
        d=preds.loc[preds.evaluation_year.eq(year)]
        for model,g in d.groupby("model"):
            rows.append({"model":model,"evaluation_year":year,"sample":"all candidates",**forecast_metrics(g.actual_usd,g.predicted_usd)})
            excl=g.loc[~g.destination_iso3.isin(dominant)]
            rows.append({"model":model,"evaluation_year":year,"sample":"excluding top 5 historically dominant destinations",**forecast_metrics(excl.actual_usd,excl.predicted_usd),
                         "excluded_destinations":",".join(sorted(dominant))})
    result=pd.DataFrame(rows)
    result.to_csv(path("results_dir")/"tables"/"R12_excluding_dominant_destinations.csv",index=False)
    return result


def _alternative_eligibility(panel:pd.DataFrame,rankings:pd.DataFrame,lag:int)->pd.DataFrame:
    # HS2-specific historical median import demand is calculated only from each evaluation origin's training set.
    rows=[]
    for year in CFG["study"]["evaluation_years"]:
        train,test=split_outer(panel,int(year),lag)
        med=train.groupby("hs2").destination_import_demand.median()
        thresholds=test.hs2.map(med)
        eligible=test.loc[test.destination_import_demand.ge(thresholds)].copy()
        cand=rankings.loc[rankings.evaluation_year.eq(year)&rankings.ranking_task.eq("incremental")].copy()
        cand["hs2"] = pd.to_numeric(cand.hs2, errors="coerce").astype("Int64").astype(str).str.zfill(2)
        eligible["hs2"] = pd.to_numeric(eligible.hs2, errors="coerce").astype("Int64").astype(str).str.zfill(2)
        cand=cand.merge(eligible[["hs2","destination_iso3","destination_import_demand"]],on=["hs2","destination_iso3"],how="inner",validate="many_to_one")
        for model,g in cand.groupby("ranking_model"):
            score=g.opportunity_score.fillna(-np.inf)
            g=g.assign(_score=score).sort_values("_score",ascending=False)
            y=g.entry_success.fillna(False).astype(int).to_numpy()
            grade=np.minimum(4,np.floor(np.log10(1+g.realized_incremental_export_usd.fillna(0).to_numpy()/CFG["study"]["primary_entry_threshold_usd"]))).astype(int)
            rows.append({"evaluation_year":year,"ranking_model":model,"eligibility":"demand >= training-period HS2 median",
                "candidate_markets":len(g),"macro_precision_at_10":g.groupby("hs2").apply(lambda x:precision_at_k(x.sort_values("_score",ascending=False).entry_success.fillna(False).astype(int).to_numpy(),10),include_groups=False).mean(),
                "macro_NDCG_at_10":g.groupby("hs2").apply(lambda x:ndcg_at_k(np.minimum(4,np.floor(np.log10(1+x.sort_values("_score",ascending=False).realized_incremental_export_usd.fillna(0).to_numpy()/CFG["study"]["primary_entry_threshold_usd"]))).astype(int),10),include_groups=False).mean(),"positive_share":float(y.mean()) if len(y) else np.nan,
                "threshold_source":"median of training-period import demand by HS2"})
    result=pd.DataFrame(rows)
    result.to_csv(path("results_dir")/"tables"/"R14_alternative_eligibility.csv",index=False)
    return result


def run(lag:int|None=None,quick:bool=False)->pd.DataFrame:
    lag=int(lag or CFG["study"]["primary_lag"])
    status=[]
    panel=pd.read_parquet(path("processed_dir")/f"panel_lag{lag}.parquet")
    e1=pd.read_parquet(path("results_dir")/"predictions"/f"E1_predictions_lag{lag}.parquet")
    overall=pd.read_csv(path("results_dir")/"E1_forecasting_overall.csv")
    best=str(overall.loc[overall.model.isin(MODEL_NAMES)].sort_values("mae_usd").iloc[0].model)
    # R1: rerun the same outer experiments with the one-year retrospective information lag.
    try:
        lag1=run_e1(lag=1,quick=quick)
        primary=overall.copy(); primary["information_lag"]=2
        lag1=lag1.copy(); lag1["information_lag"]=1
        pd.concat([primary,lag1],ignore_index=True).to_csv(path("results_dir")/"tables"/"R1_lag_sensitivity.csv",index=False)
        status.append(_status("R1 two-year vs one-year lag","completed","Matched model evaluation under both retrospective lags; one-year outputs are not real-time validated.","results/tables/R1_lag_sensitivity.csv"))
    except Exception as exc:
        status.append(_status("R1 two-year vs one-year lag","failed",f"{type(exc).__name__}: {exc}"))
    try:
        specific=_hs2_specific(panel,e1,best,lag,quick)
        state="partial" if specific.status.astype(str).str.startswith("failed").any() or not specific.status.eq("completed").any() else "completed"
        status.append(_status("R2 pooled vs HS2-specific",state,"Best pooled ML family compared with chapter-specific fits for chapters meeting data thresholds; ineligible chapters are documented.","results/tables/R2_pooled_vs_hs2_specific.csv"))
    except Exception as exc:
        status.append(_status("R2 pooled vs HS2-specific","failed",f"{type(exc).__name__}: {exc}"))
    try:
        hs4=run_hs4(lag,quick)
        state="partial" if hs4.status.astype(str).str.startswith("failed").any() or not hs4.status.eq("completed").any() else "completed"
        status.append(_status("R3 HS4 robustness",state,"HS4 subset selected from pre-test destination-year coverage across economically diverse HS sections.","results/HS4_robustness.csv"))
    except Exception as exc:
        status.append(_status("R3 HS4 robustness","failed",f"{type(exc).__name__}: {exc}"))
    try:
        ab=pd.read_csv(path("results_dir")/"E4_feature_ablation.csv")
        for check,group in [("R4 governance ablation","governance"),("R5 digitalization ablation","digitalization"),("R6 historical trade ablation","historical_trade"),("R7 import demand ablation","trade_demand")]:
            passed=ab.dropped_group.eq(group).any()
            row=ab.loc[ab.dropped_group.eq(group)]
            detail="completed; see paired full vs ablated MAE" if passed else "not computed"
            if passed and not ab.loc[ab.ablation.eq("full"),"mae_usd"].empty:
                base=float(ab.loc[ab.ablation.eq("full"),"mae_usd"].iloc[0]); alt=float(row.mae_usd.iloc[0]); detail+=f"; MAE change USD {alt-base:,.2f}"
            status.append(_status(check,"completed" if passed else "not_run",detail,"results/E4_feature_ablation.csv" if passed else ""))
    except Exception as exc:
        for check in ["R4 governance ablation","R5 digitalization ablation","R6 historical trade ablation","R7 import demand ablation"]:
            status.append(_status(check,"failed",f"{type(exc).__name__}: {exc}"))
    try:
        threshold_file=path("results_dir")/"tables"/"E2_threshold_selection.csv"
        counts=pd.read_csv(threshold_file)
        primary=int(counts.loc[counts.primary_selected,"threshold_usd"].iloc[0])
        alt_frames=[]
        for threshold in counts.threshold_usd.astype(int).unique():
            if threshold==primary:
                continue
            alt=run_e2(lag=lag,quick=quick,threshold_override=int(threshold))
            if not alt.empty:
                alt=alt.copy();alt["threshold_usd"]=int(threshold);alt_frames.append(alt)
        if alt_frames:
            pd.concat(alt_frames,ignore_index=True).to_csv(path("results_dir")/"E2_threshold_robustness.csv",index=False)
        complete=len(alt_frames)==max(0,counts.threshold_usd.nunique()-1)
        status.append(_status("R8 entry thresholds","completed" if complete else "partial",
            "Full discovery models refit at every alternative threshold; development/test event counts are recorded separately.",
            "results/E2_threshold_robustness.csv" if complete else "results/tables/E2_threshold_selection.csv"))
    except Exception as exc:
        status.append(_status("R8 entry thresholds","not_run",str(exc)))
    try:
        country=_country_groups(panel,e1)
        status.append(_status("R9 country groups","completed" if not country.empty else "not_run","Forecast metrics by CEPII continent group.","results/tables/R9_country_group_performance.csv" if not country.empty else ""))
    except Exception as exc:
        status.append(_status("R9 country groups","failed",str(exc)))
    seed_rows=[]
    for (model,seed),g in e1.loc[e1.model.isin(MODEL_NAMES)].groupby(["model","seed"]):
        seed_rows.append({"model":model,"seed":seed,**forecast_metrics(g.actual_usd,g.predicted_usd)})
    pd.DataFrame(seed_rows).to_csv(path("results_dir")/"seed_stability.csv",index=False)
    status.append(_status("R10 model seeds","completed","Per-seed out-of-time forecast scores retained.","results/seed_stability.csv"))
    stab=path("results_dir")/"tables"/"E3_ranking_stability.csv"
    status.append(_status("R11 ranking stability","completed" if stab.exists() else "not_run","Pairwise cross-year rank correlations and Top-10 overlap by HS2.",str(stab.relative_to(path("results_dir").parent)) if stab.exists() else ""))
    try:
        _exclude_dominant(panel,e1,lag)
        status.append(_status("R12 dominant destination exclusion","completed","Top five destinations determined from each training window only.","results/tables/R12_excluding_dominant_destinations.csv"))
    except Exception as exc:
        status.append(_status("R12 dominant destination exclusion","failed",str(exc)))
    try:
        e3=pd.read_csv(path("results_dir")/"E3_ranking_by_hs2.csv")
        e3=e3.loc[e3.ranking_task.eq("incremental")]
        best_forecast=str(overall.loc[overall.model.isin(MODEL_NAMES)].sort_values("mae_usd").iloc[0].model)
        best_rank=e3.groupby("ranking_model").NDCG_at_10.mean().sort_values(ascending=False).index[0]
        same=best_forecast.lower() in best_rank.lower()
        status.append(_status("R13 forecasting vs market selection","completed",f"Lowest-MAE ML model={best_forecast}; highest macro NDCG strategy={best_rank}; same model family={same}.","results/E1_forecasting_overall.csv; results/E3_ranking_by_hs2.csv"))
    except Exception as exc:
        status.append(_status("R13 forecasting vs market selection","failed",str(exc)))
    try:
        e3all=pd.read_csv(path("results_dir")/"E3_market_ranking.csv")
        alt=_alternative_eligibility(panel,e3all,lag)
        status.append(_status("R14 alternative eligibility","completed" if not alt.empty else "not_run","Candidate demand floor is the HS2-specific median estimated from that evaluation window's training period.","results/tables/R14_alternative_eligibility.csv"))
    except Exception as exc:
        status.append(_status("R14 alternative eligibility","failed",str(exc)))
    summary=pd.DataFrame(status)
    summary.to_csv(path("results_dir")/"robustness_summary.csv",index=False)
    return summary


if __name__=="__main__":
    print(run().to_string(index=False))
