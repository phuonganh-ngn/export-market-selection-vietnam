from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import CFG, path
from statistics import holm_adjust


def _block_bootstrap_mae(g:pd.DataFrame,reps:int,seed:int)->tuple[float,float]:
    rng=np.random.default_rng(seed)
    blocks=g.destination_iso3.unique()
    vals=[]
    grouped={b:g.loc[g.destination_iso3.eq(b)] for b in blocks}
    for _ in range(reps):
        sampled=rng.choice(blocks,size=len(blocks),replace=True)
        y=np.concatenate([grouped[b].actual_usd.to_numpy(float) for b in sampled])
        p=np.concatenate([grouped[b].predicted_usd.to_numpy(float) for b in sampled])
        vals.append(float(np.mean(np.abs(y-p))))
    lo,hi=np.quantile(vals,[.025,.975])
    return float(lo),float(hi)


def _paired_destination_diff(d:pd.DataFrame,a:str,b:str,reps:int,seed:int)->dict:
    wide=d.pivot_table(index=["destination_iso3","hs2","evaluation_year"],columns="model",values=["actual_usd","predicted_usd"],aggfunc="first")
    if ("predicted_usd",a) not in wide or ("predicted_usd",b) not in wide:
        return {"difference_mae_a_minus_b":np.nan,"ci_low":np.nan,"ci_high":np.nan,"p_value":np.nan,"n_destinations":0}
    actual=wide[("actual_usd",a)].to_numpy(float)
    pa,pb=wide[("predicted_usd",a)].to_numpy(float),wide[("predicted_usd",b)].to_numpy(float)
    block=wide.index.get_level_values("destination_iso3").to_numpy()
    keep=np.isfinite(actual)&np.isfinite(pa)&np.isfinite(pb)
    actual,pa,pb,block=actual[keep],pa[keep],pb[keep],block[keep]
    blocks=np.unique(block); rng=np.random.default_rng(seed); boots=[]
    for _ in range(reps):
        chosen=rng.choice(blocks,size=len(blocks),replace=True)
        idx=np.concatenate([np.flatnonzero(block==x) for x in chosen])
        boots.append(float(np.mean(np.abs(actual[idx]-pa[idx])-np.abs(actual[idx]-pb[idx]))))
    diff=float(np.mean(np.abs(actual-pa)-np.abs(actual-pb)))
    lo,hi=np.quantile(boots,[.025,.975])
    # Add-one smoothing avoids reporting an impossible exact zero from a finite
    # number of bootstrap replicates. Destination-block Wilcoxon p-values below
    # remain the prespecified inferential test for these paired contrasts.
    n_boot=len(boots)
    p=2*min((int(np.sum(np.asarray(boots)<=0))+1)/(n_boot+1),
            (int(np.sum(np.asarray(boots)>=0))+1)/(n_boot+1))
    block_diffs=[]
    for destination in blocks:
        ix=block==destination
        block_diffs.append(float(np.mean(np.abs(actual[ix]-pa[ix])-np.abs(actual[ix]-pb[ix]))))
    try:
        wp=float(wilcoxon(block_diffs,alternative="two-sided").pvalue) if np.any(np.asarray(block_diffs)!=0) else 1.0
    except Exception: wp=np.nan
    return {"difference_mae_a_minus_b":diff,"ci_low":float(lo),"ci_high":float(hi),"bootstrap_p_value":min(1.0,p),
            "wilcoxon_destination_block_p":wp,"n_destinations":int(len(blocks))}


def _paired_rank_bootstrap(d:pd.DataFrame,a:str,b:str,reps:int,seed:int)->dict:
    x=d.loc[d.ranking_model.isin([a,b])].pivot_table(index=["hs2","evaluation_year"],columns="ranking_model",values="NDCG_at_10",aggfunc="first").dropna()
    if a not in x or b not in x or x.empty:
        return {"difference_ndcg_a_minus_b":np.nan,"ci_low":np.nan,"ci_high":np.nan,"p_value":np.nan,"n_hs2":0}
    hs2=x.index.get_level_values("hs2").unique().to_numpy(); rng=np.random.default_rng(seed)
    boots=[]
    for _ in range(reps):
        sampled=rng.choice(hs2,size=len(hs2),replace=True)
        vals=[]
        for product in sampled:
            z=x.loc[x.index.get_level_values("hs2")==product]
            vals.extend((z[a]-z[b]).dropna().to_list())
        boots.append(float(np.mean(vals)) if vals else np.nan)
    diff=float((x[a]-x[b]).mean()); lo,hi=np.nanquantile(boots,[.025,.975])
    by_hs=x.groupby(level="hs2").apply(lambda z:float((z[a]-z[b]).mean()))
    try: p=float(wilcoxon(by_hs.to_numpy(),alternative="two-sided").pvalue) if (by_hs!=0).any() else 1.0
    except Exception: p=np.nan
    return {"difference_ndcg_a_minus_b":diff,"ci_low":float(lo),"ci_high":float(hi),"p_value":p,"n_hs2":int(len(hs2))}


def run()->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    preds=pd.read_parquet(path("results_dir")/"predictions"/"E1_predictions_lag2.parquet")
    d=preds.groupby(["model","evaluation_year","destination_iso3","hs2"],as_index=False).agg(
        actual_usd=("actual_usd","first"),predicted_usd=("predicted_usd","mean"))
    reps=int(CFG["study"]["bootstrap_replicates"])
    confidence=[]; tests=[]
    for model,g in d.groupby("model"):
        lo,hi=_block_bootstrap_mae(g,reps,11)
        confidence.append({"family":"forecast absolute metric","model":model,"metric":"MAE USD","estimate":float(np.mean(np.abs(g.actual_usd-g.predicted_usd))),
                          "ci_low":lo,"ci_high":hi,"method":"destination-block bootstrap","n_destinations":g.destination_iso3.nunique()})
    forecast_pairs=[("Gravity_PPML","XGBoost"),("Gravity_PPML","LightGBM"),("RandomForest","XGBoost")]
    forecast_p=[]
    for a,b in forecast_pairs:
        res=_paired_destination_diff(d,a,b,reps,11)
        tests.append({"family":"confirmatory forecasting comparisons","model_a":a,"model_b":b,"metric":"MAE USD (negative favors model A)",**res,
                      "interpretation_note":"Destination blocks retain within-destination product and year dependence; three outer years limit temporal inference."})
        forecast_p.append(res["wilcoxon_destination_block_p"])
    for i,row in enumerate(tests):
        if row["family"]=="confirmatory forecasting comparisons":
            pass
    forecast_indices=[i for i,r in enumerate(tests) if r["family"]=="confirmatory forecasting comparisons"]
    adjusted=holm_adjust([tests[i].get("wilcoxon_destination_block_p",np.nan) for i in forecast_indices])
    for i,p in zip(forecast_indices,adjusted): tests[i]["holm_adjusted_p_value"]=p
    ranking=pd.read_csv(path("results_dir")/"E3_ranking_by_hs2.csv")
    ranking_decision=ranking.loc[ranking.ranking_task.eq("incremental")].copy()
    e1_overall=pd.read_csv(path("results_dir")/"E1_forecasting_overall.csv")
    best_forecast=str(e1_overall.loc[e1_overall.model.isin(["RandomForest","XGBoost","LightGBM"])].sort_values("mae_usd").iloc[0].model)
    macro=ranking_decision.groupby("ranking_model").NDCG_at_10.mean().sort_values(ascending=False)
    best_ranking=str(macro.index[0]) if len(macro) else ""
    matched_forecast=f"{best_forecast}_increment"
    if matched_forecast in set(ranking_decision.ranking_model):
        rr=_paired_rank_bootstrap(ranking_decision,matched_forecast,best_ranking,reps,23)
        tests.append({"family":"decision-oriented ranking comparison","model_a":matched_forecast,"model_b":best_ranking,
            "metric":"NDCG@10 (positive favors forecast winner)","difference":rr["difference_ndcg_a_minus_b"],"ci_low":rr["ci_low"],"ci_high":rr["ci_high"],
            "p_value":rr["p_value"],"n_hs2":rr["n_hs2"],"method":"paired HS2 block bootstrap; product-year sets matched",
            "interpretation_note":"Only three out-of-time years; uncertainty across years is not precisely estimated."})
        confidence.append({"family":"decision-oriented ranking comparison","model":f"{matched_forecast} minus {best_ranking}","metric":"NDCG@10",
            "estimate":rr["difference_ndcg_a_minus_b"],"ci_low":rr["ci_low"],"ci_high":rr["ci_high"],"method":"paired HS2 block bootstrap","n_destinations":np.nan})
        tests[-1]["holm_adjusted_p_value"]=holm_adjust([tests[-1]["p_value"]])[0]
    ci=pd.DataFrame(confidence); test_df=pd.DataFrame(tests)
    comp_forecast=e1_overall.copy(); comp_forecast["comparison_type"]="forecasting"
    rank_summary=ranking.groupby("ranking_model",as_index=False).agg(macro_precision_at_10=("Precision_at_10","mean"),macro_ndcg_at_10=("NDCG_at_10","mean"),
        macro_increment_top10_usd=("realized_incremental_exports_top10_usd","mean"))
    rank_summary.to_csv(path("results_dir")/"tables"/"E3_strategy_summary.csv",index=False)
    comp_rank=rank_summary.rename(columns={"ranking_model":"model"}); comp_rank["comparison_type"]="market_ranking"
    comp=pd.concat([comp_forecast,comp_rank],ignore_index=True,sort=False)
    comp.to_csv(path("results_dir")/"model_comparison.csv",index=False)
    ci.to_csv(path("results_dir")/"confidence_intervals.csv",index=False)
    test_df.to_csv(path("results_dir")/"hypothesis_tests.csv",index=False)
    if not (path("results_dir")/"seed_stability.csv").exists():
        seed_rows=[]
        for (model,seed),g in preds.loc[preds.model.isin(["RandomForest","XGBoost","LightGBM"])].groupby(["model","seed"]):
            seed_rows.append({"model":model,"seed":seed,"mae_usd":float(np.mean(np.abs(g.actual_usd-g.predicted_usd))),"n":len(g)})
        pd.DataFrame(seed_rows).to_csv(path("results_dir")/"seed_stability.csv",index=False)
    text=["# Statistical inference", "", f"Destination-block bootstrap replicates: {reps}.",
        "Forecast comparisons are paired on destination–HS2–evaluation-year rows and use destinations as resampling blocks. Holm correction is applied within the three prespecified forecasting contrasts (PPML–XGBoost, PPML–LightGBM, and Random Forest–XGBoost).",
        "Ranking comparison pairs HS2 × evaluation-year pools and resamples HS2 blocks. With only three outer evaluation years, temporal uncertainty remains weakly identified; intervals and p-values should be read cautiously.",
        "The estimand is predictive performance in a retrospective observational backtest. It does not identify causal economic benefits from adopting rankings.", "",
        "## Forecasting comparisons", "", test_df.to_markdown(index=False) if not test_df.empty else "No eligible paired comparisons.", ""]
    (path("reports_dir")/"STATISTICAL_INFERENCE.md").write_text("\n".join(text))
    return comp,ci,test_df


if __name__=="__main__":
    run()
