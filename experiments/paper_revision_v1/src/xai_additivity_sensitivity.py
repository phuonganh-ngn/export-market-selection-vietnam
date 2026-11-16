import numpy as np, pandas as pd, shap, warnings
warnings.filterwarnings("ignore")
import xai_decision as X
from revision_lib import KEY, PRED_DIR, RES_DIR, TA, load_stored_candidates, zfill_hs2
from experiments.explainable_reliability_gate_v1.src.task_audit import load_and_freeze
panel,_=load_and_freeze(write=False); panel=zfill_hs2(panel)
stored=load_stored_candidates(); stored=stored[stored.task==TA]
refit=zfill_hs2(pd.read_parquet(PRED_DIR/"lgbm_refit_scores.parquet"))
stored=stored.drop(columns="lightgbm_score").merge(refit[KEY+["lightgbm_score"]],on=KEY,validate="one_to_one")
feats=[c for c in panel.columns if c not in stored.columns or c in ("destination_iso3","hs2","outcome_year")]
rows=[]; imp=[]
for year in X.YEARS:
    cand=stored[stored.outcome_year==year].merge(panel[feats],on=["destination_iso3","hs2","outcome_year"],how="left",validate="one_to_one").reset_index(drop=True)
    cand["lgb_rank"]=0; cand["hyb_rank"]=0
    for _,g in cand.groupby("hs2"):
        iso=g.destination_iso3.to_numpy()
        cand.loc[g.index,"lgb_rank"]=X.pool_rank(g.lightgbm_score.to_numpy(),iso)
        cand.loc[g.index,"hyb_rank"]=X.hybrid_rank(g.ppml_score.to_numpy(),g.lightgbm_score.to_numpy(),iso)
    expl=cand[(cand.lgb_rank<=20)|(cand.hyb_rank<=20)].copy()
    bg=cand.sample(min(100,len(cand)),random_state=year)
    models=X.load_lgbm(year); meta=None
    errs=np.zeros((len(expl),len(models))); gsum=None
    for j,(s,m) in enumerate(zip(X.SEEDS,models)):
        cols=m.numeric_features+m.categorical_features
        xe=X.dense(m.preprocessor.transform(expl[cols])).astype(np.float32).astype(np.float64)
        xb=X.dense(m.preprocessor.transform(bg[cols])).astype(np.float32).astype(np.float64)
        ex=shap.TreeExplainer(m.estimator,data=xb,feature_perturbation="interventional",model_output="raw")
        sv=ex.shap_values(xe,check_additivity=False); sv=sv[1] if isinstance(sv,list) else sv
        if sv.ndim==3: sv=sv[:,:,1]
        ev=np.atleast_1d(ex.expected_value)[-1]; margin=m.estimator.predict(xe,raw_score=True)
        errs[:,j]=np.abs(sv.sum(1)+ev-margin)
        if meta is None: meta=X.column_meta(m.preprocessor.get_feature_names_out())
        ga=pd.DataFrame(np.abs(sv),columns=meta.group).T.groupby(level=0).sum().T
        gsum=ga if gsum is None else gsum+ga
    gmean=gsum/len(models)
    e=expl[["hs2","destination_iso3","lgb_rank","hyb_rank","lightgbm_score","entry_success"]].copy()
    e["year"]=year; e["max_err"]=errs.max(1); e["n_seeds_err"]=(errs>1e-3).sum(1)
    rows.append(e)
    flag=(e.max_err>1e-3).to_numpy()
    for lab,mask in [("all",np.ones(len(e),bool)),("excl_flagged",~flag)]:
        s=gmean[mask].mean(); s=s/s.sum()
        for gname,v in s.items(): imp.append({"year":year,"sample":lab,"group":gname,"share":v})
r=pd.concat(rows)
OUT=RES_DIR/"xai"; r.to_csv(OUT/"additivity_rows.csv",index=False)
f=r[r.max_err>1e-3]
print("flagged rows:",len(f),"of",len(r)); print(f.sort_values("max_err",ascending=False).to_string(index=False))
print("flagged in LGBM top10:",(f.lgb_rank<=10).sum()," in hybrid top10:",(f.hyb_rank<=10).sum(), " ranks 8-13 (lgb):",f.lgb_rank.between(8,13).sum())
im=pd.DataFrame(imp); im.to_csv(OUT/"additivity_sensitivity_importance.csv",index=False)
p=im.pivot_table(index=["year","group"],columns="sample",values="share")
p["diff"]=p.excl_flagged-p["all"]; print("max abs share change:",p["diff"].abs().max())
print(p.loc[2019].round(4).to_string())
# faithfulness and swaps excluding flagged destinations
fk=set(zip(f.year,f.hs2.astype(str).str.zfill(2),f.destination_iso3))
fa=pd.read_csv(OUT/"faithfulness_rows.csv"); fa["hs2"]=fa.hs2.astype(str).str.zfill(2)
fa["flag"]=[ (y,h,d) in fk for y,h,d in zip(fa.year,fa.hs2,fa.destination)]
print("faithfulness rows flagged:",fa.flag.sum())
print(pd.concat({"all":fa.groupby("kind").left_top10.mean(),"excl":fa[~fa.flag].groupby("kind").left_top10.mean()},axis=1).round(4).to_string())
