from __future__ import annotations

import traceback

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance

from common import CFG, path
from ml_models import FEATURE_GROUPS, MODEL_NAMES, fit_regressor, tune_regressor
from statistics import forecast_metrics
from temporal_validation import split_outer


def _map_transformed_feature(name: str) -> tuple[str, str]:
    clean = name.split("__", 1)[-1]
    if clean.startswith("missingindicator_"):
        clean = clean.replace("missingindicator_", "", 1) + "_missing_indicator"
    base = clean.removesuffix("_missing_indicator")
    for group, features in FEATURE_GROUPS.items():
        if base in features or any(base.startswith(f + "_") for f in features):
            return clean, group
    if clean.startswith("destination_iso3_"):
        return "destination fixed effects", "destination_fixed_effects"
    if clean.startswith("hs2_"):
        return "HS2 fixed effects", "product"
    return clean, "other"


def _sample_oot(panel: pd.DataFrame, lag: int, n: int) -> pd.DataFrame:
    selected = []
    for year in CFG["study"]["evaluation_years"]:
        _, test = split_outer(panel, int(year), lag)
        # Fixed-seed random sampling from the full eligible OOT population, independent of outcomes.
        if len(test) > n:
            test = test.sample(n=n, random_state=20261011)
        selected.append(test)
    return pd.concat(selected, ignore_index=True)


def _ale_numeric(bundle, train: pd.DataFrame, sample: pd.DataFrame, features: list[str], bins: int = 10) -> pd.DataFrame:
    rows=[]
    for feature in features:
        if feature not in train or feature not in sample:
            continue
        train_x = pd.to_numeric(train[feature], errors="coerce").dropna()
        sample_x = pd.to_numeric(sample[feature], errors="coerce")
        if train_x.nunique() < 5:
            continue
        edges = np.unique(np.quantile(train_x, np.linspace(0,1,bins+1)))
        if len(edges) < 4:
            continue
        if edges[0] == edges[-1]:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        bin_ids = np.digitize(sample_x.fillna(train_x.median()), edges[1:-1], right=True)
        diffs, counts, ses = [], [], []
        for j in range(len(edges)-1):
            idx = np.flatnonzero(bin_ids == j)
            if not len(idx):
                diffs.append(np.nan); counts.append(0); ses.append(np.nan); continue
            left = np.nextafter(edges[j], edges[j+1]) if np.isfinite(edges[j]) else float(train_x.min())
            right = np.nextafter(edges[j+1], edges[j]) if np.isfinite(edges[j+1]) else float(train_x.max())
            low, high = sample.iloc[idx].copy(), sample.iloc[idx].copy()
            low[feature], high[feature] = left, right
            delta = bundle.predict(high) - bundle.predict(low)
            diffs.append(float(np.mean(delta))); counts.append(len(idx)); ses.append(float(np.std(delta, ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else np.nan)
        accumulated=[]; running=0.0
        for delta in diffs:
            running += 0.0 if not np.isfinite(delta) else delta
            accumulated.append(running)
        w=np.asarray(counts,dtype=float)
        center=float(np.average(np.asarray(accumulated),weights=np.maximum(w,1))) if w.sum() else 0.0
        for j,(effect,count,se) in enumerate(zip(accumulated,counts,ses)):
            rows.append({"feature":feature,"interval_lower":edges[j],"interval_upper":edges[j+1],"n_observations":count,
                "ale_effect_centered":effect-center,"uncertainty_se":se,"bins":len(edges)-1})
    return pd.DataFrame(rows)


def run(lag: int | None = None, quick: bool = False) -> dict:
    lag = int(lag or CFG["study"]["primary_lag"])
    panel = pd.read_parquet(path("processed_dir") / f"panel_lag{lag}.parquet")
    overall_path = path("results_dir") / "E1_forecasting_overall.csv"
    if not overall_path.exists():
        raise FileNotFoundError("Experiment 1 results are required for model selection")
    score = pd.read_csv(overall_path)
    ml_score = score.loc[score.model.isin(MODEL_NAMES)].sort_values("mae_usd")
    if ml_score.empty:
        raise RuntimeError("No fitted ML regressor is available for explainability")
    best = str(ml_score.iloc[0].model)
    test_sample = _sample_oot(panel, lag, int(CFG["study"]["shap_sample_size"]))
    shap_rows, raw_shap, perm_rows, ale_frames = [], [], [], []
    failure=[]
    numeric_importance=[]
    for year in CFG["study"]["evaluation_years"]:
        train, _ = split_outer(panel, int(year), lag)
        sample = test_sample.loc[test_sample.outcome_year.eq(year)].copy()
        model_file = path("results_dir") / "models" / f"{best.lower()}_lag{lag}_{year}_seed{CFG['study']['seeds'][0]}.joblib"
        if not model_file.exists():
            failure.append({"evaluation_year":year,"model":best,"error":"saved model bundle missing"})
            continue
        bundle = joblib.load(model_file)["bundle"]
        cats = bundle.categorical_features
        nums = bundle.numeric_features
        x = bundle.preprocessor.transform(sample[nums+cats]).astype(np.float32)
        feat_names = bundle.preprocessor.get_feature_names_out()
        try:
            explainer = shap.TreeExplainer(bundle.estimator)
            vals = explainer.shap_values(x.toarray() if hasattr(x,"toarray") else x)
            if isinstance(vals, list): vals = vals[0]
            vals = np.asarray(vals)
            if vals.ndim == 3: vals = vals[:,:,0]
            imp = np.mean(np.abs(vals),axis=0)
            grouped={}
            for col, value in zip(feat_names, imp):
                feature, group = _map_transformed_feature(str(col))
                grouped[(feature,group)] = grouped.get((feature,group),0.0)+float(value)
            for (feature,group), value in grouped.items():
                shap_rows.append({"model":best,"evaluation_year":year,"feature":feature,"feature_group":group,"mean_abs_shap":value,
                                  "sample_n":len(sample),"sample_rule":"fixed-seed random OOT sample; no outcome stratification"})
            for i, col in enumerate(feat_names):
                raw_shap.extend({"model":best,"evaluation_year":year,"destination_iso3":sample.destination_iso3.iloc[r],"hs2":sample.hs2.iloc[r],
                                 "outcome_year":int(year),"transformed_feature":str(col),"shap_value":float(vals[r,i])} for r in range(len(sample)))
            # Permutation importance is evaluated on the same bounded, randomly selected OOT sample.
            base_pred=bundle.predict(sample)
            base_mae=mean_absolute_error(sample.export_value_usd,base_pred)
            perm_rng=np.random.default_rng(11)
            for col in nums:
                permuted=sample.copy()
                permuted[col]=perm_rng.permutation(permuted[col].to_numpy())
                delta=mean_absolute_error(sample.export_value_usd,bundle.predict(permuted))-base_mae
                perm_rows.append({"model":best,"evaluation_year":year,"feature":col,"importance_delta_mae_usd":delta,"sample_n":len(sample)})
            # ALE intervals are estimated from training-only quantiles and OOT rows.
            train_full,_=split_outer(panel,int(year),lag)
            base_importance={}
            for col,value in zip(feat_names,imp):
                feature,_=_map_transformed_feature(str(col))
                base_importance[feature]=base_importance.get(feature,0.0)+float(value)
            top_numeric=[feature for feature,_ in sorted(base_importance.items(),key=lambda item:item[1],reverse=True) if feature in nums]
            if len(sample)>CFG["study"]["ale_sample_size"]:
                sample_ale=sample.sample(CFG["study"]["ale_sample_size"],random_state=177)
            else: sample_ale=sample
            ale= _ale_numeric(bundle,train_full,sample_ale,top_numeric[:10])
            if not ale.empty:
                ale["model"],ale["evaluation_year"]=best,year
                ale_frames.append(ale)
        except Exception:
            failure.append({"evaluation_year":year,"model":best,"error":traceback.format_exc()})
    shap_df=pd.DataFrame(shap_rows)
    if not shap_df.empty:
        global_df=shap_df.groupby(["feature","feature_group"],as_index=False).agg(mean_abs_shap=("mean_abs_shap","mean"),
            sd_across_years=("mean_abs_shap","std"),evaluation_years=("evaluation_year","nunique"),sample_n=("sample_n","sum"))
        global_df.sort_values("mean_abs_shap",ascending=False).to_csv(path("results_dir")/"E4_global_shap.csv",index=False)
        shap_df.to_csv(path("results_dir")/"E4_shap_by_year.csv",index=False)
        shap_df.to_csv(path("results_dir")/"E4_shap_by_hs2.csv",index=False)
    if raw_shap:
        raw_df=pd.DataFrame(raw_shap)
        raw_df.to_parquet(path("results_dir")/"predictions"/"E4_shap_sample.parquet",index=False)
        hs2_rows=[]
        for (hs2code,year,feature),g in raw_df.groupby(["hs2","evaluation_year","transformed_feature"],sort=False):
            mapped,group=_map_transformed_feature(feature)
            hs2_rows.append({"hs2":hs2code,"evaluation_year":year,"feature":mapped,"feature_group":group,
                "mean_abs_shap":g.shap_value.abs().mean(),"sample_n":g.destination_iso3.nunique()})
        pd.DataFrame(hs2_rows).groupby(["hs2","evaluation_year","feature","feature_group"],as_index=False).agg(
            mean_abs_shap=("mean_abs_shap","sum"),sample_n=("sample_n","max")).to_csv(path("results_dir")/"E4_shap_by_hs2.csv",index=False)
    if perm_rows:
        pd.DataFrame(perm_rows).to_csv(path("results_dir")/"tables"/"E4_permutation_importance.csv",index=False)
    if ale_frames:
        pd.concat(ale_frames,ignore_index=True).to_csv(path("results_dir")/"E4_ale_numeric.csv",index=False)
    # Hold the selected hyperparameters fixed to isolate the predictive information removed by each ablation.
    eval_year=2024
    train,test=split_outer(panel,eval_year,lag)
    full_params=joblib.load(path("results_dir")/"models"/f"{best.lower()}_lag{lag}_{eval_year}_seed{CFG['study']['seeds'][0]}.joblib")["params"]
    ablation_rows=[]
    # Feature-group ablation is an exploratory sensitivity analysis. Keep it to
    # the pre-specified seed 11 to control CPU cost; main E1 metrics use all five.
    seeds=[CFG["study"]["seeds"][0]]
    groups=list(FEATURE_GROUPS.keys())
    base_models=[]
    for seed in seeds:
        bundle=joblib.load(path("results_dir")/"models"/f"{best.lower()}_lag{lag}_{eval_year}_seed{seed}.joblib")["bundle"]
        base_models.append((seed,bundle.predict(test)))
    base_pred=np.mean([p for _,p in base_models],axis=0)
    base_metric=forecast_metrics(test.export_value_usd,base_pred)
    ablation_rows.append({"model":best,"ablation":"full","evaluation_year":eval_year,"dropped_group":"none",**base_metric,"seeds":len(seeds)})
    for group in groups:
        preds=[]
        for seed in seeds:
            bundle=fit_regressor(best,train,full_params,int(seed),lag,drop_groups=[group])
            preds.append(bundle.predict(test))
        m=forecast_metrics(test.export_value_usd,np.mean(preds,axis=0))
        ablation_rows.append({"model":best,"ablation":f"without_{group}","evaluation_year":eval_year,"dropped_group":group,**m,"seeds":len(seeds)})
    ablation=pd.DataFrame(ablation_rows)
    ablation.to_csv(path("results_dir")/"E4_feature_ablation.csv",index=False)
    if failure:
        pd.DataFrame(failure).to_csv(path("results_dir")/"tables"/"E4_failures.csv",index=False)
    if shap_df.empty:
        status=f"Model selected: {best}; SHAP computation failed or returned no values. See E4_failures.csv."
    else:
        status=f"Numerical SHAP and ALE computed for {best} on bounded random OOT samples; explanations are predictive, not causal."
    return {"model":best,"status":status,"sample_size":len(test_sample),"shap_features":int(shap_df.feature.nunique()) if not shap_df.empty else 0}


if __name__ == "__main__":
    print(run())
