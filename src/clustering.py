from __future__ import annotations

import itertools

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from common import CFG, path
from ml_models import MODEL_NAMES, fit_regressor
from market_discovery import add_entry_targets
from statistics import ndcg_at_k, forecast_metrics
from temporal_validation import split_outer


CLUSTER_FEATURES = ["log_gdp", "log_gdp_per_capita", "log_population", "gdp_growth", "urban_population_share",
                    "internet_usage", "government_effectiveness", "regulatory_quality", "rule_of_law", "political_stability",
                    "log_destination_import_demand"]


def _destination_characteristics(train: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CLUSTER_FEATURES if c in train.columns]
    return train.groupby(["destination_iso3", "destination_name"], as_index=False)[cols].mean()


def _fit_segments(train: pd.DataFrame, test: pd.DataFrame, seed: int):
    dest = _destination_characteristics(train)
    cols = [c for c in CLUSTER_FEATURES if c in dest.columns]
    x = dest[cols].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    xi = imputer.fit_transform(x)
    scaler = StandardScaler().fit(xi)
    xs = scaler.transform(xi)
    max_k = max(2, min(6, len(dest)-1))
    candidate_k = list(range(2, max_k+1))
    k_scores=[]
    for k in candidate_k:
        labels=KMeans(n_clusters=k,n_init=10,random_state=seed).fit_predict(xs)
        sil=float(silhouette_score(xs,labels)) if len(np.unique(labels))>1 else np.nan
        k_scores.append((k,sil))
    best_k=max(k_scores,key=lambda p: p[1] if np.isfinite(p[1]) else -np.inf)[0]
    km=KMeans(n_clusters=best_k,n_init=20,random_state=seed).fit(xs)
    dest["kmeans_cluster"]=km.labels_.astype(int)
    gm_candidates=[]
    for k in candidate_k:
        gm=GaussianMixture(n_components=k,random_state=seed,n_init=3,covariance_type="diag",reg_covar=1e-5).fit(xs)
        gm_candidates.append((k,float(gm.bic(xs)),gm))
    best_gmm_k,bic,gmm=min(gm_candidates,key=lambda p:p[1])
    dest["gmm_cluster"]=gmm.predict(xs).astype(int)
    diag={"kmeans_k":best_k,"kmeans_silhouette":dict(k_scores).get(best_k,np.nan),"gmm_k":best_gmm_k,"gmm_bic":bic,
          "n_destinations":len(dest),"n_features":len(cols),"features":",".join(cols)}
    # Cluster choices are estimated on development-period destinations only.
    mapping=dest.set_index("destination_iso3")[["kmeans_cluster","gmm_cluster"]]
    test_map=test["destination_iso3"].map(mapping.kmeans_cluster)
    dest["kmeans_silhouette_selected"] = dict(k_scores).get(best_k,np.nan)
    return dest, test_map, mapping, diag, (imputer, scaler, cols, km, gmm)


def run(lag: int | None = None, quick: bool = False) -> dict:
    lag=int(lag or CFG["study"]["primary_lag"])
    panel=pd.read_parquet(path("processed_dir")/f"panel_lag{lag}.parquet")
    e1=pd.read_parquet(path("results_dir")/"predictions"/f"E1_predictions_lag{lag}.parquet")
    threshold=int(CFG["study"]["primary_entry_threshold_usd"])
    entry_labels=add_entry_targets(panel.loc[panel.outcome_year.isin(CFG["study"]["evaluation_years"])],lag,threshold)
    scores=pd.read_csv(path("results_dir")/"E1_forecasting_overall.csv")
    best=str(scores.loc[scores.model.isin(MODEL_NAMES)].sort_values("mae_usd").iloc[0].model)
    eval_year=2024
    train,test=split_outer(panel,eval_year,lag)
    test=test.merge(entry_labels[["destination_iso3","hs2","outcome_year","entry_eligible"]],
        on=["destination_iso3","hs2","outcome_year"],how="left",validate="one_to_one")
    profiles=[]; clusters=[]; stability=[]; model_comp=[]; silhouette_rows=[]
    temporal_labels={"kmeans_cluster":{},"gmm_cluster":{}}
    for year in CFG["study"]["evaluation_years"]:
        tr,te=split_outer(panel,int(year),lag)
        dest_year,_,map_year,diag_year,artifacts_year=_fit_segments(tr,te,11)
        imputer_year,scaler_year,feature_cols,_,gmm_year=artifacts_year
        xs_year=scaler_year.transform(imputer_year.transform(dest_year[feature_cols].replace([np.inf,-np.inf],np.nan)))
        for algorithm in ["kmeans_cluster","gmm_cluster"]:
            labels=map_year[algorithm].rename("cluster").reset_index()
            temporal_labels[algorithm][int(year)]=labels
            clusters.extend([{"destination_iso3":row.destination_iso3,"destination_name":row.destination_name,
                "evaluation_year":int(year),"algorithm":algorithm.replace("_cluster",""),"cluster":int(row.cluster),
                "assignment_information_cutoff":int(tr.outcome_year.max()),"status":"training-period assignment"}
                for row in dest_year.merge(labels,on="destination_iso3").itertuples(index=False)])
            lab=dest_year[algorithm].to_numpy()
            sil=float(silhouette_score(xs_year,lab)) if len(np.unique(lab))>1 else np.nan
            silhouette_rows.append({"algorithm":"KMeans" if algorithm=="kmeans_cluster" else "GaussianMixture",
                "evaluation_year":int(year),**diag_year,"selected_cluster_silhouette":sil})
    for algorithm,year_maps in temporal_labels.items():
        for (y1,l1),(y2,l2) in itertools.combinations(sorted(year_maps.items()),2):
            common=year_maps[y1].merge(year_maps[y2],on="destination_iso3",suffixes=("_1","_2"))
            stability.append({"algorithm":"KMeans" if algorithm=="kmeans_cluster" else "GaussianMixture","stability_type":"across_outer_years",
                "year_1":y1,"year_2":y2,"paired_destinations":len(common),"adjusted_rand_index":adjusted_rand_score(common.cluster_1,common.cluster_2) if len(common)>=2 else np.nan})
    dest, test_map, mapping, diag, artifacts = _fit_segments(train,test,11)
    pooled=e1.loc[e1.evaluation_year.eq(eval_year)&e1.model.eq(best)].groupby(["destination_iso3","hs2"],as_index=False).predicted_usd.mean()
    test=test.merge(mapping.reset_index(),on="destination_iso3",how="left",validate="many_to_one").merge(pooled,on=["destination_iso3","hs2"],how="left",validate="one_to_one")
    test["pooled_increment"]=(test.predicted_usd-test.historical_export_value).clip(lower=0)
    train=train.merge(mapping.reset_index(),on="destination_iso3",how="left",validate="many_to_one")
    # K-means stability over fixed seeds (same training destinations/features).
    x=dest[CLUSTER_FEATURES].replace([np.inf,-np.inf],np.nan)
    imp=SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True).fit(x)
    xs=StandardScaler().fit_transform(imp.transform(x))
    k=diag["kmeans_k"]
    label_seeds={s:KMeans(n_clusters=k,n_init=10,random_state=int(s)).fit_predict(xs) for s in CFG["study"]["seeds"]}
    for (s1,l1),(s2,l2) in itertools.combinations(label_seeds.items(),2):
        stability.append({"algorithm":"KMeans","stability_type":"across_seeds","k":k,"seed_1":s1,"seed_2":s2,"adjusted_rand_index":adjusted_rand_score(l1,l2)})
    # Profiles and pooled versus cluster-specific predictions.
    params=joblib.load(path("results_dir")/"models"/f"{best.lower()}_lag{lag}_{eval_year}_seed{CFG['study']['seeds'][0]}.joblib")["params"]
    seeds=[CFG["study"]["seeds"][0]] if quick else CFG["study"]["seeds"]
    for algorithm in ["kmeans_cluster","gmm_cluster"]:
        for cid, dest_group in dest.groupby(algorithm):
            ids=set(dest_group.destination_iso3)
            trg=test.loc[test.destination_iso3.isin(ids)].copy()
            trn=train.loc[train.destination_iso3.isin(ids)].copy()
            characteristics={c:float(dest_group[c].mean()) for c in CLUSTER_FEATURES if c in dest_group.columns}
            profile={"algorithm":algorithm.replace("_cluster",""),"cluster":int(cid),"destinations":dest_group.destination_iso3.nunique(),
                     "evaluation_year":eval_year,"members":",".join(sorted(ids)),**characteristics,
                     "mean_realized_incremental_export_usd":float(trg.realized_incremental_export_usd.mean()) if len(trg) else np.nan,
                     "entry_success_rate":float((trg.entry_eligible.fillna(False)&trg.export_value_usd.ge(threshold)).mean()) if len(trg) else np.nan}
            profiles.append(profile)
            # K-means drives the main comparison; GMM receives descriptive/performance summaries.
            if algorithm=="kmeans_cluster" and len(trg):
                pooled_metrics=forecast_metrics(trg.export_value_usd,trg.predicted_usd)
                pooled_ndcg=[]
                for _,rank_d in trg.groupby("hs2",sort=False):
                    rank_d=rank_d.sort_values("pooled_increment",ascending=False)
                    grade=np.minimum(4,np.floor(np.log10(1+rank_d.realized_incremental_export_usd.fillna(0)/CFG["study"]["primary_entry_threshold_usd"]))).astype(int)
                    pooled_ndcg.append(ndcg_at_k(grade,10))
                pooled_ndcg=float(np.nanmean(pooled_ndcg)) if np.isfinite(pooled_ndcg).any() else np.nan
                model_comp.append({"algorithm":"KMeans","cluster":int(cid),"model":"pooled_best_ML","n_train":len(trn),"n_test":len(trg),
                    **pooled_metrics,"NDCG_at_10":pooled_ndcg,"status":"evaluated"})
                if len(trn)>=max(250,int(CFG["study"]["min_training_rows"]/5)):
                    preds=[]
                    for seed in seeds:
                        fitted=fit_regressor(best,trn,params,int(seed),lag)
                        preds.append(fitted.predict(trg))
                    cluster_pred=np.mean(preds,axis=0)
                    metrics=forecast_metrics(trg.export_value_usd,cluster_pred)
                    inc=np.maximum(0,cluster_pred-trg.historical_export_value.to_numpy(float))
                    ranked=trg[["hs2","realized_incremental_export_usd"]].copy(); ranked["score"]=inc
                    cluster_ndcg=[]
                    for _,pool in ranked.groupby("hs2",sort=False):
                        pool=pool.sort_values("score",ascending=False)
                        grades=np.minimum(4,np.floor(np.log10(1+pool.realized_incremental_export_usd.fillna(0).to_numpy()/CFG["study"]["primary_entry_threshold_usd"]))).astype(int)
                        cluster_ndcg.append(ndcg_at_k(grades,10))
                    model_comp.append({"algorithm":"KMeans","cluster":int(cid),"model":"cluster_specific_best_ML","n_train":len(trn),"n_test":len(trg),
                        **metrics,"NDCG_at_10":float(np.nanmean(cluster_ndcg)) if np.isfinite(cluster_ndcg).any() else np.nan,
                        "status":"evaluated; ranking metric macro-averaged within HS2 pools; same parameters as pooled fit"})
                else:
                    model_comp.append({"algorithm":"KMeans","cluster":int(cid),"model":"cluster_specific_best_ML","n_train":len(trn),"n_test":len(trg),"NDCG_at_10":np.nan,"status":"insufficient training rows"})
    pd.DataFrame(clusters).to_csv(path("results_dir")/"E5_clusters.csv",index=False)
    pd.DataFrame(profiles).to_csv(path("results_dir")/"E5_cluster_profiles.csv",index=False)
    pd.DataFrame(stability).to_csv(path("results_dir")/"E5_cluster_stability.csv",index=False)
    pd.DataFrame(model_comp).to_csv(path("results_dir")/"E5_cluster_model_comparison.csv",index=False)
    pd.DataFrame(silhouette_rows).to_csv(path("results_dir")/"tables"/"E5_cluster_fit_diagnostics.csv",index=False)
    return {"best_model":best,"kmeans_k":diag["kmeans_k"],"gmm_k":diag["gmm_k"],"profiles":len(profiles)}


if __name__=="__main__":
    print(run())
