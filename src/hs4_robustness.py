from __future__ import annotations

import numpy as np
import pandas as pd

from common import CFG, path
from country_mapping import load_baci_metadata
from features import build_geography, build_macro_features
from gravity_ppml import fit_ppml
from ml_models import MODEL_NAMES, fit_regressor, tune_regressor
from statistics import forecast_metrics


HS_SECTIONS = [(1,5),(6,14),(15,15),(16,24),(25,27),(28,38),(39,40),(41,43),(44,46),(47,49),(50,63),
               (64,67),(68,70),(71,71),(72,83),(84,85),(86,89),(90,92),(93,93),(94,96),(97,97)]


def section(chapter: str) -> str:
    n=int(chapter)
    for ix,(lo,hi) in enumerate(HS_SECTIONS,1):
        if lo<=n<=hi: return f"section_{ix:02d}"
    return "unassigned"


def select_subset(trade4: pd.DataFrame) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    # Chapter and HS4 choices use historical coverage only (2012–2019), before every outer evaluation year.
    dev=trade4.loc[trade4.year.le(min(2019,CFG["study"]["evaluation_years"][0]-CFG["study"]["primary_lag"]))].copy()
    chapter_stats=dev.assign(hs2=dev.hs4.str[:2]).groupby("hs2",as_index=False).agg(
        hs4_codes=("hs4","nunique"),historical_destination_year_markets=("baci_importer_id","size"),
        destinations=("baci_importer_id","nunique"),pretest_exports_usd=("vietnam_exports_usd","sum"))
    chapter_stats["section"]=chapter_stats.hs2.map(section)
    chapter_stats=chapter_stats.sort_values(["historical_destination_year_markets","destinations","hs4_codes"],ascending=False)
    section_top=chapter_stats.sort_values(["section","historical_destination_year_markets"],ascending=[True,False]).groupby("section",as_index=False).first()
    chapter_selected=section_top.sort_values("historical_destination_year_markets",ascending=False).head(int(CFG["study"]["hs4_max_chapters"]))
    selected_chapters=chapter_selected.hs2.tolist()
    hs4_stats=dev.loc[dev.hs4.str[:2].isin(selected_chapters)].groupby("hs4",as_index=False).agg(
        historical_destination_year_markets=("baci_importer_id","size"),destinations=("baci_importer_id","nunique"),
        pretest_exports_usd=("vietnam_exports_usd","sum"))
    selected=[]
    for chapter in selected_chapters:
        selected.extend(hs4_stats.loc[hs4_stats.hs4.str.startswith(chapter)].sort_values(
            ["historical_destination_year_markets","destinations"],ascending=False).head(3).hs4.tolist())
    selection=pd.DataFrame({"hs4":selected})
    selection["hs2"]=selection.hs4.str[:2]
    selection["selection_rule"]="Top 3 HS4 by 2012–2019 destination-year BACI import coverage within one highest-coverage HS2 per represented HS section; no test outcomes used"
    selection=selection.merge(hs4_stats,on="hs4",how="left",validate="one_to_one")
    return selected,selection,chapter_stats


def make_hs4_panel(selected: list[str],lag:int) -> pd.DataFrame:
    trade=pd.read_parquet(path("interim_dir")/"baci_hs4_trade.parquet")
    cmap,_=load_baci_metadata()
    trade["baci_importer_id"]=trade.baci_importer_id.astype(int)
    trade=trade.merge(cmap[["baci_id","iso3","country_name"]],left_on="baci_importer_id",right_on="baci_id",how="left",validate="many_to_one")
    trade=trade.loc[trade.iso3.str.fullmatch(r"[A-Z]{3}",na=False)&trade.iso3.ne("VNM")&trade.hs4.isin(selected)].copy()
    trade["hs4"]=trade.hs4.astype(str).str.zfill(4)
    trade["hs2"]=trade.hs4.str[:2]
    history=trade[["iso3","country_name","hs4","hs2","year","world_imports_usd","vietnam_exports_usd","supplier_hhi"]].copy()
    info=history.rename(columns={"year":"feature_year","country_name":"destination_name","world_imports_usd":"destination_import_demand",
        "vietnam_exports_usd":"historical_export_value","supplier_hhi":"destination_supplier_concentration"})
    info["outcome_year"]=info.feature_year+lag
    future=history[["iso3","hs4","year","vietnam_exports_usd","world_imports_usd"]].rename(columns={"year":"outcome_year",
        "vietnam_exports_usd":"export_value_usd","world_imports_usd":"outcome_year_world_imports_usd"})
    base=info.merge(future,on=["iso3","hs4","outcome_year"],how="left",validate="one_to_one")
    base=base.loc[base.destination_import_demand.gt(0)].copy()
    base["outcome_observed"]=base.outcome_year_world_imports_usd.notna()
    base.loc[~base.outcome_observed,"export_value_usd"]=np.nan
    prior=history.rename(columns={"year":"prior_year","vietnam_exports_usd":"export_y_minus_four","world_imports_usd":"demand_y_minus_four"})
    prior["feature_year"]=prior.prior_year+2
    base=base.merge(prior[["iso3","hs4","feature_year","export_y_minus_four","demand_y_minus_four"]],on=["iso3","hs4","feature_year"],how="left",validate="many_to_one")
    base["destination_import_demand_growth"]=np.where(base.demand_y_minus_four.gt(0),np.log(base.destination_import_demand/base.demand_y_minus_four),np.nan)
    base["historical_export_presence"]=base.historical_export_value.gt(0).astype("int8")
    base["historical_market_share"]=base.historical_export_value/base.destination_import_demand.replace(0,np.nan)
    base["lagged_export_growth"]=np.where(base.export_y_minus_four.gt(0),(base.historical_export_value-base.export_y_minus_four)/base.export_y_minus_four,np.nan)
    base["historical_export_volatility"]=np.nan
    base["log_lagged_export_value"]=np.log1p(base.historical_export_value.clip(lower=0))
    base["log_destination_import_demand"]=np.log1p(base.destination_import_demand.clip(lower=0))
    macro=build_macro_features().rename(columns={"year":"feature_year","iso3":"destination_iso3"})
    base=base.rename(columns={"iso3":"destination_iso3"})
    base=base.merge(macro,on=["destination_iso3","feature_year"],how="left",validate="many_to_one")
    geography=build_geography()
    base=base.merge(geography,on="destination_iso3",how="left",validate="many_to_one")
    base["hs4"]=base.hs4.astype(str).str.zfill(4)
    base["hs2"]=base.hs4
    base["market_share_change"]=np.where(base.outcome_year_world_imports_usd.gt(0),base.export_value_usd/base.outcome_year_world_imports_usd-base.historical_market_share,np.nan)
    base["realized_incremental_export_usd"]=(base.export_value_usd-base.historical_export_value).clip(lower=0)
    base.loc[~base.outcome_observed,"realized_incremental_export_usd"]=np.nan
    base["destination_iso3"]=base.destination_iso3.astype(str)
    base["lag"]=lag
    if (base.feature_year>base.outcome_year-lag).any(): raise AssertionError("HS4 feature timing leak")
    return base


def run(lag:int|None=None,quick:bool=False)->pd.DataFrame:
    lag=int(lag or CFG["study"]["primary_lag"])
    trade4=pd.read_parquet(path("interim_dir")/"baci_hs4_trade.parquet")
    selected,selection,chapter_stats=select_subset(trade4)
    selection.to_csv(path("results_dir")/"tables"/"HS4_subset_selection.csv",index=False)
    chapter_stats.to_csv(path("results_dir")/"tables"/"HS4_chapter_coverage.csv",index=False)
    panel=make_hs4_panel(selected,lag)
    panel.to_parquet(path("processed_dir")/f"panel_hs4_lag{lag}.parquet",index=False)
    rows=[];pred_rows=[]
    for year in CFG["study"]["evaluation_years"]:
        valid=panel.loc[panel.outcome_observed&panel.export_value_usd.notna()]
        train=valid.loc[valid.outcome_year.le(int(year)-lag)].copy()
        test=valid.loc[valid.outcome_year.eq(int(year))].copy()
        if train.empty or test.empty: continue
        for ix,row in enumerate(test.itertuples(index=False)):
            pred_rows.append({"model":"Persistence","seed":0,"evaluation_year":year,"hs4":row.hs4,"destination_iso3":row.destination_iso3,
                "actual_usd":row.export_value_usd,"predicted_usd":row.historical_export_value})
        try:
            ppml,_,diag,_=fit_ppml(train,test)
            for ix,row in enumerate(test.itertuples(index=False)):
                pred_rows.append({"model":"Gravity_PPML","seed":0,"evaluation_year":year,"hs4":row.hs4,"destination_iso3":row.destination_iso3,
                    "actual_usd":row.export_value_usd,"predicted_usd":ppml[ix]})
        except Exception as exc:
            rows.append({"model":"Gravity_PPML","evaluation_year":year,"status":f"failed: {type(exc).__name__}: {exc}"})
        seeds=[CFG["study"]["seeds"][0]] if quick else CFG["study"]["seeds"]
        for kind in MODEL_NAMES:
            try:
                params,_=tune_regressor(kind,train,lag)
                for seed in seeds:
                    bundle=fit_regressor(kind,train,params,int(seed),lag)
                    pred=bundle.predict(test)
                    for ix,row in enumerate(test.itertuples(index=False)):
                        pred_rows.append({"model":kind,"seed":seed,"evaluation_year":year,"hs4":row.hs4,"destination_iso3":row.destination_iso3,
                            "actual_usd":row.export_value_usd,"predicted_usd":pred[ix]})
            except Exception as exc:
                rows.append({"model":kind,"evaluation_year":year,"status":f"failed: {type(exc).__name__}: {exc}"})
    predictions=pd.DataFrame(pred_rows)
    if predictions.empty:
        raise RuntimeError("HS4 robustness produced no forecasts")
    predictions.to_parquet(path("results_dir")/"predictions"/"E_HS4_predictions.parquet",index=False)
    predictions=predictions.groupby(["model","evaluation_year","hs4","destination_iso3","actual_usd"],as_index=False).predicted_usd.mean()
    for (model,year,hs4),d in predictions.groupby(["model","evaluation_year","hs4"]):
        rows.append({"model":model,"evaluation_year":year,"hs4":hs4,**forecast_metrics(d.actual_usd,d.predicted_usd),"status":"completed"})
    out=pd.DataFrame(rows)
    out.to_csv(path("results_dir")/"HS4_robustness.csv",index=False)
    return out


if __name__=="__main__":
    print(run())
