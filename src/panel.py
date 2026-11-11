from __future__ import annotations

import numpy as np
import pandas as pd

from common import CFG, hs2_chapter_descriptions, hs2_code, path, write_csv
from country_mapping import load_baci_metadata
from features import build_geography, build_macro_features, feature_timing_audit


TRADE_FEATURES = ["world_imports_usd", "vietnam_exports_usd", "supplier_hhi"]


def make_lagged_panel(lag: int) -> pd.DataFrame:
    hs2 = pd.read_parquet(path("interim_dir") / "baci_hs2_trade.parquet")
    countries, _ = load_baci_metadata()
    countries = countries.dropna(subset=["baci_id"]).copy()
    hs2["baci_importer_id"] = hs2.baci_importer_id.astype(int)
    hs2 = hs2.merge(countries, left_on="baci_importer_id", right_on="baci_id", how="left", validate="many_to_one")
    # Keep foreign destinations represented by a real three-letter ISO code. Non-country aggregates are excluded.
    exclusions = pd.DataFrame([
        {"exclusion_rule":"self_destination_Vietnam","rows_before_filter":int(hs2.iso3.eq(CFG["study"]["exporter_iso3"]).sum()),"unit":"destination-HS2-year"},
        {"exclusion_rule":"destination_without_alpha3_mapping","rows_before_filter":int((~hs2.iso3.str.fullmatch(r"[A-Z]{3}",na=False)).sum()),"unit":"destination-HS2-year"},
        {"exclusion_rule":"positive_world_imports_required_for_market_observation","rows_before_filter":int((hs2.world_imports_usd.le(0)).sum()),"unit":"destination-HS2-year"},
    ])
    write_csv(exclusions, path("reports_dir") / "data_exclusions.csv")
    hs2 = hs2.loc[hs2.iso3.str.fullmatch(r"[A-Z]{3}", na=False) & hs2.iso3.ne(CFG["study"]["exporter_iso3"])].copy()
    hs2["hs2"] = hs2.hs2.astype(str).str.zfill(2)
    hs2["year"] = hs2.year.astype(int)
    hs2["export_value_usd"] = hs2.vietnam_exports_usd.astype(float)
    hs2["market_share"] = hs2.vietnam_exports_usd / hs2.world_imports_usd.replace(0, np.nan)
    hs2["key"] = hs2.iso3 + "_" + hs2.hs2 + "_" + hs2.year.astype(str)
    # Candidate markets are defined using positive, observed world demand at the information date.
    history = hs2[["iso3", "country_name", "hs2", "year", "world_imports_usd", "vietnam_exports_usd", "supplier_hhi", "market_share"]].copy()
    info = history.rename(columns={"year": "feature_year", "country_name": "destination_name",
        "world_imports_usd": "destination_import_demand", "vietnam_exports_usd": "historical_export_value",
        "supplier_hhi": "destination_supplier_concentration", "market_share": "historical_market_share"})
    info["outcome_year"] = info.feature_year + lag
    # Exact historical reference uses the last observed year available at the specified lag.
    base = info.merge(hs2[["iso3", "hs2", "year", "export_value_usd", "world_imports_usd"]].rename(
        columns={"year": "outcome_year", "export_value_usd": "export_value_usd", "world_imports_usd": "outcome_year_world_imports_usd"}),
        on=["iso3", "hs2", "outcome_year"], how="left", validate="one_to_one")
    base = base.loc[base.destination_import_demand.gt(0)].copy()
    # Absence in the bilateral table is a zero only if BACI observed positive imports in this destination-product-year.
    # Rows without an outcome-year demand observation are retained as unknown for prospective scoring only.
    base["outcome_observed"] = base.outcome_year_world_imports_usd.notna()
    base.loc[~base.outcome_observed, "export_value_usd"] = np.nan
    base["destination_iso3"] = base.iso3
    # Compare the information-year observations with year t-2, using no post-origin data.
    prior = history.rename(columns={"year": "prior_year", "vietnam_exports_usd": "export_y_minus_four"})
    prior["feature_year"] = prior.prior_year + 2
    base = base.drop(columns=["export_y_minus_four"], errors="ignore").merge(
        prior[["iso3", "hs2", "feature_year", "export_y_minus_four"]], on=["iso3", "hs2", "feature_year"], how="left", validate="many_to_one")
    base["historical_export_presence"] = (base.historical_export_value.gt(0)).astype("int8")
    base["lagged_export_growth"] = np.where(base.export_y_minus_four.gt(0),
        (base.historical_export_value - base.export_y_minus_four) / base.export_y_minus_four, np.nan)
    base["destination_import_demand_growth"] = np.nan
    demand_prior = history.rename(columns={"year": "prior_year", "world_imports_usd": "demand_y_minus_four"})
    demand_prior["feature_year"] = demand_prior.prior_year + 2
    base = base.merge(demand_prior[["iso3", "hs2", "feature_year", "demand_y_minus_four"]], on=["iso3", "hs2", "feature_year"], how="left", validate="many_to_one")
    base["destination_import_demand_growth"] = np.where(base.demand_y_minus_four.gt(0),
        np.log(base.destination_import_demand / base.demand_y_minus_four), np.nan)
    base["historical_export_volatility"] = np.nan
    # A three-year pre-origin coefficient of variation, including the information-year flow.
    hist3 = history.loc[history.year.between(CFG["study"]["first_year"], CFG["study"]["last_year"])].copy()
    vol_rows = []
    for (iso3, hs2code), d in hist3.groupby(["iso3", "hs2"], sort=False):
        s = d.set_index("year").vietnam_exports_usd
        for fyear in s.index:
            vals = s.reindex([fyear - 2, fyear - 1, fyear]).dropna()
            if len(vals) >= 2 and vals.mean() > 0:
                vol_rows.append((iso3, hs2code, fyear, vals.std(ddof=0) / vals.mean()))
    if vol_rows:
        vol = pd.DataFrame(vol_rows, columns=["iso3", "hs2", "feature_year", "historical_export_volatility"])
        base = base.drop(columns="historical_export_volatility").merge(vol, on=["iso3", "hs2", "feature_year"], how="left", validate="many_to_one")
    # Country-level predictors and time-invariant gravity features are joined at feature_year.
    macro = build_macro_features()
    macro = macro.rename(columns={"year": "feature_year", "iso3": "destination_iso3"})
    base = base.merge(macro, on=["destination_iso3", "feature_year"], how="left", validate="many_to_one")
    geography = build_geography()
    base = base.merge(geography, on="destination_iso3", how="left", validate="many_to_one")
    base["log_lagged_export_value"] = np.log1p(base.historical_export_value.clip(lower=0))
    base["log_destination_import_demand"] = np.log1p(base.destination_import_demand.clip(lower=0))
    base["log_gdp"] = pd.to_numeric(base.get("log_gdp"), errors="coerce")
    base["log_gdp_per_capita"] = pd.to_numeric(base.get("log_gdp_per_capita"), errors="coerce")
    # Keep outcome year and its observed demand field for integrity checks only; never use it as a predictor.
    base["market_share_change"] = np.where(base.outcome_year_world_imports_usd.gt(0),
        base.export_value_usd / base.outcome_year_world_imports_usd - base.historical_market_share, np.nan)
    base["realized_incremental_export_usd"] = (base.export_value_usd - base.historical_export_value).clip(lower=0)
    base.loc[~base.outcome_observed, "realized_incremental_export_usd"] = np.nan
    # Source and key assertions.
    if (base.feature_year > base.outcome_year - lag).any():
        raise AssertionError(f"Feature timing violates {lag}-year lag")
    if base.duplicated(["destination_iso3", "hs2", "outcome_year"]).any():
        raise AssertionError("Duplicate destination-HS2-outcome-year panel key")
    base["lag"] = lag
    cols = ["destination_iso3", "destination_name", "hs2", "outcome_year", "feature_year", "lag",
            "export_value_usd", "outcome_observed", "outcome_year_world_imports_usd", "destination_import_demand",
            "historical_export_value", "historical_market_share", "destination_supplier_concentration",
            "destination_import_demand_growth", "historical_export_presence", "lagged_export_growth",
            "historical_export_volatility", "log_gdp", "log_gdp_per_capita", "gdp_growth", "log_population",
            "urban_population_share", "internet_usage", "digitalization_missing_indicator", "inflation", "trade_openness",
            "government_effectiveness", "regulatory_quality", "rule_of_law", "political_stability", "distance_km",
            "log_distance", "population_weighted_distance_km", "contiguous", "common_official_language", "colonial_link",
            "common_colonizer", "landlocked", "log_lagged_export_value", "log_destination_import_demand",
            "market_share_change", "realized_incremental_export_usd", "continent"]
    cols = [c for c in cols if c in base.columns]
    return base[cols].sort_values(["destination_iso3", "hs2", "outcome_year"]).reset_index(drop=True)


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    for lag in CFG["study"]["lags"]:
        panel = make_lagged_panel(lag)
        panel.to_parquet(path("processed_dir") / f"panel_lag{lag}.parquet", index=False)
        write_csv(feature_timing_audit(lag), path("reports_dir") / "feature_timing_audit.csv")
        results.append(panel)
    primary = results[CFG["study"]["lags"].index(CFG["study"]["primary_lag"])]
    primary_observed = primary.loc[primary.outcome_year.le(CFG["study"]["last_year"]) & primary.outcome_observed].copy()
    product_cov = primary_observed.groupby("hs2", as_index=False).agg(panel_observations=("outcome_year", "size"),
        destinations=("destination_iso3", "nunique"), first_year=("outcome_year", "min"), last_year=("outcome_year", "max"))
    descriptions = hs2_chapter_descriptions()
    product_cov["hs2_description"] = product_cov.hs2.map(lambda x: descriptions[hs2_code(x)])
    write_csv(product_cov, path("reports_dir") / "product_coverage.csv")
    country_cov = primary_observed.groupby(["destination_iso3", "destination_name"], as_index=False).agg(
        panel_observations=("outcome_year", "size"), hs2_chapters=("hs2", "nunique"), years_observed=("outcome_year", "nunique"))
    write_csv(country_cov, path("reports_dir") / "country_coverage.csv")
    miss = pd.DataFrame({"variable": primary_observed.columns, "missing_count": [int(primary_observed[c].isna().sum()) for c in primary_observed.columns]})
    miss["missing_rate"] = miss.missing_count / max(len(primary_observed), 1)
    write_csv(miss, path("reports_dir") / "missingness.csv")
    # A data audit summary that distinguishes eligible observed outcomes from unavailable market outcomes.
    audit = {"eligible_hs2_chapters": int(primary_observed.hs2.nunique()), "destinations": int(primary_observed.destination_iso3.nunique()),
             "panel_rows_primary_lag": int(len(primary_observed)), "unique_keys": int(primary_observed[["destination_iso3", "hs2", "outcome_year"]].drop_duplicates().shape[0]),
             "years": f"{int(primary_observed.outcome_year.min())}–{int(primary_observed.outcome_year.max())}",
             "zero_export_share": float(primary_observed.export_value_usd.eq(0).mean()),
             "prospective_unknown_rows": int((~primary.outcome_observed).sum())}
    write_csv(pd.DataFrame([audit]), path("reports_dir") / "panel_summary.csv")
    return primary, results[0 if CFG["study"]["lags"][0] != CFG["study"]["primary_lag"] else 1]


def run() -> None:
    primary, _ = build_panel()
    print(f"Panel: {len(primary):,} rows, {primary.destination_iso3.nunique()} destinations, {primary.hs2.nunique()} HS2 chapters")


if __name__ == "__main__":
    run()
