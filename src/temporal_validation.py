from __future__ import annotations

import numpy as np
import pandas as pd

from common import CFG, path, write_csv


NUMERIC_FEATURES = [
    "log_gdp", "log_gdp_per_capita", "gdp_growth", "log_population", "urban_population_share",
    "internet_usage", "digitalization_missing_indicator", "inflation", "trade_openness", "log_distance",
    "contiguous", "common_official_language", "colonial_link", "common_colonizer", "landlocked",
    "log_lagged_export_value", "historical_export_presence", "log_destination_import_demand",
    "destination_import_demand_growth", "historical_market_share", "destination_supplier_concentration",
    "lagged_export_growth", "historical_export_volatility", "government_effectiveness", "regulatory_quality",
    "rule_of_law", "political_stability",
]
CATEGORICAL_FEATURES = ["destination_iso3", "hs2"]
FEATURE_GROUPS = {
    "macroeconomic_size": ["log_gdp", "gdp_growth", "log_population", "urban_population_share", "inflation", "trade_openness"],
    "purchasing_power": ["log_gdp_per_capita"],
    "geography": ["log_distance", "contiguous", "common_official_language", "colonial_link", "common_colonizer", "landlocked"],
    "trade_demand": ["log_destination_import_demand", "destination_import_demand_growth", "destination_supplier_concentration"],
    "governance": ["government_effectiveness", "regulatory_quality", "rule_of_law", "political_stability"],
    "digitalization": ["internet_usage", "digitalization_missing_indicator"],
    "historical_trade": ["log_lagged_export_value", "historical_export_presence", "historical_market_share", "lagged_export_growth", "historical_export_volatility"],
}


def split_outer(panel: pd.DataFrame, evaluation_year: int, lag: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = panel.loc[panel.outcome_observed & panel.export_value_usd.notna()].copy()
    train = valid.loc[valid.outcome_year.le(evaluation_year - lag)].copy()
    test = valid.loc[valid.outcome_year.eq(evaluation_year)].copy()
    assert_no_leakage(train, test, evaluation_year, lag)
    if train.empty or test.empty:
        raise ValueError(f"Empty train/test split for eval={evaluation_year}, lag={lag}")
    return train, test


def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame, evaluation_year: int, lag: int) -> None:
    if (train.outcome_year > evaluation_year - lag).any():
        raise AssertionError("Training outcome extends beyond evaluation origin's information cutoff")
    if (train.feature_year > train.outcome_year - lag).any() or (test.feature_year > test.outcome_year - lag).any():
        raise AssertionError("Time-varying predictor exceeds declared outcome lag")
    if test.outcome_year.nunique() != 1 or int(test.outcome_year.iloc[0]) != int(evaluation_year):
        raise AssertionError("Evaluation set contains a different outcome year")


def write_split_manifest(panel: pd.DataFrame, lag: int) -> pd.DataFrame:
    valid = panel.loc[panel.outcome_observed & panel.export_value_usd.notna()]
    rows = []
    for year in CFG["study"]["evaluation_years"]:
        tr, te = split_outer(panel, int(year), lag)
        rows.append({"lag": lag, "evaluation_year": year, "train_outcome_max": int(tr.outcome_year.max()),
                     "train_feature_max": int(tr.feature_year.max()), "test_feature_year": int(te.feature_year.max()),
                     "train_rows": len(tr), "test_rows": len(te), "train_destinations": tr.destination_iso3.nunique(),
                     "test_destinations": te.destination_iso3.nunique(), "train_hs2": tr.hs2.nunique(), "test_hs2": te.hs2.nunique(),
                     "leakage_assertion": "passed"})
    manifest = pd.DataFrame(rows)
    p = path("reports_dir") / "split_manifest.csv"
    old = pd.read_csv(p) if p.exists() else pd.DataFrame()
    old = pd.concat([old.loc[old.lag.ne(lag)] if not old.empty and "lag" in old else old, manifest], ignore_index=True)
    write_csv(old, p)
    return manifest


def run() -> None:
    from common import path
    panel = pd.read_parquet(path("processed_dir") / "panel_lag2.parquet")
    print(write_split_manifest(panel, 2).to_string(index=False))


if __name__ == "__main__":
    run()
