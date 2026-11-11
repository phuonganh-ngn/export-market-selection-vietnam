from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from market_discovery import add_entry_targets  # noqa: E402
from temporal_validation import CATEGORICAL_FEATURES, FEATURE_GROUPS, NUMERIC_FEATURES  # noqa: E402

YEARS = [2022, 2023, 2024]
LAG = 2
ENTRY_THRESHOLD_USD = 50_000
SEEDS = [11, 23, 37, 51, 73]
KEY = ["destination_iso3", "hs2", "outcome_year"]
CAT_COLS = list(CATEGORICAL_FEATURES)
FEATURE_SETS = {
    "F0_history_only": ["historical_trade"],
    "F1_history_plus_size": ["historical_trade", "macroeconomic_size", "purchasing_power"],
    "F2_plus_import_demand": ["historical_trade", "macroeconomic_size", "purchasing_power", "trade_demand"],
    "F3_plus_governance": ["historical_trade", "macroeconomic_size", "purchasing_power", "trade_demand", "governance"],
    "F4_full": ["historical_trade", "macroeconomic_size", "purchasing_power", "trade_demand", "governance", "digitalization", "geography"],
}


def feature_columns(feature_set: str = "F4_full") -> tuple[list[str], list[str]]:
    groups = FEATURE_SETS[feature_set]
    numerical = []
    for group in groups:
        numerical.extend(FEATURE_GROUPS[group])
    numerical = list(dict.fromkeys(c for c in numerical if c in NUMERIC_FEATURES))
    return numerical, CAT_COLS.copy()


def stable_candidate_id(row: pd.Series) -> str:
    raw = f"{row.destination_iso3}|{str(row.hs2).zfill(2)}|{int(row.outcome_year)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def load_frozen_population(write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild, then verify the audited ex-ante universe and entry history labels."""
    panel = pd.read_parquet(PROJECT / "data/processed/panel_lag2.parquet")
    panel["hs2"] = panel.hs2.astype(str).str.zfill(2)
    panel["destination_iso3"] = panel.destination_iso3.astype(str)
    panel["outcome_year"] = panel.outcome_year.astype(int)
    panel["feature_year"] = panel.feature_year.astype(int)
    candidates = panel.loc[panel.outcome_year.isin(YEARS)].copy()
    if candidates.duplicated(KEY).any():
        raise AssertionError("Duplicate outcome key in frozen population")
    candidates = add_entry_targets(candidates, LAG, ENTRY_THRESHOLD_USD)
    candidates["eligible_ex_ante"] = candidates.destination_import_demand.gt(0)
    candidates["label_observed"] = candidates.outcome_observed.astype(bool) & candidates.export_value_usd.notna()
    candidates["missingness_reason"] = np.where(
        candidates.label_observed,
        "",
        "Outcome-year destination-HS2 import observation absent in processed BACI; zero versus unavailable reporting is unresolved",
    )
    candidates["entry_candidate_ex_ante"] = candidates.entry_history_verified.astype(bool) & candidates.entry_inactive.astype(bool)
    candidates["entry_label_observed"] = candidates.entry_candidate_ex_ante & candidates.label_observed
    candidates["candidate_id"] = candidates.apply(stable_candidate_id, axis=1)
    if not candidates.eligible_ex_ante.all():
        raise AssertionError("Panel contains a non-positive-demand row in the ex-ante universe")

    counts = candidates.groupby("outcome_year").agg(
        ex_ante_candidates=("candidate_id", "size"),
        observed_labels=("label_observed", "sum"),
        missing_labels=("label_observed", lambda x: int((~x).sum())),
        entry_candidates=("entry_candidate_ex_ante", "sum"),
        observed_entry_labels=("entry_label_observed", "sum"),
    ).reset_index()
    events = candidates.loc[candidates.entry_candidate_ex_ante & candidates.entry_success.eq(1)].groupby("outcome_year").size()
    counts["entry_events"] = counts.outcome_year.map(events).fillna(0).astype(int)
    expected = {2022: (20921, 20734, 187), 2023: (20899, 20749, 150), 2024: (20977, 20755, 222)}
    for row in counts.itertuples(index=False):
        if (int(row.ex_ante_candidates), int(row.observed_labels), int(row.missing_labels)) != expected[int(row.outcome_year)]:
            raise AssertionError(f"Candidate counts diverge from audit_v2 for {row.outcome_year}: {row}")
    if int(counts.ex_ante_candidates.sum()) != 62797 or int(counts.observed_labels.sum()) != 62238:
        raise AssertionError("Frozen candidate population does not match audit_v2 denominators")

    manifest_columns = ["candidate_id", *KEY, "destination_name", "feature_year", "eligible_ex_ante",
                        "label_observed", "missingness_reason", "entry_candidate_ex_ante",
                        "entry_history_verified", "entry_inactive", "entry_success", "export_value_usd",
                        "realized_incremental_export_usd", "destination_import_demand", "log_gdp",
                        "historical_export_value", "historical_market_share"]
    manifest = candidates[manifest_columns].copy()
    observed = candidates.loc[candidates.label_observed].copy()
    if write:
        (HERE / "results").mkdir(exist_ok=True)
        manifest.to_csv(HERE / "results/candidate_manifest.csv", index=False, na_rep="")
        counts.to_csv(HERE / "results/evaluation_population.csv", index=False, na_rep="")
        missing = candidates.loc[~candidates.label_observed].copy()
        missing_rows = []
        for (year, hs2), g in missing.groupby(["outcome_year", "hs2"]):
            missing_rows.append({"outcome_year": year, "hs2": hs2, "missing_candidates": len(g),
                                 "mean_log_gdp": g.log_gdp.mean(), "mean_log_import_demand_origin": g.log_destination_import_demand.mean(),
                                 "mean_historical_exports_usd": g.historical_export_value.mean(),
                                 "mean_market_share_origin": g.historical_market_share.mean()})
        pd.DataFrame(missing_rows).to_csv(HERE / "results/missing_label_audit.csv", index=False, na_rep="")
    return panel, candidates, observed


def load_baselines(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load saved audited predictions and verify keys before any NN comparisons."""
    pred_dir = PROJECT / "audit_v2"
    forecast = pd.read_parquet(pred_dir / "corrected_exante_forecast_predictions.parquet")
    forecast["hs2"] = forecast.hs2.astype(str).str.zfill(2)
    forecast["evaluation_year"] = forecast.evaluation_year.astype(int)
    forecast = forecast.rename(columns={"evaluation_year": "outcome_year"})
    forecast_key = ["destination_iso3", "hs2", "outcome_year"]
    if forecast.duplicated(["model", *forecast_key]).any():
        raise AssertionError("Audit-v2 forecast baseline has duplicate model-candidate keys")
    expected_keys = set(map(tuple, candidates[forecast_key].itertuples(index=False, name=None)))
    for model, group in forecast.groupby("model"):
        observed_keys = set(map(tuple, group[forecast_key].itertuples(index=False, name=None)))
        if observed_keys != expected_keys:
            raise AssertionError(f"Baseline {model} does not cover the frozen ex-ante population")

    entry = pd.read_parquet(pred_dir / "corrected_E2_candidates.parquet")
    entry["hs2"] = entry.hs2.astype(str).str.zfill(2)
    entry["evaluation_year"] = entry.evaluation_year.astype(int)
    entry = entry.rename(columns={"evaluation_year": "outcome_year"})
    entry_candidates = candidates.loc[candidates.entry_candidate_ex_ante, [*forecast_key, "entry_success", "label_observed"]]
    wanted = set(map(tuple, entry_candidates[forecast_key].itertuples(index=False, name=None)))
    existing = set(map(tuple, entry[["destination_iso3", "hs2", "outcome_year"]].drop_duplicates().itertuples(index=False, name=None)))
    if wanted != existing:
        raise AssertionError(f"Saved E2 baseline candidate keys differ: current={len(wanted)}, saved={len(existing)}")
    return forecast, entry


def write_input_audit(panel: pd.DataFrame, candidates: pd.DataFrame, forecast: pd.DataFrame, entry: pd.DataFrame) -> None:
    import importlib.metadata
    import torch
    import tabm

    year_counts = candidates.groupby("outcome_year").agg(n=("candidate_id", "size"), observed=("label_observed", "sum"),
                                                         missing=("label_observed", lambda s: (~s).sum()),
                                                         inactive=("entry_candidate_ex_ante", "sum")).reset_index()
    event_counts = candidates.loc[candidates.entry_candidate_ex_ante & candidates.entry_success.eq(1)].groupby("outcome_year").size()
    year_counts["events"] = year_counts.outcome_year.map(event_counts).fillna(0).astype(int)
    text = [
        "# Input audit and frozen evaluation population", "",
        "**Status: VERIFIED** for key-level reproduction of the audited 2022–2024 two-year-lag candidate set. This is not a point-in-time macro-data vintage.", "",
        "## Inputs", "",
        "- Processed panel: `data/processed/panel_lag2.parquet`; rows = {:,}; columns = {}.".format(len(panel), len(panel.columns)),
        "- Forecast comparators: `audit_v2/corrected_exante_forecast_predictions.parquet`; rows = {:,}; models = {}.".format(len(forecast), ", ".join(sorted(forecast.model.unique()))),
        "- Entry comparators: `audit_v2/corrected_E2_candidates.parquet`; rows = {:,}; models = {}.".format(len(entry), ", ".join(sorted(entry.model.unique()))),
        "- Primary feature cutoff: `feature_year = outcome_year - 2`; training outcomes stop at `evaluation_year - 2`.",
        "- Entry target: three years of verified reporting through the origin; exports below USD 50,000 in each; success iff observed outcome-year exports reach USD 50,000.",
        "- Missing outcome labels are preserved as NA. An absent BACI destination-HS2 outcome import aggregate is not treated as a true zero.",
        "- Revised current WDI API and WGI 2026 historical values are retrospective and do not establish point-in-time publication availability.",
        "", "## Reconstructed annual population", "", year_counts.to_markdown(index=False), "",
        "Total candidates = {:,}; observed outcomes = {:,}; unobserved outcomes = {:,}; inactive entry candidates = {:,}.".format(
            len(candidates), int(candidates.label_observed.sum()), int((~candidates.label_observed).sum()), int(candidates.entry_candidate_ex_ante.sum())),
        "", "## Neural implementation", "",
        f"PyTorch {torch.__version__}; official `tabm` package {tabm.__version__} (installed distribution {importlib.metadata.version('tabm')}).",
        "TabM uses the official package class, categorical one-hot handling, and averages member probabilities after sigmoid for classification; regression members are averaged before inverse transform.",
        "MLP and TabM use the same listed numeric features, destination identity, and HS2 identity. All imputers and scalers are fitted within each chronological training fold.",
        "", "## Unresolved integrity boundary", "",
        "The source audit cannot distinguish a true zero-import destination-product outcome from unavailable/incomplete reporting for 559 candidate keys. Complete-case performance is therefore conditional on observed outcomes. See `reports/MISSING_LABEL_SENSITIVITY.md`.",
    ]
    (HERE / "reports/INPUT_AUDIT.md").write_text("\n".join(text) + "\n")
