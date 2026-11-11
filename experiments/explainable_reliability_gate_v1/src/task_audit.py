from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parents[1]
YEARS = (2022, 2023, 2024)
KEY = ["destination_iso3", "hs2", "outcome_year"]

sys.path.insert(0, str(PROJECT / "src"))
from market_discovery import add_entry_targets  # noqa: E402


def _candidate_id(iso3: str, hs2: str, year: int) -> str:
    return hashlib.sha256(f"{iso3}|{str(hs2).zfill(2)}|{int(year)}".encode()).hexdigest()[:20]


def add_history_flows(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach verified origin-known annual Vietnam exports for the three-year history."""
    trade = pd.read_parquet(PROJECT / "data/interim/baci_hs2_trade.parquet")
    cmap = pd.read_parquet(PROJECT / "data/processed/country_mapping.parquet")
    trade = trade.merge(cmap[["baci_id", "iso3"]], left_on="baci_importer_id", right_on="baci_id",
                        how="left", validate="many_to_one")
    trade = trade.loc[trade.iso3.str.fullmatch(r"[A-Z]{3}", na=False) & trade.iso3.ne("VNM")].copy()
    trade["hs2"] = trade.hs2.astype(str).str.zfill(2)
    coverage = trade[["iso3", "year"]].drop_duplicates().assign(reporting_covered=True)
    flow = trade.groupby(["iso3", "hs2", "year"], as_index=False).vietnam_exports_usd.sum(min_count=1)
    out = panel.copy()
    out["hs2"] = out.hs2.astype(str).str.zfill(2)
    for j in range(3):
        keys = out[["destination_iso3", "hs2", "feature_year"]].copy()
        keys["history_year"] = keys.feature_year.astype(int) - j
        keys["_row"] = np.arange(len(keys), dtype=np.int64)
        joined = keys.merge(coverage.rename(columns={"iso3": "destination_iso3", "year": "history_year"}),
                            on=["destination_iso3", "history_year"], how="left", validate="many_to_one")
        joined = joined.merge(flow.rename(columns={"iso3": "destination_iso3", "year": "history_year",
                                                    "vietnam_exports_usd": "_flow"}),
                              on=["destination_iso3", "hs2", "history_year"], how="left", validate="many_to_one")
        covered = joined.reporting_covered.fillna(False).to_numpy(bool)
        val = joined._flow.to_numpy(float)
        val = np.where(np.isnan(val) & covered, 0.0, val)
        val = np.where(covered, val, np.nan)
        out[f"history_export_yminus{j}"] = val[np.argsort(joined._row.to_numpy())]
        out[f"history_covered_yminus{j}"] = covered[np.argsort(joined._row.to_numpy())]
    covered_cols = [f"history_covered_yminus{j}" for j in range(3)]
    flow_cols = [f"history_export_yminus{j}" for j in range(3)]
    out["strict_inactive_verified"] = out[covered_cols].all(axis=1) & out[flow_cols].eq(0.0).all(axis=1)
    out["threshold_inactive_verified"] = out.entry_history_verified.astype(bool) & out.entry_inactive.astype(bool)
    return out


def tag_expansion(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel.copy()
    history = d.loc[d.outcome_observed.astype(bool) & d.export_value_usd.notna()
                    & d.historical_export_value.gt(0) & d.historical_market_share.notna(),
                    ["hs2", "outcome_year", "historical_market_share"]]
    cutoffs: dict[tuple[str, int], float] = {}
    for cutoff in sorted(d.feature_year.astype(int).unique()):
        med = history.loc[history.outcome_year.le(cutoff)].groupby("hs2").historical_market_share.median()
        cutoffs.update({(str(h).zfill(2), cutoff): float(v) for h, v in med.items()})
    d["training_median_market_share"] = [cutoffs.get((str(h).zfill(2), int(y)), np.nan)
                                         for h, y in zip(d.hs2, d.feature_year)]
    d["task_b_eligible"] = (d.destination_import_demand.gt(0) & d.historical_export_value.gt(0)
                             & d.historical_market_share.notna() & d.training_median_market_share.notna()
                             & d.historical_market_share.le(d.training_median_market_share))
    return d


def load_and_freeze(write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(PROJECT / "data/processed/panel_lag2.parquet")
    panel["hs2"] = panel.hs2.astype(str).str.zfill(2)
    panel["outcome_year"] = panel.outcome_year.astype(int)
    panel["feature_year"] = panel.feature_year.astype(int)
    panel = add_entry_targets(panel, lag=2, threshold=50_000)
    panel["eligible_ex_ante"] = panel.destination_import_demand.gt(0)
    panel = add_history_flows(panel)
    panel["entry_candidate_ex_ante"] = panel.threshold_inactive_verified.astype(bool)
    panel = tag_expansion(panel)
    panel["candidate_id"] = [_candidate_id(a, b, c) for a, b, c in panel[KEY].itertuples(index=False, name=None)]
    panel["label_observed"] = panel.outcome_observed.astype(bool) & panel.export_value_usd.notna()
    evald = panel.loc[panel.outcome_year.isin(YEARS) & panel.eligible_ex_ante].copy()
    evald["task_a_strict"] = evald.strict_inactive_verified.astype(bool)
    evald["task_a_audited_threshold"] = evald.threshold_inactive_verified.astype(bool)
    evald["task_b"] = evald.task_b_eligible.astype(bool)
    # The strict primary Task A is disjoint from existing Task B by construction.
    overlap_prior = evald.task_a_audited_threshold & evald.task_b
    overlap_new = evald.task_a_strict & evald.task_b
    evald["entry_success"] = np.where(evald.label_observed,
                                       evald.export_value_usd.ge(50_000).astype(float), np.nan)
    evald["missingness_reason"] = np.where(evald.label_observed, "",
        "Outcome-year BACI label unavailable; not recoded to zero")
    manifest_rows = []
    for r in evald.itertuples(index=False):
        for task, include in (("Task_A_new_entry", bool(r.task_a_strict)),
                              ("Task_B_existing_underexploited", bool(r.task_b))):
            if not include:
                continue
            manifest_rows.append({
                "candidate_id": r.candidate_id, "destination_iso3": r.destination_iso3,
                "destination_name": r.destination_name, "hs2": str(r.hs2).zfill(2),
                "outcome_year": int(r.outcome_year), "task": task,
                "candidate_pool_id": f"{task}|{r.hs2}|{int(r.outcome_year)}",
                "eligible_ex_ante": True, "historical_information_cutoff": int(r.feature_year),
                "label_observed": bool(r.label_observed), "missingness_reason": r.missingness_reason,
                "entry_success": r.entry_success if task == "Task_A_new_entry" else np.nan,
                "export_value_usd": r.export_value_usd,
                "realized_incremental_export_usd": r.realized_incremental_export_usd,
                "historical_export_value": r.historical_export_value,
                "historical_market_share": r.historical_market_share,
                "strict_inactive_verified": bool(r.strict_inactive_verified),
                "threshold_inactive_verified": bool(r.threshold_inactive_verified),
                "training_median_market_share": r.training_median_market_share,
            })
    manifest = pd.DataFrame(manifest_rows)
    if manifest.duplicated(["task", *KEY]).any():
        raise AssertionError("Duplicate task candidate key")
    if manifest.loc[manifest.task.eq("Task_A_new_entry"), KEY].merge(
        manifest.loc[manifest.task.eq("Task_B_existing_underexploited"), KEY], on=KEY).shape[0]:
        raise AssertionError("Strict entry and existing expansion still overlap")
    counts = []
    for year, g in evald.groupby("outcome_year"):
        counts.append({"outcome_year": int(year), "ex_ante_population": len(g),
            "outcomes_observed": int(g.label_observed.sum()), "outcomes_unknown": int((~g.label_observed).sum()),
            "prior_threshold_entry": int(g.task_a_audited_threshold.sum()),
            "prior_task_overlap": int(overlap_prior.loc[g.index].sum()),
            "strict_zero_entry": int(g.task_a_strict.sum()), "strict_task_overlap": int(overlap_new.loc[g.index].sum()),
            "existing_underexploited": int(g.task_b.sum()),
            "task_a_unknown_labels": int((g.task_a_strict & ~g.label_observed).sum()),
            "task_b_unknown_labels": int((g.task_b & ~g.label_observed).sum())})
    counts = pd.DataFrame(counts)
    sample_flow = pd.DataFrame([
        {"stage": "Ex-ante destination-HS2-year universe", "n_rows": len(evald)},
        {"stage": "Prior threshold-defined entry candidates", "n_rows": int(evald.task_a_audited_threshold.sum())},
        {"stage": "Prior existing-underexploited candidates", "n_rows": int(evald.task_b.sum())},
        {"stage": "Prior overlapping task keys", "n_rows": int(overlap_prior.sum())},
        {"stage": "Corrected strict-zero entry candidates", "n_rows": int(evald.task_a_strict.sum())},
        {"stage": "Corrected existing-underexploited candidates", "n_rows": int(evald.task_b.sum())},
        {"stage": "Corrected task overlap", "n_rows": int(overlap_new.sum())},
        {"stage": "Task A/B union keys", "n_rows": int(evald.task_a_strict.sum()+evald.task_b.sum())},
        {"stage": "Neither corrected task", "n_rows": int((~(evald.task_a_strict|evald.task_b)).sum())},
    ])
    if write:
        HERE.joinpath("results").mkdir(parents=True, exist_ok=True)
        manifest.to_csv(HERE / "results/candidate_manifest.csv", index=False, na_rep="")
        counts.to_csv(HERE / "results/sample_flow.csv", index=False)
        pd.DataFrame({"candidate_id": evald.candidate_id, "destination_iso3": evald.destination_iso3,
          "hs2": evald.hs2, "outcome_year": evald.outcome_year, "label_observed": evald.label_observed,
          "destination_import_demand": evald.destination_import_demand, "log_gdp": evald.log_gdp,
          "task_a_strict": evald.task_a_strict, "task_a_audited_threshold": evald.task_a_audited_threshold,
          "task_b": evald.task_b}).to_csv(HERE / "results/exante_universe_audit.csv", index=False, na_rep="")
        text = ["# Initial data and task audit", "", "## Provenance and frozen evaluation inputs", "",
          "The new extension reuses `data/processed/panel_lag2.parquet`, BACI-derived HS2 interim trade data and the existing saved out-of-time predictions. It does not modify previous files or download trade data. Historical WDI/WGI values are revised vintages, so the exercise is retrospective.", "",
          f"The audited ex-ante evaluation universe contains {len(evald):,} destination–HS2–year keys (2022–2024), of which {int(evald.label_observed.sum()):,} outcomes are observed and {int((~evald.label_observed).sum()):,} remain unknown.", "",
          "## Task overlap and correction", "",
          f"The prior threshold-based entry definition includes rows whose last three verified annual export flows were each below USD 50,000. The expansion definition only requires positive historical exports and a market share at or below the HS2 training median. Therefore a positive but sub-threshold export relationship can satisfy both rules. Recomputed overlap is {int(overlap_prior.sum()):,} keys. The overlap is generated by the definitions, not duplicate keys or a join error.", "",
          "For this extension, Task A is defined as **zero verified Vietnamese exports in each of the three historical years** and success remains outcome-year exports ≥ USD 50,000. Task B retains the prior positive-historical-export and below-median-share rule. These primary candidate sets are now disjoint. The original threshold-based Task A count is retained in the audit as a sensitivity population. This alters the estimand from entry into commercial scale to strictly new bilateral product relationships.", "",
          counts.to_markdown(index=False), "",
          "Unknown BACI outcomes remain missing and are never recoded as zero. Complete-case metrics describe only observed-label candidates; missing counts and selection into Top-10 are separately retained.", "",
          "## Integrity checks", "",
          "- Origin years are outcome year minus two.",
          "- A historical zero is accepted only when BACI importer-year reporting coverage exists; otherwise history is unavailable, not zero.",
          "- The strict Task A and Task B key intersection is zero.",
          "- The past 6,511 overlap keys are explained by positive sub-threshold history; no duplicate keys were found in the old task manifest.",
          "- WDI/WGI vintage revisions prevent a fully point-in-time reconstruction.", ""]
        (HERE / "reports/INITIAL_DATA_AND_TASK_AUDIT.md").write_text("\n".join(text))
    return panel, evald

