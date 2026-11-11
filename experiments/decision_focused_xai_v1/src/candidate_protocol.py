from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parents[1]
YEARS = (2022, 2023, 2024)
LAG = 2
THRESHOLD = 50_000
KEY = ["destination_iso3", "hs2", "outcome_year"]
POOL = ["task", "hs2", "outcome_year"]

sys.path.insert(0, str(PROJECT / "src"))
from market_discovery import add_entry_targets  # noqa: E402
from temporal_validation import NUMERIC_FEATURES, CATEGORICAL_FEATURES  # noqa: E402

NUM_FEATURES = list(NUMERIC_FEATURES)
CAT_FEATURES = list(CATEGORICAL_FEATURES)
FEATURES = NUM_FEATURES + CAT_FEATURES


def candidate_id(row: pd.Series) -> str:
    raw = f"{row.destination_iso3}|{str(row.hs2).zfill(2)}|{int(row.outcome_year)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def missingness_audit(test: pd.DataFrame) -> pd.DataFrame:
    """Describe outcome observability by year, HS2, and historical market size."""
    d = test.copy()
    d["size_quartile"] = pd.qcut(d.log_destination_import_demand, 4, labels=False, duplicates="drop")
    d["gdp_quartile"] = pd.qcut(d.log_gdp, 4, labels=False, duplicates="drop")
    records = []
    specifications = [
        ("year", ["outcome_year"]),
        ("hs2_by_year", ["hs2", "outcome_year"]),
        ("historical_import_demand_quartile_by_year", ["size_quartile", "outcome_year"]),
        ("historical_gdp_quartile_by_year", ["gdp_quartile", "outcome_year"]),
    ]
    for group_type, cols in specifications:
        for key, g in d.groupby(cols, dropna=False, sort=True):
            values = key if isinstance(key, tuple) else (key,)
            row = {"group_type": group_type}
            row.update({c: v for c, v in zip(cols, values)})
            row.update({"eligible_candidates": len(g),
                        "observed_outcomes": int(g.label_observed.sum()),
                        "unknown_outcomes": int((~g.label_observed).sum()),
                        "unknown_share": float((~g.label_observed).mean()),
                        "mean_log_historical_import_demand": float(g.log_destination_import_demand.mean()),
                        "mean_log_gdp": float(g.log_gdp.mean())})
            records.append(row)
    return pd.DataFrame(records)


def tag_expansion_eligibility(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply the E3 audited existing-underexploited rule at each row's origin."""
    d = panel.copy()
    d["hs2"] = d.hs2.astype(str).str.zfill(2)
    d["feature_year"] = pd.to_numeric(d.feature_year).astype(int)
    history = d.loc[
        d.outcome_observed.astype(bool)
        & d.export_value_usd.notna()
        & d.historical_export_value.gt(0)
        & d.historical_market_share.notna(),
        ["hs2", "outcome_year", "historical_market_share"],
    ]
    cutoff_map = {}
    for cutoff_year in sorted(d.feature_year.unique()):
        eligible_history = history.loc[history.outcome_year.le(int(cutoff_year))]
        medians = eligible_history.groupby("hs2").historical_market_share.median()
        cutoff_map.update({(str(h).zfill(2), int(cutoff_year)): float(v) for h, v in medians.items()})
    thresholds = [cutoff_map.get((str(h).zfill(2), int(cutoff)), np.nan)
                  for h, cutoff in zip(d.hs2, d.feature_year)]
    d["training_median_market_share"] = np.asarray(thresholds, dtype=float)
    d["task_b_eligible"] = (
        d.destination_import_demand.gt(0)
        & d.historical_export_value.gt(0)
        & d.training_median_market_share.notna()
        & d.historical_market_share.notna()
        & d.historical_market_share.le(d.training_median_market_share)
    )
    return d


def load_panel_and_candidates(write: bool = True):
    panel_path = PROJECT / "data/processed/panel_lag2.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)
    panel = pd.read_parquet(panel_path)
    panel["hs2"] = panel.hs2.astype(str).str.zfill(2)
    panel["outcome_year"] = panel.outcome_year.astype(int)
    panel["feature_year"] = panel.feature_year.astype(int)
    panel = add_entry_targets(panel, LAG, THRESHOLD)
    panel["eligible_ex_ante"] = panel.destination_import_demand.gt(0)
    panel = tag_expansion_eligibility(panel)
    panel["candidate_id"] = panel.apply(candidate_id, axis=1)
    panel["label_observed"] = panel.outcome_observed.astype(bool) & panel.export_value_usd.notna()
    panel["entry_candidate_ex_ante"] = panel.entry_history_verified.astype(bool) & panel.entry_inactive.astype(bool)
    panel["entry_success"] = panel.entry_success.astype(float)

    test = panel.loc[panel.outcome_year.isin(YEARS) & panel.eligible_ex_ante].copy()
    if test.duplicated(KEY).any():
        raise AssertionError("Duplicate candidate key")
    rows = []
    for r in test.itertuples(index=False):
        for task, flag in (("Task_A_new_entry", bool(r.entry_candidate_ex_ante)),
                           ("Task_B_existing_underexploited", bool(r.task_b_eligible))):
            if not flag:
                continue
            observed = bool(r.label_observed)
            rows.append({
                "candidate_id": r.candidate_id,
                "destination_iso3": r.destination_iso3,
                "destination_name": r.destination_name,
                "hs2": str(r.hs2).zfill(2),
                "outcome_year": int(r.outcome_year),
                "task": task,
                "candidate_pool_id": f"{task}|{str(r.hs2).zfill(2)}|{int(r.outcome_year)}",
                "eligible_ex_ante": True,
                "historical_information_cutoff": int(r.feature_year),
                "label_observed": observed,
                "missingness_reason": "" if observed else "Outcome-year BACI label unavailable; zero versus unavailable reporting unresolved",
                "entry_success": r.entry_success if task == "Task_A_new_entry" else np.nan,
                "export_value_usd": r.export_value_usd,
                "realized_incremental_export_usd": r.realized_incremental_export_usd,
                "historical_export_value": r.historical_export_value,
                "historical_market_share": r.historical_market_share,
                "training_median_market_share": r.training_median_market_share,
            })
    manifest = pd.DataFrame(rows)
    if manifest.empty or manifest.duplicated(["task", *KEY]).any():
        raise AssertionError("Frozen task manifest is empty or duplicates candidate-task keys")
    manifest["candidate_pool_id"] = manifest.candidate_pool_id.astype(str)
    miss_audit = missingness_audit(test)

    # Reconcile the audited evaluation counts before creating task-specific candidate pools.
    counts = test.groupby("outcome_year").agg(
        ex_ante_candidates=("candidate_id", "size"),
        observed_labels=("label_observed", "sum"),
        missing_labels=("label_observed", lambda s: int((~s).sum())),
        entry_candidates=("entry_candidate_ex_ante", "sum"),
    ).reset_index()
    expected = {2022: (20921, 20734, 187, 13791), 2023: (20899, 20749, 150, 13687), 2024: (20977, 20755, 222, 13653)}
    for r in counts.itertuples(index=False):
        found = (int(r.ex_ante_candidates), int(r.observed_labels), int(r.missing_labels), int(r.entry_candidates))
        if found != expected[int(r.outcome_year)]:
            raise AssertionError(f"Audit population mismatch in {r.outcome_year}: {found}")
    if write:
        manifest.to_csv(HERE / "results/candidate_manifest.csv", index=False, na_rep="")
        counts.to_csv(HERE / "results/evaluation_population.csv", index=False, na_rep="")
        miss_audit.to_csv(HERE / "results/missing_label_audit.csv", index=False, na_rep="")
    return panel, test, manifest, counts


def training_rows(panel: pd.DataFrame, task: str, outer_year: int, outcome_cutoff: int | None = None) -> pd.DataFrame:
    cutoff = int(outer_year - LAG if outcome_cutoff is None else outcome_cutoff)
    d = panel.loc[
        panel.eligible_ex_ante
        & panel.label_observed
        & panel.outcome_year.le(cutoff)
    ].copy()
    if task == "Task_A_new_entry":
        d = d.loc[d.entry_candidate_ex_ante & d.entry_success.notna()].copy()
        d["relevance"] = d.entry_success.astype(np.int8)
    elif task == "Task_B_existing_underexploited":
        d = d.loc[d.task_b_eligible & d.realized_incremental_export_usd.notna()].copy()
    else:
        raise ValueError(task)
    return d


def assert_outer_timing(train: pd.DataFrame, test: pd.DataFrame, evaluation_year: int) -> None:
    cutoff = int(evaluation_year) - LAG
    if train.empty or test.empty:
        raise AssertionError("Outer training and test samples must both be nonempty")
    if int(train.outcome_year.max()) > cutoff:
        raise AssertionError("Training outcomes exceed y-2 information cutoff")
    if (pd.to_numeric(train.feature_year) > pd.to_numeric(train.outcome_year) - LAG).any():
        raise AssertionError("A training predictor is later than its outcome-year lag cutoff")
    test_feature_year = test["feature_year"] if "feature_year" in test else test["historical_information_cutoff"]
    if (pd.to_numeric(test_feature_year) > pd.to_numeric(test.outcome_year) - LAG).any():
        raise AssertionError("A test predictor is later than its outcome-year lag cutoff")
    if not test.outcome_year.eq(int(evaluation_year)).all():
        raise AssertionError("Outer test rows include a different evaluation year")


def validation_rows(panel: pd.DataFrame, task: str, validation_year: int) -> pd.DataFrame:
    d = panel.loc[panel.eligible_ex_ante & panel.outcome_year.eq(int(validation_year)) & panel.label_observed].copy()
    if task == "Task_A_new_entry":
        d = d.loc[d.entry_candidate_ex_ante & d.entry_success.notna()].copy()
        d["relevance"] = d.entry_success.astype(np.int8)
    else:
        d = d.loc[d.task_b_eligible & d.realized_incremental_export_usd.notna()].copy()
    return d


def load_baseline_predictions(test: pd.DataFrame) -> pd.DataFrame:
    """Read previously verified prediction artifacts and map their actual task scores."""
    nn = HERE.parent / "post_audit_nn_dl_v1"
    e2 = pd.read_parquet(nn / "predictions/E2_neural_predictions.parquet")
    e3 = pd.read_parquet(nn / "predictions/E3_neural_rankings.parquet")
    e2["hs2"] = e2.hs2.astype(str).str.zfill(2)
    e2["outcome_year"] = e2.outcome_year.astype(int)
    e3["hs2"] = e3.hs2.astype(str).str.zfill(2)
    e3["outcome_year"] = e3.evaluation_year.astype(int)
    key = ["destination_iso3", "hs2", "outcome_year"]
    out = []
    a = test.loc[test.entry_candidate_ex_ante].copy()
    e2 = e2.loc[e2.task.isin(["classification", "ranking_score"])].copy()
    e2 = e2.rename(columns={"score": "baseline_score"})
    for model, g in e2.groupby("model", sort=False):
        score = g.groupby(key, as_index=False).baseline_score.mean()
        score["task"] = "Task_A_new_entry"
        score["model"] = model
        out.append(score)

    b = test.loc[test.task_b_eligible].copy()
    e3 = e3.loc[e3.task.eq("Task_B_existing_underexploited")].copy()
    for model, g in e3.groupby("ranking_model", sort=False):
        score = g.groupby(key, as_index=False).opportunity_score.mean().rename(columns={"opportunity_score": "baseline_score"})
        score["task"] = "Task_B_existing_underexploited"
        score["model"] = model
        out.append(score)
    base = pd.concat(out, ignore_index=True)
    check_rows = []
    for task, pool in (("Task_A_new_entry", a), ("Task_B_existing_underexploited", b)):
        wanted = set(map(tuple, pool[key].itertuples(index=False, name=None)))
        for model, g in base.loc[base.task.eq(task)].groupby("model"):
            have = set(map(tuple, g[key].itertuples(index=False, name=None)))
            if have != wanted:
                raise AssertionError(f"Baseline score population mismatch for {task}/{model}: missing={len(wanted-have)} extra={len(have-wanted)}")
            if not np.isfinite(g.baseline_score).all():
                raise AssertionError(f"Nonfinite baseline score {task}/{model}")
            check_rows.append({"task": task, "model": model, "candidate_keys": len(have), "matched": True})
    pd.DataFrame(check_rows).to_csv(HERE / "results/baseline_population_check.csv", index=False)
    return base


def write_input_audit(panel: pd.DataFrame, test: pd.DataFrame, manifest: pd.DataFrame, counts: pd.DataFrame) -> None:
    import lightgbm
    lines = [
        "# Input and protocol audit", "", "**Population: VERIFIED against audit_v2 and the post-audit NN extension.**",
        "", f"Processed input: `data/processed/panel_lag2.parquet` ({len(panel):,} rows, {len(panel.columns)} columns).",
        "Baseline predictions are reused from `audit_v2/corrected_E2_candidates.parquet` via the checked neural-extension artifacts, and from `post_audit_nn_dl_v1/predictions/E3_neural_rankings.parquet` for expansion scores.",
        "", "## Frozen evaluation candidate universe", "", counts.to_markdown(index=False),
        "", f"Test task manifest has {len(manifest):,} candidate-task rows: {int(manifest.task.eq('Task_A_new_entry').sum()):,} new-entry candidates and {int(manifest.task.eq('Task_B_existing_underexploited').sum()):,} existing-underexploited candidates.",
        "The 559 outcome labels remain missing. Complete-case ranking metrics are conditional on observed labels; ranks and missing-label positions are saved for the full ex-ante candidate pool.",
        "", "## Time cutoffs", "", "Outer years are 2022, 2023, and 2024. At each origin, the maximum feature reference year and training outcome year are y−2. Inner validation uses the latest historically admissible outcome year y−2, with its fitting data ending at y−4. This preserves the audited rolling-origin protocol; the same validation year may be an earlier outer test year in a later walk-forward fold, so cross-fold estimates are dependent.",
        "Historical WDI/WGI values use revised vintages; this is a retrospective evaluation, not archival real-time validation.",
        "", "## Implementation", "", f"Installed LightGBM version: {lightgbm.__version__}. `LGBMRanker` is fitted with `objective='lambdarank'`, contiguous `group` sizes, and explicit `label_gain`. The official API documents that group sizes must sum to the number of rows and that `pred_contrib` returns an extra expected-value column ([LightGBM API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html)).",
        "RankingSHAP is an established listwise explanation proposal, not a method claimed here; its reference is Heuss, de Rijke and Anand (SIGIR 2025), https://doi.org/10.1145/3726302.3729971. This extension uses native additive ranking-score contributions and explicit rank/Top-K perturbation diagnostics, and does not claim to implement RankingSHAP.",
    ]
    (HERE / "reports/INPUT_AUDIT.md").write_text("\n".join(lines) + "\n")
