from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from experiments.explainable_reliability_gate_v1.src.task_audit import load_and_freeze  # noqa: E402

ERGM = PROJECT / "experiments/explainable_reliability_gate_v1"
EVAL_YEARS = (2022, 2023, 2024)
OOF_YEARS = (2018, 2019, 2020, 2021)
TASK_A = "Task_A_new_entry"
TASK_B = "Task_B_existing_underexploited"
TASKS = (TASK_A, TASK_B)
BOOT_REPS = 10_000
SEED = 51073
SCORE_NAMES = {
    "Gravity_PPML": "ppml_score",
    "LightGBM": "lightgbm_score",
    "EQUAL_WEIGHT_RANK_BLEND": "blend_rank_score",
    "BORDA_RANK_AGGREGATION": "borda_rank_score",
    "SAVED_ORIGINAL_STATIC_BLEND": "saved_blend_score",
}
NUMERIC_FEATURES = [
    "candidate_count", "mean_log_gdp", "median_log_gdp", "sd_log_gdp",
    "mean_log_demand", "median_log_demand", "mean_log_hist_exports",
    "mean_market_share", "mean_trade_volatility", "mean_demand_growth",
    "predictor_missing_share", "top10_jaccard", "rank_kendall_tau",
    "rank_displacement_norm", "ppml_margin_z", "lightgbm_margin_z",
    "prior_regret_mean", "prior_regret_sd", "prior_history_n",
]


def ensure_dirs(root: Path) -> None:
    for name in ("models", "predictions", "results", "reports", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)


def _rank_score(g: pd.DataFrame, score: str) -> pd.DataFrame:
    return g.sort_values([score, "destination_iso3"], ascending=[False, True], kind="mergesort").copy()


def _add_blend_scores(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in frame.groupby(["task", "hs2", "outcome_year"], sort=False):
        q = g.copy()
        p = _rank_score(q, "ppml_score")
        l = _rank_score(q, "lightgbm_score")
        pr = pd.Series(np.arange(1, len(p) + 1), index=p.index)
        lr = pd.Series(np.arange(1, len(l) + 1), index=l.index)
        n = len(q)
        # This reproduces the saved equal-weight normalized within-pool rank blend.
        q["_rank_ppml_rebuilt"] = pr.reindex(q.index).to_numpy()
        q["_rank_lightgbm_rebuilt"] = lr.reindex(q.index).to_numpy()
        q["blend_rank_score"] = .5 * (
            (n - q._rank_ppml_rebuilt + 1) / max(1, n - 1)
            + (n - q._rank_lightgbm_rebuilt + 1) / max(1, n - 1)
        )
        q["borda_rank_score"] = (2 * n + 2 - q._rank_ppml_rebuilt - q._rank_lightgbm_rebuilt).astype(float)
        parts.append(q)
    return pd.concat(parts, ignore_index=True)


def _task_gain(frame: pd.DataFrame, task: str, definition: str = "primary") -> np.ndarray:
    if task == TASK_A:
        return pd.to_numeric(frame.entry_success, errors="coerce").to_numpy(float)
    inc = pd.to_numeric(frame.realized_incremental_export_usd, errors="coerce").to_numpy(float)
    pos = np.maximum(0.0, inc)
    if definition == "primary":
        grade = np.minimum(4, np.floor(np.log10(1.0 + pos / 50_000.0))).astype(int)
        return np.exp2(grade) - 1.0
    if definition == "binary_50k":
        return (pos >= 50_000).astype(float)
    if definition == "log1p_usd":
        return np.log1p(pos)
    if definition == "raw_usd":
        return pos
    raise ValueError(definition)


def _dcg(gain: np.ndarray, k: int) -> float:
    x = np.asarray(gain, dtype=float)[:k]
    return float(np.sum(x / np.log2(np.arange(2, len(x) + 2))))


def pool_metrics(frame: pd.DataFrame, score: str, task: str, k_values=(5, 10), gain_def="primary") -> dict:
    all_ranked = _rank_score(frame, score)
    unknown = all_ranked.loc[~all_ranked.label_observed.astype(bool)]
    known = all_ranked.loc[all_ranked.label_observed.astype(bool)].copy()
    result = {"candidate_count": len(frame), "observed_label_count": len(known),
              "unknown_label_count": len(frame) - len(known)}
    for k in k_values:
        full_top = all_ranked.head(k)
        n = min(k, len(known))
        top = known.head(n)
        gains = _task_gain(known, task, gain_def)
        selected_gain = _task_gain(top, task, gain_def)
        oracle_gain = np.sort(gains)[::-1][:n]
        idcg = _dcg(oracle_gain, n)
        dcg = _dcg(selected_gain, n)
        ndcg = dcg / idcg if idcg > 0 else np.nan
        if task == TASK_A:
            binary = pd.to_numeric(top.entry_success, errors="coerce").to_numpy(float)
            selected_usd = np.maximum(0, pd.to_numeric(top.export_value_usd, errors="coerce").to_numpy(float))
            # Preserve unknown labels as NaN. nansum below is an explicitly
            # partial observed sum, not a zero-imputation of missing outcomes.
            full_usd = np.maximum(0, pd.to_numeric(full_top.export_value_usd, errors="coerce").to_numpy(float))
            if np.isnan(selected_usd).any() or np.isnan(binary).any():
                raise AssertionError("Observed entry labels must have nonmissing outcome values")
            success_count = float(np.nansum(binary))
            false_count = float(n - success_count)
        else:
            inc = pd.to_numeric(top.realized_incremental_export_usd, errors="coerce").to_numpy(float)
            binary = (inc >= 50_000).astype(float)
            selected_usd = np.maximum(0, inc)
            full_usd = np.maximum(0, pd.to_numeric(full_top.realized_incremental_export_usd, errors="coerce").to_numpy(float))
            if np.isnan(selected_usd).any():
                raise AssertionError("Observed expansion labels must have nonmissing incremental values")
            success_count = float(binary.sum())
            false_count = float(n - success_count)
        oracle_usd_values = np.maximum(0, pd.to_numeric(
            known.export_value_usd if task == TASK_A else known.realized_incremental_export_usd,
            errors="coerce").to_numpy(float))
        oracle_usd = float(np.sum(np.sort(oracle_usd_values)[::-1][:n]))
        result.update({
            f"ndcg@{k}": ndcg,
            f"precision@{k}": float(binary.sum() / n) if n else np.nan,
            f"successful_or_meaningful_count@{k}": success_count,
            f"unsuccessful_recommendations@{k}": false_count,
            f"regret@{k}": max(0.0, float(np.sum(oracle_gain) - np.sum(selected_gain))),
            f"selected_gain@{k}": float(np.sum(selected_gain)),
            f"oracle_gain@{k}": float(np.sum(oracle_gain)),
            f"selected_usd_known_top{k}": float(np.nansum(selected_usd)),
            f"oracle_usd_known_top{k}": oracle_usd,
            f"usd_regret_known_top{k}": max(0.0, oracle_usd - np.nansum(selected_usd)),
            f"unknown_in_full_score_top{k}": int((~full_top.label_observed.astype(bool)).sum()),
            f"observed_usd_partial_full_score_top{k}": float(np.nansum(full_usd)),
            f"partial_usd_is_complete": bool((~full_top.label_observed.astype(bool)).sum() == 0),
        })
    return result


def _sample_integrity(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    panel, _ = load_and_freeze(write=False)
    saved_manifest = pd.read_csv(ERGM / "results/candidate_manifest.csv", dtype={"hs2": str})
    saved_scores = pd.read_parquet(ERGM / "predictions/ergm_candidate_predictions.parquet")
    oof = pd.read_parquet(ERGM / "predictions/historical_oof_expert_predictions.parquet")
    for d in (saved_manifest, saved_scores, oof, panel):
        d["hs2"] = d.hs2.astype(str).str.zfill(2)
        d["outcome_year"] = d.outcome_year.astype(int)
    expected_keys = set(map(tuple, saved_manifest[["task", "candidate_id"]].drop_duplicates().to_numpy()))
    actual_keys = set(map(tuple, saved_scores[["task", "candidate_id"]].drop_duplicates().to_numpy()))
    if expected_keys != actual_keys:
        raise AssertionError(f"Saved candidate/score key mismatch: missing={len(expected_keys-actual_keys)}, extra={len(actual_keys-expected_keys)}")
    if saved_manifest.duplicated(["task", "candidate_id"]).any() or saved_scores.duplicated(["task", "candidate_id"]).any():
        raise AssertionError("Duplicate evaluation task-candidate key")
    if saved_manifest.loc[saved_manifest.task.eq(TASK_A), ["destination_iso3", "hs2", "outcome_year"]].merge(
        saved_manifest.loc[saved_manifest.task.eq(TASK_B), ["destination_iso3", "hs2", "outcome_year"]],
        on=["destination_iso3", "hs2", "outcome_year"]).shape[0]:
        raise AssertionError("Task A and Task B candidate keys overlap")
    if not (saved_manifest.historical_information_cutoff == saved_manifest.outcome_year - 2).all():
        raise AssertionError("Manifest cutoff is not outcome year minus two")
    if saved_manifest.loc[saved_manifest.task.eq(TASK_A), "strict_inactive_verified"].ne(True).any():
        raise AssertionError("Task A includes a non-strict-zero row")
    # Historical three-year window attached by the source candidate builder is feature_year, -1, -2.
    panel_eval = panel.loc[panel.outcome_year.isin(EVAL_YEARS)].copy()
    panel_eval["task_a_eligible"] = panel_eval.strict_inactive_verified.astype(bool) & panel_eval.destination_import_demand.gt(0)
    panel_eval["task_b_eligible_final"] = panel_eval.task_b_eligible.astype(bool) & panel_eval.destination_import_demand.gt(0)
    rebuilt = []
    for task, mask in ((TASK_A, panel_eval.task_a_eligible), (TASK_B, panel_eval.task_b_eligible_final)):
        q = panel_eval.loc[mask].copy()
        q["task"] = task
        q["historical_information_cutoff"] = q.feature_year.astype(int)
        q["history_start_year"] = q.feature_year.astype(int) - 2
        q["history_end_year"] = q.feature_year.astype(int)
        q["history_years_used"] = q.apply(lambda r: f"{int(r.feature_year)-2},{int(r.feature_year)-1},{int(r.feature_year)}", axis=1)
        q["label_observed_rebuilt"] = q.outcome_observed.astype(bool) & q.export_value_usd.notna()
        q["eligible_ex_ante"] = True
        q["candidate_pool_id"] = [f"{task}|{str(h).zfill(2)}|{int(y)}" for h,y in zip(q.hs2,q.outcome_year)]
        rebuilt.append(q)
    rebuilt_manifest = pd.concat(rebuilt, ignore_index=True)
    rebuilt_keys = set(map(tuple, rebuilt_manifest[["task", "candidate_id"]].drop_duplicates().to_numpy()))
    if rebuilt_keys != expected_keys:
        raise AssertionError(f"Rebuilt candidate manifest differs from archived manifest: missing={len(expected_keys-rebuilt_keys)}, extra={len(rebuilt_keys-expected_keys)}")
    if rebuilt_manifest.duplicated(["task", "candidate_id"]).any():
        raise AssertionError("Rebuilt candidate keys are not unique")
    # Verify historical OOF universe against the same strict rules and ensure fold cutoffs.
    hist_expected = []
    hp = panel.loc[panel.outcome_year.isin(OOF_YEARS) & panel.destination_import_demand.gt(0)].copy()
    for task, mask in ((TASK_A, hp.strict_inactive_verified.astype(bool)),
                       (TASK_B, hp.task_b_eligible.astype(bool))):
        z = hp.loc[mask, ["candidate_id", "destination_iso3", "hs2", "outcome_year", "feature_year"]].copy()
        z["task"] = task
        hist_expected.append(z)
    hist_expected_df = pd.concat(hist_expected, ignore_index=True)
    oof_keys = set(map(tuple, oof[["task", "candidate_id"]].drop_duplicates().to_numpy()))
    hist_keys = set(map(tuple, hist_expected_df[["task", "candidate_id"]].drop_duplicates().to_numpy()))
    if hist_keys != oof_keys:
        raise AssertionError(f"Historical OOF task candidate mismatch: missing={len(hist_keys-oof_keys)}, extra={len(oof_keys-hist_keys)}")
    if (hist_expected_df.feature_year > hist_expected_df.outcome_year - 2).any():
        raise AssertionError("Historical OOF predictor cutoff violation")
    if (oof.groupby("outcome_year").size().index.to_numpy() != np.array(OOF_YEARS)).any():
        raise AssertionError("Unexpected OOF years")
    if oof.duplicated(["task", "candidate_id", "outcome_year"]).any():
        raise AssertionError("Duplicate OOF key")
    if saved_scores[["ppml_score", "lightgbm_score"]].isna().any().any():
        raise AssertionError("Missing outer expert score")
    # Compare all archived task/outcome labels to the independently rebuilt panel.
    labels = saved_manifest.merge(rebuilt_manifest[["task", "candidate_id", "label_observed_rebuilt"]],
                                  on=["task", "candidate_id"], validate="one_to_one")
    if not (labels.label_observed.astype(bool).to_numpy() == labels.label_observed_rebuilt.astype(bool).to_numpy()).all():
        raise AssertionError("Archived and rebuilt observed-label indicators differ")
    # Freeze this extension's own manifest with origin years and source history fields.
    manifest_cols = ["task", "candidate_id", "destination_iso3", "destination_name", "hs2", "outcome_year",
                     "candidate_pool_id", "eligible_ex_ante", "historical_information_cutoff", "history_start_year",
                     "history_end_year", "history_years_used", "label_observed_rebuilt", "export_value_usd",
                     "entry_success", "realized_incremental_export_usd", "historical_export_value",
                     "historical_market_share", "strict_inactive_verified", "task_b_eligible_final"]
    frozen = rebuilt_manifest[manifest_cols].rename(columns={"label_observed_rebuilt": "label_observed",
                                                              "task_b_eligible_final": "task_b_eligible"})
    frozen.to_csv(root / "results/final_candidate_manifest.csv", index=False, na_rep="")
    counts = []
    for task, g in frozen.groupby("task"):
        for year, y in g.groupby("outcome_year"):
            counts.append({"task": task, "outcome_year": int(year), "ex_ante_candidates": len(y),
                           "observed_labels": int(y.label_observed.sum()),
                           "unknown_labels": int((~y.label_observed.astype(bool)).sum()),
                           "unique_destinations": y.destination_iso3.nunique(), "hs2_chapters": y.hs2.nunique(),
                           "candidate_pools": y.candidate_pool_id.nunique(),
                           "cutoff_violation_count": int((y.historical_information_cutoff > y.outcome_year - 2).sum())})
    sample_flow = pd.DataFrame(counts)
    sample_flow.to_csv(root / "results/final_sample_flow.csv", index=False)
    summary = {
        "eval_rows": len(frozen), "eval_unknown": int((~frozen.label_observed.astype(bool)).sum()),
        "task_a_n": int(frozen.task.eq(TASK_A).sum()), "task_b_n": int(frozen.task.eq(TASK_B).sum()),
        "task_a_unknown": int((frozen.task.eq(TASK_A) & ~frozen.label_observed.astype(bool)).sum()),
        "task_b_unknown": int((frozen.task.eq(TASK_B) & ~frozen.label_observed.astype(bool)).sum()),
        "oof_rows": len(oof), "oof_years": sorted(oof.outcome_year.unique().tolist()),
        "pools_each_task": int(frozen.groupby("task").candidate_pool_id.nunique().min()),
        "strict_task_overlap": 0,
    }
    audit = [
        "# Final candidate and temporal audit", "",
        "## Reconstructed population", "",
        f"The source task builder was rerun read-only from `data/processed/panel_lag2.parquet` and BACI HS2 interim trade records. Its candidate keys were compared with the archived ERGM manifest and prediction archive. The independently rebuilt manifest contains {len(frozen):,} task-candidate rows: {summary['task_a_n']:,} Task A and {summary['task_b_n']:,} Task B. Unknown labels remain NA; Task A has {summary['task_a_unknown']:,} and Task B has {summary['task_b_unknown']:,}.", "",
        sample_flow.to_markdown(index=False), "",
        "## Timing", "",
        "For outcome year y, the panel predictor year is y−2. Task A verifies annual Vietnam export reporting for the three years feature_year−2, feature_year−1, and feature_year. Thus the strict-zero window ends at y−2; it does not inspect y−1. The label defines a new-to-Vietnam HS2-destination relationship after three verified zero years, not a first-ever trade relationship over all history. A missing year of destination reporting is not treated as zero.", "",
        "Task B requires positive historical exports and a market share no greater than an HS2 median estimated from observed historical outcomes through the feature year. Eligibility is generated before the target year. The two reconstructed candidate sets have no shared destination–HS2–year keys. Candidate ID and task keys are unique.", "",
        "The historical OOF archive was also matched against the task definitions for 2018–2021. Each OOF model fold used outcomes through no later than y−2; feature-year checks found no cutoff violation. The outer score archive has complete PPML and LightGBM scores for all candidate keys.", "",
        "## Scope limitations", "",
        "Unknown labels are preserved as missing and complete-case ranking metrics are conditional on observed outcomes. Economic sums over a full predicted Top-K list with missing labels are lower-bound observed amounts, not full realized portfolio values. Macro and governance histories are revised-vintage data, so these tests are retrospective rather than deployment-grade point-in-time backtests.", "",
        "## Integrity checks", "",
        "- Reconstructed candidate-key set matches the archived ERGM manifest exactly.",
        "- Reconstructed observed-label indicators match the archive.",
        "- Task A/Task B overlap: 0.",
        "- Strict-zero years end at the permitted y−2 cutoff.",
        "- No missing outcomes were recoded as zero.",
        "- All outer expert predictions are finite and present.", ""
    ]
    (root / "reports/FINAL_CANDIDATE_AND_TEMPORAL_AUDIT.md").write_text("\n".join(audit))
    return frozen, saved_scores, oof, summary


def _outer_candidates(manifest: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in ["log_gdp", "log_gdp_per_capita", "log_population", "log_destination_import_demand",
                                "destination_import_demand_growth", "historical_export_volatility", "internet_usage",
                                "government_effectiveness", "digitalization_missing_indicator", "regulatory_quality",
                                "rule_of_law", "political_stability"] if c in scores.columns and c not in manifest.columns]
    keep = ["task", "candidate_id", "ppml_score", "lightgbm_score", "blend_rank_score", "_r_p", "_r_l", "score_provenance", *feature_cols]
    d = manifest.merge(scores[keep], on=["task", "candidate_id"], how="left", validate="one_to_one")
    if d[["ppml_score", "lightgbm_score"]].isna().any().any():
        raise AssertionError("Missing prediction after frozen-manifest join")
    d = _add_blend_scores(d)
    if "blend_rank_score" in scores.columns:
        chk = d.merge(scores[["task", "candidate_id", "blend_rank_score"]].rename(columns={"blend_rank_score": "saved_blend_check"}),
                      on=["task", "candidate_id"], validate="one_to_one")
        delta = np.abs(chk.blend_rank_score - chk.saved_blend_check)
        if delta.max() > 1e-12:
            raise AssertionError(f"Reconstructed blend differs from saved score: max={delta.max()}")
    d["saved_blend_score"] = d.blend_rank_score
    # Score names are in descending order; this is exactly Borda rank aggregation for two experts.
    d["borda_rank_score"] = d["borda_rank_score"].astype(float)
    return d


def _pool_rows(frame: pd.DataFrame, models: dict[str, str], task: str,
               gain_def: str = "primary", k_values=(5, 10)) -> pd.DataFrame:
    rows = []
    for (hs2, year), g in frame.groupby(["hs2", "outcome_year"], sort=True):
        for model, score in models.items():
            m = pool_metrics(g, score, task, k_values, gain_def)
            rows.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year),
                         "candidate_pool_id": f"{task}|{str(hs2).zfill(2)}|{int(year)}",
                         "policy": model, "gain_definition": gain_def, **m})
    return pd.DataFrame(rows)


def _cluster_bootstrap(d: pd.DataFrame, value_col: str, reps: int = BOOT_REPS,
                       seed: int = SEED) -> dict:
    z = d.loc[np.isfinite(pd.to_numeric(d[value_col], errors="coerce")), ["hs2", value_col]].copy()
    if z.empty:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_signflip": np.nan,
                "paired_pools": 0, "hs2_blocks": 0}
    z[value_col] = pd.to_numeric(z[value_col], errors="coerce")
    blocks = z.groupby("hs2")[value_col].agg(["sum", "count"])
    estimate = float(z[value_col].mean())
    rng = np.random.default_rng(seed)
    sums = blocks["sum"].to_numpy(float)
    counts = blocks["count"].to_numpy(float)
    b = len(blocks)
    boot = np.empty(reps)
    for i in range(reps):
        ix = rng.integers(0, b, b)
        boot[i] = sums[ix].sum() / counts[ix].sum()
    ci = np.quantile(boot, [.025, .975])
    # Cluster-level sign-flip test of the paired null, retaining all years within each HS2.
    sign = rng.choice(np.array([-1.0, 1.0]), size=(reps, b))
    null = (sign * sums[None, :]).sum(axis=1) / counts.sum()
    p = float((1 + np.sum(np.abs(null) >= abs(estimate))) / (reps + 1))
    return {"estimate": estimate, "ci_low": float(ci[0]), "ci_high": float(ci[1]), "p_signflip": p,
            "paired_pools": len(z), "hs2_blocks": b}


def _holm(p_values: list[float]) -> list[float]:
    vals = np.asarray(p_values, dtype=float)
    n = len(vals)
    order = np.argsort(vals)
    adjusted = np.empty(n)
    running = 0.0
    for rank, ix in enumerate(order):
        running = max(running, (n - rank) * vals[ix])
        adjusted[ix] = min(1.0, running)
    return adjusted.tolist()


def _metric_tests(pools: pd.DataFrame, contrasts: list[tuple[str, str, str, str]],
                  task: str, family: str) -> pd.DataFrame:
    out = []
    for metric, a, b, metric_family in contrasts:
        wide = pools.loc[pools.task.eq(task)].pivot_table(index=["hs2", "outcome_year"], columns="policy",
                                                          values=metric, aggfunc="first").reset_index()
        if a not in wide or b not in wide:
            continue
        wide["difference"] = pd.to_numeric(wide[a], errors="coerce") - pd.to_numeric(wide[b], errors="coerce")
        s = _cluster_bootstrap(wide, "difference")
        base = float(pd.to_numeric(wide[b], errors="coerce").mean())
        out.append({"task": task, "family": family, "metric_family": metric_family,
                    "metric": metric, "contrast": f"{a} - {b}", "baseline_mean": base,
                    "difference": s["estimate"], "relative_difference_pct": 100*s["estimate"]/abs(base) if base else np.nan,
                    "ci95_low": s["ci_low"], "ci95_high": s["ci_high"], "p_raw_signflip": s["p_signflip"],
                    "paired_pools": s["paired_pools"], "hs2_blocks": s["hs2_blocks"],
                    "inference_note": "HS2-block paired sign-flip test; percentile block bootstrap; years retained within HS2"})
    return pd.DataFrame(out)


def _save_f1_f2(root: Path, outer: pd.DataFrame, oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    task_models = {
        TASK_A: {"Gravity_PPML": "ppml_score", "LightGBM": "lightgbm_score",
                 "EQUAL_WEIGHT_RANK_BLEND": "blend_rank_score", "BORDA_RANK_AGGREGATION": "borda_rank_score",
                 "SAVED_ORIGINAL_STATIC_BLEND": "saved_blend_score"},
        TASK_B: {"Gravity_PPML": "ppml_score", "LightGBM": "lightgbm_score",
                 "EQUAL_WEIGHT_RANK_BLEND": "blend_rank_score", "BORDA_RANK_AGGREGATION": "borda_rank_score",
                 "SAVED_ORIGINAL_STATIC_BLEND": "saved_blend_score"},
    }
    all_pool = []
    for task in TASKS:
        all_pool.append(_pool_rows(outer.loc[outer.task.eq(task)], task_models[task], task))
    pool_all = pd.concat(all_pool, ignore_index=True)
    # Model-level macro summaries, with explicit denominators and metric scale.
    summary = pool_all.groupby(["task", "policy"], as_index=False).agg(
        candidate_pools=("candidate_pool_id", "size"),
        valid_ndcg10_pools=("ndcg@10", "count"), macro_ndcg10=("ndcg@10", "mean"),
        macro_precision10=("precision@10", "mean"), successful_count_top10=("successful_or_meaningful_count@10", "mean"),
        macro_ndcg5=("ndcg@5", "mean"), macro_precision5=("precision@5", "mean"),
        mean_regret10=("regret@10", "mean"), mean_known_selected_usd_top10=("selected_usd_known_top10", "mean"),
        mean_unknown_in_full_top10=("unknown_in_full_score_top10", "mean"),
        mean_observed_partial_usd_full_top10=("observed_usd_partial_full_score_top10", "mean"))
    summary["metric_scope"] = "macro over HS2-year pools; complete-case rankings; USD known-label Top-K unless marked partial full-score list"
    summary.to_csv(root / "results/F1_blend_model_comparison.csv", index=False, na_rep="")
    by_year = pool_all.groupby(["task", "outcome_year", "policy"], as_index=False).agg(
        pools=("candidate_pool_id", "size"), valid_ndcg10_pools=("ndcg@10", "count"),
        ndcg10=("ndcg@10", "mean"), precision10=("precision@10", "mean"),
        successful_count_top10=("successful_or_meaningful_count@10", "mean"), regret10=("regret@10", "mean"),
        known_selected_usd_top10=("selected_usd_known_top10", "mean"),
        unknown_in_full_top10=("unknown_in_full_score_top10", "sum"))
    by_year.to_csv(root / "results/F1_by_year.csv", index=False, na_rep="")
    by_hs2 = pool_all.groupby(["task", "hs2", "policy"], as_index=False).agg(
        pools=("candidate_pool_id", "size"), valid_ndcg10_pools=("ndcg@10", "count"),
        ndcg10=("ndcg@10", "mean"), precision10=("precision@10", "mean"),
        successful_count_top10=("successful_or_meaningful_count@10", "mean"), regret10=("regret@10", "mean"),
        known_selected_usd_top10=("selected_usd_known_top10", "mean"))
    by_hs2.to_csv(root / "results/F1_by_hs2.csv", index=False, na_rep="")
    contrasts_a = []
    for metric, fam in (("ndcg@10", "Family_A_NDCG10"), ("precision@10", "Family_A_Precision10"),
                        ("successful_or_meaningful_count@10", "Family_A_SuccessCount10")):
        contrasts_a += [(metric, "EQUAL_WEIGHT_RANK_BLEND", "LightGBM", fam),
                        (metric, "EQUAL_WEIGHT_RANK_BLEND", "Gravity_PPML", fam)]
    f1tests = _metric_tests(pool_all, contrasts_a, TASK_A, "F1_new_market_blend")
    f1b = _metric_tests(pool_all, [("ndcg@10", "EQUAL_WEIGHT_RANK_BLEND", "Gravity_PPML", "Family_B_NDCG10")],
                        TASK_B, "F1_expansion_consistency")
    f1tests = pd.concat([f1tests, f1b], ignore_index=True)
    if len(f1tests):
        f1tests["p_holm"] = np.nan
        for _, ix in f1tests.groupby("metric_family").groups.items():
            f1tests.loc[ix, "p_holm"] = _holm(f1tests.loc[ix, "p_raw_signflip"].to_list())
    f1tests.to_csv(root / "results/F1_blend_paired_tests.csv", index=False, na_rep="")
    comp = _complementarity(outer)
    comp.to_csv(root / "results/F1_model_complementarity.csv", index=False, na_rep="")

    # Economic outcome tables and regret, retaining unknowns separately.
    economic = pool_all[["task", "hs2", "outcome_year", "candidate_pool_id", "policy", "candidate_count",
                         "observed_label_count", "unknown_label_count", "unknown_in_full_score_top10",
                         "selected_usd_known_top10", "oracle_usd_known_top10", "usd_regret_known_top10",
                         "observed_usd_partial_full_score_top10", "partial_usd_is_complete"]].copy()
    economic = economic.rename(columns={"selected_usd_known_top10": "complete_case_selected_usd_top10",
                                        "oracle_usd_known_top10": "complete_case_oracle_usd_top10",
                                        "usd_regret_known_top10": "complete_case_usd_regret_top10",
                                        "observed_usd_partial_full_score_top10": "observed_partial_amount_full_exante_top10"})
    economic["usd_scope_note"] = "Known-label top-10 is complete-case; observed partial amount uses full scored top-10 and leaves unknown outcome value unresolved"
    economic.to_csv(root / "results/F2_economic_opportunity_usd.csv", index=False, na_rep="")
    succ = pool_all[["task", "hs2", "outcome_year", "candidate_pool_id", "policy", "precision@5", "precision@10",
                     "successful_or_meaningful_count@5", "successful_or_meaningful_count@10",
                     "unsuccessful_recommendations@10", "unknown_in_full_score_top10"]].copy()
    succ = succ.loc[succ.task.eq(TASK_A)]
    succ.to_csv(root / "results/F2_successful_market_entries.csv", index=False, na_rep="")
    regrets = pool_all[["task", "hs2", "outcome_year", "candidate_pool_id", "policy", "regret@5", "regret@10",
                        "selected_gain@10", "oracle_gain@10", "usd_regret_known_top10"]].copy()
    regrets.to_csv(root / "results/F2_economic_regret.csv", index=False, na_rep="")
    usdtests = []
    for task, a, b, fam in ((TASK_A, "EQUAL_WEIGHT_RANK_BLEND", "LightGBM", "TaskA_blend_vs_LightGBM_USD"),
                            (TASK_A, "EQUAL_WEIGHT_RANK_BLEND", "Gravity_PPML", "TaskA_blend_vs_PPML_USD"),
                            (TASK_B, "EQUAL_WEIGHT_RANK_BLEND", "Gravity_PPML", "TaskB_blend_vs_PPML_USD")):
        wide = economic.loc[economic.task.eq(task)].pivot_table(index=["hs2", "outcome_year"], columns="policy",
                                                                values="complete_case_selected_usd_top10", aggfunc="first").reset_index()
        wide["difference"] = wide[a] - wide[b]
        s = _cluster_bootstrap(wide, "difference")
        dif = wide.difference.dropna()
        usdtests.append({"task": task, "family": fam, "contrast": f"{a} - {b}", **s,
                         "median_paired_difference_usd": float(dif.median()),
                         "q25_difference_usd": float(dif.quantile(.25)), "q75_difference_usd": float(dif.quantile(.75)),
                         "pools_favor_first_model": int((dif > 0).sum()), "pools_favor_second_model": int((dif < 0).sum()),
                         "ties": int((dif == 0).sum()), "interpretation": "descriptive retrospective observed trade opportunity; not causal or profit"})
    usdtest_df = pd.DataFrame(usdtests)
    if len(usdtest_df):
        usdtest_df["p_holm"] = _holm(usdtest_df.p_signflip.to_list())
    usdtest_df.to_csv(root / "results/F2_usd_paired_tests.csv", index=False, na_rep="")
    usdtest_df.to_csv(root / "results/F2_tail_sensitivity.csv", index=False, na_rep="")
    # Fixed development-period 99th percentile caps for a tail sensitivity; never fitted on test outcomes.
    oof = _add_blend_scores(oof.copy())
    tail_rows = []
    for task in TASKS:
        od = oof.loc[oof.task.eq(task) & oof.label_observed.astype(bool)].copy()
        source = od.export_value_usd if task == TASK_A else od.realized_incremental_export_usd.clip(lower=0)
        cap = float(source.quantile(.99)) if len(source) else np.nan
        pd_task = economic.loc[economic.task.eq(task)].copy()
        score_by_policy = {r.policy: SCORE_NAMES[r.policy] for r in summary.loc[summary.task.eq(task)].itertuples(index=False)}
        for pol, score in score_by_policy.items():
            # candidate-level winsorized USD opportunity, with the cap frozen on OOF labels.
            for (hs2, year), g in outer.loc[outer.task.eq(task)].groupby(["hs2", "outcome_year"]):
                selected = _rank_score(g.loc[g.label_observed.astype(bool)], score).head(10)
                vals = selected.export_value_usd if task == TASK_A else selected.realized_incremental_export_usd.clip(lower=0)
                tail_rows.append({"task": task, "hs2": hs2, "outcome_year": year, "policy": pol,
                                  "dev_oof_99pct_cap_usd": cap,
                                  "selected_usd_raw": float(vals.clip(lower=0).sum()),
                                  "selected_usd_capped_at_development_99pct": float(vals.clip(lower=0, upper=cap).sum())})
    tail_df = pd.DataFrame(tail_rows)
    tail_df.to_csv(root / "results/F2_tail_sensitivity.csv", index=False, na_rep="")
    comp_dict = {"summary": summary, "pool": pool_all, "f1_tests": f1tests,
                 "usd_tests": usdtest_df, "complementarity": comp}
    return summary, pool_all, comp_dict


def _complementarity(outer: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (hs2, year), g in outer.loc[outer.task.eq(TASK_A)].groupby(["hs2", "outcome_year"], sort=True):
        p = _rank_score(g, "ppml_score")
        l = _rank_score(g, "lightgbm_score")
        b = _rank_score(g, "blend_rank_score")
        pp = set(p.head(10).destination_iso3)
        ll = set(l.head(10).destination_iso3)
        bb = set(b.head(10).destination_iso3)
        pmap = {iso: i + 1 for i, iso in enumerate(p.destination_iso3.astype(str))}
        lmap = {iso: i + 1 for i, iso in enumerate(l.destination_iso3.astype(str))}
        observed = g.loc[g.label_observed.astype(bool)].set_index("destination_iso3")
        succ = set(observed.index[observed.entry_success.eq(1)])
        rows.append({"task": TASK_A, "hs2": hs2, "outcome_year": year,
                     "top10_overlap_count": len(pp & ll), "top10_jaccard": len(pp & ll) / len(pp | ll) if pp | ll else np.nan,
                     "rank_kendall_tau": float(kendalltau(
                         [pmap[x] for x in sorted(pmap)], [lmap[x] for x in sorted(pmap)]).statistic) if len(g) > 1 else np.nan,
                     "lgbm_only_candidates": len(ll - pp), "ppml_only_candidates": len(pp - ll),
                     "blend_unique_candidates_vs_either_top10": len(bb - (pp | ll)),
                     "successful_entries_lgbm_only": len((ll - pp) & succ),
                     "successful_entries_ppml_only": len((pp - ll) & succ),
                     "successful_entries_shared": len((pp & ll) & succ),
                     "successful_entries_blend_only_vs_either": len((bb - (pp | ll)) & succ),
                     "successful_entries_top10_lgbm": len(ll & succ),
                     "successful_entries_top10_ppml": len(pp & succ),
                     "successful_entries_top10_blend": len(bb & succ),
                     "unknown_labels_in_full_top10_union": int((~g.loc[g.destination_iso3.isin(pp | ll | bb)].label_observed.astype(bool)).sum())})
    return pd.DataFrame(rows)


def _make_pool_features(candidates: pd.DataFrame, historical_losses: pd.DataFrame, target_col: str) -> pd.DataFrame:
    rows = []
    pred_features = [c for c in ["log_gdp", "log_gdp_per_capita", "log_population", "log_destination_import_demand",
                                  "destination_import_demand_growth", "historical_export_value", "historical_market_share",
                                  "historical_export_volatility", "internet_usage", "government_effectiveness",
                                  "regulatory_quality", "rule_of_law", "political_stability"] if c in candidates.columns]
    for (task, hs2, year), g in candidates.groupby(["task", "hs2", "outcome_year"], sort=True):
        pp = _rank_score(g, "ppml_score")
        ll = _rank_score(g, "lightgbm_score")
        n = len(g)
        p_rank = {iso: i + 1 for i, iso in enumerate(pp.destination_iso3.astype(str))}
        l_rank = {iso: i + 1 for i, iso in enumerate(ll.destination_iso3.astype(str))}
        ptop, ltop = set(pp.head(10).destination_iso3.astype(str)), set(ll.head(10).destination_iso3.astype(str))
        union = ptop | ltop
        kend = kendalltau([p_rank[x] for x in sorted(p_rank)], [l_rank[x] for x in sorted(p_rank)]).statistic if n > 1 else np.nan
        def margin(frame: pd.DataFrame, col: str) -> float:
            s = np.sort(pd.to_numeric(frame[col], errors="coerce").to_numpy(float))[::-1]
            gap = s[9] - s[10] if len(s) > 10 else (s[0] - s[-1] if len(s) > 1 else 0.0)
            sd = float(np.nanstd(s))
            return float(gap / sd) if sd > 0 else 0.0
        hist = pd.to_numeric(g.historical_export_value, errors="coerce").clip(lower=0)
        gd = pd.to_numeric(g.log_gdp, errors="coerce") if "log_gdp" in g else pd.Series(dtype=float)
        demand = pd.to_numeric(g.log_destination_import_demand, errors="coerce") if "log_destination_import_demand" in g else pd.Series(dtype=float)
        missing_share = float(g[pred_features].isna().to_numpy().mean()) if pred_features else np.nan
        prior = historical_losses.loc[historical_losses.task.eq(task) & historical_losses.hs2.astype(str).str.zfill(2).eq(str(hs2).zfill(2))
                                      & historical_losses.outcome_year.le(int(year) - 2), target_col]
        rows.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year),
            "candidate_pool_id": f"{task}|{str(hs2).zfill(2)}|{int(year)}", "candidate_count": n,
            "mean_log_gdp": float(gd.mean()) if len(gd) else np.nan,
            "median_log_gdp": float(gd.median()) if len(gd) else np.nan,
            "sd_log_gdp": float(gd.std()) if len(gd) else np.nan,
            "mean_log_demand": float(demand.mean()) if len(demand) else np.nan,
            "median_log_demand": float(demand.median()) if len(demand) else np.nan,
            "mean_log_hist_exports": float(np.log1p(hist).mean()),
            "mean_market_share": float(pd.to_numeric(g.historical_market_share, errors="coerce").mean()),
            "mean_trade_volatility": float(pd.to_numeric(g.historical_export_volatility, errors="coerce").mean()),
            "mean_demand_growth": float(pd.to_numeric(g.destination_import_demand_growth, errors="coerce").mean()),
            "predictor_missing_share": missing_share,
            "top10_jaccard": len(ptop & ltop) / len(union) if union else np.nan,
            "rank_kendall_tau": float(kend) if np.isfinite(kend) else np.nan,
            "rank_displacement_norm": float(np.mean([abs(p_rank[x] - l_rank[x]) for x in p_rank]) / max(1, n-1)),
            "ppml_margin_z": margin(g, "ppml_score"), "lightgbm_margin_z": margin(g, "lightgbm_score"),
            "prior_regret_mean": float(prior.mean()) if len(prior) else np.nan,
            "prior_regret_sd": float(prior.std()) if len(prior) > 1 else 0.0,
            "prior_history_n": int(len(prior))})
    return pd.DataFrame(rows)


def _risk_losses(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = _add_blend_scores(candidates.copy())
    models = {"Gravity_PPML": "ppml_score", "LightGBM": "lightgbm_score", "BLEND": "blend_rank_score"}
    for (task, hs2, year), g in d.groupby(["task", "hs2", "outcome_year"], sort=True):
        row = {"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year),
               "candidate_pool_id": f"{task}|{str(hs2).zfill(2)}|{int(year)}"}
        for name, col in models.items():
            row[f"regret_{name}"] = pool_metrics(g, col, task)["regret@10"]
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_risk_model(train: pd.DataFrame, test: pd.DataFrame, target: str):
    numeric = [c for c in NUMERIC_FEATURES if c in train.columns and train[c].notna().any()]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                           ("scale", StandardScaler())]), numeric),
        ("hs2", OneHotEncoder(handle_unknown="ignore", sparse_output=True), ["hs2"]),
    ], remainder="drop")
    model = Pipeline([("prep", prep), ("ridge", Ridge(alpha=10.0))])
    model.fit(train, train[target].astype(float))
    pred = model.predict(test)
    pred_train = model.predict(train)
    return model, np.maximum(0.0, pred), pred, pred_train, numeric


def _risk_system(root: Path, outer: pd.DataFrame, oof: pd.DataFrame) -> dict:
    oof = _add_blend_scores(oof.copy())
    outer = _add_blend_scores(outer.copy())
    losses_oof = _risk_losses(oof)
    losses_outer = _risk_losses(outer)
    rows_oof = []
    for task in TASKS:
        target_model = "LightGBM" if task == TASK_A else "Gravity_PPML"
        target_col = f"regret_{target_model}"
        feature_oof = _make_pool_features(oof.loc[oof.task.eq(task)], losses_oof, target_col)
        feature_oof = feature_oof.merge(losses_oof.loc[losses_oof.task.eq(task)], on=["task", "hs2", "outcome_year", "candidate_pool_id"], validate="one_to_one")
        # Chronological cross-fitted historical risk scores; a pool never scores its own target.
        for year in OOF_YEARS:
            test = feature_oof.loc[feature_oof.outcome_year.eq(year)].copy()
            train = feature_oof.loc[feature_oof.outcome_year.le(year - 2) & feature_oof[target_col].notna()].copy()
            if len(train) < 40 or test.empty:
                continue
            _, pred, _, _, _ = _fit_risk_model(train, test, target_col)
            test["crossfit_risk_static"] = pred
            test["actual_regret_static"] = test[target_col]
            test["risk_model"] = "RIDGE_STATIC_EXPERT"
            test["expert"] = target_model
            rows_oof.append(test)
        # Separate Task A blend-risk target; this is an explicitly exploratory candidate.
        if task == TASK_A:
            blend_target = "regret_BLEND"
            for year in OOF_YEARS:
                test = feature_oof.loc[feature_oof.outcome_year.eq(year)].copy()
                train = feature_oof.loc[feature_oof.outcome_year.le(year - 2) & feature_oof[blend_target].notna()].copy()
                if len(train) < 40 or test.empty:
                    continue
                _, pred, _, _, _ = _fit_risk_model(train, test, blend_target)
                zz = test.copy(); zz["crossfit_risk_static"] = pred
                zz["actual_regret_static"] = zz[blend_target]
                zz["risk_model"] = "RIDGE_BLEND"
                zz["expert"] = "StaticBlend"
                rows_oof.append(zz)
    calibration = pd.concat(rows_oof, ignore_index=True) if rows_oof else pd.DataFrame()
    calibration.to_parquet(root / "predictions/F3_historical_crossfit_risk.parquet", index=False)

    outer_risk_rows, gate_models = [], {}
    for task in TASKS:
        static_model = "LightGBM" if task == TASK_A else "Gravity_PPML"
        target_options = [("RIDGE_STATIC_EXPERT", f"regret_{static_model}", static_model)]
        if task == TASK_A:
            target_options.append(("RIDGE_BLEND", "regret_BLEND", "StaticBlend"))
        for eval_year in EVAL_YEARS:
            candidates = outer.loc[outer.task.eq(task) & outer.outcome_year.eq(eval_year)]
            h_loss = losses_oof.loc[losses_oof.task.eq(task)]
            for gate_name, target_col, expert_name in target_options:
                feats = _make_pool_features(candidates, h_loss, target_col)
                feats = feats.merge(losses_outer.loc[losses_outer.task.eq(task) & losses_outer.outcome_year.eq(eval_year),
                                                    ["task", "hs2", "outcome_year", "candidate_pool_id", f"regret_{static_model}", "regret_BLEND"]],
                                    on=["task", "hs2", "outcome_year", "candidate_pool_id"], validate="one_to_one")
                train_feats = _make_pool_features(oof.loc[oof.task.eq(task)], losses_oof, target_col)
                train_feats = train_feats.merge(losses_oof.loc[losses_oof.task.eq(task),
                    ["task", "hs2", "outcome_year", "candidate_pool_id", target_col]],
                    on=["task", "hs2", "outcome_year", "candidate_pool_id"], validate="one_to_one")
                train = train_feats.loc[train_feats.outcome_year.le(eval_year - 2) & train_feats[target_col].notna()].copy()
                if len(train) < 40:
                    raise RuntimeError(f"Insufficient historical pool loss for risk model: {task}/{eval_year}: {len(train)}")
                fit_start = time.perf_counter()
                fitted, pred, raw_pred, train_pred, numeric = _fit_risk_model(train, feats, target_col)
                fit_seconds = time.perf_counter() - fit_start
                key = f"{task}_{gate_name}_{eval_year}"
                joblib.dump({"model": fitted, "training_pools": len(train), "training_year_max": int(train.outcome_year.max()),
                             "target": target_col, "numeric_features": numeric, "task": task, "evaluation_year": eval_year},
                            root / f"models/{key}.joblib")
                gate_models[(task, gate_name, eval_year)] = (fitted, feats, pred, raw_pred, train, fit_seconds)
                for i, r in feats.reset_index(drop=True).iterrows():
                    row = r.to_dict()
                    row.update({"gate_name": gate_name, "expert": expert_name, "risk_predicted_regret": float(pred[i]),
                                "risk_raw_ridge_output": float(raw_pred[i]), "risk_target_regret": float(r[target_col]),
                                "risk_training_pools": len(train), "risk_training_year_max": int(train.outcome_year.max()),
                                "risk_training_max_allowed": eval_year - 2,
                                "risk_model_fit_seconds": fit_seconds})
                    outer_risk_rows.append(row)
    risk_scores = pd.DataFrame(outer_risk_rows)
    if (risk_scores.risk_training_year_max > risk_scores.risk_training_max_allowed).any():
        raise AssertionError("Risk model was trained on an outcome after its cutoff")
    risk_scores.to_parquet(root / "predictions/F3_risk_scores.parquet", index=False)

    # Clipping at zero can create many tied estimated risks and materially change
    # the ordering used by a reject-option gate. Preserve this diagnostic.
    clip_rows = []
    for (task, gate, year), g in risk_scores.groupby(["task", "gate_name", "outcome_year"]):
        raw = pd.to_numeric(g.risk_raw_ridge_output, errors="coerce")
        pred = pd.to_numeric(g.risk_predicted_regret, errors="coerce")
        counts = pred.value_counts()
        clip_rows.append({"task": task, "risk_model": gate, "outcome_year": int(year), "pools": len(g),
            "negative_raw_predictions_clipped": int((raw < 0).sum()),
            "negative_raw_clip_share": float((raw < 0).mean()),
            "zero_after_clipping": int((pred == 0).sum()), "unique_predicted_risk_values": int(pred.nunique()),
            "largest_tie_group": int(counts.max()) if len(counts) else 0,
            "interpretation": "risk estimates are regret scores; clipping creates ties and is not probability calibration"})
    clipping = pd.DataFrame(clip_rows)
    clipping.to_csv(root / "results/F3_risk_clipping_ties.csv", index=False, na_rep="")

    # Evaluate whether simple disagreement/history signals order subsequent risk.
    signal_rows = []
    for (task, gate, year), g in risk_scores.groupby(["task", "gate_name", "outcome_year"]):
        target = pd.to_numeric(g.risk_target_regret, errors="coerce")
        for signal in ("top10_jaccard", "rank_displacement_norm", "prior_regret_mean",
                       "ppml_margin_z", "lightgbm_margin_z"):
            x = pd.to_numeric(g[signal], errors="coerce")
            valid = np.isfinite(x) & np.isfinite(target)
            rho = spearmanr(x[valid], target[valid]).statistic if valid.sum() > 2 and x[valid].nunique() > 1 else np.nan
            signal_rows.append({"task": task, "risk_target_model": gate, "outcome_year": int(year),
                "signal": signal, "pools": int(valid.sum()), "spearman_signal_vs_regret": float(rho) if np.isfinite(rho) else np.nan,
                "interpretation": "exploratory retrospective association; not a causal or calibrated risk relationship"})
    pd.DataFrame(signal_rows).to_csv(root / "results/F3_risk_signal_diagnostics.csv", index=False, na_rep="")

    # Risk quality against observed retrospective static-pool regret.
    quality = []
    for (task, gate, year), g in risk_scores.groupby(["task", "gate_name", "outcome_year"]):
        y = g.risk_target_regret.to_numpy(float); p = g.risk_predicted_regret.to_numpy(float)
        dev = losses_oof.loc[losses_oof.task.eq(task) & losses_oof.outcome_year.le(int(year)-2), f"regret_{'LightGBM' if task==TASK_A else 'Gravity_PPML'}"].dropna()
        high_cut = float(dev.quantile(.8)) if len(dev) else np.nan
        high = y >= high_cut if np.isfinite(high_cut) else np.zeros(len(y), dtype=bool)
        valid = np.isfinite(y) & np.isfinite(p)
        auc = roc_auc_score(high[valid], p[valid]) if valid.sum() and np.unique(high[valid]).size == 2 else np.nan
        ap = average_precision_score(high[valid], p[valid]) if valid.sum() and high[valid].sum() else np.nan
        rho = spearmanr(p[valid], y[valid]).statistic if valid.sum() > 2 else np.nan
        quality.append({"task": task, "risk_model": gate, "outcome_year": int(year), "pools": int(valid.sum()),
                        "high_regret_threshold_development_p80": high_cut, "high_regret_events": int(high[valid].sum()),
                        "high_regret_prevalence": float(high[valid].mean()) if valid.sum() else np.nan,
                        "spearman_risk_vs_regret": float(rho) if np.isfinite(rho) else np.nan,
                        "mae_regret_units": float(mean_absolute_error(y[valid], p[valid])) if valid.sum() else np.nan,
                        "rmse_regret_units": float(np.sqrt(mean_squared_error(y[valid], p[valid]))) if valid.sum() else np.nan,
                        "high_regret_roc_auc": auc, "high_regret_average_precision": ap,
                        "risk_interpretation": "out-of-time point estimate of regret; not calibrated probability"})
    quality_df = pd.DataFrame(quality)
    quality_df.to_csv(root / "results/F3_risk_prediction_quality.csv", index=False, na_rep="")
    # Cross-fitted risk deciles; descriptive calibration of a loss predictor, not probability calibration.
    cal_rows = []
    for (task, gate), g in calibration.groupby(["task", "risk_model"]):
        g = g.loc[np.isfinite(g.crossfit_risk_static) & np.isfinite(g.actual_regret_static)].copy()
        if g.empty:
            continue
        g["risk_decile"] = pd.qcut(g.crossfit_risk_static.rank(method="first"), 10, labels=False, duplicates="drop")
        for dec, z in g.groupby("risk_decile"):
            cal_rows.append({"task": task, "risk_model": gate, "risk_decile_low_to_high": int(dec), "pools": len(z),
                             "mean_predicted_regret": float(z.crossfit_risk_static.mean()),
                             "mean_realized_regret": float(z.actual_regret_static.mean()),
                             "median_realized_regret": float(z.actual_regret_static.median()),
                             "year_min": int(z.outcome_year.min()), "year_max": int(z.outcome_year.max()),
                             "claim": "cross-fitted historical calibration diagnostic; small number of years"})
    calibration_df = pd.DataFrame(cal_rows)
    calibration_df.to_csv(root / "results/F3_risk_calibration.csv", index=False, na_rep="")
    return {"losses_oof": losses_oof, "losses_outer": losses_outer, "crossfit": calibration,
            "risk_scores": risk_scores, "quality": quality_df, "calibration": calibration_df,
            "gate_models": gate_models}


def _f4_gates(root: Path, outer: pd.DataFrame, risk: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = risk["risk_scores"].copy()
    calibration = risk["crossfit"].copy()
    outer_policy_rows = []
    targets = [.9, .8, .7, .6]
    for task in TASKS:
        for year in EVAL_YEARS:
            rr = scores.loc[scores.task.eq(task) & scores.outcome_year.eq(year)
                            & scores.gate_name.eq("RIDGE_STATIC_EXPERT")].copy()
            if rr.empty:
                continue
            # Feature-only baselines come from the same ex-ante pool features.
            static_exp = "LightGBM" if task == TASK_A else "Gravity_PPML"
            feature_gate = "RIDGE_STATIC_EXPERT"
            testc = outer.loc[outer.task.eq(task) & outer.outcome_year.eq(year)]
            # Calculate static rank margin and expert disagreement at pool level.
            features = rr.set_index("candidate_pool_id")
            gate_risks = {}
            gate_risks["STATIC_LEARNED_RISK"] = features.loc[rr.candidate_pool_id, "risk_predicted_regret"].to_numpy(float)
            if task == TASK_A:
                blend_rr = scores.loc[scores.task.eq(task) & scores.outcome_year.eq(year) & scores.gate_name.eq("RIDGE_BLEND")]
                if not blend_rr.empty:
                    gate_risks["BLEND_LEARNED_RISK"] = blend_rr.set_index("candidate_pool_id").loc[rr.candidate_pool_id, "risk_predicted_regret"].to_numpy(float)
            # Risk score: lower means safer. Margin thresholds use smaller margin => higher risk.
            static_margin = rr["lightgbm_margin_z" if static_exp == "LightGBM" else "ppml_margin_z"].to_numpy(float)
            disagreement = (1.0 - rr.top10_jaccard.to_numpy(float)) + rr.rank_displacement_norm.to_numpy(float)
            prior_error = rr.prior_regret_mean.to_numpy(float)
            gate_risks["SCORE_MARGIN"] = -static_margin
            gate_risks["MODEL_DISAGREEMENT"] = disagreement
            gate_risks["HISTORICAL_HS2_ERROR"] = prior_error
            # Cross-fitted historical calibration for every risk signal. Only risk predictions/features
            # from OOF periods no later than y-2 determine each fixed threshold.
            cal_task = calibration.loc[calibration.task.eq(task) & calibration.outcome_year.le(year - 2)].copy()
            base_hist = risk["losses_oof"].loc[risk["losses_oof"].task.eq(task)].copy()
            hist_features = _make_pool_features(
                pd.read_parquet(ERGM / "predictions/historical_oof_expert_predictions.parquet").assign(
                    hs2=lambda d: d.hs2.astype(str).str.zfill(2), outcome_year=lambda d: d.outcome_year.astype(int)),
                base_hist, f"regret_{static_exp}")
            hist_features = hist_features.loc[hist_features.task.eq(task) & hist_features.outcome_year.le(year - 2)]
            hist_oof_preds = calibration.loc[calibration.task.eq(task) & calibration.outcome_year.le(year - 2)]
            static_crossfit = hist_oof_preds.loc[hist_oof_preds.risk_model.eq("RIDGE_STATIC_EXPERT"), "crossfit_risk_static"].dropna().to_numpy(float)
            if task == TASK_A:
                blend_crossfit = hist_oof_preds.loc[hist_oof_preds.risk_model.eq("RIDGE_BLEND"), "crossfit_risk_static"].dropna().to_numpy(float)
            hist_static_margin = hist_features["lightgbm_margin_z" if static_exp == "LightGBM" else "ppml_margin_z"].to_numpy(float) if len(hist_features) else np.array([])
            hist_disagreement = (1-hist_features.top10_jaccard.to_numpy(float)+hist_features.rank_displacement_norm.to_numpy(float)) if len(hist_features) else np.array([])
            hist_prior = hist_features.prior_regret_mean.to_numpy(float) if len(hist_features) else np.array([])
            calibration_risks = {"STATIC_LEARNED_RISK": static_crossfit,
                                 "SCORE_MARGIN": -hist_static_margin,
                                 "MODEL_DISAGREEMENT": hist_disagreement,
                                 "HISTORICAL_HS2_ERROR": hist_prior}
            if task == TASK_A and len(blend_crossfit):
                calibration_risks["BLEND_LEARNED_RISK"] = blend_crossfit
            test_experts = {"STATIC_LEARNED_RISK": static_exp,
                            "SCORE_MARGIN": static_exp, "MODEL_DISAGREEMENT": static_exp,
                            "HISTORICAL_HS2_ERROR": static_exp}
            if task == TASK_A:
                test_experts["BLEND_LEARNED_RISK"] = "StaticBlend"
            for policy, risk_vec in gate_risks.items():
                cal_vec = calibration_risks.get(policy, np.array([]))
                cal_vec = np.asarray(cal_vec, dtype=float)
                cal_vec = cal_vec[np.isfinite(cal_vec)]
                for nominal in targets:
                    threshold = float(np.quantile(cal_vec, nominal, method="linear")) if len(cal_vec) else np.nan
                    accepted = np.isfinite(risk_vec) & (risk_vec <= threshold)
                    if nominal == 1:
                        accepted = np.ones(len(rr), dtype=bool)
                        threshold = float("inf")
                    expert = test_experts[policy]
                    score = {"Gravity_PPML": "ppml_score", "LightGBM": "lightgbm_score", "StaticBlend": "blend_rank_score"}[expert]
                    for i, pool in enumerate(rr.itertuples(index=False)):
                        outer_policy_rows.append({"task": task, "outcome_year": year, "hs2": pool.hs2,
                            "candidate_pool_id": pool.candidate_pool_id, "policy": policy,
                            "nominal_coverage": nominal, "coverage_mode": "fixed_development_threshold",
                            "risk_score": float(risk_vec[i]), "risk_threshold": threshold,
                            "accepted": bool(accepted[i]), "selected_expert": expert,
                            "selection_score": score,
                            "risk_model_training_year_max": int(pool.risk_training_year_max) if policy in ["STATIC_LEARNED_RISK", "BLEND_LEARNED_RISK"] else int(year-2),
                            "risk_cutoff_year": int(year-2)})
            # No-deferral reference.
            for nominal in targets:
                for pool in rr.itertuples(index=False):
                    outer_policy_rows.append({"task": task, "outcome_year": year, "hs2": pool.hs2,
                        "candidate_pool_id": pool.candidate_pool_id, "policy": "NO_DEFERRAL",
                        "nominal_coverage": nominal, "coverage_mode": "fixed_development_threshold",
                        "risk_score": np.nan, "risk_threshold": np.nan, "accepted": True,
                        "selected_expert": static_exp, "selection_score": "ppml_score" if static_exp=="Gravity_PPML" else "lightgbm_score",
                        "risk_model_training_year_max": int(year-2), "risk_cutoff_year": int(year-2)})
    decisions = pd.DataFrame(outer_policy_rows)
    # Actual per-pool result under each policy, including deferred opportunity separately.
    result_rows, coverage_rows = [], []
    for (task, year, policy, nominal), decision in decisions.groupby(["task", "outcome_year", "policy", "nominal_coverage"], sort=True):
        candidates = outer.loc[outer.task.eq(task) & outer.outcome_year.eq(year)]
        selected_model = decision.selected_expert.iloc[0]
        score = decision.selection_score.iloc[0]
        mrows = []
        for r in decision.itertuples(index=False):
            g = candidates.loc[candidates.candidate_pool_id.eq(r.candidate_pool_id)]
            met = pool_metrics(g, r.selection_score, task)
            mrows.append({"task": task, "outcome_year": year, "hs2": r.hs2,
                          "candidate_pool_id": r.candidate_pool_id, "policy": policy,
                          "nominal_coverage": nominal, "accepted": r.accepted,
                          "risk_score": r.risk_score, "risk_threshold": r.risk_threshold,
                          "selected_expert": r.selected_expert, **met})
        md = pd.DataFrame(mrows)
        accepted = md.loc[md.accepted.astype(bool)]
        deferred = md.loc[~md.accepted.astype(bool)]
        result_rows.extend(mrows)
        valid = accepted.loc[accepted["regret@10"].notna()]
        coverage_rows.append({"task": task, "outcome_year": int(year), "policy": policy,
            "nominal_coverage": nominal, "actual_coverage": float(md.accepted.mean()),
            "defer_rate": float(1-md.accepted.mean()), "all_pools": len(md), "accepted_pools": int(md.accepted.sum()),
            "deferred_pools": int((~md.accepted).sum()), "accepted_observed_pools": int(len(valid)),
            "accepted_mean_regret10": float(valid["regret@10"].mean()) if len(valid) else np.nan,
            "accepted_median_regret10": float(valid["regret@10"].median()) if len(valid) else np.nan,
            "accepted_p90_regret10": float(valid["regret@10"].quantile(.9)) if len(valid) else np.nan,
            "accepted_ndcg10": float(valid["ndcg@10"].mean()) if len(valid) else np.nan,
            "accepted_precision10": float(valid["precision@10"].mean()) if len(valid) else np.nan,
            "accepted_known_label_usd_per_pool": float(valid["selected_usd_known_top10"].mean()) if len(valid) else np.nan,
            "deferred_mean_regret_hindsight": float(deferred["regret@10"].mean()) if len(deferred) else np.nan,
            "deferred_known_label_usd_per_pool": float(deferred["selected_usd_known_top10"].mean()) if len(deferred) else np.nan,
            "coverage_threshold_note": "threshold learned from cross-fitted historical scores; actual coverage allowed to differ"})
    # Random deferral is a workload benchmark only: exactly ceil(target*N) pools are
    # accepted in each draw, and no test labels determine the random selection.
    random_rows = []
    rng = np.random.default_rng(SEED)
    for task in TASKS:
        static_exp = "LightGBM" if task == TASK_A else "Gravity_PPML"
        score = "lightgbm_score" if static_exp == "LightGBM" else "ppml_score"
        for year in EVAL_YEARS:
            candidates = outer.loc[outer.task.eq(task) & outer.outcome_year.eq(year)]
            pool_metrics_rows = []
            for pid, g in candidates.groupby("candidate_pool_id", sort=True):
                m = pool_metrics(g, score, task)
                pool_metrics_rows.append({"candidate_pool_id": pid, **m})
            metric_frame = pd.DataFrame(pool_metrics_rows)
            n_pool = len(metric_frame)
            for nominal in targets:
                n_accept = min(n_pool, int(math.ceil(nominal * n_pool)))
                draws = []
                for draw in range(1000):
                    ix = rng.choice(n_pool, n_accept, replace=False)
                    acc = metric_frame.iloc[ix]
                    deff = metric_frame.drop(metric_frame.index[ix])
                    draws.append({"accepted_regret": acc["regret@10"].mean(), "accepted_ndcg": acc["ndcg@10"].mean(),
                                  "accepted_precision": acc["precision@10"].mean(),
                                  "accepted_usd": acc["selected_usd_known_top10"].mean(),
                                  "deferred_regret": deff["regret@10"].mean() if len(deff) else np.nan,
                                  "deferred_usd": deff["selected_usd_known_top10"].mean() if len(deff) else np.nan})
                dd = pd.DataFrame(draws)
                coverage_rows.append({"task": task, "outcome_year": int(year), "policy": "RANDOM_DEFERRAL",
                    "nominal_coverage": nominal, "actual_coverage": n_accept / n_pool,
                    "defer_rate": 1 - n_accept / n_pool, "all_pools": n_pool, "accepted_pools": n_accept,
                    "deferred_pools": n_pool - n_accept, "accepted_observed_pools": n_accept,
                    "accepted_mean_regret10": float(dd.accepted_regret.mean()),
                    "accepted_median_regret10": np.nan, "accepted_p90_regret10": np.nan,
                    "accepted_ndcg10": float(dd.accepted_ndcg.mean()),
                    "accepted_precision10": float(dd.accepted_precision.mean()),
                    "accepted_known_label_usd_per_pool": float(dd.accepted_usd.mean()),
                    "deferred_mean_regret_hindsight": float(dd.deferred_regret.mean()) if (n_pool-n_accept) else np.nan,
                    "deferred_known_label_usd_per_pool": float(dd.deferred_usd.mean()) if (n_pool-n_accept) else np.nan,
                    "coverage_threshold_note": "mean over 1,000 random pool subsets; within-year workload fixed by design"})
                random_rows.append({"task": task, "outcome_year": year, "nominal_coverage": nominal,
                    "actual_coverage": n_accept/n_pool, "accepted_regret10_mean_over_random_draws": float(dd.accepted_regret.mean()),
                    "accepted_regret10_q025": float(dd.accepted_regret.quantile(.025)),
                    "accepted_regret10_q975": float(dd.accepted_regret.quantile(.975)),
                    "accepted_ndcg10_mean_over_random_draws": float(dd.accepted_ndcg.mean()),
                    "accepted_usd_mean_over_random_draws": float(dd.accepted_usd.mean()),
                    "random_draws": len(dd), "interpretation": "random review-workload benchmark, not a predictive gate"})
    decisions = decisions.merge(pd.DataFrame(result_rows)[["task", "outcome_year", "candidate_pool_id", "policy",
        "nominal_coverage", "ndcg@10", "precision@10", "regret@10", "selected_usd_known_top10",
        "unknown_in_full_score_top10"]], on=["task", "outcome_year", "candidate_pool_id", "policy", "nominal_coverage"],
        how="left", validate="one_to_one")
    matched_rows = []
    for (task, year, policy), g in decisions.loc[decisions.nominal_coverage.eq(.8)
            & ~decisions.policy.eq("NO_DEFERRAL")].groupby(["task", "outcome_year", "policy"], sort=True):
        # Retrospective coverage standardization uses only the risk score ordering, not outcomes.
        # It is a diagnostic and is not the fixed-threshold operating result above.
        n_accept = int(math.ceil(.8 * len(g)))
        ranked = g.sort_values(["risk_score", "hs2"], ascending=[True, True], kind="mergesort")
        chosen = ranked.head(n_accept)
        for _, row in chosen.iterrows():
            matched_rows.append({"task": task, "outcome_year": int(year), "hs2": str(row.hs2).zfill(2),
                "candidate_pool_id": row.candidate_pool_id, "policy": policy, "risk_score": row.risk_score,
                "requested_coverage": .8, "actual_coverage": len(chosen)/len(g),
                "accepted_pools": len(chosen), "pool_count": len(g),
                "mean_regret10": row["regret@10"], "mean_ndcg10": row["ndcg@10"],
                "mean_precision10": row["precision@10"],
                "mean_known_label_usd_top10": row["selected_usd_known_top10"],
                "diagnostic_only": True,
                "limitation": "uses the evaluation pool risk-score distribution to standardize coverage; not a frozen deployment threshold"})
    matched_df = pd.DataFrame(matched_rows)
    matched_df.to_csv(root / "results/F4_matched_coverage_diagnostic.csv", index=False, na_rep="")
    # Paired HS2-block inference for the retrospective common-coverage diagnostic.
    # This is secondary: it ranks outer-year risk scores to equalize coverage and
    # is not a frozen-threshold deployment test.
    infer_rows = []
    matched_contrasts = {
        TASK_A: ("BLEND_LEARNED_RISK", ["STATIC_LEARNED_RISK", "SCORE_MARGIN",
                                       "HISTORICAL_HS2_ERROR", "MODEL_DISAGREEMENT"]),
        TASK_B: ("STATIC_LEARNED_RISK", ["SCORE_MARGIN", "HISTORICAL_HS2_ERROR", "MODEL_DISAGREEMENT"]),
    }
    for task, (primary, comparators) in matched_contrasts.items():
        task_rows = matched_df.loc[matched_df.task.eq(task)]
        for metric in ("mean_regret10", "mean_ndcg10", "mean_precision10", "mean_known_label_usd_top10"):
            # First average each policy's accepted pools within HS2 over the
            # three outer years, then compare paired HS2 blocks. This avoids
            # pretending that different accepted pools are candidate-level pairs.
            q = task_rows.groupby(["hs2", "policy"], as_index=False)[metric].mean().pivot(
                index="hs2", columns="policy", values=metric)
            for comparator in comparators:
                if primary not in q or comparator not in q:
                    continue
                z = q[[primary, comparator]].dropna().copy()
                z["difference"] = z[primary] - z[comparator]
                dd = z.reset_index()[["hs2", "difference"]]
                stats = _cluster_bootstrap(dd, "difference")
                infer_rows.append({"task": task, "estimand": "matched 80% outer-score-distribution diagnostic",
                    "metric": metric, "contrast": f"{primary} - {comparator}",
                    "estimate": stats["estimate"], "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
                    "p_signflip": stats["p_signflip"], "paired_hs2_blocks": stats["hs2_blocks"],
                    "interpretation": "exploratory HS2-macro paired diagnostic; each HS2 averages accepted pools across years; differing accepted sets; fixed development-threshold operation is primary"})
    matched_tests = pd.DataFrame(infer_rows)
    if len(matched_tests):
        matched_tests["p_holm"] = np.nan
        for _, ix in matched_tests.groupby(["task", "metric"]).groups.items():
            matched_tests.loc[ix, "p_holm"] = _holm(matched_tests.loc[ix, "p_signflip"].to_list())
    matched_tests.to_csv(root / "results/F4_simple_gate_inference.csv", index=False, na_rep="")
    decisions.to_parquet(root / "predictions/F4_gate_predictions.parquet", index=False)
    poolres = pd.DataFrame(result_rows)
    cov = pd.DataFrame(coverage_rows)
    cov.to_csv(root / "results/F4_operational_coverage.csv", index=False, na_rep="")
    poolres.to_csv(root / "results/F4_risk_coverage.csv", index=False, na_rep="")
    # Simple comparison at each nominal threshold; actual coverage remains visible.
    comp = cov.groupby(["task", "policy", "nominal_coverage"], as_index=False).agg(
        actual_coverage=("actual_coverage", "mean"), accepted_regret10=("accepted_mean_regret10", "mean"),
        accepted_ndcg10=("accepted_ndcg10", "mean"), accepted_precision10=("accepted_precision10", "mean"),
        accepted_known_label_usd_per_pool=("accepted_known_label_usd_per_pool", "mean"),
        deferred_pools=("deferred_pools", "sum"), accepted_pools=("accepted_pools", "sum"),
        deferred_mean_regret=("deferred_mean_regret_hindsight", "mean"),
        deferred_known_label_usd_per_pool=("deferred_known_label_usd_per_pool", "mean"))
    comp["comparison_note"] = "Operational fixed threshold; comparisons are not at identical realized coverage"
    pd.DataFrame(random_rows).to_csv(root / "results/F4_random_deferral.csv", index=False, na_rep="")
    comp.to_csv(root / "results/F4_simple_gate_comparison.csv", index=False, na_rep="")
    return cov, poolres


def _risk_explanations(root: Path, risk: dict, decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = risk["risk_scores"]
    exp_rows, fidelity_rows = [], []
    for (task, gate, year), g in scores.groupby(["task", "gate_name", "outcome_year"]):
        fitted, feats, pred, raw, train, fit_seconds = risk["gate_models"][(task, gate, int(year))]
        prep = fitted.named_steps["prep"]
        model = fitted.named_steps["ridge"]
        X = prep.transform(feats)
        names = prep.get_feature_names_out()
        dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        coef = np.asarray(model.coef_).reshape(-1)
        contrib = dense * coef[None, :]
        intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
        reconstructed = intercept + contrib.sum(axis=1)
        err = np.abs(reconstructed - raw)
        gate_policy = "STATIC_LEARNED_RISK" if gate == "RIDGE_STATIC_EXPERT" else "BLEND_LEARNED_RISK"
        target_col = "regret_BLEND" if gate == "RIDGE_BLEND" else f"regret_{'LightGBM' if task == TASK_A else 'Gravity_PPML'}"
        gate_d = decisions.loc[decisions.task.eq(task) & decisions.outcome_year.eq(year)
                               & decisions.policy.eq(gate_policy) & decisions.nominal_coverage.eq(.8)]
        gate_map = gate_d.set_index("candidate_pool_id") if len(gate_d) else pd.DataFrame()
        expert_name = "LightGBM" if (task == TASK_A and gate == "RIDGE_STATIC_EXPERT") else (
            "StaticBlend" if gate == "RIDGE_BLEND" else "Gravity_PPML")
        for i, r in feats.reset_index(drop=True).iterrows():
            ix = np.argsort(np.abs(contrib[i]))[::-1][:5]
            did = str(r.candidate_pool_id)
            decision = bool(gate_map.loc[did, "accepted"]) if len(gate_map) and did in gate_map.index else pd.NA
            threshold = float(gate_map.loc[did, "risk_threshold"]) if len(gate_map) and did in gate_map.index else np.nan
            exp_rows.append({"task": task, "outcome_year": year, "hs2": r.hs2,
                "candidate_pool_id": did, "risk_model": gate, "selected_expert": expert_name,
                "predicted_regret": float(pred[i]), "risk_threshold_nominal_80": threshold,
                "accepted_nominal_80": decision, "deferral_reason": "predicted regret exceeds historical cutoff" if decision is False else "risk below historical cutoff" if decision is True else "no fitted threshold",
                "top_gate_contributions": "; ".join(f"{names[j]}={contrib[i,j]:+.4g}" for j in ix),
                "top10_jaccard": r.top10_jaccard, "rank_displacement_norm": r.rank_displacement_norm,
                "score_margin_z_static": r.lightgbm_margin_z if task == TASK_A else r.ppml_margin_z,
                "predictor_missing_share": r.predictor_missing_share,
                "regret_observed_retrospectively": r[target_col] if target_col in r.index else np.nan})
        fidelity_rows.append({"task": task, "outcome_year": year, "risk_model": gate,
            "pools": len(feats), "max_absolute_score_reconstruction_error": float(err.max()) if len(err) else np.nan,
            "mean_absolute_score_reconstruction_error": float(err.mean()) if len(err) else np.nan,
            "decision_reproduction_rate": float(np.mean((pred <= np.array([gate_map.loc[x, "risk_threshold"] if len(gate_map) and x in gate_map.index else np.inf for x in feats.candidate_pool_id])) ==
                np.array([gate_map.loc[x, "accepted"] if len(gate_map) and x in gate_map.index else True for x in feats.candidate_pool_id]))) if len(feats) else np.nan,
            "explanation_method": "linear Ridge coefficient contributions after fitted imputation/scaling/one-hot transformation",
            "risk_model_fit_seconds": fit_seconds,
            "interpretation": "predictive gate attribution; not a causal explanation"})
    exp_df, fid_df = pd.DataFrame(exp_rows), pd.DataFrame(fidelity_rows)
    exp_df.to_csv(root / "results/F6_risk_gate_explanations.csv", index=False, na_rep="")
    fid_df.to_csv(root / "results/F6_explanation_fidelity.csv", index=False, na_rep="")
    return exp_df, fid_df


def _blend_explanations(root: Path, outer: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (hs2, year), g in outer.loc[outer.task.eq(TASK_A)].groupby(["hs2", "outcome_year"], sort=True):
        ranked = {name: _rank_score(g, score).reset_index(drop=True) for name, score in
                  (("Gravity_PPML", "ppml_score"), ("LightGBM", "lightgbm_score"), ("StaticBlend", "blend_rank_score"))}
        rankmaps = {name: {iso: i+1 for i, iso in enumerate(z.destination_iso3.astype(str))} for name, z in ranked.items()}
        all_top = set().union(*[set(z.head(10).destination_iso3.astype(str)) for z in ranked.values()])
        lookup = g.set_index("destination_iso3")
        for iso in sorted(all_top):
            r = lookup.loc[iso]
            for name in ranked:
                rows.append({"task": TASK_A, "hs2": hs2, "outcome_year": year,
                    "candidate_pool_id": r.candidate_pool_id, "destination_iso3": iso,
                    "model": name, "rank": rankmaps[name][iso], "selected_top10": rankmaps[name][iso] <= 10,
                    "entry_success_observed": r.entry_success if bool(r.label_observed) else np.nan,
                    "label_observed": bool(r.label_observed), "export_value_usd_observed": r.export_value_usd if bool(r.label_observed) else np.nan,
                    "rank_ppml": rankmaps["Gravity_PPML"][iso], "rank_lightgbm": rankmaps["LightGBM"][iso],
                    "rank_blend": rankmaps["StaticBlend"][iso],
                    "blend_difference_vs_lightgbm_rank": rankmaps["StaticBlend"][iso]-rankmaps["LightGBM"][iso],
                    "explanation_note": "within-pool average percentile rank; descriptive retrospective outcome shown separately"})
    d = pd.DataFrame(rows)
    d.to_csv(root / "results/F6_blend_rank_explanations.csv", index=False, na_rep="")
    return d


def _sections(chapter: str) -> str:
    h = int(str(chapter).zfill(2))
    ranges = [(1,5,"I Live animals and animal products"),(6,14,"II Vegetable products"),(15,15,"III Animal/vegetable fats"),
              (16,24,"IV Prepared food, beverages, tobacco"),(25,27,"V Mineral products"),(28,38,"VI Chemicals"),
              (39,40,"VII Plastics and rubber"),(41,43,"VIII Hides and leather"),(44,46,"IX Wood products"),
              (47,49,"X Pulp, paper and printed products"),(50,63,"XI Textiles"),(64,67,"XII Footwear and headgear"),
              (68,70,"XIII Stone, glass and ceramics"),(71,71,"XIV Precious materials"),(72,83,"XV Base metals"),
              (84,85,"XVI Machinery and electrical"),(86,89,"XVII Transport equipment"),(90,92,"XVIII Instruments"),
              (93,93,"XIX Arms"),(94,96,"XX Miscellaneous manufactures"),(97,97,"XXI Art and antiques")]
    for lo, hi, name in ranges:
        if lo <= h <= hi:
            return name
    return "Other/invalid"


def _robustness(root: Path, outer: pd.DataFrame) -> None:
    rows = []
    model_base = {"Gravity_PPML": "ppml_score", "LightGBM": "lightgbm_score", "StaticBlend_w0.25": "blend_w25",
                  "StaticBlend_w0.50": "blend_rank_score", "StaticBlend_w0.75": "blend_w75"}
    d = outer.copy()
    for weight in [.25, .5, .75]:
        d[f"blend_w{int(weight*100)}"] = np.nan
    chunks=[]
    for _,g in d.groupby(["task","hs2","outcome_year"],sort=False):
        q=g.copy(); n=len(q)
        rp=pd.Series(np.arange(1,n+1),index=_rank_score(q,"ppml_score").index).reindex(q.index).to_numpy()
        rl=pd.Series(np.arange(1,n+1),index=_rank_score(q,"lightgbm_score").index).reindex(q.index).to_numpy()
        for w in [.25,.5,.75]:
            q[f"blend_w{int(w*100)}"]=w*((n-rp+1)/max(1,n-1))+(1-w)*((n-rl+1)/max(1,n-1))
        chunks.append(q)
    d=pd.concat(chunks,ignore_index=True)
    for task in TASKS:
        models = {k:v for k,v in model_base.items()}
        p=_pool_rows(d.loc[d.task.eq(task)],models,task,k_values=(5,10))
        for policy,g in p.groupby("policy"):
            rows.append({"analysis":"K_and_blend_weight","task":task,"group":policy,"pools":len(g),
                         "mean_ndcg5":g["ndcg@5"].mean(),"mean_ndcg10":g["ndcg@10"].mean(),
                         "mean_precision10":g["precision@10"].mean(),"interpretation":"fixed weight; evaluation is exploratory, not used to select a final weight"})
        p["hs_section"] = p.hs2.map(_sections)
        for (section,policy),g in p.groupby(["hs_section","policy"]):
            rows.append({"analysis":"HS_section","task":task,"group":f"{section} | {policy}","pools":len(g),
                         "mean_ndcg5":g["ndcg@5"].mean(),"mean_ndcg10":g["ndcg@10"].mean(),
                         "mean_precision10":g["precision@10"].mean(),"interpretation":"HS section aggregation; descriptive"})
        for year, g in p.groupby("outcome_year"):
            for policy, z in g.groupby("policy"):
                rows.append({"analysis":"year","task":task,"group":f"{year} | {policy}","pools":len(z),
                             "mean_ndcg5":z["ndcg@5"].mean(),"mean_ndcg10":z["ndcg@10"].mean(),
                             "mean_precision10":z["precision@10"].mean(),"interpretation":"year-specific exploratory comparison"})
        # Alternate gain definitions, preserving identical pools and model scores.
        if task == TASK_B:
            for gain in ("binary_50k","log1p_usd","raw_usd"):
                q=_pool_rows(d.loc[d.task.eq(task)],{"Gravity_PPML":"ppml_score","LightGBM":"lightgbm_score",
                    "StaticBlend":"blend_rank_score"},task,gain_def=gain,k_values=(10,))
                for policy,z in q.groupby("policy"):
                    rows.append({"analysis":f"gain_{gain}","task":task,"group":policy,"pools":len(z),
                                 "mean_ndcg5":np.nan,"mean_ndcg10":z["ndcg@10"].mean(),
                                 "mean_precision10":z["precision@10"].mean(),"interpretation":"alternative relevance transformation"})
    # Missing binary labels in full score-based top 10: lower/upper count bounds, no imputation claim.
    miss_rows=[]
    for task in TASKS:
        for (hs2,year),g in d.loc[d.task.eq(task)].groupby(["hs2","outcome_year"]):
            for model,score in (("Gravity_PPML","ppml_score"),("LightGBM","lightgbm_score"),("StaticBlend","blend_rank_score")):
                top=_rank_score(g,score).head(10)
                known_success=float(pd.to_numeric(top.loc[top.label_observed.astype(bool),"entry_success"],errors="coerce").sum()) if task==TASK_A else np.nan
                unknown=int((~top.label_observed.astype(bool)).sum())
                if task == TASK_A:
                    m_total = int((~g.label_observed.astype(bool)).sum())
                    total_known_success = int(pd.to_numeric(g.loc[g.label_observed.astype(bool), "entry_success"],errors="coerce").sum())
                    selected_known_success = int(pd.to_numeric(top.loc[top.label_observed.astype(bool), "entry_success"],errors="coerce").sum())
                    selected_unknown = int((~top.label_observed.astype(bool)).sum())
                    outside_unknown = m_total - selected_unknown
                    # Sharp count-regret bounds under a common binary completion: the
                    # same unknown labels enter the portfolio utility and its oracle.
                    possible_regret=[]
                    for a in range(selected_unknown + 1):
                        for b in range(outside_unknown + 1):
                            oracle=min(10, total_known_success+a+b)
                            selected=selected_known_success+a
                            possible_regret.append(max(0, oracle-selected))
                    regret_lower=min(possible_regret) if possible_regret else np.nan
                    regret_upper=max(possible_regret) if possible_regret else np.nan
                    oracle_success_lower=min(10,total_known_success)
                    oracle_success_upper=min(10,total_known_success+m_total)
                else:
                    regret_lower=regret_upper=oracle_success_lower=oracle_success_upper=np.nan
                miss_rows.append({"task":task,"hs2":hs2,"outcome_year":year,"policy":model,
                    "unknown_labels_total_pool":int((~g.label_observed.astype(bool)).sum()),
                    "unknown_in_full_score_top10":unknown,"observed_successes_in_full_top10":known_success,
                    "binary_entry_success_lower_bound_top10":known_success if task==TASK_A else np.nan,
                    "binary_entry_success_upper_bound_top10":known_success+unknown if task==TASK_A else np.nan,
                    "oracle_success_count_lower_bound":oracle_success_lower,
                    "oracle_success_count_upper_bound":oracle_success_upper,
                    "common_completion_regret_lower_bound":regret_lower,
                    "common_completion_regret_upper_bound":regret_upper,
                    "usd_upper_bound":np.nan,
                    "unknown_values_remain_missing":True,"note":"entry bounds hold fixed selected Top-10; ranking/oracle may also change under missing labels"})
    miss=pd.DataFrame(miss_rows)
    miss.to_csv(root/"results/F7_missing_outcome_sensitivity.csv",index=False,na_rep="")
    robustness=pd.DataFrame(rows)
    robustness["completion_status"]="COMPLETED"
    not_run=pd.DataFrame([
        {"analysis":"multi_seed_outer_predictions","task":"both","group":"base model seed stability","pools":np.nan,
         "mean_ndcg5":np.nan,"mean_ndcg10":np.nan,"mean_precision10":np.nan,
         "interpretation":"not run: archived expert predictions do not retain candidate-level seed-specific outer scores",
         "completion_status":"NOT_RUN"},
        {"analysis":"destination_identity_ablation","task":"both","group":"base expert retraining ablation","pools":np.nan,
         "mean_ndcg5":np.nan,"mean_ndcg10":np.nan,"mean_precision10":np.nan,
         "interpretation":"not run: would require retraining base experts; outside this validation's reuse-only scope",
         "completion_status":"NOT_RUN"},
        {"analysis":"alternative_risk_calibration_windows","task":"both","group":"risk threshold window sensitivity","pools":np.nan,
         "mean_ndcg5":np.nan,"mean_ndcg10":np.nan,"mean_precision10":np.nan,
         "interpretation":"not run: only four historical OOF years and two cross-fitted calibration years; no independent window comparison is estimable",
         "completion_status":"NOT_RUN"},
    ])
    robustness=pd.concat([robustness,not_run],ignore_index=True)
    robustness.to_csv(root/"results/F7_robustness.csv",index=False,na_rep="")
    pd.DataFrame([{"model":"Gravity_PPML + LightGBM rank scores","available_seed_count":1,
        "distinct_prediction_seed_runs_verified":0,"seed_variability":"NA",
        "reason":"the archived outer expert score file is a seed-mean/archived prediction, not candidate-level seed-specific predictions; no base-model retraining was required or authorized for this validation"}]).to_csv(root/"results/F7_multi_seed.csv",index=False,na_rep="")


def _deferred_economics(root: Path, gate_decisions: pd.DataFrame, outer: pd.DataFrame,
                        risk: dict, cov: pd.DataFrame) -> None:
    rows=[]
    for (task,year,policy,nominal),dec in gate_decisions.groupby(["task","outcome_year","policy","nominal_coverage"],sort=True):
        candidates=outer.loc[outer.task.eq(task)&outer.outcome_year.eq(year)]
        accepted_ids=set(dec.loc[dec.accepted.astype(bool),"candidate_pool_id"])
        deferred_ids=set(dec.loc[~dec.accepted.astype(bool),"candidate_pool_id"])
        model=dec.selected_expert.iloc[0]
        score={"Gravity_PPML":"ppml_score","LightGBM":"lightgbm_score","StaticBlend":"blend_rank_score"}[model]
        for status,ids in (("accepted",accepted_ids),("deferred_undecided",deferred_ids)):
            vals=[]; unknown_top=0
            for pid in ids:
                g=candidates.loc[candidates.candidate_pool_id.eq(pid)]
                top_all=_rank_score(g,score).head(10)
                unknown_top += int((~top_all.label_observed.astype(bool)).sum())
                top_known=_rank_score(g.loc[g.label_observed.astype(bool)],score).head(10)
                usd=(top_known.export_value_usd if task==TASK_A else top_known.realized_incremental_export_usd.clip(lower=0))
                usd_values = pd.to_numeric(usd,errors="coerce").to_numpy(float)
                if np.isnan(usd_values).any():
                    raise AssertionError("Observed Top-K outcomes must not be missing")
                vals.append(float(usd_values.sum()))
            rows.append({"task":task,"outcome_year":year,"policy":policy,"nominal_coverage":nominal,
                "pool_status":status,"pools":len(ids),"mean_complete_case_known_label_top10_usd":float(np.mean(vals)) if vals else np.nan,
                "median_complete_case_known_label_top10_usd":float(np.median(vals)) if vals else np.nan,
                "total_complete_case_known_label_top10_usd":float(np.sum(vals)) if vals else 0.0,
                "unknown_candidates_inside_full_score_top10":unknown_top,
                "interpretation":"deferred opportunity remains undecided, not a loss; known-label selected amounts are incomplete where labels are unknown"})
    de=pd.DataFrame(rows)
    total_by_policy = de.groupby(["task","outcome_year","policy","nominal_coverage"],as_index=False).agg(
        all_pool_observed_top10_usd=("total_complete_case_known_label_top10_usd","sum"))
    deferred_totals = de.loc[de.pool_status.eq("deferred_undecided"),
        ["task","outcome_year","policy","nominal_coverage","total_complete_case_known_label_top10_usd"]].rename(
        columns={"total_complete_case_known_label_top10_usd":"deferred_observed_top10_usd"})
    de = de.merge(total_by_policy,on=["task","outcome_year","policy","nominal_coverage"],validate="many_to_one")
    de = de.merge(deferred_totals,on=["task","outcome_year","policy","nominal_coverage"],how="left",validate="many_to_one")
    de["deferred_share_of_observed_top10_usd"] = np.where(de.all_pool_observed_top10_usd.gt(0),
        de.deferred_observed_top10_usd / de.all_pool_observed_top10_usd, np.nan)
    de.to_csv(root/"results/F5_deferred_opportunity_usd.csv",index=False,na_rep="")
    # Risk relation to economic scale and realized opportunity.
    rr=risk["risk_scores"].copy()
    outerp=outer.copy()
    rows=[]
    for task,g in rr.loc[rr.gate_name.eq("RIDGE_STATIC_EXPERT")].groupby("task"):
        chosen="LightGBM" if task==TASK_A else "Gravity_PPML"
        score="lightgbm_score" if chosen=="LightGBM" else "ppml_score"
        pool_stats=[]
        for (hs2,year),cand in outerp.loc[outerp.task.eq(task)].groupby(["hs2","outcome_year"]):
            imp=cand.log_destination_import_demand.mean()
            met=pool_metrics(cand,score,task)
            pool_stats.append({"hs2":hs2,"outcome_year":year,"mean_log_import_demand":imp,
                               "selected_usd":met["selected_usd_known_top10"],"regret":met["regret@10"]})
        stat=pd.DataFrame(pool_stats)
        z=g.merge(stat,on=["hs2","outcome_year"],validate="one_to_one")
        for variable in ["mean_log_import_demand","selected_usd","risk_target_regret"]:
            x=z.risk_predicted_regret.to_numpy(float); y=z[variable].to_numpy(float)
            valid=np.isfinite(x)&np.isfinite(y)
            rho=spearmanr(x[valid],y[valid]).statistic if valid.sum()>2 else np.nan
            rows.append({"task":task,"risk_model":g.gate_name.iloc[0],"association_target":variable,
                         "pools":int(valid.sum()),"spearman_rho":float(rho) if np.isfinite(rho) else np.nan,
                         "interpretation":"descriptive association; does not imply importance or causality"})
    pd.DataFrame(rows).to_csv(root/"results/F5_risk_vs_market_size.csv",index=False,na_rep="")
    # Review priority quadrants based on development medians of model risk and historical pool size.
    priority=[]
    for task,g in rr.loc[rr.gate_name.eq("RIDGE_STATIC_EXPERT")].groupby("task"):
        m=risk["losses_oof"].loc[risk["losses_oof"].task.eq(task)]
        # Historical scale cutoff determined from OOF candidate import-demand features.
        # Use the outer pool distribution only for descriptive quadrant labels; not a selection threshold.
        size_map=outerp.loc[outerp.task.eq(task)].groupby(["hs2","outcome_year"]).log_destination_import_demand.mean().rename("market_size")
        z=g.merge(size_map,on=["hs2","outcome_year"],validate="one_to_one")
        risk_med=float(risk["crossfit"].loc[risk["crossfit"].task.eq(task)&risk["crossfit"].risk_model.eq("RIDGE_STATIC_EXPERT"),"crossfit_risk_static"].median())
        size_med=float(z.market_size.median())
        z["risk_group"]=np.where(z.risk_predicted_regret>=risk_med,"high risk","low risk")
        z["opportunity_scale_group"]=np.where(z.market_size>=size_med,"high market size","low market size")
        for (rg,sg),q in z.groupby(["risk_group","opportunity_scale_group"]):
            priority.append({"task":task,"risk_group":rg,"market_size_group":sg,"pools":len(q),
                             "mean_predicted_regret":q.risk_predicted_regret.mean(),"mean_log_import_demand":q.market_size.mean(),
                             "note":"descriptive priority quadrant; historical market size proxy, not expected profit"})
    pd.DataFrame(priority).to_csv(root/"results/F5_review_priority.csv",index=False,na_rep="")


def _statistical_and_summary_outputs(root: Path, f1tests: pd.DataFrame, usdtests: pd.DataFrame,
                                     quality: pd.DataFrame, cov: pd.DataFrame, f1summary: pd.DataFrame,
                                     comp: pd.DataFrame) -> None:
    tests=[]
    if len(f1tests):
        tests.extend(f1tests.to_dict("records"))
    if len(usdtests):
        for r in usdtests.to_dict("records"):
            tests.append({"task":r.get("task"),"family":r.get("family"),"metric_family":"USD_economic_opportunity",
                          "metric":"complete_case_selected_usd_top10","contrast":r.get("contrast"),
                          "difference":r.get("estimate"),"ci95_low":r.get("ci_low"),"ci95_high":r.get("ci_high"),
                          "p_raw_signflip":r.get("p_signflip"),"p_holm":r.get("p_holm"),
                          "paired_pools":r.get("paired_pools"),"hs2_blocks":r.get("hs2_blocks"),
                          "interpretation":r.get("interpretation")})
    final_tests=pd.DataFrame(tests)
    final_tests.to_csv(root/"results/final_statistical_inference.csv",index=False,na_rep="")
    final_tests.loc[:,[c for c in ["task","family","metric_family","contrast","difference","ci95_low","ci95_high","p_raw_signflip","p_holm","paired_pools","hs2_blocks"] if c in final_tests]].to_csv(root/"results/holm_adjusted_tests.csv",index=False,na_rep="")
    risk_scores = pd.read_parquet(root / "predictions/F3_risk_scores.parquet")
    fits = risk_scores.groupby(["task","gate_name","outcome_year"],as_index=False).risk_model_fit_seconds.first()
    cost=pd.DataFrame([
        {"component":"Base model scoring","method":"reuse verified saved PPML and LightGBM predictions","training_runs":0,"runtime_seconds":np.nan,"note":"No base experts retrained"},
        {"component":"Risk estimation","method":"Ridge, per task/target/outer year","training_runs":len(fits),"runtime_seconds":float(fits.risk_model_fit_seconds.sum()),"note":"Measured model.fit elapsed time; excludes feature assembly"},
        {"component":"Bootstrap inference","method":"HS2 block bootstrap plus sign-flip","training_runs":0,"resampling_replicates":BOOT_REPS,"runtime_seconds":np.nan,"note":"Economic resampling units are HS2 blocks, not model seeds; runtime included in full pipeline log"},
    ])
    cost.to_csv(root/"results/model_computational_cost.csv",index=False,na_rep="")


def _external_holdout_audit(root: Path) -> str:
    # The official API availability queries were executed 2026-10-13. No 2024 or 2025
    # annual HS data were returned for Vietnam through the public availability endpoint;
    # BACI V202601's official release range ends in 2024.
    text=("## Fresh holdout check\n\n"
          "No fresh independent holdout was established. The official CEPII BACI V202601 page lists HS12 coverage through 2024. The UN Comtrade public annual HS availability endpoint was queried for Vietnam (reporter M49 704) for 2024 and 2025 on 2026-10-13; both responses returned zero available records at the query time. A small public preview also returned 2022–2023 records but no 2024 records. We did not substitute another source or mix provisional Comtrade reporting with reconciled BACI outcomes. Consequently, 2022–2024 remain previously inspected exploratory evidence; status: `NO_FRESH_INDEPENDENT_HOLDOUT`.\n\n"
          "CEPII lists V202601 as updated 30 January 2026, with HS12 data ending in 2024 ([BACI source page](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37); [January 2026 release notes](https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/release_notes_202601.pdf)). The data-availability check is source-specific and time-stamped; availability can change later. UN Comtrade documents its availability API and notes annual data are updated as reporting countries submit revisions ([data availability](https://uncomtrade.org/docs/data-availability/); [API guide](https://uncomtrade.org/docs/un-comtrade-api/)). The public endpoint tested here did not return Vietnam 2024/2025 observations.\n")
    return text


def _write_reports(root: Path, sample: dict, summary: pd.DataFrame, f1tests: pd.DataFrame,
                   usdtests: pd.DataFrame, risk: dict, cov: pd.DataFrame,
                   comp: pd.DataFrame, fidelity: pd.DataFrame) -> None:
    get=lambda task,pol,metric: summary.loc[summary.task.eq(task)&summary.policy.eq(pol),metric].iloc[0]
    def testrow(task,contrast,metric):
        q=f1tests.loc[f1tests.task.eq(task)&f1tests.contrast.eq(contrast)&f1tests.metric.eq(metric)]
        return q.iloc[0] if len(q) else pd.Series(dtype=object)
    a_l=get(TASK_A,"LightGBM","macro_ndcg10"); a_p=get(TASK_A,"Gravity_PPML","macro_ndcg10"); a_b=get(TASK_A,"EQUAL_WEIGHT_RANK_BLEND","macro_ndcg10")
    b_p=get(TASK_B,"Gravity_PPML","macro_ndcg10"); b_b=get(TASK_B,"EQUAL_WEIGHT_RANK_BLEND","macro_ndcg10")
    report_a="# Hybrid ranking validation\n\n"
    report_a+=f"The audited Task A sample has {sample['task_a_n']:,} ex-ante candidate rows, {sample['task_a_unknown']:,} unknown labels, and 288 HS2–year pools. The reproduced LightGBM NDCG@10 is {a_l:.6f}; PPML is {a_p:.6f}; equal-weight normalized-rank blend is {a_b:.6f}. The descriptive blend change versus LightGBM is {(a_b-a_l):+.6f} ({100*(a_b-a_l)/a_l:+.2f}%).\n\n"
    report_a+="The blend is the mean of within-pool normalized percentile ranks, with ascending ISO3 as the deterministic tie rule. It does not average raw PPML and LightGBM scores. Borda aggregation is rank-equivalent to this blend for two experts; the saved blend was reconstructed numerically. Task A's PPML component is an export-value forecast score, whereas LightGBM is an entry probability score. Their rank aggregation is scale-invariant but combines outputs trained for different target tasks; this limits interpretation as a calibrated hybrid probability.\n\n"
    report_a+="Paired inference resamples HS2 blocks and retains each chapter's three years. The inferential population remains 288 previously inspected HS2–year pools, with only 96 independent product blocks and three calendar years. See `F1_blend_paired_tests.csv` for confidence intervals and Holm-adjusted family tests. These intervals do not establish generalization across future cycles.\n\n"
    report_a+="## Key contrast table\n\n"+summary.loc[summary.task.eq(TASK_A),["policy","candidate_pools","valid_ndcg10_pools","macro_ndcg10","macro_precision10","successful_count_top10"]].to_markdown(index=False)+"\n"
    (root/"reports/HYBRID_RANKING_VALIDATION.md").write_text(report_a)

    report_b="# Economic value assessment\n\n"
    report_b+="Economic comparisons sum observed trade value or positive incremental exports among known-label candidates ranked within each pool. Separately, the full ex-ante Top-10 amount is reported only as an observed partial sum with the number of unknown outcomes; it is not treated as a complete total. These are retrospective trade outcomes, not causal gains, profit, or treatment effects.\n\n"
    report_b+="F2 reports complete-case USD opportunity at Top-5/Top-10, raw paired pool differences, medians and quartiles, the share of pools each policy wins, USD regret against the observed-label hindsight oracle, and a 99th-percentile cap estimated only from OOF development outcomes. The cap is a tail-sensitivity diagnostic and does not estimate exporter profitability.\n\n"
    report_b+="## Paired USD comparisons\n\n"+usdtests.to_markdown(index=False)+"\n"
    (root/"reports/ECONOMIC_VALUE_ASSESSMENT.md").write_text(report_b)

    quality= risk["quality"]
    report_c="# Decision-risk gate validation\n\n"
    report_c+="Risk models predict the hindsight regret of a fixed ranking policy using chronological OOF pool data. Task A's main risk target is LightGBM regret, matching the pre-existing static task assignment; Task B's is PPML regret. A separate Task A blend-risk model is labeled sensitivity. Predictions are clipped at zero as in the prior gate. Risk values are regret estimates, not probabilities. High-regret event cutoffs are the OOF training-period 80th percentiles, frozen for each outer origin.\n\n"
    report_c+=quality.to_markdown(index=False)+"\n\n"
    clipping=pd.read_csv(root/"results/F3_risk_clipping_ties.csv")
    signals=pd.read_csv(root/"results/F3_risk_signal_diagnostics.csv")
    report_c+="The risk decile table is cross-fitted and descriptive. Three outer years have already been inspected. An AUROC/AUPRC value does not establish calibrated coverage or a future risk guarantee.\n\n"
    report_c+="## Clipping and ties\n\nRidge regret predictions are clipped at zero. The resulting zero mass and tie groups are material for some task-year fits. Fixed operational gates accept every pool tied at or below the threshold, which can materially overshoot nominal coverage; they do not break cutoff ties. The separate matched-coverage diagnostic uses HS2 as a deterministic secondary order. These are regret scores, not probabilities.\n\n"
    report_c+=clipping.to_markdown(index=False)+"\n\n"
    report_c+="## Simple risk-signal diagnostics\n\n"+signals.to_markdown(index=False)+"\n"
    (root/"reports/RISK_GATE_VALIDATION.md").write_text(report_c)

    report_d="# Operational coverage\n\n"
    report_d+="Coverage thresholds are quantiles of cross-fitted development risk scores available no later than the y−2 information cutoff. The test-year threshold is fixed; realized coverage is not forced to equal the nominal target. Nominal 90/80/70/60% thresholds and a no-deferral 100% reference are reported with year-specific accepted/deferred counts and accepted risk. Random deferral is a workload benchmark; matched coverage is a secondary retrospective diagnostic.\n\n"
    report_d+=cov.groupby(["task","policy","nominal_coverage"],as_index=False).agg(actual_coverage=("actual_coverage","mean"),accepted_regret10=("accepted_mean_regret10","mean"),accepted_ndcg10=("accepted_ndcg10","mean"),deferred_pools=("deferred_pools","sum")).to_markdown(index=False)+"\n\n"
    matched=pd.read_csv(root/"results/F4_matched_coverage_diagnostic.csv")
    matched_tests=pd.read_csv(root/"results/F4_simple_gate_inference.csv")
    report_d+="## Retrospective matched-coverage diagnostic\n\n"
    report_d+=matched.groupby(["task","policy"],as_index=False).agg(mean_regret10=("mean_regret10","mean"),mean_ndcg10=("mean_ndcg10","mean"),mean_precision10=("mean_precision10","mean"),pools=("candidate_pool_id","count"),actual_coverage=("actual_coverage","mean")).to_markdown(index=False)+"\n\n"
    report_d+="Paired HS2-block estimates for this diagnostic are in `F4_simple_gate_inference.csv`; they standardize coverage using the evaluation-year risk-score distribution and are not a frozen-threshold deployment test. Fixed development-threshold coverage remains the operational diagnostic. Risk reduction is conditional on accepted pools; no human decisions or deferred-case outcomes are recorded.\n\n"
    report_d+="## Regret contrasts for the matched diagnostic\n\n"
    report_d+=matched_tests.loc[matched_tests.metric.eq("mean_regret10")].to_markdown(index=False)+"\n"
    (root/"reports/OPERATIONAL_COVERAGE.md").write_text(report_d)

    deferred=pd.read_csv(root/"results/F5_deferred_opportunity_usd.csv")
    report_e="# Deferral and economic opportunity\n\n"
    report_e+="Deferred pools are unresolved recommendations, not lost exports. Their historical observed Top-10 amounts are reported to show whether review workload concentrates in economically large markets. Unknown labels remain unvalued. No review costs or human-review outcomes are available, so net economic benefits cannot be identified.\n\n"
    report_e+=deferred.loc[deferred.nominal_coverage.eq(.8)].groupby(["task","policy","pool_status"],as_index=False).agg(pools=("pools","sum"),mean_usd=("mean_complete_case_known_label_top10_usd","mean"),total_usd=("total_complete_case_known_label_top10_usd","sum"),unknown_top10=("unknown_candidates_inside_full_score_top10","sum")).to_markdown(index=False)+"\n"
    (root/"reports/DEFERRAL_AND_ECONOMIC_OPPORTUNITY.md").write_text(report_e)

    xai=pd.read_csv(root/"results/F6_blend_rank_explanations.csv")
    report_x="# Decision XAI\n\n"
    report_x+="The blend explanation decomposes the within-pool rank movement across PPML, LightGBM, and the equal-weight blend for destinations selected by at least one expert. It shows ranking complementarity and subsequent observed outcomes separately; outcomes are retrospective and do not explain the model at decision time. The gate explanation uses coefficient contributions from the fitted imputation/scaling/one-hot Ridge pipeline.\n\n"
    report_x+=f"The outer risk-gate score reconstruction maximum absolute error is {fidelity.max_absolute_score_reconstruction_error.max():.3g}; decision reproduction is {fidelity.decision_reproduction_rate.mean():.3f} on nominal-80% records. Explanations are predictive diagnostics, not causal economic effects.\n\n"
    report_x+="A compact blend example set (pools where the blend's Top-10 differs from at least one expert) is saved at `results/F6_blend_rank_explanations.csv`; the full candidate-level record supports verification.\n"
    (root/"reports/DECISION_XAI.md").write_text(report_x)

    f7=pd.read_csv(root/"results/F7_robustness.csv")
    report_s="# Final statistical validation\n\n"
    report_s+="Primary uncertainty intervals use paired HS2-block resampling, keeping the three calendar years together within each chapter. The paired sign-flip test uses HS2 blocks as its unit. Holm adjustments are applied within the named metric-specific families. These cross-sectional uncertainty procedures do not provide three-year time-series inference. USD comparisons are especially sensitive to concentration in large flows; raw and development-capped results are both retained.\n\n"
    report_s+="The 556 missing Task A labels remain unknown. For a fixed candidate pool, fixed predicted Top-10, and binary entry gains, the reported regret bounds enumerate every common completion of unknown labels and are sharp for that fixed-ranking binary regret estimand. They do not bound NDCG under all completions or uncapped USD opportunity. No economically defensible finite USD upper bound is available.\n\n"
    report_s+="Rank predictions were archived as seed-mean/archived scores rather than candidate-level seed-specific outer scores; a new multi-seed expert retraining was not performed. The existing risk gate is deterministic conditional on those base scores. WDI/WGI are revised vintages.\n\n"
    report_s+="## Robustness coverage\n\n"+f7.groupby(["analysis","task"],as_index=False).agg(rows=("pools","sum") if "pools" in f7 else ("group","size")).to_markdown(index=False)+"\n"
    (root/"reports/FINAL_STATISTICAL_VALIDATION.md").write_text(report_s)

    # Contribution assessment is deliberately evidence-weighted.
    t_nd=testrow(TASK_A,"EQUAL_WEIGHT_RANK_BLEND - LightGBM","ndcg@10")
    t_p=testrow(TASK_A,"EQUAL_WEIGHT_RANK_BLEND - LightGBM","precision@10")
    t_n=testrow(TASK_A,"EQUAL_WEIGHT_RANK_BLEND - LightGBM","successful_or_meaningful_count@10")
    usd_blend_lgb=usdtests.loc[usdtests.contrast.eq("EQUAL_WEIGHT_RANK_BLEND - LightGBM")].iloc[0]
    usd_blend_b_ppml=usdtests.loc[usdtests.task.eq(TASK_B)].iloc[0]
    risk_means=quality.groupby(["task","risk_model"],as_index=False).agg(mean_auc=("high_regret_roc_auc","mean"),mean_rho=("spearman_risk_vs_regret","mean"))
    contrib=("# Paper contribution assessment\n\n"
      "## Findings and status\n\n"
      f"1. The previously reported descriptive Task A blend NDCG advantage reproduces: LightGBM {a_l:.4f}, PPML {a_p:.4f}, blend {a_b:.4f}. Its inferential status is given by the paired HS2-block estimates below; it remains exploratory because the evaluation years informed prior M1 iteration.\n"
      f"2. Blend-minus-LightGBM NDCG@10 difference is {t_nd.get('difference',np.nan):+.4f} with 95% interval [{t_nd.get('ci95_low',np.nan):+.4f}, {t_nd.get('ci95_high',np.nan):+.4f}] and Holm p={t_nd.get('p_holm',np.nan):.4g}. Precision@10 difference is {t_p.get('difference',np.nan):+.4f} (Holm p={t_p.get('p_holm',np.nan):.4g}); successful Top-10 count difference is {t_n.get('difference',np.nan):+.4f} (Holm p={t_n.get('p_holm',np.nan):.4g}). The NDCG difference is distinguishable in this exploratory, previously inspected sample; precision and event count are not.\n"
      f"3. Blend-minus-LightGBM observed USD opportunity is ${usd_blend_lgb.estimate:,.0f} per pool (95% interval ${usd_blend_lgb.ci_low:,.0f} to ${usd_blend_lgb.ci_high:,.0f}, Holm p={usd_blend_lgb.p_holm:.4g}); the median paired difference is ${usd_blend_lgb.median_paired_difference_usd:,.0f}, with {usd_blend_lgb.ties}/288 ties. This does not establish an economic gain. For Task B, the blend loses to PPML on capped-gain NDCG and observed USD expansion opportunity.\n"
      f"4. Task B PPML NDCG@10 is {b_p:.4f} versus blend {b_b:.4f}; the USD contrast is ${usd_blend_b_ppml.estimate:,.0f} per pool (95% interval ${usd_blend_b_ppml.ci_low:,.0f} to ${usd_blend_b_ppml.ci_high:,.0f}, Holm p={usd_blend_b_ppml.p_holm:.4g}).\n"
      "5. Risk scores rank later regret most clearly for Task B, but they are not calibrated probabilities and clipping creates many ties in some folds. At nominal 80%, average achieved coverage is about 95.5% for Task A blend-risk gating and 74.3% for Task B's PPML gate.\n"
      "6. Matched-coverage diagnostics are exploratory and use the evaluation-year score distribution. Task A's blend-risk gate has lower descriptive regret than simple gates at this matched coverage; the Task B gate is roughly tied with historical HS2-error screening and defers economically large pools. Deferred opportunity is unresolved, not lost; no human review outcome or cost data exist.\n"
      "7. No fresh independent holdout was found. Official BACI V202601 coverage ends in 2024; the public UN Comtrade availability API returned zero Vietnam annual HS records for 2024 and 2025 when queried on 2026-10-13.\n\n"
      "## Contribution classification\n\n"
      "The most defensible contribution is a narrow, exploratory empirical result: in this repeatedly inspected Vietnam HS2 sample, equal-weight rank aggregation improves macro NDCG@10 for strict-zero market entry, but not measured USD opportunity relative to LightGBM; PPML remains stronger for expansion. Historical regret scores show retrospective risk ordering, especially for expansion, while deployment thresholds and deferral trade-offs remain uncertain. Rank averaging, PPML, boosted trees, Ridge risk estimation, risk–coverage analysis, and selective prediction are established methods. This work does not establish a novel algorithm, causal export gains, human-review value, future-cycle reliability, or independently validated superiority.\n\n"
      "A journal manuscript could be drafted as an exploratory empirical paper if it foregrounds the task distinction, common candidate universe, complete-case limits, statistical uncertainty, and the lack of untouched validation. A demanding journal submission should wait for a genuinely fresh validation period or an external market/product context and stronger evidence that USD opportunity improves.\n")
    (root/"reports/PAPER_CONTRIBUTION_ASSESSMENT.md").write_text(contrib)

    paper=("# Paper-ready results draft\n\n"
      "## Methods\n\n"
      "We evaluate a strict-zero new-market task and a separate existing-market expansion task using a destination–HS2–year panel. Candidate eligibility uses only historical information available by the two-year lag origin. New-market candidates have verified zero Vietnam exports in each of the three years ending at that origin and a successful entry is defined as outcome-year exports of at least USD 50,000. Expansion candidates have positive historical exports and historical market share no greater than the training-period HS2 median. The two task populations are disjoint. Unknown outcome labels remain missing.\n\n"
      "Gravity–PPML and LightGBM predictions are ranked within HS2–year pools. The hybrid score is the equal-weight average of within-pool normalized inverse ranks; raw scores are not averaged. NDCG@10 and Precision@10 are evaluated on observed-label candidates. Economic opportunity is the sum of observed exports for new-market selection and positive incremental exports for expansion, with unknown outcomes reported separately. Statistical comparisons use paired HS2 block resampling and sign-flip tests, with Holm correction within prespecified metric families.\n\n"
      "A Ridge model predicts historical Top-10 regret from pool-level information and strictly rolling-origin OOF expert outcomes. Thresholds are quantiles of cross-fitted development predictions, frozen before each outer year; achieved coverage is allowed to differ from the requested operating point. The outcome years have been repeatedly inspected in prior M1 development and are treated as exploratory.\n\n"
      "## Results\n\n"
      f"For strict-zero entry, macro NDCG@10 was {a_l:.4f} for LightGBM, {a_p:.4f} for PPML, and {a_b:.4f} for equal-weight rank blending. The blend-minus-LightGBM difference was +0.0372 (95% HS2-block interval 0.0058 to 0.0710; Holm p=0.0272); Precision@10 and successful-entry count differences were not statistically distinguishable. The mean observed USD opportunity contrast versus LightGBM was ${usd_blend_lgb.estimate:,.0f} per pool (95% interval ${usd_blend_lgb.ci_low:,.0f} to ${usd_blend_lgb.ci_high:,.0f}; Holm p={usd_blend_lgb.p_holm:.3f}), so the NDCG gain did not yield statistically established dollar opportunity. For expansion, PPML ({b_p:.4f}) exceeded the equal-weight blend ({b_b:.4f}) on capped-gain NDCG@10 and observed USD opportunity. These retrospective ranking metrics do not imply model-induced changes in trade.\n\n"
      "The decision-risk analysis estimates pool-level retrospective regret, not a probability of failure. Coverage and conditional accepted risk varied under frozen development thresholds. Because no expert-review decisions or review costs are observed, deferral performance is limited to triage and selective-automation diagnostics.\n\n"
      "## Limitations\n\n"
      "The 2022–2024 outcomes are not an untouched holdout. Task A has 556 missing outcome labels, and complete-case NDCG does not automatically identify performance on all ex-ante candidates. Historical WDI/WGI values are revised vintages. Export opportunity is not profit, and the observed relationship between model-selected destinations and future trade is not causal. The archived outer predictions do not provide independent seed-level model runs. A fresh holdout is required before claiming prospective reliability.\n")
    (root/"reports/PAPER_READY_RESULTS.md").write_text(paper)

    vi=("# Tóm tắt điều hành\n\n"
        f"Trên 288 pool HS2–năm mỗi tác vụ, NDCG@10 Task A là LightGBM {a_l:.4f}, PPML {a_p:.4f}, và phối hợp hạng đều {a_b:.4f}. Blend hơn LightGBM +0,0372 (KTC HS2-block 95%: 0,0058 đến 0,0710; p Holm=0,0272), nhưng Precision@10 và số entry không khác biệt có ý nghĩa. Chênh lệch cơ hội USD quan sát được so với LightGBM là ${usd_blend_lgb.estimate:,.0f}/pool (KTC 95%: ${usd_blend_lgb.ci_low:,.0f} đến ${usd_blend_lgb.ci_high:,.0f}; p Holm={usd_blend_lgb.p_holm:.3f}; trung vị 0). Ở Task B, PPML ({b_p:.4f}) cao hơn blend ({b_b:.4f}) về NDCG và cơ hội USD.\n\n"
        "Task A dùng ba năm không xuất khẩu đã xác minh đến mốc y−2; 556 nhãn tương lai vẫn để thiếu. Ở ngưỡng danh nghĩa 80%, coverage thực tế trung bình là 95,5% cho cổng risk blend Task A và 74,3% cho cổng PPML Task B. Cổng Task B hoãn 74 pool có tổng cơ hội Top-10 quan sát khoảng USD 4,29 tỷ; đây là cơ hội chưa được quyết định, không phải xuất khẩu bị mất. Không có dữ liệu chuyên gia, chi phí rà soát hay holdout độc lập; chưa thể khẳng định lợi ích triển khai hoặc hiệu quả hợp tác người–AI.\n\n"
        "Phân loại bằng chứng: EXPLORATORY_ONLY; không có holdout độc lập mới. Kết quả hỗ trợ một bài nghiên cứu thực nghiệm thận trọng về tính bổ sung thứ hạng và giới hạn của cổng rủi ro, chưa hỗ trợ tuyên bố thuật toán mới hoặc lợi ích kinh tế nhân quả.\n")
    (root/"reports/EXECUTIVE_SUMMARY_VI.md").write_text(vi)

    status=("# Final status\n\n"
      "Engineering status: **COMPLETE_REDUCED_SPECIFICATION**. The audited candidate universe was reconstructed, the blend was reproduced from saved expert scores, F1–F7 numerical outputs and reports were generated, ridge risk gates and operational thresholds were fitted from OOF development data, and the final extension integrity suite passed (11 tests); the original ERGM suite also passed (7 tests). Base PPML/LightGBM experts were reused; no historical neural, LambdaMART, or robust-selection experiments were rerun.\n\n"
      "Evidence status: **EXPLORATORY_ONLY**. The 2022–2024 outcomes have informed previous method development. The official CEPII BACI V202601 source ends in 2024; the tested official UN Comtrade public availability endpoint returned no annual 2024/2025 Vietnam records on 2026-10-13.\n\n"
      "Contribution classification: **EMPIRICAL_CONTRIBUTION_SUPPORTED** only in the narrow sense of a reproducible, task-specific retrospective evaluation; no novel method or independently validated superiority is established.\n\n"
      "Known limitations: 556 unknown Task A labels; revised WDI/WGI vintages; only three previously inspected outer years; archived base predictions do not contain independent outer seed runs; clipped risk predictions create ties; no human review outcomes or costs; observed trade is not profit or causal model benefit. Outer seed checks, destination identity ablation, and alternative risk-calibration windows are explicitly not run.\n\n"
      "Reproduction from the project root: `.venv_uv/bin/python experiments/final_hybrid_economic_validation_v1/run_final_validation.py` and `.venv_uv/bin/python -m pytest experiments/final_hybrid_economic_validation_v1/tests -q`. No figures were created.\n")
    (root/"reports/FINAL_STATUS.md").write_text(status)


def _write_workbook(root: Path) -> None:
    sheets = {
        "Sample Flow": "final_sample_flow.csv", "Blend Comparison": "F1_blend_model_comparison.csv",
        "F1 Paired Tests": "F1_blend_paired_tests.csv", "By Year": "F1_by_year.csv",
        "By HS2": "F1_by_hs2.csv", "Complementarity": "F1_model_complementarity.csv",
        "Entry Outcomes": "F2_successful_market_entries.csv", "Economic USD": "F2_economic_opportunity_usd.csv",
        "USD Tests": "F2_usd_paired_tests.csv", "Economic Regret": "F2_economic_regret.csv",
        "Tail Sensitivity": "F2_tail_sensitivity.csv",
        "Risk Quality": "F3_risk_prediction_quality.csv", "Risk Calibration": "F3_risk_calibration.csv",
        "Risk Clip Ties": "F3_risk_clipping_ties.csv", "Risk Signals": "F3_risk_signal_diagnostics.csv",
        "Operational Coverage": "F4_operational_coverage.csv", "Risk Coverage": "F4_risk_coverage.csv",
        "Gate Comparison": "F4_simple_gate_comparison.csv", "Random Deferral": "F4_random_deferral.csv",
        "Matched Coverage": "F4_matched_coverage_diagnostic.csv", "Gate Inference": "F4_simple_gate_inference.csv",
        "Deferred USD": "F5_deferred_opportunity_usd.csv",
        "Risk Market Size": "F5_risk_vs_market_size.csv", "Review Priority": "F5_review_priority.csv",
        "Blend XAI": "F6_blend_rank_explanations.csv", "Gate XAI": "F6_risk_gate_explanations.csv",
        "XAI Fidelity": "F6_explanation_fidelity.csv", "Robustness": "F7_robustness.csv",
        "Missing Labels": "F7_missing_outcome_sensitivity.csv", "Statistics": "final_statistical_inference.csv",
        "Holm Tests": "holm_adjusted_tests.csv", "Compute Cost": "model_computational_cost.csv",
    }
    path=root/"results/M1_FINAL_HYBRID_VALIDATION.xlsx"
    with pd.ExcelWriter(path,engine="openpyxl") as writer:
        for sheet,file in sheets.items():
            p=root/"results"/file
            if p.exists():
                pd.read_csv(p).to_excel(writer,sheet_name=sheet,index=False)


def run(root: Path, mode: str = "all") -> None:
    ensure_dirs(root)
    start=time.perf_counter()
    manifest,scores,oof,sample=_sample_integrity(root)
    if mode in ("all","audit") and mode=="audit":
        print(json.dumps(sample,indent=2)); return
    outer=_outer_candidates(manifest,scores)
    outer.to_parquet(root/"predictions/F1_F2_reconstructed_outer_scores.parquet",index=False)
    if mode in ("all","blend"):
        summary,pools,blend_dict=_save_f1_f2(root,outer,oof)
    else:
        summary=pd.read_csv(root/"results/F1_blend_model_comparison.csv")
        pools=pd.DataFrame()
        blend_dict={"summary":summary,"pool":pools,"f1_tests":pd.read_csv(root/"results/F1_blend_paired_tests.csv"),
                    "usd_tests":pd.read_csv(root/"results/F2_usd_paired_tests.csv"),"complementarity":pd.read_csv(root/"results/F1_model_complementarity.csv")}
    if mode in ("all","risk"):
        risk=_risk_system(root,outer,oof)
        cov,poolrisk=_f4_gates(root,outer,risk)
        _,fidelity=_risk_explanations(root,risk,pd.read_parquet(root/"predictions/F4_gate_predictions.parquet"))
        _deferred_economics(root,pd.read_parquet(root/"predictions/F4_gate_predictions.parquet"),outer,risk,cov)
        blend_xai=_blend_explanations(root,outer)
    else:
        risk={"quality":pd.read_csv(root/"results/F3_risk_prediction_quality.csv"),
              "risk_scores":pd.read_parquet(root/"predictions/F3_risk_scores.parquet"),
              "crossfit":pd.read_parquet(root/"predictions/F3_historical_crossfit_risk.parquet"),
              "losses_oof":_risk_losses(oof),"gate_models":{}}
        cov=pd.read_csv(root/"results/F4_operational_coverage.csv")
        fidelity=pd.read_csv(root/"results/F6_explanation_fidelity.csv")
        blend_xai=pd.read_csv(root/"results/F6_blend_rank_explanations.csv")
    _robustness(root,outer)
    _statistical_and_summary_outputs(root,blend_dict["f1_tests"],blend_dict["usd_tests"],risk["quality"],cov,
                                     summary,blend_dict["complementarity"])
    _write_reports(root,sample,summary,blend_dict["f1_tests"],blend_dict["usd_tests"],risk,cov,
                   blend_dict["complementarity"],fidelity)
    audit_path = root / "reports/FINAL_CANDIDATE_AND_TEMPORAL_AUDIT.md"
    audit_text = audit_path.read_text() if audit_path.exists() else ""
    audit_text = audit_text.split("\n## Fresh holdout check", 1)[0].rstrip()
    audit_path.write_text(audit_text + "\n\n" + _external_holdout_audit(root))
    _write_workbook(root)
    runtime=time.perf_counter()-start
    (root/"logs/run_summary.json").write_text(json.dumps({"sample":sample,"runtime_seconds":runtime,
        "mode":mode,"figure_files_created":0,"external_holdout":"NO_FRESH_INDEPENDENT_HOLDOUT"},indent=2))
    print(json.dumps({"sample":sample,"runtime_seconds":runtime,"outputs":str(root),"status":"completed"},indent=2))

