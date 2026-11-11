from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(PROJECT / "experiments") not in sys.path:
    sys.path.insert(0, str(PROJECT / "experiments"))

from explainable_reliability_gate_v1.src.task_audit import (  # noqa: E402
    add_entry_targets, add_history_flows, tag_expansion,
)
from src.gravity_ppml import fit_ppml  # noqa: E402
from src.ml_models import fit_classifier, fit_regressor, model_configs  # noqa: E402
from src.temporal_validation import NUMERIC_FEATURES, CATEGORICAL_FEATURES  # noqa: E402

DATA = PROJECT / "data"
ERGM = PROJECT / "experiments/explainable_reliability_gate_v1"
FINAL = PROJECT / "experiments/final_hybrid_economic_validation_v1"
THRESHOLD = 50_000.0
YEARS = (2022, 2023, 2024)
COUNTRIES = {"VNM": 704, "THA": 764, "IDN": 360}
TASK_A = "Task_A_new_entry"
TASK_B = "Task_B_existing_underexploited"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def _outdirs() -> None:
    for d in ["predictions", "models", "results", "reports", "data_manifest", "logs"]:
        (HERE / d).mkdir(parents=True, exist_ok=True)


def audit_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Verify existing audited Vietnam manifests/predictions without changing them."""
    _outdirs()
    manifest = pd.read_csv(FINAL / "results/final_candidate_manifest.csv")
    scores = pd.read_parquet(FINAL / "predictions/F1_F2_reconstructed_outer_scores.parquet")
    oof = pd.read_parquet(ERGM / "predictions/historical_oof_expert_predictions.parquet")
    panel = pd.read_parquet(DATA / "processed/panel_lag2.parquet")
    checks = [
        {"check": "source_archive_exists", "value": (DATA / "raw/baci/BACI_HS12_V202601.zip").exists(), "detail": "BACI V202601 local archive"},
        {"check": "candidate_rows", "value": len(manifest), "detail": "audited 2022-2024 Task A and Task B manifest"},
        {"check": "manifest_unique_task_key", "value": not manifest.duplicated(["task", "destination_iso3", "hs2", "outcome_year"]).any(), "detail": "task x destination x HS2 x year"},
        {"check": "unknown_task_a_labels", "value": int((manifest.task.eq(TASK_A) & ~manifest.label_observed.astype(bool)).sum()), "detail": "retained as missing"},
        {"check": "score_key_coverage", "value": int(scores.candidate_id.nunique()), "detail": "saved outer candidates"},
        {"check": "score_completeness", "value": bool(scores.loc[scores.task.eq(TASK_A), "lightgbm_score"].notna().all() and scores.loc[scores.task.eq(TASK_A), "ppml_score"].notna().all()), "detail": "Task A LightGBM and PPML scores"},
        {"check": "historical_oof_years", "value": ",".join(map(str, sorted(oof.outcome_year.unique()))), "detail": "rolling-origin OOF archive"},
        {"check": "panel_years", "value": f"{panel.outcome_year.min()}-{panel.outcome_year.max()}", "detail": "two-year-lag processed Vietnam panel"},
    ]
    pd.DataFrame(checks).to_csv(HERE / "results/input_checks.csv", index=False)
    if not checks[2]["value"] or not checks[5]["value"]:
        raise AssertionError("Existing candidate/prediction integrity check failed")
    report = f"""# Input and temporal audit

This extension reuses the audited VNM candidate manifest, archived out-of-time scores, historical rolling-origin classifier predictions, and HS2 panel. The raw BACI archive is CEPII HS12 V202601. No previous research artifact was modified.

- Candidate rows in the paired Task A/B manifest: {len(manifest):,}.
- Task A unknown outcome labels: {int((manifest.task.eq(TASK_A) & ~manifest.label_observed.astype(bool)).sum()):,}; retained as unknown.
- Outer years: {', '.join(map(str, YEARS))}; the years were repeatedly inspected in prior M1 development and are exploratory for this extension.
- Historical OOF classifier years: {', '.join(map(str, sorted(oof.outcome_year.unique())))}. Fold provenance states training outcomes end no later than prediction year minus two.
- Outcome in the source panel is observed only when BACI provides a positive destination-HS2 import market; missing outcomes are not treated as zero.
- WDI/WGI are revised historical vintages, so point-in-time deployment validity is not claimed.

The Task A label is `1{{V >= USD 50,000}}` within a three-year verified-zero history window ending at y-2. It measures a qualifying new-to-exporter product-market relationship after the observed zero window; it does not prove first-ever historical trade.
"""
    (HERE / "reports/01_INPUT_AND_TEMPORAL_AUDIT.md").write_text(report, encoding="utf-8")
    return manifest, scores, oof


def _fit_platt(oof: pd.DataFrame, cutoff: int) -> LogisticRegression:
    d = oof.loc[oof.task.eq(TASK_A) & oof.outcome_year.le(cutoff) & oof.label_observed.astype(bool)
                & oof.entry_success.notna() & oof.lightgbm_score.notna()].copy()
    y = d.entry_success.astype(int).to_numpy()
    if len(d) < 100 or np.unique(y).size != 2:
        raise ValueError(f"Insufficient historical OOF calibration data through {cutoff}: n={len(d)}")
    z = np.log(np.clip(d.lightgbm_score.to_numpy(float), 1e-6, 1 - 1e-6) /
               (1 - np.clip(d.lightgbm_score.to_numpy(float), 1e-6, 1 - 1e-6))).reshape(-1, 1)
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit(z, y)


def _apply_platt(model: LogisticRegression, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    return np.clip(model.predict_proba(z)[:, 1], 0, 1)


def _rank_metrics(group: pd.DataFrame, score_col: str, outcome_col: str, k: int) -> dict:
    """Complete-case within-pool evaluation; never puts unknown labels in the metric set."""
    known = group.loc[group[outcome_col].notna()].copy()
    if known.empty:
        return {"ndcg": np.nan, "precision": np.nan, "entries": np.nan, "n_known": 0, "n_positive": 0}
    known = known.sort_values([score_col, "destination_iso3"], ascending=[False, True], kind="mergesort")
    rel = known[outcome_col].astype(float).to_numpy()
    k_eff = min(k, len(known))
    gains = (2.0 ** rel - 1.0)
    discounts = 1 / np.log2(np.arange(2, k_eff + 2))
    dcg = float(np.sum(gains[:k_eff] * discounts))
    ideal = np.sort(rel)[::-1][:k_eff]
    idcg = float(np.sum((2.0 ** ideal - 1.0) * discounts))
    top = known.head(k_eff)
    positives = int(np.sum(rel > 0))
    return {"ndcg": dcg / idcg if idcg > 0 else np.nan,
            "precision": float((top[outcome_col] > 0).sum() / k),
            "entries": int((top[outcome_col] > 0).sum()), "n_known": len(known), "n_positive": positives}


def _block_bootstrap(d: pd.DataFrame, value: str, reps: int = 4000, seed: int = 417) -> tuple[float, float, float, float]:
    """Paired HS2-block percentile interval for the macro pool-level mean difference."""
    x = d[["hs2", value]].dropna()
    if x.empty:
        return np.nan, np.nan, np.nan, np.nan
    point = float(x[value].mean())
    rng = np.random.default_rng(seed)
    vals = np.empty(reps)
    blocks=x.groupby("hs2",sort=True)[value].apply(lambda s:s.to_numpy(float))
    hs=blocks.index.to_numpy()
    by={h:blocks.loc[h] for h in hs}
    for b in range(reps):
        draw = rng.choice(hs, size=len(hs), replace=True)
        vals[b] = np.concatenate([by[h] for h in draw]).mean()
    lo, hi = np.quantile(vals, [.025, .975])
    # Paired HS2-block sign-flip test; economic observations remain pools, not seeds.
    block_sums=np.array([by[h].sum() for h in hs])
    signs=rng.choice([-1.0,1.0],size=(reps,len(block_sums)))
    null=np.sum(signs*block_sums[None,:],axis=1)/len(x)
    p=float((1+np.sum(np.abs(null)>=abs(point)))/(reps+1))
    return point, float(lo), float(hi), float(p)


def _metric_eval(d: pd.DataFrame, score_cols: dict[str, str], outcome_col: str,
                 unit: str, target_col: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, poolrows = [], []
    for (hs2, year), g in d.groupby(["hs2", "outcome_year"], sort=True):
        for model, col in score_cols.items():
            m = _rank_metrics(g, col, outcome_col, 10)
            poolrows.append({"unit": unit, "hs2": hs2, "outcome_year": year, "model": model, **m})
    pools = pd.DataFrame(poolrows)
    for model, g in pools.groupby("model"):
        rows.append({"unit": unit, "model": model, "pools_total": len(g), "valid_ndcg_pools": int(g.ndcg.notna().sum()),
                     "macro_ndcg10": g.ndcg.mean(), "macro_precision10": g.precision.mean(),
                     "mean_successes_top10": g.entries.mean(), "mean_observed_positive_pools": g.n_positive.gt(0).mean()})
    return pd.DataFrame(rows), pools


def run_vietnam_two_part(manifest: pd.DataFrame, scores: pd.DataFrame, oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit historical conditional PPML folds and evaluate VNM Task A on 2022-24."""
    start = time.perf_counter()
    panel = pd.read_parquet(DATA / "processed/panel_lag2.parquet")
    panel["hs2"] = panel.hs2.astype(str).str.zfill(2)
    panel["outcome_year"] = panel.outcome_year.astype(int)
    panel["feature_year"] = panel.feature_year.astype(int)
    panel = add_entry_targets(panel, lag=2, threshold=THRESHOLD)
    panel = add_history_flows(panel)
    panel["strict_inactive_verified"] = panel.strict_inactive_verified.astype(bool)
    task = manifest.loc[manifest.task.eq(TASK_A)].copy()
    task["hs2"] = task.hs2.astype(str).str.zfill(2)
    d = task.merge(scores[["candidate_id", "ppml_score", "lightgbm_score", "blend_rank_score"]],
                   on="candidate_id", how="left", validate="one_to_one")
    out_predictions = []
    fit_diagnostics = []
    model_metric_rows, economics_rows, calibration_rows, decomposition = [], [], [], []
    model_score_cols: dict[int, dict[str, str]] = {}
    for year in YEARS:
        cutoff = year - 2
        train = panel.loc[panel.strict_inactive_verified & panel.outcome_observed.astype(bool)
                          & panel.export_value_usd.notna() & panel.export_value_usd.ge(THRESHOLD)
                          & panel.outcome_year.le(cutoff)].copy()
        test = panel.loc[panel.outcome_year.eq(year) & panel.destination_import_demand.gt(0)
                         & panel.strict_inactive_verified].copy()
        if train.empty or test.empty:
            raise RuntimeError(f"No conditional success training/test rows at year {year}")
        pred_cond, model, diag, _ = fit_ppml(train, test)
        pred_cond = np.maximum(np.asarray(pred_cond, float), 50_000.0)
        calib = _fit_platt(oof, cutoff=min(cutoff, max(oof.outcome_year)))
        base = d.loc[d.outcome_year.eq(year)].merge(
            test[["destination_iso3", "hs2", "outcome_year", "export_value_usd", "outcome_observed",
                  "destination_import_demand", "historical_export_value"]],
            on=["destination_iso3", "hs2", "outcome_year"], how="left", validate="one_to_one",
            suffixes=("", "_panel"))
        # `test` follows the panel row order; key merge aligns the conditional fitted values.
        cond = pd.DataFrame({"destination_iso3": test.destination_iso3.to_numpy(), "hs2": test.hs2.to_numpy(),
                             "outcome_year": test.outcome_year.to_numpy(), "conditional_ppml_usd": pred_cond})
        base = base.merge(cond, on=["destination_iso3", "hs2", "outcome_year"], how="left", validate="one_to_one")
        base["prob_raw"] = base.lightgbm_score.astype(float)
        base["prob_calibrated"] = _apply_platt(calib, base.prob_raw.to_numpy(float))
        base["two_part_qualified_usd_score"] = base.prob_calibrated * base.conditional_ppml_usd
        base["conditional_training_n"] = len(train)
        base["conditional_training_successes"] = int(train.export_value_usd.ge(THRESHOLD).sum())
        base["training_outcome_max"] = int(train.outcome_year.max())
        base["calibration_oof_year_max"] = min(cutoff, int(oof.outcome_year.max()))
        base["prediction_provenance"] = "historical VNM candidate universe; y-2 training cutoff; probability Platt calibrated on prior rolling OOF; conditional PPML fit on verified qualifying entries only"
        out_predictions.append(base)
        fit_diagnostics.append({"outcome_year": year, "training_cutoff": cutoff, "conditional_train_n": len(train),
                                "conditional_success_n": int(train.export_value_usd.ge(THRESHOLD).sum()),
                                "train_outcome_max": int(train.outcome_year.max()), "conditional_ppml_converged": diag["converged"],
                                "conditional_ppml_iterations": diag["iterations"], "n_parameters": diag["n_parameters"],
                                "calibration_oof_n": int((oof.task.eq(TASK_A) & oof.outcome_year.le(min(cutoff, max(oof.outcome_year))) & oof.label_observed.astype(bool)).sum()),
                                "runtime_seconds": time.perf_counter()-start})
        calibration_rows.append({"evaluation_year": year, "calibration_method": "Platt logistic on logit archived OOF LightGBM probabilities",
                                 "calibration_year_max": min(cutoff, int(oof.outcome_year.max())),
                                 "raw_oof_prevalence": float(oof.loc[oof.task.eq(TASK_A)&oof.outcome_year.le(min(cutoff,max(oof.outcome_year)))&oof.label_observed.astype(bool),"entry_success"].mean()),
                                 "n_calibration": int((oof.task.eq(TASK_A)&oof.outcome_year.le(min(cutoff,max(oof.outcome_year)))&oof.label_observed.astype(bool)).sum()),
                                 "test_outcomes_used_for_calibration": False})
    pred = pd.concat(out_predictions, ignore_index=True)
    # Reattach exact outer outcomes and ensure the candidate score is not trained on them.
    pred = pred.drop(columns=[c for c in ["export_value_usd", "outcome_observed", "label_observed", "entry_success"] if c in pred.columns], errors="ignore")
    pred = pred.merge(task[["candidate_id", "export_value_usd", "label_observed", "entry_success"]],
                      on="candidate_id", how="left", validate="one_to_one")
    pred["qualified_export_value_usd"] = np.where(pred.label_observed.astype(bool) & pred.export_value_usd.ge(THRESHOLD), pred.export_value_usd, np.where(pred.label_observed.astype(bool), 0.0, np.nan))
    pred["positive_subthreshold_export"] = pred.label_observed.astype(bool) & pred.export_value_usd.gt(0) & pred.export_value_usd.lt(THRESHOLD)
    pred.to_parquet(HERE / "predictions/A_candidate_predictions.parquet", index=False)
    pd.DataFrame(fit_diagnostics).to_csv(HERE / "results/A_conditional_value_accuracy.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(HERE / "results/A_entry_probability_calibration.csv", index=False)

    pred["rank_blend"] = pred.blend_rank_score
    score_cols = {"LightGBM_probability": "lightgbm_score", "Gravity_PPML": "ppml_score",
                  "Original_equal_rank_blend": "rank_blend", "TwoPart_calibrated_probability_x_conditional_PPML": "two_part_qualified_usd_score"}
    summary, pools = _metric_eval(pred, score_cols, "entry_success", "Task_A")
    # Add pool-level Top-K economics from full candidate universe; missing labels remain unknown.
    econ = []
    for (hs2, year), g in pred.groupby(["hs2", "outcome_year"], sort=True):
        for model, col in score_cols.items():
            top = g.sort_values([col, "destination_iso3"], ascending=[False, True], kind="mergesort").head(10)
            known = top.loc[top.label_observed.astype(bool)]
            q = known.qualified_export_value_usd.sum(min_count=1)
            tot = known.export_value_usd.sum(min_count=1)
            econ.append({"hs2": hs2, "outcome_year": year, "model": model, "k": 10, "selected_n": len(top),
                         "observed_labels_selected": len(known), "unknown_labels_selected": len(top)-len(known),
                         "qualified_usd_observed_lower_bound": q, "total_export_usd_observed_lower_bound": tot,
                         "successful_entries_top10_observed": int(known.entry_success.sum()),
                         "subthreshold_positive_count": int(known.positive_subthreshold_export.sum()),
                         "unit_note": "USD among observed labels; if an unknown is selected, observed sum is a lower bound"})
    eco = pd.DataFrame(econ)
    eco_summary = eco.groupby("model").agg(pools=("hs2", "size"), mean_qualified_usd=("qualified_usd_observed_lower_bound", "mean"),
        median_qualified_usd=("qualified_usd_observed_lower_bound", "median"), mean_total_export_usd=("total_export_usd_observed_lower_bound", "mean"),
        mean_successes=("successful_entries_top10_observed", "mean"), mean_subthreshold_positive=("subthreshold_positive_count", "mean"),
        unknown_selected_total=("unknown_labels_selected", "sum")).reset_index()
    eco_summary["unit"] = "USD per HS2-year pool at selected Top-10; observed-label sum"
    eco_summary.to_csv(HERE / "results/A_topk_economic_opportunity.csv", index=False)
    eco.to_csv(HERE / "results/A_topk_pool_economics.csv", index=False)
    summary = summary.merge(eco_summary, on="model", how="left", suffixes=("", "_economic"))
    summary.to_csv(HERE / "results/A_two_part_model_comparison.csv", index=False)
    pools.to_csv(HERE / "results/A_pool_ranking_metrics.csv", index=False)
    # Paired contrasts on same 288 complete-case metric pools; HS2 is bootstrap block.
    infer = []
    for metric in ["ndcg", "precision", "entries"]:
        wide = pools.pivot_table(index=["hs2", "outcome_year"], columns="model", values=metric, aggfunc="first").reset_index()
        for baseline in ["LightGBM_probability", "Gravity_PPML", "Original_equal_rank_blend"]:
            if baseline == "TwoPart_calibrated_probability_x_conditional_PPML":
                continue
            diff = wide["TwoPart_calibrated_probability_x_conditional_PPML"] - wide[baseline]
            t = pd.DataFrame({"hs2": wide.hs2, "diff": diff}).dropna()
            point, lo, hi, pval = _block_bootstrap(t, "diff") if len(t) else (np.nan, np.nan, np.nan, np.nan)
            infer.append({"contrast": f"TwoPart - {baseline}", "metric": metric, "difference": point,
                          "ci95_low_hs2_block_bootstrap": lo, "ci95_high_hs2_block_bootstrap": hi,
                          "bootstrap_two_sided_p_approx": pval, "paired_pools": len(t), "hs2_blocks": t.hs2.nunique(),
                          "status": "exploratory retrospective; VNM 2022-24 previously inspected"})
    infer_df = pd.DataFrame(infer)
    if len(infer_df):
        infer_df["holm_p_metric_family"] = np.nan
        for metric, idx in infer_df.groupby("metric").groups.items():
            infer_df.loc[idx, "holm_p_metric_family"] = multipletests(infer_df.loc[idx, "bootstrap_two_sided_p_approx"].fillna(1), method="holm")[1]
    infer_df.to_csv(HERE / "results/A_two_part_paired_tests.csv", index=False)
    # Decomposition and gains; robustly avoid log(0).
    rank_rows = []
    for (hs2, year), g in pred.groupby(["hs2", "outcome_year"], sort=True):
        order_p = set(g.nlargest(10, "prob_calibrated").candidate_id)
        order_m = set(g.nlargest(10, "conditional_ppml_usd").candidate_id)
        order_q = set(g.nlargest(10, "two_part_qualified_usd_score").candidate_id)
        rank_rows.append({"hs2":hs2,"outcome_year":year,"top10_probability_conditional_value_overlap":len(order_p&order_m),
                          "two_part_vs_probability_overlap":len(order_q&order_p),"two_part_vs_conditional_value_overlap":len(order_q&order_m),
                          "two_part_unique_vs_probability":len(order_q-order_p),"two_part_unique_vs_value":len(order_q-order_m),
                          "probability_value_spearman":spearmanr(g.prob_calibrated,g.conditional_ppml_usd,nan_policy="omit").statistic})
    pd.DataFrame(rank_rows).to_csv(HERE / "results/A_rank_change_decomposition.csv", index=False)
    # Calibration diagnostics on OOF development only and outer scores separately.
    cal_metrics=[]
    for y in YEARS:
        p=pred.loc[pred.outcome_year.eq(y)&pred.label_observed.astype(bool)]
        yy=p.entry_success.astype(int); pp=p.prob_calibrated.astype(float)
        cal_metrics.append({"year":y,"n":len(p),"prevalence":yy.mean(),"brier":brier_score_loss(yy,pp),
                            "log_loss":log_loss(yy,pp,labels=[0,1]),"roc_auc":roc_auc_score(yy,pp) if yy.nunique()>1 else np.nan,
                            "pr_auc":average_precision_score(yy,pp) if yy.sum()>0 else np.nan,
                            "calibration_in_large_observed_minus_mean_pred":yy.mean()-pp.mean(),
                            "status":"retrospective exploratory evaluation"})
    pd.DataFrame(cal_metrics).to_csv(HERE / "results/A_probability_test_diagnostics.csv",index=False)
    # NDCG denominator and threshold diagnostic.
    sub = pred.loc[pred.label_observed.astype(bool)]
    subthreshold = sub.export_value_usd.gt(0)&sub.export_value_usd.lt(THRESHOLD)
    pd.DataFrame([{"observed_candidates":len(sub),"observed_positive_exports_below_50000":int(subthreshold.sum()),
                   "subthreshold_share_of_observed_candidates":float(subthreshold.mean()),
                   "subthreshold_share_of_positive_export_value":float(sub.loc[subthreshold,"export_value_usd"].sum()/sub.loc[sub.export_value_usd.gt(0),"export_value_usd"].sum()),
                   "pools_total":pred.groupby(["hs2","outcome_year"]).ngroups,
                   "pools_with_observed_positive_entry":int(pools.loc[pools.model.eq("LightGBM_probability"),"n_positive"].gt(0).sum()),
                   "pools_without_identified_positive_entry":int((pools.loc[pools.model.eq("LightGBM_probability"),"n_positive"]==0).sum()),
                   "estimand_note":"NDCG macro is complete-case over observed candidates; NDCG undefined when no observed positive entry; unknown labels remain unidentified"}]).to_csv(HERE/"results/A_target_diagnostics.csv",index=False)
    report = f"""# Vietnam two-part results

The two-part target is `Q = V × 1(V ≥ USD 50,000)`. The score is a Platt-calibrated LightGBM probability, calibrated only from archived historical OOF predictions, multiplied by a conditional PPML mean trained on observed strict-zero candidates with `V ≥ USD 50,000` and outcomes no later than y−2. This estimates expected qualified export value, not total expected exports. The target omits positive sub-threshold exports from its economic utility; total V is reported separately.

The 2022–2024 candidate population is the previously audited strict-zero Task A manifest; missing labels remain NA. Complete-case NDCG uses the identical observed-label candidates for every model and omits pools without an identified positive event from the valid-NDCG denominator. The Top-10 USD table instead ranks the full eligible universe; sums are explicitly observed-label lower bounds if an unknown label enters Top-10.

## Model results

{summary.to_markdown(index=False)}

## Inference

Paired differences use HS2-block bootstrap over the same HS2-year pools. These intervals concern only the three inspected retrospective years and do not establish future-cycle generalization.

{infer_df.to_markdown(index=False)}

Training diagnostics and row-level predictions are saved with training cutoffs, calibrator cutoffs, sample counts, and provenance.
"""
    (HERE / "reports/03_VIETNAM_TWO_PART_RESULTS.md").write_text(report, encoding="utf-8")
    return pred, summary


def extract_baci_targets(force: bool = False) -> pd.DataFrame:
    """Stream BACI by year and retain global importer-HS2 demand, supplier HHI, and 3 exporter flows."""
    _outdirs()
    outpath = HERE / "data_manifest/baci_hs2_external_source.parquet"
    if outpath.exists() and not force:
        return pd.read_parquet(outpath)
    archive = DATA / "raw/baci/BACI_HS12_V202601.zip"
    cmap = pd.read_parquet(DATA / "processed/country_mapping.parquet").dropna(subset=["baci_id"])
    id_iso = dict(zip(cmap.baci_id.astype(int), cmap.iso3.astype(str)))
    exporter_ids = set(COUNTRIES.values())
    annual = []
    logs = []
    with zipfile.ZipFile(archive) as zf:
        for year in range(2012, 2025):
            member = f"BACI_HS12_Y{year}_V202601.csv"
            if member not in zf.namelist():
                raise FileNotFoundError(member)
            parts = []
            raw_n = 0
            with zf.open(member) as fh:
                for chunk in pd.read_csv(fh, usecols=["t", "i", "j", "k", "v"],
                                         dtype={"k": "string", "i": "int32", "j": "int32", "t": "int16", "v": "float64"},
                                         chunksize=500_000):
                    raw_n += len(chunk)
                    if not chunk.t.eq(year).all():
                        raise ValueError(f"BACI year field mismatch in {year}")
                    chunk["hs2"] = chunk.k.str.replace(r"\.0$", "", regex=True).str.zfill(6).str[:2]
                    chunk = chunk.loc[chunk.hs2.str.fullmatch(r"\d{2}") & chunk.v.ge(0), ["j", "hs2", "i", "v"]]
                    if not len(chunk):
                        continue
                    p = chunk.groupby(["j", "hs2", "i"], observed=True, sort=False, as_index=False).v.sum()
                    parts.append(p)
            detail = pd.concat(parts, ignore_index=True).groupby(["j", "hs2", "i"], observed=True, sort=False, as_index=False).v.sum()
            detail["world"] = detail.groupby(["j", "hs2"], observed=True).v.transform("sum")
            detail["share2"] = (detail.v / detail.world).pow(2)
            market = detail.groupby(["j", "hs2"], observed=True, sort=False, as_index=False).agg(
                world_imports_thousand_usd=("v", "sum"), supplier_hhi=("share2", "sum"))
            exporter = detail.loc[detail.i.isin(exporter_ids), ["j", "hs2", "i", "v"]].pivot_table(
                index=["j", "hs2"], columns="i", values="v", aggfunc="sum", fill_value=0).reset_index()
            for i in exporter_ids:
                if i not in exporter.columns:
                    exporter[i] = 0.0
            exporter = exporter.rename(columns={v:f"flow_{k}" for k,v in COUNTRIES.items()})
            market = market.merge(exporter, on=["j", "hs2"], how="left", validate="one_to_one")
            for c in [f"flow_{iso}" for iso in COUNTRIES]:
                market[c] = market[c].fillna(0.0) * 1000.0
            market["world_imports_usd"] = market.world_imports_thousand_usd * 1000.0
            market["year"] = year
            market["destination_iso3"] = market.j.map(id_iso)
            market = market.loc[market.destination_iso3.str.fullmatch(r"[A-Z]{3}", na=False)].copy()
            market["hs2"] = market.hs2.astype(str).str.zfill(2)
            market = market.drop(columns=["j", "world_imports_thousand_usd"])
            annual.append(market)
            logs.append({"year":year,"raw_rows":raw_n,"aggregated_destination_hs2":len(market),"source_member":member,
                         "importers":market.destination_iso3.nunique(),"hs2":market.hs2.nunique(),"value_unit":"current USD; BACI v thousand USD multiplied by 1,000"})
            print(f"BACI {year}: {raw_n:,} raw rows -> {len(market):,} destination-HS2 rows")
    source = pd.concat(annual, ignore_index=True)
    if source.duplicated(["destination_iso3", "hs2", "year"]).any():
        raise AssertionError("Duplicate destination-HS2-year source aggregates")
    source.to_parquet(outpath, index=False)
    pd.DataFrame(logs).to_csv(HERE / "data_manifest/baci_extraction_log.csv", index=False)
    (HERE / "data_manifest/baci_source.json").write_text(json.dumps({"archive":str(archive),"sha256":_sha(archive),"version":"V202601","rows":len(source),"years":"2012-2024","created_utc":pd.Timestamp.utcnow().isoformat()},indent=2))
    return source


def build_country_panel(source: pd.DataFrame, exporter: str) -> pd.DataFrame:
    """Build same-semantics y-2 panels for an exporter; labels are in this internal panel only."""
    if exporter not in COUNTRIES:
        raise ValueError(exporter)
    flows = source[["destination_iso3", "hs2", "year", f"flow_{exporter}"]].copy()
    demand = source[["destination_iso3", "hs2", "year", "world_imports_usd", "supplier_hhi"]].copy()
    # Any destination import record indicates BACI destination-year reporting coverage.
    coverage = demand[["destination_iso3", "year"]].drop_duplicates().assign(report_covered=True)
    macros = pd.read_parquet(DATA / "processed/country_year_features.parquet")
    macros["iso3"] = macros.iso3.astype(str)
    geo_zip = DATA / "raw/cepii/dist_cepii.zip"
    with zipfile.ZipFile(geo_zip) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".xls"))
        dist = pd.read_excel(zf.open(member))
    dist.columns = [c.lower() for c in dist.columns]
    geo = dist.loc[dist.iso_o.eq(exporter), [c for c in ["iso_d","dist","distw","contig","comlang_off","colony","comcol"] if c in dist.columns]].copy()
    geo = geo.rename(columns={"iso_d":"destination_iso3","dist":"distance_km","distw":"population_weighted_distance_km","contig":"contiguous","comlang_off":"common_official_language","colony":"colonial_link","comcol":"common_colonizer"})
    geo["log_distance"] = np.log(pd.to_numeric(geo.distance_km,errors="coerce").where(pd.to_numeric(geo.distance_km,errors="coerce").gt(0)))
    country_xls = pd.read_excel(DATA / "raw/cepii/geo_cepii.xls", usecols=["iso3","landlocked","continent"])
    country_xls.columns = [c.lower() for c in country_xls.columns]
    geo = geo.merge(country_xls.drop_duplicates("iso3").rename(columns={"iso3":"destination_iso3"}),on="destination_iso3",how="left")
    demand = demand.rename(columns={"year":"feature_year","supplier_hhi":"destination_supplier_concentration"})
    flows = flows.rename(columns={"year":"feature_year","flow_"+exporter:"historical_export_value"})
    base = demand.copy()
    base["outcome_year"] = base.feature_year.astype(int)+2
    base = base.loc[base.world_imports_usd.gt(0) & base.destination_iso3.ne(exporter)].copy()
    # Current and lagged bilateral values, with missing bilateral rows treated as zero only when the destination reported in that year.
    flow_key = flows.rename(columns={"feature_year":"year","historical_export_value":"flow_usd"})
    for j in [0,1,2]:
        keys=base[["destination_iso3","hs2","feature_year"]].copy(); keys["year"]=keys.feature_year-j
        keys=keys.merge(coverage.rename(columns={"year":"_year"}),left_on=["destination_iso3","year"],right_on=["destination_iso3","_year"],how="left")
        keys=keys.merge(flow_key,left_on=["destination_iso3","hs2","year"],right_on=["destination_iso3","hs2","year"],how="left")
        keys["flow_usd"]=np.where(keys.report_covered.fillna(False),keys.flow_usd.fillna(0),np.nan)
        base[f"history_flow_{j}"]=keys.flow_usd.to_numpy()
        base[f"history_covered_{j}"]=keys.report_covered.fillna(False).to_numpy(bool)
    # Feature year flow from the current trade table; join by key.
    base=base.merge(flows,on=["destination_iso3","hs2","feature_year"],how="left",validate="one_to_one")
    base["historical_export_value"]=base.historical_export_value.fillna(0.0)
    base["historical_market_share"]=base.historical_export_value/base.world_imports_usd
    base["log_lagged_export_value"]=np.log1p(base.historical_export_value.clip(lower=0))
    base["historical_export_presence"]=base.historical_export_value.gt(0).astype("int8")
    base["strict_inactive_verified"]=base[["history_covered_0","history_covered_1","history_covered_2"]].all(axis=1)&base[["history_flow_0","history_flow_1","history_flow_2"]].eq(0).all(axis=1)
    # Historical growth/volatility use only origin-known reporting years.
    base["lagged_export_growth"]=np.where(base.history_flow_2.gt(0),(base.historical_export_value-base.history_flow_2)/base.history_flow_2,np.nan)
    base["historical_export_volatility"]=base[["history_flow_0","history_flow_1","history_flow_2"]].std(axis=1,ddof=0)/base[["history_flow_0","history_flow_1","history_flow_2"]].mean(axis=1).replace(0,np.nan)
    # Resolve time-indexed demand lag with an explicit lookup to avoid accidental future joins.
    dem_lookup=demand.set_index(["destination_iso3","hs2","feature_year"]).world_imports_usd
    base["demand_yminus2"]=[dem_lookup.get((d,h,int(y)-2),np.nan) for d,h,y in base[["destination_iso3","hs2","feature_year"]].itertuples(index=False,name=None)]
    base["destination_import_demand_growth"]=np.where(base.demand_yminus2.gt(0),np.log(base.world_imports_usd/base.demand_yminus2),np.nan)
    base=base.merge(macros.rename(columns={"iso3":"destination_iso3","year":"feature_year"}),on=["destination_iso3","feature_year"],how="left",suffixes=("","_dest"))
    base=base.merge(geo,on="destination_iso3",how="left",validate="many_to_one")
    # Outcome labels are joined only after prediction artifact creation by evaluate_external().
    outcomes=demand.rename(columns={"feature_year":"outcome_year"})[["destination_iso3","hs2","outcome_year","world_imports_usd"]].rename(columns={"world_imports_usd":"_outcome_world_imports"})
    target_flows=flows.rename(columns={"feature_year":"outcome_year","historical_export_value":"_export_usd"})[["destination_iso3","hs2","outcome_year","_export_usd"]]
    base=base.merge(outcomes,on=["destination_iso3","hs2","outcome_year"],how="left",validate="one_to_one")
    base=base.merge(target_flows,on=["destination_iso3","hs2","outcome_year"],how="left",validate="one_to_one")
    base["outcome_observed"]=base._outcome_world_imports.notna()
    base["export_value_usd"]=np.where(base.outcome_observed,base._export_usd.fillna(0),np.nan)
    # H2 expansion eligibility median is computed only from historical market shares through feature year.
    positive=demand.merge(flows,on=["destination_iso3","hs2","feature_year"],how="left")
    positive=positive.loc[positive.historical_export_value.gt(0)].copy()
    positive["share"]=positive.historical_export_value/positive.world_imports_usd
    med_by_year=[]
    for fy in sorted(base.feature_year.unique()):
        hist=positive.loc[positive.feature_year.le(fy)].groupby("hs2").share.median()
        med_by_year.extend((h,int(fy),float(v)) for h,v in hist.items())
    med=pd.DataFrame(med_by_year,columns=["hs2","feature_year","training_median_market_share"])
    base=base.merge(med,on=["hs2","feature_year"],how="left",validate="many_to_one")
    base["task_b_eligible"]=base.historical_export_value.gt(0)&base.training_median_market_share.notna()&base.historical_market_share.le(base.training_median_market_share)
    base["entry_success"]=np.where(base.outcome_observed,base.export_value_usd.ge(THRESHOLD).astype(float),np.nan)
    base["qualified_export_value_usd"]=np.where(base.outcome_observed & base.export_value_usd.ge(THRESHOLD),base.export_value_usd,np.where(base.outcome_observed,0.0,np.nan))
    base["destination_import_demand"]=base.world_imports_usd
    base["log_destination_import_demand"]=np.log1p(base.world_imports_usd)
    base["hs2"]=base.hs2.astype(str).str.zfill(2)
    base["destination_name"]=base.destination_iso3
    base["candidate_pool_id"]=[f"{exporter}|{h}|{int(y)}" for h,y in base[["hs2","outcome_year"]].itertuples(index=False,name=None)]
    base["exporter_iso3"]=exporter
    base["historical_information_cutoff"]=base.feature_year.astype(int)
    base["lag"]=2
    # Every feature year is exactly y-2.
    assert (base.feature_year==base.outcome_year-2).all()
    assert not base.duplicated(["destination_iso3","hs2","outcome_year"]).any()
    return base


def run_external_predictions(source: pd.DataFrame, manifest_hash: str) -> pd.DataFrame:
    """Build all frozen THA/IDN predictions; test outcome columns are excluded before saving."""
    predictions=[]; quality=[]; flow=[]
    vpanel=build_country_panel(source,"VNM")
    # Local chronological OOF probabilities are needed to calibrate the local country model.
    local_oof={}
    for exporter in ["THA","IDN"]:
        panel=build_country_panel(source,exporter)
        # Fitted VNM transfer models depend only on the cutoff, but cached arrays
        # must not be reused across countries because their candidate counts differ.
        vnm_models={}
        oof_rows=[]
        for oy in [2018,2019,2020,2021,2022]:
            ocut=oy-2
            otr=panel.loc[panel.strict_inactive_verified&panel.outcome_observed&panel.entry_success.notna()&panel.outcome_year.le(ocut)].copy()
            ote=panel.loc[panel.outcome_year.eq(oy)&panel.strict_inactive_verified].copy()
            if otr.entry_success.nunique()<2 or ote.empty:
                continue
            # Frozen M1 LightGBM parameters; score is generated only out of fold.
            op=fit_classifier("LightGBM",otr,"entry_success",{
                "n_estimators":120,"num_leaves":31,"learning_rate":0.05,"max_depth":-1,
                "min_child_samples":40,"class_weight":"balanced","n_jobs":4,"verbosity":-1},seed=11)
            cols=op.numeric_features+op.categorical_features
            pr=op.estimator.predict_proba(op.preprocessor.transform(ote[cols]).astype(np.float32))[:,1]
            oof_rows.append(pd.DataFrame({"country":exporter,"outcome_year":oy,"entry_success":ote.entry_success.to_numpy(float),
                "label_observed":ote.outcome_observed.to_numpy(bool),"p_oof":pr}))
        local_oof[exporter]=pd.concat(oof_rows,ignore_index=True) if oof_rows else pd.DataFrame()
        local_oof[exporter].to_parquet(HERE/f"predictions/B_{exporter}_historical_oof_probabilities.parquet",index=False)
        for year in YEARS:
            candidates=panel.loc[panel.outcome_year.eq(year)&panel.destination_import_demand.gt(0)&panel.strict_inactive_verified].copy()
            pools=int(candidates.hs2.nunique())
            quality.append({"country":exporter,"year":year,"all_eligible_candidates":int((panel.outcome_year.eq(year)&panel.destination_import_demand.gt(0)).sum()),
                            "strict_zero_candidates":len(candidates),"hs2_pools":pools,"unknown_labels":int((~candidates.outcome_observed).sum()),
                            "country_code_mapping":"verified BACI codebook","data_source":"BACI HS12 V202601","objective_feasibility":pools>=80})
            if pools<80:
                continue
            cutoff=year-2
            train=panel.loc[panel.outcome_observed&panel.export_value_usd.notna()&panel.outcome_year.le(cutoff)].copy()
            train_entry=panel.loc[panel.strict_inactive_verified&panel.outcome_observed&panel.entry_success.notna()&panel.outcome_year.le(cutoff)].copy()
            train_success=train_entry.loc[train_entry.export_value_usd.ge(THRESHOLD)].copy()
            if train_entry.entry_success.nunique()<2:
                quality[-1]["objective_feasibility"]=False;quality[-1]["failure_reason"]="training entry labels lack both classes";continue
            params={"n_estimators":120,"num_leaves":31,"learning_rate":0.05,"max_depth":-1,"min_child_samples":40,"class_weight":"balanced","n_jobs":4,"verbosity":-1}
            # Include destination-pool predictors only. Strip every outcome-derived column before model scoring.
            test_x=candidates.drop(columns=["export_value_usd","outcome_observed","entry_success","qualified_export_value_usd","_outcome_world_imports","_export_usd"],errors="ignore")
            # Country-specific B2 model.
            clf=fit_classifier("LightGBM",train_entry,"entry_success",params,seed=11)
            xcols=clf.numeric_features+clf.categorical_features
            p_local=clf.estimator.predict_proba(clf.preprocessor.transform(test_x[xcols]).astype(np.float32))[:,1]
            if len(train_success)>=50:
                cond_local,_,diag_local,_=fit_ppml(train_success,candidates)
            else:
                cond_local=np.full(len(candidates),np.nan);diag_local={"converged":False,"iterations":0,"n_parameters":np.nan}
            # B1 frozen-origin transfer: VNM-only models are fit using the same frozen features and cutoff.
            if year not in vnm_models:
                vtrain_entry=vpanel.loc[vpanel.strict_inactive_verified&vpanel.outcome_observed&vpanel.entry_success.notna()&vpanel.outcome_year.le(cutoff)].copy()
                vtrain_success=vtrain_entry.loc[vtrain_entry.export_value_usd.ge(THRESHOLD)].copy()
                vtrain_all=vpanel.loc[vpanel.outcome_observed&vpanel.export_value_usd.notna()&vpanel.outcome_year.le(cutoff)].copy()
                transfer_clf=fit_classifier("LightGBM",vtrain_entry,"entry_success",params,seed=11)
                ppml_transfer,_,diag_t,_=fit_ppml(vtrain_all,candidates)
                if len(vtrain_success)>=50:
                    cond_transfer,_,diag_ct,_=fit_ppml(vtrain_success,candidates)
                else:
                    cond_transfer=np.full(len(candidates),np.nan);diag_ct={"converged":False,"iterations":0,"n_parameters":np.nan}
                vnm_models[year]=(vtrain_entry,vtrain_success,vtrain_all,transfer_clf,ppml_transfer,cond_transfer,diag_t,diag_ct)
            vtrain_entry,vtrain_success,vtrain_all,transfer_clf,ppml_transfer,cond_transfer,diag_t,diag_ct=vnm_models[year]
            p_transfer=transfer_clf.estimator.predict_proba(transfer_clf.preprocessor.transform(test_x[transfer_clf.numeric_features+transfer_clf.categorical_features]).astype(np.float32))[:,1]
            ppml_local,_,diag_p,_=fit_ppml(train,candidates)
            # Calibrated probabilities use only earlier rolling-origin OOF predictions.
            baseline_v=pd.read_parquet(ERGM/"predictions/historical_oof_expert_predictions.parquet")
            calib=_fit_platt(baseline_v,min(cutoff,2021))
            p_transfer_cal=_apply_platt(calib,p_transfer)
            loof=local_oof[exporter]
            loof=loof.loc[loof.outcome_year.le(cutoff)&loof.label_observed.astype(bool)]
            if len(loof)>=100 and loof.entry_success.nunique()==2:
                z=np.log(np.clip(loof.p_oof.to_numpy(float),1e-6,1-1e-6)/(1-np.clip(loof.p_oof.to_numpy(float),1e-6,1-1e-6))).reshape(-1,1)
                local_cal=LogisticRegression(C=1.0,solver="lbfgs",max_iter=2000).fit(z,loof.entry_success.astype(int))
                p_local_cal=_apply_platt(local_cal,p_local)
            else:
                p_local_cal=p_local.copy()
            for mode, p, ppml, cond in [("B1_VNM_zero_shot",p_transfer_cal,ppml_transfer,cond_transfer),
                                        ("B2_country_specific",p_local_cal,ppml_local,cond_local)]:
                g=candidates[["destination_iso3","destination_name","hs2","outcome_year","feature_year","candidate_pool_id","historical_information_cutoff","historical_export_value","destination_import_demand","historical_market_share","strict_inactive_verified","task_b_eligible","training_median_market_share"]].copy()
                g["country"]=exporter;g["validation_mode"]=mode;g["training_cutoff"]=cutoff;g["training_n_all"]=len(vtrain_all if mode.startswith("B1") else train);g["training_n_entry"]=len(vtrain_entry if mode.startswith("B1") else train_entry);g["training_n_success"]=len(vtrain_success if mode.startswith("B1") else train_success)
                g["entry_probability_raw"]=p
                g["entry_probability_calibrated"]=p
                g["conditional_value_ppml_usd"]=np.maximum(cond,THRESHOLD) if np.isfinite(cond).all() else cond
                g["two_part_qualified_value_score"]=g.entry_probability_raw*g.conditional_value_ppml_usd
                g["ppml_export_value_score"]=np.maximum(ppml,0)
                g["lightgbm_entry_score"]=p
                g["rank_blend_score"] = g.groupby(["hs2","outcome_year"])["ppml_export_value_score"].rank(pct=True,method="average") + g.groupby(["hs2","outcome_year"])["lightgbm_entry_score"].rank(pct=True,method="average")
                g["score_provenance"]="outer predictions generated before evaluation labels merged; all training outcomes <= y-2"
                g["task"]=TASK_A
                predictions.append(g)
                flow.append({"country":exporter,"year":year,"mode":mode,"training_cutoff":cutoff,"train_all_n":len(vtrain_all if mode.startswith("B1") else train),
                             "train_entry_n":len(vtrain_entry if mode.startswith("B1") else train_entry),"train_success_n":len(vtrain_success if mode.startswith("B1") else train_success),
                             "classifier":"LightGBM fixed M1 fallback parameters","ppml_converged":diag_t["converged"] if mode.startswith("B1") else diag_p["converged"],
                             "conditional_ppml_converged":diag_ct["converged"] if mode.startswith("B1") else diag_local["converged"],
                             "probability_calibration":"VNM historical OOF Platt" if mode.startswith("B1") else ("country-specific historical OOF Platt" if len(loof)>=100 else "uncalibrated; insufficient OOF")})
            # Existing-market expansion replication: use the same historical model
            # features and compare incremental forecasts, not absolute scores.
            test_b=panel.loc[panel.outcome_year.eq(year)&panel.task_b_eligible].copy()
            if len(test_b):
                local_ppml_b,_,diag_lp_b,_=fit_ppml(train,test_b)
                lgb_params={"n_estimators":120,"num_leaves":31,"learning_rate":0.05,"max_depth":-1,"min_child_samples":40,"n_jobs":4,"verbosity":-1}
                local_reg=fit_regressor("LightGBM",train,lgb_params,seed=11,lag=2)
                local_lgb_b=local_reg.predict(test_b,log_target=True)
                vtrain_all=vpanel.loc[vpanel.outcome_observed&vpanel.export_value_usd.notna()&vpanel.outcome_year.le(cutoff)].copy()
                transfer_ppml_b,_,diag_tp_b,_=fit_ppml(vtrain_all,test_b)
                transfer_reg=fit_regressor("LightGBM",vtrain_all,lgb_params,seed=11,lag=2)
                transfer_lgb_b=transfer_reg.predict(test_b,log_target=True)
                for mode,pp,lgb in [("B1_VNM_zero_shot",transfer_ppml_b,transfer_lgb_b),("B2_country_specific",local_ppml_b,local_lgb_b)]:
                    b=test_b[["destination_iso3","destination_name","hs2","outcome_year","feature_year","candidate_pool_id","historical_information_cutoff","historical_export_value","destination_import_demand","historical_market_share","strict_inactive_verified","task_b_eligible","training_median_market_share"]].copy()
                    b["country"]=exporter;b["validation_mode"]=mode;b["task"]=TASK_B;b["training_cutoff"]=cutoff
                    b["ppml_incremental_score"]=np.maximum(0,np.asarray(pp,float)-b.historical_export_value.to_numpy(float))
                    b["lightgbm_incremental_score"]=np.maximum(0,np.asarray(lgb,float)-b.historical_export_value.to_numpy(float))
                    b["ppml_export_value_score"]=b.ppml_incremental_score
                    b["lightgbm_entry_score"]=b.lightgbm_incremental_score
                    b["rank_blend_score"]=b.groupby(["hs2","outcome_year"])["ppml_incremental_score"].rank(pct=True,method="average")+b.groupby(["hs2","outcome_year"])["lightgbm_incremental_score"].rank(pct=True,method="average")
                    b["entry_probability_raw"]=np.nan;b["entry_probability_calibrated"]=np.nan;b["conditional_value_ppml_usd"]=np.nan;b["two_part_qualified_value_score"]=np.nan
                    b["training_n_all"]=len(vtrain_all if mode.startswith("B1") else train);b["training_n_entry"]=np.nan;b["training_n_success"]=np.nan
                    b["score_provenance"]="Task B incremental-score forecasts generated before evaluation labels merged; training outcomes <= y-2"
                    predictions.append(b)
                    flow.append({"country":exporter,"year":year,"mode":mode,"task":TASK_B,"training_cutoff":cutoff,"train_all_n":len(vtrain_all if mode.startswith("B1") else train),"train_entry_n":np.nan,"train_success_n":np.nan,
                                 "classifier":"not used","ppml_converged":diag_tp_b["converged"] if mode.startswith("B1") else diag_lp_b["converged"],"conditional_ppml_converged":np.nan,"probability_calibration":"not applicable"})
        print(f"Generated frozen external predictions for {exporter}")
    if not predictions:
        raise RuntimeError("No external predictions generated")
    pred=pd.concat(predictions,ignore_index=True)
    # Save candidate predictions before touching/merging evaluation labels.
    out=HERE/"predictions/B_country_prediction_manifest.parquet"
    pred.to_parquet(out,index=False)
    checksum=_sha(out)
    (HERE/"data_manifest/external_prediction_manifest.json").write_text(json.dumps({"prediction_file":str(out),"sha256":checksum,"frozen_protocol_sha256":manifest_hash,"rows":len(pred),"label_columns_absent":True},indent=2))
    pd.DataFrame(quality).to_csv(HERE/"results/B_country_data_quality.csv",index=False)
    pd.DataFrame(flow).to_csv(HERE/"results/B_model_training_log.csv",index=False)
    return pred


def evaluate_external(pred: pd.DataFrame, source: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Only after prediction manifest is persisted, join outcomes and score external experiments."""
    if not (HERE/"data_manifest/external_prediction_manifest.json").exists():
        raise RuntimeError("External prediction manifest/hash is required before outcome evaluation")
    ann=source[["destination_iso3","hs2","year","world_imports_usd","flow_VNM","flow_THA","flow_IDN"]].copy()
    evals=[]; paired=[]; flows=[]
    for exporter in ["THA","IDN"]:
        p=pred.loc[pred.country.eq(exporter)].copy()
        if p.empty: continue
        # Outcomes merge after fixed predictions were saved. Unknown destination-HS2 outcomes remain NaN.
        lab=ann.rename(columns={"year":"outcome_year","world_imports_usd":"_world_imports",f"flow_{exporter}":"export_value_usd"})[["destination_iso3","hs2","outcome_year","_world_imports","export_value_usd"]]
        p=p.merge(lab,on=["destination_iso3","hs2","outcome_year"],how="left",validate="many_to_one")
        p["label_observed"]=p._world_imports.notna()
        p.loc[~p.label_observed,"export_value_usd"]=np.nan
        p["entry_success"]=np.where(p.label_observed,p.export_value_usd.ge(THRESHOLD).astype(float),np.nan)
        p["qualified_export_value_usd"]=np.where(p.label_observed&p.export_value_usd.ge(THRESHOLD),p.export_value_usd,np.where(p.label_observed,0.0,np.nan))
        p["realized_incremental_export_usd"]=(p.export_value_usd-p.historical_export_value).clip(lower=0)
        for (task,mode,year,hs2),g in p.groupby(["task","validation_mode","outcome_year","hs2"],sort=True):
            observed=g.loc[g.label_observed].copy()
            if task==TASK_A:
                model_score=[("LightGBM_probability", "lightgbm_entry_score"),("TwoPart", "two_part_qualified_value_score"),("Gravity_PPML","ppml_export_value_score"),("RankBlend","rank_blend_score")]
                observed["evaluation_gain"]=observed.entry_success.astype(float)
            else:
                model_score=[("LightGBM_incremental","lightgbm_incremental_score"),("Gravity_PPML_incremental","ppml_incremental_score"),("RankBlend","rank_blend_score")]
                observed["evaluation_gain"]=np.minimum(4,np.floor(np.log10(1+observed.realized_incremental_export_usd.fillna(0).clip(lower=0)/50_000))).astype(int)
                observed["evaluation_gain"]=2**observed.evaluation_gain-1
            for model,score in model_score:
                top=g.sort_values([score,"destination_iso3"],ascending=[False,True],kind="mergesort").head(10)
                known=top.loc[top.label_observed]
                knowncc=g.loc[g.label_observed].sort_values([score,"destination_iso3"],ascending=[False,True],kind="mergesort")
                if task==TASK_A:
                    rel=knowncc.entry_success.to_numpy(float)
                    selected_usd=known.qualified_export_value_usd.to_numpy(float)
                    value=known.qualified_export_value_usd.sum(min_count=1)
                else:
                    rel=np.minimum(4,np.floor(np.log10(1+knowncc.realized_incremental_export_usd.fillna(0).clip(lower=0).to_numpy(float)/50_000))).astype(int)
                    rel=2**rel-1
                    selected_usd=known.realized_incremental_export_usd.to_numpy(float)
                    value=known.realized_incremental_export_usd.sum(min_count=1)
                n=min(10,len(rel));gains=rel[:n];disc=1/np.log2(np.arange(2,n+2));idcg=np.sum(np.sort(rel)[::-1][:n]/np.log2(np.arange(2,n+2))) if n else 0
                dcg=np.sum(gains*disc) if n else 0
                precision=float(knowncc.head(10).entry_success.sum()/10) if task==TASK_A else float((knowncc.head(10).realized_incremental_export_usd.ge(50_000)).sum()/10)
                evals.append({"country":exporter,"task":task,"mode":mode,"year":year,"hs2":hs2,"model":model,"candidates":len(g),"observed_labels":len(observed),"unknown_labels":len(g)-len(observed),
                              "ndcg10_complete_case":dcg/idcg if idcg>0 else np.nan,"ndcg_denominator_positive":bool(idcg>0),"precision10":precision,
                              "successes_top10_complete_case_reranked":int(knowncc.head(10).entry_success.sum()) if task==TASK_A else int(knowncc.head(10).realized_incremental_export_usd.ge(50_000).sum()),"qualified_usd_top10_observed_lower_bound":value if task==TASK_A else np.nan,
                              "positive_incremental_usd_top10_observed_lower_bound":value if task==TASK_B else np.nan,
                              "total_usd_top10_observed_lower_bound":known.export_value_usd.sum(min_count=1),"unknown_selected":len(top)-len(known),"unit":"Observed-label sum within full-ranking Top-10; lower bound if unknown selected"})
                flows.append({"country":exporter,"task":task,"mode":mode,"year":year,"hs2":hs2,"model":model,"positive_subthreshold_n":int((observed.export_value_usd.gt(0)&observed.export_value_usd.lt(THRESHOLD)).sum()),"observed_n":len(observed)})
        p.to_parquet(HERE/f"predictions/B_{exporter}_evaluated.parquet",index=False)
    detail=pd.DataFrame(evals)
    if detail.empty:
        raise RuntimeError("No external evaluation rows")
    summary=detail.groupby(["country","task","mode","year","model"],as_index=False).agg(
        pools_total=("hs2","size"),valid_ndcg_pools=("ndcg_denominator_positive","sum"),ndcg10=("ndcg10_complete_case","mean"),
        precision10=("precision10","mean"),entries10=("successes_top10_complete_case_reranked","mean"),qualified_usd10=("qualified_usd_top10_observed_lower_bound","mean"),
        positive_incremental_usd10=("positive_incremental_usd_top10_observed_lower_bound","mean"),
        total_export_usd10=("total_usd_top10_observed_lower_bound","mean"),unknown_selected=("unknown_selected","sum"))
    summary["evidence_status"]="geographic replication; not a new temporal holdout; WDI/WGI revised vintages"
    detail.to_csv(HERE/"results/B_external_pool_metrics.csv",index=False)
    summary.to_csv(HERE/"results/B_country_replication.csv",index=False)
    pd.DataFrame(flows).to_csv(HERE/"results/B_external_target_diagnostics.csv",index=False)
    # Paired, common-pool comparisons. Dollar contrasts exclude pools where either policy
    # selected an unknown outcome; this is complete-case rather than zero imputation.
    for task in [TASK_A,TASK_B]:
        if task==TASK_A:
            comparisons=[("TwoPart","LightGBM_probability"),("TwoPart","RankBlend"),("RankBlend","LightGBM_probability"),("TwoPart","Gravity_PPML")]
            metrics=[("ndcg10_complete_case",False),("precision10",False),("qualified_usd_top10_observed_lower_bound",True)]
        else:
            comparisons=[("Gravity_PPML_incremental","LightGBM_incremental"),("Gravity_PPML_incremental","RankBlend"),("RankBlend","LightGBM_incremental")]
            metrics=[("ndcg10_complete_case",False),("precision10",False),("positive_incremental_usd_top10_observed_lower_bound",True)]
        subd=detail.loc[detail.task.eq(task)].copy()
        for metric,complete_usd in metrics:
            w=subd.pivot_table(index=["country","mode","year","hs2"],columns="model",values=metric,aggfunc="first").reset_index()
            miss=subd.pivot_table(index=["country","mode","year","hs2"],columns="model",values="unknown_selected",aggfunc="first").reset_index()
            for left,right in comparisons:
                if left not in w or right not in w: continue
                q=w.merge(miss,on=["country","mode","year","hs2"],suffixes=("","_miss"))
                q=q.dropna(subset=[left,right]).copy()
                if complete_usd:
                    lm=left+"_miss";rm=right+"_miss"
                    if lm in q and rm in q: q=q.loc[q[lm].eq(0)&q[rm].eq(0)]
                q["difference"]=q[left]-q[right]
                for (country,mode),g in q.groupby(["country","mode"]):
                    z=g[["hs2","difference"]].dropna()
                    if z.empty: continue
                    point,lo,hi,pval=_block_bootstrap(z,"difference",reps=3000,seed=51073)
                    paired.append({"country":country,"task":task,"mode":mode,"metric":metric,"contrast":f"{left}-{right}","difference":point,"median_difference":z.difference.median(),"ci95_hs2_bootstrap_low":lo,"ci95_hs2_bootstrap_high":hi,"signflip_p_raw":pval,"paired_pools":len(z),"hs2_blocks":z.hs2.nunique(),"year_count":g.year.nunique(),"usd_complete_case_only":complete_usd,"status":"geographic replication; year-wise walk-forward; 2022-24 repeated outer years"})
    pair=pd.DataFrame(paired)
    if len(pair):
        pair["holm_p_by_task_metric"] = np.nan
        for _,idx in pair.groupby(["task","metric"]).groups.items():
            pair.loc[idx,"holm_p_by_task_metric"]=multipletests(pair.loc[idx,"signflip_p_raw"].fillna(1).to_numpy(float),method="holm")[1]
    pair.to_csv(HERE/"results/B_country_paired_tests.csv",index=False)
    return detail,summary

