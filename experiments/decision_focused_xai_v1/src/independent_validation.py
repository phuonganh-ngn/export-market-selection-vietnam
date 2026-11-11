from __future__ import annotations

import numpy as np
import pandas as pd
import json

from .candidate_protocol import HERE


def _manual_ndcg(ranked: pd.DataFrame, task: str, k: int = 10) -> float:
    d = ranked.loc[ranked.label_observed.astype(bool)].sort_values(
        ["score", "destination_iso3"], ascending=[False, True], kind="mergesort")
    if task == "Task_A_new_entry":
        gain = d.entry_success.astype(float).to_numpy()
    else:
        inc = np.maximum(0.0, d.realized_incremental_export_usd.to_numpy(float))
        grade = np.minimum(4, np.floor(np.log10(1.0 + inc / 50_000.0)))
        gain = np.exp2(grade) - 1.0
    discount = 1.0 / np.log2(np.arange(2, min(k, len(gain)) + 2))
    ideal_gain = np.sort(gain)[::-1]
    ideal_discount = 1.0 / np.log2(np.arange(2, min(k, len(ideal_gain)) + 2))
    ideal = float(np.dot(ideal_gain[:k], ideal_discount))
    if ideal <= 0:
        return np.nan
    return float(np.dot(gain[:k], discount) / ideal)


def validate_saved_results() -> dict:
    ranked = pd.read_parquet(HERE / "predictions/E6_ranking_predictions.parquet")
    lambda_pred = pd.read_parquet(HERE / "predictions/E6_lambdamart_predictions.parquet")
    manifest = pd.read_csv(HERE / "results/candidate_manifest.csv")
    metrics = pd.read_csv(HERE / "results/E6_pool_metrics.csv")
    regret = pd.read_csv(HERE / "results/E7_regret_at_5_10.csv")
    checks = []

    # Frozen task sets, every prediction policy and row uniqueness.
    all_set_match = True
    for task, g_task in manifest.groupby("task"):
        wanted = set(g_task.candidate_id.astype(str))
        for (model, seed), g in ranked.loc[ranked.task.eq(task)].groupby(["model", "seed"]):
            actual = set(g.candidate_id.astype(str))
            if actual != wanted or g.candidate_id.duplicated().any():
                all_set_match = False
                raise AssertionError(f"Candidate mismatch/duplicate for {task} {model} seed={seed}")
    checks.append(("All models use identical frozen ex-ante candidates within task", all_set_match))

    lambda_seeds = set(lambda_pred.seed.unique().astype(int))
    checks.append(("All five prespecified LambdaMART seeds are saved", lambda_seeds == {11, 23, 37, 51, 73}))
    if lambda_seeds != {11, 23, 37, 51, 73}:
        raise AssertionError(f"Unexpected saved LambdaMART seed set: {lambda_seeds}")
    checks.append(("LambdaMART raw predictions contain no baseline rows", lambda_pred.model.eq("LambdaMART").all()))
    if not lambda_pred.model.eq("LambdaMART").all():
        raise AssertionError("Raw LambdaMART file includes non-Lambda model outputs")
    lambda_key = ["candidate_id", "task", "seed"]
    pure_scores = lambda_pred[lambda_key + ["score"]].rename(columns={"score": "pure_score"})
    ranked_scores = ranked.loc[ranked.model.eq("LambdaMART") & ranked.seed.ge(11),
                               lambda_key + ["score"]].rename(columns={"score": "ranked_score"})
    score_check = pure_scores.merge(ranked_scores, on=lambda_key, how="outer", validate="one_to_one", indicator=True)
    prediction_match = (score_check._merge.eq("both").all()
                        and np.allclose(score_check.pure_score, score_check.ranked_score, rtol=0, atol=0))
    checks.append(("Final rankings preserve every raw LambdaMART score exactly", bool(prediction_match)))
    if not prediction_match:
        raise AssertionError("Raw LambdaMART scores do not match the final merged ranking output")

    from .lambdamart import SEEDS, _cache_paths
    timing_ok = True
    for task in ["Task_A_new_entry", "Task_B_existing_underexploited"]:
        for year in [2022, 2023, 2024]:
            for seed in SEEDS:
                _, meta_path = _cache_paths(task, year, seed)
                if not meta_path.exists():
                    timing_ok = False
                    raise AssertionError(f"Missing fitted model metadata: {meta_path.name}")
                meta = json.loads(meta_path.read_text())
                timing_ok &= (int(meta["training_outcome_max"]) <= year - 2
                              and int(meta["training_outcome_cutoff"]) == year - 2
                              and int(meta["feature_cutoff"]) == year - 2)
                if not timing_ok:
                    raise AssertionError(f"Temporal cutoff violation in {meta_path.name}")
    checks.append(("All 30 fitted models satisfy the audited y-2 cutoffs", timing_ok))

    unknown_manifest = manifest.loc[~manifest.label_observed.astype(bool)]
    unknown_eval = int(unknown_manifest.loc[unknown_manifest.task.eq("Task_A_new_entry")].candidate_id.nunique())
    missing_success_is_na = bool(unknown_manifest.entry_success.isna().all())
    checks.append(("Task A unknown labels remain NA", unknown_eval == 559 and missing_success_is_na))
    if unknown_eval != 559 or not missing_success_is_na:
        raise AssertionError(f"Unknown-label integrity failed: n={unknown_eval}, entry NA={missing_success_is_na}")

    # Independently reproduce several saved pool metrics by direct array arithmetic.
    pairs = [
        ("Task_A_new_entry", "LambdaMART", -2, "01", 2022),
        ("Task_A_new_entry", "LightGBM", -1, "48", 2024),
        ("Task_B_existing_underexploited", "Gravity_PPML", -1, "01", 2022),
        ("Task_B_existing_underexploited", "LambdaMART", -2, "48", 2024),
    ]
    max_metric_error = 0.0
    for task, model, seed, hs2, year in pairs:
        g = ranked.loc[ranked.task.eq(task) & ranked.model.eq(model) & ranked.seed.eq(seed)
                       & ranked.hs2.astype(str).str.zfill(2).eq(hs2) & ranked.outcome_year.eq(year)].copy()
        if g.empty:
            raise AssertionError(f"Missing saved validation pool {task}/{model}/{hs2}/{year}")
        actual = _manual_ndcg(g, task)
        saved = metrics.loc[metrics.task.eq(task) & metrics.model.eq(model) & metrics.seed.eq(seed)
                            & metrics.hs2.astype(str).str.zfill(2).eq(hs2) & metrics.outcome_year.eq(year), "ndcg_at_10"]
        if len(saved) != 1:
            raise AssertionError("Expected one saved pool metric for independent verification")
        saved_value = float(saved.iloc[0])
        if np.isfinite(actual) != np.isfinite(saved_value):
            raise AssertionError(f"Finite/NA NDCG status mismatch for {task}/{model}/{hs2}/{year}")
        error = abs(actual - saved_value) if np.isfinite(actual) else 0.0
        max_metric_error = max(max_metric_error, error)
        if error > 1e-12:
            raise AssertionError(f"Independent NDCG mismatch for {task}/{model}/{hs2}/{year}: {actual} vs {saved_value}")
    checks.append(("Independent NDCG@10 recomputation matches four saved pools", max_metric_error <= 1e-12))

    if regret.regret.lt(-1e-12).any():
        raise AssertionError("Negative realized-outcome regret found")
    checks.append(("All observed-pool regret values are nonnegative", bool(regret.regret.ge(-1e-12).all())))
    if not np.isfinite(ranked.score.to_numpy(float)).all():
        raise AssertionError("Nonfinite model ranking scores found")
    checks.append(("All saved candidate ranking scores are finite", True))

    report = ["# Independent saved-output validation", "", "Validation is recomputed from the stored prediction files, rather than the report tables alone.", ""]
    report.extend([f"- {'PASS' if ok else 'FAIL'} — {description}" for description, ok in checks])
    report += ["", f"Maximum independently recalculated NDCG@10 absolute error: {max_metric_error:.3e}.",
               f"Final ranked prediction rows: {len(ranked):,}; raw LambdaMART rows: {len(lambda_pred):,}; frozen manifest rows: {len(manifest):,}."]
    (HERE / "reports/INDEPENDENT_VALIDATION.md").write_text("\n".join(report) + "\n")
    return {"checks": len(checks), "passed": sum(ok for _, ok in checks),
            "max_ndcg_absolute_error": max_metric_error, "ranked_prediction_rows": len(ranked),
            "lambda_prediction_rows": len(lambda_pred), "unknown_entry_labels": unknown_eval}
