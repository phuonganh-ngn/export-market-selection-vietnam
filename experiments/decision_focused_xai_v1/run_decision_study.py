from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd

from .src.candidate_protocol import (HERE, YEARS, load_baseline_predictions,
                                     load_panel_and_candidates)
from .src.decision_explainability import run_explanations, run_retrained_ablation
from .src.decision_regret import compute_regret
from .src.evaluation import evaluate_all
from .src.lambdamart import train_outer
from .src.missing_label_sensitivity import run_missing_label_sensitivity
from .src.ranking_stability import (run_candidate_eligibility_sensitivity,
                                    run_stability)
from .src.reporting import finalize_reports
from .src.statistical_inference import (destination_block_ndcg_sensitivity,
                                        paired_model_comparisons)
from .src.xendcg import evaluate_rank_xendcg, train_rank_xendcg


def _log(stage: str, status: str, started: float, extra=None):
    path = HERE / "logs/progress.json"
    try:
        state = json.loads(path.read_text()) if path.exists() else {"stages": []}
    except Exception:
        state = {"stages": []}
    state.setdefault("stages", []).append({"stage": stage, "status": status,
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3), "details": extra or {}})
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def audit():
    panel, test, manifest, counts = load_panel_and_candidates(write=True)
    base = load_baseline_predictions(test)
    from .src.candidate_protocol import write_input_audit
    write_input_audit(panel, test, manifest, counts)
    timing = []
    for year in YEARS:
        for task in ["Task_A_new_entry", "Task_B_existing_underexploited"]:
            pool = manifest.loc[manifest.task.eq(task) & manifest.outcome_year.eq(year)]
            timing.append({"task": task, "evaluation_year": year, "feature_cutoff_y_minus_2": year - 2,
                           "training_outcome_max_y_minus_2": year - 2,
                           "candidates": len(pool), "observed_labels": int(pool.label_observed.sum()),
                           "unknown_labels": int((~pool.label_observed).sum()),
                           "candidate_pools": int(pool.candidate_pool_id.nunique()),
                           "leakage_assertion": "passed"})
    timing = pd.DataFrame(timing)
    timing.to_csv(HERE / "results/temporal_protocol_manifest.csv", index=False)
    report = ["# Temporal validation protocol", "", "Outer years: 2022, 2023, 2024. Predictor and observed training outcome cutoff is y−2. Inner validation outcome year is y−2 and inner training outcomes end at y−4. All comparisons are product-year pool matched.",
              "The rolling-origin folds are dependent: the y−2 validation labels for a later origin may have been an earlier outer test year. This follows the audited sequential information set and is disclosed; it is not independent repeated holdout evidence.",
              "", timing.to_markdown(index=False), "", "WDI/WGI historical vintages are revised, so no archival point-in-time claim is made."]
    (HERE / "reports/TEMPORAL_VALIDATION.md").write_text("\n".join(report) + "\n")
    return {"candidate_task_rows": len(manifest), "baseline_rows": len(base), "years": list(YEARS)}


def train():
    panel, test, manifest, _ = load_panel_and_candidates(write=True)
    pred, cost = train_outer(panel, manifest)
    return {"ranker_predictions": len(pred),
            "outer_final_fits": int(cost.stage.eq("outer_final_fit").sum()),
            "inner_tuning_configurations": int(cost.stage.eq("inner_chronological_tuning").sum()),
            "seeds": sorted(pred.seed.unique().astype(int).tolist())}


def evaluate():
    panel, test, manifest, _ = load_panel_and_candidates(write=True)
    path = HERE / "predictions/E6_lambdamart_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError("Run --stage train before --stage evaluate")
    lam = pd.read_parquet(path)
    if not lam.model.eq("LambdaMART").all() or not set(lam.seed.unique()).issuperset({11, 23, 37, 51, 73}):
        raise AssertionError("LambdaMART seed prediction file is incomplete or contaminated by baseline rows")
    results = evaluate_all(test, lam)
    return {"candidate_prediction_rows": len(results[0]), "pool_metric_rows": len(results[1]),
            "models": sorted(results[2].model.unique().tolist())}


def decisions():
    panel, _, _, _ = load_panel_and_candidates(write=False)
    path = HERE / "predictions/E6_ranking_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError("Run --stage evaluate before decision metrics")
    ranked = pd.read_parquet(path)
    regret = compute_regret(ranked)
    detail, summary, temporal, robust = run_stability(ranked, pd.read_csv(HERE / "results/E6_pool_metrics.csv"))
    eligibility = run_candidate_eligibility_sensitivity(panel, ranked)
    return {"regret_rows": len(regret), "stability_rows": len(detail),
            "training_window_rows": len(temporal), "robust_policy_rows": len(robust),
            "candidate_eligibility_rows": len(eligibility)}


def explain():
    panel, test, manifest, _ = load_panel_and_candidates(write=True)
    ranked = pd.read_parquet(HERE / "predictions/E6_ranking_predictions.parquet")
    pair, groups, fidelity, stability, perturb = run_explanations(panel, test, manifest)
    ablation, preds = run_retrained_ablation(panel, manifest, ranked)
    return {"pairwise_feature_rows": len(pair), "group_attribution_rows": len(groups),
            "fidelity_rows": len(fidelity), "fixed_perturbation_rows": len(perturb),
            "retrained_ablation_rows": len(ablation), "retrained_ablation_prediction_rows": len(preds)}


def missing():
    ranked = pd.read_parquet(HERE / "predictions/E6_ranking_predictions.parquet")
    coverage, sensitivity = run_missing_label_sensitivity(ranked)
    return {"coverage_rows": len(coverage), "sensitivity_scenarios": len(sensitivity)}


def rank_xendcg():
    panel, _, manifest, _ = load_panel_and_candidates(write=False)
    pred, cost = train_rank_xendcg(panel, manifest)
    pool, summary = evaluate_rank_xendcg(pred)
    cost_path = HERE / "results/computational_cost.csv"
    main_cost = pd.read_csv(cost_path)
    main_cost = main_cost.loc[~main_cost.stage.eq("outer_rank_xendcg_sensitivity")]
    pd.concat([main_cost, cost], ignore_index=True, sort=False).to_csv(cost_path, index=False, na_rep="")
    return {"rank_xendcg_predictions": len(pred), "outer_fits": len(cost),
            "pool_metrics": len(pool), "summary_rows": len(summary)}


def reports():
    metrics = pd.read_csv(HERE / "results/E6_pool_metrics.csv")
    regret = pd.read_csv(HERE / "results/E7_regret_at_5_10.csv")
    inference = paired_model_comparisons(metrics, regret)
    ranked = pd.read_parquet(HERE / "predictions/E6_ranking_predictions.parquet")
    dest_boot = destination_block_ndcg_sensitivity(ranked)
    result = finalize_reports()
    return {"paired_comparisons": len(inference), "destination_block_comparisons": len(dest_boot), "reports": result}


def run(stage: str):
    stages = {
        "audit": audit, "train": train, "evaluate": evaluate,
        "decision": decisions, "xai": explain, "missing": missing,
        "rank_xendcg": rank_xendcg, "report": reports,
    }
    if stage == "full":
        ordered = ["audit", "train", "rank_xendcg", "evaluate", "decision", "missing", "xai", "report"]
    else:
        ordered = [stage]
    output = {}
    for current in ordered:
        start = time.perf_counter()
        print(f"[{current}] running", flush=True)
        details = stages[current]()
        _log(current, "COMPLETED", start, details)
        output[current] = details
        print(f"[{current}] completed: {details}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description="M1 decision-focused LambdaMART and ranking-XAI study")
    parser.add_argument("--stage", choices=["audit", "train", "rank_xendcg", "evaluate", "decision", "missing", "xai", "report", "full"], default="full")
    args = parser.parse_args()
    run(args.stage)


if __name__ == "__main__":
    main()
