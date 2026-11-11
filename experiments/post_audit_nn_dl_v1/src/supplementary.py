from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .analysis import (destination_block_ranking_bootstrap, run_feature_ablation_2024,
                       run_heterogeneity, run_missing_label_sensitivity)
from .data import (HERE, LAG, ENTRY_THRESHOLD_USD, stable_candidate_id,
                   load_baselines, load_frozen_population)
from .pipeline import OUT
from market_discovery import add_entry_targets


def run(skip_ablation: bool = False, bootstrap_reps: int = 1000) -> None:
    panel, candidates, _ = load_frozen_population(write=False)
    panel = add_entry_targets(panel, LAG, ENTRY_THRESHOLD_USD)
    panel["eligible_ex_ante"] = panel.destination_import_demand.gt(0)
    panel["label_observed"] = panel.outcome_observed.astype(bool) & panel.export_value_usd.notna()
    panel["entry_candidate_ex_ante"] = panel.entry_history_verified.astype(bool) & panel.entry_inactive.astype(bool)
    panel["candidate_id"] = panel.apply(stable_candidate_id, axis=1)
    forecast, _ = load_baselines(candidates)
    if not (OUT / "predictions/E1_neural_predictions.parquet").exists():
        raise FileNotFoundError("Run the primary full stage before supplementary analyses")
    if not (OUT / "predictions/E3_neural_rankings.parquet").exists():
        raise FileNotFoundError("E3 rankings are required before supplementary analyses")

    completed = {}
    if skip_ablation:
        ablation_table = OUT / "results/feature_ablation.csv"
        cached_predictions = [OUT / "predictions/E2_feature_ablation.parquet",
                             OUT / "predictions/E3_feature_ablation.parquet"]
        if ablation_table.exists() and all(path.exists() for path in cached_predictions):
            completed["feature_ablation"] = {
                "status": "COMPLETED_REUSED_SAVED_FITS",
                "rows": int(len(pd.read_csv(ablation_table))),
                "year": 2024, "seed": 11,
            }
        else:
            completed["feature_ablation"] = "NOT_RUN_BY_REQUEST"
    else:
        ablation = run_feature_ablation_2024(panel)
        completed["feature_ablation"] = {"status": "COMPLETED_EXPLORATORY_SINGLE_YEAR_SEED",
                                          "rows": len(ablation), "year": 2024, "seed": 11}

    heterogeneity = run_heterogeneity(panel, candidates)
    completed["heterogeneity"] = {"status": "COMPLETED_DESCRIPTIVE_COMPLETE_CASE",
                                   "rows": len(heterogeneity)}
    missing = run_missing_label_sensitivity(panel, candidates, forecast)
    completed["missing_label_sensitivity"] = {"status": "COMPLETED_NO_IMPUTATION",
                                               "rows": len(missing)}
    ranking = pd.read_parquet(OUT / "predictions/E3_neural_rankings.parquet")
    destination = destination_block_ranking_bootstrap(ranking, reps=bootstrap_reps)
    completed["destination_block_ranking_bootstrap"] = {
        "status": "COMPLETED_SENSITIVITY",
        "rows": len(destination), "repetitions": bootstrap_reps,
    }
    (OUT / "logs/supplementary_status.json").write_text(json.dumps(completed, indent=2) + "\n")
    print(json.dumps(completed, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Post-fit sensitivity analyses; no model refit except stated ablations")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    args = parser.parse_args()
    run(args.skip_ablation, args.bootstrap_reps)


if __name__ == "__main__":
    main()
