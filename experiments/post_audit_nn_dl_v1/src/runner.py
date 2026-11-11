from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from .data import HERE, PROJECT, YEARS, LAG, ENTRY_THRESHOLD_USD, SEEDS, load_frozen_population, load_baselines
from .pipeline import (OUT, build_e2_predictions, evaluate_forecasting, evaluate_rankings,
                       fit_neural_fold, load_neural_runs, run_discovery, run_regression,
                       build_forecast_predictions)
from market_discovery import add_entry_targets


def _prepare_panels():
    raw, candidates, observed = load_frozen_population(write=True)
    target = add_entry_targets(raw, LAG, ENTRY_THRESHOLD_USD)
    target["eligible_ex_ante"] = target.destination_import_demand.gt(0)
    target["label_observed"] = target.outcome_observed.astype(bool) & target.export_value_usd.notna()
    target["entry_candidate_ex_ante"] = target.entry_history_verified.astype(bool) & target.entry_inactive.astype(bool)
    # Candidate IDs and flags are exactly those in the frozen manifest.
    from .data import stable_candidate_id
    target["candidate_id"] = target.apply(stable_candidate_id, axis=1)
    rebuilt = target.loc[target.outcome_year.isin(YEARS)]
    if set(rebuilt.candidate_id) != set(candidates.candidate_id):
        raise AssertionError("Targeted all-years panel does not reproduce frozen candidate keys")
    return target, candidates, observed


def run(stage: str = "full", years=YEARS, seeds=SEEDS, force: bool = False):
    start = time.perf_counter()
    panel, candidates, observed = _prepare_panels()
    forecast_base, entry_base = load_baselines(candidates)
    from .data import write_input_audit
    write_input_audit(panel, candidates, forecast_base, entry_base)
    status = {"stage": stage, "years": list(map(int, years)), "seeds": list(map(int, seeds)),
              "primary_lag": LAG, "threshold_usd": ENTRY_THRESHOLD_USD,
              "candidate_n": len(candidates), "observed_n": len(observed)}
    (OUT / "logs/run_config.json").write_text(json.dumps(status, indent=2) + "\n")

    e2pred = None
    if stage in {"full", "discovery"}:
        e2pred, e2overall, e2hs, e2year, e2pool, e2tests = run_discovery(
            panel, candidates, forecast_base, entry_base, years=years, seeds=seeds, force=force)
        status["E2"] = {"status": "COMPLETED", "predictions": len(e2pred), "models": sorted(e2pred.model.unique())}
    elif (OUT / "predictions/E2_neural_predictions.parquet").exists():
        e2pred = pd.read_parquet(OUT / "predictions/E2_neural_predictions.parquet")

    neural_reg = None
    if stage in {"full", "regression", "ranking"}:
        neural_reg = run_regression(panel, years=years, seeds=seeds, force=force)
        e1pred = build_forecast_predictions(neural_reg)
        e1overall, e1hs, e1year, e1hs_year, e1seed, e1tests = evaluate_forecasting(e1pred)
        status["E1"] = {"status": "COMPLETED", "neural_prediction_rows": int(neural_reg.shape[0]),
                        "models": sorted(e1pred.model.unique())}
    elif (OUT / "predictions/E1_neural_predictions.parquet").exists():
        neural_reg = load_neural_runs("regression", "F4_full")

    if stage in {"full", "ranking"}:
        if e2pred is None:
            ep = OUT / "predictions/E2_neural_predictions.parquet"
            if not ep.exists():
                raise RuntimeError("E3 requires the completed E2 neural discovery predictions")
            e2pred = pd.read_parquet(ep)
        if neural_reg is None or neural_reg.empty:
            neural_reg = load_neural_runs("regression", "F4_full")
        if neural_reg.empty:
            raise RuntimeError("E3 requires completed MLP and TabM regression runs")
        ranking = evaluate_rankings(panel, forecast_base, neural_reg, e2pred)
        status["E3"] = {"status": "COMPLETED", "ranking_rows": len(ranking[4]),
                        "models": sorted(ranking[0].ranking_model.unique())}

    status["elapsed_seconds"] = time.perf_counter() - start
    (OUT / "logs/run_status.json").write_text(json.dumps(status, indent=2, default=float) + "\n")
    print(json.dumps(status, indent=2, default=float))
    return status


def main():
    parser = argparse.ArgumentParser(description="Audited CPU MLP + official TabM extension")
    parser.add_argument("--stage", choices=["audit", "discovery", "regression", "ranking", "full"], default="full")
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--force", action="store_true", help="refit and overwrite only this versioned extension's run files")
    args = parser.parse_args()
    if args.stage == "audit":
        panel, candidates, observed = _prepare_panels()
        forecast, entry = load_baselines(candidates)
        from .data import write_input_audit
        write_input_audit(panel, candidates, forecast, entry)
        print(f"Frozen input audit passed: candidates={len(candidates):,}, observed={len(observed):,}, missing={len(candidates)-len(observed):,}")
        return
    run(args.stage, args.years, args.seeds, args.force)


if __name__ == "__main__":
    main()
