from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

from experiments.preference_robust_selection_v1.src.data_audit import run_audit
from experiments.preference_robust_selection_v1.src.predictive_uncertainty import build_scenarios
from experiments.preference_robust_selection_v1.src.robust_selection import run_optimization, append_linear_amount_utility_sensitivity
from experiments.preference_robust_selection_v1.src.portfolio_evaluation import evaluate_portfolios
from experiments.preference_robust_selection_v1.src.decision_explanations import explain_decisions
from experiments.preference_robust_selection_v1.src.statistical_inference import run_inference
from experiments.preference_robust_selection_v1.src.reporting import run_reporting


def record(stage: str, result: dict, started: float):
    log = HERE / "results/stage_log.jsonl"
    row = {"stage": stage, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "runtime_seconds": time.perf_counter() - started, "result": result}
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    print(json.dumps(row, default=str), flush=True)


def run_stage(stage: str):
    start = time.perf_counter()
    if stage == "audit":
        result = run_audit()
    elif stage == "scenarios":
        result = build_scenarios()
    elif stage == "optimize":
        result = run_optimization()
    elif stage == "utility-sensitivity":
        result = append_linear_amount_utility_sensitivity()
    elif stage == "evaluate":
        result = evaluate_portfolios()
    elif stage == "explain":
        result = explain_decisions()
    elif stage == "inference":
        result = run_inference()
    elif stage == "report":
        result = run_reporting()
    else:
        raise ValueError(stage)
    record(stage, result, start)


def main():
    parser = argparse.ArgumentParser(description="Preference-robust export-market selection research extension")
    parser.add_argument("--mode", choices=["audit", "scenarios", "optimize", "utility-sensitivity", "evaluate", "explain", "inference", "report", "full"], default="full")
    args = parser.parse_args()
    stages = ["audit", "scenarios", "optimize", "utility-sensitivity", "evaluate", "explain", "inference", "report"] if args.mode == "full" else [args.mode]
    for stage in stages:
        run_stage(stage)


if __name__ == "__main__":
    main()
