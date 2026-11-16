"""Update the objective-aligned baselines after a change of the expansion relevance grading, without a full re-fit.

`aligned_baselines.py` re-fits six LightGBM variants for seven years. Only part of that work depends on the
expansion grading:

* the discovery variants (D_rank, D_pois) do not depend on it at all, so their scores are reused;
* E_rank is trained on the grade, so it is always re-fitted;
* E_inc, E_pinc and E_pois use the grading only in hyperparameter tuning, so tuning is repeated and the model is
  re-fitted only when the chosen parameters differ from the previous run.

The result equals a full run of `aligned_baselines.py` under the new grading, given deterministic fits.

Usage (project root, PYTHONPATH=src:experiments/paper_revision_v1/src:.):
    python experiments/paper_revision_v1/src/aligned_regrade.py year 2018 --previous-scores OLD.parquet --previous-tuning OLD.csv
    ...one call per outcome year, which can run in parallel...
    python experiments/paper_revision_v1/src/aligned_regrade.py merge

`--previous-scores` and `--previous-tuning` are `predictions/aligned_baseline_scores.parquet` and
`results/aligned/tuning.csv` of the run made under the earlier grading.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import aligned_baselines as ab

WORK = ab.OUT / "regrade_parts"
ID = ["destination_iso3", "hs2", "outcome_year"]


def regrade_year(year: int, previous_scores: str, previous_tuning: str) -> None:
    ab.BASE["n_jobs"] = int(os.environ.get("ALIGNED_THREADS", "4"))
    old_scores = pd.read_parquet(previous_scores)
    old_scores = old_scores[old_scores.outcome_year == year]
    old_tuning = pd.read_csv(previous_tuning)
    old_tuning = old_tuning[old_tuning.year == year].set_index("model")

    panel, _ = ab.load_and_freeze(write=False)
    panel = ab.zfill_hs2(panel)
    usable = panel.destination_import_demand.gt(0)
    train = panel.loc[panel.outcome_observed.astype(bool) & panel.export_value_usd.notna() & usable & panel.outcome_year.le(year - 2)]
    test = panel.loc[panel.outcome_year.eq(year) & usable]
    candidates = {
        ab.TA: test.loc[test.entry_candidate_ex_ante.astype(bool) & test.strict_inactive_verified.astype(bool)],
        ab.TB: test.loc[test.task_b_eligible.astype(bool)],
    }

    scores, tuning = [], []
    for spec in ab.specs():
        previous = old_tuning.loc[spec.name]
        previous_params = (int(previous.num_leaves), int(previous.n_estimators))
        if spec.task == ab.TA:                       # independent of the expansion grading
            scores.append(old_scores[old_scores.model == spec.name])
            tuning.append({"year": year, "model": spec.name, "num_leaves": previous_params[0], "n_estimators": previous_params[1],
                           "inner_ndcg10": previous.inner_ndcg10, "refit": False})
            continue
        params, inner = ab.tune(spec, train)
        refit = spec.name == "E_rank" or (params["num_leaves"], params["n_estimators"]) != previous_params
        if refit:
            pred = ab.fit_predict(spec, train, candidates[spec.task], params, ab.SEEDS)
            scores.append(candidates[spec.task][ID].assign(task=spec.task, model=spec.name, score=np.asarray(pred, float)))
        else:
            scores.append(old_scores[old_scores.model == spec.name])
        tuning.append({"year": year, "model": spec.name, **params, "inner_ndcg10": inner, "refit": refit})
        ab.log(f"{year} {spec.name}: {params} inner NDCG {inner:.3f} refit={refit}")

    WORK.mkdir(parents=True, exist_ok=True)
    pd.concat(scores, ignore_index=True).to_parquet(WORK / f"scores_{year}.parquet", index=False)
    pd.DataFrame(tuning).to_csv(WORK / f"tuning_{year}.csv", index=False)


def merge() -> None:
    """Combine the yearly parts into the standard outputs and evaluate them."""
    parts = sorted(WORK.glob("scores_*.parquet"))
    if not parts:
        raise SystemExit(f"no yearly parts in {WORK}")
    scores = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    scores.to_parquet(ab.PRED_DIR / "aligned_baseline_scores.parquet", index=False)
    pd.concat([pd.read_csv(p) for p in sorted(WORK.glob("tuning_*.csv"))]).to_csv(ab.OUT / "tuning.csv", index=False)
    ab.evaluate(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("year", help="update one outcome year")
    one.add_argument("year", type=int)
    one.add_argument("--previous-scores", required=True)
    one.add_argument("--previous-tuning", required=True)
    sub.add_parser("merge", help="merge the yearly parts and evaluate")
    args = parser.parse_args()
    if args.command == "year":
        regrade_year(args.year, args.previous_scores, args.previous_tuning)
    else:
        merge()
