from __future__ import annotations

import numpy as np
import pandas as pd

from .candidate_protocol import HERE
from .ranking_metrics import ndcg_from_gains


def observed_complete_cases(frame: pd.DataFrame) -> pd.DataFrame:
    """Return labeled rows only; callers must not coerce unknown labels to zero."""
    return frame.loc[frame.label_observed.astype(bool)].copy()


def _binary_metrics(y_ranked: np.ndarray) -> dict:
    y = np.asarray(y_ranked, dtype=float)
    gains = y.copy()
    positives = int(y.sum())
    return {"ndcg10": ndcg_from_gains(gains, gains, 10),
            "precision10": float(y[:10].sum() / min(10, len(y))) if len(y) else np.nan,
            "recall10": float(y[:10].sum() / positives) if positives else np.nan,
            "positive_prevalence": float(y.mean()) if len(y) else np.nan}


def run_missing_label_sensitivity(ranked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, coverage = [], []
    primary_models = ["LambdaMART", "Gravity_PPML", "LightGBM", "TabM", "MLP",
                      "RandomForest", "XGBoost", "LogisticRegression", "Historical_entry_frequency",
                      "World_import_demand"]
    for (task, model, seed, hs2, year), full in ranked.loc[ranked.model.isin(primary_models)].groupby(
        ["task", "model", "seed", "hs2", "outcome_year"], sort=False
    ):
        if task != "Task_A_new_entry":
            continue
        full = full.sort_values(["score", "destination_iso3"], ascending=[False, True], kind="mergesort").copy()
        observed = full.label_observed.astype(bool).to_numpy()
        y_known = full.entry_success.to_numpy(float)
        n_unknown = int((~observed).sum())
        top5 = full.head(5)
        top10 = full.head(10)
        miss5 = int((~top5.label_observed.astype(bool)).sum())
        miss10 = int((~top10.label_observed.astype(bool)).sum())
        known_idx = np.flatnonzero(observed)
        known_order = full.iloc[known_idx]
        known_metrics = _binary_metrics(known_order.entry_success.astype(int).to_numpy())
        coverage.append({"task": task, "model": model, "seed": int(seed), "hs2": hs2,
                         "outcome_year": int(year), "exante_candidates": len(full),
                         "observed_outcomes": int(observed.sum()), "missing_outcomes": n_unknown,
                         "missing_in_top5": miss5, "missing_in_top10": miss10,
                         "complete_case_ndcg10": known_metrics["ndcg10"],
                         "complete_case_precision10": known_metrics["precision10"]})
        for scenario, missing_label in [("all_missing_fail", 0), ("all_missing_success", 1)]:
            y = np.where(observed, np.nan_to_num(y_known, nan=0.0), float(missing_label)).astype(int)
            vals = _binary_metrics(y)
            rows.append({"task": task, "model": model, "seed": int(seed), "hs2": hs2,
                         "outcome_year": int(year), "scenario": scenario,
                         "n_candidates": len(full), "observed_labels": int(observed.sum()),
                         "missing_labels": n_unknown, "missing_assignment_is_scenario_only": True,
                         **vals})
    cov = pd.DataFrame(coverage)
    sens = pd.DataFrame(rows)
    cov.to_csv(HERE / "results/E9_missing_label_coverage.csv", index=False, na_rep="")
    sens.to_csv(HERE / "results/E9_missing_label_sensitivity.csv", index=False, na_rep="")
    return cov, sens

