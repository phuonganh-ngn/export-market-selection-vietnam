from __future__ import annotations

import numpy as np
import pandas as pd

from .candidate_protocol import HERE


def _gain(frame: pd.DataFrame, task: str, definition: str) -> np.ndarray:
    if task == "Task_A_new_entry":
        return frame.entry_success.astype(int).to_numpy(float)
    inc = np.maximum(0.0, frame.realized_incremental_export_usd.to_numpy(float))
    if definition == "positive_increment_usd":
        return inc
    grade = np.minimum(4, np.floor(np.log10(1.0 + inc / 50_000.0))).astype(int)
    if definition == "capped_grade_gain":
        return np.exp2(grade) - 1.0
    if definition == "binary_expansion":
        return (inc >= 50_000.0).astype(float)
    raise ValueError(definition)


def compute_regret(ranked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, model, seed, hs2, year), full in ranked.groupby(
        ["task", "model", "seed", "hs2", "outcome_year"], sort=False
    ):
        d = full.loc[full.label_observed.astype(bool)].sort_values(
            ["score", "destination_iso3"], ascending=[False, True], kind="mergesort"
        ).copy()
        if d.empty:
            continue
        definitions = ["binary_entry"] if task == "Task_A_new_entry" else [
            "positive_increment_usd", "capped_grade_gain", "binary_expansion"]
        for definition in definitions:
            gains = _gain(d, task, definition)
            oracle = np.sort(gains)[::-1]
            for k in [5, 10]:
                k_eff = min(k, len(d))
                selected_total = float(gains[:k_eff].sum())
                oracle_total = float(oracle[:k_eff].sum())
                regret = max(0.0, oracle_total - selected_total)
                rows.append({"task": task, "model": model, "seed": int(seed), "hs2": str(hs2).zfill(2),
                             "outcome_year": int(year), "k": k, "gain_definition": definition,
                             "n_observed_candidates": len(d), "selected_gain": selected_total,
                             "oracle_gain": oracle_total, "regret": regret,
                             "normalized_regret": regret / oracle_total if oracle_total > 0 else np.nan,
                             "oracle_gain_zero": bool(oracle_total == 0),
                             "candidate_population": "observed-label intersection; same query for model and oracle"})
    out = pd.DataFrame(rows)
    if (out.regret < -1e-10).any():
        raise AssertionError("Negative regret indicates inconsistent oracle/model candidate sets or gains")
    out.to_csv(HERE / "results/E7_regret_at_5_10.csv", index=False, na_rep="")
    main = out.loc[
        ((out.task.eq("Task_A_new_entry")) & out.gain_definition.eq("binary_entry"))
        | ((out.task.eq("Task_B_existing_underexploited")) & out.gain_definition.eq("capped_grade_gain"))
    ]
    summary = main.groupby(["task", "model", "seed", "k"], as_index=False).agg(
        mean_regret=("regret", "mean"), mean_normalized_regret=("normalized_regret", "mean"),
        mean_selected_gain=("selected_gain", "mean"), mean_oracle_gain=("oracle_gain", "mean"),
        pools=("regret", "size"), valid_normalized=("normalized_regret", "count"))
    summary.to_csv(HERE / "results/E7_regret_summary.csv", index=False, na_rep="")
    return out

