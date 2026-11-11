from __future__ import annotations

import numpy as np
import pandas as pd


def _gain(d: pd.DataFrame, task: str) -> np.ndarray:
    if task == "Task_A_new_entry":
        return d.entry_success.to_numpy(float)
    inc = np.maximum(0.0, d.realized_incremental_export_usd.to_numpy(float))
    grade = np.minimum(4, np.floor(np.log10(1.0 + inc / 50_000.0))).astype(int)
    return np.exp2(grade) - 1.0


def _dcg(gain: np.ndarray, k: int) -> float:
    return float(np.sum(np.asarray(gain, dtype=float)[:k] / np.log2(np.arange(2, min(k, len(gain)) + 2))))


def pool_decision_metrics(frame: pd.DataFrame, score_col: str, task: str, k_values=(5, 10)) -> dict:
    """Evaluate a full ex-ante score order on known outcomes; unknowns stay NA.

    The reported NDCG/Top-K statistics use the complete-case candidate intersection,
    with a stable destination-code tie-break. Counts of unobserved names in the
    score-based Top-K are returned separately and are not coded as zero gains.
    """
    ordered_all = frame.sort_values([score_col, "destination_iso3"], ascending=[False, True], kind="mergesort")
    known = ordered_all.loc[ordered_all.label_observed.astype(bool)].copy()
    missing_top = {k: int((~ordered_all.label_observed.astype(bool).iloc[:k]).sum()) for k in k_values}
    out = {"candidate_count": int(len(frame)), "observed_label_count": int(len(known)),
           "unknown_label_count": int(len(frame)-len(known))}
    if known.empty:
        for k in k_values:
            out.update({f"ndcg@{k}": np.nan, f"precision@{k}": np.nan,
                        f"regret@{k}": np.nan, f"selected_gain@{k}": np.nan,
                        f"oracle_gain@{k}": np.nan, f"missing_in_full_score_top{k}": missing_top[k],
                        f"selected_positive_usd@{k}": np.nan})
        return out
    gains = _gain(known, task)
    # Task A precision is entry success. Task B expansion frequency uses the
    # audited economically meaningful USD 50,000 incremental threshold.
    if task == "Task_A_new_entry":
        binary = known.entry_success.to_numpy(float)
    else:
        binary = (known.realized_incremental_export_usd.to_numpy(float) >= 50_000).astype(float)
    for k in k_values:
        n = min(int(k), len(known))
        top = gains[:n]
        ideal = np.sort(gains)[::-1][:n]
        idcg = _dcg(ideal, n)
        ndcg = _dcg(top, n)/idcg if idcg > 0 else np.nan
        selected = float(np.sum(top))
        oracle = float(np.sum(ideal))
        regret = max(0.0, oracle-selected)
        known_top_fullscore = ordered_all.loc[ordered_all.label_observed.astype(bool)].head(n)
        usd = np.maximum(0.0, known_top_fullscore.export_value_usd.to_numpy(float))
        if task == "Task_B_existing_underexploited":
            usd = np.maximum(0.0, known_top_fullscore.realized_incremental_export_usd.to_numpy(float))
        out.update({f"ndcg@{k}": ndcg, f"precision@{k}": float(binary[:n].sum()/n) if n else np.nan,
                    f"regret@{k}": regret, f"selected_gain@{k}": selected,
                    f"oracle_gain@{k}": oracle, f"missing_in_full_score_top{k}": missing_top[k],
                    f"selected_positive_usd@{k}": float(np.sum(usd))})
    out["mean_entry_success_probability_observed"] = float(known.entry_success.mean()) if task == "Task_A_new_entry" else np.nan
    return out


def metrics_for_scored_candidates(candidates: pd.DataFrame, method_score: str,
                                  policy_name: str, outer_year: int, k: int = 10) -> dict:
    task = str(candidates.task.iloc[0])
    out = pool_decision_metrics(candidates, method_score, task, (5, 10))
    return {"task": task, "outcome_year": int(outer_year), "hs2": str(candidates.hs2.iloc[0]).zfill(2),
            "policy": policy_name, **out}

