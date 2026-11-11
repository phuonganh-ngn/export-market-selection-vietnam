from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from .robust_selection import _pivot_scenarios, SCENARIO_SETS
from .preference_sets import PREFERENCE_SETS, vertices

HERE = Path(__file__).resolve().parents[1]
TASK_A = "Task_A_new_entry"
TASK_B = "Task_B_existing_underexploited"


def _best_one_swap(util: np.ndarray, oracle: np.ndarray, selected_idx: np.ndarray,
                   candidate_idx: int, ids: list[str]) -> tuple[int, float]:
    """Return the selected index whose replacement minimizes worst-case regret.

    The vectorized expression is algebraically equivalent to recomputing each
    one-for-one portfolio against every scenario and its same-cardinality oracle.
    """
    base = oracle[:, :len(selected_idx)].sum(axis=1) - util[:, selected_idx].sum(axis=1)
    trial = base[None, :] + util[:, selected_idx].T - util[:, candidate_idx][None, :]
    maximum = trial.max(axis=1)
    position = min(range(len(selected_idx)), key=lambda z: (maximum[z], str(ids[selected_idx[z]])))
    return int(selected_idx[position]), float(maximum[position])


def explain_decisions() -> dict:
    start = time.perf_counter()
    selections = pd.read_parquet(HERE / "policies/selected_portfolios.parquet")
    selections["hs2"] = selections.hs2.astype(str).str.zfill(2)
    selection_groups = {key: group for key, group in selections.groupby(
        ["task", "hs2", "outcome_year", "decision_budget_k", "selection_method"], sort=False)}
    manifest = pd.read_csv(HERE / "results/candidate_task_manifest.csv", dtype={"hs2": str})
    manifest["hs2"] = manifest.hs2.astype(str).str.zfill(2)
    scenarios = pd.read_parquet(HERE / "scenarios/predictive_utility_scenarios.parquet")
    scenarios["hs2"] = scenarios.hs2.astype(str).str.zfill(2)
    scenario_indices = scenarios.groupby(["task", "outcome_year", "hs2"], sort=False).indices
    explanations, swaps, frequency, pareto_rows = [], [], [], []
    for (task, year, hs2), candidates in manifest.groupby(["task", "outcome_year", "hs2"], sort=True):
        candidates = candidates.sort_values("destination_iso3", kind="mergesort").reset_index(drop=True)
        ids = candidates.candidate_id.astype(str).tolist()
        qidx = scenario_indices.get((task, int(year), str(hs2).zfill(2)))
        q = scenarios.iloc[qidx] if qidx is not None else scenarios.iloc[0:0]
        o1, o2, scenario_ids = _pivot_scenarios(q, "primary", ids)
        c1, c2 = o1.mean(axis=0), o2.mean(axis=0)
        dominated_by = {}
        for j in range(len(ids)):
            mask = ((c1 >= c1[j] - 1e-12) & (c2 >= c2[j] - 1e-12)
                    & ((c1 > c1[j] + 1e-12) | (c2 > c2[j] + 1e-12)))
            dominated_by[j] = np.flatnonzero(mask)
        for j, dominators in dominated_by.items():
            if len(dominators):
                for k in [10, 5]:
                    pr = selection_groups.get((task, str(hs2).zfill(2), int(year), k, "PR-EMS"), selections.iloc[0:0])
                    pareto_rows.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year), "k": k,
                                        "candidate_id": ids[j], "destination_iso3": candidates.iloc[j].destination_iso3,
                                        "central_objective1": float(c1[j]), "central_objective2": float(c2[j]),
                                        "dominator_count": len(dominators),
                                        "dominator_iso3": "|".join(candidates.iloc[dominators].destination_iso3.astype(str).head(20)),
                                        "selected_by_pr_ems": candidates.iloc[j].destination_iso3 in set(pr.destination_iso3.astype(str)),
                                        "dominance_scope": "weakly better on both equal-scenario-mean objectives and strictly better on one"})
        util = np.vstack([w * o1 + (1-w) * o2 for w in vertices(PREFERENCE_SETS["W_primary"])])
        oracle = np.sort(util, axis=1)[:, ::-1]
        for k in [10, 5]:
            p = selection_groups.get((task, str(hs2).zfill(2), int(year), k, "PR-EMS"), selections.iloc[0:0])
            if p.empty:
                continue
            selected_ids = set(p.candidate_id.astype(str))
            selected_idx = np.array([i for i, x in enumerate(ids) if x in selected_ids], dtype=int)
            if len(selected_idx) != min(k, len(ids)):
                raise AssertionError("Robust portfolio is not cardinality-feasible")
            scenario_regret = oracle[:, :len(selected_idx)].sum(axis=1) - util[:, selected_idx].sum(axis=1)
            worst = float(scenario_regret.max())
            avg = float(util[:, selected_idx].sum(axis=1).mean())
            worst_utility = float(util[:, selected_idx].sum(axis=1).min())
            for r in p.itertuples(index=False):
                explanations.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year), "k": k,
                                     "candidate_id": r.candidate_id, "destination_iso3": r.destination_iso3,
                                     "selection_rank": r.selection_rank, "historical_cutoff": r.historical_information_cutoff,
                                     "mean_predicted_utility_w_primary_endpoints": avg,
                                     "worst_case_predicted_utility": worst_utility,
                                     "max_predicted_regret": worst,
                                     "supporting_scenario_count": int(np.sum(scenario_regret <= worst + 1e-9)),
                                     "scenario_count": len(scenario_ids),
                                     "preference_range": "W_primary=[0.25,0.75]",
                                     "interpretation": "conditional finite-scenario optimization diagnostic; not causal"})
            # Candidate replacement diagnostic: for each excluded destination, find its
            # best one-for-one replacement of a selected market by worst-case regret.
            for j, candidate in enumerate(candidates.itertuples(index=False)):
                if j in selected_idx:
                    continue
                i, new_regret = _best_one_swap(util, oracle, selected_idx, j, ids)
                reduction = worst - new_regret
                swaps.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year), "k": k,
                              "excluded_destination": candidate.destination_iso3,
                              "replace_selected_destination": candidates.iloc[i].destination_iso3,
                              "baseline_max_regret": worst, "swap_max_regret": new_regret,
                              "regret_reduction_from_swap": reduction,
                              "decision": "swap improves worst-case finite-scenario regret" if reduction > 1e-9 else "not selected; one-for-one swap does not reduce max regret"})
            # Scenario selection frequency for each individual scenario, recomputing
            # its K-best set and comparing it with the joint robust portfolio.
            for sidx, sid in enumerate(scenario_ids):
                score = .5 * (o1[sidx] + o2[sidx])
                order = np.lexsort((np.asarray(ids, dtype=str), -score))[:min(k, len(ids))]
                for j in order:
                    frequency.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year), "k": k,
                                      "scenario_id": sid, "candidate_id": ids[j], "selected_under_scenario": True,
                                      "selected_by_pr_ems": ids[j] in selected_ids,
                                      "objective1_scenario_value": float(o1[sidx, j]),
                                      "objective2_scenario_value": float(o2[sidx, j]),
                                      "equal_weight_scenario_utility": float(score[j])})
    e = pd.DataFrame(explanations)
    sw = pd.DataFrame(swaps)
    fr = pd.DataFrame(frequency)
    e.to_csv(HERE / "results/R6_portfolio_explanations.csv", index=False, na_rep="")
    sw.to_csv(HERE / "results/R6_candidate_swap_analysis.csv", index=False, na_rep="")
    fr.to_csv(HERE / "results/R6_scenario_selection_frequency.csv", index=False, na_rep="")
    pref = pd.read_csv(HERE / "results/R1_preference_sensitivity.csv", dtype={"hs2": str})
    sensitivity_rows = []
    if not pref.empty:
        for (task, hs2, year, k, wname, method), g in pref.groupby(
                ["task", "hs2", "outcome_year", "k", "preference_set", "policy"], sort=True):
            pref_set = set(g.destination_iso3.astype(str))
            pr = selection_groups.get((task, str(hs2).zfill(2), int(year), k, "PR-EMS"), selections.iloc[0:0])
            pr_set = set(pr.destination_iso3.astype(str))
            union = pref_set | pr_set
            sensitivity_rows.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year), "k": int(k),
                                    "preference_set": wname, "policy": method,
                                    "selected_set_size": len(pref_set), "pr_ems_reference_size": len(pr_set),
                                    "intersection": len(pref_set & pr_set), "union": len(union),
                                    "jaccard_vs_joint_pr_ems": len(pref_set & pr_set) / len(union) if union else np.nan,
                                    "symmetric_difference_count": len(pref_set ^ pr_set)})
    pd.DataFrame(sensitivity_rows).to_csv(HERE / "results/R6_preference_sensitivity.csv", index=False, na_rep="")
    pd.DataFrame(pareto_rows).to_csv(HERE / "results/R1_pareto_dominance.csv", index=False, na_rep="")
    r2p = pd.read_csv(HERE / "results/R2_decision_robustness.csv", dtype={"hs2": str})
    r2_overlap = []
    if not r2p.empty:
        for keys, g in r2p.groupby(["task", "hs2", "outcome_year", "k"], sort=True):
            sets = {n: set(z.candidate_id.astype(str)) for n, z in g.groupby("scenario_set")}
            for a_name, b_name in [("narrow", "primary"), ("primary", "stress"), ("narrow", "stress")]:
                if a_name not in sets or b_name not in sets:
                    continue
                union = sets[a_name] | sets[b_name]
                r2_overlap.append({"task": keys[0], "hs2": str(keys[1]).zfill(2), "outcome_year": int(keys[2]), "k": int(keys[3]),
                                   "scenario_set_a": a_name, "scenario_set_b": b_name,
                                   "intersection": len(sets[a_name] & sets[b_name]), "union": len(union),
                                   "jaccard": len(sets[a_name] & sets[b_name])/len(union) if union else np.nan})
    pd.DataFrame(r2_overlap).to_csv(HERE / "results/R2_predictive_overlap.csv", index=False, na_rep="")
    return {"portfolio_explanation_rows": len(e), "candidate_swap_rows": len(sw),
            "scenario_selection_frequency_rows": len(fr), "pareto_dominance_rows": len(pareto_rows),
            "predictive_overlap_rows": len(r2_overlap),
            "runtime_seconds": time.perf_counter() - start}
