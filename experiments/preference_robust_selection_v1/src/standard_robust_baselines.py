from __future__ import annotations

import numpy as np

from .preference_sets import PREFERENCE_GRID, vertices
from .robust_optimization import solve_maximin, solve_minimax_regret, solve_relative_robust, topk_set


def _scenario_objective(scenarios, objective: str, weight: float | None = None):
    if objective == "scalar":
        return weight * scenarios.objective1.to_numpy(float) + (1.0 - weight) * scenarios.objective2.to_numpy(float)
    raise ValueError(objective)


def robust_policy_sets(obj1_primary: np.ndarray, obj2_primary: np.ndarray,
                       obj1_central: np.ndarray, obj2_central: np.ndarray,
                       candidate_ids: list[str], k: int, interval: tuple[float, float]):
    """Return established decision rules and their solver diagnostics."""
    wv = vertices(interval)
    central1 = np.asarray(obj1_central, float)
    central2 = np.asarray(obj2_central, float)
    s1, n = np.asarray(obj1_primary, float).shape
    o1 = np.asarray(obj1_primary, float)
    o2 = np.asarray(obj2_primary, float)
    if o1.shape != o2.shape or n != len(candidate_ids):
        raise ValueError("Scenario matrices do not match candidates")
    k = min(int(k), n)
    result = {}
    # Point-weight baseline, equal weight by construction.
    point_scores = .5 * central1 + .5 * central2
    result["Equal_weight_MCDM"] = (topk_set(point_scores, candidate_ids, k), float(np.sort(point_scores)[::-1][:k].sum()), "direct weighted sum")
    # Borda-style aggregation of the two centrally predicted criteria.
    r1 = np.empty(n, float); r2 = np.empty(n, float)
    r1[np.lexsort((np.asarray(candidate_ids), -central1))] = np.arange(1, n + 1)
    r2[np.lexsort((np.asarray(candidate_ids), -central2))] = np.arange(1, n + 1)
    borda = -(r1 + r2)
    result["Borda_rank_aggregation"] = (topk_set(borda, candidate_ids, k), float(np.sort(borda)[-k:].sum()), "mean rank of two criteria")
    # Central-prediction preference robustness.
    pref_util = np.vstack([w * central1 + (1 - w) * central2 for w in wv])
    mm_pref = solve_maximin(pref_util, k)
    result["Maximin_preference"] = (mm_pref.selected, mm_pref.objective, "finite preference vertices; central prediction")
    mr_pref = solve_minimax_regret(pref_util, k)
    result["Standard_minimax_regret"] = (mr_pref.selected, mr_pref.objective, "preference ambiguity only; central prediction")
    rr = solve_relative_robust(central1, central2, wv, k)
    result["Relative_robust_Weber"] = (rr.selected, rr.objective, "Weber relative performance index; central prediction")
    # Prediction ambiguity only at the equal weight vector.
    pred_util = .5 * o1 + .5 * o2
    mm_pred = solve_maximin(pred_util, k)
    result["Predictive_maximin"] = (mm_pred.selected, mm_pred.objective, "predictive scenarios only; fixed 0.5 preference")
    # Joint finite model/predictive scenarios and preference extremes.
    joint = np.vstack([w * o1 + (1 - w) * o2 for w in wv])
    joint_mm = solve_maximin(joint, k)
    joint_mr = solve_minimax_regret(joint, k)
    result["Joint_maximin_utility"] = (joint_mm.selected, joint_mm.objective,
                                       "maximin utility over joint preference and predictive scenarios")
    result["PR-EMS"] = (joint_mr.selected, joint_mr.objective, "standard finite-scenario minimax regret on joint preference-prediction set")
    # Expected utility is a descriptive fixed-probability scenario benchmark. Scenarios do
    # not carry calibrated probabilities, so this is a uniform scenario average only.
    grid = PREFERENCE_GRID
    avg = np.mean(np.vstack([w * o1 + (1 - w) * o2 for w in grid]), axis=0)
    result["Uniform_scenario_expected_utility"] = (topk_set(avg, candidate_ids, k), float(np.sort(avg)[::-1][:k].sum()),
                                                    "uniform average over finite scenario and preference grid; not probabilistic")
    diagnostics = {
        "maximin_preference": mm_pref, "minimax_regret_preference": mr_pref,
        "relative_robust": rr, "maximin_predictive": mm_pred, "joint_minimax_regret": joint_mr,
        "joint_maximin": joint_mm,
    }
    return result, diagnostics


def joint_mr_for_scenario_set(obj1: np.ndarray, obj2: np.ndarray, k: int,
                              interval: tuple[float, float]):
    weights = vertices(interval)
    utility = np.vstack([w * obj1 + (1 - w) * obj2 for w in weights])
    return solve_minimax_regret(utility, k)
