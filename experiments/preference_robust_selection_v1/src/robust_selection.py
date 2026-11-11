from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from .preference_sets import PREFERENCE_GRID, PREFERENCE_SETS, vertices
from .robust_optimization import topk_set, solve_maximin, solve_minimax_regret, solve_relative_robust, verify_solver_small_instances
from .standard_robust_baselines import robust_policy_sets, joint_mr_for_scenario_set

HERE = Path(__file__).resolve().parents[1]
TASK_A = "Task_A_new_entry"
TASK_B = "Task_B_existing_underexploited"
SCENARIO_SETS = ("narrow", "primary", "stress")


def _pivot_scenarios(pool: pd.DataFrame, scenario_set: str, ids: list[str]):
    q = pool.loc[pool.scenario_set.eq(scenario_set)].copy()
    a = q.pivot(index="scenario_id", columns="candidate_id", values="objective1").reindex(columns=ids)
    b = q.pivot(index="scenario_id", columns="candidate_id", values="objective2").reindex(index=a.index, columns=ids)
    if a.empty or a.isna().any().any() or b.isna().any().any():
        raise AssertionError(f"Scenario set {scenario_set} does not cover every candidate in a pool")
    return a.to_numpy(float), b.to_numpy(float), list(a.index)


def _central(o1: np.ndarray, o2: np.ndarray):
    return np.mean(o1, axis=0), np.mean(o2, axis=0)


def _baseline_score_pivots(base: pd.DataFrame, task: str, year: int, hs2: str, ids: list[str]):
    q = base.loc[base.task.eq(task) & base.outcome_year.eq(year) & base.hs2.astype(str).str.zfill(2).eq(hs2)].copy()
    q = q.loc[(q.model.eq("LambdaMART") & q.seed.eq(-2)) | (~q.model.eq("LambdaMART") & q.seed.eq(-1))]
    piv = q.pivot(index="model", columns="candidate_id", values="score").reindex(columns=ids)
    if piv.empty or piv.isna().any().any():
        raise AssertionError(f"Baseline predictions not matched on {task}/{year}/HS2{hs2}")
    return piv


def _breakpoints(c1: np.ndarray, c2: np.ndarray, ids: list[str], k: int,
                 interval: tuple[float, float], task: str, hs2: str, year: int):
    # Every change in a top-K linear weighted sum can occur only at a pairwise
    # intersection of candidate score lines. Intersections are derived from training-only
    # normalized predicted objectives, not test outcomes.
    lo, hi = interval
    n = len(ids)
    vals = [lo, hi]
    for i in range(n):
        for j in range(i + 1, n):
            den = (c1[i] - c2[i]) - (c1[j] - c2[j])
            if abs(den) < 1e-14:
                continue
            w = (c2[j] - c2[i]) / den
            if lo < w < hi:
                vals.append(float(w))
    pts = sorted(set(round(v, 12) for v in vals))
    rows = []
    last = None
    for left, right in zip(pts[:-1], pts[1:]):
        w = (left + right) / 2
        selected = tuple(sorted(topk_set(w * c1 + (1 - w) * c2, ids, k)))
        if selected != last:
            rows.append({"task": task, "hs2": hs2, "outcome_year": year, "k": k,
                         "interval_left": left, "interval_right": right, "weight_midpoint": w,
                         "selected_count": len(selected), "selected_candidate_ids": "|".join(ids[j] for j in selected)})
            last = selected
    return rows


def run_optimization() -> dict:
    solver_validation = verify_solver_small_instances()
    pd.DataFrame(solver_validation).to_csv(HERE / "results/solver_validation.csv", index=False, na_rep="")
    scenarios = pd.read_parquet(HERE / "scenarios/predictive_utility_scenarios.parquet")
    manifest = pd.read_csv(HERE / "results/candidate_task_manifest.csv", dtype={"hs2": str})
    manifest["hs2"] = manifest.hs2.astype(str).str.zfill(2)
    base = pd.read_parquet(HERE / "scenarios/baseline_ranking_scores.parquet")
    base["hs2"] = base.hs2.astype(str).str.zfill(2)
    scenario_pools, selection_rows, optimization_rows = [], [], []
    pref_rows, breakpoint_rows, overlap_rows, r2_rows, r3_rows, freq_rows = [], [], [], [], [], []
    start = time.perf_counter()
    pool_keys = ["task", "outcome_year", "hs2"]
    for (task, year, hs2), d0 in manifest.groupby(pool_keys, sort=True):
        d = d0.sort_values("destination_iso3", kind="mergesort").reset_index(drop=True)
        ids = d.candidate_id.astype(str).tolist()
        if len(ids) < 5:
            continue
        sx = scenarios.loc[scenarios.task.eq(task) & scenarios.outcome_year.eq(year) & scenarios.hs2.astype(str).str.zfill(2).eq(str(hs2).zfill(2))]
        pivot = {s: _pivot_scenarios(sx, s, ids) for s in SCENARIO_SETS}
        o1, o2, scen_ids = pivot["primary"]
        c1, c2 = _central(o1, o2)
        bscore = _baseline_score_pivots(base, task, int(year), str(hs2).zfill(2), ids)
        for k in [10, 5]:
            kk = min(k, len(ids))
            try:
                central_policy, diagnostics = robust_policy_sets(o1, o2, c1, c2, ids, kk, PREFERENCE_SETS["W_primary"])
            except RuntimeError as exc:
                raise RuntimeError(f"Central robust solve failed for {task}, HS2={hs2}, year={year}, K={kk}: {exc}") from exc
            solver_meta = {
                "Maximin_preference": diagnostics["maximin_preference"],
                "Standard_minimax_regret": diagnostics["minimax_regret_preference"],
                "Relative_robust_Weber": diagnostics["relative_robust"],
                "Predictive_maximin": diagnostics["maximin_predictive"],
                "Joint_maximin_utility": diagnostics["joint_maximin"],
                "PR-EMS": diagnostics["joint_minimax_regret"],
            }
            method_specs = {}
            for method, (sel, obj, desc) in central_policy.items():
                if method in solver_meta:
                    solved = solver_meta[method]
                    method_specs[method] = (np.asarray(sel, dtype=int), float(obj), solved.status,
                                            solved.gap, solved.absolute_gap, desc)
                else:
                    method_specs[method] = (np.asarray(sel, dtype=int), float(obj),
                                            "deterministic_exact_sort", 0.0, 0.0, desc)
            # Forecast/ranking baseline portfolios use their saved task-matched scores.
            for model, score in bscore.iterrows():
                score_vec = score.to_numpy(float)
                sel = topk_set(score_vec, ids, kk)
                method_specs[model] = (sel, float(np.sort(score_vec)[::-1][:kk].sum()), "saved_prediction", 0.0, 0.0,
                                       "previously audited rolling-origin task score; higher is better")

            for method, (sel, objective, solver_status, gap, absolute_gap, desc) in method_specs.items():
                # A stable within-portfolio display order follows the mean equal-weight utility;
                # membership remains the optimizer's exact set.
                display = sorted(sel, key=lambda j: (-float(.5*c1[j] + .5*c2[j]), ids[j]))
                for rank, j in enumerate(display, 1):
                    selection_rows.append({
                        "task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year), "decision_budget_k": int(k),
                        "selection_method": method, "candidate_id": ids[j], "destination_iso3": d.iloc[j].destination_iso3,
                        "destination_name": d.iloc[j].destination_name, "selection_rank": rank,
                        "preference_specification": "W_primary" if method in {"Maximin_preference", "Standard_minimax_regret", "Relative_robust_Weber", "PR-EMS"} else "fixed/embedded policy",
                        "predictive_scenario_specification": "primary" if method in {"Predictive_maximin", "PR-EMS", "Uniform_scenario_expected_utility"} else "central or model-specific",
                        "historical_information_cutoff": int(d.iloc[j].historical_information_cutoff),
                        "eligible_ex_ante": bool(d.iloc[j].eligible_ex_ante),
                        "label_observed": bool(d.iloc[j].label_observed),
                        "policy_objective": objective, "solver_status": solver_status,
                        "relative_optimality_gap": gap, "absolute_optimality_gap": absolute_gap,
                        "max_constraint_violation": solver_meta[method].max_constraint_violation if method in solver_meta else 0.0,
                        "method_definition": desc,
                        "policy_score_at_0_5": float(.5*c1[j] + .5*c2[j]) if method in central_policy else np.nan,
                    })
                optimization_rows.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year),
                                          "k": int(k), "method": method, "objective": objective,
                                          "solver_status": solver_status, "relative_optimality_gap": gap,
                                          "absolute_optimality_gap": absolute_gap,
                                          "max_constraint_violation": solver_meta[method].max_constraint_violation if method in solver_meta else 0.0,
                                          "solver_runtime_seconds": solver_meta[method].runtime_seconds if method in solver_meta else 0.0,
                                          "candidate_count": len(ids),
                                          "scenario_count": solver_meta[method].scenarios if method in solver_meta else len(scen_ids),
                                          "description": desc})

            # Preference-set sensitivity for standard minimax regret and Weber relative robustness.
            pref_cache = {"W_primary": (diagnostics["minimax_regret_preference"], diagnostics["relative_robust"])}
            for wname, interval in PREFERENCE_SETS.items():
                if wname not in pref_cache:
                    wv = vertices(interval)
                    util = np.vstack([w*c1 + (1-w)*c2 for w in wv])
                    try:
                        pref_cache[wname] = (solve_minimax_regret(util, kk), solve_relative_robust(c1, c2, wv, kk))
                    except RuntimeError as exc:
                        raise RuntimeError(f"Solver failure for {task}, HS2={hs2}, year={year}, K={kk}, preference={wname}: {exc}") from exc
                mr, rr = pref_cache[wname]
                for name, sel, value in [("Standard_minimax_regret", mr.selected, mr.objective),
                                         ("Relative_robust_Weber", rr.selected, rr.objective)]:
                    for j in sel:
                        pref_rows.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year),
                                          "k": int(k), "preference_set": wname, "policy": name,
                                          "candidate_id": ids[j], "destination_iso3": d.iloc[j].destination_iso3,
                                          "objective_value": value,
                                          "objective_definition": "worst absolute regret" if name == "Standard_minimax_regret" else "worst relative utility ratio",
                                          "solver_status": mr.status if name == "Standard_minimax_regret" else rr.status,
                                          "relative_optimality_gap": mr.gap if name == "Standard_minimax_regret" else rr.gap,
                                          "absolute_optimality_gap": mr.absolute_gap if name == "Standard_minimax_regret" else rr.absolute_gap,
                                          "max_constraint_violation": mr.max_constraint_violation if name == "Standard_minimax_regret" else rr.max_constraint_violation})

            # Exact preference breakpoints for the central predicted objective tradeoff.
            breakpoint_rows.extend(_breakpoints(c1, c2, ids, kk, PREFERENCE_SETS["W_broad"], task, str(hs2).zfill(2), int(year)))
            for w in PREFERENCE_GRID:
                sel = topk_set(w*c1 + (1-w)*c2, ids, kk)
                freq_rows.extend({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year), "k": int(k),
                                  "weight_objective1": w, "candidate_id": ids[j], "destination_iso3": d.iloc[j].destination_iso3}
                                 for j in sel)

            # Preference-set top-K comparisons.
            selected_by_w = {}
            for wname in PREFERENCE_SETS:
                sol = pref_cache[wname][0]
                selected_by_w[wname] = set(sol.selected.tolist())
            for x, sxset in selected_by_w.items():
                for y, syset in selected_by_w.items():
                    if x >= y:
                        continue
                    denom = len(sxset | syset)
                    overlap_rows.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year), "k": int(k),
                                         "policy": "Standard_minimax_regret", "preference_set_a": x,
                                         "preference_set_b": y, "intersection": len(sxset & syset),
                                         "union": denom, "jaccard": len(sxset & syset)/denom if denom else np.nan})

            # Scenario-set and four-way uncertainty ablation. These are model choices,
            # not probability-weighted distributions.
            for scen_name in SCENARIO_SETS:
                if scen_name == "primary":
                    sol = diagnostics["joint_minimax_regret"]
                else:
                    aa, bb, _ = pivot[scen_name]
                    sol = joint_mr_for_scenario_set(aa, bb, kk, PREFERENCE_SETS["W_primary"])
                for j in sol.selected:
                    r2_rows.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year), "k": int(k),
                                    "scenario_set": scen_name, "candidate_id": ids[j], "destination_iso3": d.iloc[j].destination_iso3,
                                    "worst_case_regret_predicted": sol.objective, "solver_status": sol.status,
                                    "relative_optimality_gap": sol.gap, "absolute_optimality_gap": sol.absolute_gap,
                                    "max_constraint_violation": sol.max_constraint_violation,
                                    "scenario_count": sol.scenarios})
            # M0: no explicit uncertainty; M1: preference only; M2: predictive only; M3: both.
            ablation_policies = {
                "M0_point_equal_weight": (topk_set(.5*c1+.5*c2, ids, kk), "central predicted utility at w=.5"),
                "M1_preference_only": (diagnostics["minimax_regret_preference"].selected, "minimax regret over W_primary, central prediction"),
                "M2_predictive_only": (diagnostics["maximin_predictive"].selected, "maximin over primary scenarios at w=.5"),
                "M3_joint_preference_prediction": (diagnostics["joint_minimax_regret"].selected, "minimax regret over primary scenarios and W_primary"),
            }
            for name, (sel, desc) in ablation_policies.items():
                r3_rows.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year), "k": int(k),
                                "component_specification": name, "selected_candidate_ids": "|".join(ids[j] for j in sorted(sel)),
                                "policy_definition": desc, "predicted_worst_regret": diagnostics["joint_minimax_regret"].objective
                                if name == "M3_joint_preference_prediction" else np.nan})

    selections = pd.DataFrame(selection_rows)
    selections.to_parquet(HERE / "policies/selected_portfolios.parquet", index=False)
    selections.to_csv(HERE / "results/selected_portfolios.csv", index=False, na_rep="")
    pd.DataFrame(optimization_rows).to_csv(HERE / "results/optimization_diagnostics.csv", index=False, na_rep="")
    pd.DataFrame(pref_rows).to_csv(HERE / "results/R1_preference_sensitivity.csv", index=False, na_rep="")
    pd.DataFrame(breakpoint_rows).to_csv(HERE / "results/R1_preference_breakpoints.csv", index=False, na_rep="")
    pd.DataFrame(overlap_rows).to_csv(HERE / "results/R1_topk_overlap.csv", index=False, na_rep="")
    pd.DataFrame(freq_rows).to_csv(HERE / "results/R1_selection_frequency.csv", index=False, na_rep="")
    r2_df = pd.DataFrame(r2_rows)
    r2_df.to_csv(HERE / "results/R2_decision_robustness.csv", index=False, na_rep="")
    overlap_pairs = []
    for keys, g in r2_df.groupby(["task", "hs2", "outcome_year", "k"], sort=True):
        sets = {name: set(z.candidate_id.astype(str)) for name, z in g.groupby("scenario_set")}
        for a_name, b_name in [("narrow", "primary"), ("primary", "stress"), ("narrow", "stress")]:
            if a_name not in sets or b_name not in sets:
                continue
            union = sets[a_name] | sets[b_name]
            overlap_pairs.append({"task": keys[0], "hs2": str(keys[1]).zfill(2), "outcome_year": int(keys[2]), "k": int(keys[3]),
                                  "scenario_set_a": a_name, "scenario_set_b": b_name,
                                  "intersection": len(sets[a_name] & sets[b_name]), "union": len(union),
                                  "jaccard": len(sets[a_name] & sets[b_name])/len(union) if union else np.nan,
                                  "symmetric_difference_count": len(sets[a_name] ^ sets[b_name])})
    pd.DataFrame(overlap_pairs).to_csv(HERE / "results/R2_predictive_overlap.csv", index=False, na_rep="")
    pd.DataFrame(r3_rows).to_csv(HERE / "results/R3_joint_uncertainty_ablation.csv", index=False, na_rep="")
    return {"portfolio_rows": len(selections), "policy_pool_rows": len(optimization_rows),
            "preference_rows": len(pref_rows), "breakpoints": len(breakpoint_rows),
            "r2_portfolio_rows": len(r2_rows), "r2_overlap_rows": len(overlap_pairs), "r3_pool_rows": len(r3_rows),
            "runtime_seconds": time.perf_counter() - start}


def append_joint_maximin_policy() -> dict:
    """Append the joint preference×prediction maximin comparator to completed outputs."""
    start = time.perf_counter()
    manifest = pd.read_csv(HERE / "results/candidate_task_manifest.csv", dtype={"hs2": str})
    manifest["hs2"] = manifest.hs2.astype(str).str.zfill(2)
    scenarios = pd.read_parquet(HERE / "scenarios/predictive_utility_scenarios.parquet")
    scenarios["hs2"] = scenarios.hs2.astype(str).str.zfill(2)
    selections = pd.read_parquet(HERE / "policies/selected_portfolios.parquet")
    selections["hs2"] = selections.hs2.astype(str).str.zfill(2)
    selections = selections.loc[~selections.selection_method.eq("Joint_maximin_utility")].copy()
    diagnostics = []
    new_rows = []
    for (task, year, hs2), d0 in manifest.groupby(["task", "outcome_year", "hs2"], sort=True):
        d = d0.sort_values("destination_iso3", kind="mergesort").reset_index(drop=True)
        ids = d.candidate_id.astype(str).tolist()
        pool = scenarios.loc[scenarios.task.eq(task) & scenarios.outcome_year.eq(year)
                             & scenarios.hs2.eq(str(hs2).zfill(2)) & scenarios.scenario_set.eq("primary")]
        a, b, sid = _pivot_scenarios(pool, "primary", ids)
        util = np.vstack([w*a + (1-w)*b for w in vertices(PREFERENCE_SETS["W_primary"])])
        central1, central2 = a.mean(axis=0), b.mean(axis=0)
        for k in [10, 5]:
            kk = min(k, len(ids))
            sol = solve_maximin(util, kk)
            display = sorted(sol.selected, key=lambda j: (-float(.5*central1[j] + .5*central2[j]), ids[j]))
            for rank, j in enumerate(display, 1):
                new_rows.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year),
                                 "decision_budget_k": int(k), "selection_method": "Joint_maximin_utility",
                                 "candidate_id": ids[j], "destination_iso3": d.iloc[j].destination_iso3,
                                 "destination_name": d.iloc[j].destination_name, "selection_rank": rank,
                                 "preference_specification": "W_primary",
                                 "predictive_scenario_specification": "primary",
                                 "historical_information_cutoff": int(d.iloc[j].historical_information_cutoff),
                                 "eligible_ex_ante": bool(d.iloc[j].eligible_ex_ante),
                                 "label_observed": bool(d.iloc[j].label_observed),
                                 "policy_objective": sol.objective, "solver_status": sol.status,
                                 "relative_optimality_gap": sol.gap, "absolute_optimality_gap": sol.absolute_gap,
                                 "max_constraint_violation": sol.max_constraint_violation,
                                 "method_definition": "maximin utility over joint preference and predictive scenarios",
                                 "policy_score_at_0_5": float(.5*central1[j] + .5*central2[j])})
            diagnostics.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year), "k": int(k),
                                "method": "Joint_maximin_utility", "objective": sol.objective,
                                "solver_status": sol.status, "relative_optimality_gap": sol.gap,
                                "absolute_optimality_gap": sol.absolute_gap,
                                "max_constraint_violation": sol.max_constraint_violation,
                                "solver_runtime_seconds": sol.runtime_seconds, "candidate_count": len(ids),
                                "scenario_count": sol.scenarios, "description": "maximin utility over joint preference and predictive scenarios"})
    all_selections = pd.concat([selections, pd.DataFrame(new_rows)], ignore_index=True)
    all_selections.to_parquet(HERE / "policies/selected_portfolios.parquet", index=False)
    all_selections.to_csv(HERE / "results/selected_portfolios.csv", index=False, na_rep="")
    diag_path = HERE / "results/optimization_diagnostics.csv"
    old_diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    old_diag = old_diag.loc[~old_diag.method.eq("Joint_maximin_utility")] if not old_diag.empty else old_diag
    pd.concat([old_diag, pd.DataFrame(diagnostics)], ignore_index=True).to_csv(diag_path, index=False, na_rep="")
    return {"added_portfolio_rows": len(new_rows), "pool_diagnostics": len(diagnostics),
            "runtime_seconds": time.perf_counter() - start}


def append_linear_amount_utility_sensitivity() -> dict:
    """Re-solve the joint regret policy under capped-linear instead of capped-log scale utility."""
    start = time.perf_counter()
    manifest = pd.read_csv(HERE / "results/candidate_task_manifest.csv", dtype={"hs2": str})
    manifest["hs2"] = manifest.hs2.astype(str).str.zfill(2)
    scenarios = pd.read_parquet(HERE / "scenarios/predictive_utility_scenarios.parquet")
    scenarios["hs2"] = scenarios.hs2.astype(str).str.zfill(2)
    caps_df = pd.read_csv(HERE / "results/R2_utility_scales.csv")
    cap_map = {}
    for r in caps_df.itertuples(index=False):
        task = TASK_A if r.model == "conditional_size_historical_quantiles" else TASK_B
        cap_map[(task, int(r.evaluation_year))] = max(float(r.scale_cap_usd_q95_train), 50_001.0)
    selections = pd.read_parquet(HERE / "policies/selected_portfolios.parquet")
    selections["hs2"] = selections.hs2.astype(str).str.zfill(2)
    selections = selections.loc[~selections.selection_method.eq("PR-EMS_linear_amount_utility")].copy()
    rows, diagnostics, sensitivity = [], [], []
    for (task, year, hs2), d0 in manifest.groupby(["task", "outcome_year", "hs2"], sort=True):
        d = d0.sort_values("destination_iso3", kind="mergesort").reset_index(drop=True)
        ids = d.candidate_id.astype(str).tolist()
        pool = scenarios.loc[scenarios.task.eq(task) & scenarios.outcome_year.eq(year)
                             & scenarios.hs2.eq(str(hs2).zfill(2)) & scenarios.scenario_set.eq("primary")]
        a, b_log, _ = _pivot_scenarios(pool, "primary", ids)
        cap = cap_map[(task, int(year))]
        threshold = 50_000.0
        raw_capped = threshold * np.expm1(np.clip(b_log, 0, 1) * np.log1p(cap / threshold))
        b_linear = np.clip(raw_capped / cap, 0, 1)
        utility = np.vstack([w*a + (1-w)*b_linear for w in vertices(PREFERENCE_SETS["W_primary"])])
        original = selections.loc[selections.task.eq(task) & selections.hs2.eq(str(hs2).zfill(2))
                                  & selections.outcome_year.eq(year) & selections.selection_method.eq("PR-EMS")]
        primary_sets = {int(k): set(g.candidate_id.astype(str)) for k, g in original.groupby("decision_budget_k")}
        central_score = .5*a.mean(axis=0) + .5*b_linear.mean(axis=0)
        for k in [10, 5]:
            kk = min(k, len(ids))
            sol = solve_minimax_regret(utility, kk)
            chosen_ids = {ids[j] for j in sol.selected}
            primary_ids = primary_sets.get(k, set())
            union = chosen_ids | primary_ids
            sensitivity.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year), "k": int(k),
                                "primary_utility": "capped log1p amount scale",
                                "alternative_utility": "capped linear amount scale",
                                "primary_log_scale_jaccard": len(chosen_ids & primary_ids)/len(union) if union else np.nan,
                                "intersection": len(chosen_ids & primary_ids), "union": len(union),
                                "symmetric_difference": len(chosen_ids ^ primary_ids),
                                "solver_status": sol.status, "relative_optimality_gap": sol.gap,
                                "absolute_optimality_gap": sol.absolute_gap,
                                "max_constraint_violation": sol.max_constraint_violation})
            ordered = sorted(sol.selected, key=lambda j: (-float(central_score[j]), ids[j]))
            for rank, j in enumerate(ordered, 1):
                rows.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year),
                             "decision_budget_k": int(k), "selection_method": "PR-EMS_linear_amount_utility",
                             "candidate_id": ids[j], "destination_iso3": d.iloc[j].destination_iso3,
                             "destination_name": d.iloc[j].destination_name, "selection_rank": rank,
                             "preference_specification": "W_primary", "predictive_scenario_specification": "primary",
                             "historical_information_cutoff": int(d.iloc[j].historical_information_cutoff),
                             "eligible_ex_ante": bool(d.iloc[j].eligible_ex_ante),
                             "label_observed": bool(d.iloc[j].label_observed), "policy_objective": sol.objective,
                             "solver_status": sol.status, "relative_optimality_gap": sol.gap,
                             "absolute_optimality_gap": sol.absolute_gap,
                             "max_constraint_violation": sol.max_constraint_violation,
                             "method_definition": "joint finite-scenario minimax regret with capped-linear amount utility",
                             "policy_score_at_0_5": float(central_score[j])})
            diagnostics.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year), "k": int(k),
                                "method": "PR-EMS_linear_amount_utility", "objective": sol.objective,
                                "solver_status": sol.status, "relative_optimality_gap": sol.gap,
                                "absolute_optimality_gap": sol.absolute_gap,
                                "max_constraint_violation": sol.max_constraint_violation,
                                "solver_runtime_seconds": sol.runtime_seconds, "candidate_count": len(ids),
                                "scenario_count": sol.scenarios,
                                "description": "sensitivity to capped-linear instead of capped-log amount utility"})
    all_selection = pd.concat([selections, pd.DataFrame(rows)], ignore_index=True)
    all_selection.to_parquet(HERE / "policies/selected_portfolios.parquet", index=False)
    all_selection.to_csv(HERE / "results/selected_portfolios.csv", index=False, na_rep="")
    diag_path = HERE / "results/optimization_diagnostics.csv"
    old_diag = pd.read_csv(diag_path)
    old_diag = old_diag.loc[~old_diag.method.eq("PR-EMS_linear_amount_utility")]
    pd.concat([old_diag, pd.DataFrame(diagnostics)], ignore_index=True).to_csv(diag_path, index=False, na_rep="")
    pd.DataFrame(sensitivity).to_csv(HERE / "results/R1_utility_transform_sensitivity.csv", index=False, na_rep="")
    return {"portfolio_rows": len(rows), "pool_comparisons": len(sensitivity),
            "runtime_seconds": time.perf_counter() - start}
