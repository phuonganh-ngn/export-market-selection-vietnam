from __future__ import annotations

from pathlib import Path
import itertools
import time

import numpy as np
import pandas as pd

from .preference_sets import PREFERENCE_SETS

HERE = Path(__file__).resolve().parents[1]
TASK_A = "Task_A_new_entry"
TASK_B = "Task_B_existing_underexploited"
KEY = ["task", "hs2", "outcome_year"]


def _scale(value, cap, threshold=50_000.0):
    x = np.maximum(np.asarray(value, dtype=float), 0.0)
    cap = max(float(cap), float(threshold) + 1.0)
    return np.clip(np.log1p(np.minimum(x, cap) / threshold) / np.log1p(cap / threshold), 0, 1)


def _caps() -> dict[tuple[str, int], float]:
    d = pd.read_csv(HERE / "results/R2_utility_scales.csv")
    out = {}
    for r in d.itertuples(index=False):
        if getattr(r, "model", "") == "conditional_size_historical_quantiles":
            out[(TASK_A, int(r.evaluation_year))] = float(r.scale_cap_usd_q95_train)
        elif getattr(r, "model", "") == "saved_increment_model_scenarios":
            out[(TASK_B, int(r.evaluation_year))] = float(r.scale_cap_usd_q95_train)
    return out


def _actual_outcomes(task: str, d: pd.DataFrame, cap: float):
    observed = d.label_observed.astype(bool).to_numpy()
    if task == TASK_A:
        entry = d.entry_success.to_numpy(float)
        export = d.export_value_usd.to_numpy(float)
        scale = _scale(export, cap)
        # The size criterion is the realized scale only when the entry threshold is met.
        scale = np.where(np.isfinite(entry) & (entry == 1.0), scale,
                         np.where(np.isfinite(entry), 0.0, np.nan))
        return entry, scale, observed
    inc = d.realized_incremental_export_usd.to_numpy(float)
    inc = np.where(np.isfinite(inc), np.maximum(inc, 0.0), np.nan)
    positive = (inc > 0).astype(float)
    positive[~np.isfinite(inc)] = np.nan
    return positive, _scale(inc, cap), observed


def _pool_policy_metrics(selected: pd.DataFrame, candidates: pd.DataFrame, cap: float):
    task = str(candidates.task.iloc[0])
    k = int(selected.decision_budget_k.iloc[0])
    candidates = candidates.copy().sort_values("destination_iso3", kind="mergesort").reset_index(drop=True)
    selected_ids = set(selected.candidate_id.astype(str))
    selected_mask = candidates.candidate_id.astype(str).isin(selected_ids).to_numpy()
    u1, u2, observed = _actual_outcomes(task, candidates, cap)
    unknown_sel = int((selected_mask & ~observed).sum())
    sel_known = selected_mask & observed
    k_obs = int(sel_known.sum())
    if task == TASK_A:
        success = candidates.entry_success.to_numpy(float)
        exports = candidates.export_value_usd.to_numpy(float)
        selected_success_count = int(np.sum(sel_known & (success == 1.0)))
        selected_trade_usd = float(np.nansum(exports[sel_known & (success == 1.0)])) if selected_success_count else 0.0
        oracle_entry_count = int(np.sort(np.nan_to_num(success[observed], nan=0.0))[::-1][:min(k_obs, int(observed.sum()))].sum()) if k_obs else np.nan
        oracle_entry_exports = float(np.sort(np.where(success[observed] == 1.0, exports[observed], 0.0))[::-1][:min(k_obs, int(observed.sum()))].sum()) if k_obs else np.nan
        task_metrics = {"selected_successful_entries": selected_success_count,
                        "selected_success_rate_observed_slots": selected_success_count / k_obs if k_obs else np.nan,
                        "selected_entry_export_value_usd": selected_trade_usd,
                        "hindsight_topk_entry_count": oracle_entry_count,
                        "hindsight_topk_entry_export_value_usd": oracle_entry_exports}
    else:
        inc = candidates.realized_incremental_export_usd.to_numpy(float)
        selected_positive = sel_known & np.isfinite(inc) & (inc > 0)
        selected_positive_count = int(selected_positive.sum())
        selected_increment_usd = float(inc[selected_positive].sum()) if selected_positive_count else 0.0
        known_inc = inc[observed]
        oracle_increment = float(np.sort(np.maximum(known_inc, 0.0))[::-1][:min(k_obs, int(observed.sum()))].sum()) if k_obs else np.nan
        task_metrics = {"selected_positive_expansion_count": selected_positive_count,
                        "selected_positive_expansion_rate_observed_slots": selected_positive_count / k_obs if k_obs else np.nan,
                        "selected_positive_increment_usd": selected_increment_usd,
                        "hindsight_topk_positive_increment_usd": oracle_increment}
    # Unknown labels remain unobserved. Observed outcome utility is calculated only over
    # the selected candidates whose outcomes are available; no value is assigned to missing rows.
    endpoints = PREFERENCE_SETS["W_primary"]
    regrets, oracle_utils, selected_utils = [], [], []
    for w in sorted(set(endpoints)):
        utility = w * u1 + (1 - w) * u2
        known_utility = utility[observed]
        oracle_k = min(k_obs, int(observed.sum()))
        oracle = float(np.sort(known_utility)[::-1][:oracle_k].sum()) if oracle_k else np.nan
        chosen = float(utility[sel_known].sum()) if k_obs else np.nan
        selected_utils.append(chosen)
        oracle_utils.append(oracle)
        regrets.append(max(0.0, oracle - chosen) if np.isfinite(oracle) and np.isfinite(chosen) else np.nan)
    imputed_n = 0
    result = {
        "observed_selected_count": k_obs, "missing_selected_count": unknown_sel,
        "candidate_pool_size": int(len(candidates)),
        "available_selection_slots": int(min(k, len(candidates))),
        "selected_observed_utility_w025": selected_utils[0], "selected_observed_utility_w075": selected_utils[-1],
        "selected_observed_utility_w050": float(.5 * (selected_utils[0] + selected_utils[-1])) if k_obs else np.nan,
        "oracle_observed_utility_w025": oracle_utils[0], "oracle_observed_utility_w075": oracle_utils[-1],
        "oracle_observed_utility_w050": float(.5 * (oracle_utils[0] + oracle_utils[-1])) if k_obs else np.nan,
        "observed_case_worst_preference_regret": float(np.nanmax(regrets)) if np.isfinite(regrets).any() else np.nan,
        "observed_case_regret_at_w050": float(.5 * (regrets[0] + regrets[-1])) if np.isfinite(regrets).all() else np.nan,
        "evaluation_slots": k_obs, "requested_slots": k, "is_full_k_observed": k_obs == k,
        "all_available_slots_observed": k_obs == min(k, len(candidates)),
        "outcome_imputation_count": imputed_n,
        **task_metrics,
    }
    if k_obs == 0:
        result["selected_observed_utility_per_selected"] = np.nan
    else:
        result["selected_observed_utility_per_selected"] = result["selected_observed_utility_w050"] / k_obs
    return result


def _r4_pool_bounds(task: str, policy: str, selected: pd.DataFrame, candidates: pd.DataFrame, cap: float):
    d = candidates.copy().sort_values("destination_iso3", kind="mergesort").reset_index(drop=True)
    task = str(task); k = min(int(selected.decision_budget_k.iloc[0]), len(d))
    sids = set(selected.candidate_id.astype(str))
    smask = d.candidate_id.astype(str).isin(sids).to_numpy()
    u1, u2, observed = _actual_outcomes(task, d, cap)
    missing = ~observed
    rows = []
    for w in sorted(set(PREFERENCE_SETS["W_primary"])):
        util = w * u1 + (1-w) * u2
        sel_known = float(util[smask & observed].sum())
        sel_unknown = int((smask & missing).sum())
        utility_lo, utility_hi = sel_known, sel_known + sel_unknown
        # For a fixed portfolio, regret is max_T [U(T)-U(S)]. Maximizing over
        # bounded missing utilities and T can be interchanged; unknown selected items
        # cancel if T includes them, while unknown nonselected items can contribute at most 1.
        weights = np.zeros(len(d), dtype=float)
        weights[observed] = util[observed]
        weights[missing & ~smask] = 1.0
        oracle_minus_selected_upper = float(np.sort(weights)[::-1][:k].sum() - sel_known)
        regret_upper = max(0.0, oracle_minus_selected_upper)
        rows.append({"task": task, "hs2": str(d.hs2.iloc[0]).zfill(2), "outcome_year": int(d.outcome_year.iloc[0]),
                     "k": int(selected.decision_budget_k.iloc[0]), "policy": policy, "weight_objective1": w,
                     "candidate_count": len(d), "observed_candidates": int(observed.sum()),
                     "missing_candidates": int(missing.sum()), "missing_selected": sel_unknown,
                     "utility_lower": utility_lo, "utility_upper": utility_hi,
                     "worst_preference_regret_lower": 0.0,
                     "worst_preference_regret_upper": regret_upper,
                     "regret_upper_status": "exact upper bound over utility support [0,1] and same completion; lower bound 0 is conservative",
                     "missing_candidates_that_may_enter_oracle": int(missing.sum()), "outcome_imputation": False})
    return rows


def _exact_paired_utility_bounds(pr: pd.DataFrame, other: pd.DataFrame, candidates: pd.DataFrame, cap: float):
    d = candidates.sort_values("destination_iso3", kind="mergesort").reset_index(drop=True)
    u1, u2, observed = _actual_outcomes(str(d.task.iloc[0]), d, cap)
    pr_ids = set(pr.candidate_id.astype(str)); ot_ids = set(other.candidate_id.astype(str))
    a = d.candidate_id.astype(str).isin(pr_ids).to_numpy().astype(float)
    b = d.candidate_id.astype(str).isin(ot_ids).to_numpy().astype(float)
    coeff = a - b
    missing = ~observed
    rows = []
    for w in sorted(set(PREFERENCE_SETS["W_primary"])):
        util = w*u1+(1-w)*u2
        known = float(np.dot(coeff[observed], util[observed]))
        c = coeff[missing]
        # The same missing outcome is used for both policies. The difference is linear,
        # so each missing candidate's support endpoints give sharp bounds.
        lo = known + float(np.minimum(c, 0).sum())
        hi = known + float(np.maximum(c, 0).sum())
        rows.append({"task": str(d.task.iloc[0]), "hs2": str(d.hs2.iloc[0]).zfill(2),
                     "outcome_year": int(d.outcome_year.iloc[0]), "k": int(pr.decision_budget_k.iloc[0]),
                     "policy_a": "PR-EMS", "policy_b": str(other.selection_method.iloc[0]),
                     "weight_objective1": w, "difference_a_minus_b_lower": lo,
                     "difference_a_minus_b_upper": hi, "missing_count": int(missing.sum()),
                     "missing_selected_a": int((a[missing] > 0).sum()), "missing_selected_b": int((b[missing] > 0).sum()),
                     "bounds": "sharp for the rectangular [0,1] utility-support relaxation under common completions; conservative for entry's joint binary/scale support"})
    return rows


def _aggregate(table: pd.DataFrame, group_cols, metric_cols):
    return table.groupby(group_cols, as_index=False)[metric_cols].mean()


def evaluate_portfolios() -> dict:
    start = time.perf_counter()
    selected = pd.read_parquet(HERE / "policies/selected_portfolios.parquet")
    selected["hs2"] = selected.hs2.astype(str).str.zfill(2)
    manifest = pd.read_csv(HERE / "results/candidate_task_manifest.csv", dtype={"hs2": str})
    manifest["hs2"] = manifest.hs2.astype(str).str.zfill(2)
    caps = _caps()
    realized, regrets, bounds, paired_bounds = [], [], [], []
    preference_sensitivity = []
    pool_groups = manifest.groupby(KEY, sort=True)
    candidate_dict = {key: g.copy() for key, g in pool_groups}
    for key, pgroup in selected.groupby(["task", "hs2", "outcome_year", "decision_budget_k", "selection_method"], sort=False):
        task, hs2, year, k, method = key
        poolkey = (task, str(hs2).zfill(2), int(year))
        cand = candidate_dict.get(poolkey)
        if cand is None:
            raise AssertionError(f"Missing evaluation candidate pool {poolkey}")
        cap = caps[(task, int(year))]
        metrics = _pool_policy_metrics(pgroup, cand, cap)
        selected_join = pgroup.merge(cand, on=["task", "hs2", "outcome_year", "candidate_id", "destination_iso3"],
                                     how="left", suffixes=("_policy", ""), validate="one_to_one")
        u1, u2, obs = _actual_outcomes(task, selected_join, cap)
        pred_sum = (float(selected_join.policy_score_at_0_5.sum())
                    if selected_join.policy_score_at_0_5.notna().all() else np.nan)
        realized.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year), "k": int(k),
                         "policy": method, "selected_count": len(pgroup), "predicted_central_utility_sum": pred_sum,
                         **metrics})
        regret_rows = []
        for w in [.25, .50, .75]:
            util = w*u1+(1-w)*u2
            known = obs.astype(bool)
            nsel = int(known.sum())
            chosen = float(util[known].sum()) if nsel else np.nan
            allcand = cand.sort_values("destination_iso3", kind="mergesort").reset_index(drop=True)
            aa1, aa2, aaobs = _actual_outcomes(task, allcand, cap)
            v = w*aa1+(1-w)*aa2
            oracle = float(np.sort(v[aaobs])[::-1][:nsel].sum()) if nsel else np.nan
            rg = max(0.0, oracle-chosen) if nsel else np.nan
            regret_rows.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year), "k": int(k),
                               "policy": method, "weight_objective1": w, "selected_observed_count": nsel,
                               "selected_utility": chosen, "hindsight_oracle_utility_same_k": oracle,
                               "realized_outcome_regret": rg, "normalized_regret": rg/oracle if np.isfinite(rg) and oracle > 0 else np.nan,
                               "complete_case_policy_slots": nsel == int(k), "missing_labels_replaced": False})
        r = pd.DataFrame(regret_rows)
        wmax = r.realized_outcome_regret.max(skipna=True)
        regrets.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year), "k": int(k),
                        "policy": method, "mean_regret_primary_preferences": float(r.realized_outcome_regret.mean(skipna=True)),
                        "worst_preference_regret_primary": float(wmax) if pd.notna(wmax) else np.nan,
                        "mean_oracle_utility_primary": float(r.hindsight_oracle_utility_same_k.mean(skipna=True)),
                        "selected_unknown_count": metrics["missing_selected_count"],
                        "fully_observed_topk": metrics["is_full_k_observed"]})
        bounds.extend(_r4_pool_bounds(task, method, pgroup, cand, cap))
        # Evaluated preference sensitivity uses the same observed outcomes and the
        # same known selected alternatives for every weight; unknowns remain NA.
        ordered = cand.sort_values("destination_iso3", kind="mergesort").reset_index(drop=True)
        selected_ids = set(pgroup.candidate_id.astype(str))
        smask = ordered.candidate_id.astype(str).isin(selected_ids).to_numpy()
        u1, u2, obs = _actual_outcomes(task, ordered, cap)
        n_selected_observed = int((smask & obs).sum())
        for w in [0.0, .25, .40, .50, .60, .75, 1.0]:
            utility = w*u1 + (1-w)*u2
            chosen = float(utility[smask & obs].sum()) if n_selected_observed else np.nan
            oracle = float(np.sort(utility[obs])[::-1][:n_selected_observed].sum()) if n_selected_observed else np.nan
            preference_sensitivity.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year),
                                           "k": int(k), "policy": method, "weight_objective1": w,
                                           "selected_observed_utility": chosen, "oracle_same_observed_slots": oracle,
                                           "complete_case_regret": max(0.0, oracle-chosen) if n_selected_observed else np.nan,
                                           "selected_observed_slots": n_selected_observed,
                                           "missing_selected_slots": int((smask & ~obs).sum()),
                                           "full_k_observed": n_selected_observed == int(k)})

    realized_df, regret_df = pd.DataFrame(realized), pd.DataFrame(regrets)
    realized_df.to_csv(HERE / "results/R5_realized_utility.csv", index=False, na_rep="")
    regret_df.to_csv(HERE / "results/R5_regret_at_5_10.csv", index=False, na_rep="")
    bounds_df = pd.DataFrame(bounds)
    bounds_df.to_csv(HERE / "results/R4_missing_outcome_bounds.csv", index=False, na_rep="")

    # Paired sharp utility bounds for PR-EMS against the benchmark portfolios. Missing labels
    # are assigned jointly across both portfolios, never independently per policy/oracle.
    lookup = {(k[0], k[1], int(k[2]), int(k[3]), k[4]): g for k, g in selected.groupby(
        ["task", "hs2", "outcome_year", "decision_budget_k", "selection_method"], sort=False)}
    for (task, hs2, year, k), g in selected.groupby(["task", "hs2", "outcome_year", "decision_budget_k"], sort=False):
        pr = lookup.get((task, hs2, int(year), int(k), "PR-EMS"))
        if pr is None:
            continue
        cand = candidate_dict[(task, str(hs2).zfill(2), int(year))]
        for cmp in ["Standard_minimax_regret", "Relative_robust_Weber", "Gravity_PPML", "LightGBM", "LambdaMART", "Equal_weight_MCDM"]:
            other = lookup.get((task, hs2, int(year), int(k), cmp))
            if other is None:
                continue
            paired_bounds.extend(_exact_paired_utility_bounds(pr, other, cand, caps[(task, int(year))]))
    paired_bounds_df = pd.DataFrame(paired_bounds)
    paired_bounds_df.to_csv(HERE / "results/R4_partial_identification.csv", index=False, na_rep="")
    pd.DataFrame(preference_sensitivity).to_csv(HERE / "results/R1_realized_preference_sensitivity.csv", index=False, na_rep="")

    # Evaluate policy sets under narrow / primary / stress predictive ambiguity.
    r2_eval = []
    r2p = pd.read_csv(HERE / "results/R2_decision_robustness.csv", dtype={"hs2": str})
    for key, g in r2p.groupby(["task", "hs2", "outcome_year", "k", "scenario_set"], sort=False):
        task, hs2, year, k, set_name = key
        cand = candidate_dict[(task, str(hs2).zfill(2), int(year))]
        fake = g.rename(columns={"k": "decision_budget_k"}).copy()
        metrics = _pool_policy_metrics(fake, cand, caps[(task, int(year))])
        r2_eval.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year), "k": int(k),
                        "predictive_scenario_set": set_name, **metrics})
    r2_eval_df = pd.DataFrame(r2_eval)
    r2_eval_df.to_csv(HERE / "results/R2_scenario_policy_evaluation.csv", index=False, na_rep="")

    # Evaluate the M0--M3 uncertainty-component ablation on identical outcomes.
    r3_eval = []
    r3p = pd.read_csv(HERE / "results/R3_joint_uncertainty_ablation.csv", dtype={"hs2": str})
    for r in r3p.itertuples(index=False):
        pool = candidate_dict[(r.task, str(r.hs2).zfill(2), int(r.outcome_year))]
        chosen = set(str(r.selected_candidate_ids).split("|")) if pd.notna(r.selected_candidate_ids) and str(r.selected_candidate_ids) else set()
        fake = pool.loc[pool.candidate_id.astype(str).isin(chosen)].copy()
        fake["decision_budget_k"] = int(r.k)
        fake["selection_method"] = r.component_specification
        metrics = _pool_policy_metrics(fake, pool, caps[(r.task, int(r.outcome_year))])
        r3_eval.append({"task": r.task, "hs2": str(r.hs2).zfill(2), "outcome_year": int(r.outcome_year),
                        "k": int(r.k), "component_specification": r.component_specification, **metrics})
    r3_eval_df = pd.DataFrame(r3_eval)
    r3_eval_df.to_csv(HERE / "results/R3_joint_uncertainty_evaluation.csv", index=False, na_rep="")
    r3_summary = r3_eval_df.groupby(["task", "component_specification", "k"], as_index=False).agg(
        mean_utility_w050=("selected_observed_utility_w050", "mean"),
        mean_utility_per_selected=("selected_observed_utility_per_selected", "mean"),
        mean_worst_preference_regret=("observed_case_worst_preference_regret", "mean"),
        mean_missing_selected=("missing_selected_count", "mean"),
        share_full_k_observed=("is_full_k_observed", "mean"), pools=("task", "size"))
    r3_summary.to_csv(HERE / "results/R3_policy_comparison.csv", index=False, na_rep="")

    # Summary by HS2 and year; rows are means across the matching product-year pools.
    realized_df.groupby(["task", "hs2", "policy", "k"], as_index=False).mean(numeric_only=True).to_csv(
        HERE / "results/R5_by_hs2.csv", index=False, na_rep="")
    realized_df.groupby(["task", "outcome_year", "policy", "k"], as_index=False).mean(numeric_only=True).to_csv(
        HERE / "results/R5_by_year.csv", index=False, na_rep="")

    # Preference robustness and joint-uncertainty ablation evaluated on the same outcome utility.
    primary = realized_df.groupby(["task", "policy", "k"], as_index=False).agg(
        mean_utility=("selected_observed_utility_w050", "mean"),
        mean_utility_per_selected=("selected_observed_utility_per_selected", "mean"),
        mean_worst_preference_regret=("observed_case_worst_preference_regret", "mean"),
        mean_missing_selected=("missing_selected_count", "mean"),
        share_full_k_observed=("is_full_k_observed", "mean"), pools=("observed_case_worst_preference_regret", "count"))
    primary.to_csv(HERE / "results/R5_policy_summary.csv", index=False, na_rep="")

    # Missingness by historical market-size quartile (training-time information only).
    broad = pd.read_csv(HERE / "results/candidate_manifest.csv")
    miss_rows = []
    broad["size_quartile"] = broad.groupby("outcome_year").log_destination_import_demand.transform(
        lambda s: pd.qcut(s, 4, labels=False, duplicates="drop"))
    for (year, q), z in broad.groupby(["outcome_year", "size_quartile"], dropna=False):
        miss_rows.append({"outcome_year": year, "historical_import_demand_quartile": q,
                          "eligible_candidates": len(z), "unknown_outcomes": int((~z.label_observed.astype(bool)).sum()),
                          "unknown_share": float((~z.label_observed.astype(bool)).mean())})
    pd.DataFrame(miss_rows).to_csv(HERE / "results/R4_missingness_by_market_size.csv", index=False, na_rep="")

    # Out-of-time envelope coverage is descriptive diagnostic evaluation, never used to construct scenarios.
    coverage_rows = []
    scen = pd.read_parquet(HERE / "scenarios/predictive_utility_scenarios.parquet")
    scen["hs2"] = scen.hs2.astype(str).str.zfill(2)
    # Index scenario rows once. Filtering the full scenario table separately for
    # every HS2-year pool made this post-fit audit needlessly quadratic in pools.
    scenario_indices = scen.groupby(["task", "outcome_year", "hs2", "scenario_set"], sort=False).indices
    def scenario_pool(task: str, year: int, hs2: str, scenario_set: str) -> pd.DataFrame:
        idx = scenario_indices.get((task, int(year), str(hs2).zfill(2), scenario_set))
        return scen.iloc[idx] if idx is not None else scen.iloc[0:0]
    task_manifest = pd.read_csv(HERE / "results/candidate_task_manifest.csv", dtype={"hs2": str})
    for (task, year, hs2), c in task_manifest.groupby(["task", "outcome_year", "hs2"], sort=False):
        if task == TASK_B:
            primary = scenario_pool(task, year, hs2, "primary")
            amt = primary.groupby("candidate_id").objective2.agg(["min", "max"])
            c2 = c.set_index("candidate_id").join(amt, how="left")
            # Transform observed increment with the same fold-specific scale.
            cap = caps[(task, int(year))]
            obs = c2.label_observed.astype(bool).to_numpy()
            actual_raw = c2.realized_incremental_export_usd.to_numpy(float)
            actual = _scale(np.where(obs, actual_raw, np.nan), cap)
            cov = obs & (actual >= c2["min"].to_numpy(float)) & (actual <= c2["max"].to_numpy(float))
            coverage_rows.append({"task": task, "evaluation_year": year, "hs2": hs2, "envelope_type": "range across central model scenarios",
                                  "evaluated_n": int(obs.sum()), "covered_n": int(cov.sum()),
                                  "descriptive_coverage": float(cov.sum()/obs.sum()) if obs.sum() else np.nan,
                                  "calibration_claim": "none; out-of-time check only, test outcomes did not set the envelope"})
        else:
            primary = scenario_pool(task, year, hs2, "stress")
            qlo = primary.loc[primary.scenario_id.str.contains("q10"), ["candidate_id", "objective2"]].drop_duplicates("candidate_id").set_index("candidate_id").objective2
            qhi = primary.loc[primary.scenario_id.str.contains("q90"), ["candidate_id", "objective2"]].drop_duplicates("candidate_id").set_index("candidate_id").objective2
            c2 = c.set_index("candidate_id")
            cap = caps[(task, int(year))]
            obs = c2.label_observed.astype(bool).to_numpy()
            success_raw = c2.entry_success.to_numpy(float)
            success = np.isfinite(success_raw) & (success_raw == 1.0)
            actual_raw = c2.export_value_usd.to_numpy(float)
            actual = _scale(np.where(obs, actual_raw, np.nan), cap)
            actual = np.where(success, actual, 0)
            lo = qlo.reindex(c2.index).to_numpy(float); hi=qhi.reindex(c2.index).to_numpy(float)
            evalmask = obs & success
            cov = evalmask & (actual >= lo) & (actual <= hi)
            coverage_rows.append({"task": task, "evaluation_year": year, "hs2": hs2, "envelope_type": "training conditional-entry q10–q90 stress values",
                                  "evaluated_n": int(evalmask.sum()), "covered_n": int(cov.sum()),
                                  "descriptive_coverage": float(cov.sum()/evalmask.sum()) if evalmask.sum() else np.nan,
                                  "calibration_claim": "none; historical quantile envelope, conditional on observed successful entries"})
    pd.DataFrame(coverage_rows).to_csv(HERE / "results/R2_forecast_uncertainty_coverage.csv", index=False, na_rep="")

    # Independently reconstruct each selected set's worst-case predicted regret over
    # the same finite primary preference×prediction set. This gives fixed-score
    # baselines a comparable predicted-regret diagnostic and verifies PR-EMS from
    # saved candidate membership rather than trusting solver objective columns.
    predicted_regret_rows = []
    solver_objectives = pd.read_csv(HERE / "results/optimization_diagnostics.csv")
    solver_objectives["hs2"] = solver_objectives.hs2.astype(str).str.zfill(2)
    for key, candidates in candidate_dict.items():
        task, hs2, year = key
        candidates = candidates.sort_values("destination_iso3", kind="mergesort").reset_index(drop=True)
        ids = candidates.candidate_id.astype(str).tolist()
        pool = scenario_pool(task, year, hs2, "primary")
        oa = pool.pivot(index="scenario_id", columns="candidate_id", values="objective1").reindex(columns=ids)
        ob = pool.pivot(index="scenario_id", columns="candidate_id", values="objective2").reindex(index=oa.index, columns=ids)
        if oa.empty or oa.isna().any().any() or ob.isna().any().any():
            raise AssertionError(f"Primary predicted-regret scenarios do not match candidate pool {key}")
        a, b = oa.to_numpy(float), ob.to_numpy(float)
        utility = np.vstack([.25*a+.75*b, .75*a+.25*b])
        for (k, method), chosen in selected.loc[
                selected.task.eq(task) & selected.hs2.eq(str(hs2).zfill(2))
                & selected.outcome_year.eq(year)].groupby(["decision_budget_k", "selection_method"], sort=False):
            chosen_ids = set(chosen.candidate_id.astype(str))
            mask = np.asarray([candidate_id in chosen_ids for candidate_id in ids], dtype=bool)
            expected_k = min(int(k), len(ids))
            if int(mask.sum()) != expected_k:
                raise AssertionError(f"Predicted regret set cardinality mismatch for {key}/{k}/{method}")
            oracle = np.sort(utility, axis=1)[:, ::-1][:, :expected_k].sum(axis=1)
            set_utility = utility[:, mask].sum(axis=1)
            maximum_regret = float(np.max(oracle-set_utility))
            if maximum_regret < -1e-7:
                raise AssertionError(f"Negative matched-oracle regret for {key}/{k}/{method}")
            maxreg = max(0.0, maximum_regret)
            opt = solver_objectives.loc[solver_objectives.task.eq(task)
                                        & solver_objectives.hs2.eq(str(hs2).zfill(2))
                                        & solver_objectives.outcome_year.eq(year)
                                        & solver_objectives.k.eq(int(k))
                                        & solver_objectives.method.eq(method)]
            is_joint_mr = method == "PR-EMS"
            solver_obj = float(opt.objective.iloc[0]) if is_joint_mr and not opt.empty else np.nan
            abs_gap = float(opt.absolute_optimality_gap.iloc[0]) if is_joint_mr and not opt.empty else np.nan
            predicted_regret_rows.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year),
                                          "k": int(k), "policy": method, "candidate_count": len(ids),
                                          "scenario_count": len(utility), "predicted_worst_case_regret": maxreg,
                                          "solver_objective": solver_obj, "absolute_mip_gap": abs_gap,
                                          "solver_objective_comparable": is_joint_mr,
                                          "matches_solver_objective": (bool(abs(maxreg-solver_obj) <= max(1e-6, abs_gap)+1e-8)
                                                                       if is_joint_mr and pd.notna(solver_obj) else np.nan),
                                          "calibration_claim": "none; maximum over frozen finite stress scenarios, not a probability distribution"})
    predicted_regret_df = pd.DataFrame(predicted_regret_rows)
    if not predicted_regret_df.loc[predicted_regret_df.policy.eq("PR-EMS"), "matches_solver_objective"].all():
        raise AssertionError("Reconstructed PR-EMS regret differs from saved solver objective beyond gap tolerance")
    predicted_regret_df.to_csv(HERE / "results/R5_predicted_worst_case_regret.csv", index=False, na_rep="")

    return {"policy_pool_rows": len(realized_df), "regret_rows": len(regret_df),
            "missing_bound_rows": len(bounds_df), "paired_partial_id_rows": len(paired_bounds_df),
            "reconstructed_predicted_regret_rows": len(predicted_regret_df),
            "runtime_seconds": time.perf_counter() - start}
