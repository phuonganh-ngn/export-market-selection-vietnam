from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "experiments" / "decision_focused_xai_v1" / "src"))
from candidate_protocol import load_panel_and_candidates  # noqa: E402

KEY = ["destination_iso3", "hs2", "outcome_year"]
TASK_A = "Task_A_new_entry"
TASK_B = "Task_B_existing_underexploited"
FEATURES = [
    "log_gdp", "log_gdp_per_capita", "gdp_growth", "log_population", "urban_population_share",
    "internet_usage", "inflation", "trade_openness", "government_effectiveness", "regulatory_quality",
    "rule_of_law", "political_stability", "log_distance", "log_destination_import_demand",
    "log_lagged_export_value", "historical_market_share", "historical_export_volatility",
    "destination_supplier_concentration", "destination_import_demand_growth",
]


def _utility_scale(values: pd.Series | np.ndarray, cap: float, threshold: float) -> np.ndarray:
    x = np.maximum(np.asarray(values, dtype=float), 0.0)
    cap = max(float(cap), float(threshold) + 1.0)
    den = np.log1p(cap / threshold)
    return np.clip(np.log1p(np.minimum(x, cap) / threshold) / den, 0.0, 1.0)


def _hs2_quantile_bins(train: pd.DataFrame, target: pd.DataFrame, column: str, q: int = 5):
    cuts = {}
    out_t = pd.Series(index=target.index, dtype="Int64")
    out_r = pd.Series(index=train.index, dtype="Int64")
    for h in sorted(set(train.hs2.astype(str)) | set(target.hs2.astype(str))):
        tr = train.loc[train.hs2.astype(str).eq(h), column].dropna().to_numpy(float)
        if len(tr) == 0:
            cut = np.array([], dtype=float)
        else:
            cut = np.unique(np.quantile(tr, np.linspace(0, 1, q + 1)[1:-1]))
        cuts[h] = cut
        ti = target.index[target.hs2.astype(str).eq(h)]
        ri = train.index[train.hs2.astype(str).eq(h)]
        out_t.loc[ti] = np.searchsorted(cut, target.loc[ti, column].fillna(-np.inf).to_numpy(float), side="right")
        out_r.loc[ri] = np.searchsorted(cut, train.loc[ri, column].fillna(-np.inf).to_numpy(float), side="right")
    return out_t.astype(int), out_r.astype(int), cuts


def _entry_conditional_scales(train: pd.DataFrame, target: pd.DataFrame, cutoff: int, threshold: float):
    train = train.loc[
        train.outcome_year.le(cutoff) & train.entry_candidate_ex_ante & train.label_observed
        & train.entry_success.eq(1) & train.export_value_usd.ge(threshold)
    ].copy()
    all_train_candidates = train.loc[:, ["hs2", "log_destination_import_demand"]].copy()
    target = target.copy()
    target_bins, train_bins, _ = _hs2_quantile_bins(all_train_candidates, target, "log_destination_import_demand", 5)
    train["demand_bin"] = train_bins.to_numpy()
    target["demand_bin"] = target_bins.to_numpy()
    quantiles = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]
    qcols = {q: f"scale_q{int(q*100):02d}" for q in quantiles}
    grp = train.groupby([train.hs2.astype(str), "demand_bin"]).export_value_usd
    cell = grp.quantile(quantiles).unstack(-1)
    cell.columns = [qcols[float(c)] for c in cell.columns]
    cell["n_cell"] = grp.size()
    chapter = train.groupby(train.hs2.astype(str)).export_value_usd.quantile(quantiles).unstack(-1)
    chapter.columns = [qcols[float(c)] for c in chapter.columns]
    chapter["n_chapter"] = train.groupby(train.hs2.astype(str)).size()
    overall = train.export_value_usd.quantile(quantiles).to_dict()
    global_values = {qcols[q]: float(overall[q]) for q in quantiles}
    if train.empty:
        raise ValueError(f"No historical successful entries available by cutoff {cutoff}")
    out = target.copy()
    for q, col in qcols.items():
        vals = []
        for row in out.itertuples(index=False):
            h = str(row.hs2)
            b = int(row.demand_bin)
            if (h, b) in cell.index and int(cell.loc[(h, b), "n_cell"]) >= 8:
                val = cell.loc[(h, b), col]
            elif h in chapter.index and int(chapter.loc[h, "n_chapter"]) >= 8:
                val = chapter.loc[h, col]
            else:
                val = global_values[col]
            vals.append(float(val))
        out[col] = vals
    out["conditional_entry_n_train"] = [
        int(cell.loc[(str(r.hs2), int(r.demand_bin)), "n_cell"])
        if (str(r.hs2), int(r.demand_bin)) in cell.index else 0
        for r in out.itertuples(index=False)
    ]
    return out, train


def _fit_expansion_probability(train: pd.DataFrame, target: pd.DataFrame, cutoff: int):
    tr = train.loc[
        train.outcome_year.le(cutoff) & train.task_b_eligible & train.label_observed
        & train.realized_incremental_export_usd.notna()
    ].copy()
    if tr.empty or tr.realized_incremental_export_usd.gt(0).nunique() != 2:
        raise ValueError(f"Expansion classifier lacks both target classes through {cutoff}")
    med = SimpleImputer(strategy="median", add_indicator=True)
    model = make_pipeline(med, StandardScaler(), LogisticRegression(C=1.0, max_iter=600, solver="liblinear"))
    model.fit(tr[FEATURES].replace([np.inf, -np.inf], np.nan), tr.realized_incremental_export_usd.gt(0).astype(int))
    p_lr = model.predict_proba(target[FEATURES].replace([np.inf, -np.inf], np.nan))[:, 1]

    # A smoothed, training-only HS2 × lagged-share-bin event rate is a deliberately simple alternative.
    tbin, rbin, _ = _hs2_quantile_bins(tr, target, "historical_market_share", 4)
    tr["share_bin"] = rbin.to_numpy()
    target = target.copy()
    target["share_bin"] = tbin.to_numpy()
    tr["positive"] = tr.realized_incremental_export_usd.gt(0).astype(int)
    cell = tr.groupby([tr.hs2.astype(str), "share_bin"]).positive.agg(["sum", "count"])
    chapter = tr.groupby(tr.hs2.astype(str)).positive.agg(["sum", "count"])
    global_p = (tr.positive.sum() + 1.0) / (len(tr) + 2.0)
    p_emp = []
    for row in target.itertuples(index=False):
        key = (str(row.hs2), int(row.share_bin))
        if key in cell.index and cell.loc[key, "count"] >= 10:
            s, n = cell.loc[key, ["sum", "count"]]
            p_emp.append(float((s + 1.0) / (n + 2.0)))
        elif str(row.hs2) in chapter.index and chapter.loc[str(row.hs2), "count"] >= 10:
            s, n = chapter.loc[str(row.hs2), ["sum", "count"]]
            p_emp.append(float((s + 1.0) / (n + 2.0)))
        else:
            p_emp.append(float(global_p))
    return np.clip(p_lr, 0, 1), np.asarray(p_emp), tr, model


def _make_long_scenarios(d: pd.DataFrame, set_name: str, specs: list[tuple[str, str, float, float]], task: str):
    rows = []
    # specs are scenario_id, probability source, probability vector, amount/scale vector.
    for scenario_id, psource, pvals, avals in specs:
        z = d[["candidate_id", "destination_iso3", "hs2", "outcome_year"]].copy()
        z["task"] = task
        z["scenario_set"] = set_name
        z["scenario_id"] = scenario_id
        z["probability_source"] = psource
        z["objective1"] = np.asarray(pvals, dtype=float)
        z["objective2"] = np.asarray(avals, dtype=float)
        rows.append(z)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_scenarios() -> dict:
    panel, broad, task_manifest, _ = load_panel_and_candidates(write=False)
    panel["hs2"] = panel.hs2.astype(str).str.zfill(2)
    broad["hs2"] = broad.hs2.astype(str).str.zfill(2)
    task_manifest["hs2"] = task_manifest.hs2.astype(str).str.zfill(2)
    task_manifest["label_observed"] = task_manifest.label_observed.astype(bool)
    all_scenarios, baseline_rows, cal_rows, cost_rows = [], [], [], []
    input_e2 = pd.read_parquet(PROJECT / "experiments/post_audit_nn_dl_v1/predictions/E2_neural_predictions.parquet")
    input_e3 = pd.read_parquet(PROJECT / "experiments/post_audit_nn_dl_v1/predictions/E3_neural_rankings.parquet")
    e6 = pd.read_parquet(PROJECT / "experiments/decision_focused_xai_v1/predictions/E6_ranking_predictions.parquet")
    for frame in (input_e2, input_e3, e6):
        if "hs2" in frame:
            frame["hs2"] = frame.hs2.astype(str).str.zfill(2)
    e2key = ["destination_iso3", "hs2", "outcome_year"]
    e2p = input_e2.loc[input_e2.task.eq("classification")].copy()
    if "training_outcome_max" in e2p and (e2p.training_outcome_max > e2p.outcome_year - 2).any():
        raise AssertionError("Reused entry predictions violate y-2 training outcome cutoff")
    if "feature_year" in e2p and (e2p.feature_year > e2p.outcome_year - 2).any():
        raise AssertionError("Reused entry predictions violate y-2 feature cutoff")
    e2p["probability"] = e2p.predicted_probability
    if e2p.probability.isna().any():
        e2p["probability"] = e2p.probability.fillna(e2p.score)
    e2p = e2p.groupby(["model", *e2key], as_index=False).probability.mean()
    if not e2p.probability.between(0, 1).all():
        raise AssertionError("Task-A classifier predictions are not probabilities in [0,1]")

    e3b = input_e3.loc[input_e3.task.eq(TASK_B)].copy()
    e3b["outcome_year"] = e3b.evaluation_year.astype(int)
    e3b["hs2"] = e3b.hs2.astype(str).str.zfill(2)
    e3b["predicted_increment_usd"] = np.maximum(e3b.opportunity_score.fillna(0).to_numpy(float), 0.0)
    e3b = e3b.groupby(["ranking_model", *e2key], as_index=False).predicted_increment_usd.mean()

    for year in [2022, 2023, 2024]:
        fold_start = time.perf_counter()
        cutoff = year - 2
        fold = broad.loc[broad.outcome_year.eq(year)].copy()
        task_a = task_manifest.loc[task_manifest.task.eq(TASK_A) & task_manifest.outcome_year.eq(year)].copy()
        task_b = task_manifest.loc[task_manifest.task.eq(TASK_B) & task_manifest.outcome_year.eq(year)].copy()

        # Task A: probability disagreement comes from previously saved classifiers; conditional
        # entry scale is estimated only from successful historically inactive relationships.
        a = task_a.merge(fold, on=e2key, suffixes=("", "_panel"), validate="one_to_one")
        ep = e2p.loc[e2p.outcome_year.eq(year)].pivot(index=e2key, columns="model", values="probability")
        a = a.merge(ep.reset_index(), on=e2key, how="left", validate="one_to_one")
        p_models = [m for m in ["Historical_entry_frequency", "LogisticRegression", "RandomForest", "LightGBM", "XGBoost", "MLP", "TabM"] if m in a]
        if len(p_models) < 5 or a[p_models].isna().any().any():
            raise AssertionError(f"Task-A model probabilities fail to match the candidate manifest for {year}")
        a_train = panel.loc[
            panel.outcome_year.le(cutoff) & panel.eligible_ex_ante & panel.entry_candidate_ex_ante
            & panel.label_observed & panel.entry_success.eq(1)
        ].copy()
        scale_target = a[["hs2", "log_destination_import_demand"]].copy()
        scale_target.index = a.index
        scale_source = panel.loc[panel.outcome_year.le(cutoff) & panel.eligible_ex_ante & panel.entry_candidate_ex_ante & panel.label_observed].copy()
        scale_est, successful = _entry_conditional_scales(scale_source, scale_target, cutoff, 50_000.0)
        a["demand_bin"] = scale_est.demand_bin.to_numpy()
        a["conditional_entry_n_train"] = scale_est.conditional_entry_n_train.to_numpy()
        pos_train = successful.export_value_usd.to_numpy(float)
        scale_cap = float(np.quantile(pos_train, .95))
        pred_probs = {m: a[m].to_numpy(float) for m in p_models}
        q_utility = {}
        for q in [.10, .25, .40, .50, .60, .75, .90]:
            col = f"scale_q{int(q*100):02d}"
            q_utility[q] = _utility_scale(scale_est[col].to_numpy(float), scale_cap, 50_000.0)
        for set_name, mods, qs in [
            ("narrow", [m for m in ["LightGBM", "XGBoost"] if m in p_models], [.40, .60]),
            ("primary", p_models, [.25, .50, .75]),
            ("stress", p_models, [.10, .50, .90]),
        ]:
            specs = []
            for m in mods:
                for q in qs:
                    sid = f"entryprob={m}|conditional_size=q{int(q*100)}"
                    specs.append((sid, m, pred_probs[m], q_utility[q]))
            all_scenarios.append(_make_long_scenarios(a, set_name, specs, TASK_A))
        for model in p_models:
            cal_rows.append({"task": TASK_A, "evaluation_year": year, "cutoff": cutoff,
                             "model": model, "n": len(a), "mean_probability": float(a[model].mean()),
                             "scenario_role": "existing rolling-origin classifier; probability not recalibrated"})
        cal_rows.append({"task": TASK_A, "evaluation_year": year, "cutoff": cutoff,
                         "model": "conditional_size_historical_quantiles", "n": len(successful),
                         "scale_cap_usd_q95_train": scale_cap,
                         "scenario_role": "development-only historical successful entry amount quantiles; not a predictive interval"})

        # Task B: training-only logistic and smoothed product-share cell expansion probabilities;
        # saved regressors provide coherent full-candidate amount-disagreement scenarios.
        b = task_b.merge(fold, on=e2key, suffixes=("", "_panel"), validate="one_to_one")
        b_train = panel.loc[panel.outcome_year.le(cutoff) & panel.task_b_eligible & panel.label_observed].copy()
        p_lr, p_cell, b_train_fit, model = _fit_expansion_probability(b_train, b, cutoff)
        b["p_expansion_logistic"] = p_lr
        b["p_expansion_smoothed_cell"] = p_cell
        e3year = e3b.loc[e3b.outcome_year.eq(year)].pivot(index=e2key, columns="ranking_model", values="predicted_increment_usd")
        b = b.merge(e3year.reset_index(), on=e2key, how="left", validate="one_to_one")
        amount_models = [m for m in ["Gravity_PPML", "RandomForest", "XGBoost", "LightGBM", "MLP", "TabM"] if m in b]
        if len(amount_models) < 5 or b[amount_models].isna().any().any():
            raise AssertionError(f"Task-B amount forecasts fail to match the candidate manifest for {year}")
        train_inc = b_train_fit.realized_incremental_export_usd.clip(lower=0).to_numpy(float)
        amount_cap = float(np.quantile(train_inc, .95)) if len(train_inc) else 50_000.0
        amount_cap = max(amount_cap, 1.0)
        amount_util = {m: _utility_scale(b[m].to_numpy(float), amount_cap, 50_000.0) for m in amount_models}
        p_vectors = {"LogisticRegression": p_lr, "Smoothed_HS2_share_frequency": p_cell}
        for set_name, amods in [
            ("narrow", [m for m in ["Gravity_PPML", "LightGBM", "XGBoost"] if m in amount_models]),
            ("primary", amount_models),
            ("stress", amount_models),
        ]:
            specs = []
            use_amount = [(m, amount_util[m]) for m in amods]
            if set_name == "stress":
                matrix = np.vstack([amount_util[m] for m in amount_models])
                use_amount.extend([("candidate_model_envelope_low", np.min(matrix, axis=0)),
                                   ("candidate_model_envelope_high", np.max(matrix, axis=0))])
            for pm, pvec in p_vectors.items():
                for am, avec in use_amount:
                    specs.append((f"p={pm}|increment={am}", pm, pvec, avec))
            all_scenarios.append(_make_long_scenarios(b, set_name, specs, TASK_B))
        cal_rows.extend([
            {"task": TASK_B, "evaluation_year": year, "cutoff": cutoff, "model": "LogisticRegression",
             "n": len(b), "mean_probability": float(np.mean(p_lr)), "scenario_role": "training-only binary expansion model"},
            {"task": TASK_B, "evaluation_year": year, "cutoff": cutoff, "model": "Smoothed_HS2_share_frequency",
             "n": len(b), "mean_probability": float(np.mean(p_cell)), "scenario_role": "training-only beta-smoothed HS2×share-cell rate"},
            {"task": TASK_B, "evaluation_year": year, "cutoff": cutoff, "model": "saved_increment_model_scenarios",
             "n": len(b_train_fit), "scale_cap_usd_q95_train": amount_cap,
             "scenario_role": "saved rolling-origin predicted increments; scenarios are not probabilistic"},
        ])
        # Baseline ranking scores from the previously audited, full ex-ante table.
        eb = e6.loc[e6.outcome_year.eq(year) & e6.task.isin([TASK_A, TASK_B])].copy()
        baseline_rows.append(eb[["candidate_id", "destination_iso3", "hs2", "outcome_year", "task", "model", "seed", "score", "score_source"]])
        cost_rows.append({"stage": "scenario_build", "evaluation_year": year,
                          "runtime_seconds": time.perf_counter() - fold_start,
                          "task_a_candidates": len(a), "task_b_candidates": len(b),
                          "task_a_scenarios_primary": len(p_models) * 3,
                          "task_b_scenarios_primary": len(p_vectors) * len(amount_models),
                          "thread_limit": 4})

    scenarios = pd.concat(all_scenarios, ignore_index=True)
    if scenarios.duplicated(["task", "outcome_year", "hs2", "candidate_id", "scenario_set", "scenario_id"]).any():
        raise AssertionError("Candidate-scenario table contains duplicate keys")
    if not scenarios.objective1.between(0, 1).all() or not scenarios.objective2.between(0, 1).all():
        raise AssertionError("Normalized scenario utilities must be in [0,1]")
    scenarios.to_parquet(HERE / "scenarios/predictive_utility_scenarios.parquet", index=False)
    pd.concat(baseline_rows, ignore_index=True).to_parquet(HERE / "scenarios/baseline_ranking_scores.parquet", index=False)
    calibration = pd.DataFrame(cal_rows)
    calibration["calibration_status"] = "scenario-construction metadata only; no calibrated predictive distribution"
    calibration.to_csv(HERE / "results/R2_forecast_uncertainty_calibration.csv", index=False, na_rep="")
    calibration.loc[calibration.scale_cap_usd_q95_train.notna(),
                    ["task", "evaluation_year", "cutoff", "model", "scale_cap_usd_q95_train"]].to_csv(
        HERE / "results/R2_utility_scales.csv", index=False, na_rep="")
    pd.DataFrame(cost_rows).to_csv(HERE / "results/computational_cost.csv", index=False, na_rep="")
    meta = scenarios.groupby(["task", "outcome_year", "scenario_set"], as_index=False).agg(
        candidates=("candidate_id", "nunique"), scenarios=("scenario_id", "nunique"),
        utility1_min=("objective1", "min"), utility1_max=("objective1", "max"),
        utility2_min=("objective2", "min"), utility2_max=("objective2", "max"))
    meta["scenario_interpretation"] = "Finite model-disagreement or historical conditional-size stress set; not a calibrated distribution"
    meta.to_csv(HERE / "results/R2_predictive_scenarios.csv", index=False, na_rep="")
    return {"scenario_rows": len(scenarios), "scenario_pools": len(meta), "calibration_rows": len(cal_rows)}
