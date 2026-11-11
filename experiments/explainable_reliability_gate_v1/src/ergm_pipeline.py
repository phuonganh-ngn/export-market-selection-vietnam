from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .decision_metrics import pool_decision_metrics
from .historical_oof import reconstruct_historical_oof
from .task_audit import HERE, PROJECT, load_and_freeze

NN = PROJECT / "experiments/post_audit_nn_dl_v1"
RANK = PROJECT / "experiments/decision_focused_xai_v1"
EVAL_YEARS = (2022, 2023, 2024)
K = 10
FEATURE_CANDIDATES = ["log_gdp", "log_gdp_per_capita", "log_population", "log_destination_import_demand",
                      "destination_import_demand_growth", "historical_export_value", "historical_market_share",
                      "historical_export_volatility", "internet_usage", "government_effectiveness",
                      "digitalization_missing_indicator", "regulatory_quality", "rule_of_law", "political_stability"]
ROUTING_NUMERIC = ["candidate_count", "mean_log_gdp", "median_log_gdp", "sd_log_gdp",
                   "mean_log_demand", "median_log_demand", "mean_log_hist_exports", "mean_market_share",
                   "mean_trade_volatility", "mean_demand_growth", "predictor_missing_share",
                   "top10_jaccard", "rank_kendall_tau", "rank_displacement_norm",
                   "ppml_margin_z", "lightgbm_margin_z", "prior_ppml_regret",
                   "prior_lightgbm_regret", "prior_regret_gap", "prior_history_n"]
DISAGREEMENT_FEATURES = ["top10_jaccard", "rank_kendall_tau", "rank_displacement_norm"]
MISSING_FEATURES = ["predictor_missing_share"]
HISTORY_FEATURES = ["mean_log_hist_exports", "mean_market_share", "mean_trade_volatility",
                    "prior_ppml_regret", "prior_lightgbm_regret", "prior_regret_gap", "prior_history_n"]


def _sort_ranks(g: pd.DataFrame, score: str) -> pd.DataFrame:
    return g.sort_values([score, "destination_iso3"], ascending=[False, True], kind="mergesort").copy()


def load_test_predictions(manifest: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Reuse verified outer predictions; filter to corrected, disjoint task sets."""
    saved = pd.read_parquet(RANK / "predictions/E6_ranking_predictions.parquet")
    saved["hs2"] = saved.hs2.astype(str).str.zfill(2)
    saved["outcome_year"] = saved.outcome_year.astype(int)
    saved = saved.loc[saved.seed.eq(-1) & saved.model.isin(["Gravity_PPML", "LightGBM"]),
                      ["candidate_id", "task", "model", "score"]].copy()
    saved = saved.loc[saved.candidate_id.isin(manifest.candidate_id)]
    if saved.duplicated(["candidate_id", "task", "model"]).any():
        raise AssertionError("Duplicate saved baseline expert prediction key")
    wide = saved.pivot(index=["candidate_id", "task"], columns="model", values="score").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"Gravity_PPML": "ppml_score", "LightGBM": "lightgbm_score"})
    expected = manifest[["candidate_id", "task"]].drop_duplicates()
    out = expected.merge(wide, on=["candidate_id", "task"], how="left", validate="one_to_one")
    if out[["ppml_score", "lightgbm_score"]].isna().any().any():
        bad = out.loc[out[["ppml_score", "lightgbm_score"]].isna().any(axis=1)]
        raise AssertionError(f"Missing expert prediction for corrected candidates: {len(bad)} rows")
    meta_cols = [c for c in FEATURE_CANDIDATES if c not in manifest.columns]
    meta = panel[["candidate_id", *meta_cols, "continent"]].drop_duplicates("candidate_id")
    out = out.merge(manifest, on=["candidate_id", "task"], how="left", validate="one_to_one")
    out = out.merge(meta, on="candidate_id", how="left", validate="many_to_one")
    out["score_provenance"] = "saved verified out-of-time E6 baseline predictions; seed-mean/archived model"
    out["prediction_source_year"] = out.outcome_year.astype(int)
    return out


def combine_oof_and_test(oof: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    oof = oof.copy()
    if not oof.empty:
        oof["is_test"] = False
        oof["score_provenance"] = oof.prediction_provenance
    test = test.copy()
    test["is_test"] = True
    # Harmonize and retain all score/label fields needed for independent reproduction.
    cols = list(dict.fromkeys(["candidate_id", "destination_iso3", "destination_name", "hs2", "outcome_year", "task", "candidate_pool_id",
            "label_observed", "entry_success", "export_value_usd", "realized_incremental_export_usd",
            "historical_export_value", "historical_market_share", *FEATURE_CANDIDATES,
            "ppml_score", "lightgbm_score", "score_provenance", "is_test"]))
    for d in [oof, test]:
        for c in cols:
            if c not in d.columns:
                d[c] = np.nan
        d["hs2"] = d.hs2.astype(str).str.zfill(2)
        d["outcome_year"] = d.outcome_year.astype(int)
    result = pd.concat([oof[cols], test[cols]], ignore_index=True)
    if result.duplicated(["task", "candidate_id", "outcome_year", "is_test"]).any():
        raise AssertionError("Duplicate expert score candidate keys")
    return result


def pool_features(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, hs2, year), g in candidates.groupby(["task", "hs2", "outcome_year"], sort=True):
        p = _sort_ranks(g, "ppml_score")
        l = _sort_ranks(g, "lightgbm_score")
        n = len(g)
        pr = {iso: i+1 for i, iso in enumerate(p.destination_iso3.astype(str))}
        lr = {iso: i+1 for i, iso in enumerate(l.destination_iso3.astype(str))}
        ptop = set(p.head(K).destination_iso3.astype(str))
        ltop = set(l.head(K).destination_iso3.astype(str))
        union = len(ptop | ltop)
        tau = kendalltau([pr[x] for x in sorted(pr)], [lr[x] for x in sorted(pr)]).statistic if n > 1 else np.nan
        hist = pd.to_numeric(g.historical_export_value, errors="coerce").clip(lower=0)
        demand = pd.to_numeric(g.log_destination_import_demand, errors="coerce")
        gdp = pd.to_numeric(g.log_gdp, errors="coerce")
        numeric = [c for c in FEATURE_CANDIDATES if c in g.columns]
        miss = float(g[numeric].isna().to_numpy().mean()) if numeric else np.nan
        pmargin = _margin_z(p.ppml_score.to_numpy(float), K)
        lmargin = _margin_z(l.lightgbm_score.to_numpy(float), K)
        rows.append({"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year),
            "candidate_pool_id": f"{task}|{str(hs2).zfill(2)}|{int(year)}", "candidate_count": n,
            "mean_log_gdp": float(gdp.mean()), "median_log_gdp": float(gdp.median()), "sd_log_gdp": float(gdp.std()),
            "mean_log_demand": float(demand.mean()), "median_log_demand": float(demand.median()),
            "mean_log_hist_exports": float(np.log1p(hist).mean()),
            "mean_market_share": float(pd.to_numeric(g.historical_market_share, errors="coerce").mean()),
            "mean_trade_volatility": float(pd.to_numeric(g.historical_export_volatility, errors="coerce").mean()),
            "mean_demand_growth": float(pd.to_numeric(g.destination_import_demand_growth, errors="coerce").mean()),
            "predictor_missing_share": miss, "top10_jaccard": len(ptop & ltop)/union if union else np.nan,
            "rank_kendall_tau": float(tau) if np.isfinite(tau) else np.nan,
            "rank_displacement_norm": float(np.mean([abs(pr[x]-lr[x]) for x in pr])/max(1, n-1)),
            "ppml_margin_z": pmargin, "lightgbm_margin_z": lmargin,
            "top10_ppml": ";".join(p.head(K).destination_iso3.astype(str)),
            "top10_lightgbm": ";".join(l.head(K).destination_iso3.astype(str)),
        })
    return pd.DataFrame(rows)


def _margin_z(scores: np.ndarray, k: int) -> float:
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    if not len(s):
        return np.nan
    gap = s[min(k-1, len(s)-1)] - s[k] if len(s) > k else s[0] - s[-1]
    sd = float(np.std(s))
    return float(gap/sd) if sd > 0 else 0.0


def pool_losses(candidates: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, hs2, year), g in candidates.groupby(["task", "hs2", "outcome_year"], sort=True):
        row = {"task": task, "hs2": str(hs2).zfill(2), "outcome_year": int(year),
               "candidate_pool_id": f"{task}|{str(hs2).zfill(2)}|{int(year)}"}
        for expert, col in (("Gravity_PPML", "ppml_score"), ("LightGBM", "lightgbm_score")):
            met = pool_decision_metrics(g, col, task)
            row[f"regret_{expert}"] = met["regret@10"]
            row[f"ndcg10_{expert}"] = met["ndcg@10"]
            row[f"precision10_{expert}"] = met["precision@10"]
            row[f"utility_{expert}"] = met["selected_gain@10"]
            row[f"positive_usd_{expert}"] = met["selected_positive_usd@10"]
            row[f"observed_labels_{expert}"] = met["observed_label_count"]
            row[f"unknown_top10_{expert}"] = met["missing_in_full_score_top10"]
        rows.append(row)
    out = pd.DataFrame(rows).merge(features, on=["task", "hs2", "outcome_year", "candidate_pool_id"],
                                   how="left", validate="one_to_one")
    out = out.sort_values(["task", "hs2", "outcome_year"], kind="mergesort").reset_index(drop=True)
    out["prior_ppml_regret"] = np.nan
    out["prior_lightgbm_regret"] = np.nan
    out["prior_regret_gap"] = np.nan
    out["prior_history_n"] = 0
    for _, idx in out.groupby(["task", "hs2"], sort=False).groups.items():
        positions = list(idx)
        group = out.loc[positions].copy()
        for pos in positions:
            year = int(out.at[pos, "outcome_year"])
            # At an outcome year t, only regrets observed by the y-2 origin are
            # available. A simple t-1 history feature would leak one year.
            prior = group.loc[group.outcome_year.le(year-2)]
            valid_prior = prior.loc[prior.regret_Gravity_PPML.notna()&prior.regret_LightGBM.notna()]
            if len(valid_prior):
                out.at[pos, "prior_ppml_regret"] = float(valid_prior.regret_Gravity_PPML.mean())
                out.at[pos, "prior_lightgbm_regret"] = float(valid_prior.regret_LightGBM.mean())
                out.at[pos, "prior_regret_gap"] = (out.at[pos, "prior_lightgbm_regret"]-
                                                     out.at[pos, "prior_ppml_regret"])
                out.at[pos, "prior_history_n"] = len(valid_prior)
    return out


def _router(task_train: pd.DataFrame, task_test: pd.DataFrame, selected_features: list[str] | None = None):
    features = selected_features or ROUTING_NUMERIC
    # Drop development-fold columns with no observed values. Such columns carry
    # no estimable routing information and otherwise trigger an all-missing
    # imputer warning (and are silently removed by scikit-learn downstream).
    numeric = [c for c in features if c in ROUTING_NUMERIC and
               c in task_train.columns and task_train[c].notna().any()]
    cats = ["hs2"]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                           ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cats),
    ], remainder="drop")
    model = Pipeline([("prep", prep), ("reg", Ridge(alpha=10.0))])
    y = task_train[["regret_Gravity_PPML", "regret_LightGBM"]].to_numpy(float)
    model.fit(task_train, y)
    pred = np.maximum(0.0, model.predict(task_test))
    train_pred = np.maximum(0.0, model.predict(task_train))
    return model, pred, train_pred


def _crossfit_gate_predictions(task_train: pd.DataFrame,
                               selected_features: list[str] | None = None) -> pd.DataFrame:
    """Chronological held-out regret predictions for calibrating risk cutoffs.

    For a development pool in year t, the gate is fit only to observed pool
    outcomes through t-2. The target pool's regret is never used to score its
    own risk. With the available 2018-2021 OOF archive this yields a short,
    explicitly limited calibration history rather than in-sample thresholds.
    """
    pieces = []
    for year in sorted(task_train.outcome_year.astype(int).unique()):
        validation = task_train.loc[task_train.outcome_year.eq(year)].copy()
        fit = task_train.loc[task_train.outcome_year.le(int(year)-2)
            & task_train.regret_Gravity_PPML.notna() & task_train.regret_LightGBM.notna()].copy()
        if len(fit) < 40 or validation.empty:
            continue
        _, prediction, _ = _router(fit, validation, selected_features)
        validation["crossfit_predicted_regret_ppml"] = prediction[:, 0]
        validation["crossfit_predicted_regret_lightgbm"] = prediction[:, 1]
        pieces.append(validation)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _route_label(loss_g: float, loss_l: float) -> str:
    return "Gravity_PPML" if loss_g <= loss_l else "LightGBM"


def _risk_series(frame: pd.DataFrame, policy: str, route: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    ppml_margin = frame.ppml_margin_z.to_numpy(float)
    lgb_margin = frame.lightgbm_margin_z.to_numpy(float)
    disagreement = 1.0 - frame.top10_jaccard.to_numpy(float) + frame.rank_displacement_norm.to_numpy(float)
    static = "LightGBM" if frame.task.iloc[0] == "Task_A_new_entry" else "Gravity_PPML"
    static_risk = predicted[:, 0] if static == "Gravity_PPML" else predicted[:, 1]
    chosen = np.where(route == "Gravity_PPML", predicted[:, 0], predicted[:, 1])
    if policy == "ERGM_FULL":
        return chosen
    if policy == "ADAPTIVE_RISK":
        return chosen
    if policy == "STATIC_PLUS_RISK":
        return static_risk
    if policy == "DISAGREEMENT_ONLY":
        return disagreement
    if policy == "LIGHTGBM_CONFIDENCE":
        return -lgb_margin
    if policy == "PPML_CONFIDENCE":
        return -ppml_margin
    if policy == "STATIC_CONFIDENCE":
        return -lgb_margin if static == "LightGBM" else -ppml_margin
    if policy == "SCORE_MARGIN":
        return -lgb_margin if static == "LightGBM" else -ppml_margin
    raise ValueError(policy)


def _cutoff_accept(train_risk: np.ndarray, test_risk: np.ndarray, coverage: float,
                   ids: pd.Series, matched: bool = False) -> tuple[np.ndarray, float]:
    vals = train_risk[np.isfinite(train_risk)]
    if len(vals) == 0:
        return np.ones(len(test_risk), dtype=bool), np.nan
    threshold = float(np.quantile(vals, coverage, method="linear")) if coverage < 1 else float("inf")
    if matched:
        # Test labels are never used. This is an unsupervised standardization to
        # exactly the requested realized workload for paired method comparison.
        order = np.lexsort((ids.astype(str).to_numpy(), test_risk))
        accept = np.zeros(len(test_risk), dtype=bool)
        n_accept = int(np.ceil(coverage*len(test_risk)))
        accept[order[:n_accept]] = True
        threshold = float(test_risk[order[n_accept-1]]) if n_accept else np.nan
        return accept, threshold
    # Operational policy uses a frozen development quantile. Realized test
    # coverage is allowed to differ from the requested target.
    return np.asarray(test_risk, float) <= threshold, threshold


def _metrics_table_for_policy(candidates: pd.DataFrame, pool_info: pd.DataFrame,
                              score_name: str, policy: str, accepted: np.ndarray,
                              requested: float | None, threshold: float, coverage_mode: str) -> tuple[dict, list[dict]]:
    accepted_ids = set(pool_info.loc[accepted, "candidate_pool_id"])
    rows = []
    for r in pool_info.itertuples(index=False):
        g = candidates.loc[candidates.candidate_pool_id.eq(r.candidate_pool_id)].copy()
        if policy == "STATIC_BLEND":
            g["_policy_score"] = g.blend_rank_score
        else:
            expert = getattr(r, "route_expert")
            g["_policy_score"] = g.ppml_score if expert == "Gravity_PPML" else g.lightgbm_score
        m = pool_decision_metrics(g, "_policy_score", r.task)
        rows.append({"task": r.task, "outcome_year": int(r.outcome_year), "hs2": r.hs2,
            "candidate_pool_id": r.candidate_pool_id, "policy": policy,
            "requested_coverage": requested, "coverage_mode": coverage_mode,
            "accepted": r.candidate_pool_id in accepted_ids, "acceptance_threshold": threshold,
            **m})
    d = pd.DataFrame(rows)
    a = d.loc[d.accepted]
    valid = a.loc[a["regret@10"].notna()]
    overall = {"task": pool_info.task.iloc[0], "outcome_year": int(pool_info.outcome_year.iloc[0]),
        "policy": policy, "requested_coverage": requested,
        "coverage_mode": coverage_mode, "automation_coverage": float(d.accepted.mean()) if len(d) else np.nan,
        "deferral_rate": float(1-d.accepted.mean()) if len(d) else np.nan,
        "all_pools": int(len(d)), "accepted_pools": int(d.accepted.sum()),
        "accepted_pools_with_observed_metrics": int(len(valid)),
        "deferred_pools_with_observed_metrics": int((~d.accepted & d["regret@10"].notna()).sum()),
        "mean_regret10_accepted": float(valid["regret@10"].mean()) if len(valid) else np.nan,
        "median_regret10_accepted": float(valid["regret@10"].median()) if len(valid) else np.nan,
        "p90_regret10_accepted": float(valid["regret@10"].quantile(.9)) if len(valid) else np.nan,
        "mean_ndcg10_accepted": float(valid["ndcg@10"].mean()) if len(valid) else np.nan,
        "mean_precision10_accepted": float(valid["precision@10"].mean()) if len(valid) else np.nan,
        "mean_observed_positive_usd_top10": float(valid["selected_positive_usd@10"].mean()) if len(valid) else np.nan,
        "mean_unknowns_in_top10": float(valid["missing_in_full_score_top10"].mean()) if len(valid) else np.nan,
        "mean_regret10_deferred_diagnostic": float(d.loc[~d.accepted, "regret@10"].mean()) if (~d.accepted).any() else np.nan,
        "risk_definition": "complete-case historical Top-10 regret; no human-review outcome observed"}
    return overall, rows


def _calc_blend(candidates: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    for _, g in candidates.groupby(["task", "hs2", "outcome_year"], sort=False):
        q = g.copy()
        q["_r_p"] = _sort_ranks(q, "ppml_score").reset_index().reset_index().set_index("index")["level_0"].reindex(q.index).to_numpy()+1
        q["_r_l"] = _sort_ranks(q, "lightgbm_score").reset_index().reset_index().set_index("index")["level_0"].reindex(q.index).to_numpy()+1
        n = len(q)
        q["blend_rank_score"] = .5*((n-q._r_p+1)/max(1,n-1) + (n-q._r_l+1)/max(1,n-1))
        chunks.append(q)
    return pd.concat(chunks, ignore_index=True) if chunks else candidates.copy()


def _augment_scores(candidates: pd.DataFrame) -> pd.DataFrame:
    d = candidates.copy()
    # Rank values use deterministic ISO3 tie breaking. A rank-average blend is on
    # within-pool percentiles; it does not average incomparable raw scores.
    return _calc_blend(d)


def run_routing_and_gating(all_candidates: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fit historical pool-level route gates and evaluate static/adaptive selective policies."""
    start = time.perf_counter()
    candidates = _augment_scores(all_candidates)
    features = pool_features(candidates)
    losses = pool_losses(candidates, features)
    # `is_test` is constant within a pool, merged after loss construction.
    pool_is_test = candidates.groupby(["task", "hs2", "outcome_year"], as_index=False).is_test.first()
    losses = losses.merge(pool_is_test, on=["task", "hs2", "outcome_year"], how="left", validate="one_to_one")
    historical = losses.loc[~losses.is_test.astype(bool)].copy()
    test_pools = losses.loc[losses.is_test.astype(bool)].copy()
    if test_pools.empty or historical.empty:
        raise RuntimeError("No historical OOF pools or outer evaluation pools are available")

    static_rows, blend_rows, route_rows, freq_rows, risk_rows = [], [], [], [], []
    coverage_rows, matched_rows, matched_pool_rows, operational_decisions = [], [], [], []
    route_explanations, coefficients, counterfactuals = [], [], []
    ablation_rows = []
    # Per-fold models and routing scores are saved for reconstruction.
    for task in ["Task_A_new_entry", "Task_B_existing_underexploited"]:
        for year in EVAL_YEARS:
            test = test_pools.loc[test_pools.task.eq(task) & test_pools.outcome_year.eq(year)].copy()
            if test.empty:
                continue
            train = losses.loc[losses.task.eq(task) & losses.outcome_year.le(year-2)
                               & losses.regret_Gravity_PPML.notna() & losses.regret_LightGBM.notna()].copy()
            if len(train) < 40:
                raise RuntimeError(f"Only {len(train)} historical pool labels for {task}/{year}; insufficient to fit route gate")
            route_model, pred_loss, train_pred = _router(train, test)
            joblib_path = HERE / f"models/router_{'entry' if task=='Task_A_new_entry' else 'expansion'}_{year}.joblib"
            joblib.dump({"model": route_model, "training_pool_count": len(train),
                         "training_year_max": int(train.outcome_year.max()), "features": ROUTING_NUMERIC,
                         "alpha": 10.0}, joblib_path)
            test["predicted_regret_ppml"] = pred_loss[:, 0]
            test["predicted_regret_lightgbm"] = pred_loss[:, 1]
            test["route_expert"] = np.where(pred_loss[:, 0] <= pred_loss[:, 1], "Gravity_PPML", "LightGBM")
            test["predicted_route_regret"] = np.min(pred_loss, axis=1)
            # Fixed task rule is based on previously developed M1 evidence; an
            # explicitly exploratory baseline, not an untouched preregistration.
            static_expert = "LightGBM" if task == "Task_A_new_entry" else "Gravity_PPML"
            test["static_expert"] = static_expert
            hist_mean = {"Gravity_PPML": float(train.regret_Gravity_PPML.mean()),
                         "LightGBM": float(train.regret_LightGBM.mean())}
            hist_expert = min(hist_mean, key=hist_mean.get)
            test["historical_winner_expert"] = hist_expert
            routes = {
                "STATIC_TASK_ROUTING": np.repeat(static_expert, len(test)),
                "HISTORICAL_WINNER_ROUTING": np.repeat(hist_expert, len(test)),
                "ADAPTIVE_ROUTING": test.route_expert.to_numpy(),
                "STATIC_PPML": np.repeat("Gravity_PPML", len(test)),
                "STATIC_LIGHTGBM": np.repeat("LightGBM", len(test)),
            }
            for name, vec in routes.items():
                for i, r in enumerate(test.itertuples(index=False)):
                    expert = vec[i]
                    col = "regret_Gravity_PPML" if expert == "Gravity_PPML" else "regret_LightGBM"
                    ndcol = "ndcg10_Gravity_PPML" if expert == "Gravity_PPML" else "ndcg10_LightGBM"
                    pcol = "precision10_Gravity_PPML" if expert == "Gravity_PPML" else "precision10_LightGBM"
                    usdcol = "positive_usd_Gravity_PPML" if expert == "Gravity_PPML" else "positive_usd_LightGBM"
                    route_rows.append({"task": task, "outcome_year": year, "hs2": r.hs2,
                        "candidate_pool_id": r.candidate_pool_id, "policy": name,
                        "selected_expert": str(expert),
                        "predicted_regret_ppml": float(r.predicted_regret_ppml),
                        "predicted_regret_lightgbm": float(r.predicted_regret_lightgbm),
                        "predicted_route_regret": float(min(r.predicted_regret_ppml,r.predicted_regret_lightgbm)),
                        "realized_regret10": float(getattr(r,col)), "ndcg10": float(getattr(r,ndcol)),
                        "precision10": float(getattr(r,pcol)), "selected_positive_usd_top10": float(getattr(r,usdcol)),
                        "top10_jaccard": float(r.top10_jaccard), "rank_kendall_tau": float(r.rank_kendall_tau),
                        "predictor_missing_share": float(r.predictor_missing_share),
                        "training_pool_count": len(train), "training_outcome_max": int(train.outcome_year.max())})
                freq_rows.append({"task": task, "outcome_year": year, "policy": name,
                                  "selected_lightgbm_share": float(np.mean(vec=="LightGBM")),
                                  "selected_ppml_share": float(np.mean(vec=="Gravity_PPML")), "pool_count": len(test)})

            # Equal-weight within-pool normalized rank blend.
            cand_test = candidates.loc[candidates.task.eq(task) & candidates.outcome_year.eq(year)].copy()
            for _, g in cand_test.groupby(["hs2", "outcome_year"], sort=False):
                m = pool_decision_metrics(g, "blend_rank_score", task)
                blend_rows.append({"task": task, "outcome_year": year, "hs2": str(g.hs2.iloc[0]).zfill(2),
                    "candidate_pool_id": g.candidate_pool_id.iloc[0], "policy": "EQUAL_WEIGHT_RANK_BLEND",
                    "ndcg10": m["ndcg@10"], "precision10": m["precision@10"], "regret10": m["regret@10"],
                    "selected_positive_usd_top10": m["selected_positive_usd@10"], "coverage": 1.0})

            # Explain route choice with the additive, scaled ridge gate. Sum of
            # coefficient contributions plus intercept reproduces each loss score.
            prep = route_model.named_steps["prep"]
            reg = route_model.named_steps["reg"]
            x_test = prep.transform(test)
            x_names = prep.get_feature_names_out()
            coef = np.asarray(reg.coef_)
            if coef.ndim == 1:
                coef = coef.reshape(1, -1)
            diff_coef = coef[1]-coef[0]
            for feat_name, coef_val in zip(x_names, diff_coef):
                coefficients.append({"task": task, "outcome_year": year, "feature": str(feat_name),
                    "coefficient_lightgbm_minus_ppml_regret": float(coef_val),
                    "interpretation": "linear association in scaled/categorical gate feature space; not causal"})
            x_dense = x_test.toarray() if hasattr(x_test, "toarray") else np.asarray(x_test)
            contribution = x_dense * diff_coef[None, :]
            intercept_diff = float(np.asarray(reg.intercept_).reshape(-1)[1]-np.asarray(reg.intercept_).reshape(-1)[0])
            raw_gate_predictions = np.asarray(reg.predict(x_test), dtype=float)
            for i, r in enumerate(test.itertuples(index=False)):
                chosen = r.route_expert
                vals = contribution[i]
                order = np.argsort(np.abs(vals))[::-1][:5]
                reconstructed_diff = intercept_diff + float(vals.sum())
                direct_raw_diff = float(raw_gate_predictions[i, 1]-raw_gate_predictions[i, 0])
                direct_clipped_diff = float(r.predicted_regret_lightgbm-r.predicted_regret_ppml)
                route_explanations.append({"task": task, "outcome_year": year, "hs2": r.hs2,
                    "candidate_pool_id": r.candidate_pool_id, "selected_expert": chosen,
                    "alternative_expert": "LightGBM" if chosen=="Gravity_PPML" else "Gravity_PPML",
                    "predicted_regret_ppml": r.predicted_regret_ppml,
                    "predicted_regret_lightgbm": r.predicted_regret_lightgbm,
                    "raw_ridge_regret_score_difference_lgb_minus_ppml": direct_raw_diff,
                    "clipped_predicted_regret_difference_lgb_minus_ppml": direct_clipped_diff,
                    "both_regret_scores_clipped_to_zero_tie": bool(abs(direct_clipped_diff)<1e-12),
                    "route_reconstruction_error": abs(reconstructed_diff-direct_raw_diff),
                    "route_decision_reproduced": bool(chosen == ("Gravity_PPML" if direct_clipped_diff >= 0 else "LightGBM")),
                    "top10_overlap": r.top10_jaccard, "rank_kendall_tau": r.rank_kendall_tau,
                    "route_feature_contributions": "; ".join(f"{x_names[j]}={vals[j]:+.4g}" for j in order),
                    "realized_regret10": r.regret_Gravity_PPML if chosen=="Gravity_PPML" else r.regret_LightGBM})
                if abs(direct_clipped_diff) > 1e-12:
                    best_j = None
                    best_delta = np.inf
                    for j, name in enumerate(x_names):
                        if not str(name).startswith("num__") or abs(diff_coef[j]) < 1e-10:
                            continue
                        delta = -direct_raw_diff/diff_coef[j]
                        if abs(delta) < best_delta:
                            best_delta, best_j = abs(delta), (j, float(delta), str(name))
                    if best_j:
                        counterfactuals.append({"task": task, "outcome_year": year, "hs2": r.hs2,
                            "candidate_pool_id": r.candidate_pool_id, "current_expert": chosen,
                            "switch_feature": best_j[2], "standardized_change_to_tie": best_j[1],
                            "absolute_standardized_change": best_delta,
                            "interpretation": "model-space one-feature perturbation; not a feasible economic intervention"})

            # Gate operating points. Reliability scores are predicted regrets, not
            # calibrated probabilities. Development thresholds never use test labels.
            static_route = routes["STATIC_TASK_ROUTING"]
            risk_methods = ["ERGM_FULL", "STATIC_PLUS_RISK", "DISAGREEMENT_ONLY",
                            "LIGHTGBM_CONFIDENCE", "PPML_CONFIDENCE", "STATIC_CONFIDENCE"]
            calibration_pools = _crossfit_gate_predictions(train)
            for pol in risk_methods:
                test_risk = _risk_series(test, pol, routes["ADAPTIVE_ROUTING"], pred_loss)
                # Learned-risk thresholds use chronological held-out pool scores,
                # never the fitted values on the gate's own training targets.
                if pol in ("ERGM_FULL", "STATIC_PLUS_RISK"):
                    if calibration_pools.empty:
                        raise RuntimeError("No chronological held-out risk calibration pools")
                    calibration_pred = calibration_pools[["crossfit_predicted_regret_ppml",
                        "crossfit_predicted_regret_lightgbm"]].to_numpy(float)
                    if pol == "ERGM_FULL":
                        trisk = np.min(calibration_pred, axis=1)
                    else:
                        trisk = calibration_pred[:, 1] if static_expert == "LightGBM" else calibration_pred[:, 0]
                else:
                    if calibration_pools.empty:
                        raise RuntimeError("No chronological risk calibration pools")
                    calibration_pred = calibration_pools[["crossfit_predicted_regret_ppml",
                        "crossfit_predicted_regret_lightgbm"]].to_numpy(float)
                    calibration_route = np.where(calibration_pred[:, 0] <= calibration_pred[:, 1],
                                                  "Gravity_PPML", "LightGBM")
                    trisk = _risk_series(calibration_pools, pol, calibration_route, calibration_pred)
                for target_cov in [1.0, .9, .8, .7, .6, .5]:
                    operational_accept, threshold = _cutoff_accept(np.asarray(trisk, float), test_risk,
                        target_cov, test.hs2, matched=False)
                    route = routes["ADAPTIVE_ROUTING"] if pol=="ERGM_FULL" else static_route
                    policy_route = (routes["ADAPTIVE_ROUTING"] if pol=="ERGM_FULL" else
                        (routes["STATIC_TASK_ROUTING"] if pol in ["STATIC_PLUS_RISK","DISAGREEMENT_ONLY","STATIC_CONFIDENCE","SCORE_MARGIN"] else
                         (routes["STATIC_LIGHTGBM"] if pol=="LIGHTGBM_CONFIDENCE" else routes["STATIC_PPML"])))
                    policy_info = test.copy(); policy_info["route_expert"] = policy_route
                    overall, per_pool = _metrics_table_for_policy(candidates.loc[
                        candidates.task.eq(task)&candidates.outcome_year.eq(year)], policy_info,
                        "score", pol, operational_accept, target_cov, threshold, "development_threshold")
                    coverage_rows.append(overall)
                    operational_decisions.extend(per_pool)
                    matched_accept, matched_threshold = _cutoff_accept(test_risk, test_risk,
                        target_cov, test.hs2, matched=True)
                    # Fixed matched-coverage results are standardized by covariates only;
                    # they are not the operational threshold estimate.
                    policy_info_m = test.copy(); policy_info_m["route_expert"] = policy_route
                    overall_m, per_pool_m = _metrics_table_for_policy(candidates.loc[
                        candidates.task.eq(task)&candidates.outcome_year.eq(year)], policy_info_m,
                        "score", pol, matched_accept, target_cov, matched_threshold, "matched_test_coverage")
                    matched_rows.append(overall_m)
                    matched_pool_rows.extend(per_pool_m)
                    operational_decisions.extend(per_pool_m)
            # No-deferral policies and ranking-only comparison.
            for name, vec in routes.items():
                if name not in ("STATIC_TASK_ROUTING", "HISTORICAL_WINNER_ROUTING", "ADAPTIVE_ROUTING"):
                    continue
                for i, r in enumerate(test.itertuples(index=False)):
                    col = "regret_Gravity_PPML" if vec[i]=="Gravity_PPML" else "regret_LightGBM"
                    ndcol = "ndcg10_Gravity_PPML" if vec[i]=="Gravity_PPML" else "ndcg10_LightGBM"
                    pcol = "precision10_Gravity_PPML" if vec[i]=="Gravity_PPML" else "precision10_LightGBM"
                    usdcol = "positive_usd_Gravity_PPML" if vec[i]=="Gravity_PPML" else "positive_usd_LightGBM"
                    static_rows.append({"task": task, "outcome_year": year, "hs2": r.hs2,
                        "candidate_pool_id": r.candidate_pool_id, "policy": name,
                        "selected_expert": vec[i], "coverage": 1.0,
                        "regret10": getattr(r, col), "ndcg10": getattr(r, ndcol),
                        "precision10": getattr(r, pcol), "selected_positive_usd_top10": getattr(r, usdcol),
                        "candidate_count": r.candidate_count,
                        "observed_label_count": getattr(r, f"observed_labels_{vec[i]}")})

            # Learned-risk and feature-group ablation at development-selected 80% risk threshold.
            ablation_specs = {
                "FULL": ROUTING_NUMERIC,
                "NO_DISAGREEMENT": [f for f in ROUTING_NUMERIC if f not in DISAGREEMENT_FEATURES],
                "NO_MISSINGNESS": [f for f in ROUTING_NUMERIC if f not in MISSING_FEATURES],
                "NO_HISTORICAL_TRADE": [f for f in ROUTING_NUMERIC if f not in HISTORY_FEATURES],
            }
            outer_candidates = candidates.loc[candidates.task.eq(task)&candidates.outcome_year.eq(year)].copy()
            for ab_name, ab_features in ablation_specs.items():
                ab_model, ab_pred, ab_train_pred = _router(train, test, ab_features)
                ab_route = np.where(ab_pred[:,0]<=ab_pred[:,1],"Gravity_PPML","LightGBM")
                ab_risk = np.min(ab_pred, axis=1)
                ab_calibration = _crossfit_gate_predictions(train, ab_features)
                if ab_calibration.empty:
                    raise RuntimeError(f"No chronological ablation calibration pools for {ab_name}")
                train_ab_risk = np.minimum(ab_calibration.crossfit_predicted_regret_ppml.to_numpy(float),
                                           ab_calibration.crossfit_predicted_regret_lightgbm.to_numpy(float))
                accept, thr = _cutoff_accept(train_ab_risk, ab_risk, .8, test.hs2)
                for i, r in enumerate(test.itertuples(index=False)):
                    score_name = "ppml_score" if ab_route[i]=="Gravity_PPML" else "lightgbm_score"
                    gg = outer_candidates.loc[outer_candidates.candidate_pool_id.eq(r.candidate_pool_id)].copy()
                    met = pool_decision_metrics(gg, score_name, task)
                    ablation_rows.append({"task": task,"outcome_year":year,"hs2":r.hs2,
                        "candidate_pool_id":r.candidate_pool_id,"feature_configuration":ab_name,
                        "route_expert":ab_route[i],"predicted_regret":ab_risk[i],"development_threshold_80":thr,
                        "accepted_operational":bool(accept[i]),"regret10":met["regret@10"],"ndcg10":met["ndcg@10"],
                        "selected_positive_usd_top10":met["selected_positive_usd@10"]})
            # Per-pool risk diagnostics on the adaptive route.
            for i, r in enumerate(test.itertuples(index=False)):
                actual_loss = r.regret_Gravity_PPML if r.route_expert=="Gravity_PPML" else r.regret_LightGBM
                risk_rows.append({"task":task,"outcome_year":year,"hs2":r.hs2,"candidate_pool_id":r.candidate_pool_id,
                    "predicted_regret_score":r.predicted_route_regret,"realized_regret10":actual_loss,
                    "route_expert":r.route_expert,"is_probability":False,
                    "interpretation":"out-of-time point estimate of regret, not a calibrated probability"})

    # De-duplicate per-pool decision rows and write core research tables.
    test_candidate = candidates.loc[candidates.is_test.astype(bool)].copy()
    pool_output = test_pools.merge(
        pd.DataFrame(route_rows).drop_duplicates(["task","outcome_year","hs2","policy"], keep="first")
            .loc[lambda x: x.policy.eq("ADAPTIVE_ROUTING"),
                 ["task","outcome_year","hs2","predicted_regret_ppml","predicted_regret_lightgbm","predicted_route_regret"]],
        on=["task","outcome_year","hs2"], how="left", validate="one_to_one")
    pool_output.to_parquet(HERE / "predictions/ergm_pool_predictions.parquet", index=False)
    test_candidate.to_parquet(HERE / "predictions/ergm_candidate_predictions.parquet", index=False)
    pd.DataFrame(static_rows).to_csv(HERE / "results/T1_static_task_routing.csv", index=False, na_rep="")
    pd.DataFrame(blend_rows).to_csv(HERE / "results/T2_static_blending.csv", index=False, na_rep="")
    pd.DataFrame(route_rows).to_csv(HERE / "results/T3_adaptive_routing.csv", index=False, na_rep="")
    pd.DataFrame(freq_rows).to_csv(HERE / "results/T3_expert_selection_frequency.csv", index=False)
    pd.DataFrame(coefficients).to_csv(HERE / "results/T3_routing_feature_effects.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(HERE / "results/T4_reliability_scores.csv", index=False)
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(HERE / "results/T5_coverage_risk.csv", index=False, na_rep="")
    matched_df = pd.DataFrame(matched_rows)
    # Random-deferral benchmark: retain the adaptive router's selected expert and
    # randomly accept a fixed number of complete HS2-year pools. This estimates
    # risk reduction from workload alone, without test-label-based pool selection.
    random_rows = []
    pool_df = pd.DataFrame(matched_pool_rows)
    rng_random = np.random.default_rng(7351)
    if not pool_df.empty:
        source = pool_df.loc[pool_df.policy.eq("ERGM_FULL")].copy()
        for (task, year, coverage), group in source.groupby(["task", "outcome_year", "requested_coverage"]):
            n_accept = int(np.ceil(float(coverage) * len(group)))
            rep = []
            for _ in range(500):
                ix = rng_random.choice(len(group), size=n_accept, replace=False)
                chosen = group.iloc[ix]
                valid = chosen.loc[chosen["regret@10"].notna()]
                rep.append({"mean_regret10_accepted": float(valid["regret@10"].mean()) if len(valid) else np.nan,
                    "mean_ndcg10_accepted": float(valid["ndcg@10"].mean()) if len(valid) else np.nan,
                    "mean_precision10_accepted": float(valid["precision@10"].mean()) if len(valid) else np.nan,
                    "accepted_pools_with_observed_metrics": int(len(valid)),
                    "mean_observed_positive_usd_top10": float(valid["selected_positive_usd@10"].mean()) if len(valid) else np.nan})
            rep_df = pd.DataFrame(rep)
            random_rows.append({"task": task, "outcome_year": int(year), "policy": "RANDOM_DEFERRAL_ADAPTIVE_ROUTE",
                "requested_coverage": float(coverage), "coverage_mode": "matched_test_coverage",
                "automation_coverage": n_accept / len(group), "deferral_rate": 1-n_accept/len(group),
                "all_pools": int(len(group)), "accepted_pools": n_accept,
                "accepted_pools_with_observed_metrics": float(rep_df.accepted_pools_with_observed_metrics.mean()),
                "mean_regret10_accepted": float(rep_df.mean_regret10_accepted.mean()),
                "sd_randomization_mean_regret10": float(rep_df.mean_regret10_accepted.std()),
                "mean_ndcg10_accepted": float(rep_df.mean_ndcg10_accepted.mean()),
                "mean_precision10_accepted": float(rep_df.mean_precision10_accepted.mean()),
                "mean_observed_positive_usd_top10": float(rep_df.mean_observed_positive_usd_top10.mean()),
                "randomizations": 500, "seed": 7351,
                "note": "Pool-level random acceptance; adaptive expert ranking retained; retrospective labels used only for evaluation"})
    if random_rows:
        matched_df = pd.concat([matched_df, pd.DataFrame(random_rows)], ignore_index=True, sort=False)
    matched_df.to_csv(HERE / "results/T5_matched_coverage_comparison.csv", index=False, na_rep="")
    pd.DataFrame(random_rows).to_csv(HERE / "results/T5_random_deferral.csv", index=False, na_rep="")
    pd.DataFrame(operational_decisions).to_csv(HERE / "results/T5_pool_acceptance_decisions.csv", index=False, na_rep="")
    pd.DataFrame(route_explanations).to_csv(HERE / "results/T6_routing_explanations.csv", index=False, na_rep="")
    pd.DataFrame(counterfactuals).to_csv(HERE / "results/T6_counterfactual_switching.csv", index=False, na_rep="")
    pd.DataFrame(ablation_rows).to_csv(HERE / "results/T9_ablation.csv", index=False, na_rep="")
    calibration = _risk_calibration(pd.DataFrame(risk_rows))
    calibration.to_csv(HERE / "results/T4_risk_calibration.csv", index=False, na_rep="")
    highrisk = _high_regret_detection(pd.DataFrame(risk_rows), losses)
    highrisk.to_csv(HERE / "results/T5_high_regret_detection.csv", index=False, na_rep="")
    dossiers, reasons, deferred = _review_dossiers(candidates, test_pools, pd.DataFrame(route_rows), coverage_df)
    dossiers.to_csv(HERE / "results/T7_review_dossiers.csv", index=False, na_rep="")
    reasons.to_csv(HERE / "results/T7_deferral_reason_codes.csv", index=False, na_rep="")
    fidelity = pd.DataFrame([{"task":x["task"],"outcome_year":x["outcome_year"],"hs2":x["hs2"],
        "candidate_pool_id":x["candidate_pool_id"],"raw_gate_score_difference_error":x["route_reconstruction_error"],
        "raw_gate_score_reconstructed":x["route_reconstruction_error"]<1e-8,
        "route_decision_reproduced":x["route_decision_reproduced"],
        "tie_after_nonnegative_clipping":x["both_regret_scores_clipped_to_zero_tie"]} for x in route_explanations])
    fidelity.to_csv(HERE / "results/T7_explanation_fidelity.csv", index=False)
    econ = _economic_summary(coverage_df)
    econ.to_csv(HERE / "results/T8_economic_opportunity.csv", index=False, na_rep="")
    econ.to_csv(HERE / "results/T8_export_value_usd.csv", index=False, na_rep="")
    coverage_df.to_csv(HERE / "results/T8_regret.csv", index=False, na_rep="")
    deferred_summary = coverage_df.groupby(["task","policy","coverage_mode","requested_coverage"],as_index=False).agg(
        all_pools=("all_pools","sum"),accepted_pools=("accepted_pools","sum"),
        mean_regret_diagnostic=("mean_regret10_deferred_diagnostic","mean"),
        unknown_share_accepted=("mean_unknowns_in_top10","mean"))
    deferred_summary["deferred_pools"] = deferred_summary.all_pools-deferred_summary.accepted_pools
    deferred_summary.to_csv(HERE / "results/T5_deferred_pool_characteristics.csv",index=False,na_rep="")
    _missing_sensitivity(test_candidate, pd.DataFrame(route_rows))
    _inference(pd.DataFrame(matched_pool_rows))
    compute_cost = pd.DataFrame([{"component":"routing/evaluation (OOF model training excluded)",
        "wall_seconds":time.perf_counter()-start,
        "historical_oof_candidate_task_rows":int((~candidates.is_test.astype(bool)).sum()),
        "historical_oof_pool_loss_labels":int(len(historical)),
        "outer_candidate_task_rows":int(len(test_candidate)),"max_threads":4,
        "note":"Ridge gates deterministic; historical LightGBM OOF uses seed 11; saved outer scores reuse audited archive means"}])
    compute_cost.to_csv(HERE / "results/computational_cost.csv", index=False)
    return {"candidates":candidates,"pool_losses":losses,"historical":historical,"test_pools":test_pools,
            "coverage":coverage_df,"matched":matched_df,"static":pd.DataFrame(static_rows),
            "blend":pd.DataFrame(blend_rows),"route":pd.DataFrame(route_rows),"frequency":pd.DataFrame(freq_rows),
            "risk":pd.DataFrame(risk_rows),"calibration":calibration,"ablation":pd.DataFrame(ablation_rows),
            "economic":econ,"dossiers":dossiers}


def _risk_calibration(risk: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for task, g in risk.groupby("task"):
        g=g.loc[g.predicted_regret_score.notna()&g.realized_regret10.notna()].copy()
        if g.empty: continue
        g["risk_group"]=pd.qcut(g.predicted_regret_score.rank(method="first"),q=min(5,len(g)),labels=False,duplicates="drop")
        rho=spearmanr(g.predicted_regret_score,g.realized_regret10,nan_policy="omit").statistic
        for b,d in g.groupby("risk_group"):
            rows.append({"task":task,"risk_group_low_to_high":int(b),"pools":len(d),
                "mean_predicted_regret_score":float(d.predicted_regret_score.mean()),
                "mean_realized_regret10":float(d.realized_regret10.mean()),
                "median_realized_regret10":float(d.realized_regret10.median()),"spearman_all_pools":float(rho),
                "calibration_claim":"descriptive; score is not a probability"})
    return pd.DataFrame(rows)


def _high_regret_detection(risk: pd.DataFrame, all_pool_losses: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    from sklearn.metrics import roc_auc_score, average_precision_score
    for (task,year),g in risk.groupby(["task","outcome_year"]):
        h=all_pool_losses.loc[all_pool_losses.task.eq(task)&all_pool_losses.outcome_year.le(int(year)-2)]
        static_loss = h.regret_LightGBM if task=="Task_A_new_entry" else h.regret_Gravity_PPML
        hist_loss=static_loss.dropna()
        cut=float(hist_loss.quantile(.8)) if len(hist_loss) else np.nan
        d=g.loc[g.predicted_regret_score.notna()&g.realized_regret10.notna()].copy()
        d["high_regret_event"]=d.realized_regret10.gt(cut) if np.isfinite(cut) else False
        auc=roc_auc_score(d.high_regret_event,d.predicted_regret_score) if d.high_regret_event.nunique()==2 else np.nan
        ap=average_precision_score(d.high_regret_event,d.predicted_regret_score) if d.high_regret_event.nunique()==2 else np.nan
        rows.append({"task":task,"evaluation_year":year,"development_static_regret_p80_cutoff":cut,
            "evaluation_pools":len(d),"high_regret_events":int(d.high_regret_event.sum()),
            "risk_score_roc_auc":auc,"risk_score_pr_auc":ap,
            "note":"Event threshold is static task-routing regret from outcomes available by y-2; exploratory"})
    return pd.DataFrame(rows)


def _economic_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty: return coverage
    return coverage.groupby(["task","policy","coverage_mode","requested_coverage"],as_index=False).agg(
        automation_coverage=("automation_coverage","mean"),mean_regret10_accepted=("mean_regret10_accepted","mean"),
        mean_ndcg10_accepted=("mean_ndcg10_accepted","mean"),mean_precision10_accepted=("mean_precision10_accepted","mean"),
        observed_positive_usd_top10=("mean_observed_positive_usd_top10","mean"),
        accepted_pools=("accepted_pools","sum"),all_pools=("all_pools","sum"),
        deferred_regret_diagnostic=("mean_regret10_deferred_diagnostic","mean"))


def _review_dossiers(candidates: pd.DataFrame, test_pools: pd.DataFrame,
                     routes: pd.DataFrame, coverage: pd.DataFrame):
    dossiers=[]; reasons=[]
    for task in ["Task_A_new_entry","Task_B_existing_underexploited"]:
        for year in EVAL_YEARS:
            pools=test_pools.loc[test_pools.task.eq(task)&test_pools.outcome_year.eq(year)]
            r=routes.loc[routes.task.eq(task)&routes.outcome_year.eq(year)&routes.policy.eq("ADAPTIVE_ROUTING")].copy()
            # Use exact matched 80% deferral ordering for dossiers; it is a workload
            # illustration, not a calibrated safety claim.
            risk=np.minimum(r.predicted_regret_ppml.to_numpy(float),r.predicted_regret_lightgbm.to_numpy(float))
            n_defer=max(0,len(r)-int(np.ceil(.8*len(r))))
            order=np.lexsort((r.hs2.astype(str).to_numpy(),-risk))
            defer_idx=set(order[:n_defer])
            candidate_year=candidates.loc[candidates.task.eq(task)&candidates.outcome_year.eq(year)]
            for i,rr in enumerate(r.itertuples(index=False)):
                if i not in defer_idx: continue
                pp=_sort_ranks(candidate_year.loc[candidate_year.candidate_pool_id.eq(rr.candidate_pool_id)],"ppml_score")
                ll=_sort_ranks(candidate_year.loc[candidate_year.candidate_pool_id.eq(rr.candidate_pool_id)],"lightgbm_score")
                pp_order=pp.head(10).destination_iso3.astype(str).tolist()
                ll_order=ll.head(10).destination_iso3.astype(str).tolist()
                a=set(pp_order); b=set(ll_order)
                onlyp=sorted(a-b); onlyl=sorted(b-a)
                union_candidates=sorted(a|b)
                context_cols=[c for c in ["destination_iso3","destination_name","historical_export_value",
                    "historical_market_share","log_destination_import_demand"] if c in candidate_year.columns]
                context=(candidate_year.loc[candidate_year.destination_iso3.astype(str).isin(union_candidates),context_cols]
                    .drop_duplicates("destination_iso3").sort_values("destination_iso3").to_dict("records"))
                main=[]
                if rr.predicted_route_regret>0: main.append("predicted_regret_score_ranked_in_deferred_tail")
                if rr.top10_jaccard<.5: main.append("low_expert_top10_overlap")
                if rr.predictor_missing_share>.25: main.append("high_predictor_missingness")
                if not main: main.append("relative_risk_rank_within_deferred_workload")
                dossiers.append({"decision":"DEFER_TO_EXPERT","task":task,"hs2":rr.hs2,"outcome_year":year,
                    "historical_information_cutoff":year-2,"preferred_expert":rr.selected_expert,
                    "alternative_expert":"LightGBM" if rr.selected_expert=="Gravity_PPML" else "Gravity_PPML",
                    "predicted_regret_score":rr.predicted_route_regret,"risk_cutpoint_for_matched_workload":float(np.sort(risk)[-max(1,n_defer)]),
                    "top10_overlap_count":len(a&b),"top10_overlap_share":rr.top10_jaccard,
                    "main_deferral_reasons":"; ".join(main),"ppml_top10":";".join(pp_order),
                    "lightgbm_top10":";".join(ll_order),"ppml_only_candidates":";".join(onlyp),
                    "lightgbm_only_candidates":";".join(onlyl),"human_review_result":"not observed"})
                dossiers[-1]["top10_candidate_context_json"] = json.dumps(context,ensure_ascii=False,default=str)
                for reason in main:
                    reasons.append({"task":task,"hs2":rr.hs2,"outcome_year":year,"reason_code":reason,
                        "predicted_regret_score":rr.predicted_route_regret,"top10_overlap":rr.top10_jaccard,
                        "predictor_missing_share":rr.predictor_missing_share})
    return pd.DataFrame(dossiers),pd.DataFrame(reasons),pd.DataFrame()


def _entry_partial_id_bounds(known_total_success: int, selected_known_success: int,
                             selected_missing: int, unselected_missing: int,
                             k: int = 10) -> dict[str, int]:
    """Sharp binary-entry utility/regret bounds under a common label completion."""
    selected_success=[]; regrets=[]
    for selected_true in range(selected_missing+1):
        for unselected_true in range(unselected_missing+1):
            selected_gain=selected_known_success+selected_true
            oracle_gain=min(k,known_total_success+selected_true+unselected_true)
            selected_success.append(selected_gain)
            regrets.append(max(0,oracle_gain-selected_gain))
    return {"selected_lower":min(selected_success),"selected_upper":max(selected_success),
        "regret_lower":min(regrets),"regret_upper":max(regrets),
        "completion_count_pairs_checked":(selected_missing+1)*(unselected_missing+1)}


def _missing_sensitivity(test_candidates: pd.DataFrame, routes: pd.DataFrame) -> None:
    rows=[]; coverage=[]
    for (task,year,hs2),g in test_candidates.groupby(["task","outcome_year","hs2"],sort=True):
        for expert,score in (("Gravity_PPML","ppml_score"),("LightGBM","lightgbm_score")):
            order=_sort_ranks(g,score).head(10)
            m=int((~order.label_observed.astype(bool)).sum())
            known=int(order.label_observed.astype(bool).sum())
            row={"task":task,"outcome_year":year,"hs2":hs2,"expert":expert,
                "top10_missing_labels":m,"top10_known_candidates":known,
                "binary_entry_selected_success_lower_bound":np.nan,"binary_entry_selected_success_upper_bound":np.nan,
                "all_pool_entry_regret_lower_bound":np.nan,"all_pool_entry_regret_upper_bound":np.nan,
                "usd_outcome_bound":"not identified: no finite verified upper bound"}
            if task=="Task_A_new_entry":
                # Exact bounds by enumerating feasible success counts, not binary
                # assignments. The same completion enters both selected utility
                # and the hindsight oracle, so bounds are sharp under Y in {0,1}.
                full=g.sort_values([score,"destination_iso3"],ascending=[False,True],kind="mergesort")
                selected=full.head(10)
                known=full.loc[full.label_observed.astype(bool)]
                unknown=full.loc[~full.label_observed.astype(bool)]
                known_total=int(known.entry_success.astype(int).sum())
                selected_known=int(selected.loc[selected.label_observed.astype(bool),"entry_success"].astype(int).sum())
                selected_unknown=int((~selected.label_observed.astype(bool)).sum())
                unselected_unknown=int(len(unknown)-selected_unknown)
                bounds=_entry_partial_id_bounds(known_total,selected_known,selected_unknown,unselected_unknown,10)
                row["binary_entry_selected_success_lower_bound"]=bounds["selected_lower"]
                row["binary_entry_selected_success_upper_bound"]=bounds["selected_upper"]
                row["all_pool_entry_regret_lower_bound"]=bounds["regret_lower"]
                row["all_pool_entry_regret_upper_bound"]=bounds["regret_upper"]
                row["missing_completion_count_pairs_checked"]=bounds["completion_count_pairs_checked"]
                row["entry_bound_status"]="sharp exact bound under binary support; joint feasible completion and fixed Top-10 policy"
            rows.append(row)
    out=pd.DataFrame(rows)
    out.to_csv(HERE/"results/T10_missing_outcome_sensitivity.csv",index=False,na_rep="")
    miss=[]
    d=test_candidates.copy()
    d["demand_quartile"]=pd.qcut(pd.to_numeric(d.log_destination_import_demand,errors="coerce"),4,labels=False,duplicates="drop")
    for (task,year),g in d.groupby(["task","outcome_year"]):
        miss.append({"group":"task_year","task":task,"outcome_year":year,"n":len(g),"observed":int(g.label_observed.sum()),
            "unknown":int((~g.label_observed.astype(bool)).sum()),"unknown_share":float((~g.label_observed.astype(bool)).mean())})
    for (task,year,q),g in d.groupby(["task","outcome_year","demand_quartile"],dropna=False):
        miss.append({"group":"task_year_import_demand_quartile","task":task,"outcome_year":year,"quartile":q,"n":len(g),
            "observed":int(g.label_observed.sum()),"unknown":int((~g.label_observed.astype(bool)).sum()),
            "unknown_share":float((~g.label_observed.astype(bool)).mean())})
    pd.DataFrame(miss).to_csv(HERE/"results/T10_missingness_by_task.csv",index=False,na_rep="")


def _inference(matched: pd.DataFrame, reps: int = 2000) -> None:
    # Compare accepted-pool risk at matched coverage. Resampling HS2 blocks keeps
    # repeated years together; accepted candidate pools need not be identical
    # because selective policies intentionally defer different product-years.
    rows=[]
    if matched.empty: return
    d=matched.loc[matched.coverage_mode.eq("matched_test_coverage")&matched.requested_coverage.eq(.8)].copy()
    for task,g in d.groupby("task"):
        accepted=g.loc[g.accepted & g["regret@10"].notna()]
        per_hs=accepted.groupby(["hs2","policy"],as_index=False).agg(
            accepted_pool_mean_regret=("regret@10","mean"),accepted_pools=("candidate_pool_id","nunique"))
        piv=per_hs.pivot(index="hs2",columns="policy",values="accepted_pool_mean_regret")
        count_piv=per_hs.pivot(index="hs2",columns="policy",values="accepted_pools")
        if "ERGM_FULL" not in piv: continue
        comparators=[c for c in ["STATIC_PLUS_RISK","DISAGREEMENT_ONLY","STATIC_CONFIDENCE",
                                  "LIGHTGBM_CONFIDENCE","PPML_CONFIDENCE"] if c in piv]
        for comp in comparators:
            paired=piv[["ERGM_FULL",comp]].dropna()
            if paired.empty: continue
            paired["difference"] = paired.ERGM_FULL-paired[comp]
            byhs=paired.difference
            clusters=byhs.index.to_numpy()
            if len(clusters)<2: continue
            rng=np.random.default_rng(1701)
            boots=np.array([byhs.iloc[rng.integers(0,len(byhs),len(byhs))].mean() for _ in range(reps)])
            diff=float(byhs.mean())
            null=np.array([np.mean(byhs.to_numpy()*rng.choice([-1,1],size=len(byhs))) for _ in range(reps)])
            p=(1+np.sum(np.abs(null)>=abs(diff)))/(reps+1)
            rows.append({"task":task,"metric":"paired_mean_regret10_difference_at_matched_80pct_coverage",
                "contrast":"ERGM_FULL minus "+comp,"difference":diff,"ci95_low":float(np.quantile(boots,.025)),
                "ci95_high":float(np.quantile(boots,.975)),"two_sided_hs2_signflip_p":float(min(1,p)),
                "hs2_blocks":int(len(clusters)),"accepted_pools_ergm":int(count_piv.get("ERGM_FULL",pd.Series(dtype=float)).sum()),
                "accepted_pools_comparator":int(count_piv.get(comp,pd.Series(dtype=float)).sum()),"bootstrap_reps":reps,
                "direction":"negative favors ERGM" if diff<0 else "positive favors comparator",
                "limitation":"Conditional risk among each policy's accepted pools; HS2 blocks retain years. Three inspected years; not future-cycle inference."})
    out=pd.DataFrame(rows)
    if len(out):
        out["holm_adjusted_p"]=_holm(out.two_sided_hs2_signflip_p.to_numpy())
    out.to_csv(HERE/"results/inference.csv",index=False,na_rep="")
    out.to_csv(HERE/"results/holm_adjusted_tests.csv",index=False,na_rep="")


def _holm(pvals: np.ndarray) -> np.ndarray:
    p=np.asarray(pvals,float); n=len(p); order=np.argsort(p); adj=np.empty(n); running=0.
    for rank,idx in enumerate(order):
        running=max(running,(n-rank)*p[idx]); adj[idx]=min(1.,running)
    return adj

