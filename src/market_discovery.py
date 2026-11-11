from __future__ import annotations

import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import CFG, path
from ml_models import MODEL_NAMES, fit_classifier, logistic_model, tune_classifier
from statistics import ndcg_at_k, precision_at_k
from temporal_validation import split_outer


def add_entry_targets(panel: pd.DataFrame, lag: int, threshold: int) -> pd.DataFrame:
    trade = pd.read_parquet(path("interim_dir") / "baci_hs2_trade.parquet")
    cmap = pd.read_parquet(path("processed_dir") / "country_mapping.parquet")
    trade = trade.merge(cmap[["baci_id", "iso3"]], left_on="baci_importer_id", right_on="baci_id", how="left", validate="many_to_one")
    trade = trade.loc[trade.iso3.str.fullmatch(r"[A-Z]{3}", na=False) & trade.iso3.ne("VNM")]
    result = panel.copy()
    coverage = trade[["iso3", "year"]].drop_duplicates().assign(importer_year_covered=True)
    history_values = []
    for offset in [0, 1, 2]:
        keys = result[["destination_iso3", "hs2", "feature_year"]].copy()
        keys["history_year"] = keys.feature_year - offset
        flow = trade[["iso3", "hs2", "year", "vietnam_exports_usd"]].rename(columns={"year": "history_year"})
        keys = keys.merge(flow, left_on=["destination_iso3", "hs2", "history_year"], right_on=["iso3", "hs2", "history_year"], how="left", validate="many_to_one")
        keys = keys.merge(coverage.rename(columns={"year":"history_year"}), left_on=["destination_iso3","history_year"], right_on=["iso3","history_year"], how="left", validate="many_to_one", suffixes=("","_coverage"))
        covered = keys.importer_year_covered.astype("boolean").fillna(False).to_numpy(dtype=bool)
        result[f"_history_covered_{offset}"] = covered
        values = keys.vietnam_exports_usd.where(keys.vietnam_exports_usd.notna(), 0.0).where(covered, np.nan)
        history_values.append(values.to_numpy(float))
    result["entry_history_verified"] = result[["_history_covered_0","_history_covered_1","_history_covered_2"]].all(axis=1)
    inactive = result.entry_history_verified.to_numpy() & np.isfinite(np.column_stack(history_values)).all(axis=1) & (np.nanmax(np.column_stack(history_values),axis=1) < threshold)
    result["entry_inactive"] = inactive
    result["entry_success"] = np.where(result.outcome_observed & result.export_value_usd.notna(),
                                        result.export_value_usd.ge(threshold), np.nan)
    result["entry_eligible"] = result.entry_history_verified & result.entry_inactive & result.outcome_observed & result.export_value_usd.notna()
    return result


def threshold_counts(panel: pd.DataFrame, lag: int) -> pd.DataFrame:
    rows = []
    # Threshold selection uses the development period only: outcomes no later than 2019, before any outer test year.
    development_end = min(2019, int(CFG["study"]["evaluation_years"][0]) - lag)
    for threshold in CFG["study"]["entry_thresholds_usd"]:
        tagged = add_entry_targets(panel.loc[panel.outcome_year.le(development_end)], lag, int(threshold))
        candidates = tagged.loc[tagged.entry_eligible]
        rows.append({"threshold_usd": int(threshold), "development_end_year": development_end,
                     "eligible_cases": int(len(candidates)), "positive_events": int(candidates.entry_success.sum()),
                     "prevalence": float(candidates.entry_success.mean()) if len(candidates) else np.nan})
    counts = pd.DataFrame(rows)
    preferred = int(CFG["study"]["primary_entry_threshold_usd"])
    preferred_row = counts.loc[counts.threshold_usd.eq(preferred)]
    if preferred_row.empty or int(preferred_row.positive_events.iloc[0]) < int(CFG["study"]["min_entry_events"]):
        viable = counts.loc[counts.positive_events.ge(int(CFG["study"]["min_entry_events"]))]
        if not viable.empty:
            preferred = int(viable.iloc[0].threshold_usd)
    counts["primary_selected"] = counts.threshold_usd.eq(preferred)
    counts["selection_rule"] = f"Prefer configured threshold; if fewer than {CFG['study']['min_entry_events']} development events, choose the lowest threshold with sufficient events."
    # Report held-out case and event counts for every threshold without using them for threshold selection.
    holdout_rows=[]
    holdout_panel=panel.loc[panel.outcome_year.isin(CFG["study"]["evaluation_years"])].copy()
    for threshold in CFG["study"]["entry_thresholds_usd"]:
        held=add_entry_targets(holdout_panel,lag,int(threshold))
        for year,d in held.groupby("outcome_year"):
            eligible=d.loc[d.entry_eligible]
            holdout_rows.append({"threshold_usd":int(threshold),"evaluation_year":int(year),"test_eligible_cases":len(eligible),
                "test_positive_events":int(eligible.entry_success.sum()),"test_prevalence":float(eligible.entry_success.mean()) if len(eligible) else np.nan})
    if holdout_rows:
        test_counts=pd.DataFrame(holdout_rows)
        counts=counts.merge(test_counts,on="threshold_usd",how="left",validate="one_to_many")
    suffix = "" if lag == int(CFG["study"]["primary_lag"]) else f"_lag{lag}"
    counts.to_csv(path("results_dir") / "tables" / f"E2_threshold_selection{suffix}.csv", index=False)
    return counts


def _pool_metrics(d: pd.DataFrame, score_col: str, label_col: str = "entry_success") -> dict:
    d = d.loc[d[label_col].notna()].copy()
    if d.empty:
        return {"positive_prevalence": np.nan, "PR_AUC": np.nan, "ROC_AUC": np.nan, "Precision_at_5": np.nan,
                "Precision_at_10": np.nan, "Recall_at_10": np.nan, "NDCG_at_10": np.nan, "entry_success_rate_top10": np.nan}
    y = d[label_col].astype(int).to_numpy()
    score = d[score_col].to_numpy(float)
    prevalence = float(y.mean())
    pr = average_precision_score(y, score) if y.sum() and np.unique(y).size > 1 else np.nan
    roc = roc_auc_score(y, score) if np.unique(y).size > 1 else np.nan
    return {"positive_prevalence": prevalence, "PR_AUC": pr, "ROC_AUC": roc,
            "Precision_at_5": precision_at_k(np.asarray(d.sort_values(score_col, ascending=False)[label_col], dtype=float), 5),
            "Precision_at_10": precision_at_k(np.asarray(d.sort_values(score_col, ascending=False)[label_col], dtype=float), 10),
            "Recall_at_10": float(d.sort_values(score_col, ascending=False)[label_col].head(10).sum()/y.sum()) if y.sum() else np.nan,
            "NDCG_at_10": ndcg_at_k(np.asarray(d.sort_values(score_col, ascending=False)[label_col], dtype=float), 10),
            "entry_success_rate_top10": float(d.sort_values(score_col, ascending=False)[label_col].head(10).mean())}


def run(lag: int | None = None, quick: bool = False, threshold_override: int | None = None) -> pd.DataFrame:
    lag = int(lag or CFG["study"]["primary_lag"])
    panel = pd.read_parquet(path("processed_dir") / f"panel_lag{lag}.parquet")
    counts = threshold_counts(panel, lag)
    selected_threshold = int(counts.loc[counts.primary_selected, "threshold_usd"].iloc[0])
    threshold = int(threshold_override) if threshold_override is not None else selected_threshold
    if threshold not in set(counts.threshold_usd.astype(int)):
        raise ValueError(f"Unsupported entry threshold {threshold}; expected one of {sorted(counts.threshold_usd.astype(int).unique())}")
    threshold_suffix = f"_threshold{threshold}" if threshold_override is not None else ""
    tagged = add_entry_targets(panel, lag, threshold)
    predictions, metric_rows, failures = [], [], []
    seeds = [CFG["study"]["seeds"][0]] if quick or CFG["compute"].get("quick_mode") else CFG["study"]["seeds"]
    for eval_year in CFG["study"]["evaluation_years"]:
        train, test = split_outer(tagged, int(eval_year), lag)
        train = train.loc[train.entry_eligible].copy()
        test = test.loc[test.entry_eligible].copy()
        if len(train) < 50 or train.entry_success.nunique() < 2 or test.empty or test.entry_success.nunique() < 2:
            failures.append({"evaluation_year": eval_year, "error": "insufficient verified eligible entries or positive class"})
            continue
        try:
            test["score_history_frequency"] = train.groupby("hs2").entry_success.mean().reindex(test.hs2).to_numpy()
            test["score_history_frequency"] = test.score_history_frequency.fillna(train.entry_success.mean())
            test["score_import_demand"] = test.log_destination_import_demand
            e1_file = path("results_dir") / "predictions" / f"E1_predictions_lag{lag}.parquet"
            e1 = pd.read_parquet(e1_file)
            e1 = e1.loc[e1.evaluation_year.eq(eval_year) & e1.seed.eq(0)]
            ppml = e1.loc[e1.model.eq("Gravity_PPML"), ["destination_iso3", "hs2", "predicted_usd"]].rename(columns={"predicted_usd": "score_gravity_opportunity"})
            test = test.merge(ppml, on=["destination_iso3", "hs2"], how="left", validate="one_to_one")
            test["score_gravity_opportunity"] = np.log1p(test.score_gravity_opportunity.fillna(0))
            base_methods = {"Historical_entry_frequency": "score_history_frequency", "World_import_demand": "score_import_demand", "Gravity_PPML": "score_gravity_opportunity"}
            for method, col in base_methods.items():
                for r in test[["destination_iso3", "hs2", "outcome_year", "entry_success", col]].itertuples(index=False):
                    predictions.append({"model": method, "seed": 0, "evaluation_year": int(eval_year), "destination_iso3": r.destination_iso3,
                        "hs2": r.hs2, "entry_success": r.entry_success, "score": float(r[-1]), "threshold_usd": threshold})
            try:
                scores, _, _ = logistic_model(train, "entry_success", test)
                for i, value in enumerate(scores):
                    predictions.append({"model": "LogisticRegression", "seed": 0, "evaluation_year": int(eval_year), "destination_iso3": test.destination_iso3.iloc[i], "hs2": test.hs2.iloc[i], "entry_success": test.entry_success.iloc[i], "score": float(value), "threshold_usd": threshold})
            except Exception:
                failures.append({"evaluation_year": eval_year, "model": "LogisticRegression", "error": traceback.format_exc()})
            for kind in MODEL_NAMES:
                try:
                    params, val_ap = tune_classifier(kind, train, "entry_success", lag)
                    for seed in seeds:
                        model = fit_classifier(kind, train, "entry_success", params, int(seed))
                        x = model.preprocessor.transform(test[model.numeric_features + model.categorical_features]).astype(np.float32)
                        prob = model.estimator.predict_proba(x)[:, 1]
                        joblib.dump({"bundle": model, "params": params, "lag": lag, "evaluation_year": eval_year},
                            path("results_dir") / "models" / f"e2_{kind.lower()}_lag{lag}_{eval_year}_seed{seed}{threshold_suffix}.joblib")
                        for i, value in enumerate(prob):
                            predictions.append({"model": kind, "seed": seed, "evaluation_year": int(eval_year), "destination_iso3": test.destination_iso3.iloc[i],
                                "hs2": test.hs2.iloc[i], "entry_success": test.entry_success.iloc[i], "score": float(value), "threshold_usd": threshold,
                                "inner_validation_pr_auc": val_ap})
                except Exception:
                    failures.append({"evaluation_year": eval_year, "model": kind, "error": traceback.format_exc()})
        except Exception:
            failures.append({"evaluation_year": eval_year, "model": "baseline/setup", "error": traceback.format_exc()})
    pred = pd.DataFrame(predictions)
    if pred.empty:
        raise RuntimeError(f"No market discovery predictions were produced: {failures}")
    pred["seed"] = pd.to_numeric(pred.seed, errors="coerce").fillna(0).astype(int)
    pred.to_parquet(path("results_dir") / "predictions" / f"E2_entry_predictions{threshold_suffix}.parquet", index=False)
    # Calculate metrics on unique candidate pools within each seed, then average
    # seeds. Repeated predictions are never counted as additional markets.
    seed_pool_rows = []
    for (model, year, hs2code, seed), d in pred.groupby(["model", "evaluation_year", "hs2", "seed"], sort=False):
        m = _pool_metrics(d, "score")
        unique = d.drop_duplicates(["destination_iso3", "hs2"])
        seed_pool_rows.append({"model": model, "evaluation_year": year, "hs2": hs2code, "seed": seed, **m,
            "candidate_cases": len(unique), "positive_events": int(unique.entry_success.sum()),
            "eligible_for_metric": len(unique) >= 10})
    seed_pools = pd.DataFrame(seed_pool_rows)
    metric_cols = ["positive_prevalence", "PR_AUC", "ROC_AUC", "Precision_at_5", "Precision_at_10", "Recall_at_10", "NDCG_at_10", "entry_success_rate_top10"]
    by_hs2 = seed_pools.groupby(["model", "evaluation_year", "hs2"], as_index=False).agg(
        **{c: (c, "mean") for c in metric_cols},
        **{f"{c}_seed_sd": (c, "std") for c in metric_cols},
        candidate_cases=("candidate_cases", "first"), positive_events=("positive_events", "first"),
        eligible_for_metric=("eligible_for_metric", "first"), seeds=("seed", "nunique"))
    by_hs2["note"] = "Metrics within unique HS2 × year pools; mean across model seeds"
    suffix = "" if lag == int(CFG["study"]["primary_lag"]) else f"_lag{lag}"
    by_hs2.to_csv(path("results_dir") / f"E2_discovery_by_hs2{suffix}{threshold_suffix}.csv", index=False)
    overall_rows = []
    for (model, year), d in pred.groupby(["model", "evaluation_year"], sort=False):
        seed_metrics = pd.DataFrame([{"seed": seed, **_pool_metrics(sd, "score")}
                                     for seed, sd in d.groupby("seed", sort=False)])
        global_metrics = {c: seed_metrics[c].mean() for c in metric_cols}
        global_metrics.update({f"{c}_seed_sd": seed_metrics[c].std() for c in metric_cols})
        # Top-K ranking is defined only within product-year pools, never over a mixed global list.
        for metric in ["Precision_at_5", "Precision_at_10", "Recall_at_10", "NDCG_at_10", "entry_success_rate_top10"]:
            global_metrics[metric] = np.nan
        row = {"model": model, "evaluation_year": year, **global_metrics,
               "macro_precision_at_10": by_hs2.loc[(by_hs2.model.eq(model)) & (by_hs2.evaluation_year.eq(year)), "Precision_at_10"].mean(),
               "macro_precision_at_10_seed_sd": by_hs2.loc[(by_hs2.model.eq(model)) & (by_hs2.evaluation_year.eq(year)), "Precision_at_10_seed_sd"].mean(),
               "macro_ndcg_at_10": by_hs2.loc[(by_hs2.model.eq(model)) & (by_hs2.evaluation_year.eq(year)), "NDCG_at_10"].mean(),
               "macro_ndcg_at_10_seed_sd": by_hs2.loc[(by_hs2.model.eq(model)) & (by_hs2.evaluation_year.eq(year)), "NDCG_at_10_seed_sd"].mean(),
               "macro_PR_AUC":by_hs2.loc[(by_hs2.model.eq(model))&(by_hs2.evaluation_year.eq(year)),"PR_AUC"].mean(),
               "macro_ROC_AUC":by_hs2.loc[(by_hs2.model.eq(model))&(by_hs2.evaluation_year.eq(year)),"ROC_AUC"].mean(),
               "threshold_usd": threshold,
               "candidate_cases": int(d[["destination_iso3", "hs2"]].drop_duplicates().shape[0]),
               "positive_events": int(d.drop_duplicates(["destination_iso3", "hs2"]).entry_success.sum()),
               "seeds": int(d.seed.nunique())}
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(path("results_dir") / f"E2_market_discovery{suffix}{threshold_suffix}.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(path("results_dir") / "tables" / f"E2_failures_lag{lag}.csv", index=False)
    return overall


if __name__ == "__main__":
    run()
