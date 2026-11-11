from __future__ import annotations

import itertools
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import CFG, hs2_chapter_descriptions, hs2_code, path
from experiment1 import run as run_e1
from gravity_ppml import fit_ppml
from market_discovery import add_entry_targets, threshold_counts
from ml_models import MODEL_NAMES, fit_regressor, tune_regressor
from statistics import ndcg_at_k, precision_at_k


def _graded_relevance(increment: pd.Series, threshold: int) -> np.ndarray:
    # Fixed, prespecified five-level scale: orders of magnitude above the primary entry threshold.
    return np.minimum(4, np.floor(np.log10(1 + np.maximum(0, increment.to_numpy(float))/threshold))).astype(int)


def _metrics_for_ranking(d: pd.DataFrame, score_col: str, threshold: int) -> dict:
    d = d.sort_values(score_col, ascending=False, na_position="last")
    binary = d.entry_success.astype(int).to_numpy()
    grade = _graded_relevance(d.realized_incremental_export_usd.fillna(0), threshold)
    result = {"Precision_at_5": precision_at_k(binary, 5), "Precision_at_10": precision_at_k(binary, 10),
              "NDCG_at_5": ndcg_at_k(grade, 5), "NDCG_at_10": ndcg_at_k(grade, 10),
              "entry_successes_top5": int(binary[:5].sum()), "entry_successes_top10": int(binary[:10].sum()),
              "entry_positive_cases": int(binary.sum()), "entry_case_count_sufficient_for_topk": int(binary.sum() >= 10),
              "realized_incremental_exports_top5_usd": float(d.realized_incremental_export_usd.head(5).sum()),
              "realized_incremental_exports_top10_usd": float(d.realized_incremental_export_usd.head(10).sum()),
              "mean_incremental_exports_top10_usd": float(d.realized_incremental_export_usd.head(10).mean()),
              "candidate_markets": len(d), "entry_prevalence": float(binary.mean()) if len(binary) else np.nan}
    return result


def _pool_rankings(eval_year: int, test: pd.DataFrame, e1: pd.DataFrame, threshold: int, random_seeds: list[int]) -> tuple[pd.DataFrame, list[dict]]:
    pool_frames, metric_rows = [], []
    joined = test.copy()
    model_preds = e1.loc[e1.evaluation_year.eq(eval_year)].groupby(["model", "destination_iso3", "hs2"], as_index=False).predicted_usd.mean()
    for kind in ["Gravity_PPML", *MODEL_NAMES]:
        model = model_preds.loc[model_preds.model.eq(kind), ["destination_iso3", "hs2", "predicted_usd"]].rename(columns={"predicted_usd": f"absolute_{kind}"})
        joined = joined.merge(model, on=["destination_iso3", "hs2"], how="left", validate="one_to_one")
    joined["score_gdp"] = joined.log_gdp
    joined["score_import_demand"] = joined.log_destination_import_demand
    joined["score_historical_trend"] = joined.lagged_export_growth
    for kind in ["Gravity_PPML", *MODEL_NAMES]:
        absolute = joined[f"absolute_{kind}"].fillna(0).clip(lower=0)
        joined[f"increment_{kind}"] = np.maximum(0, absolute - joined.historical_export_value.clip(lower=0))
    strategies = {
        "GDP": "score_gdp", "Historical_import_demand": "score_import_demand", "Historical_export_trend": "score_historical_trend",
        "Gravity_PPML_increment": "increment_Gravity_PPML", "RandomForest_increment": "increment_RandomForest",
        "XGBoost_increment": "increment_XGBoost", "LightGBM_increment": "increment_LightGBM",
        "Gravity_PPML_absolute": "absolute_Gravity_PPML", "RandomForest_absolute": "absolute_RandomForest",
        "XGBoost_absolute": "absolute_XGBoost", "LightGBM_absolute": "absolute_LightGBM",
    }
    out_rows = []
    for hs2code, d in joined.groupby("hs2", sort=False):
        d = d.copy()
        for seed in random_seeds:
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(eval_year), int(hs2code)]))
            d[f"random_{seed}"] = rng.random(len(d))
            m = _metrics_for_ranking(d, f"random_{seed}", threshold)
            metric_rows.append({"ranking_model": "Random_selection", "ranking_seed": seed, "evaluation_year": eval_year, "hs2": hs2code, **m})
            ranked = d.sort_values(f"random_{seed}", ascending=False).copy()
            ranked["rank"] = np.arange(1, len(ranked)+1)
            out_rows.extend(_ranking_rows(ranked, f"Random_selection_seed{seed}", "incremental", "random_score", f"random_{seed}", eval_year))
        for strategy, score in strategies.items():
            task = "absolute" if strategy.endswith("_absolute") else "incremental"
            m = _metrics_for_ranking(d, score, threshold)
            metric_rows.append({"ranking_model": strategy, "ranking_seed": 0, "evaluation_year": eval_year, "hs2": hs2code, **m})
            ranked = d.sort_values(score, ascending=False, na_position="last").copy()
            ranked["rank"] = np.arange(1, len(ranked)+1)
            score_kind = "predicted_export_value_usd" if task == "absolute" else "opportunity_score"
            out_rows.extend(_ranking_rows(ranked, strategy, task, score_kind, score, eval_year))
    return pd.DataFrame(out_rows), metric_rows


def _ranking_rows(d: pd.DataFrame, model: str, task: str, output_score: str, score_col: str, year: int) -> list[dict]:
    rows = []
    descriptions = hs2_chapter_descriptions()
    desc = d.hs2.map(lambda x: descriptions[hs2_code(x)])
    source_model = next((kind for kind in ["Gravity_PPML", "RandomForest", "XGBoost", "LightGBM"]
                         if model in {f"{kind}_increment", f"{kind}_absolute"}), None)
    for ix, row in enumerate(d.itertuples(index=False)):
        score = getattr(row, score_col) if score_col.isidentifier() and hasattr(row, score_col) else d.iloc[ix][score_col]
        absolute_col = f"absolute_{source_model}" if source_model else None
        predicted = float(d.iloc[ix][absolute_col]) if absolute_col else np.nan
        rows.append({"hs2": row.hs2, "hs2_description": desc.iloc[ix], "destination_iso3": row.destination_iso3,
            "destination_name": row.destination_name, "evaluation_year": year, "ranking_model": model,
            "ranking_task": task, "rank": int(row.rank), "opportunity_score": float(d.iloc[ix][score_col]) if task == "incremental" else np.nan,
            "predicted_export_value_usd": predicted if np.isfinite(predicted) else float(d.iloc[ix][score_col]) if task == "absolute" else np.nan,
            "historical_export_value_usd": float(row.historical_export_value), "historical_import_demand_usd": float(row.destination_import_demand),
            "historical_vietnam_market_share": float(row.historical_market_share) if pd.notna(row.historical_market_share) else np.nan,
            "observed_future_export_value_usd": float(row.export_value_usd), "realized_incremental_export_usd": float(row.realized_incremental_export_usd),
            "entry_success": bool(row.entry_success) if pd.notna(row.entry_success) else np.nan,
            "eligibility_status": "positive lagged HS2 import demand; new-entry relevance requires verified inactivity at cutoff"})
    return rows


def _ranking_stability(rankings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = rankings.loc[rankings.ranking_task.eq("incremental") & rankings.evaluation_year.isin(CFG["study"]["evaluation_years"])]
    for (model, hs2code), group in d.groupby(["ranking_model", "hs2"], sort=False):
        years = sorted(group.evaluation_year.unique())
        for y1, y2 in itertools.combinations(years, 2):
            a = group.loc[group.evaluation_year.eq(y1), ["destination_iso3", "rank"]].rename(columns={"rank": "rank1"})
            b = group.loc[group.evaluation_year.eq(y2), ["destination_iso3", "rank"]].rename(columns={"rank": "rank2"})
            pair = a.merge(b, on="destination_iso3")
            if len(pair) >= 3:
                rho = spearmanr(pair.rank1, pair.rank2).statistic
                top1, top2 = set(a.nsmallest(10, "rank1").destination_iso3), set(b.nsmallest(10, "rank2").destination_iso3)
                jaccard = len(top1 & top2) / len(top1 | top2) if top1 | top2 else np.nan
            else:
                rho, jaccard = np.nan, np.nan
            rows.append({"ranking_model": model, "hs2": hs2code, "year_1": y1, "year_2": y2,
                         "paired_destinations": len(pair), "spearman_rank_correlation": rho, "top10_jaccard": jaccard})
    return pd.DataFrame(rows)


def _prospective(panel: pd.DataFrame, lag: int, threshold: int, quick: bool = False) -> pd.DataFrame:
    # A primary-specification prospective period uses the same two-year lag (2024 inputs imply 2026).
    future_year = CFG["study"]["last_year"] + lag
    cand = panel.loc[panel.outcome_year.eq(future_year)].copy()
    valid = panel.loc[panel.outcome_observed & panel.export_value_usd.notna() & panel.outcome_year.le(future_year-lag)].copy()
    if cand.empty or valid.empty:
        return pd.DataFrame()
    joined = cand.copy()
    joined["score_gdp"] = joined.log_gdp
    joined["score_import_demand"] = joined.log_destination_import_demand
    joined["score_historical_trend"] = joined.lagged_export_growth
    for kind in ["Gravity_PPML", *MODEL_NAMES]:
        if kind == "Gravity_PPML":
            pred, result, diag, _ = fit_ppml(valid, cand)
            joblib.dump({"result": result, "diagnostics": diag}, path("results_dir") / "models" / f"ppml_prospective_lag{lag}.joblib")
        else:
            params, _ = tune_regressor(kind, valid, lag)
            pred_parts = []
            seeds = [CFG["study"]["seeds"][0]] if quick else CFG["study"]["seeds"]
            for seed in seeds:
                bundle = fit_regressor(kind, valid, params, int(seed), lag)
                pred_parts.append(bundle.predict(cand))
            pred = np.mean(pred_parts, axis=0)
            joblib.dump({"params": params, "seeds": seeds, "status": "fitted through 2023; predictions are prospective and unvalidated"},
                path("results_dir") / "models" / f"{kind.lower()}_prospective_lag{lag}.joblib")
        joined[f"absolute_{kind}"] = np.maximum(0, pred)
        joined[f"increment_{kind}"] = np.maximum(0, joined[f"absolute_{kind}"] - joined.historical_export_value.clip(lower=0))
    strategies = {"GDP": "score_gdp", "Historical_import_demand": "score_import_demand", "Historical_export_trend": "score_historical_trend",
        "Gravity_PPML_increment": "increment_Gravity_PPML", "RandomForest_increment": "increment_RandomForest", "XGBoost_increment": "increment_XGBoost",
        "LightGBM_increment": "increment_LightGBM", "Gravity_PPML_absolute": "absolute_Gravity_PPML", "RandomForest_absolute": "absolute_RandomForest",
        "XGBoost_absolute": "absolute_XGBoost", "LightGBM_absolute": "absolute_LightGBM"}
    rows=[]
    descriptions = hs2_chapter_descriptions()
    for model, col in strategies.items():
        task = "absolute" if model.endswith("_absolute") else "incremental"
        d = joined.sort_values(col, ascending=False, na_position="last").copy()
        for ix, row in enumerate(d.itertuples(index=False), 1):
            abs_name = model.replace("_increment", "").replace("_absolute", "")
            pred_col = f"absolute_{abs_name}" if f"absolute_{abs_name}" in d else None
            predicted = float(getattr(row, pred_col)) if pred_col and hasattr(row, pred_col) else np.nan
            rows.append({"hs2": row.hs2, "hs2_description": descriptions[hs2_code(row.hs2)], "destination_iso3": row.destination_iso3,
                "destination_name": row.destination_name, "evaluation_year": future_year, "ranking_model": model,
                "ranking_task": task, "rank": ix, "opportunity_score": float(getattr(row,col)) if task=="incremental" else np.nan,
                "predicted_export_value_usd": predicted, "historical_export_value_usd": float(row.historical_export_value),
                "historical_import_demand_usd": float(row.destination_import_demand), "historical_vietnam_market_share": float(row.historical_market_share),
                "observed_future_export_value_usd": np.nan, "realized_incremental_export_usd": np.nan, "entry_success": np.nan,
                "eligibility_status": f"positive {future_year-lag} HS2 import demand; prospective {future_year} rank not externally validated"})
    return pd.DataFrame(rows)


def run(lag: int | None = None, quick: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    lag = int(lag or CFG["study"]["primary_lag"])
    panel = pd.read_parquet(path("processed_dir") / f"panel_lag{lag}.parquet")
    e1_path = path("results_dir") / "predictions" / f"E1_predictions_lag{lag}.parquet"
    if not e1_path.exists():
        raise FileNotFoundError("Run Experiment 1 before Experiment 3")
    e1 = pd.read_parquet(e1_path)
    counts = threshold_counts(panel, lag)
    threshold = int(counts.loc[counts.primary_selected, "threshold_usd"].iloc[0])
    tagged = add_entry_targets(panel, lag, threshold)
    ranking_frames, metrics = [], []
    for eval_year in CFG["study"]["evaluation_years"]:
        test = tagged.loc[tagged.outcome_year.eq(eval_year) & tagged.outcome_observed & tagged.export_value_usd.notna()].copy()
        test = test.loc[test.destination_import_demand.gt(0)]
        # For E3, entry success means a verified historically inactive relationship
        # crossed the threshold. Existing exporters are not labeled as market entries;
        # their realized expansion remains evaluated separately via incremental exports.
        test["entry_success"] = (test.entry_eligible & test.export_value_usd.ge(threshold)).astype("int8")
        ranks, ms = _pool_rankings(int(eval_year), test, e1, threshold, CFG["study"]["seeds"])
        ranking_frames.append(ranks)
        metrics.extend(ms)
    historical = pd.concat(ranking_frames, ignore_index=True) if ranking_frames else pd.DataFrame()
    future = _prospective(panel, lag, threshold, quick=quick)
    all_rankings = pd.concat([historical, future], ignore_index=True) if not future.empty else historical
    all_rankings.to_csv(path("results_dir") / "E3_market_ranking.csv", index=False)
    all_rankings.loc[all_rankings["rank"].le(5)].to_csv(path("results_dir") / "E3_top5_markets.csv", index=False)
    all_rankings.loc[all_rankings["rank"].le(10)].to_csv(path("results_dir") / "E3_top10_markets.csv", index=False)
    future.to_csv(path("results_dir") / "E3_latest_prospective_rankings.csv", index=False)
    raw_metrics = pd.DataFrame(metrics)
    # Random results are summarized across repeated seeds. All model scores are macro-averaged across HS2–year pools.
    metrics_by = raw_metrics.groupby(["ranking_model", "evaluation_year", "hs2"], as_index=False).mean(numeric_only=True)
    metrics_by.to_csv(path("results_dir") / "E3_ranking_by_hs2.csv", index=False)
    macro = metrics_by.groupby("ranking_model", as_index=False).agg(
        macro_precision_at_5=("Precision_at_5", "mean"), macro_precision_at_10=("Precision_at_10", "mean"),
        macro_ndcg_at_5=("NDCG_at_5", "mean"), macro_ndcg_at_10=("NDCG_at_10", "mean"),
        macro_realized_increment_top10_usd=("realized_incremental_exports_top10_usd", "mean"),
        macro_entry_successes_top10=("entry_successes_top10", "mean"), pools=("hs2", "size"))
    ranking_rows = historical.loc[historical.ranking_task.eq("incremental")]
    stability = _ranking_stability(historical)
    stability.to_csv(path("results_dir") / "tables" / "E3_ranking_stability.csv", index=False)
    overlap = []
    for (year, hs2code), d in ranking_rows.groupby(["evaluation_year", "hs2"], sort=False):
        tops = {model: set(g.nsmallest(10, "rank").destination_iso3) for model, g in d.groupby("ranking_model")}
        for a,b in itertools.combinations(sorted(tops),2):
            union=tops[a]|tops[b]
            overlap.append({"evaluation_year":year,"hs2":hs2code,"model_a":a,"model_b":b,"top10_jaccard":len(tops[a]&tops[b])/len(union) if union else np.nan})
    pd.DataFrame(overlap).to_csv(path("results_dir") / "tables" / "E3_ranking_overlap.csv", index=False)
    return macro, all_rankings


if __name__ == "__main__":
    run()
