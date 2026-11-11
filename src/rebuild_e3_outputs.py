"""Rebuild E3 derived tables from saved full rankings without refitting models."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import CFG, path
from market_ranking import _ranking_stability
from statistics import ndcg_at_k, precision_at_k


def rebuild() -> pd.DataFrame:
    ranking_path = path("results_dir") / "E3_market_ranking.csv"
    if not ranking_path.exists():
        raise FileNotFoundError(ranking_path)
    d = pd.read_csv(ranking_path)
    d["rank"] = pd.to_numeric(d["rank"], errors="coerce")
    # Random selection has no model-based forecast attached to it.
    d.loc[d.ranking_model.str.startswith("Random_selection", na=False), "predicted_export_value_usd"] = np.nan
    d["eligibility_status"] = (
        "positive lagged HS2 import demand; new-entry relevance requires verified inactivity at cutoff"
    )
    d.to_csv(ranking_path, index=False)
    d.loc[d["rank"].le(5)].to_csv(path("results_dir") / "E3_top5_markets.csv", index=False)
    d.loc[d["rank"].le(10)].to_csv(path("results_dir") / "E3_top10_markets.csv", index=False)

    future_year = int(CFG["study"]["last_year"]) + int(CFG["study"]["primary_lag"])
    future = d.loc[d.evaluation_year.eq(future_year)].copy()
    if not future.empty:
        future["eligibility_status"] = (
            f"positive {future_year-int(CFG['study']['primary_lag'])} HS2 import demand; prospective {future_year} rank not externally validated"
        )
    future.to_csv(path("results_dir") / "E3_latest_prospective_rankings.csv", index=False)

    historical = d.loc[d.evaluation_year.isin(CFG["study"]["evaluation_years"])].copy()
    threshold = int(CFG["study"]["primary_entry_threshold_usd"])
    rows = []
    for (model, year, hs2), pool in historical.groupby(["ranking_model", "evaluation_year", "hs2"], sort=False):
        pool = pool.sort_values("rank")
        binary = pool.entry_success.fillna(0).astype(int).to_numpy()
        increments = np.maximum(0, pool.realized_incremental_export_usd.fillna(0).to_numpy(float))
        grade = np.minimum(4, np.floor(np.log10(1 + increments / threshold))).astype(int)
        seed_match = re.search(r"_seed(\d+)$", str(model))
        rows.append({
            "ranking_model": "Random_selection" if seed_match else model,
            "ranking_seed": int(seed_match.group(1)) if seed_match else 0,
            "evaluation_year": int(year), "hs2": int(hs2),
            "ranking_task": str(pool.ranking_task.iloc[0]),
            "Precision_at_5": precision_at_k(binary, 5),
            "Precision_at_10": precision_at_k(binary, 10),
            "NDCG_at_5": ndcg_at_k(grade, 5), "NDCG_at_10": ndcg_at_k(grade, 10),
            "entry_successes_top5": int(binary[:5].sum()), "entry_successes_top10": int(binary[:10].sum()),
            "entry_positive_cases": int(binary.sum()),
            "entry_case_count_sufficient_for_topk": int(binary.sum() >= 10),
            "realized_incremental_exports_top5_usd": float(increments[:5].sum()),
            "realized_incremental_exports_top10_usd": float(increments[:10].sum()),
            "mean_incremental_exports_top10_usd": float(increments[:10].mean()) if len(increments) else np.nan,
            "candidate_markets": int(len(pool)),
            "entry_prevalence": float(binary.mean()) if len(binary) else np.nan,
        })
    by_hs2 = pd.DataFrame(rows)
    # Random baseline metrics are averaged across the five predeclared fixed seeds.
    numeric = [c for c in by_hs2.select_dtypes(include=[np.number]).columns if c not in ["evaluation_year", "hs2", "ranking_seed"]]
    by_hs2 = by_hs2.groupby(["ranking_model", "evaluation_year", "hs2", "ranking_task"], as_index=False).agg(
        **{c: (c, "mean") for c in numeric},
        ranking_seed_sd=("Precision_at_10", "std"),
        random_seed_count=("ranking_seed", "nunique"),
    )
    by_hs2.to_csv(path("results_dir") / "E3_ranking_by_hs2.csv", index=False)

    # Preserve rank stability by individual random seed; product-year pools stay paired.
    _ranking_stability(historical).to_csv(path("results_dir") / "tables" / "E3_ranking_stability.csv", index=False)
    overlap = []
    rank_rows = historical.loc[historical.ranking_task.eq("incremental")]
    for (year, hs2), pool in rank_rows.groupby(["evaluation_year", "hs2"], sort=False):
        tops = {model: set(g.nsmallest(10, "rank").destination_iso3)
                for model, g in pool.groupby("ranking_model", sort=False)}
        models = sorted(tops)
        for i, model_a in enumerate(models):
            for model_b in models[i+1:]:
                union = tops[model_a] | tops[model_b]
                overlap.append({"evaluation_year": int(year), "hs2": int(hs2), "model_a": model_a,
                    "model_b": model_b, "top10_jaccard": len(tops[model_a] & tops[model_b]) / len(union) if union else np.nan})
    pd.DataFrame(overlap).to_csv(path("results_dir") / "tables" / "E3_ranking_overlap.csv", index=False)
    return by_hs2


if __name__ == "__main__":
    result = rebuild()
    print(f"Rebuilt E3 outputs from saved ranking rows: {len(result):,} HS2-year-model metrics")
