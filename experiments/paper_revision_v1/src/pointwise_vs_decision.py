"""Do pointwise accuracy metrics pick the same model as shortlist (decision) metrics? 2018-2024.

Pointwise, as in most of the trade-forecasting literature: AUC, average precision and Precision@1000 over all
candidates of a year (pooled across chapters); RMSE/MAE of log1p predicted vs realised positive increments.
Decision: NDCG@10 and Hit@10 within each HS2-year pool, plus the within-pool AUC averaged over pools.
The hybrid is a within-pool construct; for pooled metrics its normalised fused score in [0, 1] is used.
Writes results/pointwise_vs_decision.csv and results/pointwise_vs_decision_ranks.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from revision_lib import (KEY, PRED_DIR, RES_DIR, TA, TB, add_rules_and_hybrid, all_pool_metrics, load_stored_candidates,
                          zfill_hs2)

MODELS = {"PPML": "ppml_score", "LightGBM": "lightgbm_score", "Hybrid": "fused_score", "ImportDemand": "import_demand_score"}


def _safe(fn, y, s):
    try:
        return float(fn(y, s)) if len(np.unique(y)) > 1 else np.nan
    except ValueError:
        return np.nan


def main() -> pd.DataFrame:
    d = load_stored_candidates().drop(columns="lightgbm_score")
    refit = zfill_hs2(pd.read_parquet(PRED_DIR / "lgbm_refit_scores.parquet"))[KEY + ["lightgbm_score"]]
    d = add_rules_and_hybrid(d.merge(refit, on=KEY, how="left", validate="one_to_one"))
    d = d[d.label_observed.astype(bool)]
    rows = []
    for task in (TA, TB):
        t = d[d.task == task].copy()
        if task == TA:
            t["y"] = t.entry_success.astype(float)
        else:
            t["inc"] = np.maximum(0, t.realized_incremental_export_usd.astype(float))
            t["y"] = (t.inc >= 50_000).astype(float)
        for name, col in MODELS.items():
            per_year = []
            for yr, g in t.groupby("outcome_year"):
                s = g[col].astype(float).fillna(g[col].min())
                top = g.assign(_s=s).sort_values("_s", ascending=False).head(1000)
                rec = {"auc_pooled": _safe(roc_auc_score, g.y, s), "ap_pooled": _safe(average_precision_score, g.y, s),
                       "precision_at_1000_pooled": float(top.y.mean())}
                if task == TB:
                    rec["spearman_increment_pooled"] = float(spearmanr(s, g.inc).correlation)
                    if name in ("PPML", "LightGBM"):
                        e = np.log1p(np.maximum(0, s)) - np.log1p(g.inc)
                        rec["rmse_log_increment"] = float(np.sqrt(np.mean(e ** 2)))
                        rec["mae_log_increment"] = float(np.mean(np.abs(e)))
                within = [_safe(roc_auc_score, p.y, p[col].astype(float).fillna(p[col].min())) for _, p in g.groupby("hs2")]
                rec["auc_within_pool_mean"] = float(np.nanmean(within))
                per_year.append(rec)
            row = {"task": task, "model": name, **pd.DataFrame(per_year).mean().to_dict()}
            rows.append(row)
    point = pd.DataFrame(rows)
    pools = all_pool_metrics(d.assign(hybrid_score=d.hybrid_score), models={"PPML": "ppml_score", "LightGBM": "lightgbm_score",
                             "Hybrid": "hybrid_score", "ImportDemand": "import_demand_score"}, ks=(10,), persistent_until=0)
    dec = pools.groupby(["task", "model"])[["ndcg", "hit"]].mean().rename(columns={"ndcg": "ndcg10_within_pool", "hit": "hit10_within_pool"}).reset_index()
    out = point.merge(dec, on=["task", "model"])
    out.to_csv(RES_DIR / "pointwise_vs_decision.csv", index=False)
    # model ranks under each metric (1 = best; RMSE/MAE lower is better)
    ranks = []
    for task, g in out.groupby("task"):
        for m in [c for c in out.columns if c not in ("task", "model")]:
            if g[m].notna().sum() < 2:
                continue
            asc = m.startswith(("rmse", "mae"))
            r = g.set_index("model")[m].rank(ascending=asc, method="min")
            ranks.append({"task": task, "metric": m, **r.to_dict(), "best": r.idxmin()})
    rk = pd.DataFrame(ranks)
    rk.to_csv(RES_DIR / "pointwise_vs_decision_ranks.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", lambda x: f"{x:,.4f}"):
        print(out.to_string(index=False)); print(); print(rk.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
