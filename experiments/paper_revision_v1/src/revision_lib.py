"""Shared helpers for the paper revision (2026-11): paths, stored inputs, pool metrics, contrasts.

All metric definitions follow experiments/final_hybrid_economic_validation_v1/src/final_validation.py
(the code behind the archived numbers): binary entry gain for discovery, capped graded gain
2^grade - 1 for expansion (grade 1 from USD 50k, 2 from 500k, 3 from 5m, 4 from 50m), DCG with log2(rank + 1) discounts,
NDCG undefined only for pools without any relevant destination.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]          # experiments/paper_revision_v1
PROJECT = HERE.parents[1]                            # project root
for p in (PROJECT / "src", PROJECT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import CFG  # noqa: E402
import fusion  # noqa: E402
import statistics as st  # noqa: E402

TA, TB = "Task_A_new_entry", "Task_B_existing_underexploited"
KEY = ["task", "destination_iso3", "hs2", "outcome_year"]
STORED_ALL_PERIODS = PROJECT / "experiments/explainable_reliability_gate_v1/predictions/ergm_candidate_predictions_all_periods.parquet"
PANEL = PROJECT / "data/processed/panel_lag2.parquet"
# REVISION_TAG keeps side runs (e.g. the legacy-config reproduction check) out of the main outputs.
import os  # noqa: E402
_TAG = os.environ.get("REVISION_TAG", "").strip()
PRED_DIR, RES_DIR, TAB_DIR = (HERE / "predictions" / _TAG, HERE / "results" / _TAG, HERE / "reports/tables" / _TAG) if _TAG \
    else (HERE / "predictions", HERE / "results", HERE / "reports/tables")
for d in (PRED_DIR, RES_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)


def zfill_hs2(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy(); frame["hs2"] = frame.hs2.astype(str).str.zfill(2); return frame


def load_stored_candidates() -> pd.DataFrame:
    """Candidates 2018-2024 with labels, features and the stored (deterministic) PPML and LightGBM scores."""
    return zfill_hs2(pd.read_parquet(STORED_ALL_PERIODS))


def attach_next_year_value(frame: pd.DataFrame) -> pd.DataFrame:
    pn = zfill_hs2(pd.read_parquet(PANEL, columns=["destination_iso3", "hs2", "outcome_year", "export_value_usd"]))
    nxt = pn.assign(outcome_year=pn.outcome_year - 1).rename(columns={"export_value_usd": "value_next_year"})
    return frame.merge(nxt, on=["destination_iso3", "hs2", "outcome_year"], how="left", validate="many_to_one")


# ---------------------------------------------------------------- metrics
def expansion_grade(pos, threshold: float = 50_000) -> np.ndarray:
    """Graded relevance of a positive increment: 0 below the threshold, then 1 + floor(log10(pos / threshold)), capped at 4
    (grades start at USD 50k, 500k, 5m and 50m). Earlier drafts used floor(log10(1 + pos / threshold)), whose first grade
    starts only at about USD 450k; see grading_sensitivity.py."""
    pos = np.maximum(0.0, np.asarray(pos, float))
    return np.where(pos >= threshold, np.minimum(4, 1 + np.floor(np.log10(np.maximum(pos, threshold) / threshold))), 0).astype(int)


def gain(frame: pd.DataFrame, task: str, persistent: bool = False) -> np.ndarray:
    if task == TA:
        y = pd.to_numeric(frame.entry_success, errors="coerce").to_numpy(float)
        if persistent:
            nxt = frame.value_next_year.to_numpy(float)
            y = np.where(np.isnan(y) | np.isnan(nxt), np.nan,
                         ((frame.export_value_usd.to_numpy(float) >= 50_000) & (nxt >= 50_000)).astype(float))
        return y
    pos = np.maximum(0.0, pd.to_numeric(frame.realized_incremental_export_usd, errors="coerce").to_numpy(float))
    return np.exp2(expansion_grade(pos)) - 1.0


def _dcg(x: np.ndarray, k: int) -> float:
    x = np.asarray(x, float)[:k]
    return float(np.sum(x / np.log2(np.arange(2, len(x) + 2))))


def pool_metrics(g: pd.DataFrame, score_col: str, task: str, k: int, persistent: bool = False) -> dict:
    r = g.sort_values([score_col, "destination_iso3"], ascending=[False, True], kind="mergesort")
    r = r[r.label_observed.astype(bool)]
    y = gain(r, task, persistent)
    if persistent:
        keep = ~np.isnan(y); r, y = r[keep], y[keep]
    n = min(k, len(r))
    if n == 0:
        return {"ndcg": np.nan, "hit": np.nan, "succ": np.nan}
    idcg = _dcg(np.sort(y)[::-1][:n], n)
    if task == TA:
        hits = y[:n]
    else:
        hits = (np.maximum(0, r.realized_incremental_export_usd.to_numpy(float)[:n]) >= 50_000).astype(float)
    return {"ndcg": _dcg(y[:n], n) / idcg if idcg > 0 else np.nan, "hit": float(hits.sum() > 0), "succ": float(hits.sum())}


def add_rules_and_hybrid(frame: pd.DataFrame, lgb_col: str = "lightgbm_score", tie_rule: str | None = None) -> pd.DataFrame:
    """Hybrid = equal-weight rank fusion of PPML and LightGBM; plus the import-demand rule."""
    tie_rule = tie_rule or CFG["study"].get("fusion_tie_rule", "iso3")
    out = fusion.fuse(frame, ["task", "hs2", "outcome_year"], ["ppml_score", lgb_col], tie_rule=tie_rule)
    out["hybrid_score"] = -out.fused_rank                     # higher is better, ties already resolved
    out["import_demand_score"] = out.log_destination_import_demand.fillna(-1e9)
    return out


MODELS = {"PPML": "ppml_score", "LightGBM": "lightgbm_score", "Hybrid": "hybrid_score", "ImportDemand": "import_demand_score"}


def all_pool_metrics(frame: pd.DataFrame, models: dict = MODELS, ks=(5, 10, 20), persistent_until: int = 2023) -> pd.DataFrame:
    rows = []
    for (task, hs2, yr), g in frame.groupby(["task", "hs2", "outcome_year"]):
        for k in ks:
            for name, col in models.items():
                rows.append({"task": task, "hs2": hs2, "year": yr, "K": k, "model": name, "persistent": False, **pool_metrics(g, col, task, k)})
        if task == TA and yr <= persistent_until:
            for name, col in models.items():
                rows.append({"task": task, "hs2": hs2, "year": yr, "K": 10, "model": name, "persistent": True, **pool_metrics(g, col, task, 10, True)})
    return pd.DataFrame(rows)


CONTRASTS = {TA: [("Hybrid", "LightGBM"), ("Hybrid", "PPML"), ("Hybrid", "ImportDemand")],
             TB: [("PPML", "Hybrid"), ("PPML", "LightGBM"), ("PPML", "ImportDemand")]}
PERIODS = {"2018-2024": range(2018, 2025), "2022-2024": range(2022, 2025), "2018-2021": range(2018, 2022)}


def contrast_table(pools: pd.DataFrame, reps: int = 10000) -> pd.DataFrame:
    """Paired contrasts with HS2-block bootstrap, two-way clustered t, year sign-flip; Holm within a family of three."""
    out = []
    for task, pairs in CONTRASTS.items():
        for period, years in PERIODS.items():
            for persistent in (False, True):
                if persistent and task != TA:
                    continue
                for k in ((10,) if persistent else (5, 10, 20)):
                    for metric in (("ndcg", "hit", "succ") if (k == 10 and not persistent) else ("ndcg",)):
                        q = pools[(pools.task == task) & (pools.K == k) & (pools.persistent == persistent) & pools.year.isin(list(years))]
                        w = q.pivot_table(index=["hs2", "year"], columns="model", values=metric).reset_index()
                        fam = []
                        for a, b in pairs:
                            x = (w[a] - w[b]).to_numpy(float)
                            res = st.paired_contrast_inference(x, w.hs2.to_numpy(), w.year.to_numpy(), reps=reps)
                            fam.append({"task": task, "period": period, "K": k, "persistent": persistent, "metric": metric,
                                        "contrast": f"{a} - {b}", "mean_a": float(np.nanmean(w[a])), "mean_b": float(np.nanmean(w[b])), **res})
                        holm = st.holm_adjust([f["p_signflip"] for f in fam])
                        for f, h in zip(fam, holm):
                            f["p_holm_family3"] = float(h)
                        out.extend(fam)
    return pd.DataFrame(out)
