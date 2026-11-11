from __future__ import annotations

import numpy as np
import pandas as pd


def capped_export_grade(increment_usd, threshold: float = 50_000.0) -> np.ndarray:
    x = np.asarray(increment_usd, dtype=float)
    x = np.maximum(0.0, np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
    return np.minimum(4, np.floor(np.log10(1.0 + x / threshold))).astype(np.int8)


def training_increment_cutoffs(train: pd.DataFrame, quantiles=(0.25, 0.50, 0.75)) -> np.ndarray:
    inc = pd.to_numeric(train.realized_incremental_export_usd, errors="coerce").to_numpy(float)
    positive = inc[np.isfinite(inc) & (inc > 0)]
    if not len(positive):
        raise ValueError("Task B training fold has no positive incremental exports")
    return np.quantile(positive, np.asarray(quantiles, dtype=float)).astype(float)


def task_b_quantile_grade(increment_usd, cutoffs) -> np.ndarray:
    x = np.asarray(increment_usd, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros(len(x), dtype=np.int8)
    pos = x > 0
    # With three distinct cutoffs this yields four positive categories (1–4).
    out[pos] = 1 + np.searchsorted(np.unique(np.asarray(cutoffs, dtype=float)), x[pos], side="left")
    return out


def query_sort(frame: pd.DataFrame, relevance: np.ndarray, features: list[str]):
    if len(frame) != len(relevance):
        raise ValueError("Feature rows and relevance labels differ in length")
    d = frame.copy()
    d["_rel"] = np.asarray(relevance, dtype=np.int8)
    d["_query"] = d.hs2.astype(str).str.zfill(2) + "|" + d.outcome_year.astype(str)
    d = d.sort_values(["_query", "destination_iso3"], kind="mergesort").reset_index(drop=True)
    group_sizes = d.groupby("_query", sort=False).size().to_numpy(np.int32)
    labels = d._rel.to_numpy(np.int32)
    x = d[features].copy()
    if int(group_sizes.sum()) != len(d):
        raise AssertionError("Query group sizes do not sum to training rows")
    if (labels < 0).any() or not np.issubdtype(labels.dtype, np.integer):
        raise AssertionError("LambdaMART relevance must be nonnegative integer labels")
    return d, x, labels, group_sizes


def assert_query_groups(frame: pd.DataFrame, groups: np.ndarray, labels: np.ndarray) -> None:
    if len(groups) == 0 or np.any(groups <= 0):
        raise AssertionError("Empty or nonpositive query group")
    if int(groups.sum()) != len(frame) or len(labels) != len(frame):
        raise AssertionError("Query groups/labels do not match model rows")
    if frame["_query"].duplicated(keep=False).any():
        q = frame["_query"].to_numpy()
        # Each query must occupy one contiguous block.
        seen = set()
        previous = None
        for value in q:
            if value != previous and value in seen:
                raise AssertionError("Rows for one query are not contiguous")
            seen.add(value)
            previous = value

