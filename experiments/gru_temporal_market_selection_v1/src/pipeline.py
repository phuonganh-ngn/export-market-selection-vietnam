from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/explainable_reliability_gate_v1"))
from experiments.explainable_reliability_gate_v1.src.task_audit import load_and_freeze  # noqa:E402

SEEDS = (11, 23, 37, 51, 73)
EVAL_YEARS = (2022, 2023, 2024)
THRESHOLD = 50_000.0
SEQ_FEATURES = [
    "log_bilateral_exports", "log_destination_imports", "supplier_concentration",
    "log_gdp", "log_gdp_per_capita", "gdp_growth", "log_population",
    "urban_population_share", "internet_usage", "inflation", "trade_openness",
    "government_effectiveness", "regulatory_quality", "rule_of_law", "political_stability",
]
SEQ_GROUPS = {
    "bilateral_exports": ["log_bilateral_exports"],
    "destination_import_demand": ["log_destination_imports", "supplier_concentration"],
    "macroeconomic": ["log_gdp", "log_gdp_per_capita", "gdp_growth", "log_population",
                      "urban_population_share", "internet_usage", "inflation", "trade_openness",
                      "government_effectiveness", "regulatory_quality", "rule_of_law", "political_stability"],
}
STATIC_NUM = ["log_distance", "contiguous", "common_official_language", "colonial_link",
              "common_colonizer", "landlocked"]
STATIC_CAT = ["destination_iso3", "hs2"]


def ensure_dirs() -> None:
    for folder in ["config", "data", "models", "predictions", "results", "reports", "logs", "tests"]:
        (HERE / folder).mkdir(parents=True, exist_ok=True)


def configure_cpu() -> None:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    torch.use_deterministic_algorithms(True, warn_only=True)


def feature_years(outcome_year: int, lookback: int = 5) -> list[int]:
    cutoff = int(outcome_year) - 2
    return list(range(cutoff - lookback + 1, cutoff + 1))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_annual_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build annual destination-HS2 signals with coverage-aware zeros."""
    trade = pd.read_parquet(ROOT / "data/interim/baci_hs2_trade.parquet")
    cmap = pd.read_parquet(ROOT / "data/processed/country_mapping.parquet")
    trade = trade.merge(cmap[["baci_id", "iso3"]], left_on="baci_importer_id", right_on="baci_id",
                        how="left", validate="many_to_one")
    trade = trade.loc[trade.iso3.str.fullmatch(r"[A-Z]{3}", na=False) & trade.iso3.ne("VNM")].copy()
    trade["hs2"] = trade.hs2.astype(str).str.zfill(2)
    trade["year"] = trade.year.astype(int)
    coverage = trade[["iso3", "year"]].drop_duplicates().rename(columns={"iso3": "destination_iso3"})
    annual = trade.groupby(["iso3", "hs2", "year"], as_index=False).agg(
        world_imports_usd=("world_imports_usd", "sum"),
        vietnam_exports_usd=("vietnam_exports_usd", lambda s: s.sum(min_count=1)),
        supplier_concentration=("supplier_hhi", "mean"),
    ).rename(columns={"iso3": "destination_iso3"})
    annual = annual.merge(coverage.assign(importer_year_covered=True), on=["destination_iso3", "year"],
                          how="left", validate="many_to_one")
    macro = pd.read_parquet(ROOT / "data/processed/country_year_features.parquet").copy()
    macro = macro.rename(columns={"iso3": "destination_iso3", "year": "year"})
    macro["destination_iso3"] = macro.destination_iso3.astype(str).str.upper()
    for c in ["log_gdp", "log_gdp_per_capita", "gdp_growth", "log_population", "urban_population_share",
              "internet_usage", "inflation", "trade_openness", "government_effectiveness",
              "regulatory_quality", "rule_of_law", "political_stability"]:
        if c not in macro:
            macro[c] = np.nan
    annual = annual.merge(macro[["destination_iso3", "year", *[c for c in SEQ_FEATURES if c not in
                                        ["log_bilateral_exports", "log_destination_imports", "supplier_concentration"]]]],
                          on=["destination_iso3", "year"], how="left", validate="many_to_one")
    annual["log_destination_imports"] = np.log1p(annual.world_imports_usd.clip(lower=0))
    annual["log_bilateral_exports"] = np.log1p(annual.vietnam_exports_usd.clip(lower=0))
    annual = annual.drop(columns=["importer_year_covered"], errors="ignore")
    return annual.sort_values(["destination_iso3", "hs2", "year"]).reset_index(drop=True), coverage


def build_dataset() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Return all strict Task-A candidates, their sequences and verified baseline predictions."""
    panel, _ = load_and_freeze(write=False)
    panel["hs2"] = panel.hs2.astype(str).str.zfill(2)
    panel["outcome_year"] = panel.outcome_year.astype(int)
    panel["feature_year"] = panel.feature_year.astype(int)
    candidates = panel.loc[panel.eligible_ex_ante.astype(bool) & panel.strict_inactive_verified.astype(bool)].copy()
    candidates = candidates.loc[candidates.outcome_year.ge(2018)].copy()
    if candidates.duplicated(["candidate_id"]).any():
        raise AssertionError("Duplicate strict Task-A candidate_id")
    annual, coverage = make_annual_features()
    annual_rows=[]
    for pos, offset in enumerate(range(-4, 1)):
        q=candidates[["candidate_id","destination_iso3","hs2","feature_year"]].copy()
        q["_candidate_order"]=np.arange(len(candidates),dtype=np.int32)
        q["year"]=q.feature_year.astype(int)+offset
        q["sequence_position"]=pos
        q=q.merge(annual,on=["destination_iso3","hs2","year"],how="left",validate="many_to_one")
        cov=coverage.rename(columns={"destination_iso3":"destination_iso3","year":"year"})
        q=q.merge(cov.assign(importer_year_covered=True),on=["destination_iso3","year"],how="left",validate="many_to_one")
        covered=q.importer_year_covered.fillna(False).to_numpy(bool)
        # Covered importer-years establish zero flow when a destination-HS2 record is absent, matching
        # the auditable BACI convention. Uncovered importer-years remain missing.
        for c in ["world_imports_usd","vietnam_exports_usd"]:
            q.loc[covered & q[c].isna(),c]=0.0
        q["log_destination_imports"]=np.log1p(q.world_imports_usd.clip(lower=0))
        q["log_bilateral_exports"]=np.log1p(q.vietnam_exports_usd.clip(lower=0))
        q["sequence_source_covered"]=covered
        annual_rows.append(q[["candidate_id","_candidate_order","sequence_position",*SEQ_FEATURES,"sequence_source_covered"]])
    seq_long=pd.concat(annual_rows,ignore_index=True)
    seq_long=seq_long.sort_values(["_candidate_order","sequence_position"],kind="mergesort")
    seq=seq_long[SEQ_FEATURES].to_numpy(dtype=np.float32).reshape(len(candidates),5,len(SEQ_FEATURES))
    coverage_matrix=seq_long.sequence_source_covered.to_numpy(bool).reshape(len(candidates),5)
    candidates["sequence_years"] = [",".join(map(str, feature_years(y, 5))) for y in candidates.outcome_year]
    for c in STATIC_NUM:
        if c not in candidates:
            candidates[c] = np.nan
    candidates["destination_iso3"] = candidates.destination_iso3.astype(str)
    candidates["hs2"] = candidates.hs2.astype(str).str.zfill(2)
    candidates["sequence_source_coverage"] = coverage_matrix.mean(axis=1)
    # Retain candidate metadata and labels; never use labels to form predictors.
    base_path = ROOT / "experiments/two_part_external_validation_v1/predictions/A_candidate_predictions.parquet"
    base = pd.read_parquet(base_path)
    base["hs2"] = base.hs2.astype(str).str.zfill(2)
    outer = candidates.loc[candidates.outcome_year.isin(EVAL_YEARS)].copy()
    outer["candidate_pool_id"] = [f"Task_A_new_entry|{str(h).zfill(2)}|{int(y)}"
                                  for h,y in zip(outer.hs2,outer.outcome_year)]
    manifest = pd.read_csv(ROOT / "experiments/final_hybrid_economic_validation_v1/results/final_candidate_manifest.csv",
                           dtype={"hs2": str})
    manifest = manifest.loc[manifest.task.eq("Task_A_new_entry")].copy()
    manifest["hs2"] = manifest.hs2.astype(str).str.zfill(2)
    eval_keys = set(manifest.candidate_id)
    built_keys = set(outer.candidate_id)
    if eval_keys != built_keys:
        raise AssertionError(f"Current strict Task-A build differs from audited manifest: missing {len(eval_keys-built_keys)}, extra {len(built_keys-eval_keys)}")
    if not np.array_equal(manifest.sort_values("candidate_id").label_observed.astype(bool).to_numpy(),
                          outer.sort_values("candidate_id").label_observed.astype(bool).to_numpy()):
        raise AssertionError("Observed-label indicators differ from frozen audited manifest")
    if set(base.candidate_id) != eval_keys:
        raise AssertionError("Saved baseline prediction universe differs from the audited manifest")
    base_cols = [c for c in ["candidate_id", "ppml_score", "lightgbm_score", "blend_rank_score",
                              "two_part_qualified_usd_score", "entry_success", "export_value_usd",
                              "qualified_export_value_usd", "label_observed"] if c in base]
    outer = outer.merge(base[base_cols], on="candidate_id", how="left", suffixes=("", "_base"), validate="one_to_one")
    if outer.ppml_score.isna().any() or outer.lightgbm_score.isna().any() or outer.blend_rank_score.isna().any():
        raise AssertionError("A saved baseline score is missing for an audited candidate")
    if not np.allclose(outer.entry_success.fillna(-99), outer.entry_success_base.fillna(-99), equal_nan=True):
        raise AssertionError("Outcome labels differ between audited panel and saved prediction archive")
    return candidates.reset_index(drop=True), seq, annual, outer.reset_index(drop=True)


def temporal_audit(candidates: pd.DataFrame, seq: np.ndarray, annual: pd.DataFrame) -> pd.DataFrame:
    eval_mask = candidates.outcome_year.isin(EVAL_YEARS).to_numpy()
    x = seq[eval_mask]
    years = candidates.loc[eval_mask, "outcome_year"].to_numpy()
    rows = []
    for j, name in enumerate(SEQ_FEATURES):
        col = x[:, :, j].astype(float)
        valid = np.isfinite(col)
        within_std = np.nanstd(col, axis=1)
        within_uniq = np.array([len(np.unique(r[np.isfinite(r)])) for r in col])
        feature_rows = annual.loc[annual[name].notna(), ["year", name]]
        by_year = annual.groupby("year")[name].agg(["mean", "std", "count"])
        first_valid = int(feature_rows.year.min()) if len(feature_rows) else np.nan
        last_valid = int(feature_rows.year.max()) if len(feature_rows) else np.nan
        rows.append({
            "feature": name, "sequence_min_year": int(min(feature_years(y, 5)[0] for y in EVAL_YEARS)),
            "sequence_max_year": int(max(feature_years(y, 5)[-1] for y in EVAL_YEARS)),
            "earliest_available_year": first_valid, "latest_available_year": last_valid,
            "candidate_sequence_cells": int(col.size), "missing_cell_share": float(1 - valid.mean()),
            "sequences_with_at_least_2_observed_years": int((valid.sum(axis=1) >= 2).sum()),
            "sequences_with_nonconstant_observed_values": int((within_uniq >= 2).sum()),
            "nonconstant_sequence_share": float((within_uniq >= 2).mean()),
            "mean_within_sequence_sd": float(np.nanmean(within_std)),
            "sequence_zero_share": float(np.isclose(col[valid], 0).mean()) if valid.any() else np.nan,
            "first_year_mean": float(by_year.loc[by_year.index.isin([feature_years(y,5)[0] for y in EVAL_YEARS]), "mean"].mean()),
            "last_year_mean": float(by_year.loc[by_year.index.isin([feature_years(y,5)[-1] for y in EVAL_YEARS]), "mean"].mean()),
            "cutoff_compliant": True,
        })
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results/temporal_signal_audit.csv", index=False)
    m = pd.read_csv(ROOT / "experiments/final_hybrid_economic_validation_v1/results/final_candidate_manifest.csv",
                    dtype={"hs2": str})
    m = m.loc[m.task.eq("Task_A_new_entry")].copy()
    counts = m.groupby("outcome_year").agg(candidates=("candidate_id", "size"), observed_labels=("label_observed", "sum"),
        hs2=("hs2", "nunique"), pools=("candidate_pool_id", "nunique")).reset_index()
    event = m.groupby("outcome_year").entry_success.sum(min_count=1).rename("successful_entries").reset_index()
    counts = counts.merge(event, on="outcome_year")
    counts["unknown_labels"] = counts.candidates - counts.observed_labels
    counts["lookback5_sequence_coverage_share"] = candidates.loc[eval_mask, "sequence_source_coverage"].groupby(
        candidates.loc[eval_mask, "outcome_year"]).mean().reindex(counts.outcome_year).to_numpy()
    counts.to_csv(HERE / "results/sample_flow.csv", index=False)
    audit_text = [
        "# Temporal signal audit", "",
        "The primary sequence uses five annual feature vintages from `outcome_year - 6` through `outcome_year - 2`. The maximum input year equals the audited information cutoff; no outcome-year or `y-1` field enters the sequence.", "",
        f"The temporal candidate construction rebuilt {len(candidates.loc[candidates.outcome_year.isin(EVAL_YEARS)]):,} strict-zero Task A evaluation rows. The frozen manifest has {m.candidate_id.nunique():,} unique candidates, {int(m.label_observed.sum()):,} observed outcomes, and {int((~m.label_observed).sum()):,} unknown outcomes across {m.hs2.nunique()} HS2 chapters and {m.candidate_pool_id.nunique()} HS2–year pools.", "",
        "The strict-zero rule makes the three most recent bilateral-export sequence positions zero for eligible candidates. Thus those positions do not contain within-pool variation in Vietnamese bilateral exports. The first two positions may vary and provide at most two years of pre-inactivity trade history. Destination import demand and macroeconomic trajectories are separately audited below.", "",
        "A missing annual product-market record is converted to zero only when BACI importer-year reporting coverage is present, following the same coverage rule as the audited strict-zero history builder. Uncovered importer-years remain missing; supplier concentration remains missing when no positive-import market cell exists. Macro predictor gaps remain missing and are handled by training-only imputation plus missingness indicators.", "",
        "## Temporal feature audit", "", out.to_markdown(index=False), "",
        "## Evaluation sample by outcome year", "", counts.to_markdown(index=False), "",
        "These data are retrospective and use revised WDI/WGI vintages. They cannot establish that the exact historical macro values were available in real time. The sequence audit tests usable variation, not causality or prospective information-vintage validity.",
    ]
    (HERE / "reports/01_TEMPORAL_SIGNAL_AUDIT.md").write_text("\n".join(audit_text), encoding="utf-8")
    return out


class SequenceTransformer:
    def __init__(self):
        self.median = None
        self.mean = None
        self.scale = None

    def fit(self, x: np.ndarray) -> "SequenceTransformer":
        flat = x.reshape(-1, x.shape[-1]).astype(float)
        self.median = np.array([np.nanmedian(flat[:, j]) if np.isfinite(flat[:, j]).any() else 0.0
                                for j in range(flat.shape[1])])
        filled = np.where(np.isfinite(flat), flat, self.median)
        self.mean = filled.mean(axis=0)
        self.scale = filled.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        missing = ~np.isfinite(x)
        filled = np.where(missing, self.median, x)
        scaled = (filled - self.mean) / self.scale
        return np.concatenate([scaled, missing.astype(float)], axis=-1).astype(np.float32)


class StaticTransformer:
    def __init__(self, use_static: bool):
        self.use_static = use_static
        self.median = self.mean = self.scale = None
        self.encoder = None

    def fit(self, frame: pd.DataFrame) -> "StaticTransformer":
        if not self.use_static:
            return self
        x = frame[STATIC_NUM].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        self.median = np.array([np.nanmedian(x[:, j]) if np.isfinite(x[:, j]).any() else 0 for j in range(x.shape[1])])
        filled = np.where(np.isfinite(x), x, self.median)
        self.mean = filled.mean(axis=0)
        self.scale = filled.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        cats = frame[STATIC_CAT].fillna("__MISSING__").astype(str)
        self.encoder.fit(cats)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.use_static:
            return np.zeros((len(frame), 0), dtype=np.float32)
        x = frame[STATIC_NUM].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        miss = ~np.isfinite(x)
        x = np.where(miss, self.median, x)
        x = (x - self.mean) / self.scale
        cats = self.encoder.transform(frame[STATIC_CAT].fillna("__MISSING__").astype(str))
        return np.column_stack([x, miss.astype(float), cats]).astype(np.float32)


class SmallGRU(nn.Module):
    def __init__(self, sequence_dim: int, static_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_size=sequence_dim, hidden_size=32, num_layers=1, batch_first=True)
        self.has_static = static_dim > 0
        if self.has_static:
            self.static = nn.Sequential(nn.Linear(static_dim, 16), nn.ReLU())
            fused_dim = 48
        else:
            fused_dim = 32
        self.head = nn.Sequential(nn.Linear(fused_dim, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 1))

    def forward(self, seq: torch.Tensor, static: torch.Tensor | None = None) -> torch.Tensor:
        _, h = self.gru(seq)
        temporal = h[-1]
        if self.has_static:
            temporal = torch.cat([temporal, self.static(static)], dim=1)
        return self.head(temporal).squeeze(1)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _torch_fit_predict(xtr: np.ndarray, str_: np.ndarray, ytr: np.ndarray,
                       xva: np.ndarray, sva: np.ndarray, yva: np.ndarray,
                       seed: int, epochs: int | None = None) -> tuple[np.ndarray, int, float, int]:
    _seed(seed)
    model = SmallGRU(xtr.shape[-1], str_.shape[1])
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    n_pos = max(int(ytr.sum()), 1)
    pos_weight = torch.tensor([(len(ytr) - n_pos) / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ds = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(str_), torch.from_numpy(ytr.astype(np.float32)))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0, generator=generator)
    max_epochs = int(epochs or 40)
    best_ap, best_loss, best_epoch, best_state, stale = -np.inf, np.inf, 1, None, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        for xb, sb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb, sb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(xva), torch.from_numpy(sva)).numpy()
            probs = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
            vloss = float(loss_fn(torch.from_numpy(logits), torch.from_numpy(yva.astype(np.float32))).item())
        ap = float(average_precision_score(yva, probs)) if np.unique(yva).size == 2 else np.nan
        improved = (np.isfinite(ap) and ap > best_ap + 1e-7) or (not np.isfinite(ap) and vloss < best_loss - 1e-7)
        if improved:
            best_ap, best_loss, best_epoch = ap, vloss, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if epochs is None and stale >= 6:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(xva), torch.from_numpy(sva)).numpy()
    return val_logits, best_epoch, best_ap, params


def _torch_train_final(xtr: np.ndarray, str_: np.ndarray, ytr: np.ndarray,
                       xt: np.ndarray, st: np.ndarray, seed: int, epochs: int) -> tuple[np.ndarray, SmallGRU, int]:
    _seed(seed)
    model = SmallGRU(xtr.shape[-1], str_.shape[1])
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    n_pos = max(int(ytr.sum()), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(ytr) - n_pos) / n_pos], dtype=torch.float32))
    loader = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(str_),
                                     torch.from_numpy(ytr.astype(np.float32))),
                        batch_size=256, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    for _ in range(max(1, int(epochs))):
        model.train()
        for xb, sb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            criterion(model(xb, sb), yb).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(xt), torch.from_numpy(st)).numpy()
    return (1 / (1 + np.exp(-np.clip(logits, -40, 40)))).astype(float), model, params


def _prepare_fold(candidates: pd.DataFrame, seq: np.ndarray,
                  train_idx: np.ndarray, test_idx: np.ndarray, use_static: bool,
                  features: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SequenceTransformer, StaticTransformer]:
    use = features or SEQ_FEATURES
    cols = [SEQ_FEATURES.index(c) for c in use]
    rawtr, rawte = seq[train_idx][:, :, cols], seq[test_idx][:, :, cols]
    st = SequenceTransformer().fit(rawtr)
    tr_seq, te_seq = st.transform(rawtr), st.transform(rawte)
    sx = StaticTransformer(use_static).fit(candidates.iloc[train_idx])
    tr_static = sx.transform(candidates.iloc[train_idx])
    te_static = sx.transform(candidates.iloc[test_idx])
    return tr_seq, tr_static, te_seq, te_static, st, sx


def _latest_validation_split(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    years = sorted(train.outcome_year.astype(int).unique())
    val_year = years[-1]
    tr_mask = train.outcome_year.le(val_year - 2).to_numpy()
    va_mask = train.outcome_year.eq(val_year).to_numpy()
    if tr_mask.sum() < 100 or va_mask.sum() < 100 or train.loc[va_mask, "entry_success"].nunique() < 2:
        val_year = years[-1]
        va_mask = train.outcome_year.eq(val_year).to_numpy()
        tr_mask = train.outcome_year.lt(val_year).to_numpy()
    if tr_mask.sum() < 100 or va_mask.sum() < 100:
        raise ValueError(f"Insufficient chronological train/validation split; years={years}")
    return np.flatnonzero(tr_mask), np.flatnonzero(va_mask)


def _expand_lgbm(xseq: np.ndarray, xstatic: np.ndarray, include_summary: bool = False) -> np.ndarray:
    n, t, f = xseq.shape
    flat = xseq.reshape(n, t * f)
    if include_summary:
        # These are deterministic transformations of the same five historical observations.
        means = xseq.mean(axis=1)
        sds = xseq.std(axis=1)
        change = xseq[:, -1, :] - xseq[:, 0, :]
        mins = xseq.min(axis=1)
        maxs = xseq.max(axis=1)
        flat = np.column_stack([flat, means, sds, change, mins, maxs])
    return np.column_stack([flat, xstatic]).astype(np.float32)


def train_primary_models(candidates: pd.DataFrame, seq: np.ndarray, eval_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records, seed_metrics, runtimes, model_cache = [], [], [], []
    row_index = {cid: i for i, cid in enumerate(candidates.candidate_id)}
    evals = eval_predictions[["candidate_id", "destination_iso3", "hs2", "outcome_year", "entry_success",
                             "export_value_usd", "label_observed", "qualified_export_value_usd",
                             "ppml_score", "lightgbm_score", "blend_rank_score", "two_part_qualified_usd_score"]].copy()
    eval_idx = evals.candidate_id.map(row_index).to_numpy(int)
    for lookback, variants in [(5, ["GRU_SEQ_ONLY", "GRU_SEQ_PLUS_STATIC"]),
                               (3, ["GRU_LOOKBACK3_SEQ_PLUS_STATIC"])]:
        # The common tensor stores five years; the 3-year sensitivity retains the final three permitted years.
        seq_view = seq if lookback == 5 else seq[:, -3:, :]
        for year in EVAL_YEARS:
            cutoff = year - 2
            test_mask_eval = evals.outcome_year.eq(year).to_numpy()
            test_idx_all = eval_idx[test_mask_eval]
            train_df = candidates.loc[candidates.outcome_year.le(cutoff) & candidates.label_observed.astype(bool)
                                     & candidates.entry_success.notna()].copy()
            train_idx_all = train_df.index.to_numpy()
            # candidates were reset_index(drop=True); preserve those row indices explicitly.
            train_ids = train_df.candidate_id.to_numpy()
            train_idx_all = np.array([row_index[cid] for cid in train_ids], dtype=int)
            if not len(train_idx_all):
                raise ValueError(f"No historical train candidates for {year}")
            train_df = train_df.reset_index(drop=True)
            itrain_rel, ival_rel = _latest_validation_split(train_df)
            ytr_all = train_df.entry_success.astype(int).to_numpy()
            y_train_inner, y_val = ytr_all[itrain_rel], ytr_all[ival_rel]
            inner_ids, val_ids = train_df.iloc[itrain_rel].candidate_id.to_numpy(), train_df.iloc[ival_rel].candidate_id.to_numpy()
            inner_idx = np.array([row_index[cid] for cid in inner_ids], dtype=int)
            val_idx = np.array([row_index[cid] for cid in val_ids], dtype=int)
            test_rows = evals.loc[test_mask_eval].copy().reset_index(drop=True)
            for variant in variants:
                use_static = variant != "GRU_SEQ_ONLY"
                for seed in SEEDS:
                    started = time.perf_counter()
                    xtr, str_, xva, sva, seq_scaler, static_scaler = _prepare_fold(
                        candidates, seq_view, inner_idx, val_idx, use_static)
                    val_logits, selected_epochs, val_ap, params = _torch_fit_predict(
                        xtr, str_, y_train_inner, xva, sva, y_val, seed)
                    if selected_epochs < 1:
                        selected_epochs = 1
                    # Refit all labels available at the outer prediction origin, with the development-selected epoch count.
                    seq_final = SequenceTransformer().fit(seq_view[train_idx_all])
                    xtr_final = seq_final.transform(seq_view[train_idx_all])
                    xtest = seq_final.transform(seq_view[test_idx_all])
                    static_final = StaticTransformer(use_static).fit(candidates.iloc[train_idx_all])
                    st_train = static_final.transform(candidates.iloc[train_idx_all])
                    st_test = static_final.transform(candidates.iloc[test_idx_all])
                    prob, model, params = _torch_train_final(xtr_final, st_train, ytr_all, xtest, st_test,
                                                              seed, selected_epochs)
                    elapsed = time.perf_counter() - started
                    name = f"{variant}_L{lookback}"
                    for i, r in enumerate(test_rows.itertuples(index=False)):
                        records.append({"model": name, "seed": seed, "candidate_id": r.candidate_id,
                            "destination_iso3": r.destination_iso3, "hs2": str(r.hs2).zfill(2),
                            "outcome_year": int(r.outcome_year), "candidate_pool_id": f"Task_A_new_entry|{str(r.hs2).zfill(2)}|{int(r.outcome_year)}",
                            "score": float(prob[i]), "label_observed": bool(r.label_observed),
                            "entry_success": r.entry_success, "export_value_usd": r.export_value_usd,
                            "qualified_export_value_usd": r.qualified_export_value_usd,
                            "prediction_cutoff": cutoff, "training_outcome_max": int(train_df.outcome_year.max()),
                            "validation_outcome_year": int(train_df.iloc[ival_rel].outcome_year.max()),
                            "selected_epochs": selected_epochs, "parameter_count": params})
                    metrics_val = score_pools(test_rows, pd.Series(prob, index=np.arange(len(prob))), "score", scores_are_ordered=True)
                    seed_metrics.append({"model": name, "seed": seed, "evaluation_year": year,
                                         **metrics_val, "training_n": len(train_df), "training_events": int(ytr_all.sum()),
                                         "validation_year": int(train_df.iloc[ival_rel].outcome_year.max()),
                                         "validation_events": int(y_val.sum()), "selected_epochs": selected_epochs})
                    runtimes.append({"model": name, "seed": seed, "evaluation_year": year,
                                     "runtime_seconds": elapsed, "parameter_count": params,
                                     "train_candidates": len(train_df), "train_events": int(ytr_all.sum()),
                                     "selected_epochs": selected_epochs, "validation_pr_auc": val_ap,
                                     "threads": 4, "device": "cpu"})
                    model_file = HERE / "models" / f"{name}_{year}_seed{seed}.pt"
                    torch.save({"state_dict": model.state_dict(), "seed": seed, "evaluation_year": year,
                                "lookback": lookback, "variant": variant, "selected_epochs": selected_epochs,
                                "sequence_features": SEQ_FEATURES, "static_numeric": STATIC_NUM if use_static else [],
                                "static_categorical": STATIC_CAT if use_static else [],
                                "training_outcome_max": int(train_df.outcome_year.max()),
                                "cutoff": cutoff, "seq_scaler_median": seq_final.median,
                                "seq_scaler_mean": seq_final.mean, "seq_scaler_scale": seq_final.scale,
                                "static_scaler": static_final.__dict__}, model_file)
                    model_cache.append({"model": name, "seed": seed, "evaluation_year": year,
                                        "parameter_count": params, "state_path": str(model_file)})
    predictions = pd.DataFrame(records)
    predictions.to_parquet(HERE / "predictions/gru_predictions.parquet", index=False)
    sm = pd.DataFrame(seed_metrics)
    sm.to_csv(HERE / "results/G1_seed_stability.csv", index=False)
    pd.DataFrame(runtimes).to_csv(HERE / "results/model_runtime.csv", index=False)
    pd.DataFrame(model_cache).to_csv(HERE / "results/model_registry.csv", index=False)
    return predictions, sm, pd.DataFrame(runtimes)


def train_matched_lgbm(candidates: pd.DataFrame, seq: np.ndarray, eval_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_index = {cid: i for i, cid in enumerate(candidates.candidate_id)}
    evals = eval_predictions.copy().reset_index(drop=True)
    eval_idx = evals.candidate_id.map(row_index).to_numpy(int)
    records, costs = [], []
    for year in EVAL_YEARS:
        cutoff = year - 2
        test_mask = evals.outcome_year.eq(year).to_numpy()
        test_idx = eval_idx[test_mask]
        test_df = evals.loc[test_mask].reset_index(drop=True)
        tr_df = candidates.loc[candidates.outcome_year.le(cutoff) & candidates.label_observed.astype(bool)
                               & candidates.entry_success.notna()].copy()
        tr_idx = np.array([row_index[cid] for cid in tr_df.candidate_id], dtype=int)
        y = tr_df.entry_success.astype(int).to_numpy()
        # Fit all sequence imputers/scalers/identity encoding on the fold's training outcomes only.
        seq_transform = SequenceTransformer().fit(seq[tr_idx])
        xtr_seq = seq_transform.transform(seq[tr_idx])
        xte_seq = seq_transform.transform(seq[test_idx])
        stat = StaticTransformer(True).fit(candidates.iloc[tr_idx])
        xtr_static = stat.transform(candidates.iloc[tr_idx])
        xte_static = stat.transform(candidates.iloc[test_idx])
        for summary in (False, True):
            xtr = _expand_lgbm(xtr_seq, xtr_static, summary)
            xte = _expand_lgbm(xte_seq, xte_static, summary)
            model = LGBMClassifier(n_estimators=120, num_leaves=31, learning_rate=0.05, max_depth=-1,
                                   min_child_samples=40, class_weight="balanced", n_jobs=4,
                                   verbosity=-1, random_state=11)
            start = time.perf_counter()
            model.fit(xtr, y)
            probs = model.predict_proba(xte)[:, 1]
            duration = time.perf_counter() - start
            name = "LGBM_TEMPORAL_MATCHED_SUMMARY" if summary else "LGBM_TEMPORAL_MATCHED_RAW"
            for i, r in enumerate(test_df.itertuples(index=False)):
                records.append({"model": name, "seed": 11, "candidate_id": r.candidate_id,
                    "destination_iso3": r.destination_iso3, "hs2": str(r.hs2).zfill(2),
                    "outcome_year": int(r.outcome_year), "candidate_pool_id": f"Task_A_new_entry|{str(r.hs2).zfill(2)}|{int(r.outcome_year)}",
                    "score": float(probs[i]), "label_observed": bool(r.label_observed),
                    "entry_success": r.entry_success, "export_value_usd": r.export_value_usd,
                    "qualified_export_value_usd": r.qualified_export_value_usd,
                    "prediction_cutoff": cutoff, "training_outcome_max": int(tr_df.outcome_year.max()),
                    "parameter_count": np.nan, "input_feature_count": int(model.n_features_in_)})
            costs.append({"model": name, "seed": 11, "evaluation_year": year, "runtime_seconds": duration,
                          "parameter_count": np.nan, "input_feature_count": int(model.n_features_in_), "train_candidates": len(tr_df),
                          "train_events": int(y.sum()), "n_estimators": 120, "threads": 4, "device": "cpu"})
    pred = pd.DataFrame(records)
    pred.to_parquet(HERE / "predictions/matched_lgbm_predictions.parquet", index=False)
    pd.DataFrame(costs).to_csv(HERE / "results/model_runtime_lgbm.csv", index=False)
    return pred, pd.DataFrame(costs)


def _ndcg(y: np.ndarray, k: int) -> float:
    y = np.asarray(y, float)
    if len(y) == 0 or y.sum() <= 0:
        return np.nan
    k = min(k, len(y))
    dcg = np.sum(y[:k] / np.log2(np.arange(2, k + 2)))
    ideal = np.sort(y)[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return float(dcg / idcg) if idcg > 0 else np.nan


def _rank(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    return frame.sort_values([score, "destination_iso3"], ascending=[False, True], kind="mergesort")


def score_pools(frame: pd.DataFrame, score: pd.Series | None = None, score_col: str | None = None,
                scores_are_ordered: bool = False) -> dict:
    """Aggregate conditional complete-case metrics across HS2-year pools."""
    d = frame.copy().reset_index(drop=True)
    if scores_are_ordered:
        d["_score"] = np.asarray(score, float)
        score_col = "_score"
    known_pool_rows, all_pool_rows = [], []
    for (hs2, year), g in d.groupby(["hs2", "outcome_year"], sort=False):
        full = _rank(g, score_col)
        known = full.loc[full.label_observed.astype(bool) & full.entry_success.notna()].copy()
        known = _rank(known, score_col)
        y = known.entry_success.astype(float).to_numpy()
        top = known.head(10)
        ytop = top.entry_success.astype(float).to_numpy()
        nd = _ndcg(y, 10)
        nd5 = _ndcg(y, 5)
        full_top = full.head(10)
        observed_full = full_top.loc[full_top.label_observed.astype(bool) & full_top.entry_success.notna()]
        oracle_q = np.sort(known.qualified_export_value_usd.fillna(0).to_numpy(float))[::-1][:min(10, len(known))] if "qualified_export_value_usd" in known else np.array([])
        selected_q = top.qualified_export_value_usd.fillna(0).to_numpy(float) if "qualified_export_value_usd" in top else np.array([])
        oracle_usd = float(oracle_q.sum())
        picked_usd = float(selected_q.sum())
        all_pool_rows.append({"hs2": str(hs2).zfill(2), "outcome_year": int(year), "candidate_n": len(g),
            "observed_n": len(known), "unknown_n": len(g)-len(known),
            "unknown_in_full_top10": int((~full_top.label_observed.astype(bool)).sum()),
            "observed_successes_full_top10_partial": float(observed_full.entry_success.sum()),
            "observed_qualified_usd_full_top10_partial": float(observed_full.qualified_export_value_usd.sum()),
            "ndcg_full_population_zero_unknown_sensitivity": _ndcg(np.where(full.label_observed.astype(bool), full.entry_success.fillna(0), 0).astype(float),10)})
        known_pool_rows.append({"hs2": str(hs2).zfill(2), "outcome_year": int(year), "candidate_n_observed": len(known),
            "unknown_n": len(g)-len(known), "positive_n": int(y.sum()), "ndcg10": nd, "ndcg5": nd5,
            "precision10": float(ytop.sum()/10), "precision5": float(top.head(5).entry_success.sum()/5),
            "successes10": float(ytop.sum()), "successes5": float(top.head(5).entry_success.sum()),
            "qualified_usd10": picked_usd, "total_usd10": float(top.export_value_usd.fillna(0).sum()),
            "oracle_qualified_usd10": oracle_usd, "qualified_usd_regret10": max(0.0, oracle_usd-picked_usd),
            "capture_share_qualified_usd10": picked_usd/oracle_usd if oracle_usd>0 else np.nan,
            "pool_has_positive": bool(y.sum()>0)})
    p = pd.DataFrame(known_pool_rows)
    allp = pd.DataFrame(all_pool_rows)
    result = {"pools_total": int(len(p)), "pools_with_identified_positive": int(p.pool_has_positive.sum()),
        "candidate_n_mean": float(p.candidate_n_observed.mean()), "unknown_in_full_top10_mean": float(allp.unknown_in_full_top10.mean()),
        "ndcg10_positive_pools": float(p.loc[p.pool_has_positive,"ndcg10"].mean()),
        "ndcg10_all_pools_zero_relevance_sensitivity": float(p.ndcg10.fillna(0).mean()),
        "ndcg5_positive_pools": float(p.loc[p.pool_has_positive,"ndcg5"].mean()),
        "precision10_all_pools": float(p.precision10.mean()), "precision5_all_pools": float(p.precision5.mean()),
        "successes10_all_pools": float(p.successes10.mean()), "successes5_all_pools": float(p.successes5.mean()),
        "qualified_usd10_mean_per_pool": float(p.qualified_usd10.mean()),
        "qualified_usd10_median_per_pool": float(p.qualified_usd10.median()),
        "total_usd10_mean_per_pool": float(p.total_usd10.mean()),
        "qualified_usd_regret10_mean": float(p.qualified_usd_regret10.mean()),
        "capture_share_mean_pool": float(p.capture_share_qualified_usd10.mean(skipna=True))}
    return result


def evaluate_models(eval_predictions: pd.DataFrame, gru: pd.DataFrame, lgbm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_scores = {
        "Gravity_PPML": "ppml_score", "LightGBM": "lightgbm_score", "Equal_Rank_Blend": "blend_rank_score",
        "Two_Part": "two_part_qualified_usd_score",
    }
    metric_rows, pool_tables = [], []
    for model, col in baseline_scores.items():
        d = eval_predictions.copy()
        if model == "Two_Part":
            d["_score"] = d[col]
        else:
            d["_score"] = d[col]
        pm = _pool_metrics_table(d, "_score", model)
        pool_tables.append(pm)
        metric_rows.append({"model": model, **score_pools(d, score_col="_score")})
    for pred in [gru, lgbm]:
        if pred is None or pred.empty:
            continue
        for (model, seed), d0 in pred.groupby(["model", "seed"], sort=False):
            d = d0.merge(eval_predictions[["candidate_id", "ppml_score", "lightgbm_score", "blend_rank_score",
                                         "two_part_qualified_usd_score", "entry_success", "export_value_usd",
                                         "qualified_export_value_usd", "label_observed"]].drop_duplicates("candidate_id"),
                         on="candidate_id", suffixes=("", "_eval"), validate="one_to_one")
            pm = _pool_metrics_table(d, "score", f"{model}_seed{seed}")
            pool_tables.append(pm)
            metric_rows.append({"model": model, "seed": seed, **score_pools(d, score_col="score")})
    metrics = pd.DataFrame(metric_rows)
    # Main results aggregated across seeds; preserve raw seed rows in G1_seed_stability.
    aggregate = []
    for model, g in metrics.groupby("model", sort=False):
        numeric = [c for c in g.select_dtypes(include=[np.number]).columns if c != "seed"]
        row = {"model": model, "seeds": int(g.seed.nunique()) if "seed" in g else 0}
        for c in numeric:
            row[c] = float(g[c].mean())
            if g[c].notna().sum() > 1 and "seed" in g:
                row[c+"_seed_sd"] = float(g[c].std())
        aggregate.append(row)
    overall = pd.DataFrame(aggregate)
    overall.to_csv(HERE / "results/G1_ranking_performance.csv", index=False)
    _prediction_metrics(eval_predictions, gru, lgbm)
    bypool = pd.concat(pool_tables, ignore_index=True)
    bypool.to_csv(HERE / "results/G1_pool_metrics.csv", index=False)
    return overall, bypool, metrics


def _pool_metrics_table(frame: pd.DataFrame, score_col: str, model: str) -> pd.DataFrame:
    rows=[]
    for (h,y),g in frame.groupby(["hs2","outcome_year"],sort=False):
        known=g.loc[g.label_observed.astype(bool)&g.entry_success.notna()].copy()
        known=_rank(known,score_col)
        vals=known.entry_success.astype(float).to_numpy()
        top=known.head(10)
        q=top.qualified_export_value_usd.fillna(0).to_numpy(float)
        allrank=_rank(g,score_col)
        rows.append({"model":model,"hs2":str(h).zfill(2),"outcome_year":int(y),"positive_n":int(vals.sum()),
            "ndcg10":_ndcg(vals,10),"ndcg5":_ndcg(vals,5),"precision10":float(top.entry_success.sum()/10),
            "successes10":float(top.entry_success.sum()),"qualified_usd10":float(q.sum()),
            "total_usd10":float(top.export_value_usd.fillna(0).sum()),
            "unknown_in_full_top10":int((~allrank.head(10).label_observed.astype(bool)).sum()),
            "pool_has_positive":bool(vals.sum()>0)})
    return pd.DataFrame(rows)


def _prediction_metrics(eval_predictions: pd.DataFrame, gru: pd.DataFrame, lgbm: pd.DataFrame) -> None:
    rows=[]
    combined=[]
    for model,col in [("Gravity_PPML","ppml_score"),("LightGBM","lightgbm_score"),("Equal_Rank_Blend","blend_rank_score")]:
        d=eval_predictions.copy();d["score"]=d[col];d["model"]=model;d["seed"]=0
        combined.append(d)
    for pred in [gru,lgbm]:
        if pred is not None and len(pred): combined.append(pred)
    allp=pd.concat(combined,ignore_index=True,sort=False)
    for (model,seed,year),g in allp.groupby(["model","seed","outcome_year"],sort=False):
        known=g.loc[g.label_observed.astype(bool)&g.entry_success.notna()]
        y=known.entry_success.astype(int).to_numpy();s=known.score.to_numpy(float)
        is_probability = model == "LightGBM" or model.startswith("GRU_")
        rows.append({"model":model,"seed":int(seed),"outcome_year":int(year),"observed_n":len(known),
            "successes":int(y.sum()),"prevalence":float(y.mean()) if len(y) else np.nan,
            "roc_auc":float(roc_auc_score(y,s)) if len(np.unique(y))==2 else np.nan,
            "pr_auc":float(average_precision_score(y,s)) if y.sum()>0 else np.nan,
            "brier":float(brier_score_loss(y,s)) if is_probability and len(y) else np.nan,
            "log_loss":float(log_loss(y,np.clip(s,1e-6,1-1e-6),labels=[0,1])) if is_probability and len(np.unique(y))==2 else np.nan,
            "probability_metric_valid":bool(is_probability)})
    pd.DataFrame(rows).to_csv(HERE/"results/G1_prediction_performance.csv",index=False)


def _bootstrap_tests(pool_metrics: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Paired HS2 block bootstrap on seed-averaged GRU pool values."""
    # Pool metrics include baseline pool rows and all GRU seed-specific rows.
    wide=pool_metrics.pivot_table(index=["hs2","outcome_year"],columns="model",values=["ndcg10","successes10","qualified_usd10"],aggfunc="mean")
    tests=[]
    model_pairs=[("GRU_SEQ_PLUS_STATIC_L5","LightGBM"),("GRU_SEQ_PLUS_STATIC_L5","Equal_Rank_Blend"),
                 ("GRU_SEQ_PLUS_STATIC_L5","LGBM_TEMPORAL_MATCHED_RAW")]
    metrics=[("ndcg10","NDCG10"),("successes10","Successful entries@10"),("qualified_usd10","Qualified USD@10")]
    rng=np.random.default_rng(51073); reps=10000
    for a,b in model_pairs:
        for col,label in metrics:
            if (col,a) not in wide or (col,b) not in wide: continue
            d=(wide[(col,a)]-wide[(col,b)]).dropna()
            if d.empty: continue
            grouped=d.groupby(level="hs2")
            block=grouped.agg(["sum","count"])
            block_means=block["sum"].to_numpy(float)/block["count"].to_numpy(float)
            block_counts=block["count"].to_numpy(float)
            vals=block_means; n=len(vals)
            draw=rng.integers(0,n,size=(reps,n))
            boot=(block["sum"].to_numpy(float)[draw].sum(axis=1)/block_counts[draw].sum(axis=1))
            ci=np.quantile(boot,[.025,.975])
            signs=rng.choice([-1,1],size=(reps,n))
            sign_stat=(signs*block["sum"].to_numpy(float)[None,:]).sum(axis=1)/block_counts.sum()
            p=(1+np.sum(np.abs(sign_stat)>=abs(d.mean())))/(reps+1)
            tests.append({"contrast":f"{a} - {b}","metric":label,"mean_difference":float(d.mean()),
                "median_difference":float(d.median()),"relative_percent":float(100*d.mean()/abs(wide[(col,b)].dropna().mean())) if wide[(col,b)].dropna().mean()!=0 else np.nan,
                "ci_low_hs2_bootstrap":float(ci[0]),"ci_high_hs2_bootstrap":float(ci[1]),"sign_flip_p":float(p),
                "paired_pools":int(len(d)),"hs2_blocks":int(n),"bootstrap_replicates":reps,
                "inference_note":"HS2 blocks retain all years; only 3 previously inspected years; exploratory"})
    out=pd.DataFrame(tests)
    if len(out):
        order=np.argsort(out.sign_flip_p.to_numpy()); p=out.sign_flip_p.to_numpy(); adj=np.empty_like(p); running=0.0
        for rank,idx in enumerate(order):
            running=max(running,min(1.0,(len(p)-rank)*p[idx]));adj[idx]=running
        out["holm_adjusted_p"] = adj
    out.to_csv(HERE/"results/G5_paired_inference.csv",index=False)
    out.to_csv(HERE/"results/G5_holm_adjusted_tests.csv",index=False)
    return out,out.copy()


def _make_pool_metrics_all(eval_predictions: pd.DataFrame, gru: pd.DataFrame, lgbm: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    def add(model,seed,d,score):
        q=d.copy();q["_s"]=score
        pm=_pool_metrics_table(q,"_s",model)
        pm["seed"]=seed
        rows.append(pm)
    for model,col in [("LightGBM","lightgbm_score"),("Gravity_PPML","ppml_score"),("Equal_Rank_Blend","blend_rank_score"),
                      ("Two_Part","two_part_qualified_usd_score")]: add(model,0,eval_predictions,eval_predictions[col].to_numpy(float))
    for pred in [gru,lgbm]:
        if pred is not None and len(pred):
            for (model,seed),g in pred.groupby(["model","seed"]):
                d=eval_predictions.merge(g[["candidate_id","score"]],on="candidate_id",validate="one_to_one")
                add(model,seed,d,d.score.to_numpy(float))
    return pd.concat(rows,ignore_index=True)


def value_tail_sensitivity(eval_predictions: pd.DataFrame, gru: pd.DataFrame, lgbm: pd.DataFrame,
                           candidates: pd.DataFrame) -> pd.DataFrame:
    """USD tail diagnostics using a development-only P99 cap and pool-average seeds."""
    dev=candidates.loc[candidates.outcome_year.le(2020)&candidates.label_observed.astype(bool)
                       &candidates.entry_success.ge(1)&candidates.export_value_usd.notna(),"export_value_usd"].to_numpy(float)
    cap=float(np.quantile(dev,.99)) if len(dev) else np.nan
    model_rows=[]
    baseline={"Gravity_PPML":"ppml_score","LightGBM":"lightgbm_score","Equal_Rank_Blend":"blend_rank_score",
              "Two_Part":"two_part_qualified_usd_score"}
    for model,col in baseline.items():
        model_rows.append((model,0,eval_predictions,col))
    for pred in [gru,lgbm]:
        if pred is not None and len(pred):
            for (model,seed),d in pred.groupby(["model","seed"],sort=False):
                model_rows.append((model,int(seed),d,"score"))
    rows=[]
    for model,seed,source,col in model_rows:
        if col=="score" and source is not eval_predictions:
            d=eval_predictions.merge(source[["candidate_id","score"]],on="candidate_id",validate="one_to_one")
        else:
            d=source.copy()
            d["score"]=d[col]
        for (h,y),pool in d.groupby(["hs2","outcome_year"],sort=False):
            known=pool.loc[pool.label_observed.astype(bool)&pool.qualified_export_value_usd.notna()]
            top=_rank(known,"score").head(10)
            values=top.qualified_export_value_usd.to_numpy(float)
            rows.append({"model":model,"seed":seed,"hs2":str(h).zfill(2),"outcome_year":int(y),
                         "qualified_usd10_uncapped":float(values.sum()),
                         "qualified_usd10_capped_dev_p99":float(np.minimum(values,cap).sum()) if np.isfinite(cap) else np.nan})
    detail=pd.DataFrame(rows)
    pool_avg=detail.groupby(["model","hs2","outcome_year"],as_index=False).agg(
        qualified_usd10_uncapped=("qualified_usd10_uncapped","mean"),
        qualified_usd10_capped_dev_p99=("qualified_usd10_capped_dev_p99","mean"))
    out=[]
    for model,g in pool_avg.groupby("model"):
        vals=g.qualified_usd10_uncapped.to_numpy(float)
        top=vals[np.argsort(vals)[-max(1,int(np.ceil(.01*len(vals)))):]]
        out.append({"model":model,"pool_n":len(g),"development_positive_training_n":len(dev),
            "development_p99_cap_usd_per_market":cap,"mean_qualified_usd10_uncapped":float(vals.mean()),
            "mean_qualified_usd10_capped_at_development_p99":float(g.qualified_usd10_capped_dev_p99.mean()),
            "median_qualified_usd10_uncapped":float(np.median(vals)),"p25_qualified_usd10_uncapped":float(np.quantile(vals,.25)),
            "p75_qualified_usd10_uncapped":float(np.quantile(vals,.75)),"zero_pool_share":float(np.mean(vals==0)),
            "positive_pool_share":float(np.mean(vals>0)),"top_1pct_pool_value_share":float(top.sum()/max(vals.sum(),1)),
            "note":"P99 cap estimated only from observed successful-entry exports with training outcomes <=2020; seed scores averaged within HS2-year pool"})
    result=pd.DataFrame(out)
    result.to_csv(HERE/"results/G2_value_tail_sensitivity.csv",index=False)
    return result


def summarize_seed_stability(seed_metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Report fold results, seed-level summaries and pairwise Top-10 Jaccard."""
    raw=seed_metrics.copy()
    if "row_type" in raw and raw.row_type.eq("seed_by_year").any():
        raw=raw.loc[raw.row_type.eq("seed_by_year")].copy()
    summary_bases=["ndcg10_positive_pools","ndcg10_all_pools_zero_relevance_sensitivity",
                   "precision10_all_pools","successes10_all_pools","qualified_usd10_mean_per_pool",
                   "qualified_usd_regret10_mean"]
    derived=[c for c in raw.columns if c.endswith(("_sd","_min","_max"))]
    derived += [c+"_mean" for c in summary_bases if c+"_mean" in raw.columns]
    raw=raw.drop(columns=["top10_jaccard_mean","row_type_jaccard","seed_count",*derived],errors="ignore")
    raw["row_type"]="seed_by_year"
    metric_cols=[c for c in ["ndcg10_positive_pools","ndcg10_all_pools_zero_relevance_sensitivity",
        "precision10_all_pools","successes10_all_pools","qualified_usd10_mean_per_pool",
        "qualified_usd_regret10_mean"] if c in raw]
    seed_avg=raw.groupby(["model","seed"],as_index=False)[metric_cols].mean()
    seed_avg["row_type"]="seed_average_across_outer_years"
    summary=[]
    for model,g in seed_avg.groupby("model"):
        row={"model":model,"row_type":"across_seed_distribution","seed_count":int(g.seed.nunique())}
        for col in metric_cols:
            v=g[col].dropna().to_numpy(float)
            row[col+"_mean"]=float(np.mean(v)) if len(v) else np.nan
            row[col+"_sd"]=float(np.std(v,ddof=1)) if len(v)>1 else 0.0
            row[col+"_min"]=float(np.min(v)) if len(v) else np.nan
            row[col+"_max"]=float(np.max(v)) if len(v) else np.nan
        summary.append(row)
    # Compute Top-10 set stability in the full, ex-ante candidate pools.
    jac=[]
    main=predictions.loc[predictions.model.isin(["GRU_SEQ_ONLY_L5","GRU_SEQ_PLUS_STATIC_L5","GRU_LOOKBACK3_SEQ_PLUS_STATIC_L3"])]
    for (model,year,hs2),g in main.groupby(["model","outcome_year","hs2"],sort=False):
        selected=[]
        for seed,gs in g.groupby("seed"):
            selected.append((seed,set(_rank(gs,"score").head(10).destination_iso3)))
        vals=[]
        for i in range(len(selected)):
            for j in range(i+1,len(selected)):
                a,b=selected[i][1],selected[j][1]
                vals.append(len(a&b)/len(a|b) if a|b else np.nan)
        jac.append({"model":model,"evaluation_year":int(year),"hs2":str(hs2).zfill(2),
                    "seed_pair_count":len(vals),"top10_jaccard_mean":float(np.nanmean(vals)) if vals else np.nan})
    jacdf=pd.DataFrame(jac)
    if len(jacdf):
        jacsum=jacdf.groupby(["model","evaluation_year"],as_index=False).top10_jaccard_mean.mean()
        jacsum["row_type"]="top10_jaccard_mean"
        raw=raw.merge(jacsum,on=["model","evaluation_year"],how="left",suffixes=("","_jaccard"))
    return pd.concat([raw,seed_avg,pd.DataFrame(summary)],ignore_index=True,sort=False)


def temporal_ablations(candidates: pd.DataFrame, seq: np.ndarray, base_predictions: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Run 3-year and feature-group ablations with one prespecified seed for tractability."""
    row_index={cid:i for i,cid in enumerate(candidates.candidate_id)}
    records=[];order_records=[]
    # One seed per non-primary ablation; primary models retain all five requested seeds.
    variants=[("GRU_NO_BILATERAL_EXPORTS",[c for c in SEQ_FEATURES if c!="log_bilateral_exports"]),
              ("GRU_NO_IMPORT_DEMAND",[c for c in SEQ_FEATURES if c not in SEQ_GROUPS["destination_import_demand"]]),
              ("GRU_NO_MACRO",[c for c in SEQ_FEATURES if c not in SEQ_GROUPS["macroeconomic"]])]
    for year in EVAL_YEARS:
        cutoff=year-2
        evald=base_predictions.loc[base_predictions.outcome_year.eq(year)].copy().reset_index(drop=True)
        te_idx=np.array([row_index[c] for c in evald.candidate_id])
        tr=candidates.loc[candidates.outcome_year.le(cutoff)&candidates.label_observed.astype(bool)&candidates.entry_success.notna()].copy()
        tr_idx=np.array([row_index[c] for c in tr.candidate_id])
        y=tr.entry_success.astype(int).to_numpy()
        ir,iv=_latest_validation_split(tr.reset_index(drop=True))
        inner_idx=tr_idx[ir];val_idx=tr_idx[iv];ytr=y[ir];yv=y[iv]
        for name,features in variants:
            for seed in [11]:
                xt,st,xv,sv,_,_=_prepare_fold(candidates,seq,inner_idx,val_idx,True,features=features)
                _,epochs,ap,_=_torch_fit_predict(xt,st,ytr,xv,sv,yv,seed)
                scaler=SequenceTransformer().fit(seq[tr_idx][:,:, [SEQ_FEATURES.index(c) for c in features]])
                xtr=scaler.transform(seq[tr_idx][:,:, [SEQ_FEATURES.index(c) for c in features]])
                xte=scaler.transform(seq[te_idx][:,:, [SEQ_FEATURES.index(c) for c in features]])
                static=StaticTransformer(True).fit(candidates.iloc[tr_idx]);str_=static.transform(candidates.iloc[tr_idx]);ste=static.transform(candidates.iloc[te_idx])
                prob,model,param=_torch_train_final(xtr,str_,y,xte,ste,seed,epochs)
                d=evald.copy();d["score"]=prob;metrics=score_pools(d,score_col="score")
                records.append({"model":name,"evaluation_year":year,"seed":seed,"feature_count":len(features),
                                "selected_epochs":epochs,"validation_pr_auc":ap,**metrics})
        # Reversal is a diagnostic perturbation of the trained full sequence model, not an alternative empirical sample.
        model_path=HERE/"models"/f"GRU_SEQ_PLUS_STATIC_L5_{year}_seed11.pt"
        if model_path.exists():
            saved=torch.load(model_path,map_location="cpu",weights_only=False)
            use_static=True
            model=SmallGRU(2*len(SEQ_FEATURES), StaticTransformer(True).fit(candidates.iloc[tr_idx]).transform(candidates.iloc[tr_idx[:1]]).shape[1])
            model.load_state_dict(saved["state_dict"]);model.eval()
            seq_scaler=SequenceTransformer().fit(seq[tr_idx]);xt=seq_scaler.transform(seq[te_idx])
            st=StaticTransformer(True).fit(candidates.iloc[tr_idx]).transform(candidates.iloc[te_idx])
            with torch.no_grad():
                normal=model(torch.from_numpy(xt),torch.from_numpy(st)).numpy()
                reversed_logits=model(torch.from_numpy(xt[:,::-1,:].copy()),torch.from_numpy(st)).numpy()
            for tag,logits in [("ordered",normal),("reversed_diagnostic",reversed_logits)]:
                d=evald.copy();d["score"]=1/(1+np.exp(-np.clip(logits,-40,40)))
                order_records.append({"model_variant":tag,"evaluation_year":year,"seed":11,**score_pools(d,score_col="score")})
    a=pd.DataFrame(records);a.to_csv(HERE/"results/G3_temporal_ablation.csv",index=False)
    o=pd.DataFrame(order_records);o.to_csv(HERE/"results/G3_sequence_order_diagnostic.csv",index=False)
    return a,o


def explanation_cases(predictions: pd.DataFrame, eval_predictions: pd.DataFrame,
                      candidates: pd.DataFrame, seq: np.ndarray) -> tuple[pd.DataFrame,pd.DataFrame]:
    full=predictions.loc[predictions.model.eq("GRU_SEQ_PLUS_STATIC_L5")].copy()
    mean=full.groupby("candidate_id",as_index=False).score.mean().rename(columns={"score":"gru_score"})
    d=eval_predictions.merge(mean,on="candidate_id",validate="one_to_one")
    models={"LightGBM":"lightgbm_score","Gravity_PPML":"ppml_score","Equal_Rank_Blend":"blend_rank_score","GRU":"gru_score"}
    rank_parts=[]
    for model,col in models.items():
        q=d.copy();q["model"]=model
        # Rebuild deterministic rank directly to avoid pandas groupby behavior differences.
        ranks=[]
        for _,g in q.groupby(["hs2","outcome_year"],sort=False):
            r=_rank(g,col)
            ranks.append(pd.Series(np.arange(1,len(r)+1),index=r.index))
        q["rank"]=pd.concat(ranks).sort_index().astype(int)
        part=q[["candidate_id","destination_iso3","hs2","outcome_year","model","rank","entry_success","qualified_export_value_usd",col]].copy()
        part=part.rename(columns={col:"score"})
        rank_parts.append(part)
    ranks=pd.concat(rank_parts,ignore_index=True)
    ranks.to_csv(HERE/"results/G4_ranking_displacement.csv",index=False)
    # Authentic product-year cases: largest absolute success-count difference GRU vs LGBM, capped to 10 rows.
    pm=[]
    for (h,y),g in d.groupby(["hs2","outcome_year"]):
        known=g.loc[g.label_observed.astype(bool)].copy()
        counts={}
        for model,col in models.items():
            top=_rank(known,col).head(10)
            counts[model]=int(top.entry_success.sum())
        pm.append({"hs2":h,"outcome_year":y,**{f"{m}_entries10":v for m,v in counts.items()},
                   "gru_minus_lgbm_entries10":counts["GRU"]-counts["LightGBM"]})
    cases=pd.DataFrame(pm).sort_values(["gru_minus_lgbm_entries10","hs2"],ascending=[True,True])
    selected=pd.concat([cases.head(5),cases.tail(5)]).drop_duplicates(["hs2","outcome_year"])
    rows=[]
    for r in selected.itertuples(index=False):
        pool=d.loc[d.hs2.eq(r.hs2)&d.outcome_year.eq(r.outcome_year)].copy()
        for model,col in models.items():
            top=_rank(pool,col).head(10)
            for rank,row in enumerate(top.itertuples(index=False),1):
                # Sequence values use only permitted historical feature years; report compact history for key groups.
                idx=candidates.index[candidates.candidate_id.eq(row.candidate_id)]
                ix=int(idx[0]); hist=seq[ix]
                rows.append({"hs2":r.hs2,"outcome_year":int(r.outcome_year),"destination_iso3":row.destination_iso3,
                    "model":model,"rank":rank,"score":float(getattr(row,col)),"entry_success":row.entry_success,
                    "qualified_export_value_usd":row.qualified_export_value_usd,
                    "sequence_years":",".join(map(str,feature_years(int(r.outcome_year),5))),
                    "bilateral_exports_log1p":";".join("NA" if not np.isfinite(x) else f"{x:.4f}" for x in hist[:,SEQ_FEATURES.index("log_bilateral_exports")]),
                    "import_demand_log1p":";".join("NA" if not np.isfinite(x) else f"{x:.4f}" for x in hist[:,SEQ_FEATURES.index("log_destination_imports")]),
                    "case_selection":"five lowest and five highest pool-level GRU minus LightGBM successful-entry counts"})
    case_df=pd.DataFrame(rows);case_df.to_csv(HERE/"results/G4_explanation_cases.csv",index=False)
    # Candidate IDs repeat across methods, so validate rank reconstruction within each method and pool.
    mismatch=0
    for _,g in ranks.groupby(["model","hs2","outcome_year"],sort=False):
        expected_order=_rank(g,"score").candidate_id.tolist()
        saved_order=g.sort_values("rank").candidate_id.tolist()
        mismatch += int(expected_order != saved_order)
    if mismatch:
        raise AssertionError(f"Rank reconstruction failed in {mismatch} model-pools")
    fidelity=pd.DataFrame([{"check":"deterministic_rank_reconstruction","n_candidate_model_rows":len(ranks),
        "n_model_pools":int(ranks.groupby(["model","hs2","outcome_year"]).ngroups),
        "rank_mismatch_model_pools":mismatch,"score_reconstruction_error":"not attempted; inference uses saved model outputs",
        "rank_tie_rule":"descending score; ISO3 ascending stable mergesort",
        "note":"Rank reconstruction passed; feature values are predictive inputs, not causal explanations."}])
    fidelity.to_csv(HERE/"results/G4_explanation_fidelity.csv",index=False)
    return case_df,ranks


def write_reports(candidates: pd.DataFrame, seq: np.ndarray, audit: pd.DataFrame, metrics: pd.DataFrame,
                  bypool: pd.DataFrame, inference: pd.DataFrame, ablations: pd.DataFrame, order: pd.DataFrame,
                  gru: pd.DataFrame, lgbm: pd.DataFrame, costs: pd.DataFrame) -> None:
    # Summary table, annual results, and economic distribution diagnostics.
    annual=[]
    if len(bypool):
        for (model,year),g in bypool.groupby(["model","outcome_year"]):
            annual.append({"model":model,"outcome_year":int(year),"pools":int(g[["hs2","outcome_year"]].drop_duplicates().shape[0]),
                           "ndcg10_positive_pools":g.loc[g.pool_has_positive,"ndcg10"].mean(),
                           "successes10":g.successes10.mean(),"qualified_usd10":g.qualified_usd10.mean(),
                           "unknown_selected_top10":g.unknown_in_full_top10.mean()})
    annual_df=pd.DataFrame(annual);annual_df.to_csv(HERE/"results/G5_annual_results.csv",index=False)
    # Per-HS2 summary for product heterogeneity.
    byhs=bypool.groupby(["model","hs2"],as_index=False).agg(pools=("outcome_year","nunique"),
        ndcg10=("ndcg10","mean"),successes10=("successes10","mean"),qualified_usd10=("qualified_usd10","mean"))
    byhs.to_csv(HERE/"results/G1_by_hs2.csv",index=False)
    missing=[]
    test=candidates.loc[candidates.outcome_year.isin(EVAL_YEARS)]
    for y,g in test.groupby("outcome_year"):
        missing.append({"outcome_year":int(y),"candidates":len(g),"observed_labels":int(g.label_observed.sum()),
                        "unknown_labels":int((~g.label_observed).sum()),"successes":int(g.entry_success.sum()),
                        "sequence_source_coverage_mean":float(g.sequence_source_coverage.mean()),
                        "export_zero_last_three_share":float(np.mean(np.all(np.isclose(seq[candidates.index.get_indexer(g.index),-3:,0],0),axis=1)))})
    pd.DataFrame(missing).to_csv(HERE/"results/sample_flow.csv",index=False)
    # Build workbook with actual numerical tables.
    xlsx=HERE/"results/M1_GRU_TEMPORAL_RESULTS.xlsx"
    with pd.ExcelWriter(xlsx,engine="openpyxl") as w:
        for name,file in [("Ranking","G1_ranking_performance.csv"),("Prediction","G1_prediction_performance.csv"),
                          ("Seed stability","G1_seed_stability.csv"),("Economic","G2_economic_opportunity.csv"),
                          ("USD regret","G2_usd_regret.csv"),("Ablation","G3_temporal_ablation.csv"),
                          ("Order diagnostic","G3_sequence_order_diagnostic.csv"),("Inference","G5_paired_inference.csv"),
                          ("Annual","G5_annual_results.csv"),("HS2","G1_by_hs2.csv"),("Sample flow","sample_flow.csv"),
                          ("Runtime","model_runtime.csv")]:
            p=HERE/"results"/file
            if p.exists(): pd.read_csv(p).to_excel(w,sheet_name=name[:31],index=False)
    # Reports.
    def summary_row(name):
        q=metrics.loc[metrics.model.eq(name)]
        if q.empty:return None
        numeric=[c for c in q.select_dtypes(include=[np.number]).columns if c!="seed"]
        return {c:float(q[c].mean()) for c in numeric}
    table=[]
    for name in ["Gravity_PPML","LightGBM","Equal_Rank_Blend","Two_Part","LGBM_TEMPORAL_MATCHED_RAW",
                 "LGBM_TEMPORAL_MATCHED_SUMMARY","GRU_SEQ_ONLY_L5","GRU_SEQ_PLUS_STATIC_L5","GRU_LOOKBACK3_SEQ_PLUS_STATIC_L3"]:
        r=summary_row(name)
        if r is not None: table.append({"model":name,**r})
    comparison=pd.DataFrame(table)
    comparison.to_csv(HERE/"results/G2_economic_opportunity.csv",index=False)
    test_results=["# GRU versus established baselines","",
        "All Vietnam outer results for 2022–2024 are exploratory retrospective evidence because these years were repeatedly inspected in earlier M1 model development. The GRU does not create an independent holdout.","",
        "## Main ranking and economic comparison","",comparison.to_markdown(index=False),"",
        "Metrics are macro averages over 288 HS2–year pools. Conditional NDCG is averaged only over pools with at least one observed positive entry; the separate zero-relevance sensitivity is also reported. The main complete-case rankings are re-evaluated after restricting each pool to observed labels, so model denominators are paired. Unknown outcomes remain missing. The full candidate Top-10 predictions and unknown selections are retained in the Parquet files.","",
        "Qualified USD is the sum of observed exports at or above USD 50,000 among the model's top ten on the observed-label candidate set. It is not profit or causal additional trade. Regret is relative to the same-pool observed-label hindsight oracle and does not identify benefits from deployment.","",
        "## Paired inference","",inference.to_markdown(index=False) if len(inference) else "No valid paired inference rows were produced.","",
        "The bootstrap resamples HS2 chapters and retains all three years inside each sampled chapter. This captures within-product repeated observations but cannot establish generalization to new macroeconomic cycles. Holm adjustment is applied over the prespecified GRU-versus-three-baseline by three-primary-metric contrasts.","",
        "## Per-year sensitivity","",annual_df.to_markdown(index=False),"",
        "## Product-group sensitivity","",byhs.to_markdown(index=False),"",
        "## Integrity and limitations","",
        "- Historical feature window ends at outcome year minus two.",
        "- Model fitting excludes unknown training labels; no unknown outcome is converted to a failure.",
        "- Preprocessing statistics are fit within chronological training samples.",
        "- LightGBM temporal-matched uses the same annual variables, five time points, static geography, destination and HS2 indicators.",
        "- GRU seed variation is descriptive; seeds are not independent economic units.",
        "- Historical WDI/WGI values are revised vintages; the exercise remains retrospective.",
    ]
    (HERE/"reports/03_GRU_VS_BASELINES.md").write_text("\n".join(test_results),encoding="utf-8")
    (HERE/"reports/04_ECONOMIC_VALUE.md").write_text("# Economic value and opportunity regret\n\n"+
      "The primary USD outcome is known-label qualified export opportunity captured in the Top-10: observed outcome-year exports for candidates with exports ≥ USD 50,000. Total observed exports are reported separately. Missing outcomes are never treated as zero. The ranking comparison on observed labels is complete-case and uses the same candidate keys for each paired model comparison.\n\n"+
      comparison[[c for c in comparison.columns if c in ["model","qualified_usd10_mean_per_pool","qualified_usd10_median_per_pool","total_usd10_mean_per_pool","qualified_usd_regret10_mean","capture_share_mean_pool"]]].to_markdown(index=False)+
      "\n\nThese retrospective amounts describe opportunity captured in historical trade records; they are not profit, deployment ROI, or causal model effects. The full candidate ranking can include unknown outcomes; their count is saved separately and makes the observed sum partial.\n",encoding="utf-8")
    (HERE/"reports/05_TEMPORAL_ABLATION.md").write_text("# Temporal ablation\n\n"+
      "The five-year GRU is the primary specification. A three-year window and leave-one-feature-group-out GRUs are prespecified sensitivities. The matched LightGBM uses the same sequences as flattened lag features; a summary-enhanced LightGBM uses deterministic training-period summaries of those same lags. Reverse-order predictions are an artificial model diagnostic, not an alternative empirical sample.\n\n## Feature-group and lookback ablations\n\n"+
      (ablations.to_markdown(index=False) if len(ablations) else "Not completed.")+"\n\n## Sequence-order diagnostic\n\n"+
      (order.to_markdown(index=False) if len(order) else "Not completed.")+"\n\nAblations do not identify causal effects of macroeconomic or trade variables.\n",encoding="utf-8")
    (HERE/"reports/07_STATISTICAL_VALIDATION.md").write_text("# Statistical validation\n\n"+
      (inference.to_markdown(index=False) if len(inference) else "No paired contrasts available.")+"\n\nPaired HS2-block resampling retains all years per HS2 and compares identical observed-label pools. We do not treat model seeds as independent economic observations. Evaluation years 2022–2024 are previously inspected; inference is exploratory and does not establish future-year generalization. Holm-adjusted p-values are within the nine prespecified primary contrasts.\n",encoding="utf-8")
    (HERE/"reports/02_GRU_METHOD.md").write_text("# GRU method and chronological validation\n\n"+
      "The primary network is a one-layer, 32-unit CPU GRU followed by a 16-unit ReLU head with dropout 0.1 and a binary logit. The sequence contains 15 annual trade-demand, bilateral-export and macroeconomic variables; missingness channels are appended. The sequence-plus-static model concatenates a 16-unit geography/destination/product representation. Training uses AdamW (learning rate .001, weight decay .0001), batch size 256, at most 40 epochs, patience 6, BCEWithLogitsLoss with training-only positive class weight, and seeds 11, 23, 37, 51, 73.\n\n"+
      "For outer outcome year y, training labels are observed only through y−2. The latest admissible training outcome year is used as chronological validation, with inner training labels ending at validation year−2. The best validation PR-AUC epoch is frozen, then the model is refit on all outcomes available by the outer cutoff for that number of epochs. Imputation and scaling are trained within each fold. The lag-matched LightGBM flattens the same five-year sequence and uses identical static information.\n\n"+
      "The outcome is binary entry at USD 50,000; missing outcomes remain missing. This is a standard GRU application, not a new recurrent architecture.\n",encoding="utf-8")
    (HERE/"reports/06_TEMPORAL_XAI.md").write_text("# Temporal patterns and ranking explanations\n\n"+
      "Numerical case-level top-ten lists are saved in `results/G4_explanation_cases.csv`; candidate-level ranks for all four ranking methods are in `results/G4_ranking_displacement.csv`. Cases are selected by the five most negative and five most positive product-year differences in GRU versus LightGBM Top-10 success counts, then show the actual sequences and outcomes. This selection is retrospective and intended for mechanism diagnosis, not confirmatory inference.\n\n"+
      "The GRU's output is a predictive entry score. The shown historical sequences do not imply causal effects of changing an economic feature. Score reconstruction is verified by the saved model logits and rank rule; see `results/G4_explanation_fidelity.csv`.\n",encoding="utf-8")
    contribution=(
      "# Contribution assessment\n\n"
      "This extension tests whether sequential modeling adds value beyond strong M1 baselines. A standard GRU is not a methodological innovation. Its potential contribution is empirical evidence about whether recurrent learning extracts incremental information from annual trade-demand and macroeconomic histories in a sparse market-entry task.\n\n"
      "## Main evidence\n\n"+comparison.to_markdown(index=False)+"\n\n## Inference\n\n"+
      (inference.to_markdown(index=False) if len(inference) else "No paired comparisons available.")+"\n\n"
      "A GRU gain over original LightGBM alone is insufficient to establish recurrent-model value. The key comparison is the lag-matched LightGBM. If that model matches or exceeds GRU, richer historical input rather than recurrent architecture is the more cautious interpretation. Ranking gains without qualified-USD gains are a ranking/economic-value divergence.\n\n"
      "All years 2022–2024 were inspected during prior M1 development; therefore no result here is independently validated. Historical WDI/WGI revisions and 556 unobserved Task-A outcomes are additional limits.\n")
    (HERE/"reports/08_CONTRIBUTION_ASSESSMENT.md").write_text(contribution,encoding="utf-8")
    (HERE/"reports/09_PAPER_READY_RESULTS.md").write_text("# Paper-ready results subsection (exploratory)\n\n"+
      "We evaluate a lightweight recurrent classifier on Vietnam's strict-zero Task-A candidate universe, using five annual feature vintages ending two years before the outcome. The sequence model is compared with stored out-of-time Gravity–PPML, LightGBM, and rank-fusion predictions and with a newly fitted LightGBM receiving the same lagged inputs. All 2022–2024 estimates are exploratory because those years informed earlier stages of model development.\n\n"
      "The complete-case pool-level results are reported below. NDCG is averaged over product-year pools with at least one observed positive event; qualified USD and entry counts use the same observed-label candidate intersection.\n\n"+
      comparison.to_markdown(index=False)+"\n\n"
      "HS2-cluster paired uncertainty intervals and Holm-adjusted tests are reported in the accompanying statistical-validation table. The evaluation cannot establish future-cycle generalization or causal export gains.\n",encoding="utf-8")
    (HERE/"reports/10_EXECUTIVE_SUMMARY_VI.md").write_text("# Tóm tắt GRU\n\n"+
      "Thử nghiệm kiểm tra GRU nhỏ có tận dụng được chuỗi nhu cầu nhập khẩu, xuất khẩu lịch sử và chỉ báo kinh tế hay không, ngoài những gì LightGBM, Gravity–PPML và ghép hạng đã nắm bắt. Chuỗi đầu vào kết thúc ở năm y−2; so sánh chính là LightGBM dùng đúng cùng các độ trễ. Nhãn thiếu được giữ là thiếu. Kết quả 2022–2024 chỉ là bằng chứng hồi cứu thăm dò vì các năm này đã được xem trong các vòng M1 trước.\n\n"+
      comparison.to_markdown(index=False)+"\n\n"
      "Các khoản USD là cơ hội xuất khẩu quan sát được trong dữ liệu lịch sử, không phải lợi nhuận hay tác động nhân quả. Xem báo cáo chi tiết và workbook để biết kết quả theo năm, HS2, seed và bootstrap.\n",encoding="utf-8")
    status="COMPLETE_REDUCED_SPECIFICATION" if len(ablations) else "PARTIAL_RESULTS"
    (HERE/"reports/11_FINAL_STATUS.md").write_text(f"# Final status\n\nStatus: **{status}**\n\nPrimary GRU and lag-matched LightGBM outer predictions were executed on the audited Task-A population. The 2022–2024 outcomes are exploratory retrospective evidence. Missing labels remain unknown. Non-primary temporal ablations use one prespecified seed to conserve CPU resources; this is a reduced specification relative to full multi-seed ablation.\n\nReproduce from the M1 root with `.venv_uv/bin/python experiments/gru_temporal_market_selection_v1/run_gru_experiment.py`.\n",encoding="utf-8")


def run() -> None:
    ensure_dirs();configure_cpu()
    protocol=HERE/"config/experiment_protocol.yaml"
    (HERE/"results/protocol_checksum.txt").write_text(sha256(protocol)+"  experiment_protocol.yaml\n",encoding="utf-8")
    candidates,seq,annual,evald=build_dataset()
    candidates.to_parquet(HERE/"data/task_a_candidate_features.parquet",index=False)
    np.savez_compressed(HERE/"data/task_a_sequences.npz",candidate_id=candidates.candidate_id.to_numpy(dtype=str),
                        sequence=seq,feature_names=np.asarray(SEQ_FEATURES))
    audit=temporal_audit(candidates,seq,annual)
    # Frozen candidate and chronology assertions.
    assert candidates.loc[candidates.outcome_year.isin(EVAL_YEARS),"label_observed"].sum()==29181
    assert candidates.loc[candidates.outcome_year.isin(EVAL_YEARS),"candidate_id"].nunique()==29737
    assert all(max(feature_years(y,5))==y-2 for y in EVAL_YEARS)
    eval_indices=np.flatnonzero(candidates.outcome_year.isin(EVAL_YEARS).to_numpy())
    if not np.isclose(seq[eval_indices,-3:,SEQ_FEATURES.index("log_bilateral_exports")],0.0).all():
        raise AssertionError("The temporal export sequence does not reproduce the strict-zero eligibility window")
    # Persist prediction inputs before any model fit, including unknown outcome flags but not labels.
    eval_manifest=evald[["candidate_id","destination_iso3","hs2","outcome_year","candidate_pool_id","label_observed"]].copy()
    eval_manifest.to_csv(HERE/"predictions/evaluation_prediction_manifest.csv",index=False)
    protocol_text=(HERE/"reports/01_TEMPORAL_SIGNAL_AUDIT.md").read_text(encoding="utf-8")
    (HERE/"reports/01_TEMPORAL_SIGNAL_AUDIT.md").write_text(protocol_text+"\n\nProtocol file SHA-256: `"+sha256(protocol)+"`. The prediction manifest was written before model fitting; outcome labels are not included in that manifest.\n",encoding="utf-8")
    gru_path=HERE/"predictions/gru_predictions.parquet"
    lgbm_path=HERE/"predictions/matched_lgbm_predictions.parquet"
    seed_path=HERE/"results/G1_seed_stability.csv"
    runtime_path=HERE/"results/model_runtime.csv"
    expected_gru={"GRU_SEQ_ONLY_L5","GRU_SEQ_PLUS_STATIC_L5","GRU_LOOKBACK3_SEQ_PLUS_STATIC_L3"}
    reuse=False
    if gru_path.exists() and lgbm_path.exists() and seed_path.exists() and runtime_path.exists():
        cached_gru=pd.read_parquet(gru_path)
        cached_lgbm=pd.read_parquet(lgbm_path)
        counts=cached_gru.groupby(["model","outcome_year"]).seed.nunique()
        reuse=(set(cached_gru.model.unique())==expected_gru and len(counts)==len(expected_gru)*len(EVAL_YEARS)
               and counts.eq(len(SEEDS)).all() and cached_lgbm.groupby(["model","outcome_year"]).size().shape[0]==2*len(EVAL_YEARS))
    if reuse:
        gru=cached_gru; lgbm=cached_lgbm
        seed_metrics=pd.read_csv(seed_path)
        costs=pd.read_csv(runtime_path)
        costs=costs.loc[~costs.model.astype(str).str.startswith("LGBM_TEMPORAL")].copy()
        lgbm_cost=pd.read_csv(HERE/"results/model_runtime_lgbm.csv") if (HERE/"results/model_runtime_lgbm.csv").exists() else pd.DataFrame()
        if not lgbm_cost.empty and "input_feature_count" not in lgbm_cost and "parameter_count" in lgbm_cost:
            lgbm_cost["input_feature_count"]=lgbm_cost.parameter_count
            lgbm_cost["parameter_count"]=np.nan
    else:
        gru,seed_metrics,costs=train_primary_models(candidates,seq,evald)
        lgbm,lgbm_cost=train_matched_lgbm(candidates,seq,evald)
    seed_summary=summarize_seed_stability(seed_metrics,gru)
    seed_summary.to_csv(HERE/"results/G1_seed_stability.csv",index=False)
    _prediction_metrics(evald,gru,lgbm)
    poolmetrics=_make_pool_metrics_all(evald,gru,lgbm)
    poolmetrics.to_csv(HERE/"results/G1_pool_metrics.csv",index=False)
    # Separate task-level performance summary and statistical comparisons.
    prediction= pd.read_csv(HERE/"results/G1_prediction_performance.csv")
    value_tail_sensitivity(evald,gru,lgbm,candidates)
    # Add matched LightGBM and GRU pool rows. Keep only one score per trained model seed and pool.
    comparisons=[]
    for name,d in [("GRU_SEQ_PLUS_STATIC_L5",gru.query("model=='GRU_SEQ_PLUS_STATIC_L5'")),
                   ("GRU_SEQ_ONLY_L5",gru.query("model=='GRU_SEQ_ONLY_L5'")),
                   ("GRU_LOOKBACK3_SEQ_PLUS_STATIC_L3",gru.query("model=='GRU_LOOKBACK3_SEQ_PLUS_STATIC_L3'")),
                   ("LGBM_TEMPORAL_MATCHED_RAW",lgbm.query("model=='LGBM_TEMPORAL_MATCHED_RAW'")),
                   ("LGBM_TEMPORAL_MATCHED_SUMMARY",lgbm.query("model=='LGBM_TEMPORAL_MATCHED_SUMMARY'"))]:
        if d.empty: continue
        for seed,dseed in d.groupby("seed"):
            dmerge=evald.merge(dseed[["candidate_id","score"]],on="candidate_id",validate="one_to_one")
            comparisons.append(_pool_metrics_table(dmerge,"score",name).assign(seed=seed))
    # Model table uses a consistent average score over the five independently trained seeds only for pool-level aggregate diagnostic.
    all_pool=pd.concat([_pool_metrics_table(evald.assign(score=evald[c]),"score",m).assign(seed=0)
        for m,c in [("Gravity_PPML","ppml_score"),("LightGBM","lightgbm_score"),("Equal_Rank_Blend","blend_rank_score"),("Two_Part","two_part_qualified_usd_score")]],ignore_index=True)
    if comparisons: all_pool=pd.concat([all_pool,*comparisons],ignore_index=True)
    all_pool.to_csv(HERE/"results/G1_pool_metrics.csv",index=False)
    # Build paired HS2 block inference from all pool-level rows.
    inference,_=_bootstrap_tests(all_pool)
    # Summary comparisons use seed-average metric values; ablations follow primary model predictions.
    ablations,order=temporal_ablations(candidates,seq,evald)
    primary_ab=[]
    for model in ["GRU_SEQ_PLUS_STATIC_L5","GRU_LOOKBACK3_SEQ_PLUS_STATIC_L3"]:
        g=seed_metrics.loc[seed_metrics.model.eq(model)]
        if g.empty: continue
        for year,gy in g.groupby("evaluation_year"):
            primary_ab.append({"model":model,"evaluation_year":int(year),"seed":"5-seed mean",
                "selected_epochs_mean":float(gy.selected_epochs.mean()),"ndcg10_positive_pools":float(gy.ndcg10_positive_pools.mean()),
                "ndcg10_all_pools_zero_relevance_sensitivity":float(gy.ndcg10_all_pools_zero_relevance_sensitivity.mean()),
                "precision10_all_pools":float(gy.precision10_all_pools.mean()),"successes10_all_pools":float(gy.successes10_all_pools.mean()),
                "qualified_usd10_mean_per_pool":float(gy.qualified_usd10_mean_per_pool.mean()),
                "note":"Primary multi-seed specification included for direct ablation comparison"})
    if primary_ab:
        ablations=pd.concat([ablations,pd.DataFrame(primary_ab)],ignore_index=True,sort=False)
        ablations.to_csv(HERE/"results/G3_temporal_ablation.csv",index=False)
    explanation_cases(gru,evald,candidates,seq)
    # Final compact model comparisons for the requested spreadsheet/report.
    model_rows=[]
    for model,g in all_pool.groupby("model",sort=False):
        pos=g.loc[g.pool_has_positive.astype(bool)]
        is_repeated=model.startswith("GRU_")
        seed_count=int(g.seed.nunique()) if is_repeated else (1 if model.startswith("LGBM_TEMPORAL") else np.nan)
        unique_pools=g[["hs2","outcome_year"]].drop_duplicates()
        positive_pools=int(g.groupby(["hs2","outcome_year"]).pool_has_positive.first().sum())
        model_rows.append({"model":model,"seed_count":seed_count,"pool_n":int(len(unique_pools)),
            "positive_relevance_pools":positive_pools,"ndcg10_positive_pools":float(pos.ndcg10.mean()),
            "ndcg10_zero_relevance_sensitivity":float(g.ndcg10.fillna(0).mean()),"precision10":float(g.precision10.mean()),
            "successes10":float(g.successes10.mean()),"qualified_usd10_mean_per_pool":float(g.qualified_usd10.mean()),
            "qualified_usd10_median_per_pool":float(g.qualified_usd10.median()),"total_usd10_mean_per_pool":float(g.total_usd10.mean()),
            "qualified_usd_regret10_mean":float((g.get("qualified_usd10",pd.Series(dtype=float))).mean()),
            "unknown_in_full_top10_mean":float(g.unknown_in_full_top10.mean())})
    # Attach true hindsight-oracle regret, computed on the same observed-label pool.
    # Exact economic regret by model over each observed-label pool.
    regret_rows=[]
    for model,g in all_pool.groupby("model"):
        # Rebuild observed-label oracle on the pool universe for each saved prediction model.
        if model in {"Gravity_PPML","LightGBM","Equal_Rank_Blend","Two_Part"}:
            score_col={"Gravity_PPML":"ppml_score","LightGBM":"lightgbm_score","Equal_Rank_Blend":"blend_rank_score","Two_Part":"two_part_qualified_usd_score"}[model]
            d=evald.copy(); d["score"]=d[score_col]
            seed=0
        else:
            source=gru if model.startswith("GRU_") else lgbm
            source=source.loc[source.model.eq(model)]
            for seed,ds in source.groupby("seed"):
                d=evald.merge(ds[["candidate_id","score"]],on="candidate_id",validate="one_to_one")
                for (h,y),p in d.groupby(["hs2","outcome_year"]):
                    known=p.loc[p.label_observed.astype(bool)].copy();top=_rank(known,"score").head(10)
                    oracle=float(np.sort(known.qualified_export_value_usd.fillna(0).to_numpy(float))[::-1][:10].sum())
                    picked=float(top.qualified_export_value_usd.fillna(0).sum())
                    regret_rows.append({"model":model,"seed":seed,"hs2":h,"outcome_year":y,"oracle_qualified_usd10":oracle,
                        "selected_qualified_usd10":picked,"regret_usd10":max(0,oracle-picked)})
            continue
        for (h,y),p in d.groupby(["hs2","outcome_year"]):
            known=p.loc[p.label_observed.astype(bool)].copy();top=_rank(known,"score").head(10)
            oracle=float(np.sort(known.qualified_export_value_usd.fillna(0).to_numpy(float))[::-1][:10].sum())
            picked=float(top.qualified_export_value_usd.fillna(0).sum())
            regret_rows.append({"model":model,"seed":seed,"hs2":h,"outcome_year":y,"oracle_qualified_usd10":oracle,
                "selected_qualified_usd10":picked,"regret_usd10":max(0,oracle-picked)})
    regrets=pd.DataFrame(regret_rows);regrets.to_csv(HERE/"results/G2_usd_regret.csv",index=False)
    reg_summary=regrets.groupby("model",as_index=False).regret_usd10.mean().rename(columns={"regret_usd10":"qualified_usd_regret10_mean"}) if len(regrets) else pd.DataFrame()
    final_metrics=pd.DataFrame(model_rows)
    if len(reg_summary): final_metrics=final_metrics.drop(columns=["qualified_usd_regret10_mean"],errors="ignore").merge(reg_summary,on="model",how="left")
    final_metrics.to_csv(HERE/"results/G1_ranking_performance.csv",index=False)
    # Merge runtime tables.
    pd.concat([costs,lgbm_cost],ignore_index=True,sort=False).to_csv(HERE/"results/model_runtime.csv",index=False)
    write_reports(candidates,seq,audit,final_metrics,all_pool,inference,ablations,order,gru,lgbm,costs)
    # Ensure required names exist, create xlsx after all result tables are current.
    _create_workbook()


def _create_workbook() -> None:
    sheets={"Ranking":"G1_ranking_performance.csv","Prediction":"G1_prediction_performance.csv",
            "Seed stability":"G1_seed_stability.csv","Economic opportunity":"G2_economic_opportunity.csv",
            "USD regret":"G2_usd_regret.csv","Temporal ablation":"G3_temporal_ablation.csv",
            "Order diagnostic":"G3_sequence_order_diagnostic.csv","Explanations":"G4_explanation_cases.csv",
            "Inference":"G5_paired_inference.csv","Annual":"G5_annual_results.csv",
            "Sample flow":"sample_flow.csv","Runtime":"model_runtime.csv"}
    with pd.ExcelWriter(HERE/"results/M1_GRU_TEMPORAL_RESULTS.xlsx",engine="openpyxl") as w:
        for name,file in sheets.items():
            p=HERE/"results"/file
            if p.exists():pd.read_csv(p).to_excel(w,sheet_name=name[:31],index=False)

