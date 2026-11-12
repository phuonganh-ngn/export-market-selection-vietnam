"""Re-fit the rolling-origin LightGBM experts for 2018-2024 with the configured seeds and bagging.

Discovery: entry classifier, tuned once per outcome year (chronological inner split), fitted with every
configured seed; the score is the mean positive-class probability. Training population follows
study.entry_training_population. Expansion: log1p regressor, same procedure; the score is the positive
part of (mean predicted USD - historical exports). PPML is deterministic and is not re-fitted here.

Writes predictions/lgbm_refit_scores.parquet (one row per candidate, per-seed and mean scores).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import joblib

from revision_lib import CFG, HERE, PRED_DIR, TA, TB, zfill_hs2  # also sets sys.path
from experiments.explainable_reliability_gate_v1.src.task_audit import load_and_freeze  # write=False writes nothing
from market_discovery import entry_training_mask
from ml_models import fit_classifier, fit_regressor, lightgbm_bagging_params, tune_classifier, tune_regressor

YEARS = list(range(2018, 2025))


MODEL_DIR = HERE / "models" / "lightgbm"


def main(save_models: bool = False) -> pd.DataFrame:
    """save_models=True writes every fitted seed model (preprocessor + estimator) to models/lightgbm/."""
    seeds = [int(s) for s in CFG["study"]["seeds"]]
    if save_models:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    panel, _ = load_and_freeze(write=False)
    panel = zfill_hs2(panel)
    print(f"panel {panel.shape}; seeds {seeds}; bagging {lightgbm_bagging_params()}; "
          f"population {CFG['study'].get('entry_training_population')}; excluded {CFG['study'].get('excluded_features')}", flush=True)
    rows = []
    for year in YEARS:
        cutoff = year - 2
        train = panel.loc[panel.outcome_observed.astype(bool) & panel.export_value_usd.notna()
                          & panel.destination_import_demand.gt(0) & panel.outcome_year.le(cutoff)].copy()
        test = panel.loc[panel.outcome_year.eq(year) & panel.destination_import_demand.gt(0)].copy()
        assert int(train.outcome_year.max()) <= cutoff and not (train.feature_year > train.outcome_year - 2).any(), "leakage"
        # discovery
        et = train.loc[entry_training_mask(train) & train.entry_success.notna()].copy()
        params, _ = tune_classifier("LightGBM", et, "entry_success", lag=2, trials=CFG["study"]["tuning_trials"])
        ex = test.loc[test.entry_candidate_ex_ante.astype(bool)].copy()
        probs = []
        for s in seeds:
            m = fit_classifier("LightGBM", et, "entry_success", params, s)
            if save_models:
                joblib.dump({"bundle": m, "params": params, "year": year, "seed": s, "task": TA,
                             "train_outcome_max": int(et.outcome_year.max())}, MODEL_DIR / f"discovery_{year}_seed{s}.joblib")
            probs.append(m.estimator.predict_proba(m.preprocessor.transform(ex[m.numeric_features + m.categorical_features]).astype(np.float32))[:, 1])
        probs = np.column_stack(probs)
        strict = ex.strict_inactive_verified.astype(bool).to_numpy()
        a = ex.loc[strict, ["destination_iso3", "hs2", "outcome_year"]].assign(task=TA)
        for i, s in enumerate(seeds):
            a[f"seed_{s}"] = probs[strict, i]
        a["lightgbm_score"] = probs[strict].mean(1)
        a["num_leaves"], a["train_rows"] = params.get("num_leaves"), len(et)
        rows.append(a)
        # expansion
        rp, _ = tune_regressor("LightGBM", train, lag=2, trials=CFG["study"]["tuning_trials"])
        usd = []
        for s in seeds:
            reg = fit_regressor("LightGBM", train, rp, s, lag=2)
            if save_models:
                joblib.dump({"bundle": reg, "params": rp, "year": year, "seed": s, "task": TB,
                             "train_outcome_max": int(train.outcome_year.max())}, MODEL_DIR / f"expansion_{year}_seed{s}.joblib")
            usd.append(reg.predict(test, log_target=True))
        usd = np.column_stack(usd)
        mb = test.task_b_eligible.astype(bool).to_numpy()
        hist = test.historical_export_value.to_numpy(float)[mb]
        b = test.loc[mb, ["destination_iso3", "hs2", "outcome_year"]].assign(task=TB)
        for i, s in enumerate(seeds):
            b[f"seed_{s}"] = np.maximum(0, usd[mb, i] - hist)
        b["lightgbm_score"] = np.maximum(0, usd[mb].mean(1) - hist)
        b["num_leaves"], b["train_rows"] = rp.get("num_leaves"), len(train)
        rows.append(b)
        print(f"  {year}: discovery train {len(et):,} (positives {int(et.entry_success.sum())}), "
              f"leaves {params.get('num_leaves')} | expansion train {len(train):,} | {time.time() - t0:.0f}s", flush=True)
    out = pd.concat(rows, ignore_index=True)
    seed_cols = [c for c in out if c.startswith("seed_")]
    spread = out[seed_cols].std(axis=1)
    print(f"mean across-seed SD of scores: discovery {spread[out.task == TA].mean():.2e}, expansion {spread[out.task == TB].mean():.2e}")
    out.to_parquet(PRED_DIR / "lgbm_refit_scores.parquet", index=False)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-models", action="store_true", help="write fitted seed models to models/lightgbm/")
    main(save_models=ap.parse_args().save_models)
