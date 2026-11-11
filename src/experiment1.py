from __future__ import annotations

import json
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import CFG, path, write_csv
from gravity_ppml import fit_ppml
from ml_models import MODEL_NAMES, fit_regressor, tune_regressor
from statistics import forecast_metrics
from temporal_validation import split_outer, write_split_manifest


def _score(preds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in preds.groupby(by, dropna=False, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys))
        row.update(forecast_metrics(group.actual_usd, group.predicted_usd))
        rows.append(row)
    return pd.DataFrame(rows)


def run(lag: int | None = None, quick: bool = False) -> pd.DataFrame:
    lag = int(lag or CFG["study"]["primary_lag"])
    panel = pd.read_parquet(path("processed_dir") / f"panel_lag{lag}.parquet")
    write_split_manifest(panel, lag)
    all_preds, tune_rows, gravity_rows, fail_rows = [], [], [], []
    seeds = [CFG["study"]["seeds"][0]] if quick or CFG["compute"].get("quick_mode") else CFG["study"]["seeds"]
    years = CFG["study"]["evaluation_years"]
    for eval_year in years:
        train, test = split_outer(panel, int(eval_year), lag)
        test = test.reset_index(drop=True)
        actual = test.export_value_usd.to_numpy(float)
        # Historical persistence is the last verified Vietnam flow at the information date.
        persist = test.historical_export_value.clip(lower=0).to_numpy(float)
        for i, value in enumerate(persist):
            all_preds.append({"model": "Persistence", "seed": 0, "evaluation_year": int(eval_year),
                "destination_iso3": test.destination_iso3.iloc[i], "hs2": test.hs2.iloc[i], "actual_usd": actual[i],
                "predicted_usd": value, "historical_export_value_usd": test.historical_export_value.iloc[i]})
        try:
            ppml, result, diagnostic, _ = fit_ppml(train, test)
            gravity_rows.append({"evaluation_year": int(eval_year), **diagnostic, "status": "converged" if diagnostic["converged"] else "did not converge"})
            for i, value in enumerate(ppml):
                all_preds.append({"model": "Gravity_PPML", "seed": 0, "evaluation_year": int(eval_year),
                    "destination_iso3": test.destination_iso3.iloc[i], "hs2": test.hs2.iloc[i], "actual_usd": actual[i],
                    "predicted_usd": value, "historical_export_value_usd": test.historical_export_value.iloc[i]})
            joblib.dump({"result": result, "diagnostics": diagnostic}, path("results_dir") / "models" / f"ppml_lag{lag}_{eval_year}.joblib")
        except Exception as exc:
            gravity_rows.append({"evaluation_year": int(eval_year), "converged": False, "status": f"failed: {type(exc).__name__}: {exc}"})
            fail_rows.append({"experiment": "E1", "evaluation_year": eval_year, "model": "Gravity_PPML", "error": traceback.format_exc()})
        for kind in MODEL_NAMES:
            try:
                params, val_mae = tune_regressor(kind, train, lag)
                tune_rows.append({"evaluation_year": eval_year, "model": kind, "validation_mae_usd": val_mae,
                                  "params": json.dumps(params, sort_keys=True), "validation": "chronological inner holdout"})
            except Exception as exc:
                fail_rows.append({"experiment": "E1 tune", "evaluation_year": eval_year, "model": kind, "error": traceback.format_exc()})
                continue
            for seed in seeds:
                try:
                    fitted = fit_regressor(kind, train, params, int(seed), lag)
                    prediction = fitted.predict(test)
                    bundle_path = path("results_dir") / "models" / f"{kind.lower()}_lag{lag}_{eval_year}_seed{seed}.joblib"
                    joblib.dump({"bundle": fitted, "params": params, "lag": lag, "evaluation_year": eval_year}, bundle_path)
                    for i, value in enumerate(prediction):
                        all_preds.append({"model": kind, "seed": seed, "evaluation_year": int(eval_year),
                            "destination_iso3": test.destination_iso3.iloc[i], "hs2": test.hs2.iloc[i], "actual_usd": actual[i],
                            "predicted_usd": value, "historical_export_value_usd": test.historical_export_value.iloc[i],
                            "validation_mae_scale": fitted.usd_scale_factor})
                    print(f"E1 {eval_year} {kind} seed={seed} complete", flush=True)
                except Exception as exc:
                    fail_rows.append({"experiment": "E1 fit", "evaluation_year": eval_year, "model": kind,
                                      "seed": seed, "error": traceback.format_exc()})
                    print(f"E1 failed: {eval_year} {kind} seed={seed}: {exc}", flush=True)
    pred = pd.DataFrame(all_preds)
    if pred.empty:
        raise RuntimeError("Experiment 1 produced no predictions")
    output = path("results_dir") / "predictions" / f"E1_predictions_lag{lag}.parquet"
    pred.to_parquet(output, index=False)
    if lag == int(CFG["study"]["primary_lag"]):
        pred.to_parquet(path("results_dir") / "E1_predictions.parquet", index=False)
    average = pred.groupby(["model", "evaluation_year", "destination_iso3", "hs2", "actual_usd"], as_index=False).predicted_usd.mean()
    by_year = _score(average, ["model", "evaluation_year"])
    by_hs2 = _score(average, ["model", "evaluation_year", "hs2"])
    suffix = "" if lag == int(CFG["study"]["primary_lag"]) else f"_lag{lag}"
    by_hs2.to_csv(path("results_dir") / f"E1_forecasting_by_hs2{suffix}.csv", index=False)
    by_year.to_csv(path("results_dir") / f"E1_forecasting_by_year{suffix}.csv", index=False)
    over = []
    for model, d in average.groupby("model", sort=False):
        m = forecast_metrics(d.actual_usd, d.predicted_usd)
        macro = by_hs2.loc[by_hs2.model.eq(model)]
        seed_metrics = []
        raw = pred.loc[pred.model.eq(model)]
        for seed, sd in raw.groupby("seed"):
            seed_metrics.append((int(seed), forecast_metrics(sd.actual_usd, sd.predicted_usd)["mae_usd"]))
        m.update({"model": model, "macro_hs2_mae_usd": float(macro.groupby("hs2").mae_usd.mean().mean()),
                  "macro_hs2_mae_log1p": float(macro.groupby("hs2").mae_log1p.mean().mean()),
                  "seed_mae_mean_usd": float(np.nanmean([x[1] for x in seed_metrics])),
                  "seed_mae_sd_usd": float(np.nanstd([x[1] for x in seed_metrics], ddof=1)) if len(seed_metrics)>1 else np.nan,
                  "lag": lag, "evaluation_years": ",".join(map(str, years))})
        over.append(m)
    overall = pd.DataFrame(over).sort_values("mae_usd")
    overall.to_csv(path("results_dir") / f"E1_forecasting_overall{suffix}.csv", index=False)
    pd.DataFrame(tune_rows).to_csv(path("results_dir") / "tables" / f"E1_tuning_lag{lag}.csv", index=False)
    pd.DataFrame(gravity_rows).to_csv(path("results_dir") / "tables" / f"E1_ppml_diagnostics_lag{lag}.csv", index=False)
    if fail_rows:
        pd.DataFrame(fail_rows).to_csv(path("results_dir") / "tables" / f"E1_failures_lag{lag}.csv", index=False)
    return overall


if __name__ == "__main__":
    run()
