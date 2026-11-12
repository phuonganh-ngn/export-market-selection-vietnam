"""Sealed prospective shortlists for outcome years 2025 and 2026 (outcomes not yet observed).

Same pipeline as the main analysis: models trained on outcomes up to y-2 (2023 for 2025, 2024 for 2026) with
predictors from y-2; bagged five-seed LightGBM (tuned per year), Gravity-PPML, equal-weight rank fusion with the
configured tie rule. Recommended lists: hybrid for discovery, PPML for expansion; all model ranks are kept so that
every comparison of the paper can be repeated once BACI releases the outcomes.

2024 is re-run as a control: its LightGBM scores must equal the scores used in the paper.
Outputs in sealed/: candidates_<year>.parquet (all candidates, scores and ranks), shortlist_<year>.csv (union of
Top-10 lists), and MANIFEST.json with SHA-256 hashes of every output, the code, the config and the input panel.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from revision_lib import CFG, HERE, KEY, PANEL, PRED_DIR, PROJECT, TA, TB, zfill_hs2
import fusion
from experiments.explainable_reliability_gate_v1.src.task_audit import load_and_freeze
from gravity_ppml import fit_ppml
from market_discovery import entry_training_mask
from ml_models import fit_classifier, fit_regressor, tune_classifier, tune_regressor

OUT = HERE / "sealed"; OUT.mkdir(parents=True, exist_ok=True)
YEARS = [2024, 2025, 2026]
POOL = ["task", "hs2", "outcome_year"]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rank_within(frame, col):
    return frame.groupby(POOL)[col].rank(ascending=False, method="first").astype(int)


def score_year(panel, year, seeds):
    train = panel.loc[panel.outcome_observed.astype(bool) & panel.export_value_usd.notna() & panel.destination_import_demand.gt(0)
                      & panel.outcome_year.le(year - 2)].copy()
    test = panel.loc[panel.outcome_year.eq(year) & panel.destination_import_demand.gt(0)].copy()
    assert int(train.outcome_year.max()) <= year - 2, "leakage"
    ppml_usd, _, diag, _ = fit_ppml(train, test)
    test["ppml_usd"] = ppml_usd
    # discovery
    et = train.loc[entry_training_mask(train) & train.entry_success.notna()].copy()
    cp, _ = tune_classifier("LightGBM", et, "entry_success", lag=2, trials=CFG["study"]["tuning_trials"])
    ex = test.loc[test.entry_candidate_ex_ante.astype(bool) & test.strict_inactive_verified.astype(bool)].copy()
    probs = [m.estimator.predict_proba(m.preprocessor.transform(ex[m.numeric_features + m.categorical_features]).astype(np.float32))[:, 1]
             for m in (fit_classifier("LightGBM", et, "entry_success", cp, s) for s in seeds)]
    a = ex[["destination_iso3", "hs2", "outcome_year", "log_destination_import_demand"]].assign(
        task=TA, ppml_score=ex.ppml_usd, lightgbm_score=np.mean(probs, axis=0), historical_export_value=ex.historical_export_value)
    # expansion
    rp, _ = tune_regressor("LightGBM", train, lag=2, trials=CFG["study"]["tuning_trials"])
    usd = np.mean([fit_regressor("LightGBM", train, rp, s, lag=2).predict(test, log_target=True) for s in seeds], axis=0)
    mb = test.task_b_eligible.astype(bool).to_numpy()
    hist = test.historical_export_value.to_numpy(float)
    b = test.loc[mb, ["destination_iso3", "hs2", "outcome_year", "log_destination_import_demand"]].assign(
        task=TB, ppml_score=np.maximum(0, test.ppml_usd.to_numpy()[mb] - hist[mb]), lightgbm_score=np.maximum(0, usd[mb] - hist[mb]),
        historical_export_value=hist[mb])
    out = pd.concat([a, b], ignore_index=True)
    out = fusion.fuse(out, POOL, ["ppml_score", "lightgbm_score"], tie_rule=CFG["study"].get("fusion_tie_rule", "iso3"))
    out["rank_hybrid"] = out.fused_rank.astype(int)
    out["rank_ppml"], out["rank_lightgbm"] = rank_within(out, "ppml_score"), rank_within(out, "lightgbm_score")
    out["rank_import_demand"] = rank_within(out.assign(_d=out.log_destination_import_demand.fillna(-1e9)), "_d")
    out["recommended_rank"] = np.where(out.task == TA, out.rank_hybrid, out.rank_ppml)
    out["train_outcome_max"], out["ppml_converged"] = int(train.outcome_year.max()), diag["converged"]
    out["leaves_discovery"], out["leaves_expansion"] = cp.get("num_leaves"), rp.get("num_leaves")
    return out.drop(columns=[c for c in out if c.startswith("_rank_")])


def main():
    t0 = time.time()
    seeds = [int(s) for s in CFG["study"]["seeds"]]
    panel, _ = load_and_freeze(write=False); panel = zfill_hs2(panel)
    files, summary = [], {}
    for year in YEARS:
        s = score_year(panel, year, seeds)
        if year == 2024:   # control: must reproduce the paper's scores
            ref = zfill_hs2(pd.read_parquet(PRED_DIR / "lgbm_refit_scores.parquet"))[KEY + ["lightgbm_score"]]
            m = s.merge(ref, on=KEY, suffixes=("", "_paper"), validate="one_to_one")
            diff = float(np.abs(m.lightgbm_score - m.lightgbm_score_paper).max())
            summary["control_2024"] = {"rows_matched": len(m), "rows": len(s), "max_abs_diff_lightgbm": diff}
            print(f"2024 control: {len(m)}/{len(s)} rows matched, max |diff| {diff:.2e}", flush=True)
            continue
        p = OUT / f"candidates_{year}.parquet"; s.to_parquet(p, index=False); files.append(p)
        top = s[(s.rank_hybrid <= 10) | (s.rank_ppml <= 10) | (s.rank_lightgbm <= 10)].sort_values(["task", "hs2", "recommended_rank"])
        c = OUT / f"shortlist_{year}.csv"; top.to_csv(c, index=False, float_format="%.10g"); files.append(c)
        summary[str(year)] = {"discovery_candidates": int((s.task == TA).sum()), "expansion_candidates": int((s.task == TB).sum()),
                              "pools": int(s.groupby(POOL).ngroups), "train_outcome_max": int(s.train_outcome_max.iloc[0]),
                              "leaves": [int(s.leaves_discovery.iloc[0]), int(s.leaves_expansion.iloc[0])]}
        print(f"{year}: {summary[str(year)]} | {time.time() - t0:.0f}s", flush=True)
    code = [HERE / "src" / n for n in ("sealed_shortlists.py", "revision_lib.py")] + [PROJECT / "src" / n for n in ("ml_models.py", "gravity_ppml.py", "fusion.py", "market_discovery.py")]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True).stdout.strip()
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "code_commit_before_sealing": commit,
                "purpose": "Prospective shortlists for outcome years 2025 and 2026, fixed before BACI reports these years.",
                "recommended": {"discovery": "hybrid (rank_hybrid)", "expansion": "Gravity-PPML (rank_ppml)"}, "seeds": seeds,
                "summary": summary,
                "outputs": {f.name: sha256(f) for f in files},
                "inputs": {"panel_lag2.parquet": sha256(PANEL), "config.yaml": sha256(PROJECT / "config" / "config.yaml")},
                "code": {str(f.relative_to(PROJECT)): sha256(f) for f in code}}
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
