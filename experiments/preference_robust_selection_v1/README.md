# Preference-Robust Export Market Selection

This extension studies robust multi-criteria selection on the audited M1 candidate pools. It treats preference ambiguity, forecast disagreement, and missing outcome labels as separate objects. It is an exploratory application of established robust optimization, not a claim of a new minimax-regret algorithm.

## Run

From the `M1_Export_Market_Selection` directory:

```bash
.venv_uv/bin/python -m experiments.preference_robust_selection_v1.run_preference_robust --mode full
.venv_uv/bin/python -m pytest -q experiments/preference_robust_selection_v1/tests
```

Available stages are `audit`, `scenarios`, `optimize`, `utility-sensitivity`, `evaluate`, `explain`, `inference`, `report`, and `full`. The full mode re-solves the capped-linear utility sensitivity after the primary optimization. Outputs are isolated in this directory. No plot generation is implemented.

Pre-correction intermediate solver outputs are preserved in `archive/pre_solver_gap_correction_20261012/` and must not be used as final results. Active solver diagnostics report both relative and absolute primal–dual gaps; corrected decisions are accepted only when the absolute gap is at most `1e-6` in the relevant objective units (normalized utility for minimax regret/maximin, dimensionless for relative robustness) and feasibility checks pass.

## Evidence status

The existing BACI HS12 V202601 release covers 2012–2024. The 2022–2024 evaluation years have already informed the preceding M1 work, so they are reused as exploratory historical evidence only. The protocol therefore records `NO_FRESH_INDEPENDENT_HOLDOUT`.

## Important interpretation

The finite prediction scenarios are model-disagreement stress cases, not a calibrated probability distribution. Their minimax-regret solution is optimal only for the specified finite scenario set, candidate pool, utility scaling, and solver tolerance. Missing labels are retained as unknown and enter evaluation bounds only.
