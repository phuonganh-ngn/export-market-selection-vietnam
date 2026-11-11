# Final status: COMPLETE_VERIFIED

Engineering results generated: True. Missing required result files: none.

## Evidence status

**NO_FRESH_INDEPENDENT_HOLDOUT / EXPLORATORY_ONLY.** The tested outcomes 2022–2024 were previously inspected and influenced M1 development. BACI HS12 V202601 covers through 2024. This is not independent confirmation.

## Contribution classification

**EMPIRICAL_CONTRIBUTION_ONLY.** No new robust-optimization algorithm or theorem is claimed. PR-EMS is standard finite-scenario minimax regret applied to audited export-market portfolios.

## Completion details

- Selected portfolio rows: 176,721.
- Actual outcome-evaluation policy-pool rows: 23,616.
- Paired inference rows: 24.
- Solver status validation rows: 4.
- Missing labels: preserved as missing; bounds and complete-case estimates are distinct.
- Figures: none.

## Independent post-run validation

- `pytest`: 20 tests passed; Python source compilation passed.
- Frozen candidate keys: 62,797; task rows: 41,131 entry and 12,962 expansion; 559 labels remain missing and all corresponding trade values remain null.
- Selected-set cardinality: zero violations across 23,616 policy-pool records; no selected destination falls outside its task manifest.
- Robust optimization: 8,064 solver runs returned optimal status; maximum absolute objective gap was 9.45e-7 and maximum constraint residual was 1.00e-6.
- Independently reconstructed PR-EMS finite-scenario regret matched its solver objective for all 1,152 task × HS2 × year × K portfolios.
- The paired bootstrap table contains 24 tests and no zero-valued Monte Carlo p-values after the plus-one correction.
- No PNG, SVG, or PDF figure files were generated.

## Reproduction

From the M1 project root: `python experiments/preference_robust_selection_v1/run_preference_robust.py --mode full`.
The full workflow is sequential. It reuses existing processed data and prior model predictions; it does not overwrite prior experiment folders.
