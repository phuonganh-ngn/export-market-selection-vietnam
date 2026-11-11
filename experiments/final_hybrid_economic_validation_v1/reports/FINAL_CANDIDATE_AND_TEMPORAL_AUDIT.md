# Final candidate and temporal audit

## Reconstructed population

The source task builder was rerun read-only from `data/processed/panel_lag2.parquet` and BACI HS2 interim trade records. Its candidate keys were compared with the archived ERGM manifest and prediction archive. The independently rebuilt manifest contains 42,699 task-candidate rows: 29,737 Task A and 12,962 Task B. Unknown labels remain NA; Task A has 556 and Task B has 0.

| task                           |   outcome_year |   ex_ante_candidates |   observed_labels |   unknown_labels |   unique_destinations |   hs2_chapters |   candidate_pools |   cutoff_violation_count |
|:-------------------------------|---------------:|---------------------:|------------------:|-----------------:|----------------------:|---------------:|------------------:|-------------------------:|
| Task_A_new_entry               |           2022 |                 9846 |              9659 |              187 |                   221 |             96 |                96 |                        0 |
| Task_A_new_entry               |           2023 |                 9913 |              9764 |              149 |                   220 |             96 |                96 |                        0 |
| Task_A_new_entry               |           2024 |                 9978 |              9758 |              220 |                   220 |             96 |                96 |                        0 |
| Task_B_existing_underexploited |           2022 |                 4320 |              4320 |                0 |                   165 |             96 |                96 |                        0 |
| Task_B_existing_underexploited |           2023 |                 4387 |              4387 |                0 |                   163 |             96 |                96 |                        0 |
| Task_B_existing_underexploited |           2024 |                 4255 |              4255 |                0 |                   160 |             96 |                96 |                        0 |

## Timing

For outcome year y, the panel predictor year is y−2. Task A verifies annual Vietnam export reporting for the three years feature_year−2, feature_year−1, and feature_year. Thus the strict-zero window ends at y−2; it does not inspect y−1. The label defines a new-to-Vietnam HS2-destination relationship after three verified zero years, not a first-ever trade relationship over all history. A missing year of destination reporting is not treated as zero.

Task B requires positive historical exports and a market share no greater than an HS2 median estimated from observed historical outcomes through the feature year. Eligibility is generated before the target year. The two reconstructed candidate sets have no shared destination–HS2–year keys. Candidate ID and task keys are unique.

The historical OOF archive was also matched against the task definitions for 2018–2021. Each OOF model fold used outcomes through no later than y−2; feature-year checks found no cutoff violation. The outer score archive has complete PPML and LightGBM scores for all candidate keys.

## Scope limitations

Unknown labels are preserved as missing and complete-case ranking metrics are conditional on observed outcomes. Economic sums over a full predicted Top-K list with missing labels are lower-bound observed amounts, not full realized portfolio values. Macro and governance histories are revised-vintage data, so these tests are retrospective rather than deployment-grade point-in-time backtests.

## Integrity checks

- Reconstructed candidate-key set matches the archived ERGM manifest exactly.
- Reconstructed observed-label indicators match the archive.
- Task A/Task B overlap: 0.
- Strict-zero years end at the permitted y−2 cutoff.
- No missing outcomes were recoded as zero.
- All outer expert predictions are finite and present.

## Fresh holdout check

No fresh independent holdout was established. The official CEPII BACI V202601 page lists HS12 coverage through 2024. The UN Comtrade public annual HS availability endpoint was queried for Vietnam (reporter M49 704) for 2024 and 2025 on 2026-10-13; both responses returned zero available records at the query time. A small public preview also returned 2022–2023 records but no 2024 records. We did not substitute another source or mix provisional Comtrade reporting with reconciled BACI outcomes. Consequently, 2022–2024 remain previously inspected exploratory evidence; status: `NO_FRESH_INDEPENDENT_HOLDOUT`.

CEPII lists V202601 as updated 30 January 2026, with HS12 data ending in 2024 ([BACI source page](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37); [January 2026 release notes](https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/release_notes_202601.pdf)). The data-availability check is source-specific and time-stamped; availability can change later. UN Comtrade documents its availability API and notes annual data are updated as reporting countries submit revisions ([data availability](https://uncomtrade.org/docs/data-availability/); [API guide](https://uncomtrade.org/docs/un-comtrade-api/)). The public endpoint tested here did not return Vietnam 2024/2025 observations.
