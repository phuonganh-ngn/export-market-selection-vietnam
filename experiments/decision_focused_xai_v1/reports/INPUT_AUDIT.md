# Input and protocol audit

**Population: VERIFIED against audit_v2 and the post-audit NN extension.**

Processed input: `data/processed/panel_lag2.parquet` (272,113 rows, 56 columns).
Baseline predictions are reused from `audit_v2/corrected_E2_candidates.parquet` via the checked neural-extension artifacts, and from `post_audit_nn_dl_v1/predictions/E3_neural_rankings.parquet` for expansion scores.

## Frozen evaluation candidate universe

|   outcome_year |   ex_ante_candidates |   observed_labels |   missing_labels |   entry_candidates |
|---------------:|---------------------:|------------------:|-----------------:|-------------------:|
|           2022 |                20921 |             20734 |              187 |              13791 |
|           2023 |                20899 |             20749 |              150 |              13687 |
|           2024 |                20977 |             20755 |              222 |              13653 |

Test task manifest has 54,093 candidate-task rows: 41,131 new-entry candidates and 12,962 existing-underexploited candidates.
The 559 outcome labels remain missing. Complete-case ranking metrics are conditional on observed labels; ranks and missing-label positions are saved for the full ex-ante candidate pool.

## Time cutoffs

Outer years are 2022, 2023, and 2024. At each origin, the maximum feature reference year and training outcome year are y−2. Inner validation uses the latest historically admissible outcome year y−2, with its fitting data ending at y−4. This preserves the audited rolling-origin protocol; the same validation year may be an earlier outer test year in a later walk-forward fold, so cross-fold estimates are dependent.
Historical WDI/WGI values use revised vintages; this is a retrospective evaluation, not archival real-time validation.

## Implementation

Installed LightGBM version: 4.7.0. `LGBMRanker` is fitted with `objective='lambdarank'`, contiguous `group` sizes, and explicit `label_gain`. The official API documents that group sizes must sum to the number of rows and that `pred_contrib` returns an extra expected-value column ([LightGBM API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html)).
RankingSHAP is an established listwise explanation proposal, not a method claimed here; its reference is Heuss, de Rijke and Anand (SIGIR 2025), https://doi.org/10.1145/3726302.3729971. This extension uses native additive ranking-score contributions and explicit rank/Top-K perturbation diagnostics, and does not claim to implement RankingSHAP.
