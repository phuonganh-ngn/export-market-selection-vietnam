# Paper-ready methods and results (exploratory version)

## Data and evaluation

The extension uses the existing BACI HS12 V202601 HS2 panel and saved out-of-time baseline predictions. The primary evaluation uses the audited two-year information lag and decision pools formed by product chapter and outcome year. It covers 2022–2024, an interval already examined in prior M1 development. Revised WDI/WGI values make the backtest retrospective rather than archival point-in-time validation.

Entry pools require verified zero Vietnam exports in each of the three origin years; the success threshold is USD 50,000. Existing expansion pools require positive historical exports and a share at or below the HS2 median available at the origin. This strict entry definition resolves the previous 6,511-key overlap created when ‘inactive’ meant only sub-threshold trade.

## Routing and reliability

Gravity–PPML and LightGBM scores are reused for the outer years. Historical gate targets are constructed from new rolling-origin OOF predictions for 2018–2021, with model outcomes ending no later than y−2. A separate regularized linear model predicts each expert's observed-label Top-10 regret for every task. The lower predicted-regret expert is selected. A second threshold compares that pool-level regret score with a development quantile; rejected recommendations are marked for review and are not assigned a hindsight-oracle replacement.

## Results

| task                           | static_expert   |   static_ndcg10 |   static_regret10 |   adaptive_ndcg10 |   adaptive_regret10 |   adaptive_lgb_share |   blend_ndcg10 |   ergm_matched80_regret |   ergm_matched80_ndcg |   ergm_operational80_coverage |   ergm_operational80_regret |
|:-------------------------------|:----------------|----------------:|------------------:|------------------:|--------------------:|---------------------:|---------------:|------------------------:|----------------------:|------------------------------:|----------------------------:|
| Task_A_new_entry               | LightGBM        |        0.294694 |          0.458333 |          0.281714 |            0.458333 |             0.399306 |       0.331878 |                0.246753 |              0.300409 |                      0.951389 |                    0.404118 |
| Task_B_existing_underexploited | Gravity_PPML    |        0.592524 |          2.15278  |          0.579031 |            2.25     |             0.295139 |       0.571279 |                1.09957  |              0.592273 |                      0.722222 |                    0.866752 |

At matched coverage, selective-risk results are conditional on pools with observed outcomes; deferred pools remain undecided. The 559 unknown outcomes in the broad evaluation universe remain missing, including 556 corrected strict-entry candidates. No test-year outcome is used to construct candidate eligibility or fit the outer gate.

## Interpretation and limitations

The empirical system is a two-stage application of existing expert-routing and selective-prediction ideas. It does not establish human-AI collaboration, causal export gains, a calibrated decision-failure probability, or independent temporal validity. Evidence is exploratory because the test years were previously inspected, the number of product-year units is limited, and revised macroeconomic vintages are not point-in-time. The extension's principal contribution is the corrected evaluation of how a pool-level regret gate behaves against static and confidence/disagreement baselines in export-market ranking.
