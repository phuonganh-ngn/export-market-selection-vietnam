# Temporal validation protocol

Outer years: 2022, 2023, 2024. Predictor and observed training outcome cutoff is y−2. Inner validation outcome year is y−2 and inner training outcomes end at y−4. All comparisons are product-year pool matched.
The rolling-origin folds are dependent: the y−2 validation labels for a later origin may have been an earlier outer test year. This follows the audited sequential information set and is disclosed; it is not independent repeated holdout evidence.

| task                           |   evaluation_year |   feature_cutoff_y_minus_2 |   training_outcome_max_y_minus_2 |   candidates |   observed_labels |   unknown_labels |   candidate_pools | leakage_assertion   |
|:-------------------------------|------------------:|---------------------------:|---------------------------------:|-------------:|------------------:|-----------------:|------------------:|:--------------------|
| Task_A_new_entry               |              2022 |                       2020 |                             2020 |        13791 |             13604 |              187 |                96 | passed              |
| Task_B_existing_underexploited |              2022 |                       2020 |                             2020 |         4320 |              4320 |                0 |                96 | passed              |
| Task_A_new_entry               |              2023 |                       2021 |                             2021 |        13687 |             13537 |              150 |                96 | passed              |
| Task_B_existing_underexploited |              2023 |                       2021 |                             2021 |         4387 |              4387 |                0 |                96 | passed              |
| Task_A_new_entry               |              2024 |                       2022 |                             2022 |        13653 |             13431 |              222 |                96 | passed              |
| Task_B_existing_underexploited |              2024 |                       2022 |                             2022 |         4255 |              4255 |                0 |                96 | passed              |

WDI/WGI historical vintages are revised, so no archival point-in-time claim is made.
