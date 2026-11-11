# Temporal validation

**Primary specification: two-year lag, retrospective rolling-origin.** For outcome year y, test covariates use the panel's y−2 information fields and training outcomes stop at y−2. Inner chronological validation uses y−2 as the validation outcome year and inner training outcomes through y−4. Encoders, missing-value handling, scaling, and category vocabularies are fitted from training rows only. Hyperparameters, epochs, class weighting, and the regression calibration factor are selected only on the inner validation fold.

## Frozen outer populations

|   outcome_year |   ex_ante_candidates |   observed_labels |   missing_labels |   entry_candidates |   observed_entry_labels |   entry_events |
|---------------:|---------------------:|------------------:|-----------------:|-------------------:|------------------------:|---------------:|
|           2022 |                20921 |             20734 |              187 |              13791 |                   13604 |            576 |
|           2023 |                20899 |             20749 |              150 |              13687 |                   13537 |            520 |
|           2024 |                20977 |             20755 |              222 |              13653 |                   13431 |            362 |

## One-year population audit

|   outcome_year |   lag2_candidates |   lag1_candidates |   intersection |   lag2_only |   lag1_only |   lag2_observed |   lag1_observed |   lag2_max_feature_year |   lag1_max_feature_year | status                                                         |
|---------------:|------------------:|------------------:|---------------:|------------:|------------:|----------------:|----------------:|------------------------:|------------------------:|:---------------------------------------------------------------|
|           2022 |             20921 |             20899 |          20714 |         207 |         185 |           20734 |           20741 |                    2020 |                    2021 | candidate keys differ; model metrics require separate universe |
|           2023 |             20899 |             20977 |          20741 |         158 |         236 |           20749 |           20814 |                    2021 |                    2022 | candidate keys differ; model metrics require separate universe |
|           2024 |             20977 |             21015 |          20814 |         163 |         201 |           20755 |           20783 |                    2022 |                    2023 | candidate keys differ; model metrics require separate universe |

One-year-lag performance was not run: the ex-ante populations differ, so two-year predictions cannot serve as a matched one-year sensitivity. No one-year performance claim is made.

WDI/WGI values are revised historical vintages and BACI reference-year observations are not proven to have been available at prediction time. This is not a deployment-grade point-in-time backtest. Only three outer years are available.
