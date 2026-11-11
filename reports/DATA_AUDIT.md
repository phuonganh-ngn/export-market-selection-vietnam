# Data audit

## Source coverage

| Dataset | Provider | Version | Years | Units | Verification |
|---|---|---|---|---|---|
| CEPII BACI HS12 | CEPII | V202601 HS12 | 2012–2024 | thousand current USD (v); metric tons (q) | verified |
| World Bank WGI | World Bank | 2026 | 1996–2025 | standard normal estimates / absolute scores | verified |
| CEPII GeoDist dyadic | CEPII | GeoDist download | time-invariant | distance in km and dyadic controls | verified |
| CEPII GeoDist country | CEPII | GeoDist download | time-invariant | country geography and historical controls | verified |
| HS 2012 chapter headings | World Customs Organization | HS Nomenclature 2012 Edition | 2012 classification | HS2 chapter descriptions | verified official WCO table; 96 active chapters parsed |
| World Bank WDI NY.GDP.MKTP.CD | World Bank | API vintage accessed at run | 2012–2025 | GDP, current US$ | verified API response; CSV persisted |
| World Bank WDI NY.GDP.PCAP.CD | World Bank | API vintage accessed at run | 2012–2025 | GDP per capita, current US$ | verified API response; CSV persisted |
| World Bank WDI NY.GDP.MKTP.KD.ZG | World Bank | API vintage accessed at run | 2012–2025 | GDP growth, annual percent | verified API response; CSV persisted |
| World Bank WDI SP.POP.TOTL | World Bank | API vintage accessed at run | 2012–2025 | Population, total persons | verified API response; CSV persisted |
| World Bank WDI SP.URB.TOTL.IN.ZS | World Bank | API vintage accessed at run | 2012–2025 | Urban population, percent of total | verified API response; CSV persisted |
| World Bank WDI IT.NET.USER.ZS | World Bank | API vintage accessed at run | 2012–2025 | Individuals using the Internet, percent of population | verified API response; CSV persisted |
| World Bank WDI FP.CPI.TOTL.ZG | World Bank | API vintage accessed at run | 2012–2025 | Inflation, consumer prices, annual percent | verified API response; CSV persisted |
| World Bank WDI NE.TRD.GNFS.ZS | World Bank | API vintage accessed at run | 2012–2025 | Trade, percent of GDP | verified API response; CSV persisted |

## Verified sample

- BACI release: CEPII BACI HS12 V202601, observed product-level period 2012–2024; value field `v` is thousand current USD and the pipeline multiplies by 1,000.
- Foreign destinations represented in the eligible primary panel: **224**.
- Eligible HS2 chapters in the primary panel: **96**.
- Destination–HS2–outcome-year observations with verified outcome-year HS2 world imports: **227,719**.
- Outcome coverage in the primary panel: **2014–2024**.
- BACI source trade rows scanned: **142,112,452**.
- Zero Vietnam export share among observed panel rows: **0.5507**.

## Missingness and zero-flow rule

BACI supplies observed positive flows rather than a complete zero-filled matrix. The panel candidate exists only when the destination has positive world imports for the HS2 at the information date. A Vietnam bilateral absence is coded as zero only when the outcome-year destination–HS2 cell has positive BACI world imports; otherwise the outcome is unavailable and remains missing. This can omit destination–product cells with no recorded world imports and is deliberately conservative.
WDI API values can be revised after their reference year. The main backtest enforces reference-year lags but cannot reconstruct the original publication vintage. WGI uses the 2026 revision (historical estimates recalculated); CEPII GeoDist is a time-invariant geographic source.

## Aggregation and coverage checks

Aggregation rows passing value-conservation and HHI checks: 26/26.

|   year |   hs_level |   source_trade_thousand_usd |   aggregated_import_thousand_usd |   absolute_difference_thousand_usd |   markets |   supplier_hhi_out_of_range |   raw_records |   duplicate_records | pass   |
|-------:|-----------:|----------------------------:|---------------------------------:|-----------------------------------:|----------:|----------------------------:|--------------:|--------------------:|:-------|
|   2012 |          2 |                   1.797e+10 |                        1.797e+10 |                          3.815e-06 |     20896 |                           0 |       9012155 |                   0 | True   |
|   2012 |          4 |                   1.797e+10 |                        1.797e+10 |                          3.815e-06 |    225462 |                           0 |       9012155 |                   0 | True   |
|   2013 |          2 |                   1.823e+10 |                        1.823e+10 |                          0         |     21149 |                           0 |       9787220 |                   0 | True   |
|   2013 |          4 |                   1.823e+10 |                        1.823e+10 |                          3.815e-06 |    230383 |                           0 |       9787220 |                   0 | True   |
|   2014 |          2 |                   1.846e+10 |                        1.846e+10 |                          3.815e-06 |     21051 |                           0 |      10205742 |                   0 | True   |
|   2014 |          4 |                   1.846e+10 |                        1.846e+10 |                          3.815e-06 |    232348 |                           0 |      10205742 |                   0 | True   |
|   2015 |          2 |                   1.623e+10 |                        1.623e+10 |                          1.907e-06 |     21141 |                           0 |      10782381 |                   0 | True   |
|   2015 |          4 |                   1.623e+10 |                        1.623e+10 |                          0         |    236056 |                           0 |      10782381 |                   0 | True   |
|   2016 |          2 |                   1.581e+10 |                        1.581e+10 |                          3.815e-06 |     21132 |                           0 |      10868820 |                   0 | True   |
|   2016 |          4 |                   1.581e+10 |                        1.581e+10 |                          0         |    234922 |                           0 |      10868820 |                   0 | True   |
|   2017 |          2 |                   1.752e+10 |                        1.752e+10 |                          3.815e-06 |     21181 |                           0 |      11213140 |                   0 | True   |
|   2017 |          4 |                   1.752e+10 |                        1.752e+10 |                          0         |    236565 |                           0 |      11213140 |                   0 | True   |

## Country coverage

| destination_iso3   | destination_name     |   panel_observations |   hs2_chapters |   years_observed |
|:-------------------|:---------------------|---------------------:|---------------:|-----------------:|
| ABW                | Aruba                |                 1053 |             96 |               11 |
| AFG                | Afghanistan          |                 1045 |             96 |               11 |
| AGO                | Angola               |                 1056 |             96 |               11 |
| AIA                | Anguilla             |                  876 |             92 |               11 |
| ALB                | Albania              |                 1056 |             96 |               11 |
| AND                | Andorra              |                 1050 |             96 |               11 |
| ARE                | United Arab Emirates |                 1056 |             96 |               11 |
| ARG                | Argentina            |                 1056 |             96 |               11 |
| ARM                | Armenia              |                 1056 |             96 |               11 |
| ASM                | American Samoa       |                  860 |             91 |               11 |

## HS2 coverage

|   hs2 |   panel_observations |   destinations |   first_year |   last_year | hs2_description                                                                                                           |
|------:|---------------------:|---------------:|-------------:|------------:|:--------------------------------------------------------------------------------------------------------------------------|
|     1 |                 2238 |            218 |         2014 |        2024 | Live animals.                                                                                                             |
|     2 |                 2435 |            224 |         2014 |        2024 | Meat and edible meat offal.                                                                                               |
|     3 |                 2435 |            223 |         2014 |        2024 | Fish and crustaceans, molluscs and other aquatic invertebrates.                                                           |
|     4 |                 2431 |            223 |         2014 |        2024 | Dairy produce; birds' eggs; natural honey; edible products of animal origin, not elsewhere specified or included.         |
|     5 |                 2252 |            220 |         2014 |        2024 | Products of animal origin, not elsewhere specified or included.                                                           |
|     6 |                 2282 |            219 |         2014 |        2024 | Live trees and other plants; bulbs, roots and the like; cut flowers and ornamental foliage.                               |
|     7 |                 2436 |            224 |         2014 |        2024 | Edible vegetables and certain roots and tubers.                                                                           |
|     8 |                 2438 |            224 |         2014 |        2024 | Edible fruit and nuts; peel of citrus fruit or melons.                                                                    |
|     9 |                 2421 |            223 |         2014 |        2024 | Coffee, tea, maté and spices.                                                                                             |
|    10 |                 2399 |            221 |         2014 |        2024 | Cereals.                                                                                                                  |
|    11 |                 2413 |            222 |         2014 |        2024 | Products of the milling industry; malt; starches; inulin; wheat gluten.                                                   |
|    12 |                 2429 |            223 |         2014 |        2024 | Oil seeds and oleaginous fruits; miscellaneous grains, seeds and fruit; industrial or medicinal plants; straw and fodder. |
|    13 |                 2248 |            213 |         2014 |        2024 | Lac; gums, resins and other vegetable saps and extracts.                                                                  |
|    14 |                 2153 |            215 |         2014 |        2024 | Vegetable plaiting materials; vegetable products not elsewhere specified or included.                                     |
|    15 |                 2433 |            223 |         2014 |        2024 | Animal or vegetable fats and oils and their cleavage products; prepared edible fats; animal or vegetable waxes.           |

## Highest missingness

| variable                         |   missing_count |   missing_rate |
|:---------------------------------|----------------:|---------------:|
| lagged_export_growth             |          146598 |        0.6438  |
| historical_export_volatility     |          120300 |        0.5283  |
| destination_import_demand_growth |           42622 |        0.1872  |
| trade_openness                   |           42173 |        0.1852  |
| inflation                        |           36238 |        0.1591  |
| internet_usage                   |           31237 |        0.1372  |
| government_effectiveness         |           19368 |        0.08505 |
| regulatory_quality               |           19368 |        0.08505 |
| gdp_growth                       |           18579 |        0.08159 |
| log_gdp                          |           18003 |        0.07906 |
| log_gdp_per_capita               |           18003 |        0.07906 |
| political_stability              |           17746 |        0.07793 |
| rule_of_law                      |           17583 |        0.07721 |
| population_weighted_distance_km  |           17458 |        0.07666 |
| landlocked                       |           14068 |        0.06178 |

## Exclusions

| exclusion_rule                                         |   rows_before_filter | unit                 |
|:-------------------------------------------------------|---------------------:|:---------------------|
| self_destination_Vietnam                               |                 1248 | destination-HS2-year |
| destination_without_alpha3_mapping                     |                 1248 | destination-HS2-year |
| positive_world_imports_required_for_market_observation |                    0 | destination-HS2-year |

## Core result artifacts

- `results/E1_forecasting_overall.csv`: present
- `results/E1_forecasting_by_hs2.csv`: present
- `results/E2_market_discovery.csv`: present
- `results/E3_market_ranking.csv`: present
- `results/E4_global_shap.csv`: present
- `results/E5_clusters.csv`: present

Potential new-market case counts are reported by the entry-threshold table after the discovery stage. The development choice is based on pre-test outcomes only.
