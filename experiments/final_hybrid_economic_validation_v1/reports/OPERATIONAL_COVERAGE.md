# Operational coverage

Coverage thresholds are quantiles of cross-fitted development risk scores available no later than the y−2 information cutoff. The test-year threshold is fixed; realized coverage is not forced to equal the nominal target. Nominal 90/80/70/60% thresholds and a no-deferral 100% reference are reported with year-specific accepted/deferred counts and accepted risk. Random deferral is a workload benchmark; matched coverage is a secondary retrospective diagnostic.

| task                           | policy               |   nominal_coverage |   actual_coverage |   accepted_regret10 |   accepted_ndcg10 |   deferred_pools |
|:-------------------------------|:---------------------|-------------------:|------------------:|--------------------:|------------------:|-----------------:|
| Task_A_new_entry               | BLEND_LEARNED_RISK   |                0.6 |          0.909722 |            0.308883 |          0.347328 |               26 |
| Task_A_new_entry               | BLEND_LEARNED_RISK   |                0.7 |          0.940972 |            0.339155 |          0.335458 |               17 |
| Task_A_new_entry               | BLEND_LEARNED_RISK   |                0.8 |          0.954861 |            0.35233  |          0.334725 |               13 |
| Task_A_new_entry               | BLEND_LEARNED_RISK   |                0.9 |          0.996528 |            0.41473  |          0.324865 |                1 |
| Task_A_new_entry               | HISTORICAL_HS2_ERROR |                0.6 |          0.763889 |            0.282032 |          0.297136 |               68 |
| Task_A_new_entry               | HISTORICAL_HS2_ERROR |                0.7 |          0.763889 |            0.282032 |          0.297136 |               68 |
| Task_A_new_entry               | HISTORICAL_HS2_ERROR |                0.8 |          0.909722 |            0.408298 |          0.294536 |               26 |
| Task_A_new_entry               | HISTORICAL_HS2_ERROR |                0.9 |          0.979167 |            0.457447 |          0.278065 |                6 |
| Task_A_new_entry               | MODEL_DISAGREEMENT   |                0.6 |          0.815972 |            0.502339 |          0.282052 |               53 |
| Task_A_new_entry               | MODEL_DISAGREEMENT   |                0.7 |          0.861111 |            0.48082  |          0.293695 |               40 |
| Task_A_new_entry               | MODEL_DISAGREEMENT   |                0.8 |          0.909722 |            0.466695 |          0.298795 |               26 |
| Task_A_new_entry               | MODEL_DISAGREEMENT   |                0.9 |          0.965278 |            0.460299 |          0.289994 |               10 |
| Task_A_new_entry               | NO_DEFERRAL          |                0.6 |          1        |            0.458333 |          0.287623 |                0 |
| Task_A_new_entry               | NO_DEFERRAL          |                0.7 |          1        |            0.458333 |          0.287623 |                0 |
| Task_A_new_entry               | NO_DEFERRAL          |                0.8 |          1        |            0.458333 |          0.287623 |                0 |
| Task_A_new_entry               | NO_DEFERRAL          |                0.9 |          1        |            0.458333 |          0.287623 |                0 |
| Task_A_new_entry               | RANDOM_DEFERRAL      |                0.6 |          0.604167 |            0.459034 |          0.287054 |              114 |
| Task_A_new_entry               | RANDOM_DEFERRAL      |                0.7 |          0.708333 |            0.457186 |          0.287    |               84 |
| Task_A_new_entry               | RANDOM_DEFERRAL      |                0.8 |          0.802083 |            0.459628 |          0.287334 |               57 |
| Task_A_new_entry               | RANDOM_DEFERRAL      |                0.9 |          0.90625  |            0.458065 |          0.287333 |               27 |
| Task_A_new_entry               | SCORE_MARGIN         |                0.6 |          0.576389 |            0.373892 |          0.290487 |              122 |
| Task_A_new_entry               | SCORE_MARGIN         |                0.7 |          0.684028 |            0.365916 |          0.285265 |               91 |
| Task_A_new_entry               | SCORE_MARGIN         |                0.8 |          0.78125  |            0.366743 |          0.284758 |               63 |
| Task_A_new_entry               | SCORE_MARGIN         |                0.9 |          0.881944 |            0.385567 |          0.290173 |               34 |
| Task_A_new_entry               | STATIC_LEARNED_RISK  |                0.6 |          0.913194 |            0.342859 |          0.297659 |               25 |
| Task_A_new_entry               | STATIC_LEARNED_RISK  |                0.7 |          0.947917 |            0.399417 |          0.291076 |               15 |
| Task_A_new_entry               | STATIC_LEARNED_RISK  |                0.8 |          0.975694 |            0.43769  |          0.285642 |                7 |
| Task_A_new_entry               | STATIC_LEARNED_RISK  |                0.9 |          0.993056 |            0.450987 |          0.287525 |                2 |
| Task_B_existing_underexploited | HISTORICAL_HS2_ERROR |                0.6 |          0.614583 |            0.62305  |          0.602144 |              111 |
| Task_B_existing_underexploited | HISTORICAL_HS2_ERROR |                0.7 |          0.746528 |            0.85774  |          0.615191 |               73 |
| Task_B_existing_underexploited | HISTORICAL_HS2_ERROR |                0.8 |          0.861111 |            1.2107   |          0.614586 |               40 |
| Task_B_existing_underexploited | HISTORICAL_HS2_ERROR |                0.9 |          0.892361 |            1.37426  |          0.605539 |               31 |
| Task_B_existing_underexploited | MODEL_DISAGREEMENT   |                0.6 |          0.645833 |            2.37342  |          0.596743 |              102 |
| Task_B_existing_underexploited | MODEL_DISAGREEMENT   |                0.7 |          0.722222 |            2.29721  |          0.5887   |               80 |
| Task_B_existing_underexploited | MODEL_DISAGREEMENT   |                0.8 |          0.847222 |            2.27901  |          0.583144 |               44 |
| Task_B_existing_underexploited | MODEL_DISAGREEMENT   |                0.9 |          0.934028 |            2.24981  |          0.589403 |               19 |
| Task_B_existing_underexploited | NO_DEFERRAL          |                0.6 |          1        |            2.15278  |          0.589391 |                0 |
| Task_B_existing_underexploited | NO_DEFERRAL          |                0.7 |          1        |            2.15278  |          0.589391 |                0 |
| Task_B_existing_underexploited | NO_DEFERRAL          |                0.8 |          1        |            2.15278  |          0.589391 |                0 |
| Task_B_existing_underexploited | NO_DEFERRAL          |                0.9 |          1        |            2.15278  |          0.589391 |                0 |
| Task_B_existing_underexploited | RANDOM_DEFERRAL      |                0.6 |          0.604167 |            2.15605  |          0.589439 |              114 |
| Task_B_existing_underexploited | RANDOM_DEFERRAL      |                0.7 |          0.708333 |            2.15281  |          0.589179 |               84 |
| Task_B_existing_underexploited | RANDOM_DEFERRAL      |                0.8 |          0.802083 |            2.1539   |          0.589873 |               57 |
| Task_B_existing_underexploited | RANDOM_DEFERRAL      |                0.9 |          0.90625  |            2.15168  |          0.589339 |               27 |
| Task_B_existing_underexploited | SCORE_MARGIN         |                0.6 |          0.555556 |            2.67969  |          0.56632  |              128 |
| Task_B_existing_underexploited | SCORE_MARGIN         |                0.7 |          0.6875   |            2.48269  |          0.567868 |               90 |
| Task_B_existing_underexploited | SCORE_MARGIN         |                0.8 |          0.802083 |            2.37507  |          0.573547 |               57 |
| Task_B_existing_underexploited | SCORE_MARGIN         |                0.9 |          0.909722 |            2.28685  |          0.584093 |               26 |
| Task_B_existing_underexploited | STATIC_LEARNED_RISK  |                0.6 |          0.517361 |            0.579446 |          0.593006 |              139 |
| Task_B_existing_underexploited | STATIC_LEARNED_RISK  |                0.7 |          0.618056 |            0.670547 |          0.606979 |              110 |
| Task_B_existing_underexploited | STATIC_LEARNED_RISK  |                0.8 |          0.743056 |            0.824382 |          0.612391 |               74 |
| Task_B_existing_underexploited | STATIC_LEARNED_RISK  |                0.9 |          0.809028 |            1.02615  |          0.606996 |               55 |

## Retrospective matched-coverage diagnostic

| task                           | policy               |   mean_regret10 |   mean_ndcg10 |   mean_precision10 |   pools |   actual_coverage |
|:-------------------------------|:---------------------|----------------:|--------------:|-------------------:|--------:|------------------:|
| Task_A_new_entry               | BLEND_LEARNED_RISK   |        0.177489 |      0.362352 |          0.0337662 |     231 |          0.802083 |
| Task_A_new_entry               | HISTORICAL_HS2_ERROR |        0.294372 |      0.308751 |          0.0329004 |     231 |          0.802083 |
| Task_A_new_entry               | MODEL_DISAGREEMENT   |        0.510823 |      0.290794 |          0.04329   |     231 |          0.802083 |
| Task_A_new_entry               | SCORE_MARGIN         |        0.376623 |      0.297662 |          0.0385281 |     231 |          0.802083 |
| Task_A_new_entry               | STATIC_LEARNED_RISK  |        0.264069 |      0.308939 |          0.0320346 |     231 |          0.802083 |
| Task_B_existing_underexploited | HISTORICAL_HS2_ERROR |        0.978355 |      0.623887 |          0.454308  |     231 |          0.802083 |
| Task_B_existing_underexploited | MODEL_DISAGREEMENT   |        2.33333  |      0.586423 |          0.513616  |     231 |          0.802083 |
| Task_B_existing_underexploited | SCORE_MARGIN         |        2.39827  |      0.579755 |          0.496733  |     231 |          0.802083 |
| Task_B_existing_underexploited | STATIC_LEARNED_RISK  |        0.995671 |      0.609986 |          0.450412  |     231 |          0.802083 |

Paired HS2-block estimates for this diagnostic are in `F4_simple_gate_inference.csv`; they standardize coverage using the evaluation-year risk-score distribution and are not a frozen-threshold deployment test. Fixed development-threshold coverage remains the operational diagnostic. Risk reduction is conditional on accepted pools; no human decisions or deferred-case outcomes are recorded.

## Regret contrasts for the matched diagnostic

| task                           | estimand                                        | metric        | contrast                                   |   estimate |     ci_low |     ci_high |   p_signflip |   paired_hs2_blocks | interpretation                                                                                                                                                    |    p_holm |
|:-------------------------------|:------------------------------------------------|:--------------|:-------------------------------------------|-----------:|-----------:|------------:|-------------:|--------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------:|
| Task_A_new_entry               | matched 80% outer-score-distribution diagnostic | mean_regret10 | BLEND_LEARNED_RISK - STATIC_LEARNED_RISK   | -0.0823045 | -0.154321  | -0.0205761  |     0.010499 |                  81 | exploratory HS2-macro paired diagnostic; each HS2 averages accepted pools across years; differing accepted sets; fixed development-threshold operation is primary | 0.0419958 |
| Task_A_new_entry               | matched 80% outer-score-distribution diagnostic | mean_regret10 | BLEND_LEARNED_RISK - SCORE_MARGIN          | -0.0679012 | -0.154321  |  0.0102881  |     0.121588 |                  81 | exploratory HS2-macro paired diagnostic; each HS2 averages accepted pools across years; differing accepted sets; fixed development-threshold operation is primary | 0.159984  |
| Task_A_new_entry               | matched 80% outer-score-distribution diagnostic | mean_regret10 | BLEND_LEARNED_RISK - HISTORICAL_HS2_ERROR  | -0.0577778 | -0.12      | -0.00444444 |     0.079992 |                  75 | exploratory HS2-macro paired diagnostic; each HS2 averages accepted pools across years; differing accepted sets; fixed development-threshold operation is primary | 0.159984  |
| Task_A_new_entry               | matched 80% outer-score-distribution diagnostic | mean_regret10 | BLEND_LEARNED_RISK - MODEL_DISAGREEMENT    | -0.086758  | -0.173516  | -0.0114155  |     0.039996 |                  73 | exploratory HS2-macro paired diagnostic; each HS2 averages accepted pools across years; differing accepted sets; fixed development-threshold operation is primary | 0.119988  |
| Task_B_existing_underexploited | matched 80% outer-score-distribution diagnostic | mean_regret10 | STATIC_LEARNED_RISK - SCORE_MARGIN         |  0.0238095 | -0.0714286 |  0.127706   |     0.690031 |                  77 | exploratory HS2-macro paired diagnostic; each HS2 averages accepted pools across years; differing accepted sets; fixed development-threshold operation is primary | 0.977502  |
| Task_B_existing_underexploited | matched 80% outer-score-distribution diagnostic | mean_regret10 | STATIC_LEARNED_RISK - HISTORICAL_HS2_ERROR |  0.0299145 |  0         |  0.0683761  |     0.251775 |                  78 | exploratory HS2-macro paired diagnostic; each HS2 averages accepted pools across years; differing accepted sets; fixed development-threshold operation is primary | 0.755324  |
| Task_B_existing_underexploited | matched 80% outer-score-distribution diagnostic | mean_regret10 | STATIC_LEARNED_RISK - MODEL_DISAGREEMENT   | -0.0375587 | -0.138498  |  0.0446009  |     0.488751 |                  71 | exploratory HS2-macro paired diagnostic; each HS2 averages accepted pools across years; differing accepted sets; fixed development-threshold operation is primary | 0.977502  |
