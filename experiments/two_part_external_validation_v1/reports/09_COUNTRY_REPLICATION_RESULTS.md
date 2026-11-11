# 09 — Country-specific replication

B2 refits the same fixed LightGBM/PPML specifications within THA and IDN for each rolling origin, with outcomes no later than y−2. Local entry probabilities are Platt-calibrated using local rolling-origin OOF predictions only. Task B uses PPML and LightGBM export-level forecasts converted to nonnegative incremental scores; it is not a two-part entry model. Country/year/model summaries are in `results/B_country_replication.csv`; paired inference is in `results/B_country_paired_tests.csv`. Report within-country USD results only; absolute USD is not pooled across exporters.



The final sample-flow audit deduplicates candidates across the B1 and B2 scoring modes. Unique Task A candidate counts are THA 19,887 (532 unknown labels) and IDN 24,079 (548 unknown labels); Task B counts are THA 19,287 (one unknown label) and IDN 17,111 (none unknown). Each country-year has 96 HS2 pools. An earlier generated flow table double-counted B1/B2 rows; it was corrected and a regression test now verifies sample-flow counts against the independent source-coverage audit.
