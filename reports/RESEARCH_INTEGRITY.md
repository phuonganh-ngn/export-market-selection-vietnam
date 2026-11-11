# Research integrity

## Provenance and vintages

Raw source files are preserved under `data/raw` and hashed in `data_source_registry.csv`. BACI V202601 reports trade values in thousand current USD and is converted by 1,000. WDI values are from the World Bank API as accessed for this run; historical series can be revised. WGI 2026 recalculates historical estimates. GeoDist is a time-invariant CEPII source.

## Missing data and zeros

The panel does not globally turn missing records into zero. A candidate requires positive world import demand at its information date. A missing Vietnam flow is treated as zero only in an outcome-year destination–HS2 cell with positive observed world imports. Otherwise the target remains unavailable. Indicator imputation occurs inside each training fit with missingness indicators.

## Temporal and statistical checks

The two-year lag is the primary retrospective design; source publication vintages are not archived, so it is not a real-time deployment test. Split manifests and automated assertions record feature-year and training-outcome cutoffs. Destination-block bootstrap intervals preserve within-destination product-year dependence. Three evaluation years limit temporal inference; Holm adjustment is applied to prespecified forecast comparisons.

## Recovered execution issues

- Initial CEPII GeoDist preparation exposed dot-formatted missing values in the population-weighted distance field. The numeric parser and key mapping were corrected; the full BACI scan, aggregation checks, and panel preparation subsequently passed.
- An initial PPML fit with ridge alpha 0.01 did not converge within 2,000 iterations. The regularized PPML specification was set to alpha 0.1 and all three primary evaluation-year fits converged; diagnostics are saved.
- An initial E1 Duan-smearing retransformation produced implausible forecasts. Downstream outputs were stopped, predictions were recalibrated using an inner chronological validation scale selected by original-USD MAE, and the primary E1 tables were rerun. Superseded metrics were not used.
- The first E2 summary counted repeated stochastic seeds as duplicate candidate markets. E2 was rerun with metrics computed on unique destination–HS2 candidates within each seed; the saved predictions contain no duplicate model–seed–year–HS2–destination keys.
- The first E3 run saved all 1,226,225 historical and prospective ranking rows but stopped during Top-K extraction because a DataFrame rank method was accessed as a column. Derived Top-K, HS2-year metric, stability, and overlap files were rebuilt from those saved ranking rows without refitting; the ranking and stability column accessors were corrected in source.
- To keep CPU use bounded, E4 feature-group ablations use the prespecified seed 11 only. SHAP, ALE, and global/year/HS2 numerical explanation outputs use the full bounded random out-of-time sample.
- R14 initially failed on mixed integer/string HS2 join keys. Codes were normalized to zero-padded HS2 strings and R14 was rerun successfully; all 14 robustness checks are now marked completed.

No unresolved stage failures remain. The final outputs use the corrected/recovered artifacts above; superseded experimental metrics were not used.

## External validity

Market ranks are country–product observational backtest outputs, not firm-level recommendations and not causal estimates of realized policy or commercial gains. Results can be affected by BACI coverage, revisions, candidate eligibility, WDI/WGI missingness, and limited holdout years. Robustness rows not marked completed are not treated as passed.
