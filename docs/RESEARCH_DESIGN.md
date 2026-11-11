# Prespecified research design

## Unit, scope and time

Vietnam is the sole exporter. Product classification is HS 2012; the primary grain is importer × HS2 chapter × outcome year. The sample uses the verified 2012–2024 BACI HS12 period. Outer test years are 2022, 2023 and 2024. Annual predictors in the primary specification are lagged two years; the one-year specification is a retrospective sensitivity check.

## Coverage and zeros

BACI reports positive bilateral flows rather than a complete zero-filled matrix. We form a market candidate only when the destination has positive world imports in the relevant HS2 chapter at the information date. Within an outcome year, an absent Vietnam flow is coded zero only if BACI records positive world imports for that destination–HS2 cell. If no such import record exists, the target remains unavailable. The analysis therefore does not interpret all absent source rows as verified zeros.

## Targets

Export value is nominal current USD. Market entry is reaching USD 50,000 in an outcome year after no destination–HS2 Vietnam flow over USD 50,000 in the prior three observed years through the information date. USD 10,000 and USD 100,000 thresholds are robustness analyses. Incremental opportunity is `max(0, predicted export value − export value at the information date)`. Realized increment uses the same reference date. Market-share changes and relative growth are reported only when denominators are positive and meaningful.

## Evaluation

For evaluation year y, only outcome rows dated at or before y−L are used for fitting, where L is the selected information lag. Inner chronological validation selects among a bounded, predeclared set of configurations. Preprocessing is fitted within each training split. All market discovery and ranking metrics are evaluated within HS2 × year candidate pools; global Top-K lists are not used.

Forecast metrics include USD MAE/RMSE, log1p MAE, safe WAPE, and zero/positive-flow errors, reported both pooled and macro-averaged over HS2. Ranking metrics include precision, NDCG, realized incremental exports and entry success. Results are predictive and observational, not causal or firm-level recommendations.
