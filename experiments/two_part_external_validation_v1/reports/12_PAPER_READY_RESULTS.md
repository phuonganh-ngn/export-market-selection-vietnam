# 12 — Paper-ready results (draft)

## Empirical strategy

For strictly inactive destination–HS2 markets, the entry event is outcome-year exports of at least USD 50,000 following three verified zero-export years ending at the two-year information cutoff. A LightGBM classifier estimates entry probability. A pooled log-link PPML model estimates conditional export value among verified qualifying entries. Their product estimates expected qualified export value, not the unconditional mean of total trade. Missing destination–HS2 outcome markets are retained as unknown.

## Vietnam results

On the 2022–2024 retrospective sample, LightGBM's macro NDCG@10 is 0.2947 and the equal-weight Gravity–LightGBM rank blend's is 0.3319. The Two-Part model's NDCG@10 is 0.2774. The observed qualified export value captured in the full-ranked Top-10 averages approximately USD 534,664 per HS2-year pool for LightGBM, USD 545,938 for the blend, and USD 443,233 for Two-Part. The Two-Part model also selects fewer successful entries on average. The evaluation years have informed prior M1 development, so these are exploratory retrospective findings.

## External validation

Frozen VNM-trained scoring rules were transferred to THA and IDN, and the same fixed specifications were separately fit within each exporter. Results vary by country and task; the hurdle score does not show a consistent advantage over LightGBM or rank blending. PPML remains strong for existing-market expansion. THA evaluation is exploratory after limited outcome exposure during debugging; IDN is the prespecified external replication, but the three test years overlap the Vietnam temporal window and share global shocks. No causal effect of recommendations is identified.

## Limitations

WDI/WGI vintages are revised; the conditional successful-entry sample is relatively small; unknown outcomes limit complete population identification; export values are highly skewed; country-specific results rely on two external exporters; and no point-in-time future year is available. The paper should describe task-dependent empirical evidence rather than a novel hurdle method or universal superiority.


The paired HS2-block comparison for Vietnam estimated a rank-blend improvement over LightGBM of 0.0372 NDCG@10 (95% interval 0.0058–0.0710; Holm-adjusted p=0.0272), while the qualified-USD@10 difference was USD 11,274 (95% interval −USD 89,488 to USD 118,503; Holm-adjusted p=1.00). The two-part model was below the rank blend by 0.0545 NDCG points (Holm-adjusted p=0.0237) and did not show a qualified-value advantage. These results are exploratory because the outer years informed previous M1 development. The external replication is geographic only: the countries share the 2022–24 outcome period, and THA is exploratory after limited outcome exposure. No broad temporal or causal conclusion follows.
