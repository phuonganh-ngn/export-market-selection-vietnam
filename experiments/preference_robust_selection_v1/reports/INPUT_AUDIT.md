# Input and candidate audit

## Sources and scope

The extension reuses the audited HS12 BACI V202601-derived panel and saved rolling-origin model predictions. It does not redownload trade data or alter prior artifacts. BACI HS12 V202601 ends in 2024, so no unexamined 2025 HS12 outcome is available in that release.

The evaluation panel has 62,797 unique ex-ante destination–HS2–year keys across 224 destinations and 96 HS2 chapters, for 2022–2024. It contains 62,238 observed outcome labels and 559 unknown labels.

## Population reconciliation

Task A contains 41,131 keys; Task B contains 12,962; 6,511 keys satisfy both audited rules; 15,215 keys satisfy neither rule. Therefore the task-expanded manifest has 54,093 rows but only 47,582 distinct keys. The overlap arises because entry inactivity means each of the previous three verified annual flows is below USD 50,000, while expansion eligibility requires positive lagged export and below-median historical share. A sub-threshold positive relationship can satisfy both definitions. The rules are preserved rather than silently forced into mutually exclusive classes.

All 559 unknown outcome labels are retained as missing. Task A contains all 559 unknowns (187, 150, and 222 by evaluation year); Task B has no unknown labels in its eligible sample. The full universe includes 15,215 keys in neither task, mainly existing relationships that do not meet the audited underexploited rule or the entry inactivity rule.

## Temporal and source-vintage audit

Each outer row uses `feature_year = outcome_year - 2`. Reused prediction files are those from the previously verified two-year-lag rolling-origin runs. WDI/WGI historical series are revised vintages, so the evaluation is retrospective rather than point-in-time. Since 2022–2024 informed the prior research and this extension, this analysis is exploratory; no independent holdout exists.

## Eligibility and missing outcomes

Candidate eligibility is reconstructed from the audited panel and historical rules before reading the test outcomes. Outcome observability is stored separately. Unknown outcomes are not converted to zero. The full candidate manifest includes eligibility flags, historical cutoff, label-observed status, and task overlap.

## Decision interpretation

The 10-market and 5-market constraints are cardinality budgets, not monetary promotion budgets. No country-specific cost data are available. Predictive scenarios are finite model-disagreement stress cases, not a calibrated probability law.
