# Input and temporal audit

This extension reuses the audited VNM candidate manifest, archived out-of-time scores, historical rolling-origin classifier predictions, and HS2 panel. The raw BACI archive is CEPII HS12 V202601. No previous research artifact was modified.

- Candidate rows in the paired Task A/B manifest: 42,699.
- Task A unknown outcome labels: 556; retained as unknown.
- Outer years: 2022, 2023, 2024; the years were repeatedly inspected in prior M1 development and are exploratory for this extension.
- Historical OOF classifier years: 2018, 2019, 2020, 2021. Fold provenance states training outcomes end no later than prediction year minus two.
- Outcome in the source panel is observed only when BACI provides a positive destination-HS2 import market; missing outcomes are not treated as zero.
- WDI/WGI are revised historical vintages, so point-in-time deployment validity is not claimed.

The Task A label is `1{V >= USD 50,000}` within a three-year verified-zero history window ending at y-2. It measures a qualifying new-to-exporter product-market relationship after the observed zero window; it does not prove first-ever historical trade.

See also the candidate reconciliation and source/timing audit in `FINAL_CANDIDATE_AND_TEMPORAL_AUDIT.md` under the prior experiment.

See also the candidate reconciliation and source/timing audit in `FINAL_CANDIDATE_AND_TEMPORAL_AUDIT.md` under the prior experiment.

See also the candidate reconciliation and source/timing audit in `FINAL_CANDIDATE_AND_TEMPORAL_AUDIT.md` under the prior experiment.

See also the candidate reconciliation and source/timing audit in `FINAL_CANDIDATE_AND_TEMPORAL_AUDIT.md` under the prior experiment.

See also the candidate reconciliation and source/timing audit in `FINAL_CANDIDATE_AND_TEMPORAL_AUDIT.md` under the prior experiment.
