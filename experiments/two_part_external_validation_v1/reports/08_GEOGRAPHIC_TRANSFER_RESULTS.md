# 08 — Zero-shot geographic transfer

B1 applies VNM-fitted classifiers and PPML coefficients using the frozen feature schema, with exporter-specific trade histories and CEPII bilateral distances rebuilt. No external labels are used to fit, tune or recalibrate B1. `results/B_transfer_feasibility.csv` documents schema transport. Unknown destination one-hot categories are ignored by the saved M1 preprocessing rule. This is geographic domain transfer, not temporal validation. Full results, including all models and years, are in `results/B_zero_shot_transfer.csv`. Transfer is heterogeneous: neither Two-Part nor the blend dominates all tasks/countries.


## Final independent check and paired evidence

For Task A, VNM-trained zero-shot rank blending versus LightGBM yielded IDN macro NDCG@10 0.3033 vs 0.2756 and qualified USD@10 USD 2.939m vs USD 1.998m per HS2-year. IDN country-specific refitting yielded 0.3226 vs 0.3223 and USD 3.940m vs USD 3.605m. Neither IDN USD difference survives Holm adjustment. THA zero-shot results were 0.2876 vs 0.2437 NDCG and USD 140.4k vs USD 71.7k; this is exploratory because outcomes were exposed during debugging. The Task A Two-Part model did not dominate: its NDCG was 0.2666/0.2622 for IDN zero-shot/refit and 0.2769/0.2718 for THA, versus blend 0.3033/0.3226 and 0.2876/0.2845.

For Task B, gravity PPML had higher NDCG than LightGBM and rank blend in both exporters and modes: IDN 0.5209 (zero-shot) / 0.5144 (refit), THA 0.5623 / 0.5562. These are geographic replications over the same global 2022–24 years, not a new temporal holdout. NDCG is complete-case and averages only pools with identified positive relevance (IDN Task A 153/288; THA Task A 114/288); observed USD sums are lower bounds when an unknown label is selected.
