# Temporal patterns and ranking explanations

Numerical case-level top-ten lists are saved in `results/G4_explanation_cases.csv`; candidate-level ranks for all four ranking methods are in `results/G4_ranking_displacement.csv`. Cases are selected by the five most negative and five most positive product-year differences in GRU versus LightGBM Top-10 success counts, then show the actual sequences and outcomes. This selection is retrospective and intended for mechanism diagnosis, not confirmatory inference.

The GRU's output is a predictive entry score. The shown historical sequences do not imply causal effects of changing an economic feature. Score reconstruction is verified by the saved model logits and rank rule; see `results/G4_explanation_fidelity.csv`.
