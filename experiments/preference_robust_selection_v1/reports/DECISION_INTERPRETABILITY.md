# Decision interpretability

The explanation unit is the selected portfolio under the finite scenario minimax-regret rule, not a generic pointwise model prediction. For each robust portfolio, the numerical record gives the normalized central utility, worst utility, maximum predicted regret and scenario count. A one-for-one candidate swap is recomputed under the same joint scenario set; the swap table records whether that replacement lowers worst-case regret.

Selection-frequency outputs compare scenario-specific equal-weight top-K selections with the robust selection. These diagnose portfolio stability across the enumerated scenarios. A stable destination may be stable because it balances criteria; stability by itself does not establish high realized utility.

This run produced 8,621 portfolio-explanation rows and 99,565 swap diagnostics. Inspect `results/R6_portfolio_explanations.csv`, `results/R6_candidate_swap_analysis.csv`, and `results/R6_scenario_selection_frequency.csv` for actual country–product decisions.

The diagnostic recomputes the optimization objective under perturbed portfolio membership. It does not change real GDP, governance or trade demand and does not estimate causal effects. Standard SHAP is not relabeled as a robust-decision explanation method.
