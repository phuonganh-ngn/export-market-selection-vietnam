# Explainable routing and deferral

The fitted Ridge gate is additive in standardized numeric and HS2 one-hot features. For each pool, `T6_routing_explanations.csv` lists the selected expert, alternative, both estimated regret scores and leading raw score-difference contributions. Nonnegative clipping can make both estimated regrets tie at zero; those ties are resolved in favor of PPML and flagged. Counterfactual rows locate a one-feature change that ties the untruncated Ridge outputs; clipping can prevent that tie from changing the final route, so this is a model-score diagnostic rather than a verified decision-switch counterfactual.

| task                           | feature     |   outcome_year |   coefficient_lightgbm_minus_ppml_regret | interpretation                                                          |
|:-------------------------------|:------------|---------------:|-----------------------------------------:|:------------------------------------------------------------------------|
| Task_A_new_entry               | cat__hs2_31 |           2024 |                                -0.303471 | linear association in scaled/categorical gate feature space; not causal |
| Task_A_new_entry               | cat__hs2_30 |           2024 |                                 0.297063 | linear association in scaled/categorical gate feature space; not causal |
| Task_A_new_entry               | cat__hs2_04 |           2023 |                                 0.265517 | linear association in scaled/categorical gate feature space; not causal |
| Task_A_new_entry               | cat__hs2_24 |           2022 |                                -0.25133  | linear association in scaled/categorical gate feature space; not causal |
| Task_A_new_entry               | cat__hs2_34 |           2024 |                                 0.208224 | linear association in scaled/categorical gate feature space; not causal |
| Task_A_new_entry               | cat__hs2_51 |           2023 |                                -0.178359 | linear association in scaled/categorical gate feature space; not causal |
| Task_A_new_entry               | cat__hs2_60 |           2023 |                                 0.177188 | linear association in scaled/categorical gate feature space; not causal |
| Task_A_new_entry               | cat__hs2_20 |           2022 |                                -0.175452 | linear association in scaled/categorical gate feature space; not causal |
| Task_A_new_entry               | cat__hs2_40 |           2024 |                                -0.174357 | linear association in scaled/categorical gate feature space; not causal |
| Task_A_new_entry               | cat__hs2_07 |           2024 |                                 0.172945 | linear association in scaled/categorical gate feature space; not causal |
| Task_B_existing_underexploited | cat__hs2_89 |           2024 |                                 1.46555  | linear association in scaled/categorical gate feature space; not causal |
| Task_B_existing_underexploited | cat__hs2_85 |           2024 |                                -1.13625  | linear association in scaled/categorical gate feature space; not causal |
| Task_B_existing_underexploited | cat__hs2_09 |           2024 |                                -1.12607  | linear association in scaled/categorical gate feature space; not causal |
| Task_B_existing_underexploited | cat__hs2_71 |           2022 |                                 1.07354  | linear association in scaled/categorical gate feature space; not causal |
| Task_B_existing_underexploited | cat__hs2_46 |           2024 |                                 0.764832 | linear association in scaled/categorical gate feature space; not causal |
| Task_B_existing_underexploited | cat__hs2_73 |           2023 |                                -0.745296 | linear association in scaled/categorical gate feature space; not causal |
| Task_B_existing_underexploited | cat__hs2_15 |           2024 |                                 0.606119 | linear association in scaled/categorical gate feature space; not causal |
| Task_B_existing_underexploited | cat__hs2_04 |           2024 |                                -0.523825 | linear association in scaled/categorical gate feature space; not causal |
| Task_B_existing_underexploited | cat__hs2_72 |           2024 |                                 0.484875 | linear association in scaled/categorical gate feature space; not causal |
| Task_B_existing_underexploited | cat__hs2_31 |           2024 |                                 0.483133 | linear association in scaled/categorical gate feature space; not causal |

Raw gate score-difference reconstruction passed for 576/576 pools; the selected expert was reproduced for 576/576 pools.
The deferred review dossier gives both expert Top-10 lists, their overlap, differing destinations, predicted regret score and rule-based deferral reason. Explanations do not use human judgments and are not causal. The method does not claim a new XAI algorithm.
