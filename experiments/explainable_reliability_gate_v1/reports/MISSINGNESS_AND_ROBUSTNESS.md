# Missingness and robustness

Unknown outcome labels remain NA. There are 556 unknown labels in the corrected strict-entry population and none in the existing-expansion population (the other three unknowns are outside both task populations). No unknown is recoded to zero. Complete-case performance is conditional on outcome observability.

| group                            | task                           |   outcome_year |    n |   observed |   unknown |   unknown_share |   quartile |
|:---------------------------------|:-------------------------------|---------------:|-----:|-----------:|----------:|----------------:|-----------:|
| task_year                        | Task_A_new_entry               |           2022 | 9846 |       9659 |       187 |     0.0189925   |        nan |
| task_year                        | Task_A_new_entry               |           2023 | 9913 |       9764 |       149 |     0.0150308   |        nan |
| task_year                        | Task_A_new_entry               |           2024 | 9978 |       9758 |       220 |     0.0220485   |        nan |
| task_year                        | Task_B_existing_underexploited |           2022 | 4320 |       4320 |         0 |     0           |        nan |
| task_year                        | Task_B_existing_underexploited |           2023 | 4387 |       4387 |         0 |     0           |        nan |
| task_year                        | Task_B_existing_underexploited |           2024 | 4255 |       4255 |         0 |     0           |        nan |
| task_year_import_demand_quartile | Task_A_new_entry               |           2022 | 3618 |       3438 |       180 |     0.0497512   |          0 |
| task_year_import_demand_quartile | Task_A_new_entry               |           2022 | 3141 |       3136 |         5 |     0.00159185  |          1 |
| task_year_import_demand_quartile | Task_A_new_entry               |           2022 | 2157 |       2155 |         2 |     0.000927214 |          2 |
| task_year_import_demand_quartile | Task_A_new_entry               |           2022 |  930 |        930 |         0 |     0           |          3 |
| task_year_import_demand_quartile | Task_A_new_entry               |           2023 | 3471 |       3324 |       147 |     0.0423509   |          0 |
| task_year_import_demand_quartile | Task_A_new_entry               |           2023 | 3163 |       3161 |         2 |     0.000632311 |          1 |
| task_year_import_demand_quartile | Task_A_new_entry               |           2023 | 2254 |       2254 |         0 |     0           |          2 |
| task_year_import_demand_quartile | Task_A_new_entry               |           2023 | 1025 |       1025 |         0 |     0           |          3 |
| task_year_import_demand_quartile | Task_A_new_entry               |           2024 | 3440 |       3223 |       217 |     0.0630814   |          0 |
| task_year_import_demand_quartile | Task_A_new_entry               |           2024 | 3108 |       3106 |         2 |     0.000643501 |          1 |
| task_year_import_demand_quartile | Task_A_new_entry               |           2024 | 2364 |       2363 |         1 |     0.000423012 |          2 |
| task_year_import_demand_quartile | Task_A_new_entry               |           2024 | 1066 |       1066 |         0 |     0           |          3 |
| task_year_import_demand_quartile | Task_B_existing_underexploited |           2022 |   68 |         68 |         0 |     0           |          0 |
| task_year_import_demand_quartile | Task_B_existing_underexploited |           2022 |  439 |        439 |         0 |     0           |          1 |
| task_year_import_demand_quartile | Task_B_existing_underexploited |           2022 | 1337 |       1337 |         0 |     0           |          2 |
| task_year_import_demand_quartile | Task_B_existing_underexploited |           2022 | 2476 |       2476 |         0 |     0           |          3 |
| task_year_import_demand_quartile | Task_B_existing_underexploited |           2023 |   42 |         42 |         0 |     0           |          0 |
| task_year_import_demand_quartile | Task_B_existing_underexploited |           2023 |  434 |        434 |         0 |     0           |          1 |
| task_year_import_demand_quartile | Task_B_existing_underexploited |           2023 | 1303 |       1303 |         0 |     0           |          2 |
| task_year_import_demand_quartile | Task_B_existing_underexploited |           2023 | 2608 |       2608 |         0 |     0           |          3 |
| task_year_import_demand_quartile | Task_B_existing_underexploited |           2024 |   36 |         36 |         0 |     0           |          0 |
| task_year_import_demand_quartile | Task_B_existing_underexploited |           2024 |  390 |        390 |         0 |     0           |          1 |
| task_year_import_demand_quartile | Task_B_existing_underexploited |           2024 | 1259 |       1259 |         0 |     0           |          2 |
| task_year_import_demand_quartile | Task_B_existing_underexploited |           2024 | 2570 |       2570 |         0 |     0           |          3 |

Binary Task A regret bounds use exact enumeration of feasible success counts among missing selected and unselected candidates. The same joint completion is used for the selected portfolio and hindsight oracle, so bounds are sharp under the binary support and fixed Top-10 policy. This count-based procedure covers large missing pools without enumerating every label vector. Dollar outcomes remain unbounded from available evidence; no finite USD bounds are asserted.

Exact binary-bound rows: 576; any non-estimated rows remain explicitly marked. Missingness and decision uncertainty are separate.

The dataset has 2022–2024 outer outcomes, which prior M1 work has already inspected. WDI/WGI are revised historical vintages. No fresh independent holdout is available in the local BACI V202601 panel, whose observed outcomes end in 2024. All policy evidence is exploratory retrospective evidence.
