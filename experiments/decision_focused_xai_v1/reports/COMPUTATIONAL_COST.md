# Computational cost

The fit-time entries are read from the stored model metadata and inner-fold tuning logs; cached prediction regeneration is not counted as a fresh model fit. LambdaMART is a tree ensemble, so a neural-style scalar parameter count is not applicable; tree counts and feature counts are provided.

| stage                         | task                           |   fits |   recorded_fit_seconds |   mean_fit_seconds |   total_inference_seconds |   mean_trees |
|:------------------------------|:-------------------------------|-------:|-----------------------:|-------------------:|--------------------------:|-------------:|
| inner_chronological_tuning    | Task_A_new_entry               |      9 |                3.67461 |          0.40829   |                 0         |      60.6667 |
| inner_chronological_tuning    | Task_B_existing_underexploited |      9 |                2.27144 |          0.252382  |                 0         |      66.4444 |
| outer_final_fit               | Task_A_new_entry               |     15 |                4.55971 |          0.303981  |                 0.113281  |      83      |
| outer_final_fit               | Task_B_existing_underexploited |     15 |                1.8629  |          0.124193  |                 0.0491422 |      90.3333 |
| outer_rank_xendcg_sensitivity | Task_A_new_entry               |     15 |                3.49167 |          0.232778  |                 0.128139  |      83      |
| outer_rank_xendcg_sensitivity | Task_B_existing_underexploited |     15 |                1.28891 |          0.0859277 |                 0.0526027 |      90.3333 |

Total recorded fit time: 17.15 seconds across 78 fit/configuration records. Inference time is recorded separately for final outer-fold rankers.
