# Statistical validation

Primary paired outcome contrasts are computed on common task × HS2 × year × K pools where both policies select K markets with observed outcome labels. Descriptive all-pool complete-case-slot metrics and partial-identification bounds are shown separately.

The confidence intervals use resampling of HS2 chapters as blocks, retaining each sampled chapter's evaluation years and paired policies. Holm correction is applied within task × K × population × metric comparisons. Seeds are not treated as statistical observations.

| task                           |   k | policy_b                | population             |   paired_pool_count |   paired_hs2_blocks |   difference |      ci_low |    ci_high |   p_bootstrap |     holm_p |
|:-------------------------------|----:|:------------------------|:-----------------------|--------------------:|--------------------:|-------------:|------------:|-----------:|--------------:|-----------:|
| Task_A_new_entry               |   5 | Standard_minimax_regret | common_full_k_observed |                 288 |                  96 |  0.0595607   |  0.00978665 |  0.113499  |   0.0206598   | 0.0413196  |
| Task_A_new_entry               |   5 | Relative_robust_Weber   | common_full_k_observed |                 288 |                  96 |  0.0624793   |  0.0161838  |  0.111758  |   0.00866378  | 0.0346551  |
| Task_A_new_entry               |   5 | Gravity_PPML            | common_full_k_observed |                 288 |                  96 |  0.182551    |  0.107345   |  0.259527  |   0.000666445 | 0.00399867 |
| Task_A_new_entry               |   5 | LightGBM                | common_full_k_observed |                 288 |                  96 |  0.158125    |  0.0736965  |  0.245251  |   0.000666445 | 0.00399867 |
| Task_A_new_entry               |   5 | LambdaMART              | common_full_k_observed |                 288 |                  96 |  0.0228284   | -0.0539322  |  0.0982545 |   0.514495    | 0.514495   |
| Task_A_new_entry               |   5 | Equal_weight_MCDM       | common_full_k_observed |                 288 |                  96 |  0.0666579   |  0.0174287  |  0.117081  |   0.00866378  | 0.0346551  |
| Task_A_new_entry               |  10 | Standard_minimax_regret | common_full_k_observed |                 288 |                  96 |  0.0564392   |  0.0063244  |  0.111541  |   0.023992    | 0.11996    |
| Task_A_new_entry               |  10 | Relative_robust_Weber   | common_full_k_observed |                 288 |                  96 |  0.0413908   | -0.00974308 |  0.0949769 |   0.121959    | 0.487837   |
| Task_A_new_entry               |  10 | Gravity_PPML            | common_full_k_observed |                 288 |                  96 | -0.0169304   | -0.106086   |  0.069828  |   0.703765    | 0.703765   |
| Task_A_new_entry               |  10 | LightGBM                | common_full_k_observed |                 288 |                  96 |  0.118028    |  0.0392189  |  0.20536   |   0.00266578  | 0.0159947  |
| Task_A_new_entry               |  10 | LambdaMART              | common_full_k_observed |                 288 |                  96 | -0.0775931   | -0.177797   |  0.0279147 |   0.137288    | 0.487837   |
| Task_A_new_entry               |  10 | Equal_weight_MCDM       | common_full_k_observed |                 288 |                  96 |  0.0359459   | -0.0114722  |  0.0838536 |   0.141286    | 0.487837   |
| Task_B_existing_underexploited |   5 | Standard_minimax_regret | common_full_k_observed |                 288 |                  96 | -0.0133138   | -0.0681766  |  0.0422115 |   0.622459    | 1          |
| Task_B_existing_underexploited |   5 | Relative_robust_Weber   | common_full_k_observed |                 288 |                  96 | -0.0218893   | -0.0799518  |  0.0363875 |   0.448517    | 1          |
| Task_B_existing_underexploited |   5 | Gravity_PPML            | common_full_k_observed |                 288 |                  96 | -0.00380983  | -0.0828741  |  0.0701205 |   0.914362    | 1          |
| Task_B_existing_underexploited |   5 | LightGBM                | common_full_k_observed |                 288 |                  96 | -0.041865    | -0.120891   |  0.0371671 |   0.291903    | 1          |
| Task_B_existing_underexploited |   5 | LambdaMART              | common_full_k_observed |                 288 |                  96 |  0.0826332   | -0.00891119 |  0.179956  |   0.0733089   | 0.439853   |
| Task_B_existing_underexploited |   5 | Equal_weight_MCDM       | common_full_k_observed |                 288 |                  96 | -0.000921796 | -0.0565342  |  0.0515017 |   0.965012    | 1          |
| Task_B_existing_underexploited |  10 | Standard_minimax_regret | common_full_k_observed |                 282 |                  96 |  0.0958521   |  0.0180178  |  0.169695  |   0.0106631   | 0.0426524  |
| Task_B_existing_underexploited |  10 | Relative_robust_Weber   | common_full_k_observed |                 282 |                  96 |  0.133254    |  0.0576182  |  0.214043  |   0.00133289  | 0.00799733 |
| Task_B_existing_underexploited |  10 | Gravity_PPML            | common_full_k_observed |                 282 |                  96 | -0.0910378   | -0.196031   |  0.0068217 |   0.0679773   | 0.135955   |
| Task_B_existing_underexploited |  10 | LightGBM                | common_full_k_observed |                 282 |                  96 | -0.14596     | -0.254432   | -0.0335061 |   0.011996    | 0.0426524  |
| Task_B_existing_underexploited |  10 | LambdaMART              | common_full_k_observed |                 282 |                  96 | -0.0690891   | -0.204082   |  0.0613903 |   0.289237    | 0.289237   |
| Task_B_existing_underexploited |  10 | Equal_weight_MCDM       | common_full_k_observed |                 282 |                  96 |  0.11765     |  0.0433893  |  0.195226  |   0.00266578  | 0.0133289  |

Only three outer years are available, and they have informed previous M1 work. Cross-sectional HS2 resampling addresses product heterogeneity under observed years, not temporal transportability. The primary estimand is the paired difference in mean observed-outcome worst-preference regret; negative PR-EMS-minus-baseline values favor PR-EMS.

The finite-scenario solver is checked against exhaustive enumeration on small portfolios. Solver status, reported gap, and selected candidate cardinality are saved. HiGHS numerical failures and the retry policy are documented in `results/solver_validation.csv` and `METHOD_FORMULATION.md`.
