# Contribution assessment

## Evidence-based answers

1. LambdaMART Task A NDCG@10 is 0.4148 and Precision@10 is 0.2247. The strongest baseline by NDCG@10 is LightGBM (0.4495). This is a comparative result, not evidence of causal market-entry benefits.
2. For Task B, LambdaMART NDCG@10 is 0.5592, compared with LightGBM 0.5406 and PPML 0.5925. Mean LambdaMART capped-gain realized-outcome regret@10 is 2.3542.
3. Confirmatory paired comparisons (pool-weighted estimand; HS2-block resampling): Task A LambdaMART minus LightGBM NDCG@10: Δ=-0.0347; 95% CI [-0.0567, -0.0126]; Holm p=0.0114; Precision@10: Δ=-0.0219; 95% CI [-0.0330, -0.0104]; Holm p=0.0018. Against Task A PPML NDCG@10: Δ=-0.0221; 95% CI [-0.0460, 0.0009]; Holm p=0.1372.
4. Task B LambdaMART minus PPML NDCG@10: Δ=-0.0333; 95% CI [-0.0590, -0.0082]; Holm p=0.0216; minus LightGBM: Δ=0.0186; 95% CI [-0.0147, 0.0526]; Holm p=0.2921. Task B regret difference versus PPML: Δ=0.2014; 95% CI [-0.0069, 0.4306]; Holm p=0.0742; versus LightGBM: Δ=-0.3160; 95% CI [-0.5972, -0.0451]; Holm p=0.0504. Negative regret differences favor LambdaMART.
5. Task B gain sensitivity changes the descriptive ordering: capped NDCG LambdaMART/PPML = 0.5592/0.5925; log1p-gain = 0.6396/0.6312; raw-USD gain = 0.5381/0.5630. This metric dependence weakens a universal ranking-superiority claim.
6. LambdaMART seed-pair Top-10 Jaccard averages are 0.739 for entry and 0.863 for expansion. The largest mean absolute score-attribution groups are destination_import_demand for entry and destination_identity for expansion; destination identity is a predictive identifier, not an economic mechanism.
7. HS2-block and destination-block uncertainty are cross-sectional. Destination bootstrap sensitivity intervals and HS2-block tests are reported separately because they use different dependence structures; three years do not establish future-cycle robustness.
8. Top-K stability, feature perturbation, actual pairwise score explanations and ablations are descriptive. They do not show that export promotion would cause the observed gains.
9. Optional rank_xendcg with LambdaMART-selected settings yields seed-mean NDCG@10 0.4131 for Task A and 0.5724 for Task B. It is close to pairwise LambdaMART in entry and higher for expansion, but remains below PPML on the primary capped-gain metric; no objective-specific tuning or inferential test was performed.
10. Revised historical WDI/WGI vintages and incomplete outcome reporting preclude a fully point-in-time deployment interpretation.

## Novelty assessment

Confirmed empirical contribution: applying direct pairwise/listwise ranking optimization to Vietnam HS2 destination selection under an audited ex-ante candidate universe, and evaluating complete-case ranking quality, realized-outcome regret, stability, and score explanations on common pools.
Methods borrowed from established work: LambdaMART, NDCG, retrospective regret, cluster bootstrap, additive tree contributions, feature ablation, and listwise ranking explanation research. The implementation does not establish a new general-purpose ranking or explanation algorithm.
The strongest defensible contribution depends on paired contrasts being robust and economically interpretable. If LambdaMART does not consistently outperform strong forecast-based rankings, the paper should foreground task alignment, candidate-set validity, regret/stability measurement, and a transparent negative or mixed result rather than algorithmic novelty.
This is not sufficient by itself to claim a new methodology or Q1 readiness; additional country/product contexts, vintage-aware data, and prospective or external validation would strengthen the design.
