# Literature and novelty audit

**Status: VERIFIED SOURCES; NO NOVEL ALGORITHM CLAIM.** Bibliographic records and method summaries below were checked against publisher proceedings, author-hosted papers, and the trade-study records linked inline on 12 October 2026.

## Closest literature and overlap

| Work | Established method and overlap | Relation to ERGM |
|---|---|---|
| Geifman & El-Yaniv (2017), [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500), NeurIPS 2017 | Selective prediction/reject option trades prediction coverage for selective risk using a confidence score and a threshold; provides risk-control results under its sampling assumptions. | ERGM also has an accept/defer decision, but at an HS2-year portfolio level and with expert-regret outcomes. The threshold/coverage idea is established. |
| Geifman & El-Yaniv (2019), [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html), ICML/PMLR 97 | Jointly trains prediction and selection heads for target coverage; evaluates risk–coverage. | ERGM is a post-hoc gate over fixed forecast/ranking experts, not an integrated reject-option neural network. |
| Mozannar & Sontag (2020), [Consistent Estimators for Learning to Defer to an Expert](https://proceedings.mlr.press/v119/mozannar20b.html), ICML/PMLR 119 | Learns predictor/rejector with expert decision observations and a cost-sensitive consistent surrogate. | Human-decision labels are central to their defer-to-human estimand. ERGM has no such labels and cannot estimate human-review accuracy. |
| Hemmer et al. (2023), [Learning to Defer with Limited Expert Predictions](https://doi.org/10.1609/aaai.v37i5.25742), AAAI 37(5) | Uses representations and an expertise predictor to reduce required expert predictions; evaluates synthetic and real radiologist decisions. | Reinforces that a real human-AI collaboration claim requires expert-performance information. ERGM's review handoff is workload triage only. |
| Mao, Mohri & Zhong (2025), [Mastering Multiple-Expert Routing](https://proceedings.mlr.press/v267/mao25c.html), ICML/PMLR 267 | Studies multi-expert routing and two-stage deferral, with consistency and learning guarantees. | Expert routing itself is established. ERGM's domain-specific unit is a complete export-market Top-K pool, but the proposed ridge gate is not a new learning-to-defer algorithm. |
| Alves et al. (2024 preprint), [Cost-Sensitive Learning to Defer to Multiple Experts with Workload Constraints](https://arxiv.org/abs/2403.06906) | Models expert error and allocates deferrals under capacity/workload constraints. | Workload-aware deferral is existing. ERGM's coverage targets are evaluation operating points, not measured review capacity or costs. |
| Aucamp, Steenkamp & Bezuidenhout, [Comparing International Market Selection Methods Using Export Potential Values for South Africa](https://doi.org/10.1080/08853908.2023.2166628), *Journal of International Trade & Economic Development* (2023) | Compares export-potential ranking sources and documents substantial ranking disagreement. | Establishes that export-market ranking disagreement and post-prediction prioritization are already recognized questions. ERGM adds retrospective model routing/abstention evidence, not the general idea of rankings. |
| Gopinath, Batarseh & Beckman (2020), [Machine Learning in Gravity Models: An Application to Agricultural Trade](https://doi.org/10.3386/w27151), NBER Working Paper 27151 | Applies supervised and unsupervised ML to agricultural trade prediction and compares with traditional approaches over temporal projections. | Establishes ML/gravity trade prediction as prior art; the present question concerns selecting a fixed expert for a product-market portfolio and abstaining at the pool level. |
| [Relatedness and product complexity meet gravity models of international trade](https://doi.org/10.1016/j.joitmc.2024.100288), *Journal of Open Innovation* (2024) | Uses random forest, trade relatedness and product complexity to assess export diversification potential. | Export opportunity identification with interpretable trade features is existing; ERGM should not claim this broad objective as novel. |

## ERGM component classification

| Component | Classification | Basis |
|---|---|---|
| Routing between PPML and LightGBM | EXISTING | Multi-expert routing/mixture-of-experts are established. |
| Reject/defer at requested coverage levels | EXISTING | Selective prediction and risk–coverage are established. |
| Gate based on out-of-fold historical pool regret | EXISTING BUT ADAPTED | A two-stage, outcome-supervised model-selection gate applied to ranked destination portfolios. |
| HS2 × year as the decision unit and Top-10 regret target | EXISTING BUT ADAPTED | Domain-specific decision loss/unit; no general theoretical novelty is demonstrated by using it. |
| Reasons and review dossier for deferred pools | EXISTING BUT ADAPTED | Auditable explanation/reporting for a fixed gate; no new explanation algorithm. |
| Joint model routing plus portfolio-level selective automation in export selection | POTENTIALLY DISTINCTIVE empirical application | Potentially useful domain-specific evidence, but not a new general method. |
| Human-AI collaboration benefit | UNVERIFIED NOVELTY / NOT IDENTIFIED | No historical exporter-expert choices or post-review outcomes exist. |

## Novelty gate and contribution decision

The proposed implementation is a **two-stage application of established expert routing and selective prediction** to product-year Top-K trade-market portfolios. The testable adaptation is the pool-level target: a gate estimates subsequent portfolio regret for two fixed, heterogeneous model experts and can abstain on the entire HS2-year recommendation. This is distinguishable from candidate-level selective classification as a decision unit, but the extension does not supply a new consistency theorem, risk guarantee, or novel deferral objective.

We therefore do **not** call ERGM a new algorithm. The potentially supportable contribution is empirical: whether out-of-time pool-regret routing and a risk-ranked review queue outperform static routing and simple confidence/disagreement gates at comparable coverage. That claim remains exploratory because 2022–2024 informed earlier M1 work, only a small number of product-year pools are available, and actual human-review outcomes are absent. Any positive result would motivate independent future validation, not establish deployed human-AI value.

## Source verification notes

Publisher records were used for PMLR/ICML and AAAI citation metadata. The CEPII [BACI 202601 release notes](https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/release_notes_202601.pdf) show the source series through 2024 for the relevant current release; the existing panel also ends in 2024 for observed BACI outcomes. No 2025 trade outcome was added in this extension. The CEPII PDF endpoint returned an HTTP 403 to the browser reader, so the endpoint was verified from CEPII search metadata rather than a fully parsed document; the local data coverage was independently checked. This does not affect the no-fresh-holdout conclusion.
