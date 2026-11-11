# Method formulation

## Decision units and feasibility

For task (q\in\{A,B\}), product chapter (h), and decision origin (y), let (C_{qhy}) be the candidate destinations satisfying the frozen, historical eligibility rule. A policy selects (S\subseteq C_{qhy}), with (|S|=K\), for (K\in\{5,10\}). This is a cardinality decision budget, not a financial budget. No country-specific promotion costs are observed.

Task A is market entry: eligibility requires the audited three-year inactivity rule (each of the three prior verified annual exports is below USD 50,000); entry relevance is reaching USD 50,000 in year (y). Task B is existing-market expansion: eligibility follows the audited positive-lagged-export, positive-demand and below-training-median-share rule. These rules can overlap because a positive relationship below USD 50,000 can also be inactive under Task A. They remain separate tasks and are not forced to be mutually exclusive.

## Three distinct uncertainty objects

1. **Preference ambiguity** is the set (W\) of admissible weights. It describes unknown priorities, not random outcomes.
2. **Predictive ambiguity** is the finite scenario set (\Omega\) of historical rolling-origin model forecasts and training-only conditional outcome scales. It describes disagreement/stress in candidate utilities, not a calibrated probability distribution.
3. **Evaluation observability** is (R_i\in\{0,1\}), whether the outcome-year label is observed. If (R_i=0), its evaluation outcome is unknown. This is a partial-identification problem, not an adverse forecast scenario and not a zero.

The 559 missing labels enter outcome evaluation and identification bounds only. They do not enter policy construction or its risk set.

## Utilities

For each candidate (i\), each task has two bounded normalized objectives (u_{i1}(\omega),u_{i2}(\omega)\in[0,1]), and a preference vector (w=(w_1,1-w_1)\). The set utility is additive:

\[
U(S,w,\omega)=\sum_{i\in S}\{w_1u_{i1}(\omega)+(1-w_1)u_{i2}(\omega)\}.
\]

**Task A.** Objective 1 is predicted probability of successful entry, (p_i\). Objective 2 is predicted conditional export-scale utility given a successful entry. Historical conditional amount quantiles are estimated only from observed, successful, historically eligible training entries through the outer cutoff and within product/demand cells with chapter/global fallback. Amounts are mapped to ([0,1]) with a training-period 95th-percentile cap and a log transform with USD 50,000 scale. The two criteria stay separate: (p_i) is not multiplied by the amount, and neither is described as expected revenue. In realized evaluation, objective 1 is the observed binary entry; objective 2 is capped export-scale utility only for successful entries.

**Task B.** Objective 1 is the training-only probability of positive incremental exports. Objective 2 is the bounded utility of predicted positive incremental exports. The amount utility is (g(x)=\min\{\log(1+\max(x,0)/50{,}000),\log(1+c/50{,}000)\}/\log(1+c/50{,}000)\), where (c\) is the training-only 95th percentile of nonnegative incremental exports (with a documented minimum cap when needed). Raw USD and its log are not treated as independent objectives. Realized outcomes use positive incremental exports and the same frozen fold cap.

Normalization is fold-specific and estimated from outcomes no later than (y-2). The preference intervals are hypothetical, prespecified sensitivity assumptions, not elicited preferences: (W_{narrow}:w_1\in[.40,.60]), (W_{primary}:w_1\in[.25,.75]), (W_{broad}:w_1\in[0,1]).

## Predictive scenario construction

For Task A, candidate-specific entry probabilities come from saved, audited two-year-lag rolling-origin classifiers: historical-entry frequency, logistic regression, random forest, LightGBM, XGBoost, MLP and TabM. Conditional export-scale scenarios use the training-only q10/q25/q40/q50/q60/q75/q90 amounts among verified entry successes, with within-HS2 × historical-import-demand bin estimates when at least eight observations exist, then chapter and global fallback. Primary scenarios cross the seven probability models with q25/q50/q75 (21 scenarios); narrow scenarios use LightGBM/XGBoost crossed with q40/q60; stress scenarios use all seven models crossed with q10/q50/q90.

For Task B, training-only logistic and beta-smoothed HS2 × lagged-share-cell event rates form objective-1 alternatives. Saved audited forecasts from PPML, RF, XGBoost, LightGBM, MLP and TabM form objective-2 alternatives. Primary scenarios cross the two probability scores and six amount predictions. Narrow scenarios use PPML, LightGBM and XGBoost; stress scenarios add the candidatewise lower/upper envelope. Each scenario has equal status in worst-case optimization; no probability mass is assigned. Outer outcomes do not select or calibrate these scenarios. Scenario envelope coverage is descriptive post hoc assessment only.

This is a model-disagreement/stress set, not a prediction interval. It does not guarantee coverage. Independently crossing probability and amount forecasts can produce combinations not estimated as a joint distribution; results must be interpreted conditionally on that finite set. Scenario refinement cannot be informed by the test years and is required before confirmatory future use.

## Policies

- **Equal-weight MCDM:** maximize (U(S,.5,\bar\omega)).
- **Borda:** aggregate the two central objective ranks.
- **Maximin preference:** maximize minimum utility over preference endpoints, using the central forecast.
- **Standard preference-only minimax regret:** minimize worst regret over preference endpoints, using central forecasts.
- **Weber relative robustness:** maximize the worst preference-specific utility ratio to that preference's best feasible portfolio; central forecasts only. With two objectives, the candidate portfolio numerator is affine in (w_1), while the optimal portfolio denominator is a positive convex upper envelope of affine functions. The denominator lies below the chord between its endpoint values, so the ratio's minimum is attained at an interval endpoint. This endpoint reduction is valid only with nonnegative utilities and positive oracle utility throughout the interval.
- **Predictive maximin:** maximize minimum utility over predictive scenarios at (w_1=.5).
- **PR-EMS:** the implemented candidate policy is standard absolute minimax regret over the Cartesian product of (W_{primary})'s two endpoints and finite predictive scenarios.
- **Uniform-scenario expected utility:** descriptive mean across the enumerated scenario and preference grid. It is not expected utility under a fitted probability law.
- **Point forecast baselines:** PPML, LightGBM, LambdaMART and other saved model-score rankings, selected at K on the same full ex-ante pool.

For finite combined scenarios (s\), let (u_{is}\) be normalized candidate utility and (mathrm{OPT}_s=\max_{T\subseteq C,|T|=K}\sum_{i\in T}u_{is}\), obtained by sorting scenario utilities. The minimax-regret problem is:

\[
\min_{x,t}\ t\quad\text{s.t. }t+\sum_i u_{is}x_i\ge\mathrm{OPT}_s\ (s\in\mathcal S),\quad\sum_i x_i=K,\quad x_i\in\{0,1\},\quad t\ge0.
\]

This is an exact MILP for the stated finite set when HiGHS returns optimal status with accepted gap. It certifies the best portfolio for the chosen scenarios and utility scales only, not future outcomes outside the set. The finite candidate set and fixed cardinality make each scenario oracle a top-K sum. Regret is nonnegative by the matched oracle definition.

## Missing-label identification

For observed outcomes, utilities are calculated only where (R_i=1). Unknown labels remain NA. Under a bounded (u_i\in[0,1]) support relaxation, the fixed-portfolio utility lies in ([U_{observed},U_{observed}+m_S]), where (m_S) is the number of selected candidates without observed labels. The upper regret bound maximizes oracle-minus-policy utility under a common completion. Paired policy utility-difference bounds use the same missing outcome for both policies. These bounds are sharp for the rectangular ([0,1]) support relaxation; Task A's joint binary-entry/conditional-size support may be narrower, so those generic bounds are conservative for that task. No finite upper bound is asserted for uncapped USD exports.

## Tractability and numerical verification

SciPy `milp` with bundled HiGHS solves the finite-scenario MILPs. The implementation retries with supported presolve-on and presolve-off settings; it rejects a returned solution unless row feasibility residual is at most (2\times10^{-6}), binary integrality residual at most (10^{-7}), and the **absolute primal–dual objective gap** at most (10^{-6}) in the solver objective's units (normalized utility for absolute regret/maximin; dimensionless relative performance for the relative-robust comparator). Both relative and absolute gaps and the maximum row residual are retained in diagnostics. A relative gap can appear large when the optimum is close to zero, even if the absolute utility gap is negligible; accepted nonzero gaps/residuals are numerical tolerances, not exact feasibility/optimality claims. An earlier engineering run forwarded an unsupported feasibility-tolerance option, which led HiGHS to report status “Optimal” alongside larger absolute gaps. Those intermediate outputs were archived and are excluded from corrected results. Corrected outputs are regenerated under these gates and small instances are compared with exhaustive enumeration. A timeout or residual/gap above tolerance is not accepted as a completed solve.

## Mathematical properties used (not claimed as new)

- For a fixed portfolio, each preference-specific regret is the best feasible weighted utility minus that portfolio's weighted utility. It is a maximum of finitely many affine functions of the weight, hence convex; its maximum over a closed one-dimensional weight interval is attained at an endpoint. Checking the two endpoints is therefore sufficient for the stated linear two-objective formulation.
- If a preference interval or predictive scenario set expands while the candidate set, utility values and cardinality remain fixed, the feasible portfolio set is unchanged and the maximum regret for every portfolio cannot decrease. Consequently, the optimal minimax regret is weakly nondecreasing. The selected portfolio itself need not be nested or stable.
- If the preference set is a singleton and predictive uncertainty is a singleton, minimizing regret selects a top-K optimum and obtains zero regret (up to ties and solver tolerance).
- Regret is nonnegative because every portfolio is compared against an oracle optimized over the exact same feasible sets and scenario utility.
- The finite-scenario MILP's solver certificate is conditional on the enumerated scenarios and normalized utilities. Expanding the ambiguity sets can increase worst-case regret; this is not a guarantee about unmodeled future trade outcomes.

These are standard finite robust-optimization properties. Unit tests exercise endpoint, monotonicity and singleton cases; exhaustive enumeration validates the MILP on small instances.

## What this formulation does not claim

The policy is not a new minimax-regret method. Weights are illustrative, not estimated from firms or policymakers. Trade data do not reveal promotion costs, exporter capacity, tariffs for every historical year, or firm-level profitability. Backtested trade changes are not caused by the recommendations. The 2022–2024 holdout has already influenced prior M1 research; all new results are exploratory.
