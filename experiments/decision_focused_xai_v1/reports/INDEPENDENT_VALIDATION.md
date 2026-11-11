# Independent saved-output validation

Validation is recomputed from the stored prediction files, rather than the report tables alone.

- PASS — All models use identical frozen ex-ante candidates within task
- PASS — All five prespecified LambdaMART seeds are saved
- PASS — LambdaMART raw predictions contain no baseline rows
- PASS — Final rankings preserve every raw LambdaMART score exactly
- PASS — All 30 fitted models satisfy the audited y-2 cutoffs
- PASS — Task A unknown labels remain NA
- PASS — Independent NDCG@10 recomputation matches four saved pools
- PASS — All observed-pool regret values are nonnegative
- PASS — All saved candidate ranking scores are finite

Maximum independently recalculated NDCG@10 absolute error: 1.110e-16.
Final ranked prediction rows: 824,357; raw LambdaMART rows: 270,465; frozen manifest rows: 54,093.
