# Final status: COMPLETE_REDUCED_SPECIFICATION

Primary population: 62,797 ex-ante candidates; 62,238 observed labels; 559 missing labels.
Neural final outer fit files: E2=30/30; E1=30/30.
Primary E2/E3/E1 metrics are complete-case and use the same observed-label candidate intersections. See saved prediction files and run logs.
Feature ablation is exploratory (2024, seed 11, F4-selected settings held fixed). The one-year-lag population is audited but its model performance is not run.
Test verification: Verified test execution:
- Post-audit extension suite: 18 passed.
- Original project suite (`PYTHONPATH=src`): 12 passed.
- `py_compile` passed for all extension source and test files.

Outstanding limitations: revised macro-data vintages; unavailable-versus-true-zero ambiguity for missing BACI outcomes; three outer test years; one-year sensitivity not executed; feature ablation limited to one year/seed.
