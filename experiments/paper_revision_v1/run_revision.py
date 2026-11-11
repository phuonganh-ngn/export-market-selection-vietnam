"""Run the paper-revision pipeline end to end.

    cd <project root>
    PYTHONPATH=src python experiments/paper_revision_v1/run_revision.py            # all steps
    PYTHONPATH=src python experiments/paper_revision_v1/run_revision.py --skip-refit

Steps: re-fit LightGBM (2018-2024, configured seeds and bagging) -> evaluate -> LambdaMART check -> LaTeX tables.
Set M1_CONFIG=config/legacy_archived.yaml to run with the archived (deterministic) LightGBM settings.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import evaluate  # noqa: E402
import lambdamart_check  # noqa: E402
import make_tables  # noqa: E402
import refit_lightgbm  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-refit", action="store_true", help="reuse predictions/lgbm_refit_scores.parquet")
    ap.add_argument("--reps", type=int, default=10000, help="bootstrap / sign-flip replications")
    args = ap.parse_args()
    if not args.skip_refit:
        refit_lightgbm.main()
    evaluate.main(reps=args.reps)
    lambdamart_check.main(reps=args.reps)
    make_tables.main()
