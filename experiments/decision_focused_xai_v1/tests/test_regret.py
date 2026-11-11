import pandas as pd

from experiments.decision_focused_xai_v1.src.decision_regret import compute_regret


def test_oracle_regret_is_nonnegative_and_zero_for_oracle_order():
    rows = []
    for i, (dest, gain) in enumerate([("AAA", 100), ("BBB", 50), ("CCC", 0)]):
        rows.append({"task": "Task_B_existing_underexploited", "model": "LambdaMART", "seed": -2,
                     "hs2": "01", "outcome_year": 2022, "label_observed": True,
                     "destination_iso3": dest, "score": gain, "realized_incremental_export_usd": gain,
                     "entry_success": float("nan")})
    out = compute_regret(pd.DataFrame(rows))
    assert (out.regret >= 0).all()
    assert out.loc[out.k.eq(5), "regret"].iloc[0] == 0
