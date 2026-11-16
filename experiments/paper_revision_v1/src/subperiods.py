"""Sensitivity of the main contrasts to the length of the classifier's training history.

Repeats the paired NDCG@10 contrasts for windows that start in 2018, 2019, 2020, 2021 and 2022 and end in 2024.
The discovery classifiers for 2018 and 2019 are trained on one and two outcome years, so the later windows show
how much of the full-period gain depends on those two years.

Input:  results/aligned/pool_ndcg10.csv (written by aligned_baselines.py)
Output: results/robustness/subperiods.csv
"""
import numpy as np
import pandas as pd

import statistics as st
from revision_lib import RES_DIR, TA, TB

START_YEARS = [2018, 2019, 2020, 2021, 2022]
CONTRASTS = {
    TA: [("Hybrid", "LightGBM"), ("Hybrid", "D_rank"), ("Hybrid", "PPML"), ("Hybrid", "ImportDemand")],
    TB: [("PPML", "Hybrid"), ("PPML", "LightGBM"), ("PPML", "E_pois"), ("PPML", "ImportDemand")],
}
KEEP = ("estimate", "ci_low", "ci_high", "p_signflip", "t_twoway", "years_positive", "n_years", "p_year_signflip")
LEVEL_COLUMNS = ["PPML", "LightGBM", "Hybrid", "D_rank", "E_pois", "ImportDemand"]


def window_contrasts(pools: pd.DataFrame, start: int) -> list[dict]:
    """Contrasts of one window, with Holm adjustment within each task."""
    rows = []
    for task, pairs in CONTRASTS.items():
        q = pools[(pools.task == task) & (pools.year >= start)].dropna(subset=["Hybrid", "PPML"])
        family = []
        for a, b in pairs:
            diff = (q[a] - q[b]).to_numpy(float)
            ok = ~np.isnan(diff)
            res = st.paired_contrast_inference(diff[ok], q.hs2.to_numpy()[ok], q.year.to_numpy()[ok], reps=10000)
            family.append({"window": f"{start}-2024", "task": task[:6], "contrast": f"{a} - {b}", "pools": int(ok.sum()),
                           **{k: res[k] for k in KEEP}})
        for row, p in zip(family, st.holm_adjust([r["p_signflip"] for r in family])):
            row["p_holm"] = float(p)
        for row, p in zip(family, st.holm_adjust([r["p_year_signflip"] for r in family])):
            row["p_year_holm"] = float(p)
        rows += family
    return rows


def main() -> None:
    pools = pd.read_csv(RES_DIR / "aligned" / "pool_ndcg10.csv")
    rows = [row for start in START_YEARS for row in window_contrasts(pools, start)]
    out = pd.DataFrame(rows)
    out.to_csv(RES_DIR / "robustness" / "subperiods.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))
    for start in (2018, 2020, 2021, 2022):
        print(start, pools[pools.year >= start].groupby("task")[LEVEL_COLUMNS].mean().round(3).to_string())


if __name__ == "__main__":
    main()
