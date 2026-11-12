"""Vietnam-specific heterogeneity, 2018-2024 (bagged LightGBM, PPML, hybrid of the main analysis).

1. Free trade agreements, time-varying: a destination is an FTA partner in outcome year y if an agreement with
   Vietnam was in force by 1 January of y. "New-generation" partners are those first covered by CPTPP (2019),
   EVFTA (2020) or UKVFTA (2021) and by no earlier agreement. Pools are re-ranked within each subset.
2. Product groups (HS sections grouped into five blocks).
3. Diversification: share of Top-10 slots and of realised successes outside Vietnam's five main export markets
   (United States, China, EU-27, Japan, Korea) and outside existing FTA partners.
Outputs in results/vietnam/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from revision_lib import KEY, PRED_DIR, RES_DIR, TA, TB, add_rules_and_hybrid, load_stored_candidates, pool_metrics, zfill_hs2
import statistics as st

OUT = RES_DIR / "vietnam"; OUT.mkdir(parents=True, exist_ok=True)
EU27 = ["AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU",
        "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"]
ASEAN = ["BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA"]
# first year in force for Vietnam (bilateral or regional agreement covering the partner)
FTA_START = {**{c: 1999 for c in ASEAN}, "CHN": 2005, "KOR": 2007, "JPN": 2008, "IND": 2010, "AUS": 2010, "NZL": 2010,
             "CHL": 2014, "RUS": 2016, "BLR": 2016, "KAZ": 2016, "KGZ": 2017, "ARM": 2016, "HKG": 2019, "CAN": 2019,
             "MEX": 2019, "PER": 2021, **{c: 2020 for c in EU27}, "GBR": 2021, "ISR": 2024}
NEW_GEN = {"CAN", "MEX", "PER", "GBR", *EU27}
MAIN5 = {"USA", "CHN", "JPN", "KOR", *EU27}
GROUPS = [("Agri-food (HS 01--24)", 1, 24), ("Minerals, chemicals, plastics (25--40)", 25, 40),
          ("Wood, paper, textiles, footwear (41--67)", 41, 67), ("Stone, glass, metals (68--83)", 68, 83),
          ("Machinery, electronics, transport (84--97)", 84, 97)]
CONTRAST = {TA: [("Hybrid", "LightGBM"), ("Hybrid", "PPML")], TB: [("PPML", "Hybrid"), ("PPML", "LightGBM")]}
COLS = {"PPML": "ppml_score", "LightGBM": "lightgbm_score", "Hybrid": "hybrid_score", "ImportDemand": "import_demand_score"}


def load():
    d = load_stored_candidates().drop(columns="lightgbm_score")
    refit = zfill_hs2(pd.read_parquet(PRED_DIR / "lgbm_refit_scores.parquet"))[KEY + ["lightgbm_score"]]
    d = d.merge(refit, on=KEY, how="left", validate="one_to_one")
    d["fta"] = [FTA_START.get(i, 9999) <= y for i, y in zip(d.destination_iso3, d.outcome_year)]
    d["new_gen"] = d.destination_iso3.isin(NEW_GEN) & d.fta
    d["main5"] = d.destination_iso3.isin(MAIN5)
    h = d.hs2.astype(int)
    d["pgroup"] = np.select([(h >= a) & (h <= b) for _, a, b in GROUPS], [g for g, _, _ in GROUPS], "other")
    return d


def pools(frame, task):
    """Pool-level NDCG@10 for every model after recomputing the hybrid on this candidate set."""
    f = add_rules_and_hybrid(frame)
    rows = []
    for (hs2, yr), g in f.groupby(["hs2", "outcome_year"]):
        rows.append({"hs2": hs2, "year": yr, **{m: pool_metrics(g, c, task, 10)["ndcg"] for m, c in COLS.items()}})
    return pd.DataFrame(rows).dropna(subset=["Hybrid"])


def contrasts(w, task, label):
    out = []
    for a, b in CONTRAST[task]:
        r = st.paired_contrast_inference((w[a] - w[b]).to_numpy(float), w.hs2.to_numpy(), w.year.to_numpy(), reps=2000)
        out.append({"task": task, "subset": label, "contrast": f"{a} - {b}", "pools": len(w), "mean_a": w[a].mean(), "mean_b": w[b].mean(),
                    **{k: r[k] for k in ("estimate", "ci_low", "ci_high", "p_signflip", "years_positive", "n_years")}})
    return out


def main():
    d = load()
    res = []
    for task in (TA, TB):
        t = d[d.task == task]
        res += contrasts(pools(t, task), task, "All destinations")
        for lab, m in (("FTA partners (time-varying)", t.fta), ("Non-FTA destinations", ~t.fta), ("New-generation FTA partners", t.new_gen),
                       ("Outside the five main markets", ~t.main5)):
            res += contrasts(pools(t[m], task), task, lab)
        w = pools(t, task).merge(t.groupby(["hs2", "outcome_year"]).pgroup.first().rename("pgroup").reset_index().rename(columns={"outcome_year": "year"}), on=["hs2", "year"])
        for g, _, _ in GROUPS:
            res += contrasts(w[w.pgroup == g], task, g)
    r = pd.DataFrame(res); r.to_csv(OUT / "subgroup_contrasts.csv", index=False)

    # descriptive: candidates, successes and success rates by FTA status, discovery
    a = d[(d.task == TA) & d.label_observed.astype(bool)].copy()
    desc = a.groupby(["fta", "new_gen"]).agg(candidates=("entry_success", "size"), successes=("entry_success", "sum"))
    desc["rate"] = desc.successes / desc.candidates
    desc.to_csv(OUT / "fta_descriptives.csv")
    ng = a[a.destination_iso3.isin(NEW_GEN)].assign(after=lambda x: x.fta)
    pre_post = ng.groupby("after").agg(candidates=("entry_success", "size"), successes=("entry_success", "sum"))
    pre_post["rate"] = pre_post.successes / pre_post.candidates
    pre_post.to_csv(OUT / "new_gen_pre_post.csv")

    # diversification of the Top-10 lists
    f = add_rules_and_hybrid(d[d.label_observed.astype(bool)])
    div = []
    for task in (TA, TB):
        t = f[f.task == task]
        y = t.entry_success.astype(float) if task == TA else (np.maximum(0, t.realized_incremental_export_usd.astype(float)) >= 50_000).astype(float)
        t = t.assign(y=y)
        for m, c in COLS.items():
            top = t.sort_values([c, "destination_iso3"], ascending=[False, True], kind="mergesort").groupby(["hs2", "outcome_year"]).head(10)
            div.append({"task": task, "model": m, "slots": len(top), "share_outside_main5": (~top.main5).mean(), "share_non_fta": (~top.fta).mean(),
                        "share_new_gen": top.new_gen.mean(), "successes": top.y.sum(),
                        "successes_outside_main5": top.y[~top.main5].sum(), "successes_non_fta": top.y[~top.fta].sum(),
                        "successes_new_gen": top.y[top.new_gen].sum()})
        allc = t
        div.append({"task": task, "model": "All candidates", "slots": len(allc), "share_outside_main5": (~allc.main5).mean(),
                    "share_non_fta": (~allc.fta).mean(), "share_new_gen": allc.new_gen.mean(), "successes": allc.y.sum(),
                    "successes_outside_main5": allc.y[~allc.main5].sum(), "successes_non_fta": allc.y[~allc.fta].sum(),
                    "successes_new_gen": allc.y[allc.new_gen].sum()})
    dv = pd.DataFrame(div); dv.to_csv(OUT / "diversification.csv", index=False)
    with pd.option_context("display.width", 230, "display.max_columns", 20, "display.float_format", lambda x: f"{x:,.3f}"):
        print(r.assign(task=r.task.str[:6]).to_string(index=False)); print(desc); print(pre_post); print(dv.assign(task=dv.task.str[:6]).to_string(index=False))


if __name__ == "__main__":
    main()
