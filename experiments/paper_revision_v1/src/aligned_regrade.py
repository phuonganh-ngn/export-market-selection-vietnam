"""Re-run aligned expansion baselines under the new grading, one outcome year.
D_* scores do not depend on the expansion grading and are reused. E_rank is always refit
(its label is the grade). For other E_* specs tuning is re-run with the new validation gain;
the old score is reused only if the chosen parameters are unchanged."""
import sys, os, json
import numpy as np, pandas as pd
import aligned_baselines as ab
from revision_lib import *  # noqa
ab.BASE["n_jobs"] = int(os.environ.get("NJ", "4"))
year = int(sys.argv[1])
OLD = "/home/hoangpd/predictions_grade450k/aligned_baseline_scores.parquet"
old = pd.read_parquet(OLD); old = old[old.outcome_year == year]
oldt = pd.read_csv("/home/hoangpd/results_grade450k/aligned/tuning.csv"); oldt = oldt[oldt.year == year].set_index("model")
panel, _ = ab.load_and_freeze(write=False); panel = ab.zfill_hs2(panel)
train = panel.loc[panel.outcome_observed.astype(bool) & panel.export_value_usd.notna() & panel.destination_import_demand.gt(0) & panel.outcome_year.le(year - 2)]
test = panel.loc[panel.outcome_year.eq(year) & panel.destination_import_demand.gt(0)]
cand = {ab.TA: test.loc[test.entry_candidate_ex_ante.astype(bool) & test.strict_inactive_verified.astype(bool)], ab.TB: test.loc[test.task_b_eligible.astype(bool)]}
rows, tuned = [], []
for spec in ab.specs():
    if spec.task == ab.TA:
        rows.append(old[old.model == spec.name]); r = oldt.loc[spec.name]
        tuned.append({"year": year, "model": spec.name, "num_leaves": int(r.num_leaves), "n_estimators": int(r.n_estimators), "inner_ndcg10": r.inner_ndcg10, "refit": False})
        continue
    p, v = ab.tune(spec, train)
    same = (int(oldt.loc[spec.name].num_leaves), int(oldt.loc[spec.name].n_estimators)) == (p["num_leaves"], p["n_estimators"])
    refit = spec.name == "E_rank" or not same
    if refit:
        sc = ab.fit_predict(spec, train, cand[spec.task], p, ab.SEEDS)
        rows.append(cand[spec.task][["destination_iso3", "hs2", "outcome_year"]].assign(task=spec.task, model=spec.name, score=np.asarray(sc, float)))
    else:
        rows.append(old[old.model == spec.name])
    tuned.append({"year": year, "model": spec.name, **p, "inner_ndcg10": v, "refit": refit})
    ab.log(f"{year} {spec.name}: {p} inner {v:.3f} refit={refit}")
out = "/home/hoangpd/regrade_aligned"; os.makedirs(out, exist_ok=True)
pd.concat(rows, ignore_index=True).to_parquet(f"{out}/scores_{year}.parquet", index=False)
pd.DataFrame(tuned).to_csv(f"{out}/tuning_{year}.csv", index=False)
print("done", year)
