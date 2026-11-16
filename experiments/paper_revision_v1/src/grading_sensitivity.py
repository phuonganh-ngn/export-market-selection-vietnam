"""Expansion NDCG@10 under alternative relevance gradings (audit of the graded-gain threshold)."""
import numpy as np, pandas as pd, statistics as st
import fusion
from revision_lib import CFG, KEY, PRED_DIR, RES_DIR, TB, add_rules_and_hybrid, load_stored_candidates, zfill_hs2
d = load_stored_candidates().drop(columns="lightgbm_score")
refit = zfill_hs2(pd.read_parquet(PRED_DIR / "lgbm_refit_scores.parquet"))[KEY + ["lightgbm_score"]]
d = add_rules_and_hybrid(d.merge(refit, on=KEY, how="left", validate="one_to_one"))
sc = pd.read_parquet(PRED_DIR / "aligned_baseline_scores.parquet").pivot_table(index=KEY, columns="model", values="score").reset_index()
d = d.merge(sc, on=KEY, how="left", validate="one_to_one")
d = d[d.task == TB].copy()
sub = d[d.E_pois.notna()]
fz = fusion.fuse(sub, ["task", "hs2", "outcome_year"], ["ppml_score", "E_pois"], tie_rule=CFG["study"].get("fusion_tie_rule", "iso3"))
d.loc[sub.index, "H_E_pois"] = -fz.loc[sub.index, "fused_rank"].to_numpy()
d = d[d.label_observed.astype(bool)].reset_index(drop=True)
inc = np.maximum(0, pd.to_numeric(d.realized_incremental_export_usd, errors="coerce").fillna(0).to_numpy(float))
T = 50_000
r_paper = np.minimum(4, np.floor(np.log10(1 + inc / T)))
r_fix = np.where(inc >= T, np.minimum(4, 1 + np.floor(np.log10(np.maximum(inc, T) / T))), 0)
G = {"paper": 2 ** r_paper - 1, "aligned": 2 ** r_fix - 1, "binary": (inc >= T).astype(float), "logcont": np.log1p(inc / T)}
print("share inc>=50k", (inc >= T).mean(), " share paper grade>0", (r_paper > 0).mean(), " min inc with paper grade>0", inc[r_paper > 0].min())
models = {"PPML": "ppml_score", "Hybrid": "hybrid_score", "LightGBM": "lightgbm_score", "ImportDemand": "import_demand_score",
          "E_pois": "E_pois", "E_pinc": "E_pinc", "E_rank": "E_rank", "H_E_pois": "H_E_pois"}
def ndcg(y, s, iso, k=10):
    o = np.lexsort((iso, -s))[:k]; disc = 1 / np.log2(np.arange(2, len(o) + 2))
    ideal = (np.sort(y)[::-1][:len(o)] * disc).sum()
    return (y[o] * disc).sum() / ideal if ideal > 0 else np.nan
rows = []
for (hs2, yr), g in d.groupby(["hs2", "outcome_year"]):
    idx = g.index.to_numpy(); iso = g.destination_iso3.to_numpy()
    for gn, gv in G.items():
        rec = {"grading": gn, "hs2": hs2, "year": yr}
        for m, c in models.items():
            s = g[c].to_numpy(float)
            rec[m] = ndcg(gv[idx], s, iso) if not np.isnan(s).any() else np.nan
        rows.append(rec)
w = pd.DataFrame(rows)
OUT = RES_DIR / "robustness"; w.to_csv(OUT / "expansion_grading_pool_ndcg.csv", index=False)
lev = w.groupby("grading")[list(models)].mean().round(3); pools = w.dropna(subset=["PPML"]).groupby("grading").size()
print(lev.assign(pools=pools).to_string())
pairs = [("PPML", "Hybrid"), ("PPML", "LightGBM"), ("PPML", "ImportDemand"), ("PPML", "E_pois"), ("PPML", "E_pinc"), ("PPML", "E_rank"), ("H_E_pois", "PPML")]
res = []
for gn in G:
    q = w[(w.grading == gn)].dropna(subset=["PPML"]); fam = []
    for a, b in pairs:
        x = (q[a] - q[b]).to_numpy(float); ok = ~np.isnan(x)
        r = st.paired_contrast_inference(x[ok], q.hs2.to_numpy()[ok], q.year.to_numpy()[ok], reps=10000)
        fam.append({"grading": gn, "contrast": f"{a} - {b}", "pools": int(ok.sum()), **{k: r[k] for k in ("estimate", "ci_low", "ci_high", "p_signflip", "years_positive", "n_years", "p_year_signflip")}})
    for f, h in zip(fam[:3], st.holm_adjust([f["p_signflip"] for f in fam[:3]])): f["p_holm"] = float(h)
    for f, h in zip(fam[:3], st.holm_adjust([f["p_year_signflip"] for f in fam[:3]])): f["p_year_holm"] = float(h)
    res += fam
r = pd.DataFrame(res); r.to_csv(OUT / "expansion_grading_contrasts.csv", index=False)
pd.set_option("display.width", 250); print(r.round(4).to_string(index=False))
