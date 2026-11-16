import glob, pandas as pd, aligned_baselines as ab
o = "/home/hoangpd/regrade_aligned"
s = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(o + "/scores_*.parquet"))], ignore_index=True)
s.to_parquet(ab.PRED_DIR / "aligned_baseline_scores.parquet", index=False)
pd.concat([pd.read_csv(f) for f in sorted(glob.glob(o + "/tuning_*.csv"))]).to_csv(ab.OUT / "tuning.csv", index=False)
ab.evaluate(s)
