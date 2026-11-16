import pandas as pd, numpy as np, statistics as st
R="experiments/paper_revision_v1/results/"
w=pd.read_csv(R+"aligned/pool_ndcg10.csv")
TA,TB=sorted(w.task.unique())
fams={TA:[("Hybrid","LightGBM"),("Hybrid","D_rank"),("Hybrid","PPML"),("Hybrid","ImportDemand")],
      TB:[("PPML","Hybrid"),("PPML","LightGBM"),("PPML","E_pois"),("PPML","ImportDemand")]}
res=[]
for start in [2018,2019,2020,2021,2022]:
    for task,pairs in fams.items():
        q=w[(w.task==task)&(w.year>=start)].dropna(subset=["Hybrid","PPML"])
        fam=[]
        for a,b in pairs:
            x=(q[a]-q[b]).to_numpy(float); ok=~np.isnan(x)
            r=st.paired_contrast_inference(x[ok],q.hs2.to_numpy()[ok],q.year.to_numpy()[ok],reps=10000)
            fam.append({"window":f"{start}-2024","task":task[:6],"contrast":f"{a} - {b}","pools":int(ok.sum()),
              **{k:r[k] for k in ("estimate","ci_low","ci_high","p_signflip","t_twoway","years_positive","n_years","p_year_signflip")}})
        for f,h in zip(fam,st.holm_adjust([f["p_signflip"] for f in fam])): f["p_holm"]=float(h)
        for f,h in zip(fam,st.holm_adjust([f["p_year_signflip"] for f in fam])): f["p_year_holm"]=float(h)
        res+=fam
r=pd.DataFrame(res); r.to_csv(R+"robustness/subperiods.csv",index=False)
pd.set_option("display.width",250)
print(r.round(4).to_string(index=False))
# levels
for start in [2018,2020,2021,2022]:
    q=w[w.year>=start]; print(start, q.groupby("task")[["PPML","LightGBM","Hybrid","D_rank","E_pois","ImportDemand"]].mean().round(3).to_string())
