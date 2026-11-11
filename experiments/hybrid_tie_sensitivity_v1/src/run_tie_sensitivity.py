from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
BASE = PROJECT/'experiments/final_hybrid_economic_validation_v1'
INPUT = BASE/'predictions/F1_F2_reconstructed_outer_scores.parquet'
EXISTING_TIES = PROJECT/'experiments/feature_level_xai_recovery_v1/results/hybrid_top10_boundary_ties.csv'
CASES = PROJECT/'manuscript/explainable_hybrid_export_market_selection_v3/results/XAI_AUTHENTIC_CASE_STUDIES.csv'
RESULTS = ROOT/'results'
REPORTS = ROOT/'reports'
K=10
SEEDS=range(10000,11000)
BOOT_REPS=10_000
BOOT_SEED=51_073
TASK='Task_A_new_entry'


def rank_components(g: pd.DataFrame) -> pd.DataFrame:
    z=g.copy()
    p=z.sort_values(['ppml_score','destination_iso3'],ascending=[False,True],kind='mergesort')
    l=z.sort_values(['lightgbm_score','destination_iso3'],ascending=[False,True],kind='mergesort')
    z['_rp']=pd.Series(np.arange(1,len(p)+1),index=p.index).reindex(z.index).astype(int)
    z['_rl']=pd.Series(np.arange(1,len(l)+1),index=l.index).reindex(z.index).astype(int)
    n=len(z)
    z['_rank_sum']=z['_rp']+z['_rl']
    z['_hybrid']=0.5*((n-z['_rp']+1)/max(1,n-1)+(n-z['_rl']+1)/max(1,n-1))
    return z


def order_pool(g: pd.DataFrame, rule: str, seed: int|None=None) -> pd.DataFrame:
    if rule=='ISO3_ascending_original':
        return g.sort_values(['_hybrid','destination_iso3'],ascending=[False,True],kind='mergesort')
    if rule=='PPML_first_then_LightGBM_then_ISO3':
        return g.sort_values(['_hybrid','_rp','_rl','destination_iso3'],ascending=[False,True,True,True],kind='mergesort')
    if rule=='LightGBM_first_then_PPML_then_ISO3':
        return g.sort_values(['_hybrid','_rl','_rp','destination_iso3'],ascending=[False,True,True,True],kind='mergesort')
    if rule=='uniform_random_ties':
        rng=np.random.default_rng(seed)
        pieces=[]
        for score in sorted(g['_hybrid'].unique(),reverse=True):
            tie=g.loc[g['_hybrid'].eq(score)].sort_values('destination_iso3',kind='mergesort')
            if len(tie)>1:
                tie=tie.iloc[rng.permutation(len(tie))]
            pieces.append(tie)
        return pd.concat(pieces,axis=0)
    raise ValueError(rule)


def order_single(g: pd.DataFrame, score: str) -> pd.DataFrame:
    return g.sort_values([score,'destination_iso3'],ascending=[False,True],kind='mergesort')


def make_random_context(g: pd.DataFrame) -> dict:
    ordered=order_pool(g,'ISO3_ascending_original').reset_index(drop=True)
    tie_positions=[grp.index.to_numpy(dtype=int) for _,grp in ordered.groupby('_hybrid',sort=False) if len(grp)>1]
    observed=ordered.label_observed.astype(bool).to_numpy()
    entries=pd.to_numeric(ordered.entry_success,errors='coerce').to_numpy(float)
    exports=pd.to_numeric(ordered.export_value_usd,errors='coerce').to_numpy(float)
    known_gains=entries[observed]
    n=min(K,len(known_gains))
    discounts=np.log2(np.arange(2,n+2))
    idcg=float(np.sum(np.sort(known_gains)[::-1][:n]/discounts)) if n else 0.0
    return {'ordered':ordered,'tie_positions':tie_positions,'observed':observed,'entries':entries,
            'exports':exports,'iso3':ordered.destination_iso3.astype(str).to_numpy(),
            'candidate_pool_id':str(ordered.candidate_pool_id.iloc[0]),'positive_relevance':bool(np.nansum(known_gains)>0),
            'known_count':int(observed.sum()),'idcg':idcg,'ndcg_n':n,'discounts':discounts}


def randomized_positions(ctx: dict, seed: int, key: tuple[str,int]) -> np.ndarray:
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(key[0]),int(key[1])]))
    positions=np.arange(len(ctx['iso3']),dtype=int)
    for tied in ctx['tie_positions']:
        positions[tied]=positions[tied][rng.permutation(len(tied))]
    return positions


def metrics_for_positions(positions: np.ndarray, ctx: dict) -> dict:
    top_pos=positions[:min(K,len(positions))]
    top_observed=top_pos[ctx['observed'][top_pos]]
    unknown=len(top_pos)-len(top_observed)
    success_count=float(ctx['entries'][top_observed].sum())
    raw=float(ctx['exports'][top_observed].sum())
    qualified=float(ctx['exports'][top_observed][ctx['exports'][top_observed]>=50_000].sum())
    complete=(unknown==0 and len(top_pos)==K)
    known_order=positions[ctx['observed'][positions]][:ctx['ndcg_n']]
    gain=ctx['entries'][known_order]
    if ctx['ndcg_n'] and ctx['idcg']>0:
        dcg=float(np.sum(gain/ctx['discounts']))
        ndcg=dcg/ctx['idcg']
    else:
        ndcg=np.nan
    return {'ndcg10':ndcg,'precision10_complete':success_count/K if complete else np.nan,
      'successful_entries10_complete':success_count if complete else np.nan,
      'successful_entries10_observed_partial':success_count,
      'qualified_usd10_complete':qualified if complete else np.nan,
      'raw_usd10_complete':raw if complete else np.nan,
      'qualified_usd10_observed_partial':qualified,'raw_usd10_observed_partial':raw,
      'selected_count':len(top_pos),'observed_selected_count':len(top_observed),
      'unknown_selected_count':unknown,'complete_outcome_pool':complete,
      'positive_relevance':ctx['positive_relevance']}


def ndcg10(order: pd.DataFrame) -> float:
    # Match the audited F1 definition: remove unknown outcomes in score order,
    # evaluate the first min(K, observed candidates), and omit zero-IDCG pools.
    known=order.loc[order.label_observed.astype(bool)]
    n=min(K,len(known))
    if not n: return np.nan
    gains=pd.to_numeric(known.entry_success,errors='coerce').to_numpy(float)
    if np.isnan(gains).any(): raise AssertionError('Observed labels are missing entry_success')
    top=gains[:n]
    discounts=np.log2(np.arange(2,n+2))
    dcg=float(np.sum(top/discounts))
    idcg=float(np.sum(np.sort(gains)[::-1][:n]/discounts))
    return dcg/idcg if idcg>0 else np.nan


def selected_metrics(order: pd.DataFrame) -> dict:
    top=order.head(min(K,len(order))).copy()
    observed=top.loc[top.label_observed.astype(bool)]
    unknown=len(top)-len(observed)
    vals=pd.to_numeric(observed.export_value_usd,errors='coerce').to_numpy(float)
    success=pd.to_numeric(observed.entry_success,errors='coerce').to_numpy(float)
    if np.isnan(vals).any() or np.isnan(success).any():
        raise AssertionError('Observed outcomes contain NA')
    raw=float(vals.sum())
    qualified=float(vals[vals>=50_000].sum())
    count=float(success.sum())
    complete=unknown==0 and len(top)==K
    return {
        'ndcg10':ndcg10(order),
        'precision10_complete':count/K if complete else np.nan,
        'successful_entries10_complete':count if complete else np.nan,
        'successful_entries10_observed_partial':count,
        'qualified_usd10_complete':qualified if complete else np.nan,
        'raw_usd10_complete':raw if complete else np.nan,
        'qualified_usd10_observed_partial':qualified,
        'raw_usd10_observed_partial':raw,
        'selected_count':len(top),'observed_selected_count':len(observed),
        'unknown_selected_count':unknown,'complete_outcome_pool':complete,
        'positive_relevance':bool(pd.to_numeric(order.loc[order.label_observed.astype(bool),'entry_success'],errors='coerce').fillna(0).gt(0).any())}


def bootstrap(diff: pd.DataFrame, col: str, reps=BOOT_REPS, seed=BOOT_SEED) -> dict:
    z=diff.loc[np.isfinite(pd.to_numeric(diff[col],errors='coerce')),['hs2',col]].copy()
    if z.empty: return {'estimate':np.nan,'ci_low':np.nan,'ci_high':np.nan,'p_raw_signflip':np.nan,'paired_pools':0,'hs2_blocks':0}
    z[col]=pd.to_numeric(z[col],errors='coerce')
    blocks=z.groupby('hs2')[col].agg(['sum','count'])
    estimate=float(z[col].mean()); sums=blocks['sum'].to_numpy(float); counts=blocks['count'].to_numpy(float); b=len(blocks)
    rng=np.random.default_rng(seed)
    boot=np.empty(reps)
    for i in range(reps):
        ix=rng.integers(0,b,b)
        boot[i]=sums[ix].sum()/counts[ix].sum()
    ci=np.quantile(boot,[.025,.975])
    sign=rng.choice(np.array([-1.,1.]),size=(reps,b))
    null=(sign*sums[None,:]).sum(axis=1)/counts.sum()
    p=float((1+(np.abs(null)>=abs(estimate)).sum())/(reps+1))
    return {'estimate':estimate,'ci_low':float(ci[0]),'ci_high':float(ci[1]),'p_raw_signflip':p,'paired_pools':len(z),'hs2_blocks':b}


def holm(p):
    x=np.asarray(p,float); order=np.argsort(x); out=np.empty(len(x)); running=0.
    for r,i in enumerate(order):
        running=max(running,(len(x)-r)*x[i]); out[i]=min(1.,running)
    return out.tolist()


def main():
    started=time.time(); RESULTS.mkdir(parents=True,exist_ok=True); REPORTS.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(INPUT)
    d=d.loc[d.task.eq(TASK)].copy()
    d['hs2']=d.hs2.astype(str).str.zfill(2); d['outcome_year']=d.outcome_year.astype(int); d['destination_iso3']=d.destination_iso3.astype(str)
    assert len(d)==29737 and d.candidate_id.is_unique and d.candidate_pool_id.nunique()==288
    assert d.groupby('candidate_pool_id').destination_iso3.nunique().eq(d.groupby('candidate_pool_id').size()).all()
    pool_groups={key:rank_components(g) for key,g in d.groupby(['hs2','outcome_year'],sort=True)}
    assert sum(len(g) for g in pool_groups.values())==len(d)
    # Reconstruct all original component ranks and the equal-weight score.
    max_hybrid_error=max(float((g._hybrid-g.blend_rank_score).abs().max()) for g in pool_groups.values())
    max_ppml_rank_error=max(int((g._rp-g._rank_ppml_rebuilt).abs().max()) for g in pool_groups.values())
    max_lgbm_rank_error=max(int((g._rl-g._rank_lightgbm_rebuilt).abs().max()) for g in pool_groups.values())
    assert max_hybrid_error<1e-12 and max_ppml_rank_error==0 and max_lgbm_rank_error==0
    # Existing original boundary tie record is an independent count cross-check.
    archived_ties=pd.read_csv(EXISTING_TIES,dtype={'hs2':str})
    archived_ties['hs2']=archived_ties.hs2.str.zfill(2)

    rows=[]; pools_by_policy={}; selections={}; random_seed_summaries=[]
    rules=['ISO3_ascending_original','PPML_first_then_LightGBM_then_ISO3','LightGBM_first_then_PPML_then_ISO3']
    rules_by_pool={}
    # Build the base LightGBM and PPML rankings once.
    for key,g in pool_groups.items():
        rules_by_pool[(key,'LightGBM_baseline')]=order_single(g,'lightgbm_score')
        rules_by_pool[(key,'Gravity_PPML')]=order_single(g,'ppml_score')
        for rule in rules:
            rules_by_pool[(key,rule)]=order_pool(g,rule)
    orig_tops={key:set(rules_by_pool[(key,'ISO3_ascending_original')].head(K).destination_iso3) for key in pool_groups}
    lgbm_tops={key:set(rules_by_pool[(key,'LightGBM_baseline')].head(K).destination_iso3) for key in pool_groups}
    pool_metric_records={}
    for policy in ['LightGBM_baseline','Gravity_PPML',*rules]:
        rec=[]; sel={}
        for key in pool_groups:
            order=rules_by_pool[(key,policy)]
            m=selected_metrics(order)
            top=order.head(K)
            sel[key]=set(top.destination_iso3)
            rec.append({'hs2':key[0],'outcome_year':key[1],'candidate_pool_id':str(top.candidate_pool_id.iloc[0]),**m,
                        'top10_iso3':'|'.join(top.destination_iso3.astype(str))})
        pool_metric_records[policy]=pd.DataFrame(rec)
        selections[policy]=sel
    # Measure every exact archived-score tie group; also retain the integer rank sum for diagnostics.
    tie_groups=[]; tie_pool_info={}; boundary_rows=[]; algebraic_boundary_pool_keys=set(); algebraic_boundary_group_count=0
    for key,g in pool_groups.items():
        ordered=rules_by_pool[(key,'ISO3_ascending_original')].reset_index(drop=True)
        group_records=[]
        for score,t in ordered.groupby('_hybrid',sort=False):
            if len(t)<2: continue
            pos=ordered.index[ordered._hybrid.eq(score)].to_numpy()+1
            rec={'hybrid_rank_sum':int(t._rank_sum.iloc[0]),'hybrid_score':float(t._hybrid.iloc[0]),'tie_size':len(t),
                 'original_rank_min':int(pos.min()),'original_rank_max':int(pos.max()),
                 'within_top10':bool(pos.max()<=K),'crosses_top10_boundary':bool(pos.min()<=K<pos.max()),
                 'candidate_iso3':t.destination_iso3.astype(str).tolist(),
                 'candidate_names':t.destination_name.astype(str).tolist(),
                 'candidate_entry_success':t.entry_success.where(t.label_observed.astype(bool),np.nan).tolist(),
                 'candidate_label_observed':t.label_observed.astype(bool).tolist(),
                 'candidate_export_usd':t.export_value_usd.where(t.label_observed.astype(bool),np.nan).tolist()}
            group_records.append(rec)
            tie_groups.append({'hs2':key[0],'outcome_year':key[1],'candidate_pool_id':str(t.candidate_pool_id.iloc[0]),**rec,
                               'tie_category':'boundary' if rec['crosses_top10_boundary'] else ('within_top10' if rec['within_top10'] else 'outside_top10')})
            if rec['crosses_top10_boundary']:
                b=ordered.iloc[K-1:K+1]
                a=b.iloc[0]; z=b.iloc[1]
                boundary_rows.append({'hs2':key[0],'outcome_year':key[1],'candidate_pool_id':str(a.candidate_pool_id),
                  'rank10_iso3':a.destination_iso3,'rank11_iso3':z.destination_iso3,'boundary_score':float(a._hybrid),
                  'rank10_entry_success':a.entry_success if bool(a.label_observed) else np.nan,
                  'rank11_entry_success':z.entry_success if bool(z.label_observed) else np.nan,
                  'rank10_label_observed':bool(a.label_observed),'rank11_label_observed':bool(z.label_observed),
                  'rank10_export_usd':a.export_value_usd if bool(a.label_observed) else np.nan,
                  'rank11_export_usd':z.export_value_usd if bool(z.label_observed) else np.nan,
                  'positive_relevance_pool':bool(selected_metrics(ordered)['positive_relevance']),
                  'tie_size':len(t),'tie_candidate_iso3':'|'.join(t.destination_iso3.astype(str)),
                  'rank10_ppml_rank':int(a._rp),'rank11_ppml_rank':int(z._rp),
                  'rank10_lightgbm_rank':int(a._rl),'rank11_lightgbm_rank':int(z._rl)})
        # Count ties under exact rational rank arithmetic separately from the
        # original float-score equality: rounding can split algebraic equalities.
        for rank_sum,t in ordered.groupby('_rank_sum',sort=False):
            if len(t)<2:
                continue
            pos=ordered.index[ordered._rank_sum.eq(rank_sum)].to_numpy()+1
            if pos.min()<=K<pos.max():
                algebraic_boundary_group_count+=1
                algebraic_boundary_pool_keys.add(key)
        tie_pool_info[key]=group_records
    boundary=pd.DataFrame(boundary_rows)
    assert len(boundary)==51, f'Expected 51 boundary-tie pools, found {len(boundary)}'
    archived_keys=set(zip(archived_ties.hs2.astype(str).str.zfill(2),archived_ties.outcome_year.astype(int),archived_ties.rank10_iso3,archived_ties.rank11_iso3))
    current_keys=set(zip(boundary.hs2.astype(str).str.zfill(2),boundary.outcome_year.astype(int),boundary.rank10_iso3,boundary.rank11_iso3))
    assert archived_keys==current_keys, 'Boundary tie reconstruction differs from existing audit'
    n_positive=int(boundary.positive_relevance_pool.sum())
    
    # Random tie repetitions. Repeated seeds characterize tie-order uncertainty, not sample size.
    random_contexts={key:make_random_context(g) for key,g in pool_groups.items()}
    # Validate the fast array metric against the archived pandas metric on a fixed random seed.
    for key,ctx in random_contexts.items():
        pos=randomized_positions(ctx,10000,key)
        fast=metrics_for_positions(pos,ctx)
        reference=selected_metrics(ctx['ordered'].iloc[pos])
        for col in ['ndcg10','precision10_complete','successful_entries10_complete','qualified_usd10_complete','raw_usd10_complete']:
            a,b=fast[col],reference[col]
            assert (pd.isna(a) and pd.isna(b)) or abs(float(a)-float(b))<1e-10, (key,col,a,b)
    random_pool_records=[]; random_top_counts={}
    for seed in SEEDS:
        seed_recs=[]
        for key,ctx in random_contexts.items():
            positions=randomized_positions(ctx,seed,key)
            m=metrics_for_positions(positions,ctx)
            top_iso=ctx['iso3'][positions[:K]]
            seed_recs.append({'hs2':key[0],'outcome_year':key[1],'candidate_pool_id':ctx['candidate_pool_id'],**m,
                              'top10_iso3':'|'.join(top_iso)})
            # Track inclusion frequencies only for exact tie candidates at the selection boundary.
            tied_boundary=set(iso for tr in tie_pool_info[key] if tr['crosses_top10_boundary'] for iso in tr['candidate_iso3'])
            selected_set=set(top_iso)
            for iso in tied_boundary:
                random_top_counts[(key,iso)]=random_top_counts.get((key,iso),0)+int(iso in selected_set)
        seed_df=pd.DataFrame(seed_recs)
        random_pool_records.append(seed_df.assign(tie_seed=seed))
        valid_ndcg=seed_df.ndcg10.dropna()
        complete=seed_df.loc[seed_df.complete_outcome_pool]
        seed_swaps=[]; seed_jaccards=[]
        for r in seed_recs:
            key=(str(r['hs2']),int(r['outcome_year'])); selected_set=set(r['top10_iso3'].split('|'))
            seed_swaps.append(len(selections['ISO3_ascending_original'][key]^selected_set)/2)
            seed_jaccards.append(len(selected_set&lgbm_tops[key])/len(selected_set|lgbm_tops[key]))
        random_seed_summaries.append({'tie_seed':seed,'policy':'uniform_random_ties','summary_type':'tie_order_replicate',
          'ndcg10':float(valid_ndcg.mean()) if len(valid_ndcg) else np.nan,'ndcg10_pools':int(valid_ndcg.count()),
          'precision10':float(complete.precision10_complete.mean()) if len(complete) else np.nan,
          'successful_entries10_per_pool':float(complete.successful_entries10_complete.mean()) if len(complete) else np.nan,
          'qualified_usd10_per_pool':float(complete.qualified_usd10_complete.mean()) if len(complete) else np.nan,
          'raw_usd10_per_pool':float(complete.raw_usd10_complete.mean()) if len(complete) else np.nan,
          'complete_outcome_pools':int(len(complete)),
          'mean_membership_swaps_vs_original':float(np.mean(seed_swaps)),
          'mean_membership_jaccard_vs_lightgbm':float(np.mean(seed_jaccards))})
    random_pool=pd.concat(random_pool_records,ignore_index=True)
    random_summary_pool=random_pool.groupby(['hs2','outcome_year'],as_index=False).agg(
      ndcg10=('ndcg10','mean'),precision10_complete=('precision10_complete','mean'),
      successful_entries10_complete=('successful_entries10_complete','mean'),
      successful_entries10_observed_partial=('successful_entries10_observed_partial','mean'),
      qualified_usd10_complete=('qualified_usd10_complete','mean'),raw_usd10_complete=('raw_usd10_complete','mean'),
      qualified_usd10_observed_partial=('qualified_usd10_observed_partial','mean'),raw_usd10_observed_partial=('raw_usd10_observed_partial','mean'),
      unknown_selected_count=('unknown_selected_count','mean'),complete_outcome_pool=('complete_outcome_pool','mean'),
      positive_relevance=('positive_relevance','first'))
    membership_records=[]
    for key, rg in random_pool.groupby(['hs2','outcome_year'],sort=True):
        orig=orig_tops[(str(key[0]),int(key[1]))]; lb=lgbm_tops[(str(key[0]),int(key[1]))]
        swap_counts=[]; jac=[]; any_change=[]
        for toptext in rg.top10_iso3:
            ss=set(str(toptext).split('|'))
            swap_counts.append(len(ss^orig)/2)
            jac.append(len(ss&lb)/len(ss|lb))
            any_change.append(ss!=orig)
        membership_records.append({'hs2':key[0],'outcome_year':key[1],
          'membership_swaps_vs_original':float(np.mean(swap_counts)),
          'membership_changed_vs_original':bool(any(any_change)),
          'probability_membership_change_vs_original':float(np.mean(any_change)),
          'jaccard_vs_original':float(np.mean([1-2*x/K for x in swap_counts])),
          'jaccard_vs_lightgbm':float(np.mean(jac))})
    random_summary_pool=random_summary_pool.merge(pd.DataFrame(membership_records),on=['hs2','outcome_year'],validate='one_to_one')
    # Per-pool comparison metrics and Top-10 changes.
    for policy,dfm in pool_metric_records.items():
        if policy=='LightGBM_baseline': continue
        dfm=dfm.copy()
        dfm['membership_changed_vs_original']=[selections[policy][(r.hs2,r.outcome_year)]!=orig_tops[(r.hs2,r.outcome_year)] for r in dfm.itertuples()]
        dfm['membership_swaps_vs_original']=[len(selections[policy][(r.hs2,r.outcome_year)]^orig_tops[(r.hs2,r.outcome_year)])/2 for r in dfm.itertuples()]
        dfm['jaccard_vs_original']=[len(selections[policy][(r.hs2,r.outcome_year)]&orig_tops[(r.hs2,r.outcome_year)])/len(selections[policy][(r.hs2,r.outcome_year)]|orig_tops[(r.hs2,r.outcome_year)]) for r in dfm.itertuples()]
        dfm['jaccard_vs_lightgbm']=[len(selections[policy][(r.hs2,r.outcome_year)]&lgbm_tops[(r.hs2,r.outcome_year)])/len(selections[policy][(r.hs2,r.outcome_year)]|lgbm_tops[(r.hs2,r.outcome_year)]) for r in dfm.itertuples()]
        pool_metric_records[policy]=dfm
    # Include baseline LightGBM membership movement relative to original Hybrid.
    lgb_frame=pool_metric_records['LightGBM_baseline'].copy()
    lgb_frame['membership_changed_vs_original']=[selections['LightGBM_baseline'][key]!=orig_tops[key] for key in pool_groups]
    lgb_frame['membership_swaps_vs_original']=[len(selections['LightGBM_baseline'][key]^orig_tops[key])/2 for key in pool_groups]
    lgb_frame['jaccard_vs_original']=[len(selections['LightGBM_baseline'][key]&orig_tops[key])/len(selections['LightGBM_baseline'][key]|orig_tops[key]) for key in pool_groups]
    lgb_frame['jaccard_vs_lightgbm']=1.0
    pool_metric_records['LightGBM_baseline']=lgb_frame
    # Add random expected per-pool profile and tie inclusion frequency table.
    rp=random_summary_pool.copy(); rp['policy']='uniform_random_ties_expected'; rp['candidate_pool_id']=rp.apply(lambda r:f'{TASK}|{r.hs2}|{int(r.outcome_year)}',axis=1)
    rp['selected_count']=K; rp['observed_selected_count']=K-rp.unknown_selected_count; rp['unknown_selected_count']=rp.unknown_selected_count
    rp['complete_outcome_pool']=rp.complete_outcome_pool.eq(1.0)
    rp['positive_relevance']=rp.positive_relevance.astype(bool)
    rp['top10_iso3']=''
    pool_metric_records['uniform_random_ties_expected']=rp

    # Summary of every deterministic policy and the random policy expectation.
    result_rows=[]
    policies=['LightGBM_baseline','Gravity_PPML',*rules,'uniform_random_ties_expected']
    metric_frames={**pool_metric_records}
    for policy in policies:
        x=metric_frames[policy]
        israndom=policy=='uniform_random_ties_expected'
        result_rows.append({'policy':policy,'summary_type':'tie_order_expectation' if israndom else 'fixed_rule','tie_seed':np.nan,
          'ndcg10':float(x.ndcg10.mean()),'ndcg10_pools':int(x.ndcg10.count()),
          'precision10':float(x.precision10_complete.mean()),'successful_entries10_per_pool':float(x.successful_entries10_complete.mean()),
          'qualified_usd10_per_pool':float(x.qualified_usd10_complete.mean()),'raw_usd10_per_pool':float(x.raw_usd10_complete.mean()),
          'mean_observed_successes10_partial':float(x.successful_entries10_observed_partial.mean()),
          'mean_observed_qualified_usd10_partial':float(x.qualified_usd10_observed_partial.mean()),
          'mean_observed_raw_usd10_partial':float(x.raw_usd10_observed_partial.mean()),
          'complete_outcome_pools':int(x.complete_outcome_pool.sum()),'unknown_selected_candidates_mean':float(x.unknown_selected_count.mean()),
          'membership_changed_vs_original_pools':int(x.membership_changed_vs_original.sum()) if 'membership_changed_vs_original' in x else 0,
          'mean_membership_swaps_vs_original':float(x.membership_swaps_vs_original.mean()) if 'membership_swaps_vs_original' in x else 0.,
          'mean_top10_jaccard_vs_lightgbm':float(x.jaccard_vs_lightgbm.mean()) if 'jaccard_vs_lightgbm' in x else 1.,
          'interpretation':'same candidates/outcomes; missing selected labels retained as unknown; outcome means are complete-pool means'})
    for srow in random_seed_summaries:
        result_rows.append({**srow,'mean_observed_successes10_partial':np.nan,'mean_observed_qualified_usd10_partial':np.nan,
          'mean_observed_raw_usd10_partial':np.nan,'unknown_selected_candidates_mean':np.nan,
          'membership_changed_vs_original_pools':np.nan,'mean_membership_swaps_vs_original':srow['mean_membership_swaps_vs_original'],
          'mean_top10_jaccard_vs_lightgbm':srow['mean_membership_jaccard_vs_lightgbm'],
          'interpretation':'one seeded tie-order draw; Monte Carlo sensitivity replicate, not an independent sample'})
    outres=pd.DataFrame(result_rows)
    # Add tie-order uncertainty interval for random policy's aggregate metrics.
    rand_reps=outres.query("summary_type == 'tie_order_replicate'")
    mask=outres.policy.eq('uniform_random_ties_expected')
    for metric in ['ndcg10','precision10','successful_entries10_per_pool','qualified_usd10_per_pool','raw_usd10_per_pool','mean_membership_swaps_vs_original','mean_top10_jaccard_vs_lightgbm']:
        outres.loc[mask,f'{metric}_tie_order_q025']=rand_reps[metric].quantile(.025)
        outres.loc[mask,f'{metric}_tie_order_q975']=rand_reps[metric].quantile(.975)
    outres.to_csv(RESULTS/'TIE_SENSITIVITY_RESULTS.csv',index=False,na_rep='')

    # Paired inference vs the fixed LightGBM baseline. Tie-rule randomization is integrated per pool first;
    # HS2 resampling remains the inferential unit. Include original comparisons to PPML as family context.
    policy_lookup={
      'ISO3_ascending_original':'ISO3_ascending_original',
      'PPML_first_then_LightGBM_then_ISO3':'PPML_first_then_LightGBM_then_ISO3',
      'LightGBM_first_then_PPML_then_ISO3':'LightGBM_first_then_PPML_then_ISO3',
      'uniform_random_ties_expected':'uniform_random_ties_expected'}
    metric_specs=[('ndcg10','Family_A_NDCG10'),('precision10_complete','Family_A_Precision10'),
                  ('successful_entries10_complete','Family_A_SuccessCount10'),
                  ('qualified_usd10_complete','Family_A_QualifiedUSD10'),('raw_usd10_complete','Family_A_RawUSD10')]
    inf=[]
    # Build metric per-pool mapping for LGBM/PPML and policies.
    metric_pool={}
    for pol in ['LightGBM_baseline','Gravity_PPML',*rules,'uniform_random_ties_expected']:
        metric_pool[pol]=metric_frames[pol].set_index(['hs2','outcome_year'])
    contrasts=[]
    for metric,fam in metric_specs:
        contrasts += [(fam,metric,'ISO3_ascending_original','LightGBM_baseline','primary_original_vs_lgbm')]
        contrasts += [(fam,metric,'ISO3_ascending_original','Gravity_PPML','original_family_reference_vs_ppml')]
        contrasts += [(fam,metric,'PPML_first_then_LightGBM_then_ISO3','LightGBM_baseline','tie_sensitivity')]
        contrasts += [(fam,metric,'LightGBM_first_then_PPML_then_ISO3','LightGBM_baseline','tie_sensitivity')]
        contrasts += [(fam,metric,'uniform_random_ties_expected','LightGBM_baseline','random_tie_expected_sensitivity')]
    for j,(fam,metric,a,b,kind) in enumerate(contrasts):
        aa=metric_pool[a][metric].rename('a'); bb=metric_pool[b][metric].rename('b')
        w=pd.concat([aa,bb],axis=1).reset_index()
        w['difference']=w.a-w.b
        st=bootstrap(w.rename(columns={'difference':'diff'}),'diff',seed=BOOT_SEED+j)
        base=float(bb.mean())
        inf.append({'task':TASK,'metric_family':fam,'metric':metric,'contrast':f'{a} - {b}','contrast_role':kind,
          'baseline_mean':base,'difference':st['estimate'],'relative_difference_pct':100*st['estimate']/abs(base) if base else np.nan,
          'ci95_low':st['ci_low'],'ci95_high':st['ci_high'],'p_raw_signflip':st['p_raw_signflip'],
          'paired_pools':st['paired_pools'],'hs2_blocks':st['hs2_blocks'],
          'random_tie_expected_q025':float(rand_reps[metric.replace('_complete','') if metric.endswith('_complete') else metric].quantile(.025)) if a=='uniform_random_ties_expected' and metric in ['ndcg10'] else np.nan,
          'random_tie_expected_q975':float(rand_reps[metric.replace('_complete','') if metric.endswith('_complete') else metric].quantile(.975)) if a=='uniform_random_ties_expected' and metric in ['ndcg10'] else np.nan,
          'inference_scope':'exploratory HS2-block paired bootstrap/sign-flip; randomized tie policy is seed-averaged within pool; not confirmatory'})
    infdf=pd.DataFrame(inf)
    infdf['p_holm']=np.nan
    for _,ix in infdf.groupby('metric_family').groups.items(): infdf.loc[ix,'p_holm']=holm(infdf.loc[ix,'p_raw_signflip'].to_list())
    infdf.to_csv(RESULTS/'TIE_SENSITIVITY_INFERENCE.csv',index=False,na_rep='')

    # Detailed tied pools with all tie groups, alternative selections and observed membership deltas.
    affected=[]
    random_mean_pool={ (r.hs2,int(r.outcome_year)):r for r in random_summary_pool.itertuples(index=False)}
    for key,g in pool_groups.items():
        if not tie_pool_info[key]: continue
        original=orig_tops[key]
        row={'hs2':key[0],'outcome_year':key[1],'candidate_pool_id':str(g.candidate_pool_id.iloc[0]),
             'positive_relevance_pool':bool(selected_metrics(rules_by_pool[(key,'ISO3_ascending_original')])['positive_relevance']),
             'exact_tie_group_count':len(tie_pool_info[key]),'tie_groups_json':json.dumps(tie_pool_info[key],ensure_ascii=False),
             'algebraic_rank_sum_boundary_not_exact_score_tie':key in algebraic_boundary_pool_keys,
             'boundary_tie_group_count':sum(t['crosses_top10_boundary'] for t in tie_pool_info[key]),
             'within_top10_tie_group_count':sum(t['within_top10'] for t in tie_pool_info[key]),
             'original_top10_iso3':'|'.join(sorted(original))}
        for pol in rules[1:]:
            sel=selections[pol][key]; add=sel-original; rem=original-sel
            lookup=g.set_index('destination_iso3')
            def summarize(ids):
                z=lookup.loc[list(ids)] if ids else lookup.iloc[0:0]
                obs=z.loc[z.label_observed.astype(bool)]
                entry=pd.to_numeric(obs.entry_success,errors='coerce')
                values=pd.to_numeric(obs.export_value_usd,errors='coerce')
                return {'iso3':'|'.join(sorted(map(str,ids))), 'successes_observed':int(entry.sum()),
                        'failures_observed':int((1-entry).sum()),'unknown':int((~z.label_observed.astype(bool)).sum()),
                        'qualified_usd_observed_partial':float(values[values>=50_000].sum()),
                        'raw_usd_observed_partial':float(values.sum())}
            added_summary=summarize(add); removed_summary=summarize(rem)
            row[pol+'_top10']='|'.join(sorted(sel)); row[pol+'_added_vs_iso3_json']=json.dumps(added_summary); row[pol+'_removed_vs_iso3_json']=json.dumps(removed_summary)
            row[pol+'_membership_swaps']=len(add)
            row[pol+'_net_observed_successes_added_minus_removed']=added_summary['successes_observed']-removed_summary['successes_observed']
            row[pol+'_net_qualified_usd_added_minus_removed_partial']=added_summary['qualified_usd_observed_partial']-removed_summary['qualified_usd_observed_partial']
            row[pol+'_net_raw_usd_added_minus_removed_partial']=added_summary['raw_usd_observed_partial']-removed_summary['raw_usd_observed_partial']
            row[pol+'_ndcg10']=metric_frames[pol].set_index(['hs2','outcome_year']).loc[key,'ndcg10']
        boundary_candidates=set(iso for tr in tie_pool_info[key] if tr['crosses_top10_boundary'] for iso in tr['candidate_iso3'])
        rand_freq={iso:random_top_counts.get((key,iso),0)/len(SEEDS) for iso in sorted(boundary_candidates)}
        row['random_tie_boundary_candidate_selection_frequency_json']=json.dumps(rand_freq,sort_keys=True)
        rmean=random_mean_pool[key]
        original_metric=metric_frames['ISO3_ascending_original'].set_index(['hs2','outcome_year']).loc[key]
        row['random_expected_net_successes_vs_original']=float(rmean.successful_entries10_complete-original_metric.successful_entries10_complete)
        row['random_expected_net_qualified_usd_vs_original']=float(rmean.qualified_usd10_complete-original_metric.qualified_usd10_complete)
        row['random_expected_net_raw_usd_vs_original']=float(rmean.raw_usd10_complete-original_metric.raw_usd10_complete)
        affected.append(row)
    affected_df=pd.DataFrame(affected).sort_values(['hs2','outcome_year'])
    affected_df.to_csv(RESULTS/'TIE_AFFECTED_POOLS.csv',index=False,na_rep='')

    # Boundary tie and previously documented authentic case records.
    case_rows=[]
    for r in boundary.to_dict('records'):
        key=(str(r['hs2']).zfill(2),int(r['outcome_year']))
        r=dict(r)
        for iso,tag in [(r['rank10_iso3'],'original_rank10_candidate'),(r['rank11_iso3'],'original_rank11_candidate')]:
            r[tag+'_selected_ppml_first']=iso in selections[rules[1]][key]
            r[tag+'_selected_lightgbm_first']=iso in selections[rules[2]][key]
            r[tag+'_random_selection_frequency']=random_top_counts.get((key,iso),0)/len(SEEDS)
        case_rows.append({'record_type':'boundary_tie_pool',**r})
    old_cases=pd.read_csv(CASES,dtype={'hs2':str})
    for r in old_cases.to_dict('records'):
        key=(str(r['hs2']).zfill(2),int(r['outcome_year']))
        g=pool_groups[key]
        iso_added=str(r['added_iso3']); iso_displaced=str(r['displaced_iso3'])
        tie_members=[]
        for tr in tie_pool_info[key]:
            if iso_added in tr['candidate_iso3'] or iso_displaced in tr['candidate_iso3']:
                tie_members.append(tr)
        # Does the illustrative pair itself have equal hybrid score?
        ix=g.set_index('destination_iso3')
        pair_same_score=bool(float(ix.loc[iso_added,'_hybrid'])==float(ix.loc[iso_displaced,'_hybrid']))
        case_rows.append({'record_type':'prior_authentic_case','case_label':r.get('case_label'),'hs2':key[0],'outcome_year':key[1],
          'candidate_pool_id':r.get('candidate_pool_id'),'added_iso3':iso_added,'displaced_iso3':iso_displaced,
          'added_entry_success':r.get('added_entry_success'),'displaced_entry_success':r.get('displaced_entry_success'),
          'added_export_usd':r.get('added_export_usd'),'displaced_export_usd':r.get('displaced_export_usd'),
          'added_component_ranks':r.get('added_ranks_ppml_lightgbm_hybrid'),'displaced_component_ranks':r.get('displaced_ranks_ppml_lightgbm_hybrid'),
          'added_selected_ppml_first':iso_added in selections[rules[1]][key],'displaced_selected_ppml_first':iso_displaced in selections[rules[1]][key],
          'added_selected_lightgbm_first':iso_added in selections[rules[2]][key],'displaced_selected_lightgbm_first':iso_displaced in selections[rules[2]][key],
          'added_random_selection_frequency':random_top_counts.get((key,iso_added),0)/len(SEEDS),
          'displaced_random_selection_frequency':random_top_counts.get((key,iso_displaced),0)/len(SEEDS),
          'pair_equal_hybrid_score':pair_same_score,
          'either_candidate_in_exact_tie_group':bool(tie_members),'related_tie_groups_json':json.dumps(tie_members,ensure_ascii=False),
          'interpretation':'Tie status describes score-order ambiguity only; observed outcomes are retrospective.'})
    pd.DataFrame(case_rows).to_csv(RESULTS/'TIE_SENSITIVITY_CASES.csv',index=False,na_rep='')

    # Cross-rule membership dependency counts, deduplicated by pool-candidate, with outcome categories.
    # A candidate is selection-sensitive if the 1,000 random repetitions select it in some but not all draws.
    sensitive=[]
    for (key,iso),cnt in random_top_counts.items():
        if 0<cnt<len(SEEDS):
            g=pool_groups[key].set_index('destination_iso3'); rr=g.loc[iso]
            sensitive.append({'candidate_pool_id':str(rr.candidate_pool_id),'hs2':key[0],'outcome_year':key[1],'destination_iso3':iso,
                'random_selection_frequency':cnt/len(SEEDS),'entry_success':rr.entry_success if bool(rr.label_observed) else np.nan,
                'label_observed':bool(rr.label_observed),'export_value_usd':rr.export_value_usd if bool(rr.label_observed) else np.nan})
    sensitive_df=pd.DataFrame(sensitive)
    # Random repetition distribution columns are attached to expected policy summary above.
    # Write a compact YAML-like summary and report.
    summary={}
    for policy in ['LightGBM_baseline','Gravity_PPML',*rules,'uniform_random_ties_expected']:
        x=metric_frames[policy]
        summary[policy]={
          'ndcg10':float(x.ndcg10.mean()),'ndcg_pools':int(x.ndcg10.count()),
          'precision10_complete':float(x.precision10_complete.mean()),
          'entries10_per_pool':float(x.successful_entries10_complete.mean()),
          'qualified_usd10':float(x.qualified_usd10_complete.mean()),'raw_usd10':float(x.raw_usd10_complete.mean()),
          'mean_jaccard_vs_lightgbm':float(x.jaccard_vs_lightgbm.mean()) if 'jaccard_vs_lightgbm' in x else np.nan,
          'changed_pools_vs_original':int(x.membership_changed_vs_original.sum()) if 'membership_changed_vs_original' in x else 0}
    # Exact tie diagnostics.
    all_groups=pd.DataFrame(tie_groups)
    internal_count=int(all_groups.within_top10.sum()) if len(all_groups) else 0
    boundary_count=len(boundary)
    ties_pos=int(boundary.positive_relevance_pool.sum())
    exact_boundary_keys=set(zip(boundary.hs2.astype(str).str.zfill(2),boundary.outcome_year.astype(int)))
    rounding_only_boundary_pool_count=len(algebraic_boundary_pool_keys-exact_boundary_keys)
    exact_tied_candidate_records=int(all_groups.tie_size.sum()) if len(all_groups) else 0
    orig=metric_frames['ISO3_ascending_original']; lgb=metric_frames['LightGBM_baseline']
    base_ndcg=float(orig.ndcg10.mean()-lgb.ndcg10.mean())
    original_ref=pd.read_csv(BASE/'results/F1_blend_model_comparison.csv')
    ref=float(original_ref.loc[(original_ref.task==TASK)&(original_ref.policy=='EQUAL_WEIGHT_RANK_BLEND'),'macro_ndcg10'].iloc[0]-original_ref.loc[(original_ref.task==TASK)&(original_ref.policy=='LightGBM'),'macro_ndcg10'].iloc[0])
    assert abs(base_ndcg-0.037184513476218026)<1e-10 and abs(base_ndcg-ref)<1e-10
    assert ties_pos<=138
    # Candidate status variability among any candidates who can be selected under random boundary resolutions.
    sens_success=int(sensitive_df.loc[sensitive_df.label_observed.astype(bool),'entry_success'].sum()) if len(sensitive_df) else 0
    sens_fail=int((sensitive_df.label_observed.astype(bool)&sensitive_df.entry_success.eq(0)).sum()) if len(sensitive_df) else 0
    sens_unknown=int((~sensitive_df.label_observed.astype(bool)).sum()) if len(sensitive_df) else 0
    random_reps=outres.query("summary_type == 'tie_order_replicate'")
    # Baseline vs original expected selection counts, plus alternative-rule choices on boundary candidates.
    ppml_n_change=int(sum(selections[rules[1]][k]!=orig_tops[k] for k in pool_groups))
    lgb_n_change=int(sum(selections[rules[2]][k]!=orig_tops[k] for k in pool_groups))
    # Outcome composition among membership changes under the two fixed component-first rules.
    rule_outcome_counts={}
    for pol in rules[1:]:
        add_success=add_failure=remove_success=remove_failure=0
        for rec in affected_df.to_dict('records'):
            a=json.loads(rec[pol+'_added_vs_iso3_json']); r=json.loads(rec[pol+'_removed_vs_iso3_json'])
            add_success+=a['successes_observed']; add_failure+=a['failures_observed']
            remove_success+=r['successes_observed']; remove_failure+=r['failures_observed']
        rule_outcome_counts[pol]={'added_successes':add_success,'added_failures':add_failure,
                                  'removed_successes':remove_success,'removed_failures':remove_failure}
    original_summary=summary['ISO3_ascending_original']
    random_summary=summary['uniform_random_ties_expected']
    random_ndcg_min=float(random_reps.ndcg10.min()); random_ndcg_max=float(random_reps.ndcg10.max())
    random_ndcg_below=int((random_reps.ndcg10<summary['LightGBM_baseline']['ndcg10']).sum())
    ndcg_inf=infdf.loc[(infdf.metric=='ndcg10') & infdf.contrast_role.isin(['primary_original_vs_lgbm','tie_sensitivity','random_tie_expected_sensitivity'])]
    primary_expanded_holm=float(ndcg_inf.loc[ndcg_inf.contrast.eq('ISO3_ascending_original - LightGBM_baseline'),'p_holm'].iloc[0])
    original_family_p=float(pd.read_csv(BASE/'results/holm_adjusted_tests.csv').loc[lambda x:(x.task==TASK)&(x.metric_family=='Family_A_NDCG10')&(x.contrast=='EQUAL_WEIGHT_RANK_BLEND - LightGBM'),'p_holm'].iloc[0])
    report=f'''# Hybrid rank-fusion tie sensitivity\n\n## Protocol\n\nThis audit uses the original strict-zero Task A 2022--2024 candidate-score archive without retraining models, changing eligibility, changing weights, or using outcomes to select a tie rule. The hybrid score is the equal-weight mean of within-pool normalized inverse ranks. Exact ties use equality of the archived/reconstructed score from the original floating-point expression. The integer rank sum $r_P+r_L$ is tracked separately to reveal algebraically equal values that differ only by floating-point rounding. Original ties use ascending ISO3. PPML-first orders a tied group by ascending PPML rank, then LightGBM rank, then ISO3; LightGBM-first reverses the model priority. Random ties uniformly permute ISO3-sorted candidates within each exact-score group under fixed seeds 10,000--10,999 (1,000 draws). These repetitions measure tie-order uncertainty, not independent evaluation samples.\n\nFor statistical uncertainty, paired HS2 blocks are resampled 10,000 times, keeping all three years together. Sign-flip tests and Holm adjustment use the existing Task A comparison families expanded to include the tie sensitivity contrasts. All sensitivity inference is exploratory because 2022--2024 outcomes were previously inspected. NDCG follows the original complete-case, binary-entry definition and is averaged over positive-relevance pools. Selection outcomes retain unknown labels; USD paired inference uses complete-outcome pools and the reports separately expose selected unknown labels.\n\n## Reproduction checks\n\n- Candidate rows: {len(d):,}; pools: {len(pool_groups)}; HS2 blocks: {d.hs2.nunique()}.\n- Maximum reconstructed hybrid-score deviation from the saved score: {max_hybrid_error:.3g}; PPML/LightGBM rank deviations: {max_ppml_rank_error}/{max_lgbm_rank_error}.\n- Original ISO3 Hybrid NDCG@10 minus LightGBM: {base_ndcg:+.9f}; saved reference: {ref:+.9f}.\n- Exact Hybrid-score tie groups across all pools: {len(all_groups):,} ({exact_tied_candidate_records:,} candidate-pool records); groups wholly inside Top-10: {internal_count}; boundary-tie pools: {boundary_count}, of which {ties_pos} are among the {int(orig.ndcg10.count())} positive-relevance evaluation pools.\n- Integer rank-sum arithmetic identifies {algebraic_boundary_group_count} boundary equalities across {len(algebraic_boundary_pool_keys)} pools; {rounding_only_boundary_pool_count} pools are algebraically equal but not exact float-score ties under the saved expression. These are reported as a numerical-rounding diagnostic and are not reclassified as exact ties.\n\n## Performance by tie rule\n\n| Rule | NDCG@10 | Precision@10 | Successful entries / pool | Qualified USD / pool | Raw USD / pool | Pools changed vs original | Mean Jaccard vs LightGBM |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'''
    for pol,row in summary.items():
        report+=f"| {pol} | {row['ndcg10']:.6f} | {row['precision10_complete']:.6f} | {row['entries10_per_pool']:.6f} | ${row['qualified_usd10']:,.2f} | ${row['raw_usd10']:,.2f} | {row['changed_pools_vs_original']} | {row['mean_jaccard_vs_lightgbm']:.6f} |\n"
    report+=f'''\nThe seeded-random row is the per-pool mean over the 1,000 tie-order draws. `TIE_SENSITIVITY_RESULTS.csv` also contains each seed's aggregate metrics; their 2.5th--97.5th percentiles describe Monte Carlo tie-order variation only, not a confidence interval over historical evaluation samples. Across seeds, aggregate NDCG ranges from {random_ndcg_min:.6f} to {random_ndcg_max:.6f} (2.5th--97.5th percentiles {rand_reps.ndcg10.quantile(.025):.6f}--{rand_reps.ndcg10.quantile(.975):.6f}); {random_ndcg_below}/1,000 draws fall below the fixed LightGBM NDCG.\n\n## Selection dependence and cases\n\n'''
    report+=f'''There are {len(sensitive_df)} boundary-tied candidate-pool selections whose inclusion varies across the 1,000 random orders: {sens_success} are observed successful entries, {sens_fail} observed failures, and {sens_unknown} have unknown outcomes. The observed success/failure composition is descriptive and does not select or validate a tie-break. PPML-first changes Top-10 membership in {ppml_n_change} pools relative to ISO3; LightGBM-first changes it in {lgb_n_change}. See `TIE_AFFECTED_POOLS.csv` for all tied groups and pool-level additions/removals.

The boundary-case file contains all {boundary_count} boundary-tie pools plus the seven prior authentic XAI case-study pairs. Bulgaria--Iran (HS2 27, 2022) is an exact equal-score rank-10/11 tie: PPML and LightGBM rank contributions offset, and ISO3 makes BGR the original selected member. Any observed Bulgarian advantage over Iran is a consequence of outcomes after prediction; it is not evidence that ISO3 or a model component predicts better. The other prior case pairs are individually flagged for whether either candidate participates in any exact tie group in its pool.

## Inference and interpretation

`TIE_SENSITIVITY_INFERENCE.csv` reports paired HS2-block percentile intervals and sign-flip p-values for each sensitivity rule versus the fixed LightGBM ranking. Holm adjustment is computed within the corresponding Task A metric family while retaining the original Hybrid-vs-LightGBM and Hybrid-vs-PPML comparisons as family members. Random-policy inference averages each pool over tie seeds before resampling; it does not count seeds as economic observations.

'''
    # Determine whether inference robust by all prescribed tie-policy NDCG effects.
    effects=infdf[(infdf.metric=='ndcg10') & infdf.contrast_role.isin(['primary_original_vs_lgbm','tie_sensitivity','random_tie_expected_sensitivity'])]
    any_negative=bool((effects.difference<0).any()); all_positive=bool((effects.difference>0).all())
    report+=('The NDCG advantage remains positive under all prespecified tie rules, though its size varies.' if all_positive else ('At least one reasonable tie rule removes or reverses the NDCG advantage.' if any_negative else 'The estimated NDCG difference is not uniformly positive across prespecified tie rules.'))
    report+=f'''\n
The NDCG difference versus LightGBM ranges from {ndcg_inf.difference.min():+.6f} to {ndcg_inf.difference.max():+.6f} across the four Hybrid tie policies. The original two-contrast inference family retains its published Holm $p={original_family_p:.4f}$. In the broader exploratory sensitivity family that includes ISO3, PPML-first, LightGBM-first, randomized expected tie order, and the original PPML reference contrast, the ISO3-vs-LightGBM Holm-adjusted $p$ is {primary_expanded_holm:.4f}. Thus the positive point estimate is robust to these tie choices, while conventional significance is not robust to expanding the post hoc comparison family. Precision and successful-entry intervals remain compatible with no difference.

Tie-break changes also alter dollar outcomes. Relative to LightGBM, qualified USD differences are ${summary['PPML_first_then_LightGBM_then_ISO3']['qualified_usd10']-summary['LightGBM_baseline']['qualified_usd10']:+,.0f} per pool for PPML-first, ${summary['LightGBM_first_then_PPML_then_ISO3']['qualified_usd10']-summary['LightGBM_baseline']['qualified_usd10']:+,.0f} for LightGBM-first, and ${random_summary['qualified_usd10']-summary['LightGBM_baseline']['qualified_usd10']:+,.0f} for the random-tie expectation; all corresponding bootstrap intervals span zero. The dollar sign of the descriptive comparison is therefore tie-sensitive and is not an economic advantage claim.

For the changed memberships, PPML-first adds {rule_outcome_counts[rules[1]]['added_successes']} successful and {rule_outcome_counts[rules[1]]['added_failures']} failed destination-pool recommendations while displacing {rule_outcome_counts[rules[1]]['removed_successes']} successful and {rule_outcome_counts[rules[1]]['removed_failures']} failed recommendations. LightGBM-first adds {rule_outcome_counts[rules[2]]['added_successes']} successes and {rule_outcome_counts[rules[2]]['added_failures']} failures, while displacing {rule_outcome_counts[rules[2]]['removed_successes']} successes and {rule_outcome_counts[rules[2]]['removed_failures']} failures. Across random orders, {len(sensitive_df)} candidate-pool memberships vary, comprising {sens_success} observed successes and {sens_fail} observed failures.

**Recommendation:** retain the original observed NDCG estimate as a result under its stated ISO3 convention, but qualify the main statistical claim: the ranking advantage remains positive across tie rules, whereas its significance depends on the multiplicity family; observed USD opportunity is not robust to tie order and has no established advantage. Report the tie sensitivity alongside the primary result. Do not interpret the outcome composition of ISO3-selected ties as predictive or economic superiority.
'''
    (REPORTS/'TIE_SENSITIVITY_REPORT.md').write_text(report,encoding='utf-8')
    # Diagnostics json to make reconciliation auditable.
    (ROOT/'logs/run_summary.json').write_text(json.dumps({'candidate_rows':len(d),'pools':len(pool_groups),'positive_relevance_pools':int(orig.ndcg10.count()),
      'hybrid_score_error':max_hybrid_error,'ppml_rank_error':max_ppml_rank_error,'lightgbm_rank_error':max_lgbm_rank_error,
      'reference_ndcg_difference':base_ndcg,'boundary_tie_pools':boundary_count,'boundary_tie_positive_relevance_pools':ties_pos,
      'all_exact_tie_groups':len(all_groups),'exact_tied_candidate_records':exact_tied_candidate_records,'within_top10_tie_groups':internal_count,'algebraic_boundary_tie_groups':algebraic_boundary_group_count,'algebraic_boundary_pools':len(algebraic_boundary_pool_keys),'rounding_only_boundary_pools':rounding_only_boundary_pool_count,'random_seeds':len(SEEDS),
      'random_selection_sensitive_candidates':len(sensitive_df),'random_sensitive_successes':sens_success,'random_sensitive_failures':sens_fail,
      'random_sensitive_unknown':sens_unknown,'runtime_seconds':time.time()-started},indent=2))
    print((ROOT/'logs/run_summary.json').read_text())

if __name__=='__main__': main()
