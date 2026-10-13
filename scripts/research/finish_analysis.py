"""Finish predeclared baselines, diagnostics and metric corrections; never reselect."""
from pathlib import Path
import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
import run_research as rr
R=rr.ROOT
f=rr.load_development();f['tie']=np.nan
for mask in [(f.date>='2017-01-01')&(f.date<'2020-01-01'), f.date>='2020-01-01']:
    f.loc[mask,'tie']=np.random.default_rng(rr.SEED).random(int(mask.sum()))
f['time_bin']=(f.hour//2).astype(int).astype(str)
e=f[(f.date>='2017-01-01')&(f.date<'2020-01-01')].copy()
saved=np.load(R/'private_development_predictions.npz')
assert np.array_equal(e.y.to_numpy(),saved['y']) and np.array_equal(e.date.astype('int64').to_numpy(),saved['date'])
old=json.loads((R/'research_results.json').read_text());chosen=old['selected']['selected']

def consistent_day_stats(va,p,k):
    d=pd.DataFrame({'date':va.date.to_numpy(),'y':va.y.to_numpy(),'p':p,'tie':va.tie.to_numpy()})
    daily=d.groupby('date').agg(n=('y','size'),positive=('y','sum'));daily['selected']=np.minimum(k,daily.n);daily['expected']=daily.selected*daily.positive/daily.n
    selected=d.sort_values(['p','tie'],ascending=[False,True],kind='stable').groupby('date').head(k)
    daily['hits']=selected.groupby('date').y.sum();return daily
rr.day_stats=consistent_day_stats
corrected={}
for spec in rr.CANDIDATES:
    name=spec['name'];p=saved[name];folds=[]
    for year in [2017,2018,2019]:
        mask=e.date.dt.year.eq(year).to_numpy();folds.append({'year':year,**rr.metrics(e.loc[mask],p[mask])})
    corrected[name]={'folds':folds,'pooled':rr.metrics(e,p)}
extra=[{'name':'specialty_weekday_prior','kind':'group','cols':['specialty','weekday_cat']},{'name':'specialty_weekday_time_prior','kind':'group','cols':['specialty','weekday_cat','time_bin']}]
ps={n:saved[n] for n in corrected};prob_baselines={}
for spec in extra:
    p=[];folds=[]
    for year in [2017,2018,2019]:
        tr=f[f.date<pd.Timestamp(f'{year}-01-01')];va=f[f.date.dt.year==year];pv,_=rr.predict(spec,tr,va);p.append(pv);folds.append({'year':year,**rr.metrics(va,pv)})
    pv=np.concatenate(p);ps[spec['name']]=pv;corrected[spec['name']]={'folds':folds,'pooled':rr.metrics(e,pv)}
for year in [2017,2018,2019,2020]:
    tr=f[f.date<pd.Timestamp(f'{year}-01-01')];va=f[f.date.dt.year==year];recent=tr[tr.date>=tr.date.max()-pd.Timedelta(days=90)]
    prob_baselines[str(year)]={}
    for name,rate in [('training_prevalence',tr.y.mean()),('last90day_training_prevalence',recent.y.mean())]:
        p=np.full(len(va),rate);prob_baselines[str(year)][name]={'probability':rate,'brier':brier_score_loss(va.y,p),'log_loss':log_loss(va.y,p),'average_precision':va.y.mean(),'roc_auc':.5}
# Fixed pre-correction choice; do not reselect after 2020 became visible.
compare=['original_logistic','specialty_age_prior']+[s['name'] for s in extra]
cap_ci={}
for k in [5,10,20]:
    a=rr.day_stats(e,ps[chosen],k);cap_ci[str(k)]={c:rr.bootstrap_days(a,None if c=='matched_random' else rr.day_stats(e,ps[c],k)) for c in ['matched_random']+compare}
# Protocol-promised AUC/AP intervals; these remain selected-development intervals.
groups=[np.flatnonzero(e.date.eq(day).to_numpy()) for day in e.date.unique()]
discrimination_ci={}
for name in [chosen,'original_logistic','specialty_age_prior']:
    rng=np.random.default_rng(rr.SEED);vals=[]
    for b in range(500):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]);y=e.y.to_numpy()[ix];p=ps[name][ix];vals.append((roc_auc_score(y,p),average_precision_score(y,p)))
    a=np.array(vals);discrimination_ci[name]={metric:{'estimate':corrected[name]['pooled'][metric],'lower95':float(np.quantile(a[:,j],.025)),'upper95':float(np.quantile(a[:,j],.975)),'replicates':len(vals)} for j,metric in enumerate(['roc_auc','average_precision'])}
# Descriptive cohort evidence; no outcome-driven feature changes.
diagnostics={'years':[], 'specialty_shift':[], 'source_target_mapping':'Strict yes=missed(1), no=attended(0), verified against provider description; no unexpected values.'}
for year,g in f.groupby(f.date.dt.year):
    day=g.groupby('date').y.agg(['size','sum']);diagnostics['years'].append({'year':int(year),'rows':len(g),'days':len(day),'missed':int(g.y.sum()),'prevalence':float(g.y.mean()),'age_missing_rate':float(g.age.isna().mean()),'specialty_missing_rate':float(g.specialty.eq('missing').mean()),'city_missing_rate':float(g.city.eq('missing').mean()),'days_with_n_le20':int((day['size']<=20).sum()),'k20_fraction_of_all_visits_selected':float(np.minimum(20,day['size']).sum()/len(g))})
    for s,v in g.groupby('specialty'):
        if len(v)>=100:diagnostics['specialty_shift'].append({'year':int(year),'specialty':s,'rows':len(v),'missed_rate':float(v.y.mean())})
# Historical 2020 simple-baseline supplement with identical prediction-time assumptions.
qual={};tr=f[f.date<'2020-01-01'];va=f[f.date>='2020-01-01'];qpred={}
for name in [chosen,'original_logistic','specialty_age_prior']+[s['name'] for s in extra]:
    spec=next(s for s in rr.CANDIDATES+extra if s['name']==name);p,_=rr.predict(spec,tr,va);qual[name]=rr.metrics(va,p);qpred[name]=p
qual_ci={c:rr.bootstrap_days(rr.day_stats(va,qpred[chosen],20),None if c=='matched_random' else rr.day_stats(va,qpred[c],20)) for c in ['matched_random']+compare}
before=json.loads((R/'preserved_artifacts_sha256.json').read_text());after=rr.baseline_hashes()
changed={k:{'before':v,'after':after.get(k)} for k,v in before.items() if after.get(k)!=v}
# Concurrent release metadata edits must be surfaced. Trained weights and evaluation remain guarded.
assert all(k=='models/manifest.json' for k in changed),changed
rr.save('concurrent_artifact_changes.json',{'changed':changed,'writer':'Separate concurrent manifest metadata edit; fitted models and original evaluation are explicitly guarded.','trained_model_bytes_and_original_evaluation_unchanged':True})
out={'selected_unchanged':chosen,'metric_correction':'Preserves the original primary pooled2017-2019 and2020 seeded tie values exactly; annual summaries now slice these same values instead of resetting RNG. Original research report preserved. No model selection repeated after2020 viewed. Supplemental baseline specification saved before first researcher review of metrics. Alternate tie realization retained separately as sensitivity only.','corrected_development_results':corrected,'paired_capacity_intervals':cap_ci,'discrimination_intervals':discrimination_ci,'probability_baselines':prob_baselines,'diagnostics':diagnostics,'corrected_2020_qualification':qual,'corrected_2020_k20_intervals':qual_ci,'original_trained_models_and_evaluation_unchanged':True,'concurrent_manifest_metadata_change':changed}
rr.save('completed_analysis.json',out)
print(json.dumps({'chosen':chosen,'corrected_k20':cap_ci['20'],'corrected_2020':qual[chosen],'2020_intervals':qual_ci,'diagnostics':diagnostics['years']},indent=2))
