"""Frozen bounded development-only CareAccess model comparison; no evaluation >=2021."""
from pathlib import Path
import json, hashlib, time, os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from research_paths import resolve_paths
SOURCE, ROOT, BASE = resolve_paths()
SEED=20261013

def save(name,x): (ROOT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def baseline_hashes():return {str(p.relative_to(BASE)):sha(p) for d in ['models','reports'] for p in sorted((BASE/d).glob('*')) if p.is_file()}

def load_development():
    assert sha(SOURCE)=='903f30aa008c478dd6451550b53c2bf937653f28763ea288e43f5e99c536cbdf'
    raw=pd.read_csv(SOURCE).drop_duplicates()
    dates=pd.to_datetime(raw.appointment_date,format='%d/%m/%Y',errors='raise')
    # Discard the entire historical final cohort before any feature or outcome analysis.
    raw=raw.loc[dates<'2021-01-01'].copy();dates=dates.loc[raw.index]
    birth=pd.to_datetime(raw.date_of_birth,format='%d/%m/%Y',errors='coerce')
    age=(dates-birth).dt.days/365.2425;age=age.where(age.between(0,110))
    t=pd.to_datetime(raw.appointment_time,format='%H:%M',errors='raise')
    f=pd.DataFrame({'date':dates,'age':age,'hour':t.dt.hour+t.dt.minute/60,'weekday':dates.dt.weekday,'month':dates.dt.month,'y':raw.no_show.map({'yes':1,'no':0})})
    for c in ['specialty','city','disability']:
        f[c]=raw[c].fillna('missing').astype(str).str.strip().replace('','missing')
    f['age_band']=pd.cut(age,[-.01,6,12,18,30,45,60,111],right=False).astype(str).replace('nan','missing')
    f['weekday_cat']=f.weekday.astype(str);f['month_cat']=f.month.astype(str)
    f=f.sort_values('date',kind='stable').reset_index(drop=True)
    assert f.y.notna().all() and f.date.max()<pd.Timestamp('2021-01-01')
    return f

# These candidates and all fold boundaries are written before any research metrics.
CANDIDATES=[{'name':'specialty_prior','kind':'group','cols':['specialty']},
 {'name':'age_band_prior','kind':'group','cols':['age_band']},
 {'name':'specialty_age_prior','kind':'group','cols':['specialty','age_band']},
 {'name':'earliest_appointment','kind':'earliest'},
 {'name':'original_logistic','kind':'original'},
]
for extras in [False,True]:
    suffix='_augmented' if extras else '_basic'
    CANDIDATES += [{'name':'onehot_logistic'+suffix,'kind':'logistic','extras':extras},
                   {'name':'spline_logistic'+suffix,'kind':'spline','extras':extras}]
    for leaves in [7,15,31]:
        CANDIDATES.append({'name':f'hgb{leaves}'+suffix,'kind':'hgb','leaves':leaves,'extras':extras})
    CANDIDATES.append({'name':'random_forest'+suffix,'kind':'rf','extras':extras})
    for kind in ['logistic','hgb']:
        CANDIDATES.append({'name':kind+'_recent365'+suffix,'kind':kind,'extras':extras,'leaves':7,'recent_days':365})

PROTOCOL={
 'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),
 'purpose':'Bounded research after the original test was exposed. No independent new test claim.',
 'source_sha256':sha(SOURCE),'code_sha256':sha(Path(__file__)),
 'historical_test_policy':'All dates >=2021 removed before target/feature analysis; do not score or select using them.',
 'development_folds':[{'train_end_exclusive':f'{year}-01-01','validation_start':f'{year}-01-01','validation_end_exclusive':f'{year+1}-01-01'} for year in [2017,2018,2019]],
 'later_qualification':'After selection, evaluate selected model, original logistic and simple baselines once in 2020. Already-used calibration cohort; historical research only. No tuning after this check.',
 'candidate_models':CANDIDATES,
 'features':'Historical appointment age; scheduled time, specialty, calendar; optional city/disability assumed pre-existing. No patient linkage, DOB identifiers, raw age, outcome reason, realized weather, ICD, entry-service date or label-derived inference features.',
 'selection':'Maximize pooled daily K20 missed-visit capture above exact random expectation across 2017/18/19 out-of-fold predictions. This explicitly aligns selection with capacity. Tie-break pooled AP then candidate name. Report per-fold results; do not mistake selection-biased intervals for independent evidence.',
 'practical_development_gate':'Require date-bootstrap lower bound >0 vs exact matched random and vs best simple baseline plus positive random lift in at least two annual folds. Passing remains development evidence only.',
 'capacity':'K=5,10,20 per appointment day; label-independent stable random tie score seed20261013. Same slots/day for every model, exact random expectation uses each evaluation day outcome prevalence.',
 'uncertainty':'2000 paired date-cluster resamples for captured visits per100 outreach slots and differences; repeated persons unavailable. 500 resamples for AUC/AP. No multiple-comparison adjustment; exploratory selection.',
 'fitting':'Every imputer, category encoder, scaler, spline knot and group prior fitted only on prior dates. HGB100iters,learning_rate.05,l2=10,minleaf60; RF200trees,depth8,minleaf40. No class weighting, resampling, threshold tuning or probability calibration. Logistic C1 with max_iter2000; group priors smoothed with30 prior equivalent visits.',
 'limits':'Single retrospective clinic, no proven scheduling snapshots, no causal effect of outreach, no patient-independent validation;2020 pandemic is small and atypical.'}

def model_for(spec):
    num=['age','hour','weekday','month'];cats=['specialty']+(['city','disability'] if spec.get('extras') else [])
    kind=spec['kind']
    if kind in ['logistic','spline']:num=['age','hour'];cats+=['weekday_cat','month_cat']
    nsteps=[('impute',SimpleImputer(strategy='median',add_indicator=True))]
    if kind=='spline':nsteps.append(('spline',SplineTransformer(n_knots=5,degree=3,include_bias=False)))
    if kind in ['original','logistic','spline']:nsteps.append(('scale',StandardScaler()))
    pre=ColumnTransformer([('num',Pipeline(nsteps),num),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])
    if kind in ['original','logistic','spline']:m=LogisticRegression(C=1,max_iter=2000,random_state=SEED)
    elif kind=='hgb':m=HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_leaf_nodes=spec['leaves'],min_samples_leaf=60,l2_regularization=10,early_stopping=False,random_state=SEED)
    else:m=RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=40,max_features=.8,n_jobs=2,random_state=SEED)
    return Pipeline([('pre',pre),('model',m)])

def predict(spec,tr,va):
    if 'recent_days' in spec:tr=tr[tr.date>=va.date.min()-pd.Timedelta(days=spec['recent_days'])]
    if spec['kind']=='earliest':return -va.hour.to_numpy(),None
    if spec['kind']=='group':
        cols=spec['cols'];prior=tr.y.mean();agg=tr.groupby(cols,dropna=False).y.agg(['sum','size']);risk=(agg['sum']+30*prior)/(agg['size']+30)
        if len(cols)==1:p=va[cols[0]].map(risk).fillna(prior).to_numpy()
        else:p=va[cols].merge(risk.rename('p').reset_index(),on=cols,how='left',sort=False).p.fillna(prior).to_numpy()
        return p,None
    m=model_for(spec);m.fit(tr,tr.y);return m.predict_proba(va)[:,1],m

def day_stats(va,p,k):
    d=pd.DataFrame({'date':va.date.to_numpy(),'y':va.y.to_numpy(),'p':p,'tie':np.random.default_rng(SEED).random(len(va))})
    daily=d.groupby('date').agg(n=('y','size'),positive=('y','sum'));daily['selected']=np.minimum(k,daily.n);daily['expected']=daily.selected*daily.positive/daily.n
    selected=d.sort_values(['p','tie'],ascending=[False,True],kind='stable').groupby('date').head(k)
    daily['hits']=selected.groupby('date').y.sum();return daily

def metrics(va,p):
    y=va.y.to_numpy();r={'rows':len(y),'missed':int(y.sum()),'prevalence':float(y.mean()),'roc_auc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p))}
    if p.min()>=0 and p.max()<=1:r['brier']=float(brier_score_loss(y,p))
    r['capacity']={}
    for k in [5,10,20]:
        d=day_stats(va,p,k);s=int(d.selected.sum());h=int(d.hits.sum());e=float(d.expected.sum())
        r['capacity'][str(k)]={'days':len(d),'selected':s,'captured':h,'random_expected_captured':e,'precision':h/s,'random_precision':e/s,'precision_difference':(h-e)/s,'recall':h/y.sum(),'random_recall':e/y.sum()}
    return r

def bootstrap_days(a,b=None,n=2000):
    rng=np.random.default_rng(SEED);z=[]
    for _ in range(n):
        ix=rng.integers(0,len(a),len(a));x=a.iloc[ix];base=x.expected.to_numpy() if b is None else b.iloc[ix].hits.to_numpy()
        z.append(float((x.hits.to_numpy()-base).sum()/x.selected.sum()))
    base=a.expected.sum() if b is None else b.hits.sum()
    return {'estimate':float((a.hits.sum()-base)/a.selected.sum()),'lower95':float(np.quantile(z,.025)),'upper95':float(np.quantile(z,.975)),'replicates':n}

def main():
    t=time.time();before=baseline_hashes();save('preserved_artifacts_sha256.json',before);save('protocol.json',PROTOCOL)
    f=load_development();save('development_inventory.json',{'by_year':f.groupby(f.date.dt.year).agg(rows=('y','size'),missed=('y','sum')).reset_index().to_dict('records'),'feature_missing_by_year':{c:f.assign(missing=f[c].isna() if c=='age' else f[c].eq('missing')).groupby(f.date.dt.year).missing.mean().to_dict() for c in ['age','specialty','city','disability']}})
    folds={year:(f[f.date<pd.Timestamp(f'{year}-01-01')].copy(),f[(f.date>=pd.Timestamp(f'{year}-01-01'))&(f.date<pd.Timestamp(f'{year+1}-01-01'))].copy()) for year in [2017,2018,2019]}
    results={};predictions={};evaluation=pd.concat([folds[y][1] for y in folds],ignore_index=True)
    for spec in CANDIDATES:
        rows=[];ps=[]
        for year,(tr,va) in folds.items():
            p,_=predict(spec,tr,va);ps.append(p);rows.append({'year':year,**metrics(va,p)})
        p=np.concatenate(ps);results[spec['name']]={'folds':rows,'pooled':metrics(evaluation,p)};predictions[spec['name']]=p
        print(json.dumps({'candidate':spec['name'],'pooled_auc':results[spec['name']]['pooled']['roc_auc'],'pooled_ap':results[spec['name']]['pooled']['average_precision'],'k20_diff':results[spec['name']]['pooled']['capacity']['20']['precision_difference'],'fold_k20_diff':[r['capacity']['20']['precision_difference'] for r in rows]}),flush=True)
        save('interim_results.json',results)
    learned=[c['name'] for c in CANDIDATES if c['kind'] not in ['group','earliest']]
    selected=sorted(learned,key=lambda n:(-results[n]['pooled']['capacity']['20']['precision_difference'],-results[n]['pooled']['average_precision'],n))[0]
    simple=sorted([c['name'] for c in CANDIDATES if c['kind'] in ['group','earliest']],key=lambda n:(-results[n]['pooled']['capacity']['20']['precision_difference'],-results[n]['pooled']['average_precision'],n))[0]
    ci={};best=day_stats(evaluation,predictions[selected],20)
    for comparator in ['matched_random','original_logistic',simple]:ci[comparator]=bootstrap_days(best,None if comparator=='matched_random' else day_stats(evaluation,predictions[comparator],20))
    years_positive=sum(r['capacity']['20']['precision_difference']>0 for r in results[selected]['folds'])
    selection={'selected':selected,'best_simple':simple,'gate_passed':ci['matched_random']['lower95']>0 and ci[simple]['lower95']>0 and years_positive>=2,'positive_folds':years_positive,'paired_daily_k20_precision_intervals':ci,'warning':'Development-selection-biased intervals; not independent test evidence.'}
    save('selection.json',selection)
    # Selection written to disk before touching 2020 outcomes in fitting/evaluation.
    qual={};tr=f[f.date<'2020-01-01'];va=f[f.date>='2020-01-01']
    qnames=list(dict.fromkeys([selected,'original_logistic',simple,'specialty_prior','age_band_prior','specialty_age_prior','earliest_appointment']))
    qpred={}
    for name in qnames:
        spec=next(c for c in CANDIDATES if c['name']==name);p,_=predict(spec,tr,va);qpred[name]=p;qual[name]=metrics(va,p)
    qbest=day_stats(va,qpred[selected],20)
    qci={comp:bootstrap_days(qbest,None if comp=='matched_random' else day_stats(va,qpred[comp],20)) for comp in ['matched_random','original_logistic',simple]}
    after=baseline_hashes();assert after==before,'Original model/report artifacts changed during research.'
    out={'protocol_sha256':sha(ROOT/'protocol.json'),'selected':selection,'development_results':results,'2020_historical_qualification':qual,'2020_paired_daily_k20_precision_intervals':qci,'immutable_original_artifacts_verified':True,'seconds':time.time()-t}
    save('research_results.json',out)
    # Row-level prediction arrays are private only, never needed in published report.
    np.savez_compressed(ROOT/'private_development_predictions.npz',y=evaluation.y.to_numpy(),date=evaluation.date.astype('int64').to_numpy(),**predictions)
    print(json.dumps({'completed':True,'selected':selection,'2020':qual[selected],'2020_intervals':qci,'seconds':out['seconds']},indent=2),flush=True)

if __name__=='__main__':main()
