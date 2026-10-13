"""Independent aggregate-only verification of existing published predictions, no fitting."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import brier_score_loss,log_loss,average_precision_score,roc_auc_score
from research_paths import resolve_paths
SOURCE, OUTPUT, REPOSITORY = resolve_paths()
sys.path.insert(0,str(REPOSITORY))
from careaccess.data import prepare_source,temporal_split
from careaccess.validation import make_features
from careaccess.inference import load_bundle
from careaccess.network import TabularMLP
f,_=prepare_source(SOURCE)
parts=temporal_split(f);d=parts['test'];y=d.target.to_numpy();dates=pd.to_datetime(d.appointment_datetime).dt.normalize()
b=load_bundle();z=b['preprocessor'].transform(make_features(d)).astype('float32');predictions={}
for name,m in b['models'].items():
    if isinstance(m,TabularMLP):
        with torch.no_grad(): p=torch.sigmoid(m(torch.tensor(z))).numpy().astype(float)
    else: p=m.predict_proba(z)[:,1]
    c=b['manifest']['calibrators'][name]
    if c:
        p=np.clip(p,1e-6,1-1e-6);p=1/(1+np.exp(-(c['coef']*np.log(p/(1-p))+c['intercept'])))
    predictions[name]=p
out={'scope':'Independent fixed-artifact verification; retrospective already-reported 2021–2022 cohort. No new model fit or selection.', 'seed':20261013, 'bootstrap_replicates':5000, 'baselines':{}, 'capacity':{}, 'test':{'n':len(d),'missed':int(y.sum()),'days':int(dates.nunique())}}
for label,value in [('training_prevalence',parts['train'].target.mean()),('calibration_prevalence',parts['calibration'].target.mean()),('test_prevalence_oracle_not_deployable',y.mean())]:
    p=np.full(len(y),value)
    out['baselines'][label]={'constant_probability':float(value),'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p))}
for name,p in predictions.items():
    tab=pd.DataFrame({'date':dates,'y':y,'p':p,'tie':np.random.default_rng(20261013).random(len(y))})
    stats=[]
    for k in (5,10,20):
        rows=[]
        for date,g in tab.groupby('date'):
            n=len(g);selected=g.sort_values(['p','tie'],ascending=[False,True],kind='stable').head(k)
            rows.append({'date':date,'n':n,'m':len(selected),'hits':int(selected.y.sum()),'expected':min(k,n)*g.y.mean()})
        days=pd.DataFrame(rows); delta=(days.hits-days.expected).to_numpy();m=days.m.to_numpy();n=len(days)
        rng=np.random.default_rng(20261013);ix=rng.integers(0,n,size=(5000,n));gain=delta[ix].sum(axis=1)/m[ix].sum(axis=1)
        # Calendar-week clusters retain all available days in a week. Weeks sampled independently.
        days['week']=days.date.dt.to_period('W-SUN').dt.start_time
        w=days.groupby('week')[['m','hits','expected']].sum();wi=rng.integers(0,len(w),size=(5000,len(w)))
        wd=(w.hits-w.expected).to_numpy();wm=w.m.to_numpy();wg=wd[wi].sum(axis=1)/wm[wi].sum(axis=1)
        stats.append({'k':k,'selected':int(m.sum()),'captured':int(days.hits.sum()),'random_expected_captured':float(days.expected.sum()),'incremental_captures':float(delta.sum()),'precision':float(days.hits.sum()/m.sum()),'random_expected_precision':float(days.expected.sum()/m.sum()),'precision_difference':float(delta.sum()/m.sum()),'date_cluster_95_interval':np.quantile(gain,[.025,.975]).tolist(),'week_cluster_95_interval':np.quantile(wg,[.025,.975]).tolist(),'days':n,'weeks':len(w),'capacity_constrained_days':int((days.n>k).sum()),'unconstrained_days':int((days.n<=k).sum())})
    out['capacity'][name]=stats
    out.setdefault('metrics',{})[name]={'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p)),'roc_auc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p))}
path=OUTPUT/'frozen_operational_check.json';path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
