"""Label-independent randomized-tie policy expectation; diagnostic, no reselection."""
import json, numpy as np,pandas as pd
import run_research as rr
f=rr.load_development();e=f[(f.date>='2017-01-01')&(f.date<'2020-01-01')];z=np.load(rr.ROOT/'private_development_predictions.npz');q=f[f.date>='2020-01-01'];tr=f[f.date<'2020-01-01']

def expected_hits(va,p,k=20):
 d=pd.DataFrame({'date':va.date.to_numpy(),'y':va.y.to_numpy(),'p':p});out=[]
 for day,g in d.groupby('date'):
  n=min(k,len(g));cutoff=g.p.sort_values(ascending=False).iloc[n-1];hi=g[g.p>cutoff];tie=g[g.p==cutoff];h=hi.y.sum()+(n-len(hi))*tie.y.mean();out.append({'date':day,'selected':n,'hits':h,'expected':n*g.y.mean()})
 return pd.DataFrame(out).set_index('date')
names=['hgb7_augmented','hgb7_basic','original_logistic','specialty_age_prior'];out={}
for period,va in [('2017_2019',e),('2020',q)]:
 ds={}
 for n in names:
  if period=='2017_2019':p=z[n]
  else:p,_=rr.predict(next(s for s in rr.CANDIDATES if s['name']==n),tr,va)
  ds[n]=expected_hits(va,p)
 out[period]={'expected_policy_captures':{n:float(d.hits.sum()) for n,d in ds.items()},'selected_slots':int(ds[names[0]].selected.sum()),'matched_random_expected':float(ds[names[0]].expected.sum()),'paired_intervals':{n:rr.bootstrap_days(ds[names[0]],None if n=='matched_random' else ds[n]) for n in ['matched_random','hgb7_basic','original_logistic','specialty_age_prior']}}
rr.save('tie_policy_expectation.json',{'method':'Fractional expected captures when appointments with exactly equal risk at the cutoff receive uniformly random tie priority. Evaluation expectation only, not outcome-aware selection. Original learned candidate choice unchanged. Daily cluster resampling on fixed expected policy contributions.','results':out})
print(json.dumps(out,indent=2))
