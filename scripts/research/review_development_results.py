"""Independent capacity recomputation from private out-of-fold scores; no training."""
from pathlib import Path
import numpy as np,pandas as pd,json,hashlib
from research_paths import resolve_paths
SOURCE, R, REPOSITORY = resolve_paths()
a=np.load(R/'private_development_predictions.npz');stored=json.loads((R/'research_results.json').read_text());selection=stored['selected']
def stats(name):
    d=pd.DataFrame({'date':pd.to_datetime(a['date'],unit='us'),'y':a['y'],'p':a[name],'tie':np.random.default_rng(20261013).random(len(a['y']))})
    s=d.sort_values(['date','p','tie'],ascending=[True,False,True],kind='stable').groupby('date').head(20)
    z=d.groupby('date').agg(n=('y','size'),events=('y','sum'));z['slots']=z.n.clip(upper=20);z['expected']=z.slots*z.events/z.n;z['hits']=s.groupby('date').y.sum();return z
sel=stats(selection['selected']);base=stats(selection['best_simple']);orig=stats('original_logistic');result={'selected':selection['selected'],'independent_recomputed':{},'comparisons':{}}
for name in [selection['selected'],selection['best_simple'],'original_logistic']:
    z=stats(name);actual={'selected':int(z.slots.sum()),'captured':int(z.hits.sum()),'random_expected_captured':float(z.expected.sum()),'precision_difference':float((z.hits.sum()-z.expected.sum())/z.slots.sum())};expected=stored['development_results'][name]['pooled']['capacity']['20'];assert all(np.isclose(actual[k],expected[k],rtol=0,atol=1e-12) for k in actual);result['independent_recomputed'][name]=actual
for name,b in [('matched_random',None),(selection['best_simple'],base),('original_logistic',orig)]:
    out=[];d=(sel.hits-(sel.expected if b is None else b.hits)).to_numpy();m=sel.slots.to_numpy();rng=np.random.default_rng(20261013)
    for i in range(2000):
        ix=rng.integers(0,len(sel),len(sel));out.append(d[ix].sum()/m[ix].sum())
    actual={'estimate':float(d.sum()/m.sum()),'lower95':float(np.quantile(out,.025)),'upper95':float(np.quantile(out,.975))}
    expected=selection['paired_daily_k20_precision_intervals'][name];assert all(np.isclose(actual[k],expected[k],rtol=0,atol=1e-12) for k in actual)
    # Descriptive weekly-cluster sensitivity, explicitly no reselection or changed gate.
    week=pd.DataFrame({'delta':d,'slots':m,'week':sel.index.to_period('W-SUN').start_time}).groupby('week')[['delta','slots']].sum();wi=rng.integers(0,len(week),size=(5000,len(week)));wg=week.delta.to_numpy()[wi].sum(axis=1)/week.slots.to_numpy()[wi].sum(axis=1)
    actual['calendar_week_cluster_95_sensitivity']=np.quantile(wg,[.025,.975]).tolist();result['comparisons'][name]=actual
result['verified']=True;result['scope']='Arithmetic and paired bootstrap of pre-selected candidate reproduced, not independent validation or selection adjustment.'
(R/'development_independent_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
