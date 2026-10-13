"""Isolated process resource measurements; exploratory models are not serialized."""
import json,sys,time,resource,subprocess,os,platform
import numpy as np,pandas as pd
import run_research as rr

def child(name):
    f=rr.load_development();tr=f[f.date<'2020-01-01'];spec=next(s for s in rr.CANDIDATES if s['name']==name)
    # Actual-source adult appointment rows stay private in memory. No invented input values.
    timing_rows=tr.loc[tr.age.ge(18)&tr.age.le(110)].head(1000).copy()
    assert len(timing_rows)==1000 and timing_rows.age.ge(18).all()
    sample=timing_rows.head(1).copy()
    before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;t=time.perf_counter()
    if 'recent_days' in spec:tr=tr[tr.date>=pd.Timestamp('2020-01-01')-pd.Timedelta(days=spec['recent_days'])]
    if spec['kind']=='earliest':predict=lambda v:-v.hour.to_numpy()
    elif spec['kind']=='group':
        cols=spec['cols'];prior=tr.y.mean();agg=tr.groupby(cols,dropna=False).y.agg(['sum','size']);risk=(agg['sum']+30*prior)/(agg['size']+30)
        if len(cols)==1:predict=lambda v:v[cols[0]].map(risk).fillna(prior).to_numpy()
        else:predict=lambda v:v[cols].merge(risk.rename('p').reset_index(),on=cols,how='left',sort=False).p.fillna(prior).to_numpy()
    else:
        model=rr.model_for(spec);model.fit(tr,tr.y);predict=lambda v:model.predict_proba(v)[:,1]
    elapsed=time.perf_counter()-t;after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    speed={}
    for n in [1,1000]:
        x=timing_rows.head(n).copy();predict(x);times=[]
        for _ in range(20):
            t=time.perf_counter();predict(x);times.append(time.perf_counter()-t)
        speed[str(n)]={'median_milliseconds':float(np.median(times)*1000),'p95_milliseconds':float(np.quantile(times,.95)*1000),'repeats':20}
    print(json.dumps({'model':name,'fit_rows':len(tr),'timing_input_source':'Actual Mendeley development appointments, age>=18, chronological first1000; single input is first of these. No rows exported.','timing_input_rows':1000,'fit_wall_seconds':elapsed,'process_peak_rss_MiB_at_fit':after/1024,'peak_rss_MiB_before_fit':before/1024,'inference':speed}))

if __name__=='__main__':
    if len(sys.argv)>1:child(sys.argv[1])
    else:
        out={'method':'Each candidate in a new process; peak RSS includes Python/sklearn imports, private source parsing and feature preparation, not just estimator. Linux ru_maxrss converted KiB toMiB. Uncalibrated inference includes preprocessing; actual-source adult development rows only, private in memory. Timing inputs may overlap training and are not a predictive evaluation. No synthetic records are used. Median20 repetitions after warmup. A new comparison, not historical baseline training telemetry.','python':platform.python_version(),'machine':platform.machine(),'cpu_count_visible':os.cpu_count(),'threads_per_estimator':2,'results':[]}
        for s in rr.CANDIDATES:
            p=subprocess.run([sys.executable,__file__,s['name']],capture_output=True,text=True,check=True);r=json.loads(p.stdout);out['results'].append(r);print(p.stdout.strip(),flush=True)
        rr.save('resource_measurements.json',out)
