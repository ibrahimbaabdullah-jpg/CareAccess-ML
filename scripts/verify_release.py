"""Validate trained artifacts and release privacy; never substitute invented patient data."""
import os,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from careaccess.inference import load_bundle,predict,model_names
ROOT=Path(__file__).resolve().parents[1]
b=load_bundle()
for p in ROOT.rglob('*.csv'):
    if '.venv' in p.parts:continue
    if p.name!='appointments_template.csv' or p.parent.name!='examples':raise ValueError('Unapproved row-level CSV in release: '+str(p))
    if len(pd.read_csv(p)):raise ValueError('Input template must have zero patient rows.')
for f,expected_hash in b['manifest']['configuration_sha256'].items():
    if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=expected_hash:raise ValueError('Frozen configuration changed: '+f)
path=os.environ.get('CAREACCESS_TEST_DATA')
if path:
    frame=pd.read_csv(path);verification={}
    for name in model_names(b):
        batch=predict(frame,name,b);single=np.concatenate([predict(frame.iloc[[i]],name,b) for i in range(len(frame))]);reloaded=predict(frame,name,load_bundle())
        error=float(max(np.max(np.abs(batch-single)),np.max(np.abs(batch-reloaded))))
        if error>1e-6:raise ValueError('Numerical mismatch.')
        verification[name]={'actual_source_appointments_checked':len(frame),'maximum_absolute_error':error,'absolute_tolerance':1e-6,'passed':True}
    result={'method':'Private adult held-out source rows; batch/single/reload parity. No row inputs or predictions exported. UI parity tested separately in pytest.','models':verification}
    (ROOT/'reports/inference-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Actual-source inference parity passed for all four models; aggregate evidence saved.')
else:print('Actual-source inference parity NOT RUN: private CAREACCESS_TEST_DATA unavailable. Bundled model integrity checked.')
print('Release checks passed: checksummed trained artifacts, configuration hashes, no distributed patient/generated rows, empty CSV template.')
