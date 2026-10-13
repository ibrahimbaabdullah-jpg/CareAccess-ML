"""Real CPU inference from checked, bundled trained artifacts only."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import skops.io as sio
from safetensors.torch import load_file
from careaccess.network import TabularMLP
from careaccess.validation import validate_frame,make_features
ROOT=Path(__file__).resolve().parents[1]


def model_names(bundle=None):
    bundle=load_bundle() if bundle is None else bundle
    return list(bundle['manifest']['model_files'])


def load_bundle():
    folder=ROOT/'models';manifest=json.loads((folder/'manifest.json').read_text());models={}
    for name,digest in manifest['sha256'].items():
        if Path(name).name!=name:raise ValueError('Invalid bundled artifact name.')
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise ValueError('Model checksum mismatch; deployment must be repaired.')
    def read_skops(name):
        unknown=sio.get_untrusted_types(file=folder/name)
        # Only audited sklearn/numpy types from our own committed manifest, no user-uploaded models.
        if unknown!=manifest['skops_types'].get(name,[]):raise ValueError('Unexpected model types.')
        if any(not x.startswith(('sklearn.','numpy.')) for x in unknown):raise ValueError('Disallowed serialized model type.')
        return sio.load(folder/name,trusted=unknown)
    pre=read_skops('preprocessor.skops')
    for label,name in manifest['model_files'].items():
        if name.endswith('.safetensors'):
            model=TabularMLP(manifest['input_dim']);model.load_state_dict(load_file(str(folder/name),device='cpu'));model.eval()
        else:model=read_skops(name)
        models[label]=model
    torch.set_num_threads(2)
    return {'manifest':manifest,'models':models,'preprocessor':pre,'default_model':manifest['default_model'],
            'feature_contract':manifest['feature_contract'],'report':json.loads((ROOT/'reports/evaluation.json').read_text())}


def predict(frame,model_name=None,bundle=None):
    bundle=load_bundle() if bundle is None else bundle
    clean,errors=validate_frame(frame)
    if len(errors):raise ValueError('Invalid appointment input: '+ '; '.join(errors.message.unique()))
    name=model_name or bundle['default_model']
    if name not in bundle['models']:raise ValueError('Unknown model.')
    if len(clean)==0:return np.array([],dtype=float)
    x=bundle['preprocessor'].transform(make_features(clean)).astype('float32');model=bundle['models'][name]
    if isinstance(model,TabularMLP):
        with torch.no_grad():p=torch.sigmoid(model(torch.tensor(x))).numpy().astype(float)
    else:p=model.predict_proba(x)[:,1]
    cal=bundle['manifest']['calibrators'][name]
    if cal:
        p=np.clip(p,1e-6,1-1e-6);p=1/(1+np.exp(-(cal['coef']*np.log(p/(1-p))+cal['intercept'])))
    if not np.isfinite(p).all():raise ValueError('Non-finite model result.')
    return np.asarray(p,dtype=float)
