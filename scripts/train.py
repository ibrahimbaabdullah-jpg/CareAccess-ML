"""Reproduce frozen chronological benchmark. Run from repository root."""
import argparse,copy,hashlib,json,platform,sys,time
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import sklearn,torch,skops.io as sio
from safetensors.torch import save_file
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score,log_loss,f1_score
from sklearn.inspection import permutation_importance
from careaccess.data import prepare_source,temporal_split,SOURCE_SHA256
from careaccess.validation import make_features,INPUT_COLUMNS,SPECIALTIES
from careaccess.network import TabularMLP
from careaccess.evaluation import evaluate,curves,capacity_metrics,bootstrap_intervals

SEED=20261013
ROOT=Path(__file__).resolve().parents[1]

def save_json(path,obj): path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def main(raw):
    started=time.monotonic();np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    frame,counts=prepare_source(raw);parts=temporal_split(frame)
    # Split boundaries are fixed in source code before fitting; final test is evaluated only after selection.
    x={k:make_features(v) for k,v in parts.items()};y={k:v.target.to_numpy(dtype=int) for k,v in parts.items()}
    pre=ColumnTransformer([('numeric',Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),['age','hour','weekday','month']),
                           ('specialty',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['specialty'])])
    z={'train':pre.fit_transform(x['train']).astype('float32')}
    z.update({k:pre.transform(x[k]).astype('float32') for k in ['validation','calibration','test']})
    candidates={'Constant baseline':DummyClassifier(strategy='prior'),
      'Logistic Regression':LogisticRegression(C=1,max_iter=1000,random_state=SEED),
      'Gradient Boosting':HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=2,early_stopping=False,random_state=SEED)}
    rawp={};fit_times={}
    for name,model in candidates.items():
        t=time.monotonic();model.fit(z['train'],y['train']);fit_times[name]=time.monotonic()-t
        rawp[name]={k:model.predict_proba(z[k])[:,1] for k in ['validation','calibration']}
    net=TabularMLP(z['train'].shape[1]);optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    tx=torch.tensor(z['train']);ty=torch.tensor(y['train'],dtype=torch.float32)
    vx=torch.tensor(z['validation']);loss_fn=torch.nn.BCEWithLogitsLoss();best=float('inf');beststate=None;patience=0;history=[];t=time.monotonic()
    for epoch in range(1,41):
        net.train();order=torch.randperm(len(tx));losses=[]
        for ix in order.split(256):
            optimizer.zero_grad();loss=loss_fn(net(tx[ix]),ty[ix]);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        net.eval()
        with torch.no_grad():vp=torch.sigmoid(net(vx)).numpy()
        vl=float(log_loss(y['validation'],vp));history.append({'epoch':epoch,'train_bce':float(np.mean(losses)),'validation_log_loss':vl})
        if vl<best-1e-5:best=vl;beststate=copy.deepcopy(net.state_dict());patience=0
        else:patience+=1
        if patience>=6:break
    net.load_state_dict(beststate);net.eval();fit_times['PyTorch Neural Network']=time.monotonic()-t
    rawp['PyTorch Neural Network']={}
    with torch.no_grad():
        for k in ['validation','calibration']:rawp['PyTorch Neural Network'][k]=torch.sigmoid(net(torch.tensor(z[k]))).numpy().astype(float)
    validation=[]
    for name,p in rawp.items():validation.append({'model':name,'average_precision':float(average_precision_score(y['validation'],p['validation'])),'log_loss':float(log_loss(y['validation'],p['validation']))})
    eligible=[v for v in validation if v['model']!='Constant baseline']
    default=sorted(eligible,key=lambda v:(-v['average_precision'],v['log_loss'],v['model']))[0]['model']
    # Default chosen solely by validation AP; no final-test selection or refitting.
    calibrators={};thresholds={}
    for name,p in rawp.items():
        if name=='Constant baseline':calibrators[name]=None;cp=p['calibration']
        else:
            logit=np.log(np.clip(p['calibration'],1e-6,1-1e-6)/(1-np.clip(p['calibration'],1e-6,1-1e-6))).reshape(-1,1)
            cal=LogisticRegression(C=1,max_iter=1000,random_state=SEED).fit(logit,y['calibration'])
            calibrators[name]={'coef':float(cal.coef_[0,0]),'intercept':float(cal.intercept_[0])}
            cp=cal.predict_proba(logit)[:,1]
        grid=np.linspace(.05,.95,91);thresholds[name]=float(max(grid,key=lambda t:f1_score(y['calibration'],cp>=t,zero_division=0)))
    models=ROOT/'models';models.mkdir(exist_ok=True)
    sio.dump(pre,models/'preprocessor.skops');files={};slugs={'Constant baseline':'baseline','Logistic Regression':'logistic','Gradient Boosting':'gradient','PyTorch Neural Network':'neural'}
    for name,model in candidates.items():
        filename=slugs[name]+'.skops';sio.dump(model,models/filename);files[name]=filename
    save_file(net.state_dict(),str(models/'neural.safetensors'));files['PyTorch Neural Network']='neural.safetensors'
    report={'source':{'title':'Medical Appointments No-Show','author':'Luiz Henrique Salazar','doi':'10.17632/wm6w2fvkfj.1','version':1,'url':'https://data.mendeley.com/datasets/wm6w2fvkfj/1','license':'CC BY 4.0','sha256':SOURCE_SHA256},
      'counts':counts,'splits':{k:{'rows':len(v),'missed':int(v.target.sum()),'start':v.appointment_datetime.min(),'end':v.appointment_datetime.max()} for k,v in parts.items()},
      'selection':{'default_model':default,'rule':'Highest 2019 validation average precision among the three trained candidates; tie break log loss, then name. No final-test results used.','validation':validation},
      'metrics':[],'curves':{},'capacity':{},'subgroups':[],'uncertainty':{},'fit_seconds':fit_times,'neural_history':history,
      'limitations':['Single historical Brazilian rehabilitation service; not locally validated elsewhere.','No patient identifiers: repeated persons may span periods; unseen-patient generalization and patient-cluster intervals cannot be established.','Exact full-row duplicates removed; some identical distinct encounters may have been collapsed.','Age recomputed from birth date; provided raw age and all age flags excluded. Birth dates are never published.','Appointment snapshot availability is assumed, not proven by versioned booking records.','Outcome prevalence and service activity changed over time, including the pandemic.','Calibration uses only 827 appointments and 117 missed visits from sparse 2020 data.','Date-cluster confidence intervals do not account fully for repeated patients.','Predicted missed attendance is not predicted benefit from outreach. No causal or prospective reduction in no-shows is established.','Public application requires authorized nonidentifying inputs; identifiable health information must not be uploaded.','Probabilities outside historical support or in a new institution may be unreliable.','Daily top-K evaluation counts potential missed visits identified, not prevented.']}
    predictions={}
    for name in rawp:
        if name=='PyTorch Neural Network':
            with torch.no_grad():p=torch.sigmoid(net(torch.tensor(z['test']))).numpy().astype(float)
        else:p=candidates[name].predict_proba(z['test'])[:,1]
        cal=calibrators[name]
        if cal:
            p=np.clip(p,1e-6,1-1e-6);p=1/(1+np.exp(-(cal['coef']*np.log(p/(1-p))+cal['intercept'])))
        predictions[name]=p
        report['metrics'].append({'model':name,**evaluate(y['test'],p,thresholds[name])})
        report['curves'][name]=curves(y['test'],p)
        report['capacity'][name]=capacity_metrics(y['test'],p,parts['test'].appointment_datetime)
        report['uncertainty'][name]=bootstrap_intervals(y['test'],p,parts['test'].appointment_datetime)
        for groupcol,groups in [('gender',parts['test'].gender_audit),('age',pd.cut(parts['test'].age,[-.01,18,60,111],right=False,labels=['Under 18','18–59','60+']).astype(object).fillna('Unknown'))]:
            for group in pd.unique(groups):
                mask=np.asarray(groups==group);gy=y['test'][mask]
                if mask.sum()<100 or gy.sum()<20 or (len(gy)-gy.sum())<20: continue
                report['subgroups'].append({'model':name,'dimension':groupcol,'group':str(group),'rows':int(mask.sum()),'missed':int(gy.sum()),**evaluate(gy,p[mask],thresholds[name])})
    # Feature perturbation is validation-only, descriptive association rather than causal explanation.
    if default in candidates:
        pi=permutation_importance(candidates[default],z['validation'],y['validation'],scoring='average_precision',n_repeats=5,random_state=SEED)
        report['feature_importance']=[{'feature':str(n),'mean_ap_decrease':float(m),'std':float(s)} for n,m,s in zip(pre.get_feature_names_out(),pi.importances_mean,pi.importances_std)]
    else:report['feature_importance']=[]
    versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__,'torch':torch.__version__}
    manifest={'version':'0.1.0','created_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,'default_model':default,'model_files':files,'calibrators':calibrators,'thresholds':thresholds,'input_dim':int(z['train'].shape[1]),'versions':versions,'sha256':{},'skops_types':{}}
    for path in models.iterdir():
        if path.suffix in ['.skops','.safetensors']:
            manifest['sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            if path.suffix=='.skops':manifest['skops_types'][path.name]=sio.get_untrusted_types(file=path)
    manifest['feature_contract']={'input_columns':INPUT_COLUMNS,'specialty_values':SPECIALTIES,'age_min':0,'age_max':110,'age_optional':True,'specialty_optional':True,'datetime_format':'YYYY-MM-DDTHH:MM','train_start':parts['train'].appointment_datetime.min(),'train_end':parts['train'].appointment_datetime.max(),'data_start':frame.appointment_datetime.min(),'data_end':frame.appointment_datetime.max(),'description':'Age at appointment, specialty and scheduled calendar/time features only. Blank age imputed using training median with missingness indicator; blank specialty gets a missing category.'}
    report['training_seconds']=time.monotonic()-started;report['versions']=versions;report['created_utc']=manifest['created_utc'];report['default_model']=default
    manifest['configuration_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['scripts/train.py','careaccess/data.py','careaccess/validation.py','scripts/feature_ablation.py','careaccess/evaluation.py']}
    report['audit_order']='An initial automated fixed-config run generated test metrics before the additional-input clarification. Subsequent disability/city sensitivity analysis used only training/2019 validation and did not change the original features or selection rule. Final-test results were not used for feature, model, calibration or threshold selection. A later verification run repeated the unchanged fit and corrected only label-independent ties in capacity evaluation.'
    ablation=ROOT/'reports/feature_ablation.json'
    if ablation.exists():report['feature_ablation']=json.loads(ablation.read_text())
    save_json(models/'manifest.json',manifest);save_json(ROOT/'reports/evaluation.json',report)
    print(json.dumps({'default_model':default,'metrics':report['metrics'],'seconds':report['training_seconds']},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True);args=parser.parse_args();main(args.data)
