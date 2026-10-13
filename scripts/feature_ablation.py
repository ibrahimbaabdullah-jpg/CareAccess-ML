"""Development-only necessity audit. Never reads final-test outcomes for selection."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd,numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,brier_score_loss
from careaccess.validation import make_features
raw=pd.read_csv(sys.argv[1]).drop_duplicates();date=pd.to_datetime(raw.appointment_date,format='%d/%m/%Y');raw=raw[date<'2020-01-01'].copy();date=date[date<'2020-01-01']
birth=pd.to_datetime(raw.date_of_birth,format='%d/%m/%Y',errors='coerce');age=(date-birth).dt.days/365.2425;age=age.where(age.between(0,110))
f=pd.DataFrame({'appointment_datetime':pd.to_datetime(raw.appointment_date+' '+raw.appointment_time,format='%d/%m/%Y %H:%M').dt.strftime('%Y-%m-%dT%H:%M'),'age':age,'specialty':raw.specialty});x=make_features(f)
for c in ['disability','city']:x[c]=raw[c].fillna('missing').astype(str)
y=raw.no_show.eq('yes').astype(int);train=date<'2019-01-01';val=~train;results=[]
for extras in [[],['disability'],['city'],['disability','city']]:
    pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),['age','hour','weekday','month']),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['specialty']+extras)])
    z=pre.fit_transform(x[train]);zv=pre.transform(x[val]);m=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=2,early_stopping=False,random_state=20261013).fit(z,y[train]);p=m.predict_proba(zv)[:,1]
    results.append({'additional_features':extras,'validation_average_precision':float(average_precision_score(y[val],p)),'validation_brier':float(brier_score_loss(y[val],p))})
report={'scope':'Only train<2019 and validation2019; no2020 calibration or2021–22 finaltest rows used for fitting, evaluation or decisions','policy':'Consider additional inputs necessary only if validation AP improves by >=0.01 absolute and Brier does not worsen, with defensible preappointment availability. Threshold fixed before ablation.','results':results}
Path('reports/feature_ablation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
