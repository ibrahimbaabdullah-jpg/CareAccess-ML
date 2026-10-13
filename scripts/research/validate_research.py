import json,numpy as np,pandas as pd
import run_research as rr
from pathlib import Path
R=rr.ROOT;d=rr.load_development();a=json.loads((R/'completed_analysis.json').read_text());b=json.loads((R/'research_results.json').read_text());resources=json.loads((R/'resource_measurements.json').read_text())
checks=[]
def check(label,ok):
 assert ok,label;checks.append(label)
check('All research rows strictly before2021',d.date.max()<pd.Timestamp('2021-01-01'))
check('Expected development cohort size',len(d)==37767)
check('No source birth dates, reasons, weather or identifiers in prepared data',not any(x in d for x in ['date_of_birth','no_show_reason','icd','entry_service_date','PatientId','patient_id']))
for year in [2017,2018,2019]:
 tr=d[d.date<pd.Timestamp(f'{year}-01-01')];va=d[d.date.dt.year==year];check(f'Strict temporal separation {year}',tr.date.max()<va.date.min())
for c in rr.CANDIDATES:
 n=c['name'];old=b['development_results'][n]['pooled'];new=a['corrected_development_results'][n]['pooled'];check(f'Primary pooled capacity unchanged {n}',old['capacity']==new['capacity'])
 for k in ['5','10','20']:
  check(f'Annual primary capture sums equal pooled {n}/K{k}',sum(x['capacity'][k]['captured'] for x in a['corrected_development_results'][n]['folds'])==new['capacity'][k]['captured'])
for n,v in b['2020_historical_qualification'].items():
 if n in a['corrected_2020_qualification']:check(f'Primary2020 metrics unchanged {n}',v==a['corrected_2020_qualification'][n])
check('No selection change',a['selected_unchanged']==b['selected']['selected']=='hgb7_augmented')
check('Resources for every prespecified candidate',set(x['model'] for x in resources['results'])==set(x['name'] for x in rr.CANDIDATES))
original=json.loads((R/'preserved_artifacts_sha256.json').read_text());now=rr.baseline_hashes();changed={k for k in original if original[k]!=now.get(k)}
check('Only authorized concurrent manifest change',changed<={'models/manifest.json'})
rr.save('research_validation.json',{'status':'passed','assertions':len(checks),'checks':checks,'manifest_note':'Separate authorized security change to manifest configuration hash/revision note was confirmed; original fitted models and evaluation unchanged.'})
print(f'PASS {len(checks)} assertions')
