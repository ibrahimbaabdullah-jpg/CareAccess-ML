"""Read-only reproducible secret and artifact inventory for all reachable Git blobs.
Never writes matching secret values; findings contain only type/blob/path/line.
No serialized models are executed by this scanner.
"""
import subprocess,json,re,io,zipfile,collections,hashlib,csv,argparse
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--git-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
GIT=['git','--git-dir='+str(args.git_dir.resolve())]
def git(*args):return subprocess.check_output(GIT+list(args))
patterns={
'private-key':rb'-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY(?: BLOCK)?-----',
'github-token':rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b',
'aws-key':rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
'google-key':rb'\bAIza[0-9A-Za-z_-]{30,}\b',
'api-key':rb'\b(?:sk-(?:proj-|ant-|svcacct-)?|gsk_)[A-Za-z0-9_-]{20,}\b',
'slack-token':rb'\bxox[baprs]-[A-Za-z0-9-]{10,}\b',
'jwt':rb'\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b',
'connection-password':rb'(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s:/]+:[^\s@]+@',
'literal-secret':rb'''(?im)(?:password|passwd|secret|api[_-]?key|access[_-]?token|authorization)\s*[:=]\s*["'][^"'\r\n]{12,}["']''',
'long-bearer':rb'(?i)\bBearer [A-Za-z0-9._-]{20,}',
}
compiled={k:re.compile(v) for k,v in patterns.items()}
commits=git('rev-list','--all').decode().splitlines(); blobs={}; trees=[]
for commit in commits:
 entries=git('ls-tree','-rz',commit).split(b'\x00')
 for entry in entries:
  if not entry: continue
  meta,path=entry.split(b'\t',1);mode,kind,sha=meta.decode().split();path=path.decode()
  if kind!='blob':continue
  obj=blobs.setdefault(sha,{'paths':set(),'commits':set(),'mode':set()})
  obj['paths'].add(path);obj['commits'].add(commit);obj['mode'].add(mode)
findings=[];archive_entries=[];inventory=[];csvs=[]
def scan(data,source,sha):
 for label,regex in compiled.items():
  for match in regex.finditer(data):
   findings.append({'type':label,'blob':sha,'source':source,'line':data[:match.start()].count(b'\n')+1})
for commit in commits:
 scan(git('cat-file','commit',commit), ['<commit-message-and-metadata>'], commit)
for sha,meta in sorted(blobs.items()):
 data=git('cat-file','blob',sha);paths=sorted(meta['paths']);scan(data,paths,sha)
 inventory.append({'blob':sha,'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),'paths':paths,'commits':sorted(meta['commits']),'modes':sorted(meta['mode'])})
 if any(p.endswith('.csv') for p in paths):
  rows=list(csv.reader(io.StringIO(data.decode('utf-8-sig'))));csvs.append({'blob':sha,'paths':paths,'data_rows':max(0,len(rows)-1),'column_count':len(rows[0]) if rows else 0})
 if zipfile.is_zipfile(io.BytesIO(data)):
  with zipfile.ZipFile(io.BytesIO(data)) as z:
   for entry in z.infolist():
    if entry.file_size>20*1024*1024:raise RuntimeError('Oversized archive entry')
    content=z.read(entry);scan(content,paths+[entry.filename],sha)
    item={'blob':sha,'archive_paths':paths,'entry':entry.filename,'size':entry.file_size,'compressed_size':entry.compress_size,'magic':content[:8].hex()}
    if entry.filename.endswith('.npy'):
     stream=io.BytesIO(content)
     import numpy.lib.format as fmt
     version=fmt.read_magic(stream);reader=fmt.read_array_header_1_0 if version==(1,0) else fmt.read_array_header_2_0;shape,order,dtype=reader(stream)
     item.update(shape=list(shape),dtype=str(dtype),has_object=dtype.hasobject)
    archive_entries.append(item)
refs=git('for-each-ref','--format=%(refname) %(objectname)').decode().splitlines()
report={'main':git('rev-parse','refs/heads/main').decode().strip(),'commit_count':len(commits),'blob_count':len(blobs),'unique_blob_bytes':sum(x['size'] for x in inventory),'refs':refs,'inventory':inventory,'archive_entries_count':len(archive_entries),'archive_entries':archive_entries,'csv_inventory':csvs,'secret_patterns':list(patterns),'secret_pattern_findings':findings,'limits':['All commits reachable from server-advertised mirror refs, not deleted/unadvertised refs or orphan objects on GitHub','Pattern-based scanning can miss unknown secret formats and is not proof of absence','Public commit author metadata is excluded from personal-data leak classification','Binary content static review is not proof of absence of model memorization or vulnerability']}
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['main','commit_count','blob_count','unique_blob_bytes','refs','archive_entries_count','csv_inventory','secret_pattern_findings']},indent=2))
