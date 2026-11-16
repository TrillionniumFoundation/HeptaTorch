"""Read original PR23 reference bytes and the exact committed reference-only diff."""
from pathlib import Path
import hashlib,json,re,shutil,subprocess,zipfile
R=Path('/dev/shm/heptatorch-d26699e25fea/HeptaTorch-reference-bernoulli-gradient')
A=Path('/dev/shm/heptatorch-d26699e25fea/ref-ci-9a20ead')
O=Path('/dev/shm/heptatorch-d26699e25fea/HeptaTorch-gradient-adam-distributions/validation/hosted-reference-9a20ead/reference')
HEAD='9a20eadbe20a801c8b26727ae369e0ee0832e423';BASE='e4cf7b6d3645b5a340ec3b47489ecc326e908c13';TREE='5519a9f84b9768e8dcf9ff041ddd1d5969f3bcbc'
sha=lambda b:hashlib.sha256(b).hexdigest()
checks=[]
def check(ok,label):
 assert ok,label
 checks.append(label)
def git(*args):return subprocess.check_output(['git','-C',str(R),*args])
def source(path,ref=HEAD):return git('show',ref+':'+path)
probe_names={'torch-gradient-materialization.json':('pytorch_gradient_materialization_probe.py',16,25),'torch-gradient-alias-edge.json':('pytorch_gradient_alias_edge_probe.py',6,12),'torch-bernoulli-gradient-layout.json':('pytorch_bernoulli_gradient_layout_probe.py',8,None),'torch-zero-merge-layout.json':('pytorch_zero_merge_layout_probe.py',2,None),'torch-scalar-add-nan.json':('pytorch_scalar_add_nan_probe.py',12,48),'torch-extreme-stride.json':('pytorch_extreme_stride_probe.py',8,None)}
changed=git('diff','--name-only',BASE,HEAD).decode().splitlines()
allowed={'.github/workflows/backward-observations.yml',*('tests/'+v[0] for v in probe_names.values())}
check(len(changed)==21 and all(p in allowed or p.startswith('validation/') for p in changed),'exact21 paths: six new reference probes, reference workflow and validation only')
check(source('.github/workflows/native.yml')==source('.github/workflows/native.yml',BASE),'native workflow byte-identical to admitted main')
for path in ['tests/pytorch_oracle.py','compatibility/pytorch-oracle-lock.json','tests/pytorch_tensor_rng_backward_oracle.py','tests/tensor_object_oracle/pytorch_tensor_objects_oracle.py','tools/collect_ci_reports.py','tools/check_acceptance.py','compatibility/full-acceptance.json','compatibility/upstream-schemas.json']:
 check(source(path)==source(path,BASE),path+': original implementation/reference/ledger unchanged')
workflow=source('.github/workflows/backward-observations.yml')
old=source('.github/workflows/backward-observations.yml',BASE)
prefix=workflow.split(b'      - name: Verify fixed incoming-gradient materialization source and syntax\n',1)[0]
for probe,_,_ in probe_names.values():prefix=prefix.replace(('"tests/'+probe+'", ').encode(),b'')
check(prefix==old,'exact inverse retains all original reference steps; only new trigger paths and appended steps')
zb=(A/'reports.zip').read_bytes();check(len(zb)==389929 and sha(zb)=='c52993dd041daf0a6d512e361341934368708309f79efe01b7d70edb17dff2a7','original25-report ZIP official size and digest')
mb=(A/'ci-report-manifest.json').read_bytes();m=json.loads(mb)
check(m['status']=='COMPLETE' and m['job_status_before_export']=='success' and not m['tracked_worktree_dirty'] and not m['original_report_bytes_reserialized'],'successful clean original-byte export')
check(m['checkout_commit']=='bb7288e4a764716c20ff92064265dd27f1219a3e' and m['checkout_tree']==TREE,'actual merge checkout and tree')
check(m['github']['GITHUB_RUN_ID']=='43842245575' and m['github']['HEPTATORCH_SOURCE_HEAD_SHA']==HEAD,'actual run and head')
check(m['workflow_sha256']==sha(workflow) and m['collector_sha256']==sha(source('tools/collect_ci_reports.py')),'actual workflow and collector source binding')
envelopes=sorted(set(x.decode() for x in re.findall(rb'tools/emit_ci_report\.py build/([A-Za-z0-9_.-]+\.json)',workflow)))
check(len(envelopes)==24 and envelopes==m['required_envelope_reports'] and not m['missing_envelope_reports'],'all24 required envelopes retained')
reports={}
with zipfile.ZipFile(A/'reports.zip') as z:
 check(len(z.namelist())==len(set(z.namelist()))==26 and set(z.namelist())=={x['path'] for x in m['reports']}|{'ci-report-manifest.json'},'all25 reports and original manifest exactly once')
 check(z.read('ci-report-manifest.json')==mb,'manifest unchanged')
 for row in m['reports']:
  raw=(A/row['path']).read_bytes();check(len(raw)==row['bytes'] and sha(raw)==row['sha256'] and z.read(row['path'])==raw,row['path']+': original bytes bound')
  reports[row['path']]=json.loads(raw)
prov=0
for name,d in reports.items():
 if 'oracle_provenance' in d:
  p=d['oracle_provenance'];check(p['status']=='PASS' and len(p['checks'])==11 and all(v is True for v in p['checks'].values()),name+': eleven actual provenance checks')
  check(p['lock_sha256']==sha(source('compatibility/pytorch-oracle-lock.json')),name+': exact lock');prov+=1
 if 'probe_errors' in d:check(d['probe_errors'] in (0,[]),name+': no probe error')
 if name.endswith('-prepared.json'):check(d['status']=='PREPARED_NOT_RUN' and d['cases']==0,name+': preparation remains NOT_RUN')
for name,(probe,families,variants) in probe_names.items():
 d=reports[name];check(d['status']=='OBSERVED' and d['cases']==len(d['rows'])==families,name+': actual family count')
 check(d['probe_sha256']==sha(source('tests/'+probe)),name+': committed unchanged probe')
 check(d['reference_support_sha256']==sha(source('tests/pytorch_oracle.py')) and d['lock_sha256']==sha(source('compatibility/pytorch-oracle-lock.json')),name+': support and lock')
 check(d['native_executed'] is False and d['full_rewrite_complete'] is False and d['pytorch_api_denominator']==2590,name+': reference observations are not native parity')
 if variants is not None:check(d['variants']==d['expected_variants']==variants,name+': exact variant count')
 check(all(v is True for v in d['source_files_unchanged'].values()),name+': no input source mutation')
ledger=json.loads(source('compatibility/full-acceptance.json'));check(ledger['status']=='HOLD' and not ledger['whole_pytorch_rewrite_complete'] and len(ledger['gates'])==17,'all17 full gates and HOLD preserved')
O.mkdir(parents=True,exist_ok=False)
for name in ['reports.zip','ci-report-manifest.json']:shutil.copy2(A/name,O/name)
shutil.copy2(__file__,O/Path(__file__).name)
result={'schema':'heptatorch.pr23.reference-admission-review.v1','status':'PASS_REFERENCE_ONLY_REVIEW','head':HEAD,'base':BASE,'tree':TREE,'workflow_run':43842245575,'job':133213530745,'artifact_id':14213701692,'artifact_bytes':len(zb),'artifact_sha256':sha(zb),'original_report_count':25,'required_envelopes':24,'actual_provenance_occurrences':prov,'checks':checks,'check_count':len(checks),'changed_paths':changed,'original_reports':m['reports'],'source_sha256':sha(Path(__file__).read_bytes()),'full_rewrite_complete':False,'whole_project_decision':'HOLD','pytorch_api_denominator':2590}
b=(json.dumps(result,indent=2)+'\n').encode();(O/'root-review.json').write_bytes(b)
print(json.dumps({'status':result['status'],'checks':len(checks),'provenance_occurrences':prov,'report_sha256':sha(b)}))
