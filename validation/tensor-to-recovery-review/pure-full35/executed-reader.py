from pathlib import Path
import gzip,hashlib,importlib.util,json,subprocess,sys,traceback
base=Path('/home/qian-qi/heptatorch-continuation-20261116/runs/to09b-full35-614-01')
src=base/'execution/source/tests/tensor_to_frozen35_accounting'
actual=base/'actual-first'
out=base/'root-readmission-01'
out.mkdir(exist_ok=False)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(o):return json.dumps(o,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(p):return p.read_bytes()
inputs={}
def pinned(p,pin=None):
 b=read(p);h=sha(b)
 if pin:assert h==pin,(str(p),h,pin)
 inputs[str(p)]={'bytes':len(b),'sha256':h}
 return b
report={'schema':'heptatorch.root.to09b-full35-pure-readmission.v1','status':'RUNNING','native_executed':False,'Torch_executed':False,'new_build':False,'whole_pytorch_gate':'HOLD','closed_schema_count':0,'journals':[]}
def save():
 (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
save()
try:
 script=src/'check_frozen35_accounted.py'
 pinned(script,'a86cde9c035e622932560441ad4d496452bef6afd648c836f297240e0b0136bf')
 raw=pinned(actual/'full35-original.json','8468fc0eeaccca1051795b33f0be5a304029f260699a4f267b30bda4bb4b3b58')
 outer=pinned(actual/'full35-observer.json','ca79c0bee9b612c1446659e00774eea1acbc40bcdcf4fdc7a9bf9bb586e8da99')
 argv=[sys.executable,'-S','-B',str(script),'--observer-report',str(actual/'full35-observer.json'),'--original-report',str(actual/'full35-original.json')]
 with (out/'strict.stdout').open('xb') as so,(out/'strict.stderr').open('xb') as se:
  p=subprocess.run(argv,stdout=so,stderr=se,check=False,timeout=60)
 report['strict_original_checker']={'argv':argv,'returncode':p.returncode,'stdout_sha256':sha((out/'strict.stdout').read_bytes()),'stderr_sha256':sha((out/'strict.stderr').read_bytes())}
 save();assert p.returncode==0
 for name,pin in {
 'owner_accounting_to.py':'a496aae3ca0b9f02b7f0e8c26c7177c3ce5b5a8c4198354c22059434b1491914',
 'legacy_owner_rules.py':'281f1ea437046a9cd884de2c8da34f38649a3b6482ade23123ac4aa992faed64',
 'owner672_general.py':'6d230e2df25e04e32f76fb10fd0936ec9d220b60acda37c40b52a45633372312',
 'owner672_fixed35.py':'e58adb23bb65cdc81ec36e3e2fcbd178c0d5198f3b10c5d274afa94da7761a6f',
 'profile.json':'1e5e8db18a2e9a7bc4b98ac1063a427eb0e60613781962371dacafc76414529f'}.items():
  pinned(src/name,pin)
 spec=importlib.util.spec_from_file_location('root_to09b_pure_observer',src/'owner_accounting_to.py')
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 total_events=0;total_snapshots=0;total_nodes=0
 paths=sorted((actual/'journals/614/owners').glob('*/events.json'))
 assert len(paths)==35,len(paths)
 for path in paths:
  events=json.loads(pinned(path));snapfile=path.with_name('snapshots.jsonl.gz');pinned(snapfile)
  snapshots={}
  with gzip.open(snapfile,'rb') as f:
   for line in f:
    item=json.loads(line);snapshot=item['snapshot'];key=item['sha256']
    assert key==sha(canon(snapshot)) and key not in snapshots,(str(path),'snapshot key')
    assert snapshot['profile']=='tensor-to-copy-614-fixed35-F32-Saved'
    for node in snapshot['nodes'].values():
     assert len(node)==84
     total_nodes+=1
    snapshots[key]=snapshot
  for ordinal,e in enumerate(events):
   assert not e.get('error'),(str(path),ordinal,e.get('error'))
   before=snapshots[e['before']];after=snapshots[e['after']]
   got=module.verify_transition(before,after,e['tickets'],e['commands'],e['expected_status'])
   assert got==e['result'],(str(path),ordinal,'replayed result differs')
  report['journals'].append({'name':path.parent.name,'events':len(events),'snapshots':len(snapshots),'status':'PASS','events_sha256':inputs[str(path)]['sha256'],'snapshots_sha256':inputs[str(snapfile)]['sha256']})
  total_events+=len(events);total_snapshots+=len(snapshots);save()
 assert 'torch' not in sys.modules
 for path,v in inputs.items():
  b=read(Path(path));assert len(b)==v['bytes'] and sha(b)==v['sha256'],('input changed',path)
 inner=json.loads(raw)
 report.update(status='PASS_ORIGINAL_REPORT_AND_ALL_COMMITTED_JOURNAL_TRANSITIONS',journal_count=len(paths),replayed_events=total_events,full_snapshots=total_snapshots,full_84_word_nodes=total_nodes,all_inputs_unchanged=True,input_files=inputs,original_report_frames=inner['frames'],original_report_values=inner['compared_values'],original_state_version_differences=inner['explicit_state_absolute_version_differences'],original_parameter_version_mismatches=inner['direct_parameter_absolute_version_mismatches'],original_overall_job_stays='FAIL',original_copied_failure_preserved=True)
except BaseException:
 report.update(status='FAIL',traceback=traceback.format_exc(),input_files=inputs)
finally:
 save();b=(out/'report.json').read_bytes()
 print(json.dumps({'status':report['status'],'report':str(out/'report.json'),'sha256':sha(b),'journals':len(report['journals']),'replayed_events':report.get('replayed_events'),'snapshots':report.get('full_snapshots')}),flush=True)
 if report['status']=='FAIL':raise SystemExit(1)
