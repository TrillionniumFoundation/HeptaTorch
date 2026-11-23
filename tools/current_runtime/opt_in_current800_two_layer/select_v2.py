"""Read-only exact post-reference scoping; does not remeasure or modify original."""
import hashlib,json,struct,sys
from pathlib import Path
original,measured,baseline_source,baseline_elf,output=map(Path,sys.argv[1:])
sha=lambda b:hashlib.sha256(b).hexdigest()
def bind(path,wanted):
 raw=path.read_bytes();assert sha(raw)==wanted,path
 return raw
frozen_raw=bind(original,'d54fa50ff0a9aaf839d38a1c69946cb6582cec48e388fa130fc2dec72ee34f73')
torch_raw=bind(measured/'receipt.json','8dfe7aa9da4ef9f15b69249567798520b0549f228660fa46172bdd2fcb2c652e')
partial=bind(measured/'logical-f32le.bin','d218874796e57aec4c1816b47f668b2a3277daab2766bac940630c38b259c035')
f=json.loads(frozen_raw);m=json.loads(torch_raw)
sr=bind(baseline_source,'f8f924a82763a28d8ad12adbd371f25458ed86d1fbdee2406bb93912847afb4b')
binary=bind(baseline_elf,'5c8dfb2d329e211e7871438d64754b0f8912cc687c1d9ae3fc432f20e95d165d')
assert sha(sr)==f['source_sha256'] and sha(binary)==f['native_elf_sha256']
assert m['status']=='MISMATCH_FROZEN' and m['no_grad'] and m['eager_unfused'] and m['threads']==1
assert m['pin']['torch_git_commit']==f['locked_torch_commit'] and m['pin']['wheel_sha256']==f['locked_torch_wheel']
assert m['prereg_sha256']==sha(frozen_raw) and len(f['cases'])==len(m['cases'])==4 and len(partial)==400
assert len(m['mismatch'])==1 and m['summary']['stages']==19
def metadata(expected):
 return {k:expected[k] for k in ('dtype','shape','strides','version','offset','storage_bytes','view','requires_grad','contiguous')}|dict(device='cpu')
def assert_stage(s,obs,offset):
 assert obs['stage']==s['name'] and obs['command']==s['command']
 assert obs['metadata']==metadata(s['expected']) and obs['offset']==offset
 assert obs['nonalias_fresh_storage'] and obs['prior_inputs_and_intermediates_unchanged']
 if s['command'][0]==2:
  o=obs['bias_orientation']
  assert o['broadcast_bias_axis']==1 and o['broadcast_over_batch'] and o['explicit_per_row_equals_broadcast']
  assert o['left_shape']==s['expected']['shape'] and o['right_shape']==[o['left_shape'][1]]
 else:assert obs['bias_orientation'] is None
 want=bytes.fromhex(s['expected']['raw_f32le_hex'])
 assert len(want)==obs['length']==s['expected']['storage_bytes']
 assert sha(want)==s['expected']['logical_sha256']==s['expected']['physical_sha256']
 got=partial[offset:offset+len(want)]
 assert sha(got)==obs['logical_sha256']==obs['physical_sha256']
 return want,got
offset=0;admitted=[];prior=[]
for idx,(c,o) in enumerate(zip(f['cases'],m['cases'],strict=True)):
 assert c['label']==o['case'] and c['input_geometry']==o['geometry']
 if idx<3:
  assert len(c['stages'])==len(o['stages'])==5
  for s,obs in zip(c['stages'],o['stages'],strict=True):
   w,g=assert_stage(s,obs,offset);assert w==g;offset+=len(w)
  admitted.append(dict(original_frozen=c,exact_locked_observations=o))
 else:
  assert c['label']=='hidden-width-one-broadcast-two-outputs'
  assert len(c['stages'])==5 and len(o['stages'])==4
  for s,obs in zip(c['stages'][:3],o['stages'][:3],strict=True):
   w,g=assert_stage(s,obs,offset);assert w==g;offset+=len(w);prior.append(obs)
  obs=o['stages'][3];s=c['stages'][3];assert obs['stage']=='mm2'
  w,g=assert_stage(s,obs,offset);assert w!=g and offset+len(w)==len(partial)
  mw=m['mismatch'][0];assert mw['case']==c['label'] and mw['observed']==obs and mw['expected']==metadata(s['expected'])
  assert mw['expected_bytes']==w.hex() and mw['observed_bytes']==g.hex()
  aw=struct.unpack('<8I',w); ag=struct.unpack('<8I',g)
  changes=[dict(zero_based_row=i//2,zero_based_col=i%2,original_expected=a,actual_locked=b)
           for i,(a,b) in enumerate(zip(aw,ag,strict=True)) if a!=b]
  assert changes==[dict(zero_based_row=i,zero_based_col=0,original_expected=0,actual_locked=0x80000000) for i in (1,3)]
  excluded=dict(reason='Numerical signed-zero mismatch, never normalized or changed. Not proposed for native v2 scope.',
    original_frozen=c,original_partial_locked_observations=o,mismatch=mw,exact_raw_differences=changes,
    final_prediction_stage_unexecuted_in_locked_receipt=True)
assert len(admitted)==3 and sum(len(i['original_frozen']['stages']) for i in admitted)==15
nwords=sum(s['length']//4 for c in admitted for s in c['exact_locked_observations']['stages'])
nbytes=4*nwords; assert nbytes==sum(s['length'] for c in m['cases'][:3] for s in c['stages'])
r=dict(tier='IMMUTABLE_SCOPE_V2_AFTER_FROZEN_ORACLE_MISMATCH_ONLY',prereg_sha256=sha(frozen_raw),
 original_locked_receipt_sha256=sha(torch_raw),original_locked_partial_raw_sha256=sha(partial),
 source_sha256=sha(sr),native_elf_sha256=sha(binary),torch_pin=m['pin'],admitted=admitted,
 excluded=excluded,summary=dict(admitted_cases=3,admitted_stages=15,admitted_words=nwords,
   literal_input_words=sum(v['storage_bytes']//4 for a in admitted for v in a['original_frozen']['inputs'].values()),
   admitted_raw_sha256=sha(partial[:nbytes]),excluded_cases=1),
 future_native_requirements=f['notes'][1:3]+[
   'Under separate approval, carry the original live Tensor/Storage/Version owners stage-to-stage in the same session; check distinct fresh owners, unchanged original inputs/intermediates, both storage and wire logical bytes. Keep parent and all intermediate ownership evidence, including lifecycle checks in ordinary scope.',
   'No native stage ran for this v2 or for the excluded original case. No model/frontend/runtime performance, default, related route or full parity claim.'
 ])
raw=(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n').encode()
with output.open('xb') as stream:stream.write(raw)
print(len(raw),sha(raw),r['summary'])
