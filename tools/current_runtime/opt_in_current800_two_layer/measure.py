"""Locked eager CPU Torch-only exact-stage oracle, no native."""
import hashlib,json,sys
from pathlib import Path
H,L,I,F,O=map(Path,sys.argv[1:]); sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(F.read_bytes())=='d54fa50ff0a9aaf839d38a1c69946cb6582cec48e388fa130fc2dec72ee34f73'
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
t,pin=load_oracle(L,I);assert t.get_num_threads()==1 and t.are_deterministic_algorithms_enabled()
p=json.loads(F.read_bytes());assert pin['torch_git_commit']==p['locked_torch_commit'] and pin['wheel_sha256']==p['locked_torch_wheel']
def md(x):
 return dict(dtype=str(x.dtype).removeprefix('torch.'),shape=list(x.shape),strides=list(x.stride()),
  version=x._version,offset=x.storage_offset(),storage_bytes=x.untyped_storage().nbytes(),view=x._is_view(),
  requires_grad=x.requires_grad,contiguous=x.is_contiguous(),device=x.device.type)
def read(x):
 m=md(x)
 l=x.detach().reshape(-1).view(t.uint8).numpy().tobytes()
 r=t.empty(0,dtype=t.uint8).set_(x.untyped_storage(),0,(m['storage_bytes'],),(1,)).numpy().tobytes()
 return m,l,r
def expected(f):
 m={k:f[k] for k in ('dtype','shape','strides','version','offset','storage_bytes','view','requires_grad','contiguous')}
 raw=bytes.fromhex(f['raw_f32le_hex'])
 assert sha(raw)==f['logical_sha256']==f['physical_sha256']
 return m|dict(device='cpu'),raw,raw
O.mkdir();cases=[];mismatch=[];outputs=bytearray()
with t.no_grad():
 for c in p['cases']:
  rows=[];loaded={};prev={};inp=c['inputs'];label=c['label']
  for name,fi in inp.items():
   em,raw,_=expected(fi)
   x=t.frombuffer(bytearray(raw),dtype=t.float32).reshape(fi['shape']).clone();loaded[name]=x
   actual=read(x)
   if actual!=(em,raw,raw):
    mismatch.append(dict(case=label,input=name,observed=actual[0],expected=em));break
   prev[name]=actual
  for s in c['stages'] if not mismatch else ():
   args=s['command'];opcode=args[0];orient=None
   if opcode==32:
    y=t.mm(loaded[args[1]],loaded[args[2]])
   elif opcode==2:
    assert args[1]==1
    left,right=loaded[args[2]],loaded[args[3]]
    orient=dict(left_shape=list(left.shape),right_shape=list(right.shape),broadcast_bias_axis=1,broadcast_over_batch=True)
    assert left.ndim==2 and right.ndim==1 and left.shape[1]==right.shape[0]
    # Independent explicit row check; the target call remains eager nonfused add.
    per_batch=t.stack([left[k]+right for k in range(left.shape[0])])
    y=t.add(left,right)
    orient['explicit_per_row_equals_broadcast']=read(per_batch)[1]==read(y)[1]
   elif opcode==3:
    assert args[1]==5
    y=t.relu(loaded[args[2]])
   else:raise AssertionError(args)
   observed=read(y);exp=expected(s['expected'])
   unchanged=all(read(v)==prev[n] for n,v in loaded.items())
   fresh=all(not t._C._is_alias_of(v,y) and v.untyped_storage()._cdata!=y.untyped_storage()._cdata for v in loaded.values())
   good=observed==exp and unchanged and fresh and (not orient or orient['explicit_per_row_equals_broadcast'])
   row=dict(stage=s['name'],command=args,metadata=observed[0],logical_sha256=sha(observed[1]),physical_sha256=sha(observed[2]),
            prior_inputs_and_intermediates_unchanged=unchanged,nonalias_fresh_storage=fresh,bias_orientation=orient,
            offset=len(outputs),length=len(observed[1]))
   outputs.extend(observed[1]);rows.append(row)
   if not good:
    mismatch.append(dict(case=label,stage=s['name'],observed=row,expected=exp[0],
                        observed_bytes=observed[1].hex(),expected_bytes=exp[1].hex()));break
   loaded[s['name']]=y;prev[s['name']]=observed
  cases.append(dict(case=label,geometry=c['input_geometry'],stages=rows))
  if mismatch:break
rows=[r for c in cases for r in c['stages']]
if not mismatch:assert len(cases)==4 and len(rows)==20 and len(outputs)==432
(O/'logical-f32le.bin').write_bytes(outputs)
r=dict(status='MATCH_LOCKED_EXACT_4_20' if not mismatch else 'MISMATCH_FROZEN',threads=1,no_grad=True,
       eager_unfused=True,pin=pin,prereg_sha256=sha(F.read_bytes()),cases=cases,mismatch=mismatch,
       summary=dict(cases=len(cases),stages=len(rows),output_words=len(outputs)//4,raw_sha256=sha(outputs),
        bias_orientation_exact=sum(bool(x['bias_orientation']) and x['bias_orientation']['explicit_per_row_equals_broadcast'] for x in rows)))
(O/'receipt.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
print(r['status'],r['summary'],mismatch,flush=True);sys.exit(bool(mismatch))
