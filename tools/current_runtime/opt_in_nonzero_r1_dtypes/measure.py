"""Verify the exact frozen rank1 dtype fixture with locked CPU Torch only."""
from pathlib import Path
import hashlib,json,sys
H,L,I,F,O=map(Path,sys.argv[1:])
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(F.read_bytes())=='e75921cb1b5366eaccbb4b5dc4eab5213bda2263c1ee53b4b77955b8e7716797'
oracle=F.with_name('freeze.py').read_bytes()
assert sha(oracle)=='95a7a36e15f4ae6f7fddac7d08e2d759f07b2875288e65668c60e51743c79c40'
lib={};exec(compile(oracle.split(b'\nraw=lambda ',1)[0],str(F.with_name('freeze.py')),'exec'),lib)
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
t,pin=load_oracle(L,I);assert t.get_num_threads()==1 and t.are_deterministic_algorithms_enabled()
p=json.loads(F.read_bytes())
assert p['wheel']==pin['wheel_sha256'] and p['torch_commit']==pin['torch_git_commit']
assert sys.byteorder=='little'
def metadata(x):
 return dict(dtype=str(x.dtype).removeprefix('torch.'),shape=list(x.shape),strides=list(x.stride()),
  offset=x.storage_offset(),storage_bytes=x.untyped_storage().nbytes(),version=x._version,
  is_contiguous=x.is_contiguous(),view=x._is_view(),requires_grad=x.requires_grad,device=x.device.type)
def logical(x):
 return x.detach().flatten().view(t.uint8).numpy().tobytes()
def physical(x):
 n=x.untyped_storage().nbytes()
 if not n:return b''
 return t.empty(0,dtype=t.uint8).set_(x.untyped_storage(),0,(n,),(1,)).numpy().tobytes()
def construct(case,b):
 size,=case['input_shape'];dt=getattr(t,case['input_dtype'])
 if not size:return t.empty((0,),dtype=dt)
 # Clone bytes before reinterpreting/attaching storage to a fresh ordinary root.
 w=lib['TYPES'][case['input_dtype']][1]
 a=t.frombuffer(bytearray(b),dtype=t.uint8).clone()
 src=t.empty(0,dtype=dt).set_(a.untyped_storage(),0,(size,),(1,))
 # Tensor.clone establishes Version0 and a fresh ordinary root. Copy bit preservation
 # is checked against the frozen input SHA (both storage bytes and logical bytes).
 assert a.numel()==size*w
 return src.clone()
O.mkdir(exist_ok=False)
rows=[];mismatch=[]
with (O/'selected-i64le.bin').open('xb') as output:
 for case in p['cases']:
  label=case['label'];exp=case['expected_torch'];expected_n,=case['input_shape']
  b=lib['source'](case['input_dtype'],expected_n,case['recipe'])
  assert sha(b)==case['input_raw_sha256']
  x=construct(case,b);initial=metadata(x);input_match=(logical(x)==physical(x)==b and initial==dict(
   dtype=case['input_dtype'],shape=case['input_shape'],strides=[1],offset=0,storage_bytes=len(b),
   version=0,is_contiguous=True,view=False,requires_grad=False,device='cpu'))
  if not input_match:
   mismatch.append(dict(label=label,stage='torch-input',input_observed=initial,raw_sha256=sha(logical(x)),
    physical_sha256=sha(physical(x)),expected_raw_sha256=case['input_raw_sha256']));break
  y=t.ops.aten.nonzero.default(x);assert isinstance(y,t.Tensor)
  m=metadata(y);raw=logical(y);phys=physical(y)
  clean_input=metadata(x)==initial and logical(x)==physical(x)==b
  alias=t._C._is_alias_of(x,y); storage_distinct=x.untyped_storage()._cdata!=y.untyped_storage()._cdata
  meta_match=m=={**{k:exp[k] for k in m if k!='device'},'device':'cpu'}
  passed=meta_match and raw==phys and sha(raw)==exp['raw_sha256'] and len(raw)==m['storage_bytes'] and clean_input and not alias and storage_distinct
  passed=passed and ('raw_hex' not in exp or raw.hex()==exp['raw_hex'])
  row=dict(label=label,metadata=m,input_sha256=sha(b),raw_sha256=sha(raw),physical_sha256=sha(phys),
            input_unchanged=clean_input,fresh_nonalias_storage=not alias and storage_distinct,
            empty_data_pointer_not_used=True,preregistered_in_scope=case['admitted'],offset=output.tell(),length=len(raw))
  rows.append(row);output.write(raw)
  if not passed:
   mismatch.append(dict(label=label,stage='torch-output',observed=row,expected=exp));break
blob=(O/'selected-i64le.bin').read_bytes()
scope=[r for r in rows if r['preregistered_in_scope']]
cap=[dict(label=r['label'],k=r['metadata']['shape'][0]) for r in rows if not r['preregistered_in_scope']]
r=dict(status='MISMATCH_FROZEN' if mismatch else 'MATCH_LOCKED_EXACT_62',threads=1,lock=pin,
 prereg_sha256=sha(F.read_bytes()),cases=rows,mismatch=mismatch,scope_only=True,
 observations=dict(cases=len(rows),scoped_cases=len(scope),scoped_words=sum(r['length']//8 for r in scope),
  all_words=len(blob)//8,raw_bytes=len(blob),raw_sha256=sha(blob),overcap_torch_reference_only=cap,
  result_storage_check='PyTorch alias API & StorageImpl identity including zero-size; not a public data-pointer identity or native authority token claim.'))
(O/'receipt.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
if not mismatch:
 assert len(rows)==p['summary']['all'] and len(scope)==p['summary']['admitted']
 assert len(blob)//8==p['summary']['all_words'] and r['observations']['scoped_words']==p['summary']['admitted_words']
print(r['status'],r['observations'],mismatch,flush=True);sys.exit(bool(mismatch))
