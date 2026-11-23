"""Locked upstream Torch-only validation of frozen rank4 public dtype observations."""
from pathlib import Path
import hashlib,json,sys
H,L,I,F,O=map(Path,sys.argv[1:])
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(F.read_bytes())=='df0970b57a6a5f69fbd5faf047d1d58c0302afcd20cbaa499759e5c0c68d57f8'
generator=F.with_name('freeze.py').read_bytes()
assert sha(generator)=='685d26e2af41fd6009142102de0c8cdf451ac7505cb970bf2d9ab2c63314e7d2'
lib={};exec(compile(generator.split(b'\ncases=[]',1)[0],str(F.with_name('freeze.py')),'exec'),lib)
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
t,pin=load_oracle(L,I)
assert t.get_num_threads()==1 and t.are_deterministic_algorithms_enabled() and sys.byteorder=='little'
p=json.loads(F.read_bytes())
assert p['wheel']==pin['wheel_sha256'] and p['pinned_torch']==pin['torch_git_commit']
def md(x):
 return dict(dtype=str(x.dtype).removeprefix('torch.'),shape=list(x.shape),strides=list(x.stride()),
  offset=x.storage_offset(),storage_bytes=x.untyped_storage().nbytes(),version=x._version,
  is_contiguous=x.is_contiguous(),view=x._is_view(),requires_grad=x.requires_grad,device=x.device.type)
def logical(x):
 return x.detach().reshape(-1).view(t.uint8).numpy().tobytes()
def physical(x):
 n=x.untyped_storage().nbytes()
 if not n:return b''
 return t.empty(0,dtype=t.uint8).set_(x.untyped_storage(),0,(n,),(1,)).numpy().tobytes()
def capture_in(x):
 return md(x),logical(x),physical(x)
O.mkdir(exist_ok=False);rows=[];bad=[]
with (O/'logical-i64le.bin').open('xb') as lf,(O/'physical-i64le.bin').open('xb') as pf:
 for case in p['cases']:
  label=case['label'];ex=case['expected_torch'];shape=case['input_shape']
  dtype=case['input_dtype'];dt=getattr(t,dtype)
  b=lib['source'](dtype,shape,case['recipe']);assert sha(b)==case['input_raw_sha256']
  if dtype=='bool':assert all(w<=1 for w in b)
  x=t.frombuffer(bytearray(b),dtype=dt).reshape(shape).clone() if b else t.empty(shape,dtype=dt)
  prior=capture_in(x)
  exp_input=dict(dtype=dtype,shape=shape,strides=case['input_strides'],offset=0,storage_bytes=len(b),
                 version=0,is_contiguous=True,view=False,requires_grad=False,device='cpu')
  if prior!=(exp_input,b,b):
   bad.append(dict(label=label,stage='input',metadata=prior[0],logical_sha256=sha(prior[1]),physical_sha256=sha(prior[2]),
                   exp_input=exp_input)); break
  y=t.ops.aten.nonzero.default(x);assert isinstance(y,t.Tensor)
  # Metadata collected before the independent raw storage view.
  m=md(y);l=logical(y);v=physical(y)
  no_alias=not t._C._is_alias_of(x,y) and x.untyped_storage()._cdata!=y.untyped_storage()._cdata
  same=capture_in(x)==prior
  expected_meta={k:ex[k] for k in m if k!='device'}|dict(device='cpu')
  expected_k=ex['shape'][0]
  assert expected_k==case['k'] and 32*expected_k==ex['storage_bytes']
  nbytes=8*expected_k
  cs=[sha(v[j*nbytes:(j+1)*nbytes]) for j in range(4)]
  ls=[sha(y[:,j].contiguous().view(t.uint8).numpy().tobytes()) for j in range(4)]
  r=dict(label=label,dtype=dtype,metadata=m,input_unchanged=same,
         fresh_nonalias_storage=no_alias,logical_sha256=sha(l),physical_sha256=sha(v),physical_column_sha256=cs,logical_column_sha256=ls,
         logical_offset=lf.tell(),physical_offset=pf.tell(),bytes=len(l))
  rows.append(r);lf.write(l);pf.write(v)
  if not (m==expected_meta and len(l)==len(v)==32*expected_k and
          sha(l)==ex['logical_sha256'] and sha(v)==ex['physical_sha256'] and cs==ls==ex['column_sha256'] and
          ('logical_hex' not in ex or l.hex()==ex['logical_hex']) and
          ('physical_hex' not in ex or v.hex()==ex['physical_hex']) and same and no_alias):
   bad.append(dict(label=label,stage='output',observed=r,expected=ex));break
lr=(O/'logical-i64le.bin').read_bytes();pr=(O/'physical-i64le.bin').read_bytes()
a=[r for r in rows if r['dtype']!='bool'];c=[r for r in rows if r['dtype']=='bool']
summary=dict(cases=len(rows),new_cases=len(a),scope_words=sum(r['bytes']//8 for r in a),
 all_words=len(lr)//8,logical_sha256=sha(lr),physical_sha256=sha(pr),bytes_each=len(lr),
 measured_different_logical_physical=sum(r['logical_sha256']!=r['physical_sha256'] for r in rows),
 retained_bool=[dict(label=r['label'],shape=r['metadata']['shape']) for r in c],
 scope_k0=sum(r['metadata']['shape'][0]==0 for r in a),scope_k1=sum(r['metadata']['shape'][0]==1 for r in a))
if not bad:
 assert len(a)==65 and len(c)==3 and len(rows)==68
 assert summary['scope_words']==5752 and summary['all_words']==5776 and len(lr)==len(pr)
result=dict(status='MATCH_EXACT_LOCKED_68' if not bad else 'MISMATCH_FROZEN',threads=1,pin=pin,
 prereg_sha256=sha(F.read_bytes()),summary=summary,mismatch=bad,cases=rows,
 storage_note='Alias relation and StorageImpl identity corroborate fresh storage even for empties, not a zero data-pointer comparison. Logical via reshape flatten; physical byte tensor attached directly to underlying untyped PyTorch storage, no contiguous copy.')
(O/'receipt.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(result['status'],summary,bad,flush=True); sys.exit(bool(bad))
