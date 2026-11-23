"""One pinned eager Torch training step, immutable exact stdlib oracle."""
from pathlib import Path
import hashlib,json,sys
H,L,I,F,O=map(Path,sys.argv[1:]); sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(F.read_bytes())=='898b3b2ccb70729d124709d608063e5eec7446161fbe36896319e9c27fecc6f1'
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
t,pin=load_oracle(L,I);assert t.get_num_threads()==1 and t.are_deterministic_algorithms_enabled()
p=json.loads(F.read_bytes())
assert pin['torch_git_commit']==p['locked_torch_git'] and pin['wheel_sha256']==p['locked_torch_wheel']
def md(x):
 return dict(dtype=str(x.dtype).removeprefix('torch.'),shape=list(x.shape),strides=list(x.stride()),
  version=x._version,storage_offset=x.storage_offset(),storage_bytes=x.untyped_storage().nbytes(),view=x._is_view(),
  requires_grad=x.requires_grad,contiguous=x.is_contiguous(),device=x.device.type)
def raw(x):
 n=x.untyped_storage().nbytes()
 return x.detach().reshape(-1).view(t.uint8).numpy().tobytes(),t.empty(0,dtype=t.uint8).set_(x.untyped_storage(),0,(n,),(1,)).numpy().tobytes()
def ident(x):return (x._cdata,x.untyped_storage()._cdata)
def same(x,m):return (md(x),*raw(x),ident(x))==m
expected_keys=('dtype','shape','strides','version','storage_offset','storage_bytes','view','requires_grad','contiguous')
O.mkdir(exist_ok=False); outputs=bytearray(); rows=[]; mismatch=[]; refs={};initial={};before_snap={}
class Difference(Exception):pass
def check(label,x,f,extra=None):
 r,b=raw(x);m=md(x);e={k:f[k] for k in expected_keys}|dict(device='cpu');w=bytes.fromhex(f['raw_f32le_hex'])
 assert sha(w)==f['logical_sha256']==f['physical_sha256']
 row=dict(label=label,metadata=m,logical_sha256=sha(r),physical_sha256=sha(b),offset=len(outputs),length=len(r))
 if extra:row.update(extra)
 rows.append(row);outputs.extend(r)
 if (m,r,b)!=(e,w,w) or extra and not all(extra.values()):
  mismatch.append(dict(stage=label,metadata_expected=e,observed=row,logical_expected=w.hex(),
                       logical_actual=r.hex(),physical_actual=b.hex()))
  raise Difference(label)
 return m,r,b,ident(x)
def forward(out,stored,seq):
 for spec in seq:
  a=spec['command'];label=spec['name']
  if a[0]==32:y=t.mm(out[a[1]],out[a[2]])
  elif a[:2]==[2,2]:y=t.sub(out[a[2]],out[a[3]])
  elif a[:2]==[2,3]:y=t.mul(out[a[2]],out[a[3]])
  elif a[:2]==[4,7]:y=t.sum(out[a[2]])
  else:raise AssertionError(a)
  fresh=all(not t._C._is_alias_of(y,z) and y.untyped_storage()._cdata!=z.untyped_storage()._cdata for z in out.values())
  stored[label]=check(label,y,spec['expected'],dict(fresh_distinct_storage=fresh,all_prior_unchanged=all(same(z,stored[key]) for key,z in out.items())))
  out[label]=y
try:
 for label,fi in p['input_tensors'].items():
  b=bytes.fromhex(fi['raw_f32le_hex'])
  x=t.frombuffer(bytearray(b),dtype=t.float32).reshape(fi['shape']).clone()
  if fi['requires_grad']:x.requires_grad_(True)
  refs[label]=x;initial[label]=check(label,x,fi,dict(is_leaf=x.is_leaf,has_no_gradient=x.grad is None))
 before=dict(refs);before_snap=dict(initial);forward(before,before_snap,p['before_update'])
 before['loss'].backward()
 weight=refs['weight'];g=weight.grad
 assert isinstance(g,t.Tensor)
 grad=check('weight_gradient',g,p['backward']['expected_gradient'],dict(
   distinct_grad_authority=all(not t._C._is_alias_of(g,z) and g.untyped_storage()._cdata!=z.untyped_storage()._cdata for z in refs.values()),
   prior_inputs_unchanged=all(same(z,initial[n]) for n,z in refs.items()),
   no_grad_for_inputs_without_requirement=refs['x'].grad is None and refs['target'].grad is None))
 # Retire the old forward graph while preserving the actual leaf and its gradient; do not call backward again.
 for e in reversed(p['before_update']):del before[e['name']]
 del before,before_snap
 wver=weight._version; wids=ident(weight)
 opt=t.optim.SGD([weight],lr=1/8,momentum=0,weight_decay=0,dampening=0,nesterov=False,foreach=False,fused=False)
 opt.step()
 after_w=check('one_sgd_weight_update',weight,p['update']['weight_after'],dict(
    exact_parameter_version_delta_one=weight._version-wver==1,same_tensor_and_storage=ident(weight)==wids,
    gradient_object_bytes_metadata_unchanged=weight.grad is g and same(g,grad),
    x_target_raw_and_metadata_unchanged=same(refs['x'],initial['x']) and same(refs['target'],initial['target']),
    no_optimizer_state=len(opt.state)==0))
 aft=dict(refs); snap=dict(initial);snap['weight']=after_w
 forward(aft,snap,p['after_update'])
 assert all(same(refs[n],snap[n]) for n in refs)
except Difference:
 pass
blob=bytes(outputs)
(O/'all-f32le.bin').write_bytes(blob)
r=dict(status='MISMATCH_FROZEN' if mismatch else 'MATCH_LOCKED_ONE_STEP_EXACT',threads=1,
       prereg_sha256=sha(F.read_bytes()),pin=pin,rows=rows,mismatch=mismatch,
       total_rows=len(rows),raw_words=len(blob)//4,raw_sha256=sha(blob),
       eager_operations=['mm','sub','mul','sum','scalar backward','torch.optim.SGD.step'],
       optimizer_profile=dict(lr='1/8',momentum=0,weight_decay=0,foreach=False,fused=False))
if not mismatch:assert len(rows)==13 and len(blob)==104
(O/'receipt.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
print(r['status'],r['total_rows'],r['raw_words'],r['raw_sha256'],mismatch,flush=True);sys.exit(bool(mismatch))
