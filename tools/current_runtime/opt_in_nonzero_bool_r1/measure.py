"""Real locked Torch preflight against immutable independent preregistration."""
import hashlib,itertools,json,math,struct,sys
from pathlib import Path
H,L,I,F,O=map(Path,sys.argv[1:])
sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
assert sha(L.read_bytes())=='ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
assert sha(F.read_bytes())=='cf40ccecf20b5778aa5fe8a79ab5a3401dd9feb841b095e9563ffc62326a2946'
p=json.loads(F.read_bytes()); assert p['tier']=='FROZEN_STD_LIB_PREDICTIONS_ONLY'
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
t,pin=load_oracle(L,I);t.set_num_threads(1)
assert t.get_num_threads()==1 and pin['torch_git_commit']=='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e' and pin['wheel_sha256']=='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3'
CAP=262144
def sha(b):return hashlib.sha256(b).hexdigest()
def source(n,r):
 k=r['kind']
 if k=='all':return b'\1'*n
 if k=='none':return bytes(n)
 if k=='points':
  hits=tuple(r['positions']);assert len(set(hits))==len(hits) and all(0<=i<n for i in hits)
  b=bytearray(n)
  for i in hits:b[i]=1
  return bytes(b)
 if k=='even':return bytes(1-i%2 for i in range(n))
 if k=='pattern':return bytes(int((17*i+3)%31 in (0,3,6,7,17,29)) for i in range(n))
 raise AssertionError(k)
def io(shape,bs,dt=t.bool):
 if not bs:return t.empty(shape,dtype=dt)
 return t.frombuffer(bytearray(bs),dtype=dt).reshape(shape).clone()
def logical(x):
 return bytes(x.detach().contiguous().reshape(-1).view(t.uint8).tolist())
def physical(x):
 assert x.storage_offset()==0
 if not x.numel():return b''
 flat=x.as_strided((x.numel(),),(1,),0)
 return bytes(flat.view(t.uint8).tolist())
def md(z):
 return dict(dtype=str(z.dtype).removeprefix('torch.'),device=str(z.device),shape=list(z.shape),strides=list(z.stride()),
  offset=z.storage_offset(),storage_bytes=z.untyped_storage().nbytes(),version=z._version,requires_grad=z.requires_grad,
  view=z._is_view(),contiguous=z.is_contiguous(),numel=z.numel())
def nonzero(x):return t.ops.aten.nonzero.default(x)
def pre(x):return md(x),sha(logical(x))
cases=[]; guards=[]; mismatch=[]; threaded=[]
dest=O/'selected-i64le.bin'
with dest.open('xb') as output:
 for c in p['cases']:
  b=source(c['input_count'],c['recipe']); assert sha(b)==c['input_raw_sha256']
  x=io(c['input_shape'],b); initial=pre(x);assert initial[1]==sha(b) and not initial[0]['view'] and initial[0]['offset']==initial[0]['version']==0
  y=nonzero(x); assert isinstance(y,t.Tensor)
  result=md(y);raw=logical(y);raw_store=physical(y);e=c['expected_torch']
  match=(all(result[k]==e[k] for k in ('dtype','shape','offset','storage_bytes','requires_grad','view'))
    and result['strides']==e['strides_hypothesis'] and result['contiguous'] and result['device']=='cpu'
    and result['numel']==e['shape'][0] and len(raw)==result['storage_bytes']
    and raw==raw_store and sha(raw)==e['raw_i64le_sha256'])
  match=match and ('raw_i64le_hex' not in e or raw.hex()==e['raw_i64le_hex'])
  same=pre(x)==initial
  fresh=(not result['numel'] or y.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr())
  row=dict(label=c['label'],metadata=result,sha256=sha(raw),physical_sha256=sha(raw_store),input_count=c['input_count'],
   offset=output.tell(),length=len(raw),input_unchanged=same,distinct_nonempty_buffer=fresh if result['numel'] else 'NO_EMPTY_POINTER_IDENTITY')
  output.write(raw);cases.append(row)
  if not (match and same and fresh):mismatch.append(dict(observed=row,expected=e));break
 if not mismatch:
  archive={c['label']:c for c in p['cases']}
  for g in p['guards']:
   l=g['label'];r=dict(label=l,status='NO_ANALOGOUS_TORCH_WIRE')
   if l=='scalar-false':x=io([],b'\0')
   elif l=='scalar-true':x=io([],b'\1')
   elif l=='rank2-control':x=io([2,3],bytes([1,0,1,0,1,0]))
   elif l=='f32-truth-control':x=io([5],struct.pack('<5I',0,0x80000000,0x7fc12345,0xbf800000,1),t.float32)
   elif l=='i64-truth-control':x=io([5],struct.pack('<5q',0,-(1<<63),1,(1<<63)-1,0),t.int64)
   elif l=='actual-offset-Bool-view':
    x=io([7],bytes([0,1,0,1,0,1,1]))[1:6];assert x._is_view() and x.storage_offset()==1
   else:guards.append(r);continue
   y=nonzero(x);z=md(y);lr=logical(y);pr=physical(y)
   r.update(status='ACCEPT_OUTSIDE_CANDIDATE_SCOPE',metadata=z,logical_raw_i64le_hex=lr.hex(),physical_i64le_hex=pr.hex())
   guards.append(r)
   if 'torch_shape' in g:
    ok=z['shape']==g['torch_shape'] and z['strides']==g['torch_strides_hypothesis']
    if 'torch_storage_bytes' in g:ok=ok and z['storage_bytes']==g['torch_storage_bytes']==0
    if 'torch_coords' in g:
     coords=g['torch_coords'];m=len(coords[0]);k=len(coords)
     ok=ok and lr==b''.join(struct.pack('<q',v) for line in coords for v in line)
     ok=ok and pr==b''.join(struct.pack('<q',coords[row][col]) for col in range(m) for row in range(k))
    if not ok:mismatch.append(dict(guard=l,predicted=g,observed=r));break
# Bounded follow-up thread checks only if the single-thread design matched.
if not mismatch:
 frozen={x['label']:x for x in p['cases']};one={y['label']:y for y in cases}
 for th in (2,4):
  t.set_num_threads(th);assert t.get_num_threads()==th
  for name in ('multi-grain-unequal-chunk-count','cap-sparse','cap-dense'):
   c=frozen[name];x=io(c['input_shape'],source(c['input_count'],c['recipe']))
   y=nonzero(x);result=md(y); digest=sha(logical(y)); good=result==one[name]['metadata'] and digest==one[name]['sha256']
   row=dict(threads=th,label=name,md=result,sha256=digest,exact_thread1=good);threaded.append(row)
   if not good:mismatch.append(row);break
  if mismatch:break
 t.set_num_threads(1)
rr=dest.read_bytes()
s=dict(cases=len(cases),words=sum(x['metadata']['numel'] for x in cases),raw_bytes=len(rr),raw_sha256=sha(rr),
 versions=sorted({x['metadata']['version'] for x in cases}),all_contiguous=all(x['metadata']['contiguous'] for x in cases),
 overcap=[dict(label=x['label'],k=x['metadata']['numel']) for x in cases if x['input_count']>262144],
 threaded_repeats=len(threaded),guards=len(guards))
R=dict(status='MATCH_SELECTED_AND_STRIDES' if not mismatch else 'MISMATCH',oracle=pin,
 prereg_sha256=sha(F.read_bytes()),threads=t.get_num_threads(),summary=s,cases=cases,guards=guards,threaded=threaded,mismatches=mismatch)
with (O/'receipt.json').open('x') as f:json.dump(R,f,sort_keys=True,indent=2);f.write('\n')
print(R['status'],s,mismatch[:1]);raise SystemExit(bool(mismatch))
