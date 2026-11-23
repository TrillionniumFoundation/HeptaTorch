"""Locked CPU PyTorch only; original preregistration/snapshots immutable."""
import hashlib,json,math,sys
from pathlib import Path
H,L,I,D=map(Path,sys.argv[1:])
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
pr=(D/'preregister.json').read_bytes()
assert sha(pr)=='c343761909fb44e3fd568fa8d192b8950b3e323e756e612bf543e9b69da2e81b'
p=json.loads(pr)
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
torch,pin=load_oracle(L,I);torch.set_num_threads(1);torch.set_num_interop_threads(1)
types={0:(4,torch.int32,torch.float32),1:(1,torch.uint8,torch.bool),2:(8,torch.int64,torch.int64),
       4:(8,torch.int64,torch.float64),5:(2,torch.int16,torch.float16),6:(2,torch.int16,torch.bfloat16)}
ops={'131':torch.ops.aten.logical_and.default,'133':torch.ops.aten.logical_or.default,'134':torch.ops.aten.logical_xor.default}
def make(z):
 n,i,t=types[z['dtype']];raw=bytes.fromhex(z['raw_hex']);shape=z['shape'];assert len(raw)==n*math.prod(shape)
 return torch.frombuffer(bytearray(raw),dtype=i).view(t).reshape(shape) if raw else torch.empty(shape,dtype=t)
def raw(t):return bytes(t.contiguous().reshape(-1).view(torch.uint8).tolist())
def meta(t):return dict(shape=list(t.shape),strides=list(t.stride()),dtype=str(t.dtype),n=t.numel(),version=t._version,
                         offset=t.storage_offset(),storage_bytes=t.untyped_storage().nbytes())
def known(z,t):
 m=meta(t);assert m['shape']==z['shape'] and m['strides']==z['strides'] and m['n']==math.prod(z['shape']) and m['offset']==0
 assert m['version']==0 and m['storage_bytes']==types[z['dtype']][0]*m['n'] and raw(t).hex()==z['raw_hex']
 assert t.dtype==types[z['dtype']][2] and t.device.type=='cpu' and t.is_contiguous()
def measured(x,y,op,z=None):
 a,b=map(make,(x,y));known(x,a);known(y,b);prev=[(meta(t),t.untyped_storage()._cdata,raw(t)) for t in (a,b)]
 r=ops[op](a,b);m=meta(r);n=m['n']
 assert r.device.type=='cpu' and r.dtype==torch.bool and m['storage_bytes']==n and m['version']==m['offset']==0 and not r.requires_grad
 assert r.untyped_storage()._cdata not in [prev[0][1],prev[1][1]]
 logical=raw(r);physical=bytes(r.as_strided((n,),(1,),0).view(torch.uint8).tolist())
 assert len(logical)==len(physical)==n and set(logical)<={0,1} and set(physical)<={0,1}
 assert prev==[(meta(t),t.untyped_storage()._cdata,raw(t)) for t in (a,b)]
 if z is not None:assert m['shape']==z['shape'] and m['n']==z['numel']
 return m,logical,physical
meta_delta=[];records=[];archive=[];caps=[];controls=[]
def annotate(label,op,u,z,logical,physical):
 if u['strides']!=z['strides'] or logical!=physical:
  meta_delta.append(dict(label=label,op=op,shape=u['shape'],actual_strides=u['strides'],canonical_prediction=z['strides'],
      physical_order_matches_logical=logical==physical))
for item in p['rows']:
 for op,z in item['expect'].items():
  m,t,ph=measured(item['left'],item['right'],op,z)
  assert t.hex()==z['raw_hex'],(item['label'],op,t.hex(),z['raw_hex'])
  annotate(item['label'],op,m,z,t,ph)
  records.append(dict(label=item['label'],op=op,**m,raw_hex=t.hex(),physical_hex=ph.hex(),raw_sha256=sha(t)))
for r in p['exact_archived_refusals']:
 x,y=r['operands'];op=str(r['op']);label=r['label']
 try:
  m,t,ph=measured(x,y,op,r['prediction'])
 except RuntimeError as e:
  assert r['after_status']==7 and 'must match' in str(e)
  archive.append(dict(origin=r['origin'],frame=r['frame'],op=op,label=label,outcome='upstream-shape-error',
      source_request_sha256=r['request_sha256'],source_owner_sha256=r['owner_sha256']))
 else:
  z=r['prediction'];assert r['after_status']==0 and t.hex()==z['raw_hex']
  annotate(label,op,m,z,t,ph)
  archive.append(dict(origin=r['origin'],frame=r['frame'],op=op,label=label,outcome='upstream-admitted',**m,
      source_request_sha256=r['request_sha256'],source_owner_sha256=r['owner_sha256'],raw_hex=t.hex(),physical_hex=ph.hex()))
for c in p['capacity']:
 sh=c['output_shape'];n=c['output_count']
 for op in ops:
  m,t,ph=measured(c['left'],c['right'],op,dict(shape=sh,numel=n))
  row=dict(op=op,**m,native_policy=c['predicted_status'],ones=sum(t),sha=sha(t),physical_sha=sha(ph),
           first=t[:8].hex(),last=t[-8:].hex())
  if c['predicted_status']==0:
   y=c['result_digests'][op]
   assert (row['sha'],row['ones'],row['first'],row['last'])==(y['raw_sha256'],y['ones'],y['first'],y['last'])
   annotate('cap-exact',op,m,dict(strides=[sh[1],1]),t,ph)
  caps.append(row)
def make_guard(sh):
 n=math.prod(sh); return dict(dtype=1,shape=sh,raw_hex=('01'*n),
  strides=[math.prod(max(1,v) for v in sh[i+1:]) for i in range(len(sh))])
for c in p['geometry_refusals']:
 for op in ops:
  try:m,t,ph=measured(make_guard(c['left']),make_guard(c['right']),op)
  except RuntimeError as e:
   assert c['upstream'] is None and 'must match' in str(e)
   controls.append(dict(label=c['label'],op=op,outcome='upstream-shape-error'))
  else:
   assert m['shape']==c['upstream'] and not t and not ph
   controls.append(dict(label=c['label'],op=op,outcome='upstream-empty-success-outside-native',**m))
assert len(records)==396 and sum(x['n'] for x in records)==8280
assert len(archive)==12 and sum(r['outcome']=='upstream-admitted' for r in archive)==9
assert len(caps)==6 and len(controls)==12
r=dict(stage='locked-Torch-only',oracle=pin,capability=torch.backends.cpu.get_cpu_capability(),
 threads=torch.get_num_threads(),inter_op_threads=torch.get_num_interop_threads(),preregister_sha256=sha(pr),
 rows=records,archive=archive,capacity=caps,policy=controls,metadata_or_order_discrepancies=meta_delta)
(D/'oracle.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
print('strict',len(records),sum(x['n'] for x in records),'archive',len(archive),'caps',len(caps),'policy',len(controls),'metadata/order discrepancy',len(meta_delta))
if meta_delta:print(json.dumps(meta_delta))
