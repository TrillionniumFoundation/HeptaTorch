"""Locked CPU Torch preflight; pure frozen predictions remain unchanged."""
import hashlib,itertools,json,math,sys
from pathlib import Path
H,L,I,D=map(Path,sys.argv[1:])
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
f=(D/'preregister.json').read_bytes()
assert sha(f)=='299e9a2fc982c4a50d0b023bd0378f7dd6b1c7aaf24a9ea7bd60627173b7614e'
p=json.loads(f)
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
torch,pin=load_oracle(L,I)
torch.set_num_threads(1);torch.set_num_interop_threads(1)
types={0:(4,torch.int32,torch.float32),1:(1,torch.uint8,torch.bool),2:(8,torch.int64,torch.int64),
       4:(8,torch.int64,torch.float64),5:(2,torch.int16,torch.float16),6:(2,torch.int16,torch.bfloat16)}
def make(z):
  w,i,t=types[z['dtype']]; shape=z['shape']; raw=bytes.fromhex(z['raw_hex'])
  return torch.frombuffer(bytearray(raw),dtype=i).view(t).reshape(shape) if raw else torch.empty(shape,dtype=t)
def raw(t):return bytes(t.contiguous().reshape(-1).view(torch.uint8).tolist())
def view(t):return (tuple(t.shape),tuple(t.stride()),t._version,t.storage_offset(),t.untyped_storage()._cdata,raw(t))

P=Path('/tmp/hepta-logical-and-prereg')
and_fixture=(P/'oracle.json').read_bytes()
assert sha(and_fixture)=='c88ded5784f1fbd0e8db86bb48582bb5b5eef2d723217d0cfa0cd7a6f7f1561d'
and_ref={x['label']:x for x in json.loads(and_fixture)['rows']}
old=(D/'old-132.json').read_bytes(); assert sha(old)=='a91ff3dd8903bffb6320db83b8b0c79d4fbb556be546dfb2dd7f6296d99b6749'
archived=json.loads(old)
assert archived['op']==132 and archived['old_status']==4
assert sha((D/'old-132.request.bin').read_bytes())==archived['sha_request']
assert sha((D/'old-132.reply.bin').read_bytes())==archived['sha_reply']
assert p['target_wire_opcodes'] is None
ops={'logical_or':torch.ops.aten.logical_or.default,'logical_xor':torch.ops.aten.logical_xor.default}
all_rows={k:[] for k in ops}; discrepancies=[]
for z in p['strict_rows']:
 a=make(z['left']);b=a if z['relation']=='same_handle' else make(z['right'])
 old_a,old_b=view(a),view(b)
 assert (old_a[4]==old_b[4])==(z['relation']=='same_handle')
 for arg,t,m in ((z['left'],a,old_a),(z['right'],b,old_b)):
  assert list(m[0])==arg['shape'] and list(m[1])==arg['strides'] and m[2:4]==(0,0) and m[-1].hex()==arg['raw_hex']
  assert t.is_contiguous() and t.numel()==math.prod(arg['shape'])>0 and t.device.type=='cpu'
 outputs={}
 for key,fn in ops.items():
  r=fn(a,b); got=raw(r);want=bytes.fromhex(z['outputs'][key])
  assert r.dtype==torch.bool and not r.requires_grad and r.device.type=='cpu' and set(got)<={0,1}
  assert list(r.shape)==z['output_shape'] and list(r.stride())==z['output_strides']
  assert r._version==r.storage_offset()==0 and r.untyped_storage()._cdata not in (old_a[4],old_b[4])
  assert old_a==view(a) and old_b==view(b)
  change=[i for i,(x,y) in enumerate(zip(got,want,strict=True)) if x!=y]
  if change:discrepancies.append(dict(op=key,label=z['label'],changed=change))
  n=r.numel(); same=z['relation']=='same_handle'
  if same and key=='logical_xor':assert got==bytes(n)
  all_rows[key].append(dict(label=z['label'],dtype=z['left']['dtype'],rank=r.ndim,shape=list(r.shape),strides=list(r.stride()),
       raw_hex=got.hex(),raw_sha256=sha(got),numel=n,unchanged_raw_inputs=True,fresh_output=True,same_handle=same))
  outputs[key]=got
 # Historical AND measured receipt is read-only. Verify independent logical identities with actual OR/XOR bits.
 c=and_ref[z['label']];and_r=bytes.fromhex(z['outputs']['logical_and']);assert c['raw_hex']==and_r.hex()
 assert list(c['shape'])==list(a.shape) and list(c['strides'])==list(a.stride())
 assert all(orr==(xorr|andr) and not(xorr&andr) for orr,xorr,andr in zip(outputs['logical_or'],outputs['logical_xor'],and_r,strict=True))
 if z['relation']=='same_handle':assert outputs['logical_or']==and_r
controls={}
def guard_case(fn,z):
 a,b=[make(dict(dtype=t,shape=s,raw_hex=(bytes.fromhex('01') if t==1 else (b'\x00'*(types[t][0]-1)+b'\x3f')).hex()*math.prod(s)))
      for t,s in zip(z['tags'],(z['left_shape'],z['right_shape']),strict=True)]
 prior=(view(a),view(b))
 try:r=fn(a,b)
 except RuntimeError as e:
  assert z['label']=='equal-count-incompatible' and 'must match' in str(e)
  record=dict(outcome='upstream-shape-error',message=str(e))
 else:
  assert z['label']!='equal-count-incompatible' and r.dtype==torch.bool and list(r.shape)==list(torch.broadcast_shapes(a.shape,b.shape)) and set(raw(r))<={0,1}
  record=dict(outcome='upstream-admits-outside-native-initial-scope',shape=list(r.shape),strides=list(r.stride()),raw_hex=raw(r).hex())
 assert prior==(view(a),view(b));return dict(label=z['label'],**record)
for k,op in ops.items():
 rows=all_rows[k]
 assert len(rows)==156 and sum(x['numel'] for x in rows)==1101 and sum(x['same_handle'] for x in rows)==6 and sum(x['rank']==0 for x in rows)==24
 controls[k]=[guard_case(op,c) for c in p['scope_controls']]
assert len(p['strict_rows'])==len(and_ref)==156
r=dict(stage='locked-cpu-torch-only-no-native',oracle=pin,capability=torch.backends.cpu.get_cpu_capability(),
 threads=torch.get_num_threads(),inter_op_threads=torch.get_num_interop_threads(),preregister_sha256=sha(f),rows=all_rows,
 semantic_controls=controls,old_and_locked_receipt_sha256=sha(and_fixture),old_132_unsupported_witness_sha256=sha(old),
 truths_against_historical_and=156,discrepancies=discrepancies)
(D/'oracle.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
print('strict each=156 words each=1101 scalars each=24 shared each=6; controls each=7; AND identities=156; differences',len(discrepancies))
if discrepancies:raise SystemExit(1)
