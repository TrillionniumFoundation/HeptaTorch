"""Locked CPU Torch preflight; pure frozen predictions remain unchanged."""
import hashlib,itertools,json,math,sys
from pathlib import Path
H,L,I,D=map(Path,sys.argv[1:])
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
f=(D/'preregister.json').read_bytes()
assert sha(f)=='20d3c5f5383e2ca80cfab1f26884e6ae76031b18c804ed15b86594d5321e735c'
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
def run(a,b):return torch.ops.aten.logical_and.default(a,b)
def row(z):
 a=make(z['left']); b=a if z['owner_relationship']=='same_handle' else make(z['right'])
 before_a,before_b=view(a),view(b)
 for inp,meta,which in ((a,before_a,'left'),(b,before_b,'right')):
  old=z[which]
  assert inp.device.type=='cpu' and inp.is_contiguous() and inp.ndim==len(old['shape']) and inp.numel()==math.prod(old['shape'])
  assert list(meta[0])==old['shape'] and list(meta[1])==old['strides'] and meta[2:4]==(0,0) and meta[-1].hex()==old['raw_hex']
 same=z['owner_relationship']=='same_handle'
 assert (before_a[4]==before_b[4])==same
 r=run(a,b);got=raw(r); expected=z['expected'];n=r.numel()
 assert r.dtype==torch.bool and r.device.type=='cpu' and n==math.prod(expected['shape']) and not r.requires_grad
 assert list(r.shape)==expected['shape'] and list(r.stride())==expected['strides']
 assert r._version==r.storage_offset()==0 and r.untyped_storage()._cdata not in (before_a[4],before_b[4])
 assert before_a==view(a) and before_b==view(b) and set(got)<={0,1}
 assert r.ndim!=0 or n==1
 diff=[dict(index=i,left_class=l,right_class=m,observed=x,expected=y) for i,(l,m,x,y) in
       enumerate(zip(z['left']['raw_classes'],z['right']['raw_classes'],got,bytes.fromhex(expected['raw_hex']),strict=True)) if x!=y]
 return dict(label=z['label'],dtype=z['left']['dtype'],rank=r.ndim,shape=list(r.shape),strides=list(r.stride()),numel=n,
             raw_hex=got.hex(),raw_sha256=sha(got),same_owner_input=same,exact_raw_inputs=True,output_fresh=True,discrepancies=diff)
rows=[row(z) for z in p['rows']]
def control(z):
 a,b=[make(dict(dtype=t,shape=s,raw_hex=(bytes.fromhex('01') if t==1 else (b'\x00'*(types[t][0]-1)+b'\x3f')).hex()*math.prod(s)))
      for t,s in zip(z['tags'],(z['left_shape'],z['right_shape']),strict=True)]
 prior=(view(a),view(b))
 try:out=run(a,b)
 except RuntimeError as e:
  assert z['label']=='equal-count-incompatible' and 'must match' in str(e)
  result=dict(outcome='upstream-shape-error',error=str(e))
 else:
  assert z['label']!='equal-count-incompatible' and out.dtype==torch.bool and set(raw(out))<={0,1}
  assert out.shape==torch.broadcast_shapes(a.shape,b.shape)
  result=dict(outcome='upstream-admits-outside-native-initial-scope',shape=list(out.shape),stride=list(out.stride()),raw_hex=raw(out).hex())
 assert prior==(view(a),view(b))
 return dict(label=z['label'],**result)
guards=list(map(control,p['scope_controls']))
# One exact true transposition, a true positive-offset view, and a real requires-grad input.
base=torch.arange(10,dtype=torch.float32)
trans=base[:6].reshape(2,3).t()
offset=base[2:8].reshape(2,3)
grad=torch.tensor([[0.0,1.0],[2.0,0.0]],requires_grad=True)
single=torch.as_strided(base,(2,1,2),(2,9,1))
for label,a in [('true-noncontiguous-transpose',trans),('true-offset-view',offset),('true-requires-grad',grad),('true-noncanonical-singleton-stride',single)]:
 b=torch.ones(a.shape,dtype=a.dtype)
 old=(view(a),view(b));out=run(a,b)
 assert out.dtype==torch.bool and list(out.shape)==list(a.shape) and not out.requires_grad and raw(out)==bytes(v!=0 for v in a.contiguous().flatten().tolist())
 assert old==(view(a),view(b))
 guards.append(dict(label=label,outcome='upstream-admits-outside-native-initial-scope',shape=list(out.shape),stride=list(out.stride()),
                    source_strides=list(a.stride()),source_storage_offset=a.storage_offset(),source_requires_grad=a.requires_grad,raw_hex=raw(out).hex()))
assert len(rows)==156 and sum(x['numel'] for x in rows)==1101
assert sum(x['rank']==0 for x in rows)==24 and sum(x['same_owner_input'] for x in rows)==6
assert len(guards)==11
delta=[dict(label=x['label'],positions=x['discrepancies']) for x in rows if x['discrepancies']]
srows=sum(any('snan' in c for arg in ('left','right') for c in z[arg]['raw_classes']) for z in p['rows'])
swords=sum(sum('snan' in c for arg in ('left','right') for c in z[arg]['raw_classes']) for z in p['rows'])
output=dict(stage='locked-cpu-torch-only-no-native',op=p['op'],oracle=pin,capability=torch.backends.cpu.get_cpu_capability(),
            threads=torch.get_num_threads(),inter_op_threads=torch.get_num_interop_threads(),preregister_sha256=sha(f),rows=rows,
            semantic_controls=guards,snan_rows=srows,snan_input_words=swords,discrepancies=delta)
(D/'oracle.json').write_text(json.dumps(output,sort_keys=True,indent=2)+'\n')
print('strict',len(rows),'words',sum(x['numel'] for x in rows),'true scalars',24,'same-owner',6,'sNaN rows/inputs',srows,swords,'controls',len(guards),'discrepancies',len(delta))
if delta:raise SystemExit(1)
