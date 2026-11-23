"""Verify immutable v2 ordinary prediction fixture using exact upstream Torch."""
import hashlib,json,sys,os,math
from pathlib import Path
sha=lambda b:hashlib.sha256(b).hexdigest()
H,L,I,F,O=map(Path,sys.argv[1:])
assert sha(F.read_bytes())=='7f492ae8d36bdb1c6523a5a42d09c2009702298df6798ec69258838071d060b5'
assert sha((H/'qualify_runtime_torch.py').read_bytes())=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
t,pin=load_oracle(L,I)
t.set_num_interop_threads(1)
assert t.get_num_threads()==1 and t.get_num_interop_threads()==1 and t.are_deterministic_algorithms_enabled()
doc=json.loads(F.read_bytes()); assert doc['schema'].endswith('.v2')
assert doc['exact_draft_heads']=={'67':'270fde9f8b597c1444fd711026121fef4845b36d','68':'dc4c9fd5efb9526241ea21ac83a4fa5b9196570f','69':'2b5b2fcd8b7937d5441317da1becf9cbef360a1b'}
dtype={k:getattr(t,v) for k,v in dict(Bool='bool',I64='int64',F32='float32',F64='float64',F16='float16',BF16='bfloat16').items()}
ops={130:t.ops.aten.masked_select.default,131:t.ops.aten.logical_and.default,133:t.ops.aten.logical_or.default,134:t.ops.aten.logical_xor.default,135:t.ops.aten.nonzero.default}
def make(spec):
 b=bytes.fromhex(spec['raw_le'])
 if not b: return t.empty(spec['shape'],dtype=dtype[spec['dtype']])
 return t.frombuffer(bytearray(b),dtype=t.uint8).view(dtype[spec['dtype']]).reshape(spec['shape']).clone()
def meta(y):
 return [str(y.dtype),list(y.shape),list(y.stride()),str(y.device),y._version,y.storage_offset(),y.untyped_storage().nbytes(),y.requires_grad,y._is_view()]
def logical(y):
 return y.detach().reshape(-1).contiguous().view(t.uint8).numpy().tobytes()
def physical(y):
 n=y.untyped_storage().nbytes()
 return t.empty(0,dtype=t.uint8).set_(y.untyped_storage(),0,(n,),(1,)).numpy().tobytes() if n else b''
def expected_physical(spec,data):
 if len(spec['shape'])==2 and spec['dtype']=='I64' and spec['strides']==[1,spec['shape'][0]]:
  n,r=spec['shape']
  return b''.join(data[8*(a*r+b):8*(a*r+b+1)] for b in range(r) for a in range(n))
 return data
def getids(x): return int(x._cdata),int(x.untyped_storage()._cdata)
bads=[];rows=[]; nchecks=0; typed_values=0; distinct_physical=0
for case in doc['cases']:
 for mode in ('composed','isolated'):
  scope={key:make(spec) for key,spec in case['inputs'].items()}
  snapshots={key:(meta(val),logical(val),physical(val),getids(val)) for key,val in scope.items()}
  report={}
  for name,op,*args in case['composed_reference']:
   sources=[make(case['expect']['logic']) if mode=='isolated' and key=='logic' else scope[key] for key in args]
   y=ops[op](*sources)
   nchecks+=1
   spec=case['expect'][name]; expected=bytes.fromhex(spec['raw_le'])
   ex_meta=[str(dtype[spec['dtype']]),spec['shape'],spec['strides'],'cpu',0,0,len(expected),False,False]
   actual=meta(y); data=logical(y); storage=physical(y); ex_storage=expected_physical(spec,expected)
   t_id,s_id=getids(y)
   independent=all(getids(obj)[0]!=t_id and getids(obj)[1]!=s_id for obj in [*scope.values(),*sources])
   if mode=='composed': distinct_physical+=op==135 and data!=storage
   if mode=='composed': typed_values+=len(data)//y.element_size()
   report[name]=dict(meta=actual,logical_sha256=sha(data),physical_sha256=sha(storage),fresh_tensor_storage=independent)
   if actual!=ex_meta or data!=expected or storage!=ex_storage or not independent:
    bads.append(dict(id=case['id'],mode=mode,output=name,metadata=actual,expected_metadata=ex_meta,logical_actual=data.hex(),logical_expected=expected.hex(),physical_actual=storage.hex(),physical_expected=ex_storage.hex(),fresh=independent))
   scope[name]=y
  for key,(me,lo,ph,ids) in snapshots.items():
   y=scope[key]
   if (meta(y),logical(y),physical(y),getids(y))!=(me,lo,ph,ids): bads.append(dict(id=case['id'],mode=mode,mutated=key))
  rows.append(dict(id=case['id'],mode=mode,report=report))
O.mkdir(exist_ok=False)
res=dict(status=('PASS' if not bads else 'MISMATCH'),pin=pin,fixture_sha256=sha(F.read_bytes()),heads=doc['exact_draft_heads'],cases=len(doc['cases']),output_checks=nchecks,values_composed=typed_values,distinct_physical_composed=distinct_physical,mismatches=bads,rows=rows,threads=[t.get_num_threads(),t.get_num_interop_threads()])
(O/'receipt.json').write_text(json.dumps(res,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in res.items() if k!='rows'},sort_keys=True))
raise SystemExit(bool(bads))
