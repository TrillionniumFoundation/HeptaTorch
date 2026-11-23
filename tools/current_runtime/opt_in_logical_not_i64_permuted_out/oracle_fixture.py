"""Official Torch-only fixture; no native execution."""
from pathlib import Path
import hashlib,json,sys
D=Path(__file__).resolve().parent
H=Path(sys.argv[1]);L=Path(sys.argv[2]);I=Path(sys.argv[3])
p=H/'qualify_runtime_torch.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
torch,oracle=load_oracle(L,I)
specs=[([3,2],[1,0],[0,2,0,4,5,0]),
       ([2,3,4],[1,2,0],[0,1,1,1,1,0]+[1]*18),
       ([2,1,3],[2,1,0],[0,1,1,0,1,1])]
def data(t):return bytes(t.contiguous().view(torch.uint8).reshape(-1).tolist()).hex()
rows=[]
for dtype in (0,1,2):
 for bs,perm,values in specs:
  if dtype==2:
   probes=[0,-(1<<63),(1<<63)-1,0,-17,1<<40]
   values=(probes*((len(values)+5)//6))[:len(values)]
  ty={0:torch.float32,1:torch.bool,2:torch.int64}[dtype]
  base=torch.full(bs,True if dtype==1 else -2,dtype=ty)
  d=base.permute(perm); src=torch.tensor(values,dtype=ty).reshape(d.shape); sib=base.view(-1)
  version=d._version; shape,strides=d.size(),d.stride();orig=data(src)
  handle=d.data_ptr(); storage=d.untyped_storage().data_ptr()
  assert not d.is_contiguous() and d.storage_offset()==0 and d._base is base
  assert torch.logical_not(src,out=d) is d
  assert d._version==base._version==sib._version==version+1 and d.data_ptr()==handle
  assert d.untyped_storage().data_ptr()==storage and d.size()==shape and d.stride()==strides and data(src)==orig
  rows.append(dict(dtype=dtype,base_shape=bs,permutation=perm,input=values,shape=list(shape),strides=list(strides),initial=data(src),logical=data(d),base_physical=data(base),version_delta=1))
(D/'official.fixture.json').write_text(json.dumps(dict(rows=rows,oracle=oracle),sort_keys=True)+'\n')
