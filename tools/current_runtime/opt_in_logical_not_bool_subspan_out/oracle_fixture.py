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
for dtype in (0,1,2,4):
 for bs,perm,values in specs:
  if dtype==2:
   probes=[0,-(1<<63),(1<<63)-1,0,-17,1<<40]
   values=(probes*((len(values)+5)//6))[:len(values)]
  if dtype==4:
   groups=[
    [0,0x8000000000000000,1,0x8000000000000001,0x7ff0000000000000,0xfff0000000000000],
    [0,0x7ff8000000002345,0x7ff0000000005432,0x8000000000000000,1,0x8000000000000001,
     0x3ff0000000000000,0,0xfff8000000001111,0xfff0000000003210,0x7ff0000000000000,0xfff0000000000000,
     0x000fffffffffffff,0x800fffffffffffff,0,0x8000000000000000,0x0010000000000000,0x8010000000000000,
     0x7fefffffffffffff,0xffefffffffffffff,0x3ff0000000000000,0xbff0000000000000,0,0x8000000000000000],
    [0x7ff8000000002345,0x8000000000000000,0x7ff0000000005432,0,0x8000000000000001,0],
   ]
   values=groups[specs.index((bs,perm,values))]
  ty={0:torch.float32,1:torch.bool,2:torch.int64,4:torch.float64}[dtype]
  base=torch.full(bs,True if dtype==1 else -2,dtype=ty)
  d=base.permute(perm); src=(torch.tensor([v if v<1<<63 else v-(1<<64) for v in values],dtype=torch.int64).view(torch.float64) if dtype==4 else torch.tensor(values,dtype=ty)).reshape(d.shape); sib=base.view(-1)
  if dtype==4:assert data(src)==b''.join(v.to_bytes(8,'little') for v in values).hex()
  version=d._version; shape,strides=d.size(),d.stride();orig=data(src)
  handle=d.data_ptr(); storage=d.untyped_storage().data_ptr()
  assert not d.is_contiguous() and d.storage_offset()==0 and d._base is base
  assert torch.logical_not(src,out=d) is d
  assert d._version==base._version==sib._version==version+1 and d.data_ptr()==handle
  assert d.untyped_storage().data_ptr()==storage and d.size()==shape and d.stride()==strides and data(src)==orig
  rows.append(dict(dtype=dtype,base_shape=bs,permutation=perm,input=values,shape=list(shape),strides=list(strides),initial=data(src),logical=data(d),base_physical=data(base),version_delta=1))
for j in range(4):
 boundary=j==3
 root_values=[-2]*5 if boundary else [101+i for i in range(10)]
 shape,strides,offset=([2,2],[1,2],1) if boundary else ([2,3],[1,2],2)
 values=[0]*4 if boundary else [0 if i>>j&1 else -11-i for i in range(6)]
 base=torch.tensor(root_values,dtype=torch.float32)
 dst=base.as_strided(shape,strides,offset);sib=base.as_strided(shape,strides,offset)
 src=torch.tensor(values,dtype=torch.float32).reshape(shape)
 before=data(base);initial=data(src);version=base._version
 ptr=dst.data_ptr()
 assert not dst.is_contiguous() and dst.storage_offset()==offset
 assert torch.logical_not(src,out=dst) is dst and dst.data_ptr()==ptr
 assert dst._version==base._version==sib._version==version+1 and data(src)==initial
 assert data(sib)==data(dst)
 rows.append(dict(dtype=0,base_shape=[len(root_values)],shape=shape,strides=strides,offset=offset,
   prior_exact_boundary=boundary,root_values=root_values,base_initial=before,input=values,
   initial=initial,logical=data(dst),base_physical=data(base),version_delta=1))
for j in range(4):
 boundary=j==3
 root_values=[1]*5 if boundary else [1,0,1,1,0,0,1,0,0,1]
 shape,strides,offset=([2,2],[1,2],1) if boundary else ([2,3],[1,2],2)
 values=[0]*4 if boundary else [0 if i>>j&1 else 1 for i in range(6)]
 base=torch.tensor(root_values,dtype=torch.bool)
 dst=base.as_strided(shape,strides,offset);sib=base.as_strided(shape,strides,offset)
 src=torch.tensor(values,dtype=torch.bool).reshape(shape)
 before=data(base);initial=data(src);version=base._version
 ptr=dst.data_ptr(); storage=dst.untyped_storage().data_ptr()
 assert not dst.is_contiguous() and dst.storage_offset()==offset and dst._base is base
 assert torch.logical_not(src,out=dst) is dst and dst.data_ptr()==ptr
 assert dst.untyped_storage().data_ptr()==storage
 assert dst._version==base._version==sib._version==version+1 and data(src)==initial
 assert data(sib)==data(dst)
 rows.append(dict(dtype=1,base_shape=[len(root_values)],shape=shape,strides=strides,offset=offset,
   prior_exact_boundary=boundary,root_values=root_values,base_initial=before,input=values,
   initial=initial,logical=data(dst),base_physical=data(base),version_delta=1))
(D/'official.fixture.json').write_text(json.dumps(dict(rows=rows,oracle=oracle),sort_keys=True)+'\n')
