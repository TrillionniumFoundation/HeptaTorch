"""Pinned official CPU Torch fixture only; never imports the runtime."""
import hashlib,json,math,sys
from pathlib import Path
H,L,I,out=map(Path,sys.argv[1:])
assert hashlib.sha256((H/'qualify_runtime_torch.py').read_bytes()).hexdigest()=='b0addbda8fb3ba1d42507d7001c8486611a93cb701b397001e325223a645fdbd'
sys.path.insert(0,str(H))
from qualify_runtime_torch import load_oracle
torch,oracle=load_oracle(L,I)
DT={0:(4,torch.int32,torch.float32),1:(1,torch.uint8,torch.bool),2:(8,torch.int64,torch.int64),
    4:(8,torch.int64,torch.float64),5:(2,torch.int16,torch.float16),6:(2,torch.int16,torch.bfloat16)}
def floating(sign,inf,qnan,one):
 return [0,sign,1,sign|1,inf,sign|inf,inf|qnan|3,inf|7,sign|inf|qnan|5,sign|inf|9,one,sign|one]
BITS={0:floating(1<<31,0x7f800000,1<<22,0x3f800000),
      1:[0,1,1,0,1,0,0,1,0,1,0,1],
      2:[0,1,2**64-1,2**63,2**63-1,2**63+1,9,2**64-8,73,2**64-15,37,11],
      4:floating(1<<63,0x7ff0000000000000,1<<51,0x3ff0000000000000),
      5:floating(1<<15,0x7c00,1<<9,0x3c00),
      6:floating(1<<15,0x7f80,1<<6,0x3f80)}
def make(tag,shape,words):
 w,it,t=DT[tag]
 if words: return torch.frombuffer(bytearray(b''.join(a.to_bytes(w,'little') for a in words)),dtype=it).view(t).reshape(shape)
 return torch.empty(shape,dtype=t)
def raw(t):
 return bytes(t.reshape(-1).view(torch.uint8).tolist())
rows=[]
def add(label,tag,shape,flag,x,y,same=False):
 tt=make(1,[],[flag]);xx=make(tag,shape,x);yy=xx if same else make(tag,shape,y)
 a,b,c=map(raw,(tt,xx,yy)); vs=[t._version for t in (tt,xx,yy)]
 result=torch.ops.aten.where.self(tt,xx,yy)
 assert str(result.device)=='cpu' and result.dtype==xx.dtype and list(result.shape)==shape and result._version==0
 assert result.untyped_storage()._cdata not in {t.untyped_storage()._cdata for t in (tt,xx,yy)}
 assert list(map(raw,(tt,xx,yy)))==[a,b,c] and [t._version for t in (tt,xx,yy)]==vs
 expected=b if flag else c
 assert a==bytes([flag]) and raw(result)==expected
 rows.append(dict(label=label,dtype=tag,width=DT[tag][0],shape=shape,strides=list(result.stride()),same=same,
   mask=[flag],left=x,right=y,mask_hex=a.hex(),left_hex=b.hex(),right_hex=c.hex(),result_hex=expected.hex()))
for tag,p in BITS.items():
 add(f'{tag}-true',tag,[3,4],1,p,p[::-1])
 add(f'{tag}-false',tag,[2,2,3],0,p[::-1],p)
 add(f'{tag}-empty',tag,[2,0,3],tag & 1,[],[])
add('4-identical-input',4,[1,2,1,3],0,BITS[4][:6],BITS[4][:6],same=True)
out.write_text(json.dumps(dict(rows=rows,oracle=oracle),sort_keys=True,indent=2)+'\n')
