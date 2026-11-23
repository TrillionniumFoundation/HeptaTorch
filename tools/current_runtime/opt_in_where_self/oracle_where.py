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
def add(label,tag,shape,m,x,y,same=False):
 tt=make(1,shape,m);xx=make(tag,shape,x);yy=xx if same else make(tag,shape,y)
 a,b,c=map(raw,(tt,xx,yy)); vs=[t._version for t in (tt,xx,yy)]
 result=torch.ops.aten.where.self(tt,xx,yy)
 assert str(result.device)=='cpu' and result.dtype==xx.dtype and list(result.shape)==shape and result._version==0
 assert result.untyped_storage()._cdata not in {t.untyped_storage()._cdata for t in (tt,xx,yy)}
 assert list(map(raw,(tt,xx,yy)))==[a,b,c] and [t._version for t in (tt,xx,yy)]==vs
 width=DT[tag][0]
 independent=b''.join((b if flag else c)[i*width:(i+1)*width] for i,flag in enumerate(m))
 assert raw(result)==independent
 rows.append(dict(label=label,dtype=tag,width=width,shape=shape,strides=list(result.stride()),same=same,
   mask=m,left=x,right=y,mask_hex=a.hex(),left_hex=b.hex(),right_hex=c.hex(),result_hex=independent.hex()))
for tag,p in BITS.items():
 x=p*2;y=p[5:]+p[:5]+p[4:]+p[:4]
 # Opposite mask in the second half ensures all twelve source words of x can reach output.
 for k,sh in enumerate(([2,12],[2,3,4])):
  a=[1,0,1,1,0,0,1,1,0,1,0,0]; a=a+[1-v for v in a]
  if k: a=[1-v for v in a]
  add(f'{tag}-mixed-rank{k+2}',tag,sh,a,x,y)
 add(f'{tag}-typed-empty',tag,[2,0,3],[],[],[])
add('0-scalar-x',0,[],[1],[0x80000000],[0x7f800001])
add('2-scalar-y',2,[],[0],[1],[2**63])
add('1-all-false',1,[2,3],[0]*6,BITS[1][:6],BITS[1][6:])
add('5-all-true',5,[2,3],[1]*6,BITS[5][:6],BITS[5][6:])
add('4-identical-input',4,[1,2,1,3],[0,1,1,0,1,0],BITS[4][:6],BITS[4][:6],same=True)
out.write_text(json.dumps(dict(rows=rows,oracle=oracle),sort_keys=True,indent=2)+'\n')
