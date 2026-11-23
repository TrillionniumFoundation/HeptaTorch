"""Pinned official CPU Torch fixture only; never imports the runtime."""
import hashlib,itertools,json,math,sys
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
 return bytes(t.contiguous().reshape(-1).view(torch.uint8).tolist())
rows=[]
def add(label,xt,ms,xs,ys,mask,x,y):
 yt=6-xt;assert {xt,yt}=={0,6} and xs and ys
 assert len(x)==math.prod(xs) and len(y)==math.prod(ys) and len(mask)==math.prod(ms)
 tt=make(1,ms,mask);xx=make(xt,xs,x);yy=make(yt,ys,y)
 a,b,c=map(raw,(tt,xx,yy));vs=[t._version for t in (tt,xx,yy)]
 result=torch.ops.aten.where.self(tt,xx,yy)
 assert str(result.device)=='cpu' and result.dtype==torch.float32 and result._version==0 and not result.requires_grad
 assert result.untyped_storage()._cdata not in {t.untyped_storage()._cdata for t in (tt,xx,yy)}
 assert list(map(raw,(tt,xx,yy)))==[a,b,c] and [t._version for t in (tt,xx,yy)]==vs
 shape=list(result.shape)
 coords=lambda sh:itertools.product(*(range(d) for d in sh))
 def pos(sh,ij):
  return tuple(0 if dim==1 else k for dim,k in zip(sh,ij[-len(sh):])) if sh else ()
 source=[dict(zip(coords(sh),z,strict=True)) for sh,z in ((ms,mask),(xs,x),(ys,y))]
 words=[]
 for ij in coords(shape):
  take=1 if source[0][pos(ms,ij)] else 2
  word=source[take][pos(xs if take==1 else ys,ij)]
  st=xt if take==1 else yt
  words.append(word<<16 if st==6 else word)
 assert raw(result)==b''.join(w.to_bytes(4,'little') for w in words)
 rows.append(dict(label=label,dtype=0,width=4,left_dtype=xt,right_dtype=yt,shape=shape,mask_shape=ms,left_shape=xs,right_shape=ys,
  strides=list(result.stride()),mask=mask,left=x,right=y,mask_hex=a.hex(),left_hex=b.hex(),right_hex=c.hex(),result_hex=raw(result).hex()))
for xt in (0,6):
 yt=6-xt
 add(f'compound-broadcast-{xt}',xt,[2,1,4,1],[1,3,4,1],[2,1,1,2],[1,0,1,0,0,1,0,1],BITS[xt],BITS[yt][:4])
 add(f'empty-broadcast-{xt}',xt,[1,0,1],[2,0,4],[1,0,1],[],[],[])
 add(f'empty-last-{xt}',xt,[3,1],[3,0],[3,1],[1,0,1],[],BITS[yt][:3])
out.write_text(json.dumps(dict(rows=rows,oracle=oracle),sort_keys=True,indent=2)+'\n')
