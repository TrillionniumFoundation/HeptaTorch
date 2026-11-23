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
E=[2**63,2**63+1,2**63-1,2**64-1,0,1,2**53+1,2**64-(2**53+1),17,2**64-24,13,978]
B=[0,1,1,0,1,0,0,1,0,1,0,1]
def add(label,xt,ms,xs,ys,mask,x=None,y=None):
 yt=3-xt; assert {xt,yt}=={1,2}
 nx=math.prod(xs);ny=math.prod(ys)
 assert len(mask)==math.prod(ms)
 words={1:B,2:E}
 def fill(tag,count):
  v=words[tag];return (v*(count//len(v)+1))[:count]
 if x is None:x=fill(xt,nx)
 if y is None:y=fill(yt,ny)
 tt=make(1,ms,mask);xx=make(xt,xs,x);yy=make(yt,ys,y)
 a,b,c=map(raw,(tt,xx,yy));vs=[t._version for t in (tt,xx,yy)]
 result=torch.ops.aten.where.self(tt,xx,yy)
 assert str(result.device)=='cpu' and result.dtype==torch.int64 and result._version==0 and not result.requires_grad
 assert result.untyped_storage()._cdata not in {t.untyped_storage()._cdata for t in (tt,xx,yy)}
 assert list(map(raw,(tt,xx,yy)))==[a,b,c] and [t._version for t in (tt,xx,yy)]==vs
 shape=list(result.shape)
 mb=raw(tt.expand(shape)); xb=raw(xx.expand(shape).to(torch.int64)); yb=raw(yy.expand(shape).to(torch.int64))
 expected=b''.join((xb if flag else yb)[i*8:(i+1)*8] for i,flag in enumerate(mb))
 assert a==bytes(mask) and raw(result)==expected
 rows.append(dict(label=label,dtype=2,width=8,left_dtype=xt,right_dtype=yt,shape=shape,mask_shape=ms,left_shape=xs,right_shape=ys,
   strides=list(result.stride()),mask=mask,left=x,right=y,mask_hex=a.hex(),left_hex=b.hex(),right_hex=c.hex(),result_hex=expected.hex()))
add('scalar-bool-left-select-bool',1,[],[],[],[1],[1],[2**63])
add('scalar-bool-left-select-i64',1,[],[],[],[0],[0],[2**63])
add('scalar-i64-left-select-i64',2,[],[],[],[1],[2**63-1],[1])
add('scalar-i64-left-select-bool',2,[],[],[],[0],[2**63+1],[0])
add('scalar-bool-left-vector-i64',1,[2,4],[],[2,4],[0,0,0,0,1,1,0,0],[1])
add('scalar-i64-left-vector-bool',2,[2,4],[],[2,4],[1,0,0,0,1,1,0,0],[2**64-1])
add('vector-bool-left-scalar-i64',1,[2,4],[2,4],[],[1,0,0,0,1,1,0,0],None,[2**64-(2**53+1)])
add('vector-i64-left-scalar-bool',2,[2,4],[2,4],[],[1,1,1,1,0,0,1,1],None,[0])
add('three-shape-bool-left',1,[2,1,4],[2,3,1],[1,3,4],[1,0,1,0,0,0,1,0])
add('rank4-i64-left',2,[2,1,4,1],[1,3,4,1],[2,1,1,2],[1,0,1,0,0,1,0,1])
add('empty-last-bool-left',1,[3,1],[3,0],[3,1],[1,0,1])
add('empty-interior-i64-left',2,[1,0,1],[2,0,4],[1,0,1],[])
out.write_text(json.dumps(dict(rows=rows,oracle=oracle),sort_keys=True,indent=2)+'\n')
