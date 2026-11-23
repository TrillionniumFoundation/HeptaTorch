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
def add(label,tag,ms,xs,ys,mask,x=None,y=None):
 nx=math.prod(xs);ny=math.prod(ys)
 assert len(mask)==math.prod(ms)
 p=BITS[tag]
 if x is None:x=(p*(nx//len(p)+1))[:nx]
 if y is None:y=(p[::-1]*(ny//len(p)+1))[:ny]
 tt=make(1,ms,mask);xx=make(tag,xs,x);yy=make(tag,ys,y)
 a,b,c=map(raw,(tt,xx,yy));vs=[t._version for t in (tt,xx,yy)]
 result=torch.ops.aten.where.self(tt,xx,yy)
 assert str(result.device)=='cpu' and result.dtype==xx.dtype and result._version==0
 assert result.untyped_storage()._cdata not in {t.untyped_storage()._cdata for t in (tt,xx,yy)}
 assert list(map(raw,(tt,xx,yy)))==[a,b,c] and [t._version for t in (tt,xx,yy)]==vs
 shape=list(result.shape)
 width=DT[tag][0]; mb=raw(tt.expand(shape)); xb=raw(xx.expand(shape)); yb=raw(yy.expand(shape))
 expected=b''.join((xb if flag else yb)[i*width:(i+1)*width] for i,flag in enumerate(mb))
 assert a==bytes(mask) and raw(result)==expected
 rows.append(dict(label=label,dtype=tag,width=width,shape=shape,mask_shape=ms,left_shape=xs,right_shape=ys,
   strides=list(result.stride()),mask=mask,left=x,right=y,mask_hex=a.hex(),left_hex=b.hex(),right_hex=c.hex(),result_hex=expected.hex()))
p=BITS[0]
add('historical-scalar-broadcast-value',0,[],[3,4],[1,4],[1],p,p[:4])
add('historical-trailing-broadcast-value',0,[3,1],[3,4],[1,4],[0,1,0],p,p[:4])
add('historical-general-broadcast-value',0,[1],[3,4],[1,4],[1],p,p[:4])
add('historical-higher-mask-rank',0,[1,3,4],[3,4],[3,4],[0,1]*6,p,p[::-1])
add('three-distinct-axis-contributions',2,[2,1,1],[1,3,1],[1,1,4],[1,0])
add('two-scalar-values-mask-grows',5,[2,3],[],[],[1,0,0,1,1,0],[0x7c03],[0x8000])
add('mixed-missing-and-interior',4,[3,1],[2,1,4],[1,3,4],[0,1,0])
add('rank4-value-lane',6,[2,1,1,1],[1,2,1,3],[1,1,2,1],[0,1])
add('bool-opposite-val-choice',1,[3,1],[3,1],[1,4],[0,1,0])
add('empty-mask-lifts-rank',6,[2,1,1],[0,4],[1,4],[0,1])
add('empty-late-zero-large-prefix',0,[],[512,1,2,0],[1,512,1,1],[1])
add('exact-262144-cap-bool',1,[],[512,1],[1,512],[1],[0,1]*256,[1,0]*256)
out.write_text(json.dumps(dict(rows=rows,oracle=oracle),sort_keys=True,indent=2)+'\n')
