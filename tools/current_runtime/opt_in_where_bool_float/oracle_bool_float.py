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
ENC={0:0x3f800000,4:0x3ff0000000000000,5:0x3c00,6:0x3f80}
def add(label,d,xt,ms,xs,ys,mask,x=None,y=None):
 yt=1 if xt==d else d
 assert {xt,yt}=={1,d} and d in ENC
 nx=math.prod(xs);ny=math.prod(ys)
 assert len(mask)==math.prod(ms)
 def fill(tag,count):
  v=BITS[tag];return (v*(count//len(v)+1))[:count]
 if x is None:x=fill(xt,nx)
 if y is None:y=fill(yt,ny)
 tt=make(1,ms,mask);xx=make(xt,xs,x);yy=make(yt,ys,y)
 a,b,c=map(raw,(tt,xx,yy));vs=[t._version for t in (tt,xx,yy)]
 result=torch.ops.aten.where.self(tt,xx,yy)
 assert str(result.device)=='cpu' and result.dtype==DT[d][2] and result._version==0 and not result.requires_grad,(label,result.dtype)
 assert result.untyped_storage()._cdata not in {t.untyped_storage()._cdata for t in (tt,xx,yy)}
 assert list(map(raw,(tt,xx,yy)))==[a,b,c] and [t._version for t in (tt,xx,yy)]==vs
 shape=list(result.shape)
 w=DT[d][0]
 mb=raw(tt.expand(shape));xb=raw(xx.expand(shape).to(DT[d][2]));yb=raw(yy.expand(shape).to(DT[d][2]))
 expected=b''.join((xb if flag else yb)[i*w:(i+1)*w] for i,flag in enumerate(mb))
 assert a==bytes(mask) and raw(result)==expected,(label,raw(result).hex(),expected.hex())
 rows.append(dict(label=label,dtype=d,width=w,left_dtype=xt,right_dtype=yt,shape=shape,mask_shape=ms,left_shape=xs,right_shape=ys,
  strides=list(result.stride()),mask=mask,left=x,right=y,mask_hex=a.hex(),left_hex=b.hex(),right_hex=c.hex(),result_hex=expected.hex()))
for d in ENC:
 p=BITS[d]; prefix=str(d)+'-'
 for xt in (1,d):
  for flag in (0,1):
   add(prefix+f'rank0-both-{xt}-{flag}',d,xt,[],[],[],[flag],[0] if xt==1 else [p[1]],[p[7]] if xt==1 else [1])
  add(prefix+f'rank0-left-{xt}',d,xt,[3,4],[],[3,4],[1,0]*6,[1] if xt==1 else [p[9]])
  add(prefix+f'rank0-right-{xt}',d,xt,[3,4],[3,4],[],[0,1]*6,None,[p[1]] if xt==1 else [0])
  # Exactly one of each pair of rows selects the target; all 12 adversarial floats are observed.
  m=[1,0,1,1,0,0,1,1,0,1,0,0]
  add(prefix+f'special-words-both-selected-{xt}',d,xt,[2,3,4],[2,3,4],[2,3,4],m+[1-v for v in m])
  add(prefix+f'right-aligned-rank4-{xt}',d,xt,[2,1,4,1],[1,3,4,1],[2,1,1,2],[1,0,1,0,0,1,0,1])
  if xt==1:add(prefix+'empty-bool-left',d,xt,[3,1],[3,0],[3,1],[1,0,1])
  else:add(prefix+'empty-float-left',d,xt,[1,0,1],[2,0,4],[1,0,1],[])
for d in (6,4):
 add(f'historical-{d}-float-left',d,d,[],[2,4],[2,4],[1],list(range(8)),[0,1]*4)
 add(f'historical-{d}-bool-left',d,1,[],[2,4],[2,4],[1],[0,1]*4,list(range(8)))
out.write_text(json.dumps(dict(rows=rows,oracle=oracle),sort_keys=True,indent=2)+'\n')
