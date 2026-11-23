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
rows=[]; mismatches=[]
# Independent integer IEEE widening: direct mantissa transfer for normal/special,
# and exact normalization for binary32 subnormals. Preserve sign and payload.
def widen(w):
 sign=(w>>31)<<63
 exp=(w>>23)&255
 frac=w&0x7fffff
 if exp==255:return sign|0x7ff0000000000000|((1<<51) if frac else 0)|(frac<<29)
 if exp:return sign|((exp+896)<<52)|(frac<<29)
 if not frac:return sign
 p=frac.bit_length()-1
 return sign|((p+874)<<52)|((frac-(1<<p))<<(52-p))
SPECIAL={
 0:[0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,
  0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc12345,0xffc23456,0x7f812345,0xff823456,0x7fbfffff,0xffbfffff,0x3f800000,0xbf800000],
 4:[0,1<<63,1,(1<<63)|1,0x000fffffffffffff,0x800fffffffffffff,0x0010000000000000,0x8010000000000000,
  0x7fefffffffffffff,0xffefffffffffffff,0x7ff0000000000000,0xfff0000000000000,0x7ff8123423456789,0xfff823453456789a,
  0x7ff0123423456789,0xfff023453456789a,0x7ff7ffffffffffff,0xfff7ffffffffffff,0x3ff0000000000000,0xbff0000000000000]}
def add(label,xt,ms,xs,ys,mask,x,y):
 yt=4-xt;assert {xt,yt}=={0,4} and xs and ys
 assert len(x)==math.prod(xs) and len(y)==math.prod(ys) and len(mask)==math.prod(ms)
 tt=make(1,ms,mask);xx=make(xt,xs,x);yy=make(yt,ys,y)
 a,b,c=map(raw,(tt,xx,yy));vs=[t._version for t in (tt,xx,yy)]
 result=torch.ops.aten.where.self(tt,xx,yy)
 assert str(result.device)=='cpu' and result.dtype==torch.float64 and result._version==0 and not result.requires_grad,(label,result.dtype)
 assert result.untyped_storage()._cdata not in {t.untyped_storage()._cdata for t in (tt,xx,yy)}
 assert list(map(raw,(tt,xx,yy)))==[a,b,c] and [t._version for t in (tt,xx,yy)]==vs
 shape=list(result.shape)
 coords=lambda sh:itertools.product(*(range(d) for d in sh))
 def pos(sh,ij):
  return tuple(0 if dim==1 else k for dim,k in zip(sh,ij[-len(sh):])) if sh else ()
 source=[dict(zip(coords(sh),z,strict=True)) for sh,z in ((ms,mask),(xs,x),(ys,y))]
 pairs=[]
 for ij in coords(shape):
  take=1 if source[0][pos(ms,ij)] else 2
  word=source[take][pos(xs if take==1 else ys,ij)]
  st=xt if take==1 else yt
  pairs.append((st,word,widen(word) if st==0 else word))
 actual=raw(result)
 expected=b''.join(w.to_bytes(8,'little') for _,__,w in pairs)
 if actual!=expected:
  for i,(st,src,exp) in enumerate(pairs):
   got=int.from_bytes(actual[8*i:8*i+8],'little')
   if exp!=got:mismatches.append(dict(label=label,index=i,source_dtype=st,source=hex(src),expected=hex(exp),observed=hex(got)))
 rows.append(dict(label=label,dtype=4,width=8,left_dtype=xt,right_dtype=yt,shape=shape,mask_shape=ms,left_shape=xs,right_shape=ys,
  strides=list(result.stride()),mask=mask,left=x,right=y,mask_hex=a.hex(),left_hex=b.hex(),right_hex=c.hex(),result_hex=actual.hex()))
for xt in (0,4):
 yt=4-xt
 for n in range(1,66):
  for phase in (0,1):
   x=[SPECIAL[xt][(i+2*n+phase*7)%len(SPECIAL[xt])] for i in range(n)]
   y=[SPECIAL[yt][(i+3*n+phase*5)%len(SPECIAL[yt])] for i in range(n)]
   if n>=2:
    # Both signed signaling payloads near either vector/tail boundary.
    for j in {0,n-1,*[t for t in (7,8,15,16,31,32,63,64) if t<n]}:
     x[j]=SPECIAL[xt][14+((j+phase)&1)]
     y[j]=SPECIAL[yt][16+((j+phase)&1)]
   mask=[(i+phase)&1 for i in range(n)]
   add(f'profile-{xt}-{n}-{phase}',xt,[n],[n],[n],mask,x,y)
 for chosen in (0,4):
  flag=int(xt==chosen);p=SPECIAL[xt];q=SPECIAL[yt]
  add(f'raw-all-{xt}-from-{chosen}',xt,[],[4,5],[4,5],[flag],p,q)
 add(f'compound-broadcast-{xt}',xt,[2,1,4,1],[1,3,4,1],[2,1,1,2],[1,0,1,0,0,1,0,1],SPECIAL[xt][:12],SPECIAL[yt][:4])
 add(f'empty-broadcast-{xt}',xt,[1,0,1],[2,0,4],[1,0,1],[],[],[])
 add(f'historical-f32-f64-{xt}',xt,[],[2,4],[2,4],[1],list(range(8)),list(range(8)))
out.write_text(json.dumps(dict(rows=rows,oracle=oracle,mismatches=mismatches),sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(rows=len(rows),result_words=sum(math.prod(r['shape']) for r in rows),mismatch_count=len(mismatches),first=mismatches[:5])))
