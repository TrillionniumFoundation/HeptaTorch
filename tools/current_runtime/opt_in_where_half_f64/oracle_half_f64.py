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
def widen(t,w):
 e,f,b={5:(5,10,15),6:(8,7,127)}[t]
 sign=(w>>15)<<63; exp=(w>>f)&((1<<e)-1); frac=w&((1<<f)-1)
 if exp==(1<<e)-1:return sign|(2047<<52)|(frac<<(52-f))|((1<<51) if frac else 0)
 if exp:return sign|((exp-b+1023)<<52)|(frac<<(52-f))
 if not frac:return sign
 high=frac.bit_length()-1
 return sign|((high+1-b-f+1023)<<52)|((frac^(1<<high))<<(52-high))
def unpack(a): return [w[0] for w in struct.iter_unpack('<Q',raw(a))]
import struct
rows=[];bad=[]; nbad=0
def check(label,tag,actual,desired,source):
 global nbad
 mismatch=0
 for j,(a,d,s) in enumerate(zip(actual,desired,source,strict=True)):
  if a!=d:
   mismatch+=1
   if len(bad)<100:bad.append([label,tag,j,s,a,d])
 nbad+=mismatch
 rows.append([label,tag,len(actual),mismatch,hashlib.sha256(struct.pack(f'<{len(actual)}Q',*actual)).hexdigest()])
def direct(t,shape,words,label):
 x=make(t,shape,words)
 y=x.to(torch.float64)
 assert y.shape==x.shape
 check(label,t,unpack(y),[widen(t,w) for w in words],words)
 return x
def actual(t,half,d,mask,shape,hidx,didx,words,fw,flags,label):
 for rev in (0,1):
  x,y=(d,half) if rev else (half,d)
  m=~mask if rev else mask
  g=torch.ops.aten.where.self(m,x,y)
  assert g.dtype==torch.float64 and g.shape==torch.Size(shape) and g._version==0 and not g.requires_grad
  assert g.untyped_storage()._cdata not in {k.untyped_storage()._cdata for k in (x,y,m)}
  checked=[widen(t,words[h]) if flag else fw[f] for h,f,flag in zip(hidx,didx,flags,strict=True)]
  source=[words[h] if flag else fw[f] for h,f,flag in zip(hidx,didx,flags,strict=True)]
  check(f'where:{label}:order{rev}',t,unpack(g),checked,source)
def add(t,left_half,label,hs,ds,ms,w,f,flags):
 h,d,m=make(t,hs,w),make(4,ds,f),make(1,ms,flags)
 if left_half:x,y,xt,yt,xs,ys,xw,yw=h,d,t,4,hs,ds,w,f
 else:
  x,y,xt,yt,xs,ys,xw,yw=d,h,4,t,ds,hs,f,w
  m=~m;flags=[1-z for z in flags]
 snapshots=list(map(raw,(m,x,y)));vers=[a._version for a in (m,x,y)]
 g=torch.ops.aten.where.self(m,x,y)
 assert g.dtype==torch.float64 and g.device.type=='cpu' and g._version==0 and not g.requires_grad
 assert g.untyped_storage()._cdata not in {a.untyped_storage()._cdata for a in (m,x,y)}
 assert snapshots==list(map(raw,(m,x,y))) and vers==[a._version for a in (m,x,y)]
 rows.append(dict(label=label,dtype=4,width=8,left_dtype=xt,right_dtype=yt,shape=list(g.shape),strides=list(g.stride()),mask_shape=ms,left_shape=xs,right_shape=ys,mask=flags,left=xw,right=yw,mask_hex=snapshots[0].hex(),left_hex=snapshots[1].hex(),right_hex=snapshots[2].hex(),result_hex=raw(g).hex()))
for t in (5,6):
 for order in (0,1):
  for n in (7,8,9,15,16,17):
   for phase in (0,1):
    flags=[(i+j+phase)%2 for i in range(n) for j in range(12)]
    w=[BITS[t][(i+phase*3)%12] for i in range(n)]
    add(t,order,f'profile-{t}-{order}-{n}-{phase}',[n,1],[1,12],[n,12],w,BITS[4],flags)
  for hs,ds in [([0,1],[1,3]),([2,0,1],[1,1,4])]:
   add(t,order,f'empty-{t}-{order}-{len(hs)}',hs,ds,[],[],BITS[4][:ds[-1]],[1])
  add(t,order,f'compound-broadcast-{t}-{order}',[2,1,1,3],[1,2,2,1],[2,1,2,1],BITS[t][6:12],BITS[4][6:10],[1,0,0,1])
for order in (0,1):
 for n in (32767,32768,32769):
  add(5,order,f'large-5-{order}-{n}',[n,1],[1,2],[1,2],[0x7c03 if i%2==0 else 0xfc05 for i in range(n)],BITS[4][6:8],[1,0])
for t in (5,6):
 for order in (0,1):
  add(t,order,f'historical-{t}-{order}',[2,4],[2,4],[],list(range(8)),list(range(8)),[order])
out.write_text(json.dumps(dict(rows=rows,oracle=oracle),sort_keys=True,separators=(',',':'))+'\n')
print(json.dumps(dict(rows=len(rows),words=sum(math.prod(r['shape']) for r in rows))))
