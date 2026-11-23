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
for t in (5,6):
 for start in range(0,65536,2048):
  w=list(range(start,start+2048))
  h=direct(t,(2048,),w,f'direct:raw:{start}')
  d=make(4,(2048,),[BITS[4][i%12] for i in range(2048)])
  for phase in (0,1):
   flags=[(i+phase)%2 for i in range(2048)]
   actual(t,h,d,make(1,(2048,),flags),(2048,),range(2048),range(2048),w,[BITS[4][i%12] for i in range(2048)],flags,f'raw:{start}:phase{phase}')
 for n in (7,8,9,15,16,17):
  f=BITS[4]
  for phase in (0,1):
   w=[BITS[t][(i+phase*3)%12] for i in range(n)]
   h=direct(t,(n,1),w,f'direct:bcast:{n}:phase{phase}')
   d=make(4,(1,12),f)
   flags=[(i+j+phase)%2 for i in range(n) for j in range(12)]
   actual(t,h,d,make(1,(n,12),flags),(n,12),[i for i in range(n) for j in range(12)],list(range(12))*n,w,f,flags,f'bcast:{n}:phase{phase}')
out.write_text(json.dumps(dict(rows=rows,mismatch_count=nbad,first=bad,oracle=oracle),sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(rows=len(rows),words=sum(r[2] for r in rows),mismatch_count=nbad,first=bad[:6])))
