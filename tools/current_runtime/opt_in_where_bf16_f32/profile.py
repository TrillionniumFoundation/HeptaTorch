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
rows=[];bad=[]
from collections import Counter
from itertools import combinations
def as32(t,w):
 if t==0:return w
 if t==6:return w<<16
 exp=(w>>10)&31; frac=w&1023; sign=(w&32768)<<16
 if exp==31:return sign|0x7f800000|(frac<<13)|(0x400000 if frac else 0)
 if exp:return sign|((exp+112)<<23)|(frac<<13)
 if not frac:return sign
 high=frac.bit_length()-1
 return sign|((high+103)<<23)|((frac-(1<<high))<<(23-high))
def specs(t):
 exp,frac={0:(8,23),5:(5,10),6:(8,7)}[t]
 sign=1<<(exp+frac);inf=((1<<exp)-1)<<frac;quiet=1<<(frac-1);normal=(1<<(exp-1))-1
 return [0,sign,1,sign|1,(1<<frac)-1,sign|((1<<frac)-1),1<<frac,sign|(1<<frac),inf-1,sign|(inf-1),
       inf,sign|inf,inf|quiet|3,sign|inf|quiet|5,inf|7,sign|inf|9,inf|(quiet-1),sign|inf|(quiet-1),
       normal<<frac,sign|(normal<<frac)]
V={i:specs(i) for i in (0,5,6)}
def nan_kind(t,w):
 exp,frac={0:(8,23),5:(5,10),6:(8,7)}[t]
 if (w>>frac)&((1<<exp)-1) != (1<<exp)-1:return 'finite'
 f=w&((1<<frac)-1)
 return 'inf' if f==0 else 'qnan' if f>>(frac-1) else 'snan'
def add(pair,xt,yt,n,phase,mask,x,y,label):
 tt=make(1,[n],mask); xx=make(xt,[n],x); yy=make(yt,[n],y)
 result=torch.ops.aten.where.self(tt,xx,yy)
 assert str(result.device)=='cpu' and result.dtype==torch.float32 and result._version==0
 assert list(result.shape)==[n] and not result.requires_grad
 assert result.untyped_storage()._cdata not in {t.untyped_storage()._cdata for t in (tt,xx,yy)}
 actual=raw(result)
 expected=[as32(xt,z) if m else as32(yt,w) for m,z,w in zip(mask,x,y)]
 ob=[int.from_bytes(actual[i:i+4],'little') for i in range(0,len(actual),4)]
 source=[(xt,z) if m else (yt,w) for m,z,w in zip(mask,x,y)]
 row_misses=0
 for i,(e,g,(t,w)) in enumerate(zip(expected,ob,source)):
  if e!=g:
   row_misses+=1
   bad.append(dict(pair=pair,left_dtype=xt,right_dtype=yt,n=n,phase=phase,index=i,dtype=t,word=w,
    kind=nan_kind(t,w),expected=e,observed=g,label=label))
 rows.append(dict(pair=pair,left_dtype=xt,right_dtype=yt,n=n,phase=phase,label=label,mask=mask,left=x,right=y,actual=ob,
   expected=expected,mismatches=row_misses))
for lt,rt in combinations((0,5,6),2):
 pair=f'{lt}+{rt}'
 for xt,yt in ((lt,rt),(rt,lt)):
  for n in range(1,66):
   for phase in (0,1):
    x=[V[xt][(i+2*n+phase*7)%len(V[xt])] for i in range(n)]
    y=[V[yt][(i+3*n+phase*5)%len(V[yt])] for i in range(n)]
    near={0,n-1,*[t for t in (7,8,15,16,31,32,63,64) if t<n]}
    for j in near:
     x[j]=V[xt][14+((j+phase)&3)]
     y[j]=V[yt][14+((j+phase+1)&3)]
    mask=[(i+phase)&1 for i in range(n)]
    add(pair,xt,yt,n,phase,mask,x,y,f'{pair}-order{xt}-n{n}-p{phase}')
  for source in (xt,yt):
   n=20;flag=int(source==xt)
   add(pair,xt,yt,n,flag,[flag]*n,V[xt],V[yt],f'{pair}-order{xt}-full{source}')
summary={}
for p in sorted(set(r['pair'] for r in rows)):
 subset=[r for r in rows if r['pair']==p];m=[b for b in bad if b['pair']==p]
 groups=Counter((b['dtype'],b['kind']) for b in m)
 corrections=Counter((b['dtype'],b['kind'],b['expected']^b['observed']) for b in m)
 coverage={}
 for r in subset:
  for f,x,y in zip(r['mask'],r['left'],r['right']):
   t,w=(r['left_dtype'],x) if f else (r['right_dtype'],y)
   k,sg=nan_kind(t,w), '-' if w>>(DT[t][0]*8-1) else '+'
   coverage[f'{t}:{sg}{k}']=coverage.get(f'{t}:{sg}{k}',0)+1
 summary[p]=dict(rows=len(subset),words=sum(r['n'] for r in subset),mismatch_words=len(m),
    mismatch_rows=sum(r['mismatches']!=0 for r in subset),groups={f'{x}:{y}':v for (x,y),v in groups.items()},
    xor_corrections={f'{x}:{y}:{hex(z)}':v for (x,y,z),v in corrections.items()},
    mismatching_lengths=sorted(set(b['n'] for b in m)),selected_classes=coverage,first=m[:8])
out.write_text(json.dumps(dict(rows=rows,oracle=oracle,mismatches=bad,summary=summary),sort_keys=True,indent=2)+'\n')
print(json.dumps(summary,sort_keys=True))
