"""Pre-registered independent integer-only expectation generator. No Torch imports."""
from itertools import product
from math import prod
import hashlib,json,sys
from pathlib import Path
L=(1<<63)-1
def nearest_integer(a,precision):
 if a==0:return 0
 grid=1<<max(0,a.bit_length()-precision)
 lo=a-a%grid; hi=lo+grid
 dl,dh=a-lo,hi-a
 return lo if (dl<dh or (dl==dh and (lo//grid)%2==0)) else hi
def reduced(n,tag,intermediate=24):
 assert -(1<<63)<=n<=L and tag in (5,6)
 p,b,k={5:(11,15,5),6:(8,127,8)}[tag]
 a=abs(n)
 if intermediate is not None:a=nearest_integer(a,intermediate)
 a=nearest_integer(a,p)
 if not a:return 0
 e=a.bit_length()-1; s=(n<0)<<15
 if e+b>=(1<<k)-1:return s|(((1<<k)-1)<<(p-1))
 fraction=((a<<(p-1))>>e)&((1<<(p-1))-1)
 return s|((e+b)<<(p-1))|fraction
T={5:[0,0x8000,1,0x8001,0x7bff,0xfbff,0x7c00,0xfc00,0x7e03,0xfe05,0x7c07,0xfc09],
   6:[0,0x8000,1,0x8001,0x7f7f,0xff7f,0x7f80,0xff80,0x7fc3,0xffc5,0x7f87,0xff89]}
KEYS=[(16842753,0x4b80,0x4b81),(16973823,0x4b82,0x4b81),
 (4629700416936870912,0x5e80,0x5e81),(4665729213955832832,0x5e82,0x5e81),
 (9205357638345292800,0x5f00,0x5eff)]
def signed(values):
 return list(dict.fromkeys(y for a in values for y in (a,-a) if 0<=a<=L))+[-(1<<63)]
half_I=signed([0,1,2,2047,2048,2049,2050,2051,32751,32752,32753,32767,32768,32769,*range(65487,65522),65535,65536,65537,L])
bf=[0,1,127,128,129,255,256,257,L,L-1]
for e in (24,25,31,52,53,62):
 h=1<<(e-24)
 for j in (0,1,126,127):
  m=(1<<e)+(2*j+1)*(1<<(e-8))
  bf += [m+s for s in sorted({-h-1,-h,-h+1,-1,0,1,h-1,h,h+1})]
bf_I=signed([*(k[0] for k in KEYS),*bf])
assert all(reduced(n,6)==a and reduced(n,6,None)==reduced(n,6,53)==b and
           reduced(-n,6)==a|32768 and reduced(-n,6,None)==reduced(-n,6,53)==b|32768 for n,a,b in KEYS)
assert [(reduced(n,5),reduced(-n,5)) for n in (65504,65519,65520,65521)]==[(0x7bff,0xfbff),(0x7bff,0xfbff),(0x7c00,0xfc00),(0x7c00,0xfc00)]
def coords(s):return product(*(range(d) for d in s))
def join(shapes):
 r=max(map(len,shapes)); axes=zip(*([(1,)*(r-len(s))+tuple(s) for s in shapes]))
 out=[]
 for axis in axes:
  distinct=set(axis)-{1};assert len(distinct)<=1
  out.append(next(iter(distinct),1))
 return out
def select(src_shape,coord):
 return tuple(0 if d==1 else x for d,x in zip(src_shape,coord[-len(src_shape):])) if src_shape else ()
def fixed(seq,n,offset=0):
 return [seq[(i+offset)%len(seq)] for i in range(n)]
def positive(tag,k,isb,hs,ms,ints,rw,flags):
 assert len(ints)==prod(isb) and len(rw)==prod(hs) and len(flags)==prod(ms)
 shape=join([ms,isb,hs]); r={}
 for reverse in (False,True):
  mask=[b^reverse for b in flags]
  arguments=[(ms,mask),(hs,rw),(isb,ints)] if reverse else [(ms,mask),(isb,ints),(hs,rw)]
  tags=[1,tag,2] if reverse else [1,2,tag]
  maps=[dict(zip(coords(s),words,strict=True)) for s,words in arguments]
  actual=[];counts=[0,0,0];alt_direct=alt_f64=0
  for ijk in coords(shape):
   i=1 if maps[0][select(ms,ijk)] else 2
   v=maps[i][select(arguments[i][0],ijk)]
   if tags[i]==2:
    w=reduced(v,tag);alt_direct+=w!=reduced(v,tag,None);alt_f64+=w!=reduced(v,tag,53)
   else:w=v
   counts[i]+=1
   actual.append(w)
  name=f'{tag}:{k}:{"reduced-left" if reverse else "int-left"}'
  r[name]=dict(label=name,shape=shape,dtype=tag,operands=[dict(shape=s,tag=t,words=w) for t,(s,w) in zip(tags,arguments,strict=True)],
    expected_hex=''.join(v.to_bytes(2,'little').hex() for v in actual),
    selected_int=counts[2 if reverse else 1],selected_reduced=counts[1 if reverse else 2],
    direct_disagreements=alt_direct,f64_disagreements=alt_f64)
 return list(r.values())
def controls():
 for t in (5,6):
  for left,right in ((2,t),(t,2)):
   for sx,sy in (([],[2]),([2],[]),([],[])):
    # This is an HIA scope status, not a PyTorch result claim.
    yield dict(label=f'rank0:{left}:{right}:{len(sx)}:{len(sy)}',expected_hia_status=7,
       operands=[dict(tag=1,shape=[],words=[1]),dict(tag=left,shape=sx,words=[0]*prod(sx)),
                 dict(tag=right,shape=sy,words=[0]*prod(sy))])
rows=[]
for t,I in ((5,half_I),(6,bf_I)):
 for n in (1,7,8,9,15,16,17,31,32,33,65):
  for phase in (0,1):
   ints=fixed(I,n,phase*7+n);rw=fixed(T[t],n,phase*3+n)
   if t==6:
    for i in {0,n-1,*[x for x in (7,8,15,16,31,32,63,64) if x<n]}:
     ints[i]=KEYS[(i+phase)%len(KEYS)][0]*(1 if (i+phase)%2 else -1)
   rows.extend(positive(t,f'boundary-{n}-{phase}',[n],[n],[n],ints,rw,[(i+phase)%2 for i in range(n)]))
 for offset in range(0,len(I),32):
  vals=I[offset:offset+32];n=len(vals)
  rows.extend(positive(t,f'midpoints-{offset}',[n,1],[1,12],[n,12],vals,T[t],[(i+j)%2 for i in range(n) for j in range(12)]))
 rows.extend(positive(t,'empty-prefix',[2,0,1],[1,1,4],[],[],T[t][:4],[1]))
 rows.extend(positive(t,'empty-last',[3,1],[1,0],[1,0],I[:3],[],[]))
 rows.extend(positive(t,'rank4',[2,1,1,3],[1,2,2,1],[2,1,2,1],I[:6],T[t][5:9],[1,0,0,1]))
stop=list(controls())
summary={}
for t in (5,6):
 z=[r for r in rows if r['dtype']==t]
 summary[str(t)]=dict(rows=len(z),words=sum(prod(r['shape']) for r in z),empty=sum(prod(r['shape'])==0 for r in z),
  **{k:sum(r[k] for r in z) for k in ('selected_int','selected_reduced','direct_disagreements','f64_disagreements')})
assert summary['5']['direct_disagreements']==summary['5']['f64_disagreements']==0
assert summary['6']['direct_disagreements']>=summary['6']['f64_disagreements']>0
assert len(stop)==12
dest=Path(sys.argv[1])
blob=(json.dumps(dict(status='PREREGISTERED_NOT_MEASURED',summary=summary,rows=rows,hypothesis=stop),
 sort_keys=True,separators=(',',':'))+'\n').encode()
with dest.open('xb') as f:f.write(blob)
print(json.dumps(dict(summary=summary,rank0_refusals=len(stop),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()),sort_keys=True))
