"""Offline wire/owner readback; no runtime, protocol, producer or Torch imports."""
from pathlib import Path
import itertools,math,hashlib,json,struct,sys,zlib
B=139264
MASK=(1<<63)-1
sha=lambda b:hashlib.sha256(b).hexdigest()
def q(raw,n=1,at=0):return struct.unpack_from('<'+'Q'*n,raw,at)
def frames(p,magic):
 data=p.read_bytes();o=0
 while o<len(data):
  assert data[o:o+8]==magic
  n,=q(data,1,o+8);assert 32<=n<=1<<20 and o+16+n<=len(data)
  yield data[o+16:o+16+n];o+=16+n
 assert o==len(data)
def req(x):
 n,ln,z=q(x,3,8);assert x[:8]==b'HTORCH03' and ln==len(x) and z==0
 return [q(x,8,32+64*j) for j in range(n)]
def resp(x):
 v,n,s=q(x,3,8);assert x[:8]==b'HTORSP03' and v==3
 if s:assert n==0 and len(x)==32
 a=[];k=32
 for _ in range(n):
  op,err,t,ln=q(x,4,k);assert err==0;k+=32
  assert k+ln<=len(x);a.append((op,t,x[k:k+ln]));k+=ln
 assert k==len(x);return s,a
class Owner:
 def __init__(self,x):
  assert x[64:72]==b'HTSINT02' and x[B:B+8]==b'HTOSIN02'
  self.b=x;self.slots,self.pool=q(x,2,B+16)
  assert 4<=self.slots<=65536 and self.slots&(self.slots-1)==0
  self.start=B+144+552*self.slots
  assert len(x)==self.start+self.pool
 def row(self,kind,tok):
  slot=tok&0xffffff
  assert 0<slot<=self.slots and (tok>>24)&255==kind
  at=B+128+((kind-1)*self.slots+slot-1)*64
  r=q(self.b,8,at);assert r[:2]==(tok>>32,1)
  return at,r
 def node(self,tok):
  i=self.row(1,tok)[1][6]
  assert 0<i<q(self.b,1,8)[0]
  page,=q(self.b,1,4096+(i>>8)*8);assert page>>63
  at=self.start+(page&MASK)+(i&255)*672
  d=q(self.b,84,at);assert d[24]==tok and d[25]==d[30]==1
  return d
 def storage(self,tok):
  _,r=self.row(2,tok)
  pos=r[2]+self.start
  assert self.start<=pos<=pos+r[3]<=len(self.b)
  return pos,r[3],self.b[pos:pos+r[3]]
base,F,oracle=map(Path,sys.argv[1:])
doc=json.loads(F.read_bytes()); assert sha(F.read_bytes())=='d7b8af35d5288fd870c998e0564c3da7c6efcaf88c5723e438cd6e4999493cd1'
measured=json.loads((oracle/'receipt.json').read_bytes());O={a['label']:a for a in measured['rows']}
oracle_raw=(oracle/'selected.bin').read_bytes();assert sha(oracle_raw)=='0f7dbe529db92d62ebbcac9fa67d9c2875775d58b4493ce5ad6af36633416948'
RO={a['label']:a for a in doc['cases']}
CAP=262144
T={
'F64':(4,8,(0,0x8000000000000000,1,0x8000000000000001,0x000fffffffffffff,0x800fffffffffffff,0x0010000000000000,
 0x8010000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000001abc,0xfff8000000002abc,
 0x7ff0000000001357,0xfff0000000002468,0x3ff0000000000000,0xbff0000000000000,0x7fefffffffffffff,0xffefffffffffffff)),
'F16':(5,2,(0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,0x7c00,0xfc00,0x7e35,0xfe57,0x7c35,0xfc57,0x3c00,0xbc00,0x7bff,0xfbff)),
'BF16':(6,2,(0,0x8000,1,0x8001,0x007f,0x807f,0x0080,0x8080,0x7f80,0xff80,0x7fc5,0xffd7,0x7f85,0xff97,0x3f80,0xbf80,0x7f7f,0xff7f)),
'Bool':(1,1,(0,1,1,0,0,1,0,1,1,1,0,0,1,0,1,0,0,1))}
T['F32']=(0,4,(0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x7f800000,0xff800000,
 0x7fc12345,0xffc23456,0x7f812345,0xff823456,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff))
T['I64']=(2,8,tuple(x&((1<<64)-1) for x in (-(1<<63),-(1<<63)+1,-(1<<62),-(1<<53)-1,-(1<<32)-1,-1,0,1,
 0x7fffffff,0x80000000,0xffffffff,1<<32,(1<<53)+1,(1<<62)+0x5a,0x5555555555555555,(1<<63)-2,(1<<63)-1)))
bytag={v[0]:d for d,v in T.items()}
def points(shape):
 return itertools.product(*(range(n) for n in shape))
def merge(*shapes):
 r=max(map(len,shapes));a,b=[(1,)*(r-len(s))+tuple(s) for s in shapes]
 out=[]
 for y,z in zip(a,b,strict=True): 
  if y==z or z==1:out.append(y)
  elif y==1:out.append(z)
  else:return None
 return tuple(out)
def key(c,shape):
 return tuple(0 if d==1 else k for k,d in zip(c[len(c)-len(shape):],shape,strict=True))
def words(d,shape,seed):
 _,w,p=T[d]; return b''.join(p[(i+seed)%len(p)].to_bytes(w,'little') for i in range(math.prod(shape)))
def flags(m,style):
 count=math.prod(m)
 if style=='all':return b'\1'*count
 if style=='none':return bytes(count)
 if style=='first':return bytes([1])+bytes(count-1)
 if style=='order':return bytes(int((i*7+2)%11 in (0,1,2,3,7,9)) for i in range(count))
 raise AssertionError(style)
def independent(a):
 x=a['value_shape'];m=a['mask_shape'];lattice=a['merged_shape']; v=a['recipe']; d=a['dtype']; w=a['width']
 if 'operands' in v:xraw,mraw=map(bytes.fromhex,v['operands'])
 else: xraw=words(d,x,v['seed']); mraw=flags(m,v['mask_style'])
 # Independent mixed-radix flattened index for each input, not the prereg Cartesian dictionary.
 def idx(pt,shape):
  q=0
  for coord,dim in zip(pt[len(lattice)-len(shape):],shape,strict=True):
   q=q*dim+(coord if dim!=1 else 0)
  return q
 out=bytearray()
 for pt in points(lattice):
  j=idx(pt,m);i=idx(pt,x)
  assert 0<=j<len(mraw) and 0<=i<len(xraw)//w
  if mraw[j]:out+=xraw[i*w:(i+1)*w]
 row=O[a['label']]; n=row['at']; assert oracle_raw[n:n+row['length']]==out and sha(out)==a['expected']['raw_sha256']
 assert sha(xraw)==a['value_sha256'] and sha(mraw)==a['mask_sha256'] and merge(x,m)==tuple(lattice)
 return xraw,mraw,bytes(out)
def owner_bytes(c,e):
 rr=zlib.decompress((c/'owners'/(e['sha256']+'.zlib')).read_bytes());assert len(rr)==e['length'] and sha(rr)==e['sha256'];return rr
def node(b,h,tag,shape,v):
 e=b.node(h); st,vr=e[27:29]
 assert e[72]==tag and e[4:6]==(0,0) and e[29]==e[83]==0
 assert e[:2]==(len(shape),math.prod(shape)) and e[12:12+len(shape)]==tuple(shape)
 assert b.row(1,h)[1][2:6]==(st,vr,0,len(v)) and b.storage(st)[1:]==(len(v),v)
 assert b.row(3,vr)[1][2]==0
 return st,vr,e
def streams(p):
 rq=list(frames(p/'requests.frames.bin',b'HTOWNR01'));rs=list(frames(p/'responses.frames.bin',b'HTOWNS01'))
 return rq,rs
def read_campaign(c):
 H=json.loads((c/'report.json').read_bytes());T=json.loads((c/'transport.json').read_bytes());assert H['elf']==T['executable_sha256'] and T['exit_code']==0 and H['no_torch']
 U,V=streams(c);assert len(U)==len(V)==T['frames']==H['native_frames'];Q=list(map(req,U));R=list(map(resp,V))
 seen=set();stat={};setup={};positives=out_words=0
 def fetch(f,h,w,width):
  j=f['meta_frame']-1;assert Q[j]==[(13,h,0,0,0,0,0,0)]
  st,rr=R[j];assert st==0
  op,m,desc=rr[0];assert (op,m)==(13,h)
  n=len(w)//width; cursor=0; payload=bytearray()
  for slot in f['chunks']:
   k=slot-1;cmd,=Q[k];a,b,start,count,*unused=cmd;assert (a,b,start)==(98,h,cursor) and unused==[0]*4
   st,l=R[k];assert st==0 and l[0][:2]==(98,count) and len(l[0][2])==count*width
   cursor+=count;payload+=l[0][2]
  assert cursor==n and payload==w
  return q(desc,16)
 for a in H['rows']:
  k=a['frame']-1;rr=R[k];lbl=a.get('label')
  bb=owner_bytes(c,a['pre']);ab=owner_bytes(c,a['post'])
  if a['kind']=='setup_only':
   assert Q[k][0][0]==97 and rr==(a['status'],[]) and bb==ab and sha(U[k])==a['request']
   setup[lbl]=a['status'];continue
  if a['kind']=='refusal':
   assert rr==(a['status'],[]) and bb==ab and a['pre']['ptr']==a['post']['ptr'] and sha(U[k])==a['q_sha256']
   stat[lbl]=a['status']
   if lbl in RO:
    rec=RO[lbl];v,m,_=independent(rec);o=Owner(bb);cmd,=Q[k]
    assert cmd[0]==130 and cmd[3:]==(0,)*5
    node(o,cmd[1],rec['tag'],rec['value_shape'],v);node(o,cmd[2],1,rec['mask_shape'],m)
    assert a['status']==rec['old_hypothesized_status' if H['mode']=='old' else 'new_hypothesized_status']
    seen.add(lbl)
  elif a['kind']=='positive':
   rec=RO[lbl];v,m,raw=independent(rec);hx,hm,h=a['handles'];w=rec['width'];sz=len(raw)//w
   assert Q[k]==[(130,hx,hm,0,0,0,0,0)] and rr==(0,[(130,h,b'')])
   assert bb[128:160]==ab[128:160]
   o=Owner(bb);oa=Owner(ab); allids=[]
   for j,(x,t,sh,xraw,l) in enumerate(((hx,rec['tag'],rec['value_shape'],v,w),(hm,1,rec['mask_shape'],m,1))):
    id0=node(o,x,t,sh,xraw);assert node(oa,x,t,sh,xraw)==id0; allids.append(id0[:2])
    assert fetch(a['reads_before'][j],x,xraw,l)==fetch(a['reads_after'][j],x,xraw,l)
   outnode=node(oa,h,rec['tag'],[sz],raw);assert outnode[0] not in [x[0] for x in allids] and outnode[1] not in [x[1] for x in allids]
   meta=fetch(a['output'],h,raw,w);assert meta[:4]==(1,sz,0,sz) and meta[7]==1 and meta[13:]==(0,0,len(raw))
   order=a['order']; fetch(a['survive'],h if order=='inputs-first' else hx,raw if order=='inputs-first' else v,rec['width'])
   wf=a['weak_frame']-1;st,wl=R[wf];assert st==0 and Q[wf][0][:2]==(42,h);wt=wl[0][1]
   wf=a['weak']-1;assert R[wf]==(6,[]) and Q[wf][0][:2]==(43,wt) and Q[wf+1][0][:2]==(44,wt)
   out_words+=sz; positives+=1;seen.add(lbl)
  else:raise AssertionError(a['kind'])
 assert seen==set(RO)
 if H['mode']=='old': assert positives==18 and len(stat)==73 and sum(v==7 for v in stat.values())==55
 else:
  G={a['label']:a['status'] for a in doc['guards']}
  G.pop('existing-oversize-exact-or-oneway-still-cap')
  G.update({('existing-oversize-exact-or-oneway-still-cap-'+k):3 for k in ('exact','scalar-mask')})
  for k in ('rank5-mask','private-tag3-value-stays-forbidden'):G.pop(k)
  assert positives==61 and out_words==4554 and all(stat.get(k)==v for k,v in G.items())
 return dict(frames=len(U),positives=positives,words=out_words,refusals=stat,setup_only=setup,elf=H['elf'])
print(json.dumps({d:read_campaign(base/d) for d in ('old','new')},sort_keys=True,indent=2))
