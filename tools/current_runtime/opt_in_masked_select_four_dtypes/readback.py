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
doc=json.loads(F.read_bytes()); assert sha(F.read_bytes())=='f1be3964d390da1450e3a418e8e3dbd17fd3f1f6353cb63046b13d73133b409e'
measured=json.loads((oracle/'receipt.json').read_bytes());O={a['label']:a for a in measured['rows']}
oracle_raw=(oracle/'selected.bin').read_bytes();assert sha(oracle_raw)=='be7ce5fb0b7a2ed2d5218ee9eb3f70a88603dc6e2dd21ac831eaa3e00029579b'
RO={a['label']:a for a in doc['cases']}
CAP=262144
T={
'F64':(4,8,(0,0x8000000000000000,1,0x8000000000000001,0x000fffffffffffff,0x800fffffffffffff,0x0010000000000000,
 0x8010000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000001abc,0xfff8000000002abc,
 0x7ff0000000001357,0xfff0000000002468,0x3ff0000000000000,0xbff0000000000000,0x7fefffffffffffff,0xffefffffffffffff)),
'F16':(5,2,(0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,0x7c00,0xfc00,0x7e35,0xfe57,0x7c35,0xfc57,0x3c00,0xbc00,0x7bff,0xfbff)),
'BF16':(6,2,(0,0x8000,1,0x8001,0x007f,0x807f,0x0080,0x8080,0x7f80,0xff80,0x7fc5,0xffd7,0x7f85,0xff97,0x3f80,0xbf80,0x7f7f,0xff7f)),
'Bool':(1,1,(0,1,1,0,0,1,0,1,1,1,0,0,1,0,1,0,0,1))}
def points(s):return itertools.product(*(range(k) for k in s))
def mask(i,c,n):
 v=c['kind']
 if v=='literal':return c['bits'][i]
 if v=='strided':return int(i%c['period'] in c['keep'])
 if v=='edges':return int(i in (0,n-1))
 assert v in ('none','all');return int(v=='all')
def word(i,shape,c,d):
 _,w,p=T[d];v=c['kind'];seed=c['seed']
 if v=='pattern':return p[(i+seed)%len(p)].to_bytes(w,'little')
 if v=='archived_zeros': assert d=='F64';return bytes(w)
 assert v=='finite_f64_exact_add' and d=='F64'
 a,b=divmod(i,shape[1])
 # Both terms and their sum are exact signed integers as binary64.
 return struct.pack('<d',float(((-1 if a&1 else 1)*(2**41+8192*a+seed))+13*b))
def independent(a):
 x=a['value_shape'];m=a['mask_shape']; n=math.prod(x);u=math.prod(m)
 mraw=bytes(mask(i,a['mask'],u) for i in range(u)); vv=bytearray();out=bytearray();align=len(x)-len(m)
 for k,xyz in enumerate(points(x)):
  w=word(k,x,a['source'],a['dtype']);vv+=w
  idx=sum((xyz[align+j] if q!=1 else 0)*math.prod(m[j+1:]) for j,q in enumerate(m))
  if mraw[idx]:out+=w
 z=O[a['label']]; at=z['file_offset']
 assert oracle_raw[at:at+z['byte_len']]==out and sha(out)==a['expected']['raw_sha256']
 assert sha(vv)==a['value_sha256'] and sha(mraw)==a['mask_sha256']
 return bytes(vv),mraw,bytes(out)
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
 seen=set();stat={};positives=words=0
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
  if a['kind']=='large_setup':
   cmd,=Q[k];s0,rr0=R[k];assert cmd[:2]==(2,1) and s0==0 and rr0==[(2,a['handle'],b'')]
   continue
  bb=owner_bytes(c,a['pre']);ab=owner_bytes(c,a['post'])
  if a['kind']=='refusal':
   assert rr==(a['status'],[]) and bb==ab and a['pre']['ptr']==a['post']['ptr'] and sha(U[k])==a['q_sha256']
   stat[lbl]=a['status']
   if lbl in RO:
    rec=RO[lbl];v,m,_=independent(rec);o=Owner(bb);cmd,=Q[k]
    assert cmd[0]==130 and cmd[3:]==(0,)*5
    node(o,cmd[1],rec['dtype_tag'],rec['value_shape'],v);node(o,cmd[2],1,rec['mask_shape'],m)
    assert a['status']==rec['old_status' if H['mode']=='old' else 'new_hypothesized_status']
    seen.add(lbl)
  elif a['kind']=='positive':
   rec=RO[lbl];v,m,raw=independent(rec);hx,hm,h=a['handles'];w=rec['byte_width'];sz=len(raw)//w
   assert Q[k]==[(130,hx,hm,0,0,0,0,0)] and rr==(0,[(130,h,b'')])
   assert bb[128:160]==ab[128:160]
   o=Owner(bb);oa=Owner(ab); allids=[]
   for j,(x,t,sh,xraw,l) in enumerate(((hx,rec['dtype_tag'],rec['value_shape'],v,w),(hm,1,rec['mask_shape'],m,1))):
    id0=node(o,x,t,sh,xraw);assert node(oa,x,t,sh,xraw)==id0; allids.append(id0[:2])
    assert fetch(a['reads_before'][j],x,xraw,l)==fetch(a['reads_after'][j],x,xraw,l)
   outnode=node(oa,h,rec['dtype_tag'],[sz],raw);assert outnode[0] not in [x[0] for x in allids] and outnode[1] not in [x[1] for x in allids]
   meta=fetch(a['output'],h,raw,w);assert meta[:4]==(1,sz,0,sz) and meta[7]==1 and meta[13:]==(0,0,len(raw))
   order=a['order']; fetch(a['survive'],h if order=='inputs-first' else hx,raw if order=='inputs-first' else v,rec['byte_width'])
   wf=a['weak_frame']-1;st,wl=R[wf];assert st==0 and Q[wf][0][:2]==(42,h);wt=wl[0][1]
   wf=a['weak']-1;assert R[wf]==(6,[]) and Q[wf][0][:2]==(43,wt) and Q[wf+1][0][:2]==(44,wt)
   words+=sz; positives+=1;seen.add(lbl)
  else:raise AssertionError(a['kind'])
 assert seen==set(RO)
 if H['mode']=='old': assert positives==0 and len(stat)==73 and all(s==7 for s in stat.values())
 else:
  every={a['label']:a['status'] for a in doc['guards']}
  omitted={'actual-offset-value-view','actual-offset-mask-view','actual-grad-value','stale-capability','padding','valid-then-unknown'}
  assert (c/'error.json').read_text().strip()=='AssertionError((7, [111], 1961))' and len(Q)==1961
  assert Q[-1][0][0]==111 and R[-1]==(7,[])
  assert positives==65 and words==1966536 and all(stat.get(k)==v for k,v in every.items() if k not in omitted)
  assert not set(stat)&omitted
 return dict(frames=len(U),positives=positives,words=words,refusals=stat,elf=H['elf'])
print(json.dumps({d:read_campaign(base/d) for d in ('old','new')},sort_keys=True,indent=2))
