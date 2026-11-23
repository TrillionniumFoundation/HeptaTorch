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
base,F,oracle,prior=map(Path,sys.argv[1:])
doc=json.loads(F.read_bytes()); assert sha(F.read_bytes())=='4f782478f42b9c3639ba3bfed921a8a932c8bb3dff2772096f00e312ae916f18'
measured=json.loads((oracle/'receipt.json').read_bytes());O={a['label']:a for a in measured['rows']}
oracle_raw=(oracle/'raw.bin').read_bytes();assert sha(oracle_raw)=='622feda4edf6e5510b7cdb05bc52f71c9584c149b12738e7713ef019f2f8972d'
RO={a['label']:a for a in doc['cases']}
PAT=(0,0x80000000,1,0x80000001,0x7fffff,0x807fffff,0x800000,0x80800000,0x7f800000,0xff800000,0x7fc12345,0xffc23456,0x7f812345,0xff823456,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff)
def independent(a):
 x=a['x_shape'];m=a['mask_shape']; n=math.prod(x);u=math.prod(m)
 b=bytearray.fromhex(a.get('mask_raw_u8_hex',''))
 if 'mask_formula' in a:
  b=bytearray(u);b[a['mask_formula']['index']]=1
 assert len(b)==u
 raw=bytearray();v=bytearray();sk=a['source']['kind'];seed=a['source']['seed'];diff=len(x)-len(m)
 # Independent oracle: expand masks as nested coordinate key table, then gather slices of value bytes.
 for i,point in enumerate(itertools.product(*(range(d) for d in x))):
  eye=tuple(t for j,t in enumerate(point) if j>=diff)
  keys=[0 if m[axis]==1 else t for axis,t in enumerate(eye)]
  index=sum(t*math.prod(m[axis+1:]) for axis,t in enumerate(keys))
  w=i if sk=='archive-exact' else (0x3f000000+13*i+seed if sk=='ordinal' else PAT[(i+seed)%len(PAT)])
  word=struct.pack('<I',w);v+=word
  if b[index]:raw+=word
 opt=O[a['label']];start=opt['offset'];end=start+opt['length']
 assert len(v)==4*n and oracle_raw[start:end]==raw and sha(raw)==a['expected']['raw_u32le_sha256']
 return bytes(v),bytes(b),bytes(raw)
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
  bb=owner_bytes(c,a['pre']);ab=owner_bytes(c,a['post'])
  if a['kind']=='refusal':
   assert rr==(a['status'],[]) and bb==ab and a['pre']['ptr']==a['post']['ptr'] and sha(U[k])==a['q_sha256']
   stat[lbl]=a['status']
   if lbl in RO:
    rec=RO[lbl];v,m,_=independent(rec);o=Owner(bb);cmd,=Q[k]
    assert cmd[0]==130 and cmd[3:]==(0,)*5
    node(o,cmd[1],0,rec['x_shape'],v);node(o,cmd[2],1,rec['mask_shape'],m)
    assert a['status']==rec['old_hypothesized_status' if H['mode']=='old' else 'new_hypothesized_status']
    seen.add(lbl)
  elif a['kind']=='positive':
   rec=RO[lbl];v,m,raw=independent(rec);hx,hm,h=a['handles'];sz=len(raw)//4
   assert Q[k]==[(130,hx,hm,0,0,0,0,0)] and rr==(0,[(130,h,b'')])
   assert bb[128:160]==ab[128:160]
   o=Owner(bb);oa=Owner(ab); allids=[]
   for j,(x,t,sh,w,l) in enumerate(((hx,0,rec['x_shape'],v,4),(hm,1,rec['mask_shape'],m,1))):
    id0=node(o,x,t,sh,w);assert node(oa,x,t,sh,w)==id0; allids.append(id0[:2])
    assert fetch(a['reads_before'][j],x,w,l)==fetch(a['reads_after'][j],x,w,l)
   outnode=node(oa,h,0,[sz],raw);assert outnode[0] not in [x[0] for x in allids] and outnode[1] not in [x[1] for x in allids]
   meta=fetch(a['output'],h,raw,4);assert meta[:4]==(1,sz,0,sz) and meta[7]==1 and meta[13:]==(0,0,len(raw))
   order=a['order']; fetch(a['survive'],h if order=='inputs-first' else hx,raw if order=='inputs-first' else v,4)
   wf=a['weak_frame']-1;st,wl=R[wf];assert st==0 and Q[wf][0][:2]==(42,h);wt=wl[0][1]
   wf=a['weak']-1;assert R[wf]==(6,[]) and Q[wf][0][:2]==(43,wt) and Q[wf+1][0][:2]==(44,wt)
   words+=sz; positives+=1;seen.add(lbl)
  else:
   assert a['kind']=='scratch' and q(bb,4,128)==q(ab,4,128)==(2,3,0,0)
   assert rr[0]==0 and rr[1][-1][2]==struct.pack('<4I',1,2,4,5)
 assert seen==set(RO)
 if H['mode']=='old': assert positives==0 and len(stat)==23 and all(s==7 for s in stat.values())
 else: assert positives==20 and words==264891 and set(stat).issuperset(a['label'] for a in doc['guards'])
 return dict(frames=len(U),positives=positives,words=words,refusals=stat,elf=H['elf'])
print(json.dumps({d:read_campaign(base/d) for d in ('old','new')},sort_keys=True,indent=2))
