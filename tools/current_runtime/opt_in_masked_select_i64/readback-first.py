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
doc=json.loads(F.read_bytes()); assert sha(F.read_bytes())=='81aad2d489f77adbba131bf703456ca5e7fb04ad9876197376307bc128d3cc19'
measured=json.loads((oracle/'receipt.json').read_bytes());O={a['label']:a for a in measured['rows']}
oracle_raw=(oracle/'selected-i64le.bin').read_bytes();assert sha(oracle_raw)=='160d36a596e3d04b8e21b96491a649013fdb4e354eaee1a8215f213fb81277db'
RO={a['label']:a for a in doc['cases']}
LOW=-(1<<63); HIGH=(1<<63)-1
PAT=(LOW,LOW+1,LOW+0xffffffff,LOW+(1<<40),-(1<<62),-(1<<53)-1,-(1<<32)-1,-(1<<32),-1,
 0,1,0x7fffffff,0x80000000,0xffffffff,1<<32,(1<<53)+1,(1<<62)+0x5a,0x5555555555555555,HIGH-1,HIGH)
def independent(a):
 x=a['value_shape'];m=a['mask_shape']; n=math.prod(x);u=math.prod(m)
 mode=a['mask'];var=mode['kind'];src=a['source']; seed=src['seed']
 if var=='literal':mraw=bytes(mode['bits'])
 elif var=='strided':mraw=bytes(int(i%mode['period'] in mode['remainders']) for i in range(u))
 elif var=='edges':mraw=bytes(int(i in {0,u-1}) for i in range(u))
 else:
  assert var in ('all','none');mraw=bytes([int(var=='all')])*u
 assert len(mraw)==u
 v=bytearray();raw=bytearray();dx=len(x)-len(m)
 for i,c in enumerate(itertools.product(*(range(d) for d in x))):
  if src['kind']=='signed_pattern': w=PAT[(i+seed)%len(PAT)]
  else:
   assert src['kind']=='i64_add_cartesian'
   arow,acol=c;q0=arow*8192+seed; j=arow%4;base=[LOW,-(1<<60),1<<61,HIGH-(1<<24)][j]
   w=base+q0*(1 if j!=1 else -1)+13*acol
  word=struct.pack('<q',w);v+=word
  index=sum((c[dx+j] if md!=1 else 0)*math.prod(m[j+1:]) for j,md in enumerate(m))
  if mraw[index]:raw+=word
 opt=O[a['label']];start=opt['offset'];end=start+opt['length']
 assert len(v)==8*n and oracle_raw[start:end]==raw and sha(raw)==a['expected']['torch_raw_sha256']
 assert sha(v)==a['pre_source_sha256'] and sha(mraw)==a['pre_mask_sha256']
 return bytes(v),mraw,bytes(raw)
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
   cmd,=Q[k];s0,rr0=R[k];assert cmd[:2]==(103,1) and s0==0 and rr0==[(103,a['handle'],b'')]
   continue
  bb=owner_bytes(c,a['pre']);ab=owner_bytes(c,a['post'])
  if a['kind']=='refusal':
   assert rr==(a['status'],[]) and bb==ab and a['pre']['ptr']==a['post']['ptr'] and sha(U[k])==a['q_sha256']
   stat[lbl]=a['status']
   if lbl in RO:
    rec=RO[lbl];v,m,_=independent(rec);o=Owner(bb);cmd,=Q[k]
    assert cmd[0]==130 and cmd[3:]==(0,)*5
    node(o,cmd[1],2,rec['value_shape'],v);node(o,cmd[2],1,rec['mask_shape'],m)
    assert a['status']==rec['old_status' if H['mode']=='old' else 'new_hypothesized_status']
    seen.add(lbl)
  elif a['kind']=='positive':
   rec=RO[lbl];v,m,raw=independent(rec);hx,hm,h=a['handles'];sz=len(raw)//8
   assert Q[k]==[(130,hx,hm,0,0,0,0,0)] and rr==(0,[(130,h,b'')])
   assert bb[128:160]==ab[128:160]
   o=Owner(bb);oa=Owner(ab); allids=[]
   for j,(x,t,sh,w,l) in enumerate(((hx,2,rec['value_shape'],v,8),(hm,1,rec['mask_shape'],m,1))):
    id0=node(o,x,t,sh,w);assert node(oa,x,t,sh,w)==id0; allids.append(id0[:2])
    assert fetch(a['reads_before'][j],x,w,l)==fetch(a['reads_after'][j],x,w,l)
   outnode=node(oa,h,2,[sz],raw);assert outnode[0] not in [x[0] for x in allids] and outnode[1] not in [x[1] for x in allids]
   meta=fetch(a['output'],h,raw,8);assert meta[:4]==(1,sz,0,sz) and meta[7]==1 and meta[13:]==(0,0,len(raw))
   order=a['order']; fetch(a['survive'],h if order=='inputs-first' else hx,raw if order=='inputs-first' else v,8)
   wf=a['weak_frame']-1;st,wl=R[wf];assert st==0 and Q[wf][0][:2]==(42,h);wt=wl[0][1]
   wf=a['weak']-1;assert R[wf]==(6,[]) and Q[wf][0][:2]==(43,wt) and Q[wf+1][0][:2]==(44,wt)
   words+=sz; positives+=1;seen.add(lbl)
  else:raise AssertionError(a['kind'])
 assert seen==set(RO)
 if H['mode']=='old': assert positives==0 and len(stat)==24 and all(s==7 for s in stat.values())
 else: assert positives==22 and words==365697 and set(stat).issuperset(a['label'] for a in doc['guards'])
 return dict(frames=len(U),positives=positives,words=words,refusals=stat,elf=H['elf'])
print(json.dumps({d:read_campaign(base/d) for d in ('old','new')},sort_keys=True,indent=2))
