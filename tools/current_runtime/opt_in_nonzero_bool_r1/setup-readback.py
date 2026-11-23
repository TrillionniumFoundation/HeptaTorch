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
r,f=map(Path,sys.argv[1:]); doc=json.loads(f.read_bytes())
F={a['label']:a for a in doc['cases'] if a['hypothesized_candidate_status']==3}
receipts={}
for label in ('old-setup','new'):
 p=r/label; rr=json.loads((p/'report.json').read_bytes());rows=[a for a in rr['rows'] if a['kind']=='setup_only']
 a=list(frames(p/'requests.frames.bin',b'HTOWNR01'));b=list(frames(p/'responses.frames.bin',b'HTOWNS01'));assert len(a)==len(b)
 out=[]
 for l in rows:
  j=l['frame']-1; e=F[l['label']]; q0=a[j];r0=b[j]; cmd,=req(q0)
  assert cmd[:3]==(97,1,96) and resp(r0)==(3,[])
  assert q(q0,2,96)==(1,0) and q(q0,1,112)==(262145,) and sha(q0[160:])==e['input_raw_sha256']
  assert l['pre']==l['post'] and sha(q0)==l['q_sha256']
  before=zlib.decompress((p/'owners'/(l['pre']['sha256']+'.zlib')).read_bytes());assert sha(before)==l['pre']['sha256']
  out.append(dict(label=l['label'],frame=l['frame'],request_sha256=sha(q0),response_sha256=sha(r0),owner_sha256=sha(before)))
 assert len(out)==2 and set(x['label'] for x in out)==set(F)
 receipts[label]=out
assert [(x['request_sha256'],x['response_sha256']) for x in receipts['old-setup']]==[(x['request_sha256'],x['response_sha256']) for x in receipts['new']]
print(json.dumps(dict(status='PASS_SETUP_ONLY_NOT_NONZERO_CAP',receipts=receipts),sort_keys=True,indent=2))
