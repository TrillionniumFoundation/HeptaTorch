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
base,fixture=map(Path,sys.argv[1:])
case=json.loads(fixture.read_bytes())['cases'][0]; assert case['label']=='equal-all-signed'
rows={}
for label in ('old','new'):
 p=base/label
 qr=p/'requests.frames.bin';rr=p/'responses.frames.bin'
 Q=list(frames(qr,b'HTOWNR01'));R=list(frames(rr,b'HTOWNS01'))
 j=6; cmd,=req(Q[j]);assert cmd[0]==130 and cmd[3:]==(0,)*5
 expect=7 if label=='old' else 2
 assert resp(R[j])==(expect,[])
 pre=next(iter((p/'owners').glob('*.zlib'))) if label=='new' else p/'owners'/(json.loads((p/'report.json').read_bytes())['rows'][0]['pre']['sha256']+'.zlib')
 b=zlib.decompress(pre.read_bytes());assert sha(b)==pre.stem; o=Owner(b)
 for h,tag,want in [(cmd[1],2,case['pre_source_sha256']),(cmd[2],1,case['pre_mask_sha256'])]:
  d=o.node(h);st,vr=d[27:29];assert d[:2]==(1,20) and d[72]==tag and d[16]==1 and d[29]==d[83]==0
  assert o.row(3,vr)[1][2]==0
  assert sha(o.storage(st)[2])==want
 rows[label]=dict(total_requests=len(Q),total_responses=len(R),frame=7,
  request_sha256=sha(Q[j]),response_sha256=sha(R[j]),full_requests_sha256=sha(qr.read_bytes()),full_responses_sha256=sha(rr.read_bytes()),
  status=expect,signed_and_bool_operands_live=True,pre_owner_sha256=sha(b))
assert rows['new']['request_sha256']==rows['old']['request_sha256']
print(json.dumps(rows,sort_keys=True,indent=2))
