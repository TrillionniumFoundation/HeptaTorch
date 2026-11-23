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
latest,prior,path,receipt,fixture=map(Path,sys.argv[1:])
rec=json.loads(receipt.read_bytes()); f=json.loads(fixture.read_bytes())
assert rec['retained']==3090 and rec['retired']==1 and rec['byte_exact'] and len(rec['rows'])==11
def paired(p):
 return list(zip(frames(p/'requests.frames.bin',b'HTOWNR01'),frames(p/'responses.frames.bin',b'HTOWNS01'),strict=True))
for g in rec['rows']:
 group=g['group']; ref=latest if group=='i64' else prior/group
 a=paired(ref); b=paired(path/group.replace('/','-'))
 assert len(a)==g['predecessor_frames'] and len(b)==g['retained']
 assert sha((ref/'requests.frames.bin').read_bytes())==g['predecessor_request_sha256']
 assert sha((ref/'responses.frames.bin').read_bytes())==g['predecessor_response_sha256']
 assert g['excluded']==([685] if group=='i64' else [])
 assert [r for i,r in enumerate(a,1) if i not in g['excluded']]==b
h=f['historical'];src=paired(latest)
for k in ('old','new'):
 out=path/('witness-'+k); r=paired(out)
 assert len(r)==685 and r[:684]==src[:684] and r[684][0]==src[684][0]
 assert sha(r[683][0])==h['nonbool-value-used-as-mask']['request_sha256'] and resp(r[683][1])==(7,[])
 assert sha(r[684][0])==h['other-value-F64-remains-unadmitted']['request_sha256']
 if k=='old':
  assert r[684]==src[684] and resp(r[684][1])==(7,[])
 else:
  status,row=resp(r[684][1]);assert status==0 and row[0][0]==130
  state=Owner(zlib.decompress((out/'post.zlib').read_bytes())); n=state.node(row[0][1])
  assert n[72]==4 and n[0]==1 and n[1]==4 and state.storage(n[27])[1:]==(32,bytes(32))
print(json.dumps(dict(status='PASS',retained=3090,intended_change_only=1,unchanged_nonbool_historical=True),sort_keys=True))
