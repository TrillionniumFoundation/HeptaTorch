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
driver,proto,archive,image,out,sha_pin=map(Path,sys.argv[1:])
oldimg='f952bf025e63f6545df424dff55e0d5a64cea196fc06a14d53ea76b5a37a9487'
assert sha(image.read_bytes())==sha_pin.read_text().strip()
sys.path.insert(0,str(driver))
from test_runtime_candidate import load_wire
assert 'torch' not in sys.modules
a=json.loads((archive/'report.json').read_bytes());assert a['elf']==oldimg and a['frames']==1197 and len(a['rows'])==50
Q=list(frames(archive/'requests.frames.bin',b'HTOWNR01'));R=list(frames(archive/'responses.frames.bin',b'HTOWNS01'))
assert len(Q)==len(R)==1197
assert sha((archive/'requests.frames.bin').read_bytes())=='8aaa7fe1ff0781e029e17afa82b3a2cd2d3e0dfe509627959bf9306c2fecd117'
assert sha((archive/'responses.frames.bin').read_bytes())=='79ee5cf0b6c2933b141820285550fe9194ffb724ab2eca68410cda33665614b1'
assert all(resp(r)[0]==0 for r in R)
assert sum(req(t)[0][0]==135 for t in Q)==50
rows=[];W=load_wire(proto);S=W.Session(image,directory=out)
try:
 for i,(t,u) in enumerate(zip(Q,R,strict=True),1):
  x=S.exchange(t);assert x==u,(i,S.frames,sha(x),sha(u))
  rows.append(dict(ordinal=i,q_sha256=sha(t),r_sha256=sha(u)))
finally:
 (out/'receipt.json').write_text(json.dumps(dict(retained_commands=len(rows),ordinary_rank1_nonzero=50,rows=rows),sort_keys=True,indent=2)+'\n')
 S.finish()
assert len(rows)==1197
print('PASS 1197 exact positive ordinary frames, 50 six-dtype rank1 nonzero',flush=True)
