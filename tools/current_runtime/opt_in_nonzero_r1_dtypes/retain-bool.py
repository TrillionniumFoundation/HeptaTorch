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
oldimg='95aadc064a5a16a188988b699b81dda287443b0e7172fa163dde15db3857c6c7'
assert sha(image.read_bytes())==sha_pin.read_text().strip()
sys.path.insert(0,str(driver))
from test_runtime_candidate import load_wire
assert 'torch' not in sys.modules
a=json.loads((archive/'report.json').read_bytes());assert a['elf']==oldimg and a['native_frames']==370
bools=[x for x in a['rows'] if x['kind']=='positive'];assert len(bools)==17
end=max(x['weak_expired']+1 for x in bools)
Q=list(frames(archive/'requests.frames.bin',b'HTOWNR01'));R=list(frames(archive/'responses.frames.bin',b'HTOWNS01'))
assert len(Q)==len(R)==370
wit=[]
for i,(t,u) in enumerate(zip(Q[:end],R[:end],strict=True),1):
 status,rows=resp(u)
 if status:
  # Ignore old expired weak-acquire checks, never rerun them here.
  assert status==6 and req(t)[0][0]==43
  continue
 wit.append((i,t,u))
rows=[];W=load_wire(proto);S=W.Session(image,directory=out)
try:
 for i,t,u in wit:
  x=S.exchange(t);assert x==u,(i,S.frames,sha(x),sha(u))
  rows.append(dict(original_ordinal=i,ordinal=S.frames,q_sha256=sha(t),r_sha256=sha(u)))
finally:
 (out/'receipt.json').write_text(json.dumps(dict(retained_commands=len(rows),original_successes=len(bools),
  original_ordinary_region_end=end,skipped_expired_weak_checks=end-len(wit),rows=rows),sort_keys=True,indent=2)+'\n')
 S.finish()
assert len(rows)==len(wit)
print(f'PASS {len(rows)} exactly retained ordinary frames, 17 operation successes',flush=True)
