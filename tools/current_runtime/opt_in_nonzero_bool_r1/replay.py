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
H,P,new,pin,saved,old,out=map(Path,sys.argv[1:])
assert sha(new.read_bytes())==pin.read_text().strip()
assert sha((H/'test_runtime_candidate.py').read_bytes())=='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'
sys.path.insert(0,str(H));from test_runtime_candidate import load_wire
W=load_wire(P);assert 'torch' not in sys.modules
ns='ef82340209edc4dfb8d7ed0ecbc65e144fe26078cb29a0ba327798533476e86e'
def pairs(p):
 return list(zip(frames(p/'requests.frames.bin',b'HTOWNR01'),frames(p/'responses.frames.bin',b'HTOWNS01'),strict=True))
def digests(p):
 return [sha((p/k).read_bytes()) for k in ('requests.frames.bin','responses.frames.bin')]
assert digests(saved)==['e072990b7a315f537aec4c35fcd9dbe771f2c0ac5fa9568095094eeadd1b3b3a','e03bb7617c24ad7cde9a44f1843c4f91776c6f1a9517ee320ceb66d6a9f3de4a']
out.mkdir();result=[]
for label,p in [('main-masked-archived',saved),('main-nonzero-setup',old)]:
 proof=json.loads((p/'transport.json').read_bytes());assert proof['executable_sha256']==ns
 a=pairs(p); assert len(a)==proof['frames']
 s=W.Session(new,directory=out/label);skip=[];checked=[]
 try:
  for j,(b,c) in enumerate(a,1):
   ops=req(b);assert len(ops)==1
   if label=='main-nonzero-setup' and ops[0][0]==135:
    assert resp(c)==(4,[]);skip.append(j);continue
   rsp=s.exchange(b);assert rsp==c,(label,j,ops[0][0],sha(rsp),sha(c))
   checked.append((b,c))
 finally:s.finish()
 assert pairs(out/label)==checked
 if label=='main-masked-archived':assert s.frames==290 and skip==[]
 else:assert len(skip)==17
 result.append(dict(group=label,previous_frames=len(a),identical_frames=s.frames,skipped=skip,old_hashes=digests(p),actual_hashes=digests(out/label)))
print(json.dumps(dict(status='PASS',old_main=ns,current=sha(new.read_bytes()),groups=result,exact_comparable=sum(x['identical_frames'] for x in result),
 scope='exact saved 290 main cases plus all newly captured comparable input/metadata/drop; not full hosted aggregate'),sort_keys=True,indent=2))
