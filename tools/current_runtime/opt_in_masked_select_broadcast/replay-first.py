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
H,proto,new,old,root,out,pin=map(Path,sys.argv[1:])
assert sha(new.read_bytes())==pin.read_text().strip() and sha(old.read_bytes())=='75401b00356b8941960fae2a2207c0e046fb0da36cbcef7de26bef41ec4a3deb'
assert sha((H/'test_runtime_candidate.py').read_bytes())=='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'
sys.path.insert(0,str(H)); from test_runtime_candidate import load_wire
W=load_wire(proto)
assert 'torch' not in sys.modules
pins={
'candidate':('d9c0d7729b8e2bb5e6b1d580c454861a8b3f08f685dca94f0c1d8f4dea4fa60a','ac5e761a42d14a6fcc2545ce0bed875fad205e10b6fd4061a060b80d9c2d6216',696),
'guards':('5e2b041b6caac278c5c6e277f7524c3c1f0f0652a985408d2ae75beef2212aed','4a39bce6f56d6784cb52003e07b9248f44e6b2431a9579b0c9a07fb783fa6da7',57),
'replay-main':('cd85438f0e5dc3d59bce7d0cb160188f3b17e72c53fdb8511414b87a8e996ef0','4e569546ca8b506c5830d240f46d1288059ac0bb9c7d002bf4a323b1650e455d',262),
'mirror/sparse-clean':('cfea2e8a26ae8cfc393266f87d73bd58670472f109c87ac179c7c56d68d40323','e19d4f368e20b9727f3b956730b1753963212cce0d04361015c8a166c551e3c8',7),
'mirror/sparse-reject':('24465b4d34b81d615956001df35ca444544a99ac8d3e73d6a3f9ed7b56f5234d','244dd75753cc33c5ef2caaed8a90527be1c8933ab1eee7b9cc90b69cb9bda59f',8),
'mirror/zero-clean':('e2050025b2eedc78bdc671914aaafdba5cb4e5820a3e7681e32feedb88561e61','e19d4f368e20b9727f3b956730b1753963212cce0d04361015c8a166c551e3c8',7),
'mirror/zero-reject':('c6a98ab25be01ee8e99372865d784d6c4b85922f490de87f4bdf2ddfc89e1fa0','244dd75753cc33c5ef2caaed8a90527be1c8933ab1eee7b9cc90b69cb9bda59f',8)}
out.mkdir(); report={}
target='f1d7dbda06b63671bf182cb9bfe163cfbf726545b5bc93222da56097ddb47cea'
def source(group):
 p=root/group; qr=p/'requests.frames.bin'; rr=p/'responses.frames.bin'
 u,v,n=pins[group]; assert sha(qr.read_bytes())==u and sha(rr.read_bytes())==v
 z=list(zip(frames(qr,b'HTOWNR01'),frames(rr,b'HTOWNS01'),strict=True));assert len(z)==n
 return z
for group in pins:
 records=source(group); hits=[]; sx=W.Session(new,directory=out/group.replace('/','-'))
 try:
  for i,(a,b) in enumerate(records,1):
   if sha(a)==target:
    assert group=='guards' and i==20 and resp(b)==(7,[])
    hits.append(i);continue
   got=sx.exchange(a)
   assert got==b,(group,i,resp(got)[0],resp(b)[0],sha(got),sha(b))
 finally:sx.finish()
 assert len(records)-len(hits)==sx.frames
 report[group]=dict(replayed=sx.frames,skipped=hits)
records=source('guards')
witness={}
for kind,elf in [('old',old),('new',new)]:
 sx=W.Session(elf,directory=out/('witness-'+kind))
 try:
  for a,b in records[:19]:assert sx.exchange(a)==b
  a,b=records[19];assert sha(a)==target
  pre=sx.owner();got=sx.exchange(a);after=sx.owner()
  if kind=='old': assert got==b and pre==after
  else:
   s,r=resp(got); assert s==0 and len(r)==1 and r[0][0]==130 and r[0][2]==b''
   h=r[0][1];o=Owner(after['bytes']); n=o.node(h)
   assert n[:2]==(1,6) and o.storage(n[27])[2]==struct.pack('<6I',*range(6))
   assert pre['bytes'][128:160]==after['bytes'][128:160]
  witness[kind]=dict(response_sha256=sha(got),status=resp(got)[0],same_whole_owner=pre==after)
 finally:sx.finish()
print(json.dumps(dict(retained=report,retained_count=sum(v['replayed'] for v in report.values()),exact_changed_witness_frame=20,request_sha256=target,witness=witness),sort_keys=True,indent=2))
