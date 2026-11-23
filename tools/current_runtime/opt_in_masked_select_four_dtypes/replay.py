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
h,pp,new,old,latest,prior,out,pin,fixture=map(Path,sys.argv[1:])
assert sha(new.read_bytes())==pin.read_text().strip() and sha(old.read_bytes())=='5130bc446381049134363953a687656e7bbfe9bfe6fb87c331383e3acaf27cd8'
assert sha((h/'test_runtime_candidate.py').read_bytes())=='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'
sys.path.insert(0,str(h));from test_runtime_candidate import load_wire
W=load_wire(pp);assert 'torch' not in sys.modules
assert sha(fixture.read_bytes())=='f1be3964d390da1450e3a418e8e3dbd17fd3f1f6353cb63046b13d73133b409e'
A=json.loads(fixture.read_bytes())['historical']
intent=A['other-value-F64-remains-unadmitted'];strict=A['nonbool-value-used-as-mask']
receipt=latest.parent/'compact'/'replay-receipt.json'
assert sha(receipt.read_bytes())=='1084a2a0977e99da27660bd755116cc417655f6e11df129e21ca3e8c9cf6c078'
pins=json.loads(receipt.read_bytes())
assert len(pins['rows'])==10 and pins['frames']==2385
rows=[(x['group'],prior/x['group'],x['frames'],x['requests_sha256'],x['responses_sha256']) for x in pins['rows']]
rows.insert(0,('i64',latest,706,'514c5bfde7d6df7464fe0eda0d7f7695f4292fc7a0f6e5c834fd005fc90de634','2dcadcaa107117f941f05c1c30a217ea7e632993cc3758ae4027f6858fd77adb'))
def captured(path,n,qp=None,rp=None):
 a=path/'requests.frames.bin';b=path/'responses.frames.bin'
 if qp:assert sha(a.read_bytes())==qp and sha(b.read_bytes())==rp
 items=list(zip(frames(a,b'HTOWNR01'),frames(b,b'HTOWNS01'),strict=True));assert len(items)==n;return items
out.mkdir(); ledger=[]
for name,path,n,qp,rp in rows:
 items=captured(path,n,qp,rp); dest=out/name.replace('/','-'); skipped=[]
 s=W.Session(new,directory=dest)
 try:
  for at,(q0,r0) in enumerate(items,1):
   if name=='i64' and sha(q0)==intent['request_sha256']:
    assert at==intent['frame'] and resp(r0)==(7,[]) and sha(r0)==intent['response_sha256']
    skipped.append(at);continue
   ans=s.exchange(q0);assert ans==r0,(name,at,sha(ans),sha(r0))
 finally:s.finish()
 assert (skipped==[685] if name=='i64' else not skipped)
 assert captured(dest,n-len(skipped))==[pair for at,pair in enumerate(items,1) if at not in skipped]
 ledger.append(dict(group=name,retained=s.frames,predecessor_frames=n,excluded=skipped,predecessor_request_sha256=qp,predecessor_response_sha256=rp))
w={}
items=captured(latest,706)
for k,elf in [('old',old),('new',new)]:
 p=out/('witness-'+k);s=W.Session(elf,directory=p)
 try:
  for a,b in items[:684]:assert s.exchange(a)==b
  a,b=items[683];assert sha(a)==strict['request_sha256'] and resp(b)==(7,[])
  a,b=items[684];assert sha(a)==intent['request_sha256'];cmd,=req(a)
  owner=s.owner();o=Owner(owner['bytes'])
  for ht,tag,shape,raw in [(cmd[1],4,(2,3),bytes(48)),(cmd[2],1,(1,3),bytes([1,0,1]))]:
   nd=o.node(ht);assert nd[:2]==(2,math.prod(shape)) and nd[12:14]==shape and nd[72]==tag and nd[29]==nd[83]==0
   assert o.storage(nd[27])[1:]==(len(raw),raw) and o.row(3,nd[28])[1][2]==0
  actual=s.exchange(a);status,r=resp(actual);aft=s.owner()
  if k=='old':assert actual==b and owner==aft
  else:
   assert status==0 and len(r)==1 and r[0][0]==130 and r[0][2]==b''
   o=Owner(aft['bytes']);n=o.node(r[0][1]); assert n[:2]==(1,4) and n[72]==4 and n[16]==1 and n[29]==0
   assert o.storage(n[27])[1:]==(32,bytes(32)) and o.row(3,n[28])[1][2]==0
   assert owner['bytes'][128:160]==aft['bytes'][128:160]
  (p/'pre.zlib').write_bytes(zlib.compress(owner['bytes'],6));(p/'post.zlib').write_bytes(zlib.compress(aft['bytes'],6))
  w[k]=dict(status=status,after=sha(aft['bytes']),before=sha(owner['bytes']),frame=685,actual_response=sha(actual),strict_retained_request=strict['request_sha256'])
 finally:s.finish()
print(json.dumps(dict(byte_exact=True,retired=1,retained=sum(x['retained'] for x in ledger),rows=ledger,new_native=sha(new.read_bytes()),witness=w),sort_keys=True,indent=2))
