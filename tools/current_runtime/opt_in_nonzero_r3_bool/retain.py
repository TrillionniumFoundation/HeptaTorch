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
driver,proto,old_r1,old_r2,old_r2d,image,out,sha_pin=map(Path,sys.argv[1:])
assert sha(image.read_bytes())==sha_pin.read_text().strip()
sys.path.insert(0,str(driver))
from test_runtime_candidate import load_wire
assert 'torch' not in sys.modules
configs=[
 ('rank1',old_r1,'f952bf025e63f6545df424dff55e0d5a64cea196fc06a14d53ea76b5a37a9487',
   1197,50,'8aaa7fe1ff0781e029e17afa82b3a2cd2d3e0dfe509627959bf9306c2fecd117','79ee5cf0b6c2933b141820285550fe9194ffb724ab2eca68410cda33665614b1'),
 ('rank2bool',old_r2,'ac17c641a0e90a08a79dc913e0a3afd2476797ab40d5a48d06190f3b8fd12d33',
   392,23,'47e1e58b06df1d6f491df590c9f87cff974f8860212ec08956bd08a9d3dec025','21fec117ed5acf4f2dd770b162b6189466cf07a2319410c42973d3bcbd24ce6c'),
 ('rank2dtype',old_r2d,'c971732c76049f2b4d760177085f81f73ba602c7880e075625fb8a15bac06c54',
   624,52,'23e9599c697beae9875df4d16f8f4b8c1698de1498c7f49ff41ed0c942f8ef31','034f4bcd0df00647d8a6fe80c1012f0ac1170b946632db6909acd192c5e8b95f'),
]
out.mkdir();W=load_wire(proto);summ={}
for name,archive,oldimg,n,k,qhash,rhash in configs:
 a=json.loads((archive/'report.json').read_bytes());assert a['elf']==oldimg and a['frames']==n and len(a['rows'])==k
 Q=list(frames(archive/'requests.frames.bin',b'HTOWNR01'));R=list(frames(archive/'responses.frames.bin',b'HTOWNS01'))
 assert len(Q)==len(R)==n
 assert sha((archive/'requests.frames.bin').read_bytes())==qhash and sha((archive/'responses.frames.bin').read_bytes())==rhash
 assert all(resp(r)[0]==0 for r in R) and sum(req(t)[0][0]==135 for t in Q)==k
 rows=[];d=out/name;S=W.Session(image,directory=d)
 try:
  for i,(t,u) in enumerate(zip(Q,R,strict=True),1):
   v=S.exchange(t);assert v==u,(name,i,S.frames,sha(v),sha(u))
   rows.append(dict(ordinal=i,q_sha256=sha(t),r_sha256=sha(u)))
 finally:
  sums=dict(retained_commands=len(rows),ordinary_nonzero=k,q_sha256=qhash,r_sha256=rhash)
  (d/'receipt.json').write_text(json.dumps(sums|dict(rows=rows),sort_keys=True,indent=2)+'\n')
  S.finish()
 assert len(rows)==n
 assert sha((d/'requests.frames.bin').read_bytes())==qhash and sha((d/'responses.frames.bin').read_bytes())==rhash
 summ[name]=sums
 (out/'summary.json').write_text(json.dumps(summ,sort_keys=True,indent=2)+'\n')
 print('PASS',name,n,k,flush=True)
