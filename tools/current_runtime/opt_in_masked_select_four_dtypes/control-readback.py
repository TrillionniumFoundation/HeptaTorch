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
x,su,tail,f=map(Path,sys.argv[1:])
fs=json.loads(f.read_bytes()); expected={z['label']:z['status'] for z in fs['guards']}
expected.update({z['label']:3 for z in fs['cases'] if z['new_hypothesized_status']==3})
explicit={};seen=set();blocked={}
for c in (x,su,tail):
 p=json.loads((c/'report.json').read_bytes())
 act=[e for e in p['rows'] if e['kind']=='refusal' and not e['label'].startswith('weak-')]
 Q=list(frames(c/'requests.frames.bin',b'HTOWNR01'));R=list(map(resp,frames(c/'responses.frames.bin',b'HTOWNS01')))
 for trace in act:
  label=trace['label'];j=trace['frame']-1
  assert label not in seen;seen.add(label)
  assert sha(Q[j])==trace['q_sha256'] and R[j]==(expected[label],[])
  cmd=req(Q[j]);assert cmd[0][0]==130
  a=zlib.decompress((c/'owners'/(trace['pre']['sha256']+'.zlib')).read_bytes())
  b=zlib.decompress((c/'owners'/(trace['post']['sha256']+'.zlib')).read_bytes())
  assert a==b and sha(a)==trace['pre']['sha256']==trace['post']['sha256'] and trace['pre']['ptr']==trace['post']['ptr']
  o=Owner(a)
  if label=='actual-offset-mask-view':
   n=o.node(cmd[0][2]); assert (n[72],n[83],n[29])==(1,1,1)
   assert o.row(2,n[27])[1][1] and o.row(3,n[28])[1][1]
   explicit[label]=dict(dtype_tag=1,offset_bytes=1,view_bit=1)
  if label=='actual-grad-value':
   n=o.node(cmd[0][1]); assert n[72]==5 and n[4]==1
   explicit[label]=dict(dtype_tag=5,actual_requires_grad=1)
  if expected[label]==3:
   v=o.node(cmd[0][1]); assert v[1]==262656 and v[72] in (1,4,5,6)
  if label=='valid-then-unknown':assert len(cmd)==2 and cmd[1][0]==65500
 if c==x:
  assert len(Q)==1961 and req(Q[-1])[0][0]==111 and R[-1]==(7,[])
  blocked['actual-offset-value-view']=dict(frame=len(Q),op=111,status=7,request_sha256=sha(Q[-1]),outcome='NOT_AN_OP130_REFUSAL')
 if c==su:
  assert len(Q)==13 and req(Q[-1])[0][0]==90 and R[-1]==(5,[])
  blocked['actual-grad-value']=dict(frame=len(Q),op=90,status=5,request_sha256=sha(Q[-1]),outcome='NOT_AN_OP130_REFUSAL')
assert set(expected)-seen=={'actual-offset-value-view','actual-grad-value'}==set(blocked)
print(json.dumps(dict(status='PASS_SCOPED',controls={k:expected[k] for k in sorted(seen)},explicit=explicit,unmeasured_setup=blocked),sort_keys=True,indent=2))
