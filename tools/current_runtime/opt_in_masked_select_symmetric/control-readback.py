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
r,f=map(Path,sys.argv[1:]); raw=f.read_bytes();assert sha(raw)=='d7b8af35d5288fd870c998e0564c3da7c6efcaf88c5723e438cd6e4999493cd1'
p=json.loads(raw); D={x['label']:x for x in p['cases']}; G={x['label']:x['status'] for x in p['guards']}
report=json.loads((r/'report.json').read_bytes()); Q=list(frames(r/'requests.frames.bin',b'HTOWNR01')); R=list(map(resp,frames(r/'responses.frames.bin',b'HTOWNS01')))
assert len(Q)==len(R)==report['native_frames']==1491
check={};caps=[];setups={}
for a in report['rows']:
 label=a.get('label');j=a['frame']-1
 if a['kind']=='setup_only':
  assert req(Q[j])[0][0]==97 and R[j]==(a['status'],[])
  setups[label]=a['status'];continue
 if a['kind']!='refusal' or label.startswith('weak-'): continue
 assert R[j]==(a['status'],[]) and sha(Q[j])==a['q_sha256']
 z=a['pre'];zo=a['post']; assert z==zo
 b=zlib.decompress((r/'owners'/(z['sha256']+'.zlib')).read_bytes());assert sha(b)==z['sha256']
 o=Owner(b);c=req(Q[j]);h=c[0];assert h[0]==130
 if label in D:
  x=D[label];assert x['merged_count']==262656 and x['value_count']==512 and x['mask_count']==513 and a['status']==3
  for k,t,shape,rawsha in [(1,x['tag'],x['value_shape'],x['value_sha256']),(2,1,x['mask_shape'],x['mask_sha256'])]:
   nd=o.node(h[k]);st,vs=nd[27:29];assert nd[72]==t and nd[1]==math.prod(shape) and list(nd[12:12+len(shape)])==shape
   assert nd[4:6]==(0,0) and nd[29]==nd[83]==0 and sha(o.storage(st)[2])==rawsha and o.row(3,vs)[1][2]==0
  caps.append(dict(label=label,input_counts=[512,513],logical_lattice=262656,hypothetical_k=x['expected']['words'],status=3))
 elif label=='actual-view-mask':
  nd=o.node(h[2]);assert (nd[72],nd[29],nd[83])==(1,1,1)
 elif label=='actual-grad-f32':
  nd=o.node(h[1]);assert (nd[72],nd[4])==(0,1)
 elif label.startswith('existing-oversize'):
  nd=o.node(h[1]);assert nd[1]==262656 and nd[72]==1 and a['status']==3
 elif label=='compaction-then-unknown':
  assert len(c)==2 and c[1][0]==65500 and a['status']==4
 if label in G:assert a['status']==G[label]
 check[label]=a['status']
assert len(caps)==12 and len(check)==25 and len(setups)==2
print(json.dumps(dict(status='PASS',measured_controls=check,small_inputs_joined_cap_refusals=caps,setup_not_op130=setups),sort_keys=True,indent=2))
