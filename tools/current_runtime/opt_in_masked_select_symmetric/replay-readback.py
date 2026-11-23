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
from itertools import chain
archive,typed,new,report,fixture,census=map(Path,sys.argv[1:])
R=json.loads(report.read_bytes()); F=json.loads(fixture.read_bytes()); C=json.loads(census.read_bytes())
assert R['status']=='PASS' and R['retained']==6436 and R['new_retirements']==14 and R['previous_stage_F64_noncomparable']==1
priors={r['label']:r for r in F['cases'] if 'operands' in r['recipe']}
bound={(s['stream'],a['frame']):a for s in C['matching_streams'] for a in s['matches']}
assert len(bound)==14 and len(R['owner_identity_witnesses'])==14
def pairs(p):
 return list(zip(frames(p/'requests.frames.bin',b'HTOWNR01'),frames(p/'responses.frames.bin',b'HTOWNS01'),strict=True))
for r in R['rows']:
 stream=r['stream']; old=(archive/stream.removeprefix('retained/')) if stream.startswith('retained/') else typed/stream.removeprefix('four/')
 dst=new/stream.replace('/','-')
 a=pairs(old);b=pairs(dst)
 assert len(a)==r['prior_frames'] and len(b)==r['retained']
 exclude=[i for ss,i in bound if ss==stream]+([685] if stream=='retained/witness-old' else [])
 assert r['excluded']==exclude
 assert [it for i,it in enumerate(a,1) if i not in exclude]==b
 for i in exclude:
  if (stream,i) not in bound:continue
  key=bound[stream,i]; v=priors[key['label']]; rec=v['recipe']
  q0,z0=a[i-1];cmd,=req(q0)
  assert sha(q0)==rec['request_sha256'] and resp(z0)==(7,[])
  bb=zlib.decompress((dst/f'orig-pre-{i}.zlib').read_bytes());assert sha(bb)==rec['identical_owner_sha256']
  owner=Owner(bb)
  for t,tag,s,raw in [(cmd[1],v['tag'],v['value_shape'],rec['operands'][0]),(cmd[2],1,v['mask_shape'],rec['operands'][1])]:
   d=owner.node(t); assert d[72]==tag and list(d[12:12+len(s)])==s
   assert owner.storage(d[27])[2]==bytes.fromhex(raw)
 assert len([z for z in R['positive_context_aliases'] if z['stream']==stream])==(3 if stream=='four/new' else 0)
print(json.dumps(dict(status='PASS',retained=6436,owner_bound=14,positive_same_request_preserved=3),sort_keys=True))
