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
f,occ,stage,f32,i64,four=map(Path,sys.argv[1:])
assert sha(f.read_bytes())=='d7b8af35d5288fd870c998e0564c3da7c6efcaf88c5723e438cd6e4999493cd1'
assert sha(occ.read_bytes())=='301740430064b38939874ae3b59163fe22377c47eaf172e5e6d5891fb1072f3b'
D=json.loads(f.read_bytes());O=json.loads(occ.read_bytes())
identity={x['label']:x for x in D['cases'] if 'operands' in x['recipe']}
def pair(p):
 return list(zip(frames(p/'requests.frames.bin',b'HTOWNR01'),frames(p/'responses.frames.bin',b'HTOWNS01'),strict=True))
roots={'f32':f32,'i64':i64,'four':four}
source={k:pair(v) for k,v in roots.items()}
out=[]
for s in O['matching_streams']:
 a=s['stream'];root=stage/a.removeprefix('retained/') if a.startswith('retained/') else four
 assert sha((root/'requests.frames.bin').read_bytes())==s['full_q_sha'] and sha((root/'responses.frames.bin').read_bytes())==s['full_r_sha']
 stream=pair(root)
 for m in s['matches']:
  g=identity[m['label']];pin=g['recipe'];n=pin['frame'];sr=source[pin['corpus']]
  assert n==m['frame'] and stream[:n]==sr[:n]
  (q0,r0)=stream[n-1];assert sha(q0)==pin['request_sha256'] and resp(r0)==(7,[])
  orig=roots[pin['corpus']]
  rows=json.loads((orig/'report.json').read_bytes())['rows']; row,=[r for r in rows if r['frame']==n and r.get('label')==pin['old_label']]
  assert row['pre']==row['post'] and row['pre']['sha256']==pin['identical_owner_sha256']
  zz=zlib.decompress((orig/'owners'/(pin['identical_owner_sha256']+'.zlib')).read_bytes());assert sha(zz)==pin['identical_owner_sha256']
  owner=Owner(zz); cmd,=req(q0); assert cmd[0]==130 and cmd[3:]==(0,)*5
  for ht,t,shape,word in [(cmd[1],g['tag'],g['value_shape'],pin['operands'][0]),(cmd[2],1,g['mask_shape'],pin['operands'][1])]:
   d=owner.node(ht); st,vr=d[27:29]
   assert d[72]==t and d[0]==len(shape) and list(d[12:12+len(shape)])==shape and d[29]==d[83]==0 and d[4:6]==(0,0)
   assert owner.storage(st)[2]==bytes.fromhex(word) and owner.row(3,vr)[1][2]==0
  record=dict(stream=a,frame=n,exact_prefix=True,prior_original_live_owner=pin['identical_owner_sha256'],owner_raw_direct_in_original=(root==orig),
    input_tags=[g['tag'],1],value_shape=g['value_shape'],mask_shape=g['mask_shape'])
  out.append(record)
src=source['four']
for a in O['do_not_retire_positive_context_aliases']:
 n=a['frame'];u,v=src[n-1];assert sha(u)==a['request_sha256'] and resp(v)[0]==0
assert len(out)==14 and len(identity)==8
print(json.dumps(dict(status='PASS',physical_occurrences=len(out),original_live_owner_pins=len(identity),
 note='Original owner snapshots observed directly; duplicate context aliases proven by full exact q/r prefix, not by sampling a fresh native owner.',
 positive_same_request_bytes_keep=O['do_not_retire_positive_context_aliases'],records=out),sort_keys=True,indent=2))
