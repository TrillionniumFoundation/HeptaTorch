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
pdir,scope=map(Path,sys.argv[1:])
src=scope.read_bytes();assert sha(src)=='899f9aaecbd6621625bf2b702ab445922cb4d3d10ba03e4c5fc0a8b8eaa05065'
p=json.loads(src)
native=json.loads((pdir/'report.json').read_bytes())
assert native['elf']==p['native_elf_sha256'] and native['external_torch_backend'] is False
summary=[]
for x,g in zip(p['admitted'],native['cases'],strict=True):
 c=x['original_frozen'];label=c['label'];assert label==g['case']
 d=pdir/label;Q=list(frames(d/'requests.frames.bin',b'HTOWNR01'));R=list(frames(d/'responses.frames.bin',b'HTOWNS01'))
 assert len(Q)==len(R)==g['frames']==128 and all(resp(z)[0]==0 for z in R)
 ops=[req(w)[0][0] for w in Q];assert ops.count(97)==5 and [a for a in ops if a in (2,3,32)]==[32,2,3,32,2]
 assert ops.count(41)==10
 def packet(i,qu,op,value=None):
  r=resp(R[i-1])[1];got=req(Q[i-1]);assert len(got)==len(r)==1
  assert got[0][:len(qu)]==tuple(qu) and r[0][0]==op
  if value is not None:assert r[0][1]==value
  return r[0]
 ledger=json.loads((d/'ledger.json').read_bytes());assert len(ledger)==16
 exp={};live={};owner_states=0;words=0;checks=0
 for i,(name,fixture) in enumerate(c['inputs'].items(),1):
  e=fixture;raw=bytes.fromhex(e['raw_f32le_hex']); sh=e['shape']
  p0=packet(i,(97,0,96),97)
  assert q(Q[i-1],8,96)==(len(sh),0,*sh,*([0]*(4-len(sh))),160,len(raw))
  assert Q[i-1][160:]==raw
  live[name]=p0[1];exp[name]=e
 def readback(row,name,h):
  e=exp[name];s=e['shape'];st=e['strides'];n=e['storage_bytes']//4;raw=bytes.fromhex(e['raw_f32le_hex'])
  assert row['handle']==h
  md=q(packet(row['meta'],(13,h),13)[2],16)
  assert md[:3]==(len(s),n,0) and md[3:3+len(s)]==tuple(s) and md[7:7+len(s)]==tuple(st)
  assert md[11:]==(row['storage'],row['version'],0,0,len(raw))
  assert packet(row['raw'],(98,h,0,n),98,n)[2]==raw
  return row['storage'],row['version'],raw
 def prove_owner(li):
  nonlocal_unused=None
  osha=li['owner'];raw=zlib.decompress((d/'owners'/(osha+'.zlib')).read_bytes());assert sha(raw)==osha
  obs=li['observations'];assert obs.keys()==live.keys()
  p0=Owner(raw);auth={}
  for key,h in live.items():
   ss,vs,bs=readback(obs[key],key,h);s=exp[key]['shape'];st=exp[key]['strides'];nd=p0.node(h)
   assert nd[:2]==(len(s),len(bs)//4) and nd[12:12+len(s)]==tuple(s) and nd[16:16+len(s)]==tuple(st)
   assert nd[4:6]==(0,0) and nd[27:30]==(ss,vs,0) and nd[72]==nd[83]==0
   assert p0.row(1,h)[1][2:6]==(ss,vs,0,len(bs))
   assert p0.row(3,vs)[1][2]==0 and p0.storage(ss)[1:]==(len(bs),bs)
   auth[key]=(h,ss,vs)
  assert all(len({u[k] for u in auth.values()})==len(auth) for k in range(3))
  return auth
 assert ledger[0]['kind']=='input'; prev=prove_owner(ledger[0]);owner_states+=1;checks+=5
 for spec,l in zip(c['stages'],ledger[1:6],strict=True):
  key=spec['name'];assert l['kind']=='stage' and l['name']==key
  cmd=[live[v] if isinstance(v,str) else v for v in spec['command']]
  assert cmd==l['command'];r=packet(l['frame'],cmd,cmd[0])
  live[key]=r[1];exp[key]=spec['expected'];new=prove_owner(l);assert all(new[k]==v for k,v in prev.items());prev=new
  owner_states+=1;checks+=len(new);words+=exp[key]['storage_bytes']//4
 for l in ledger[6:]:
  key=l['name'];assert l['kind']=='drop' and key in live
  packet(l['frame'],(41,live.pop(key)),41)
  if live:
   k=l['survivor'];sa,va,_=readback(l['survival'],k,live[k]);assert (live[k],sa,va)==prev[k]
  else:assert l['survival'] is None and l['survivor'] is None
 assert not live
 assert [l['name'] for l in ledger[6:]]==(list(reversed(prev)) if g['release_order']=='result-first' else list(prev))
 summary.append(dict(case=label,frames=len(Q),stage_words=words,live_observations=checks,owner_states=owner_states,
                     q_sha256=sha((d/'requests.frames.bin').read_bytes()),r_sha256=sha((d/'responses.frames.bin').read_bytes())))
assert sum(g['frames'] for g in summary)==384
assert sum(g['stage_words'] for g in summary)==80 and sum(g['live_observations'] for g in summary)==135
print(json.dumps(dict(status='PASS_INDEPENDENT_PURE_REPLAY_OF_WIRE_AND_PHYSICAL',cases=summary),sort_keys=True))
