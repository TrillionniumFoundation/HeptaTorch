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
h,pp,elf,r,typed,archive,fixture,census,receipt,pin=map(Path,sys.argv[1:])
assert sha(elf.read_bytes())==pin.read_text().strip()
assert sha((h/'test_runtime_candidate.py').read_bytes())=='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'
sys.path.insert(0,str(h));from test_runtime_candidate import load_wire
W=load_wire(pp);assert 'torch' not in sys.modules
assert sha(fixture.read_bytes())=='d7b8af35d5288fd870c998e0564c3da7c6efcaf88c5723e438cd6e4999493cd1'
assert sha(census.read_bytes())=='301740430064b38939874ae3b59163fe22377c47eaf172e5e6d5891fb1072f3b'
assert sha(receipt.read_bytes())=='890cc3282d6d2e15f833eb81514cd9f0819c211105bbbf164266439afa1a2641'
p=json.loads(fixture.read_bytes());c=json.loads(census.read_bytes());T=json.loads(receipt.read_bytes())
H={Z['label']:Z for Z in p['cases'] if 'operands' in Z['recipe']}
wd={row['stream']:row for row in c['matching_streams']}
specs=[]
# The previous release's complete comparable streams with full q/r, as actually produced by its own ELF.
for x in T['rows']:
 name=x['group'].replace('/','-')
 specs.append(('retained/'+name,archive/name,x['retained']))
specs.extend([('retained/witness-old',archive/'witness-old',685),('retained/witness-new',archive/'witness-new',685),
 ('four/new',typed/'new',1961),('four/supplement',typed/'supplement',13),('four/supplement-tail',typed/'supplement-tail',17)])
assert len(specs)==16 and sum(n for _,_,n in specs)==6451
def stream(p,n):
 a=p/'requests.frames.bin';b=p/'responses.frames.bin'
 z=list(zip(frames(a,b'HTOWNR01'),frames(b,b'HTOWNS01'),strict=True));assert len(z)==n
 return z,sha(a.read_bytes()),sha(b.read_bytes())
r.mkdir(); ledger=[]; watched=[];aliases=[]
for name,path,n in specs:
 obs,qsha,rsha=stream(path,n)
 planned={z['frame']:z for z in wd[name]['matches']} if name in wd else {}
 if planned:assert (qsha,rsha)==(wd[name]['full_q_sha'],wd[name]['full_r_sha'])
 pd=r/name.replace('/','-')
 s=W.Session(elf,directory=pd)
 skipped=[]
 try:
  for i,(q0,z0) in enumerate(obs,1):
   if i in planned:
    ancient=planned[i];g=H[ancient['label']]; rec=g['recipe']
    assert ancient['request_sha256']==sha(q0)==rec['request_sha256'] and sha(z0)==rec['response_sha256'] and resp(z0)==(7,[])
    saved=s.owner(); content=saved['bytes']; assert sha(content)==rec['identical_owner_sha256'],(name,i,sha(content),rec['identical_owner_sha256'])
    qu,=req(q0); owner=Owner(content)
    for pointer,tag,dims,raw in [(qu[1],g['tag'],g['value_shape'],rec['operands'][0]),(qu[2],1,g['mask_shape'],rec['operands'][1])]:
     d0=owner.node(pointer);sid,vn=d0[27:29]
     assert d0[72]==tag and d0[0]==len(dims) and list(d0[12:12+len(dims)])==dims
     assert d0[29]==d0[83]==0 and owner.storage(sid)[2]==bytes.fromhex(raw)
     assert owner.row(3,vn)[1][2]==0
    (pd/(f'orig-pre-{i}.zlib')).write_bytes(zlib.compress(content,6))
    watched.append(dict(stream=name,frame=i,label=ancient['label'],owner_sha256=sha(content),request_sha256=sha(q0),old_response_sha256=sha(z0)))
    skipped.append(i);continue
   if name=='retained/witness-old' and i==685:
    # Inherited F64 stage witness. That old I64 image refused this exact command;
    # the immediately previous release and current already admit it. No new retirement.
    assert sha(q0)=='a918b15f1b156204b31f42d8b1548b34e2c81019121f92b73a28a784ea8774cb' and resp(z0)==(7,[])
    skipped.append(i);continue
   got=s.exchange(q0)
   assert got==z0,(name,i,sha(got),sha(z0),resp(got)[0],resp(z0)[0])
   if name=='four/new' and i in (679,728,784):
    assert resp(got)[0]==0
    aliases.append(dict(stream=name,frame=i,request=sha(q0),response=sha(got),kept=True))
 finally:s.finish()
 assert skipped==list(planned)+([685] if name=='retained/witness-old' else [])
 mirror,cq,cr=stream(pd,n-len(skipped));assert mirror==[raw for i,raw in enumerate(obs,1) if i not in skipped]
 ledger.append(dict(stream=name,prior_frames=n,retained=len(mirror),excluded=skipped,original_q_sha256=qsha,original_r_sha256=rsha,replayed_q_sha256=cq,replayed_r_sha256=cr))
assert len(watched)==14 and len({w['label'] for w in watched})==8 and len(aliases)==3
result=dict(status='PASS',retained=sum(s['retained'] for s in ledger),new_retirements=14,previous_stage_F64_noncomparable=1,
 rows=ledger,owner_identity_witnesses=watched,positive_context_aliases=aliases)
assert result['retained']==6436
print(json.dumps(result,sort_keys=True,indent=2))
