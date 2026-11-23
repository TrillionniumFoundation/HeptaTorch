"""Offline wire/owner readback; no runtime, protocol, producer or Torch imports."""
from pathlib import Path
import itertools,hashlib,json,struct,sys,zlib
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
def inspect(path):
 p=Path(path);r=json.loads((p/'report.json').read_bytes());assert not r.get('error') and r['no_torch']
 fixture_raw=(p.parent/'official.fixture.json').read_bytes()
 assert sha(fixture_raw)==r['fixture_sha256']
 official=json.loads(fixture_raw)
 t=json.loads((p/'transport.json').read_bytes())
 assert t['exit_code']==0 and t['responses_before_stdin_eof'] and t['executable_sha256']==r['elf']
 qs=list(map(req,frames(p/'requests.frames.bin',b'HTOWNR01')));rs=list(map(resp,frames(p/'responses.frames.bin',b'HTOWNS01')))
 assert len(qs)==len(rs)==t['frames']==r['native_frames']
 for qq,(st,rr) in zip(qs,rs):
  if not st:assert [a[0] for a in qq]==[a[0] for a in rr]
 def raw(s):
  b=zlib.decompress((p/'owners'/(s['sha256']+'.zlib')).read_bytes());assert sha(b)==s['sha256'] and len(b)==s['length']
  return b
 counts=dict(successes=0,exact_rollbacks=0,unsupported_constructor_controls=0,frames=len(qs))
 for x in r['rows']:
  if x['kind']=='setup_refusal':
   assert rs[x['frame']-1]==(x['status'],[]) and x['status']!=0
   counts['unsupported_constructor_controls']+=1;continue
  if x['kind']=='root_toggle_state':
   transactional,= [v for v in r['rows'] if v.get('case')=='stale-attempt-same-frame-root-toggle']
   rejected,= [v for v in r['rows'] if v.get('case')=='root-grad-after-creation']
   j=transactional['frame']-1
   trial,operation=qs[j]
   assert trial[0]==90 and trial[2:]==(1,0,0,0,0,0) and operation[0]==128
   root=trial[1];view=operation[2]
   assert rs[j]==(7,[]) and raw(transactional['pre'])==raw(transactional['post'])
   assert qs[j+1]==[trial] and rs[j+1]==(0,[(90,root,b'')]),(qs[j+1],rs[j+1])
   view_meta=rs[j+2][1][0];root_meta=rs[j+4][1][0]
   assert (view_meta[0:2],root_meta[0:2])==((13,view),(13,root))
   assert q(view_meta[2],16)[2]==x['view_requires_grad']==1
   assert q(root_meta[2],16)[2]==x['root_requires_grad']==1
   assert rejected['frame']==j+7 and qs[j+6]==[operation] and rs[j+6]==(7,[])
   assert raw(rejected['pre'])==raw(rejected['post'])
   old=Owner(raw(transactional['pre']));grad=Owner(raw(rejected['pre']))
   for handle in (root,view):
    a,b=old.node(handle),grad.node(handle)
    assert (a[4],b[4])==(0,1) and a[5]==b[5]==0
    assert a[27:29]==b[27:29] and a[40]==b[40] and a[80:84]==b[80:84]
   counts['successful_synchronized_root_toggle_verified']=True
   continue
  idx=x['frame']-1
  if x['kind']=='refusal':
   assert rs[idx]==(x['expected_status'],[]) and raw(x['pre'])==raw(x['post']) and x['pre']['ptr']==x['post']['ptr']
   assert any(c[0]==128 for c in qs[idx]);counts['exact_rollbacks']+=1;continue
  assert x['kind']=='positive'
  f=x['fixture'];assert f==official['rows'][x['case']];n=len(f['input']);h=x['dst']
  assert qs[idx]==[(128,x['src'],h,0,0,0,0,0),(98,0xfe000001,0,n,0,0,0,0),(98,x['root'],0,n,0,0,0,0),(98,x['sib'],0,n,0,0,0,0)]
  logical=b''.join(struct.pack('<I',0x3f800000 if v==0 else 0) for v in f['input'])
  assert logical.hex()==f['logical']
  physical=bytearray(4*n);seen=set()
  for i,ix in enumerate(itertools.product(*map(range,f['shape']))):
   k=sum(a*b for a,b in zip(ix,f['strides']));assert 0<=k<n and k not in seen;seen.add(k);physical[4*k:4*k+4]=logical[4*i:4*i+4]
  assert physical.hex()==f['base_physical']
  assert rs[idx]==(0,[(128,h,b''),(98,n,logical),(98,n,physical),(98,n,logical)])
  a,b=map(Owner,(raw(x['before']),raw(x['after'])))
  root,view,sib,src=(a.node(x[key]) for key in ('root','dst','sib','src'))
  assert all(a.node(x[key])==b.node(x[key]) for key in ('root','dst','sib','src'))
  assert root[27:29]==view[27:29]==sib[27:29] and root[27]!=src[27] and root[28]!=src[28]
  assert root[83]==0 and view[83]==sib[83]==1 and view[80:82]==sib[80:82] and view[81]==root[40]
  assert view[29]==0 and view[12:12+view[0]]==tuple(f['shape']) and view[16:16+view[0]]==tuple(f['strides'])
  start,size,old=a.storage(root[27]);nt,nb,new=b.storage(root[27])
  assert (start,size)==(nt,nb) and new[:4*n]==physical and new[4*n:]==old[4*n:]
  ss,sz,sd=a.storage(src[27]); assert a.storage(src[27])==b.storage(src[27]) and (ss+sz<=start or start+size<=ss)
  va,vr=a.row(3,root[28]);vb,vs=b.row(3,root[28]);assert va==vb and vs==vr[:2]+(vr[2]+1,)+vr[3:]
  assert a.row(3,src[28])==b.row(3,src[28])
  for kind in range(1,9):
   first=B+128+(kind-1)*a.slots*64;end=first+a.slots*64
   if kind==3:assert a.b[first:va+16]==b.b[first:va+16] and a.b[va+24:end]==b.b[va+24:end]
   else:assert a.b[first:end]==b.b[first:end]
  counts['successes']+=1
 return counts
if __name__=='__main__':print(json.dumps({str(Path(p).name):inspect(p) for p in sys.argv[1:]},sort_keys=True))
