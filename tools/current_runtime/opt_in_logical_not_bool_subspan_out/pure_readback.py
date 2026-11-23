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
 request_bytes=list(frames(p/'requests.frames.bin',b'HTOWNR01'));qs=list(map(req,request_bytes));rs=list(map(resp,frames(p/'responses.frames.bin',b'HTOWNS01')))
 assert len(qs)==len(rs)==t['frames']==r['native_frames']
 for qq,(st,rr) in zip(qs,rs):
  if not st:assert [a[0] for a in qq]==[a[0] for a in rr]
 def raw(s):
  b=zlib.decompress((p/'owners'/(s['sha256']+'.zlib')).read_bytes());assert sha(b)==s['sha256'] and len(b)==s['length']
  return b
 counts=dict(successes=0,exact_rollbacks=0,unsupported_constructor_controls=0,frames=len(qs))
 def direct_config(index,root,handle,fixture):
  c,=qs[index];assert c[:2]==(111,root) and c[3:]==(80,0,0,0,0)
  rank=len(fixture['shape']); pad=[0]*(4-rank)
  assert q(request_bytes[index],10,c[2])==(rank,*fixture['shape'],*pad,*fixture['strides'],*pad,fixture.get('offset',0))
  assert rs[index]==(0,[(111,handle,b'')])
 def validate_span(owner,root_handle,view_handle,f):
  r=owner.node(root_handle);v=owner.node(view_handle)
  _,kr=owner.row(1,root_handle);_,kv=owner.row(1,view_handle)
  elems=f['offset'];dtype=f['dtype'];width={0:4,1:1}[dtype];span=width*len(f['input'])
  assert r[72]==v[72]==dtype and r[83]==0 and v[83]==1 and v[80:82]==(r[40],r[40])
  assert kr[2:4]==kv[2:4]==r[27:29]==v[27:29]
  assert kv[4]==v[29]==v[2]-r[2]==width*elems and kr[4]==r[29]==0
  assert kv[5]>=span==width*(1+sum((a-1)*b for a,b in zip(f['shape'],f['strides'])))
  _,live,old=owner.storage(r[27])
  assert live==kr[5]==width*r[1]==width*len(f['root_values']) and 0<width*elems<=live-span and kv[5]==live-width*elems
  store=owner.row(2,r[27])[1]
  assert r[2]==(1<<63)+store[2] and v[2]<=(1<<64)-span
  assert r[4]==v[4]==r[5]==v[5]==0
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
   assert any(c[0]==128 for c in qs[idx])
   if x['case'].startswith(('offset-same-storage-','bool-offset-partially-overlapping-')):
    c,=qs[idx];assert c[0]==128 and c[1]!=c[2]
    a=Owner(raw(x['pre']));src=a.node(c[1]);dst=a.node(c[2])
    assert src[27:29]==dst[27:29] and src[83]==dst[83]==1 and src[72]==dst[72] and src[29]>0 and dst[29]>0
    difference=src[29]-dst[29]
    if x['case'].startswith('bool-offset-partially-overlapping-'):
     assert (difference,src[72])==(1,1) and src[1]==dst[1]>difference
     counts['bool_partial_offset_alias_rollbacks']=counts.get('bool_partial_offset_alias_rollbacks',0)+1
    else:
     assert difference==0
     if src[72]==1:counts['bool_same_start_offset_alias_rollbacks']=counts.get('bool_same_start_offset_alias_rollbacks',0)+1
   if r['mode']=='baseline':
    c,=qs[idx];assert c[0]==128
    source,dst=c[1:3]
    a=Owner(raw(x['pre']));vn=a.node(dst);sn=a.node(source)
    assert vn[72]==sn[72]==1 and vn[83]==1 and sn[83]==0 and vn[27]!=sn[27] and vn[28]!=sn[28]
    f=official['rows'][int(x['case'].rsplit('-',1)[1])]
    i,=[k for k,(status,rr) in enumerate(rs[:idx]) if not status and rr==[(111,dst,b'')]]
    root=qs[i][0][1];rn=a.node(root)
    assert rn[72]==1 and rn[83]==0 and rn[27:29]==vn[27:29] and rn[40]==vn[81]
    direct_config(i,root,dst,f);validate_span(a,root,dst,f)
    if f.get("prior_exact_boundary"):counts["prior_end_boundary_refusal"]=True
   counts['exact_rollbacks']+=1;continue
  assert x['kind']=='positive'
  f=x['fixture'];assert f==official['rows'][x['case']];n=len(f['input']);h=x['dst'];offset=f.get('offset',0);rn=__import__('math').prod(f['base_shape'])
  assert qs[idx]==[(128,x['src'],h,0,0,0,0,0),(98,0xfe000001,0,n,0,0,0,0),(98,x['root'],0,rn,0,0,0,0),(98,x['sib'],0,n,0,0,0,0)]
  dtype=f['dtype'];assert dtype in (0,1,2,4)
  width={0:4,1:1,2:8,4:8}[dtype]
  if dtype in (1,2,4) or offset:
   for frame,handle in zip(x['construct'],(h,x['sib'])):
    direct_config(frame-1,x['root'],handle,f)
  logical=b''.join(((0x3ff0000000000000 if v&((1<<63)-1)==0 else 0).to_bytes(8,'little') if dtype==4 else int(v==0).to_bytes(width,'little') if dtype in (1,2) else struct.pack('<I',0x3f800000 if v==0 else 0)) for v in f['input'])
  assert logical.hex()==f['logical']
  physical=bytearray.fromhex(f["base_initial"]) if offset else bytearray(width*n);seen=set()
  for i,ix in enumerate(itertools.product(*map(range,f['shape']))):
   k=sum(a*b for a,b in zip(ix,f['strides']));assert 0<=k<n and k not in seen;seen.add(k);physical[width*(k+offset):width*(k+offset)+width]=logical[width*i:width*i+width]
  assert physical.hex()==f['base_physical']
  assert rs[idx]==(0,[(128,h,b''),(98,n,logical),(98,rn,physical),(98,n,logical)])
  a,b=map(Owner,(raw(x['before']),raw(x['after'])))
  root,view,sib,src=(a.node(x[key]) for key in ('root','dst','sib','src'))
  assert all(a.node(x[key])==b.node(x[key]) for key in ('root','dst','sib','src'))
  assert root[27:29]==view[27:29]==sib[27:29] and root[27]!=src[27] and root[28]!=src[28]
  assert root[72]==view[72]==sib[72]==src[72]==dtype
  assert root[83]==0 and view[83]==sib[83]==1 and view[80:82]==sib[80:82] and view[81]==root[40]
  assert view[29]==offset*width and view[12:12+view[0]]==tuple(f['shape']) and view[16:16+view[0]]==tuple(f['strides'])
  start,size,old=a.storage(root[27]);nt,nb,new=b.storage(root[27])
  assert (start,size)==(nt,nb) and new[:width*rn]==physical and new[width*rn:]==old[width*rn:]
  if offset:
   validate_span(a,x['root'],h,f);validate_span(a,x['root'],x['sib'],f)
   assert old.hex()==f['base_initial'] and old[:width*offset]==new[:width*offset] and old[width*(offset+n):]==new[width*(offset+n):]
   key={0:'f32',1:'bool'}[dtype]+('_end_boundary_successes' if f.get('prior_exact_boundary') else '_offset_mask_successes')
   counts[key]=counts.get(key,0)+1
  else:
   assert len(old)>=width*n and old[:width*n]==(({0:struct.pack('<f',-2),1:b'\x01',2:struct.pack('<q',-2),4:struct.pack('<d',-2)}[dtype])*n)
  ss,sz,sd=a.storage(src[27]);assert sd[:width*n].hex()==f['initial']; assert a.storage(src[27])==b.storage(src[27]) and (ss+sz<=start or start+size<=ss)
  va,vr=a.row(3,root[28]);vb,vs=b.row(3,root[28]);assert va==vb and vs==vr[:2]+(vr[2]+1,)+vr[3:]
  assert a.row(3,src[28])==b.row(3,src[28])
  for kind in range(1,9):
   first=B+128+(kind-1)*a.slots*64;end=first+a.slots*64
   if kind==3:assert a.b[first:va+16]==b.b[first:va+16] and a.b[va+24:end]==b.b[va+24:end]
   else:assert a.b[first:end]==b.b[first:end]
  counts['successes']+=1
  if offset:continue
  key={0:'f32_successes',1:'bool_successes',2:'i64_successes',4:'f64_successes'}[dtype];counts[key]=counts.get(key,0)+1
 return counts
if __name__=='__main__':print(json.dumps({str(Path(p).name):inspect(p) for p in sys.argv[1:]},sort_keys=True))
