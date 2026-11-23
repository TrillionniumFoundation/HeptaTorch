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
FACTORY=97
VIEW=111
DROP=41
def inspect(path,fixture):
 p=Path(path); r=json.loads((p/'report.json').read_bytes()); assert r['no_torch'] and not r.get('error')
 raw_fixture=fixture.read_bytes();assert sha(raw_fixture)==r['fixture_sha256']
 rows={x['label']:x for x in json.loads(raw_fixture)['rows']}
 transport=json.loads((p/'transport.json').read_bytes())
 assert transport['exit_code']==0 and transport['responses_before_stdin_eof'] and transport['executable_sha256']==r['elf']
 raw_q=list(frames(p/'requests.frames.bin',b'HTOWNR01'))
 qs=list(map(req,raw_q)); rs=list(map(resp,frames(p/'responses.frames.bin',b'HTOWNS01')))
 assert len(qs)==len(rs)==transport['frames']==r['native_frames']
 for qq,(st,rr) in zip(qs,rs):
  if not st:assert [a[0] for a in qq]==[a[0] for a in rr]
 def snapshot(record):
  b=zlib.decompress((p/'owners'/(record['sha256']+'.zlib')).read_bytes())
  assert sha(b)==record['sha256'] and len(b)==record['length'];return b
 def direct(f, root, shape, strides):
  cmd,=qs[f-1];assert cmd[:2]==(VIEW,root) and cmd[3:]==(80,0,0,0,0)
  n=len(shape);pad=[0]*(4-n)
  assert q(raw_q[f-1],10,cmd[2])==(n,*shape,*pad,*strides,*pad,0)
 def read(f,h,n,expected):
  assert qs[f-1]==[(13,h,0,0,0,0,0,0),(98,h,0,n,0,0,0,0)]
  st,ret=rs[f-1]
  assert st==0 and ret[0][:2]==(13,h) and ret[1]==(98,n,expected)
  return q(ret[0][2],16)
 def owner_fields(o,root,handles,n,root_expected):
  d=o.node(root); k=o.row(1,root)[1]
  assert d[72]==6 and d[83]==0 and d[4:6]==(0,0) and d[29]==0 and d[1]==n
  stor=d[27];vers=d[28];sf=o.row(2,stor)[1]
  assert k[2:4]==(stor,vers) and k[4:6]==(0,2*n)
  assert d[2]==(1<<63)+sf[2] and o.storage(stor)[1:]==(2*n,root_expected)
  for v in handles:
   a=o.node(v);row=o.row(1,v)[1]
   assert a[72]==6 and a[83]==1 and a[4:6]==(0,0) and a[1]==n
   assert a[2]==d[2] and a[29]==0
   assert a[80:82]==(k[6],d[40]) and a[27:29]==row[2:4]==(stor,vers)
   assert row[4:6]==(0,2*n)
  return stor,vers,k[6],d[40]
 def private_node(o,i):
  page,=q(o.b,1,4096+(i>>8)*8);assert page>>63
  return q(o.b,84,o.start+(page&MASK)+(i&255)*672)
 totals=dict(cases=0,new_constructor_successes=0,exact_rollbacks=0,frames=len(qs),old_refusals=0,still_unsupported_out=0,
             root_first=0,view_first=0,grad_toggle_setup_refusals=0)
 for x in r['rows']:
  if x['kind']=='refusal':
   i=x['frame']-1
   assert rs[i]==(x['expected'],[]) and x['status']==x['expected']
   before=snapshot(x['pre']);after=snapshot(x['post'])
   assert before==after and x['pre']['ptr']==x['post']['ptr']
   totals['exact_rollbacks']+=1
   if x['case'].startswith('original-real-'):
    totals['old_refusals']+=1
    row=rows[x['case'].removeprefix('original-real-')]
    direct(i+1,qs[i][0][1],row['shape'],row['strides'])
    assert Owner(before).node(qs[i][0][1])[72]==6
   if x['case'].startswith('new-bf16-view-is-still-not-out-covered-'):
    totals['still_unsupported_out']+=1
    a,=qs[i];assert a[0]==128
    oo=Owner(before)
    assert oo.node(a[1])[72]==oo.node(a[2])[72]==6 and oo.node(a[2])[83]==1
   if x['case'].startswith('valid-view-followed-by-stale-view-'):
    assert [a[0] for a in qs[i]]==[111,111]
   continue
  if x['kind']=='setup_refusal':
   i=x['frame']-1
   assert x['case']=='bf16-root-grad-toggle' and qs[i][0][0]==90 and rs[i]==(x['status'],[]) and x['status']==5
   totals['grad_toggle_setup_refusals']+=1
   continue
  assert x['kind']=='positive'
  f=rows[x['fixture']];n=len(f['root_u16']); root,v,sib=(x[k] for k in ('root','view','sibling'))
  assert len({root,v,sib})==3
  mer=list(itertools.product(*(range(a) for a in f['shape'])))
  perm=[sum(a*b for a,b in zip(ix,f['strides'])) for ix in mer]
  assert sorted(perm)==list(range(n))
  expected_root=struct.pack('<'+'H'*n,*f['root_u16'])
  expected_view=struct.pack('<'+'H'*n,*(f['root_u16'][i] for i in perm))
  assert f['raw_root']==expected_root.hex() and f['raw_view']==expected_view.hex() and f['version_delta']==0
  assert rs[x['create']-1]==(0,[(FACTORY,root,b'')]) and qs[x['create']-1][0][:2]==(FACTORY,6)
  for idx,h in [(x['constructor'],v),(x['sibling_constructor'],sib)]:
   direct(idx,root,f['shape'],f['strides']);assert rs[idx-1]==(0,[(VIEW,h,b'')])
  a,b=map(Owner,(snapshot(x['before']),snapshot(x['after'])))
  stor,vers,r_idx,generation=owner_fields(a,root,(),n,expected_root)
  assert owner_fields(b,root,(v,sib),n,expected_root)==(stor,vers,r_idx,generation)
  assert a.row(3,vers)[1][2]==b.row(3,vers)[1][2]
  for key,h,by,sh,st in [('root_read',root,expected_root,f['root_shape'],None),('view_read',v,expected_view,f['shape'],f['strides']),('sib_read',sib,expected_view,f['shape'],f['strides']),('root_after_read',root,expected_root,f['root_shape'],None)]:
   meta=read(x[key],h,n,by);assert meta[:3]==(len(sh),n,0) and meta[3:3+len(sh)]==tuple(sh)
   assert meta[11:16]==(stor,vers,a.row(3,vers)[1][2],0,2*n)
   if st is not None:assert meta[7:7+len(sh)]==tuple(st)
  assert qs[x['weak_frame']-1][0][:2]==(42,root) and rs[x['weak_frame']-1]==(0,[(42,x['weak'],b'')])
  first,second,last=x['drop_frames'];first_id=qs[first-1][0][1]
  assert {first_id,qs[second-1][0][1]}=={root,v} and qs[last-1][0][1]==sib
  for fr in (first,second,last):assert qs[fr-1][0][0]==DROP and rs[fr-1][0]==0 and rs[fr-1][1][0][0]==DROP
  survived=x['survivor']; assert survived in (v,root) and survived!=first_id
  totals['root_first' if first_id==root else 'view_first']+=1
  read(x['mid_reads'][0],survived,n,expected_root if survived==root else expected_view)
  for fr in x['mid_reads'][1:]:read(fr,sib,n,expected_view)
  for snap in [x['middle'],x['after_two']]:
   o=Owner(snapshot(snap)); pv=private_node(o,r_idx)
   assert pv[40]==generation and pv[27:29]==(stor,vers) and pv[72]==6
   assert o.node(sib)[80:82]==(r_idx,generation) and o.storage(stor)[2]==expected_root
   assert o.row(3,vers)[1][2]==a.row(3,vers)[1][2]
  assert qs[x['hold_frame']-1][0][:2]==(43,x['weak']) and rs[x['hold_frame']-1]==(0,[(43,x['hold'],b'')])
  assert qs[x['expired']-1][0][:2]==(43,x['weak']) and rs[x['expired']-1]==(6,[])
  totals['cases']+=1;totals['new_constructor_successes']+=2
 return totals
if __name__=='__main__':
 print(json.dumps({Path(s).name:inspect(s,Path(sys.argv[1])) for s in sys.argv[2:]},sort_keys=True))
