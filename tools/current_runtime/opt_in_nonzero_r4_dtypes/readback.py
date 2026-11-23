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
stage,fixture,measure=map(Path,sys.argv[1:])
assert sha(fixture.read_bytes())=='df0970b57a6a5f69fbd5faf047d1d58c0302afcd20cbaa499759e5c0c68d57f8'
assert sha((measure/'receipt.json').read_bytes())=='70f21ba1eec4e505130c995675b298a3fdbee42c571402a0d377867a578130d1'
cases={c['label']:c for c in json.loads(fixture.read_bytes())['cases']}
L={c['label']:c for c in json.loads((measure/'receipt.json').read_bytes())['cases']}
logical=(measure/'logical-i64le.bin').read_bytes();physical=(measure/'physical-i64le.bin').read_bytes()
assert sha(logical)=='c1d40b2698ff5a0565dc0c03aa467bcac876149d9c03b529cce989cc03eabb35'
assert sha(physical)=='51201a409995209de3c1e97295ba0c68f9fe264d5ac9e3c16a57a78acd3e46cb'
width={0:4,1:1,2:8,4:8,5:2,6:2}
def port(d,entry):
 raw=zlib.decompress((d/'owners'/(entry['sha256']+'.zlib')).read_bytes())
 assert len(raw)==entry['bytes'] and sha(raw)==entry['sha256']
 return Owner(raw)
def tensor(o,h,target,shape,strd):
 d=o.node(h);s,v=d[27:29];rank=len(shape);n=math.prod(shape);sz=n*width[target]
 assert d[:2]==(rank,n) and d[12:12+rank]==tuple(shape) and d[16:16+rank]==tuple(strd)
 assert d[72]==target and d[29]==d[83]==0 and d[4:6]==(0,0)
 assert o.row(1,h)[1][2:6]==(s,v,0,sz) and o.row(3,v)[1][2]==0
 _,sz_actual,raw=o.storage(s);assert sz_actual==sz and len(raw)==sz
 return (s,v,raw)
def observation(fa,h,bs,n,shape,stride,ss,vv):
 idx=fa['meta_frame'];assert req(Q[idx-1])[0][:2]==(13,h)
 st,recs=resp(R[idx-1]);assert st==0 and len(recs)==1 and recs[0][0]==13
 md=q(recs[0][2],16)
 assert md[:3]==(len(shape),n,0) and md[3:3+len(shape)]==tuple(shape) and md[7:7+len(shape)]==tuple(stride)
 assert md[11:]==(ss,vv,0,0,len(bs))
 chunks=[]
 for i,j in enumerate(fa['raw_frames']):
  c=min(n-(i*32768),32768)
  assert req(Q[j-1])[0][:4]==(98,h,i*32768,c)
  st,recs=resp(R[j-1]);assert st==0 and len(recs)==1 and recs[0][:2]==(98,c)
  chunks.append(recs[0][2])
 assert b''.join(chunks)==bs
summ={}
for mode in ('old','new'):
 d=stage/mode;a=json.loads((d/'report.json').read_bytes())
 assert a['mode']==mode and a['no_torch'] and a['elf']==(stage/(mode+'.pin')).read_text().strip()
 Q=list(frames(d/'requests.frames.bin',b'HTOWNR01'));R=list(frames(d/'responses.frames.bin',b'HTOWNS01'))
 assert len(Q)==len(R)==a['frames'] and len(a['rows'])==len(cases)==68
 seen=set();rows=[]
 for x in a['rows']:
  label=x['label'];assert label not in seen;seen.add(label);c=cases[label]
  n=c['input_count'];tag=c['input_tag'];w=width[tag];shape=c['input_shape'];a0,a1,a2,a3=shape;stride=c['input_strides']
  old=port(d,x['pre']);h=x['input_handle'];s,v,b=tensor(old,h,tag,shape,stride)
  assert sha(b)==c['input_raw_sha256']
  if tag==1:assert all(i<=1 for i in b)
  setup=x['setup_frame'];fq=Q[setup-1];assert req(fq)==[(97,tag,96,0,0,0,0,0)]
  assert q(fq,8,96)==(4,0,a0,a1,a2,a3,160,n*w) and fq[160:]==b
  fst,frecs=resp(R[setup-1]);assert fst==0 and len(frecs)==1 and frecs[0][:2]==(97,h)
  observation(x['input_read'],h,b,n,shape,stride,s,v)
  fr=x['frame'];assert req(Q[fr-1])==[(135,h,0,0,0,0,0,0)]
  st,recs=resp(R[fr-1]);assert x['status']==st
  out=port(d,x['post']);info=dict(label=label,frame=fr,status=st)
  if st==7:
   assert mode=='old' and tag!=1 and not recs and old.b==out.b;rows.append(info);continue
  assert st==0 and (mode=='new' or tag==1)
  hy=x['output_handle'];assert len(recs)==1 and recs[0][:2]==(135,hy)
  k=c['k'];sz,vz,coords=tensor(out,hy,2,[k,4],[1,k])
  answer=L[label];by=answer['bytes'];la=answer['logical_offset'];pa=answer['physical_offset']
  wantlogical=logical[la:la+by];wantphysical=physical[pa:pa+by]
  # Independent source traversal from actual owned bytes, not fixture generator.
  mask=(1<<(8*w-int(tag in (0,4,5,6))))-1
  words=struct.iter_unpack({1:'<B',2:'<H',4:'<I',8:'<Q'}[w],b)
  pos=[]
  for i,(val,) in enumerate(words):
   if not val&mask:continue
   a,t=divmod(i,a1*a2*a3);bb,t=divmod(t,a2*a3);cc,dd=divmod(t,a3)
   pos.append((a,bb,cc,dd))
  want_l=b''.join(struct.pack('<qqqq',*p) for p in pos)
  assert [sha(coords[j*8*k:(j+1)*8*k]) for j in range(4)]==answer['physical_column_sha256']==answer['logical_column_sha256']
  want_p=b''.join(struct.pack('<q',z) for dim in zip(*pos) for z in dim)
  assert len(pos)==k and want_l==wantlogical and want_p==wantphysical==coords
  assert hy!=h and sz!=s and vz!=v and tensor(out,h,tag,shape,stride)==(s,v,b)
  assert old.b[128:160]==out.b[128:160]
  observation(x['output_read'],hy,wantlogical,4*k,[k,4],[1,k],sz,vz)
  observation(x['input_reread'],h,b,n,shape,stride,s,v)
  if x['order']=='input-first':observation(x['survival_read'],hy,wantlogical,4*k,[k,4],[1,k],sz,vz)
  else: assert x['order']=='output-first';observation(x['survival_read'],h,b,n,shape,stride,s,v)
  info.update(k=k,logical_sha256=sha(wantlogical),physical_sha256=sha(coords),order=x['order'])
  rows.append(info)
 summ[mode]=dict(rows=rows,frames=len(Q),q_sha256=sha((d/'requests.frames.bin').read_bytes()),
  r_sha256=sha((d/'responses.frames.bin').read_bytes()),admitted=sum(x['status']==0 for x in rows),
  refused=sum(x['status']==7 for x in rows),words=sum(4*x.get('k',0) for x in rows),
  physical_distinct=sum(x.get('logical_sha256')!=x.get('physical_sha256') for x in rows))
assert summ['old']['refused']==65 and summ['old']['admitted']==3 and summ['old']['words']==24
assert summ['new']['admitted']==68 and summ['new']['words']==5776 and summ['new']['physical_distinct']==31
assert sum(x['k']==0 for x in summ['new']['rows'])==31
print(json.dumps(dict(status='PASS_OFFLINE_BOTH_ORDERS_AND_OWNER',summary=summ),sort_keys=True))
