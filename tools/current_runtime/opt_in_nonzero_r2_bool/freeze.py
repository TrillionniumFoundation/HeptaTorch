"""Independent detached ordinary Bool rank2 nonzero fixtures, stdlib-only."""
from pathlib import Path
import hashlib, json, struct, sys, zlib

N_CAP,K_CAP=262144,131072
sha=lambda b:hashlib.sha256(b).hexdigest()
q=lambda arr:b''.join(struct.pack('<q',x) for x in arr)
def source(shape,recipe):
 rows,cols=shape;n=rows*cols;kind=recipe['kind']
 if kind=='all':return b'\1'*n
 if kind=='none':return bytes(n)
 if kind=='prefix':
  k=recipe['count'];assert 0<=k<=n
  return b'\1'*k+bytes(n-k)
 if kind=='on':
  points=[tuple(p) for p in recipe['points']]
  assert len(points)==len(set(points)) and all(0<=i<rows and 0<=j<cols for i,j in points)
  b=bytearray(n)
  for i,j in points:b[i*cols+j]=1
  return bytes(b)
 if kind=='coordinate':
  return bytes(int((i*13+j*7+i*j)%29 in (0,2,5,11,13,17,19)) for i in range(rows) for j in range(cols))
 raise AssertionError(kind)

def make(label,shape,r):
 a,b=shape;n=a*b;actual=source(shape,r)
 # Generate lexicographic row/column coordinates via rectangular loops, not flat div/mod.
 coords=[(i,j) for i in range(a) for j in range(b) if actual[i*b+j]]
 k=len(coords); logical=q(x for row in coords for x in row)
 physical=q(x for dim in (0,1) for row in coords for x in (row[dim],))
 exp=dict(dtype='int64',shape=[k,2],strides=[1,k],storage_bytes=16*k,offset=0,
       requires_grad=False,is_contiguous=k<=1,view=False,version=0,
       logical_sha256=sha(logical),physical_sha256=sha(physical),
       head_coords=coords[:10],tail_coords=coords[-10:])
 if k<=32:exp.update(logical_hex=logical.hex(),physical_hex=physical.hex())
 if n>N_CAP:policy='input-count'
 elif k>K_CAP:policy='output-count'
 else:policy='admit'
 return dict(label=label,input_shape=shape,input_strides=[max(b,1),1],
             input_dtype='bool',input_count=n,input_sha256=sha(actual),recipe=r,k=k,
             candidate_policy=policy,candidate_status=0 if policy=='admit' else 3,
             parent_status_if_ordinary_setup_proven=7,expected_torch=exp)
A=dict(kind='all');Z=dict(kind='none');M=dict(kind='coordinate')
P=lambda *pts:dict(kind='on',points=pts)
T=lambda k:dict(kind='prefix',count=k)
rows=[
 make('empty-both-zero',[0,0],Z),make('empty-leading-zero',[0,11],Z),make('empty-trailing-zero',[11,0],Z),
 make('empty-leading-zero-cap-axis',[0,262144],Z),make('empty-trailing-zero-cap-axis',[262144,0],Z),
 make('one-by-one-false',[1,1],Z),make('one-by-one-true',[1,1],A),
 make('nonempty-all-false',[3,7],Z),make('single-selected-not-origin',[3,7],P((2,5))),
 make('single-row-irregular',[1,19],P((0,18),(0,2),(0,3),(0,0),(0,11))),
 make('single-column-irregular',[19,1],P((18,0),(2,0),(3,0),(0,0),(11,0))),
 make('rank2-historical-control',[2,3],P((0,0),(0,2),(1,1))),
 make('asymmetric-axes',[4,7],P((1,0),(0,6),(3,6),(2,1),(1,5),(0,2))),
 make('swapped-asymmetric-axes',[7,4],P((0,1),(6,0),(6,3),(1,2),(5,1),(2,0))),
 make('dense-small',[3,5],A),make('mixed-row-order',[31,17],M),
 make('grain-near-noncrossing-cols',[129,257],P((0,0),(0,256),(1,0),(126,256),(127,128),(127,129),(128,256))),
 make('grain-near-crossing-cols',[257,128],P((0,0),(1,0),(255,126),(255,127),(256,0),(256,1),(256,127))),
 make('input-cap-all-false',[512,512],Z),
 make('input-cap-sparse',[512,512],P((0,511),(1,0),(63,511),(64,0),(255,511),(256,0),(511,511))),
 make('output-last-below-cap',[512,512],T(K_CAP-1)),
 make('output-exact-cap',[512,512],T(K_CAP)),
 make('output-first-above-cap',[512,512],T(K_CAP+1)),
 make('output-exact-cap-dense-smaller-input',[256,512],A),
 make('input-first-above-cap-even-zero-output',[5,52429],Z),
 make('input-first-above-cap-two-output',[5,52429],P((0,1),(4,52428))),
]

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

dest,older,prior_measured=map(Path,sys.argv[1:4])
head,tree=sys.argv[4:6]
assert all(len(h)==40 and all(x in '0123456789abcdef' for x in h) for h in (head,tree))
assert len(sys.argv)==6
old=json.loads((older/'report.json').read_bytes())
assert old['native_frames']==370 and old['elf']=='95aadc064a5a16a188988b699b81dda287443b0e7172fa163dde15db3857c6c7'
prior=prior_measured.read_bytes()
assert sha(prior)=='907d07066493f982e3802a870d4a3bb8110ce85e44363611abfec3ab30ecb058'
earlier,=[x for x in json.loads(prior)['guards'] if x['label']=='rank2-control']
row,=[x for x in rows if x['label']=='rank2-historical-control']
pred=row['expected_torch']
assert earlier['logical_raw_i64le_hex']==pred['logical_hex'] and earlier['physical_i64le_hex']==pred['physical_hex']
assert all(earlier['metadata'][k]==pred[k] for k in ('shape','strides','storage_bytes','version','view','offset'))
assert earlier['metadata']['contiguous']==pred['is_contiguous']
hist,=[x for x in old['rows'] if x.get('label')=='rank2-control']
assert hist['frame']==342 and hist['pre']==hist['post'] and hist['status']==7
Q=list(frames(older/'requests.frames.bin',b'HTOWNR01'));R=list(frames(older/'responses.frames.bin',b'HTOWNS01'));assert len(Q)==len(R)==370
f=hist['frame'];oq=Q[f-1];orr=R[f-1];fq=Q[f-2];fr=R[f-2]
assert sha(oq)==hist['q_sha256'] and resp(orr)==(7,[])
c,=req(oq);h=c[1];assert c==(135,h,0,0,0,0,0,0)
status,r=resp(fr);assert status==0 and len(r)==1 and r[0][:2]==(97,h)
r,=req(fq);assert r[:3]==(97,1,96)
payload=source(row['input_shape'],row['recipe'])
assert tuple(struct.unpack_from('<8Q',fq,96))==(2,0,2,3,0,0,160,6) and fq[160:]==payload
before=hist['pre'];port=zlib.decompress((older/'owners'/(before['sha256']+'.zlib')).read_bytes())
assert len(port)==before['length'] and sha(port)==before['sha256']
own=Owner(port);d=own.node(h);s,v=d[27:29]
assert d[:2]==(2,6) and d[12:14]==(2,3) and d[16:18]==(3,1) and d[72]==1 and d[29]==d[83]==0 and d[4:6]==(0,0)
assert own.row(1,h)[1][2:6]==(s,v,0,6) and own.row(3,v)[1][2]==0 and own.storage(s)[1:]==(6,payload)
hist=dict(historical_git='cb83670b272a6a9816b12c4ac8b2ef6f5473a087',
 history_capsule_report_sha256=sha((older/'report.json').read_bytes()),
 request_stream_sha256=sha((older/'requests.frames.bin').read_bytes()),
 response_stream_sha256=sha((older/'responses.frames.bin').read_bytes()),
 native_elf=old['elf'],physical_ordinal=f,setup_ordinal=f-1,setup_q_sha256=sha(fq),setup_r_sha256=sha(fr),
 query_sha256=sha(oq),response_sha256=sha(orr),pre_equals_post=True,context_owner_sha256=sha(port),
 tensor=h,storage=s,version=v,prior_locked_metadata_and_both_byte_orders_verified=True,
 current_parent_disposition='Source rank1 only; obtain same exact valid setup status7 on parent image during physical phase, not inferred from this different exact old image.')
row['historical_exact']=hist
ad=[r for r in rows if r['candidate_status']==0]
summary=dict(cases=len(rows),scope=len(ad),reference_only=len(rows)-len(ad),
 output_words=sum(2*r['k'] for r in rows),scoped_output_words=sum(2*r['k'] for r in ad),
 scope_k0=sum(r['k']==0 for r in ad),scope_k1=sum(r['k']==1 for r in ad),
 exact_input_cap_in_scope=sum(r['input_count']==N_CAP for r in ad),
 exact_output_cap_in_scope=sum(r['k']==K_CAP for r in ad),exact_context_witness=1)
obj=dict(tier='DETACHED_ORDINARY_NUMERICAL_STD_LIB_PRE_REG',parent_git=head,parent_tree=tree,
 combined='4b3274114665152bb66544a88b49b11047b1aac6d0b5438c3dc07727b01d0055',
 elf='f952bf025e63f6545df424dff55e0d5a64cea196fc06a14d53ea76b5a37a9487',
 thresholds=dict(max_input_values=N_CAP,max_result_values=2*K_CAP,max_selected_count=K_CAP),
 wheel='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3',
 torch_commit='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e',
 cases=rows,summary=summary,notes=[
 'Only rank2 ordinary owned contiguous zero-offset Bool CPU is new; rank1 all six dtypes stays exact. No new wire or public default.',
 'K=0 shapes [0,2], element strides [1,0], no output payload/address; when input has zero count skip both passes without div/mod. [K,2] noncontiguous except K=0 or K=1.',
 'Future physical require source full raw readback + ordinary owner, no synthetic ownership. Coordinates row-major logical, raw output Storage is columns; verify independent whole physical storage and op98 logical qwords separately. v1 candidate guard selected K<=131072 so 2*K<=262144 (existing ordinary validator bound), not a global allocator maximum.',
 'For valid ordinary out-of-profile controls use rank2 true N=262145 via [5,52429] (each dim within op97 max 262144 and packet <1MiB) and normal N=262144, K=131073 via [512,512]. Require actual setup readback before distinguishing input-cap or result-cap status. No fault/OOM or malformed setup.',
 'Potential native output layout: allocate only after proofs using ordinary [K,2] then require private ownership, byte span and actual 2*K, set both stride fields to 1,K before publication. If using typed_store, explicit byte width 8 and direct column-major offsets j,j+K with nonempty/bounded proof; do not call ordinary C-layout validator on a valid final strided K>1 tensor. No ambient flag or raw storage assumption.',
 'For floating/int rank2, non-scope; no fault, invalid-ownership, pressure, held Cat, or other runtime experiments. Compare fresh input/output ownership and Version, not pointer equality in empties.',
 'Old archived rank2 status7 is exact previous release context (owner+factory+ordinal) only; current parent must independently exhibit it before retagging. Never ignore old frames by bare request hash.'
 ])
blob=(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n').encode()
with dest.open('xb') as f:f.write(blob)
print(len(blob),sha(blob),summary)
