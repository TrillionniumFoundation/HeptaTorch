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
CAP=262144
T={
'F64':(4,8,(0,0x8000000000000000,1,0x8000000000000001,0x000fffffffffffff,0x800fffffffffffff,0x0010000000000000,
 0x8010000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000001abc,0xfff8000000002abc,
 0x7ff0000000001357,0xfff0000000002468,0x3ff0000000000000,0xbff0000000000000,0x7fefffffffffffff,0xffefffffffffffff)),
'F16':(5,2,(0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,0x7c00,0xfc00,0x7e35,0xfe57,0x7c35,0xfc57,0x3c00,0xbc00,0x7bff,0xfbff)),
'BF16':(6,2,(0,0x8000,1,0x8001,0x007f,0x807f,0x0080,0x8080,0x7f80,0xff80,0x7fc5,0xffd7,0x7f85,0xff97,0x3f80,0xbf80,0x7f7f,0xff7f)),
'Bool':(1,1,(0,1,1,0,0,1,0,1,1,1,0,0,1,0,1,0,0,1))}
def points(s):return itertools.product(*(range(k) for k in s))
def mask(i,c,n):
 v=c['kind']
 if v=='literal':return c['bits'][i]
 if v=='strided':return int(i%c['period'] in c['keep'])
 if v=='edges':return int(i in (0,n-1))
 assert v in ('none','all');return int(v=='all')
def word(i,shape,c,d):
 _,w,p=T[d];v=c['kind'];seed=c['seed']
 if v=='pattern':return p[(i+seed)%len(p)].to_bytes(w,'little')
 if v=='archived_zeros': assert d=='F64';return bytes(w)
 assert v=='finite_f64_exact_add' and d=='F64'
 a,b=divmod(i,shape[1])
 # Both terms and their sum are exact signed integers as binary64.
 return struct.pack('<d',float(((-1 if a&1 else 1)*(2**41+8192*a+seed))+13*b))
def make(lbl,d,x,m,src,msk,status=0):
 tag,w,_=T[d]; n=math.prod(x);nm=math.prod(m); delta=len(x)-len(m)
 assert 1<=len(x)<=4 and 0<=delta and all(a in (1,b) for a,b in zip(m,x[delta:],strict=True))
 assert n<=CAP if status==0 else status==3 and n>CAP
 bits=bytes(mask(i,msk,nm) for i in range(nm));lookup=dict(zip(points(m),bits,strict=True))
 h=hashlib.sha256();v=hashlib.sha256(); raw=bytearray(); witnesses=[]
 k=0
 for i,pos in enumerate(points(x)):
  wp=word(i,x,src,d);v.update(wp)
  key=tuple(0 if md==1 else p for md,p in zip(m,pos[delta:],strict=True))
  if not lookup[key]:continue
  h.update(wp)
  if n<=128:raw.extend(wp)
  if k<18:witnesses.append(dict(i=i,value_coord=pos,mask_coord=key,output=k,word_hex=wp.hex()))
  k+=1
 result=dict(words=k,dtype=d,shape=[k],strides=[1],version=0,offset=0,storage_bytes=k*w,raw_sha256=h.hexdigest(),first18=witnesses)
 if n<=128:result['raw_le_hex']=raw.hex()
 return dict(label=lbl,dtype=d,dtype_tag=tag,byte_width=w,value_shape=x,mask_shape=m,value_count=n,mask_count=nm,
  source=src,mask=msk,old_status=7,new_hypothesized_status=status,value_sha256=v.hexdigest(),mask_sha256=sha(bits),expected=result)
a=dict(kind='all');z=dict(kind='none');e=dict(kind='edges')
lit=lambda *bits:dict(kind='literal',bits=bits)
pat=lambda i:dict(kind='pattern',seed=i)
period=lambda n,*xs:dict(kind='strided',period=n,keep=xs)
S=[
 ('equal-all-raw-signed',[18],[18],a),('equal-nontrivial-signed',[3,6],[3,6],period(5,0,2,4)),
 ('scalar-true',[2,3,6],[],lit(1)),('scalar-false',[2,3,4],[],lit(0)),
 ('missing-leading',[2,3,4],[4],lit(0,1,0,1)),
 ('interior-singleton',[2,3,4],[2,1,4],lit(1,0,1,0,0,1,0,1)),
 ('multi-singleton',[2,3,2,4],[1,3,1,4],period(4,0,3)),
 ('equal-zero',[0],[0],z),('broadcast-zero',[2,0,4],[1,1,4],lit(0,1,0,1)),
 ('expanded-zero-mask',[0,4],[0,1],z)]
C=[
 ('cap-scalar-all',[512,512],[],lit(1),0),('cap-equal-alternating',[512,512],[512,512],period(2,0),0),
 ('cap-scalar-none',[512,512],[],lit(0),0),('overcap-two',[513,512],[513,512],e,3),
 ('overcap-none',[513,512],[],lit(0),3)]
items=[]
for d in T:
 for i,(l,x,m,msk) in enumerate(S):items.append(make(d+'-'+l,d,x,m,pat(i),msk))
 for i,(l,x,m,msk,status) in enumerate(C):
  sr=dict(kind='finite_f64_exact_add',seed=100+i) if d=='F64' else pat(i+7)
  items.append(make(d+'-'+l,d,x,m,sr,msk,status))
 # Pin max chunk plan at standard 32768-word boundary for each width.
 for k in (32767,32768,32769):items.append(make(d+'-chunk-'+str(k),d,[k],[k],pat(k%9),a))
items.append(make('F64-historical-value-accept','F64',[2,3],[1,3],dict(kind='archived_zeros',seed=0),lit(1,0,1)))
def history(p):
 qr=p/'requests.frames.bin';rr=p/'responses.frames.bin'
 assert sha(qr.read_bytes())=='514c5bfde7d6df7464fe0eda0d7f7695f4292fc7a0f6e5c834fd005fc90de634'
 assert sha(rr.read_bytes())=='2dcadcaa107117f941f05c1c30a217ea7e632993cc3758ae4027f6858fd77adb'
 Q=list(frames(qr,b'HTOWNR01'));R=list(frames(rr,b'HTOWNS01')); assert len(Q)==len(R)==706
 report=json.loads((p/'report.json').read_bytes()); A={x.get('label'):x for x in report['rows']}
 receipt={}
 for name,at,xd,md,xs,ms,rawx,rawm in (
 ('other-value-F64-remains-unadmitted',685,4,1,(2,3),(1,3),bytes(48),bytes([1,0,1])),
 ('nonbool-value-used-as-mask',684,2,2,(2,3),(1,3),struct.pack('<6q',*range(6)),struct.pack('<3q',1,0,1))):
  row=A[name]; a,b=row['pre'],row['post'];assert a==b and row['frame']==at and row['status']==7
  item=Q[at-1]; resp0=R[at-1]; assert sha(item)==row['q_sha256'] and resp(resp0)==(7,[])
  assert sum(sha(q0)==sha(item) for q0 in Q)==1
  o=zlib.decompress((p/'owners'/(a['sha256']+'.zlib')).read_bytes());assert sha(o)==a['sha256'] and len(o)==a['length']; owner=Owner(o)
  cmd,=req(item);assert cmd[0]==130 and cmd[3:]==(0,)*5
  for ht,tag,shape,w in ((cmd[1],xd,xs,rawx),(cmd[2],md,ms,rawm)):
   n=owner.node(ht);st,vr=n[27:29]
   assert n[:2]==(len(shape),math.prod(shape)) and n[12:14]==shape and n[16:18]==(3,1)
   assert n[72]==tag and n[29]==n[83]==0 and n[4:6]==(0,0)
   assert owner.row(3,vr)[1][2]==0 and owner.storage(st)[1:]==(len(w),w)
  receipt[name]=dict(frame=at,request_sha256=sha(item),response_sha256=sha(resp0),owner_sha256=sha(o),
   old_status=7,new_expected=0 if xd==4 else 7,old_operand_raw=[rawx.hex(),rawm.hex()])
 return receipt
guards=[
 dict(label='integer-mask-still-forbidden',dtype='I64',mask='I64',status=7,torch_accept=False,historical_alias='nonbool-value-used-as-mask'),
 dict(label='floating-mask-still-forbidden',dtype='BF16',mask='F64',status=7,torch_accept=False),
 dict(label='I64-mask-on-Bool-value',dtype='Bool',mask='I64',status=7,torch_accept=False),
 dict(label='F64-mask-on-F16-value',dtype='F16',mask='F64',status=7,torch_accept=False),
 dict(label='rank0-value',dtype='F64',value_shape=[],mask_shape=[],status=7,torch_accept=True),
 dict(label='symmetric-expansion',dtype='F16',value_shape=[2,1,4],mask_shape=[1,3,4],status=7,torch_accept=True),
 dict(label='incompatible-zero',dtype='BF16',value_shape=[2,0,4],mask_shape=[1,2,4],status=7,torch_accept=False),
 dict(label='higher-mask-rank',dtype='Bool',value_shape=[3,4],mask_shape=[1,3,4],status=7,torch_accept=True),
 dict(label='actual-offset-value-view',dtype='F64',status=7,torch_accept=True),
 dict(label='actual-offset-mask-view',dtype='Bool',status=7,torch_accept=True),
 dict(label='actual-grad-value',dtype='F16',status=7,torch_accept=True),
 dict(label='stale-capability',status=1),dict(label='padding',status=7),dict(label='valid-then-unknown',status=4)]
summary=dict(cases=len(items),positives=sum(x['new_hypothesized_status']==0 for x in items),
 sums={d:sum(x['expected']['words'] for x in items if x['dtype']==d and x['new_hypothesized_status']==0) for d in T},
 cap=sum(x['value_count']==CAP for x in items),overcap=sum(x['value_count']>CAP for x in items),guards=len(guards))
p=dict(kind='FROZEN_PREDICTION_ONLY',stacked_source='c5198ea3ddbbfbc3aeff244997dfa05660dba1c98d05edcf42d18c899b40768f',
 stacked_elf='5130bc446381049134363953a687656e7bbfe9bfe6fb87c331383e3acaf27cd8',
 publication_base='c08694a893c76da9c898df4e018b864ec0b1f124', publication_tree='ea4a8114ab9cdb688b3f081bf0f697d58d78578b',match_threads=1,
 torch_lock='ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944',
 torch_wheel_sha256='0955c7e3446326293f48af18fad67872fa31dee5a36a87690eb9e5be8b465be3',
 torch_commit='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e',
 semantic='Only Bool mask determines selected coordinate, independent of value bits. Value output copied raw; pinned Torch must measure payload corners, not assume casts.',
 constructor_plan={'F64':'Hypothesis ordinary row [R,1] and col [1,C] exact integer binary64 values, functional binary(op2 add1) with existing F64 add route. Require fully independent physical raw and owner before selection; no cat.',
 'F16/BF16':'Use raw typed factories: 2*N + 160 < 1MiB at both N boundaries; no arithmetic or casting.',
 'Bool':'Use typed canonical Bool source 0/1 and fresh independent canonical Bool mask; one byte per value. Verify values[mask], never replace selection with nonzero(values).',
 'output_read':'Read chunks no more than 32768 words. Full original values checked before cap tests, not inferred.'},
 cases=items,guards=guards,historical=history(Path(sys.argv[1])),summary=summary)
b=(json.dumps(p,sort_keys=True,separators=(',',':'))+'\n').encode()
with Path(sys.argv[2]).open('xb') as f:f.write(b)
print(len(b),sha(b),summary)
