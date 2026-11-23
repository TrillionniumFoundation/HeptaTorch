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
T['F32']=(0,4,(0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x7f800000,0xff800000,
 0x7fc12345,0xffc23456,0x7f812345,0xff823456,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff))
T['I64']=(2,8,tuple(x&((1<<64)-1) for x in (-(1<<63),-(1<<63)+1,-(1<<62),-(1<<53)-1,-(1<<32)-1,-1,0,1,
 0x7fffffff,0x80000000,0xffffffff,1<<32,(1<<53)+1,(1<<62)+0x5a,0x5555555555555555,(1<<63)-2,(1<<63)-1)))
bytag={v[0]:d for d,v in T.items()}
def points(shape):
 return itertools.product(*(range(n) for n in shape))
def merge(*shapes):
 r=max(map(len,shapes));a,b=[(1,)*(r-len(s))+tuple(s) for s in shapes]
 out=[]
 for y,z in zip(a,b,strict=True): 
  if y==z or z==1:out.append(y)
  elif y==1:out.append(z)
  else:return None
 return tuple(out)
def key(c,shape):
 return tuple(0 if d==1 else k for k,d in zip(c[len(c)-len(shape):],shape,strict=True))
def words(d,shape,seed):
 _,w,p=T[d]; return b''.join(p[(i+seed)%len(p)].to_bytes(w,'little') for i in range(math.prod(shape)))
def flags(m,style):
 count=math.prod(m)
 if style=='all':return b'\1'*count
 if style=='none':return bytes(count)
 if style=='first':return bytes([1])+bytes(count-1)
 if style=='order':return bytes(int((i*7+2)%11 in (0,1,2,3,7,9)) for i in range(count))
 raise AssertionError(style)
def predict(label,d,x,m,values,bits,old=7,expected=0,bind=None):
 t,w,_=T[d];n0=math.prod(x);n1=math.prod(m); lattice=merge(x,m);assert lattice is not None
 n=math.prod(lattice)
 assert 1<=len(x)<=4 and len(m)<=4 and n0<=CAP and n1<=CAP and len(values)==w*n0 and len(bits)==n1
 assert (n<=CAP and expected==0) or (n>CAP and expected==3)
 xdata=dict(zip(points(x),(values[i:i+w] for i in range(0,len(values),w)),strict=True))
 bdata=dict(zip(points(m),bits,strict=True));assert all(b in (0,1) for b in bits)
 out=hashlib.sha256(); raw=bytearray(); checks=[];k=0
 for i,p in enumerate(points(lattice)):
  mk=key(p,m)
  if not bdata[mk]:continue
  xk=key(p,x);word=xdata[xk];out.update(word)
  if n<=128:raw.extend(word)
  if k<18:checks.append(dict(linear=i,lattice=p,x_coord=xk,mask_coord=mk,word=word.hex(),result_index=k))
  k+=1
 pred=dict(words=k,shape=[k],strides=[1],offset=0,version=0,storage_bytes=k*w,raw_sha256=out.hexdigest(),selected_coordinates=checks)
 if n<=128:pred['raw_le_hex']=raw.hex()
 return dict(label=label,dtype=d,tag=t,width=w,value_shape=x,mask_shape=m,merged_shape=lattice,merged_count=n,
       value_count=n0,mask_count=n1,old_hypothesized_status=old,new_hypothesized_status=expected,
       value_sha256=sha(values),mask_sha256=sha(bits),recipe=bind,expected=pred)
def fixture(label,d,x,m,seed,style,old=7,expected=0):
 return predict(label,d,x,m,words(d,x,seed),flags(m,style),old,expected,dict(seed=seed,mask_style=style))
cases=[]
for d in T:
 rows=[
  ('crossed-signed',[2,1,18],[1,3,1],'order'),
  ('mask-dominant',[1,5],[3,5],'order'),('mixed-rank',[18],[2,3,1],'order'),
  ('rank4-both-expand',[2,1,4,1],[1,3,1,5],'order'),
  ('equal-retained',[2,3],[2,3],'order'),('one-way-retained',[2,3,4],[3,1],'order'),
  ('scalar-mask-retained',[2,3],[],'all')]
 for i,(label,x,m,style) in enumerate(rows):
  old=0 if 'retained' in label else 7
  cases.append(fixture(d+'-'+label,d,x,m,i,style,old))
 cases.append(fixture(d+'-exact-joined-cap',d,[512,1],[1,512],3,'first'))
 cases.append(fixture(d+'-over-joined-cap-K0',d,[512,1],[1,513],4,'none',7,3))
 cases.append(fixture(d+'-over-joined-cap-K512',d,[512,1],[1,513],5,'first',7,3))
zeros=[
 ('zero-leading-mask-dominant','F64',[1,4],[0,4]),
 ('zero-middle-second-singleton','F16',[2,1,4],[1,0,1]),
 ('zero-middle-first','BF16',[1,0,4],[2,1,1]),
 ('zero-late-saturation','I64',[512,1,0],[1,513,1]),
 ('zero-interleaved-two','Bool',[2,0,1,1],[1,1,0,4])]
for i,(l,d,x,m) in enumerate(zeros):cases.append(fixture(l,d,x,m,i,'all'))
def historical(title,p,records,qp,rp):
 qf=p/'requests.frames.bin';rf=p/'responses.frames.bin'
 assert sha(qf.read_bytes())==qp and sha(rf.read_bytes())==rp
 Q=list(frames(qf,b'HTOWNR01'));R=list(frames(rf,b'HTOWNS01'))
 data=json.loads((p/'report.json').read_bytes()); ledger={a.get('label'):a for a in data['rows']}
 out=[]
 for label,at,qsha in records:
  row=ledger[label]; j=at-1; item=Q[j];ans=R[j]
  assert row['frame']==at and sha(item)==row['q_sha256']==qsha and row['status']==7 and resp(ans)==(7,[])
  assert sum(sha(k)==sha(item) for k in Q)==1
  meta=row['pre'];assert meta==row['post']
  raw=zlib.decompress((p/'owners'/(meta['sha256']+'.zlib')).read_bytes());assert len(raw)==meta['length'] and sha(raw)==meta['sha256']
  o=Owner(raw); cmd,=req(item);assert cmd[0]==130 and cmd[3:]==(0,)*5
  z=[]
  for h in cmd[1:3]:
   n=o.node(h);rank,count=n[:2];st,vr=n[27:29];sh=n[12:12+rank];tag=n[72];w=T[bytag[tag]][1]
   assert count==math.prod(sh) and n[29]==n[83]==0 and n[4:6]==(0,0)
   assert o.row(3,vr)[1][2]==0 and o.row(1,h)[1][2:6]==(st,vr,0,count*w)
   s=o.storage(st)[2]; assert len(s)==count*w
   z.append((tag,sh,s))
  (t,x,v),(mt,m,b)=z;assert mt==1 and merge(x,m) is not None
  if len(m)<=len(x):assert merge(x,m)!=x
  pin=dict(corpus=title,old_label=label,frame=at,request_sha256=qsha,response_sha256=sha(ans),identical_owner_sha256=sha(raw),
   operands=[v.hex(),b.hex()],full_requests_sha256=qp,full_responses_sha256=rp)
  out.append(predict('historical-'+title+'-'+label,bytag[t],x,m,v,b,bind=pin))
 return out
f32,i64,typed,out=map(Path,sys.argv[1:])
cases.extend(historical('f32',f32,[('mask-zero-would-change-values',577,'22a8f5b761b1f9b3d6c5f120ed43135d0bfe102e9c2efb5651d73f343101db24'),
 ('symmetric-expansion-not-admitted',587,'e4e79080187cb9328de8105e53f57087d334070c59fb6fafa7f60d26d3744ffb'),
 ('higher-mask-rank-not-admitted',592,'74bb185cfd0a2f676d8caaa0344dc693594be6229d8be39b25f6a4eb17d47300')],
 '76464ee4df5aeeb351ac10f59b15cb850a76408dbd829b575e8a9252cdc00136','e48ad88e9dc1a4c07347358a5a92de524dbfa97b5e1d275c4a9d49c2456ef336'))
cases.extend(historical('i64',i64,[('mask-zero-changes-value-shape',657,'4b580b186a57775b2eee33b998c201b52cbbf6c4137590f981d1e345892371af'),
 ('symmetric-value-expansion',667,'dfa18515b44081cbbfb64ae9b5989a83dd7904e1690d4977c59d0932b41c08d3'),
 ('higher-rank-mask',672,'8e8a68b90cfd68d40528ef9db23be60f11b1713f45c8e8bc5ed4f6eccd287882')],
 '514c5bfde7d6df7464fe0eda0d7f7695f4292fc7a0f6e5c834fd005fc90de634','2dcadcaa107117f941f05c1c30a217ea7e632993cc3758ae4027f6858fd77adb'))
cases.extend(historical('four',typed,[('symmetric-expansion',1945,'274fa9c271e10f229c971e1a9644243348ba6654aa9bd211cf9c3c237c1692df'),
 ('higher-mask-rank',1955,'8f5f56d41dd860a32b29d8d880f7ca35e79789009114197b7b0abd04dcc63f24')],
 'bdb44744595d479ced0cf7f01b7883b46ec601dd02ced18c0d20d97e5f05d9b5','8c081c28daf9f985d6cd123d52f1374705f554d957d1f765a2b2b8cd1f75fb4a'))
guards=[
 dict(label='zero-then-incompatible',value=[0,2,4],mask=[1,3,4],status=7,torch=False),
 dict(label='late-zero-still-incompatible',value=[2,0,4],mask=[3,1,1],status=7,torch=False),
 dict(label='zero-vs-non-singleton',value=[2,0,4],mask=[1,3,1],status=7,torch=False),
 dict(label='equal-count-incompatible',value=[2,3,4],mask=[3,2,4],status=7,torch=False),
 dict(label='rank5-mask',value=[2,3],mask=[1,1,1,2,3],status=7,torch=True),
 dict(label='rank0-value-remains-excluded',value=[],mask=[2,3],status=7,torch=True),
 dict(label='existing-oversize-exact-or-oneway-still-cap',status=3),
 dict(label='private-tag3-value-stays-forbidden',status=7),
 dict(label='nonbool-mask-stays-forbidden',status=7),dict(label='actual-view-mask',status=7),
 dict(label='actual-grad-f32',status=7),dict(label='stale',status=1),dict(label='padding',status=7),dict(label='compaction-then-unknown',status=4)]
summary=dict(ncases=len(cases),successes=sum(x['new_hypothesized_status']==0 for x in cases),
 retained=sum(x['old_hypothesized_status']==0 for x in cases),
 cap=sum(x['merged_count']==CAP for x in cases),overcap=sum(x['merged_count']>CAP for x in cases),
 zero=sum(x['merged_count']==0 for x in cases),historical=sum('historical' in x['label'] for x in cases),guards=len(guards),
 candidate_words=sum(x['expected']['words'] for x in cases if x['new_hypothesized_status']==0))
d=dict(kind='PREDICTION_ONLY',base='0489d49f4f497f02a218a06d64d80081ae004a00',tree='58ba08dbb8f14ac045e836f664b841e250280f7c',
 combined='25a495092c57ade642c4d3648a16bc42ded060a88473c06b60d22080b99ee73f',
 baseline_elf='23ee4cc96d5f000c48dd128fbf65088a1b58b87dd72cd22add6cf0209144ad74',
 torch_commit='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e',summary=summary,cases=cases,guards=guards,
 joined_cap_plan='Small original tensor roots by construction. Result lattice width and count are logical; use 32,768-word wire slices if result needs them. Do not allocate either expanded operand.',
 historical_plan='Eight unique archived request byte strings before replay aliases. Later exact replay must census all occurrences by hash and retire only confirmed semantic changes.')
b=(json.dumps(d,sort_keys=True,separators=(',',':'))+'\n').encode()
with out.open('xb') as h:h.write(b)
print(len(b),sha(b),summary)
