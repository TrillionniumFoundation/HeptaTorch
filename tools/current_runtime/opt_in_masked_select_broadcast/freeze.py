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
ELF='75401b00356b8941960fae2a2207c0e046fb0da36cbcef7de26bef41ec4a3deb'
QSHA='5e2b041b6caac278c5c6e277f7524c3c1f0f0652a985408d2ae75beef2212aed'
RSHA='4a39bce6f56d6784cb52003e07b9248f44e6b2431a9579b0c9a07fb783fa6da7'
PAT=(0,0x80000000,1,0x80000001,0x7fffff,0x807fffff,0x800000,0x80800000,0x7f800000,0xff800000,0x7fc12345,0xffc23456,0x7f812345,0xff823456,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff)
def historical(p):
 qp=p/'requests.frames.bin';rp=p/'responses.frames.bin'
 assert sha(qp.read_bytes())==QSHA and sha(rp.read_bytes())==RSHA
 rraw=p/'report.json'; r=json.loads(rraw.read_bytes())
 t=json.loads((p/'transport.json').read_bytes()); assert r['elf']==t['executable_sha256']==ELF and t['exit_code']==0
 rr=[v for v in r['rows'] if v.get('label')=='broadcast']; item,=rr
 j=item['frame']-1
 Q=list(frames(qp,b'HTOWNR01'));R=list(frames(rp,b'HTOWNS01'))
 assert len(Q)==len(R)==t['frames']==57
 assert [i for i,body in enumerate(Q) if sha(body)==item['q_sha256']]==[j]
 cmd,=req(Q[j]);assert cmd==(130,cmd[1],cmd[2],0,0,0,0,0)
 assert resp(R[j])==(7,[]) and item['pre']==item['post'] and item['status']==7
 snap=item['pre'];payload=zlib.decompress((p/'owners'/(snap['sha256']+'.zlib')).read_bytes())
 assert len(payload)==snap['length'] and sha(payload)==snap['sha256']; owner=Owner(payload)
 for h,dims,tag,raw in [(cmd[1],(2,3),0,struct.pack('<6I',*range(6))),(cmd[2],(1,3),1,b'\1'*3)]:
  n=owner.node(h); st,ver=n[27:29]
  assert n[:2]==(2,math.prod(dims)) and n[12:14]==dims and n[16:18]==(3,1)
  assert n[4:6]==(0,0) and n[29]==n[83]==0 and n[72]==tag
  assert owner.row(1,h)[1][2:6]==(st,ver,0,len(raw))
  assert owner.storage(st)[1:]==(len(raw),raw) and owner.row(3,ver)[1][2]==0
 return dict(frame=j+1,frames=len(Q),old_status=7,elf=ELF,full_requests_sha256=QSHA,full_responses_sha256=RSHA,report_sha256=sha(rraw.read_bytes()),request_sha256=sha(Q[j]),response_sha256=sha(R[j]),request_hex=Q[j].hex(),response_hex=R[j].hex(),pre_post_identical_owner_sha256=sha(payload),request_target_handles=list(cmd[1:3]))
def coords(shape): return itertools.product(*(range(n) for n in shape))
def predict(label,x,m,bit,kind,seed,old=7,new=0):
 n=math.prod(x); nm=math.prod(m)
 assert len(bit)==nm and all(b in (0,1) for b in bit) and 0<=len(m)<=len(x)<=4
 delta=len(x)-len(m)
 assert all(a in (1,b) for a,b in zip(m,x[delta:],strict=True))
 mask=dict(zip(coords(m),bit,strict=True)); h=hashlib.sha256();k=0;raw=bytearray(); witnesses=[]
 for i,c in enumerate(coords(x)):
  mc=tuple(ci if md!=1 else 0 for ci,md in zip(c[delta:],m,strict=True))
  if not mask[mc]:continue
  w=(0x3f000000+(13*i+seed)) if kind=='ordinal' else (i if kind=='archive-exact' else PAT[(i+seed)%len(PAT)])
  record=struct.pack('<I',w);h.update(record)
  if n<=256:raw.extend(record)
  if k<16:witnesses.append(dict(i=i,value_coord=list(c),mask_coord=list(mc),word=f'{w:08x}',out_i=k))
  k+=1
 result=(dict(shape=[k],strides=[1],offset=0,version=0,dtype='float32',raw_u32le_sha256=h.hexdigest(),
  **({'raw_u32le_hex':raw.hex()} if n<=256 else {}),selected_witnesses=witnesses))
 return dict(label=label,x_shape=x,mask_shape=m,value_count=n,mask_count=nm,source=dict(kind=kind,seed=seed),mask_raw_u8_hex=bytes(bit).hex(),
  old_hypothesized_status=old,new_hypothesized_status=new,mathematical_selected=k,expected=result)
recipe=[
 ('archived-broadcast',[2,3],[1,3],[1,1,1],'archive-exact'),
 ('true-scalar-special-r1',[18],[],[1],'special'),('false-scalar-special-r2',[3,6],[],[0],'special'),
 ('true-scalar-special-r3',[2,3,6],[],[1],'special'),('false-scalar-r4',[2,3,4,5],[],[0],'ordinal'),
 ('missing-leading',[2,3,4],[4],[0,1,0,1],'ordinal'),('rank1-singleton',[7],[1],[1],'ordinal'),
 ('trailing-broadcast',[2,3,4],[3,1],[1,0,1],'ordinal'),
 ('interior',[2,3,4],[2,1,4],[0,1,0,1,1,0,1,0],'ordinal'),
 ('multi-singled',[2,3,4],[1,3,1],[0,1,1],'ordinal'),
 ('rank4-repeat',[2,3,4,5],[3,1,5],[int(i%5 in (0,3) and i//5!=1) for i in range(15)],'special'),
 ('rank4-multi-singletons',[2,3,4,5],[1,3,1,5],[int(i%5 in (0,3) and i//5!=1) for i in range(15)],'ordinal'),
 ('zero-last',[3,0],[3,1],[1,0,1],'special'),('zero-interior',[2,0,4],[1,1,4],[0,1,0,1],'special'),
 ('zero-mask',[0,4],[0,1],[],'ordinal'),('zero-with-scalar',[2,0,3],[],[1],'special'),
 ('zero-leading-omission',[0,3,4],[3,4],[1]*12,'ordinal'),
 ('exact-cap-scalar-false',[512,512],[],[0],'ordinal'),('exact-cap-scalar-true',[512,512],[],[1],'ordinal'),
 ('exact-cap-partial',[512,512],[512],[int(i%127==1) for i in range(512)],'ordinal'),
 ('over-cap-scalar-false',[513,512],[],[0],'ordinal'),('over-cap-scalar-true',[513,512],[],[1],'ordinal'),
 ('over-cap-select-two',[513,512],[513,1],[int(i==512) for i in range(513)],'ordinal'),
]
# Two selected actual positions at the over-cap boundary without an expanded mask:
# [131073,2] with mask [131073,1], one selected row. Encode that recipe separately.
recipe.pop()
cases=[predict(*r,i,new=3 if r[0].startswith('over-cap') else 0) for i,r in enumerate(recipe)]
cases.append(predict('over-cap-select-two',[2,131073],[1,131073],[int(i==131072) for i in range(131073)],'ordinal',99,new=3))
c=cases[-1]; del c['mask_raw_u8_hex'];c['mask_formula']={'kind':'one-at-flat-index','index':131072}
guards=[
 dict(label='incompatible-zero-target',x=[2,0,4],mask=[1,2,4],torch_accept=False,status=7),
 dict(label='mask-zero-would-change-values',x=[2,1,4],mask=[1,0,4],torch_accept=True,status=7),
 dict(label='equal-count-transpose',x=[2,3],mask=[3,2],torch_accept=False,status=7),
 dict(label='symmetric-expansion-not-admitted',x=[2,1,4],mask=[1,3,4],torch_accept=True,status=7),
 dict(label='higher-mask-rank-not-admitted',x=[3,4],mask=[1,3,4],torch_accept=True,status=7),
 dict(label='scalar-value-not-admitted',x=[],mask=[],torch_accept=True,status=7),
 dict(label='dtype-i64-mask',status=7),dict(label='actual-view-mask',status=7),dict(label='actual-grad-value',status=7),
 dict(label='stale-mask',status=1),dict(label='nonzero-padding',status=7),dict(label='valid-broadcast-then-unknown',status=4)]
summary=dict(cases=len(cases),positives=sum(a['new_hypothesized_status']==0 for a in cases),
 theoretical_raw_words=sum(a['mathematical_selected'] for a in cases),
 candidate_raw_words=sum(a['mathematical_selected'] for a in cases if a['new_hypothesized_status']==0),
 zero_targets=sum(a['value_count']==0 for a in cases),over_cap=sum(a['value_count']>CAP for a in cases),guards=len(guards))
doc=dict(kind='FROZEN_PREDICTION_ONLY',head='ad964ebaf3f7b3913f039d3ee44802aaff447198',tree_prefix='21bdbf9',
  combined='baa3b155e5965e2c452fae534921a378c26d8b5befafd69b68d1f542b5e249ee',
  pinned_torch_commit='d0f2af8b66b20da4baa50644c0cd3f887a2eae2e',threads_planned=[1,2],
  old_historical=historical(Path(sys.argv[1])),cases=cases,guards=guards,summary=summary)
b=(json.dumps(doc,sort_keys=True,separators=(',',':'))+'\n').encode()
with Path(sys.argv[2]).open('xb') as f:f.write(b)
print(len(b),sha(b),summary)
