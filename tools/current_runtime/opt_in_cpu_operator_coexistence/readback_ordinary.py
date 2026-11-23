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
D,F,L=map(Path,sys.argv[1:])
assert sha(F.read_bytes())=='7f492ae8d36bdb1c6523a5a42d09c2009702298df6798ec69258838071d060b5'
assert sha(L.read_bytes())=='419851b7c3259fc129a1cef9242c8ce5da3f937406cfdd0254fb8a413b38a108'
f=json.loads(F.read_bytes());lo=json.loads(L.read_bytes()); assert not lo['mismatches']
OR={(x['id'],x['mode']):x['report'] for x in lo['rows']}
DT={'Bool':(1,1),'I64':(2,8),'F32':(0,4),'F64':(4,8),'F16':(5,2),'BF16':(6,2)}
def physical(s):
 shape=s['shape']; st=s['strides']; tag,w=DT[s['dtype']]; logical=bytes.fromhex(s['raw_le'])
 b=bytearray(len(logical))
 slots=set()
 for i,idx in enumerate(itertools.product(*(range(j) for j in shape))):
  j=sum(x*k for x,k in zip(idx,st,strict=True));assert j not in slots and 0<=j<math.prod(shape)
  slots.add(j);b[j*w:(j+1)*w]=logical[i*w:(i+1)*w]
 assert len(slots)==math.prod(shape)
 return bytes(b)
def tensor(o,h,s):
 d=o.node(h);t,w=DT[s['dtype']];rank=len(s['shape']);n=math.prod(s['shape']);z=n*w
 assert d[:2]==(rank,n) and d[12:12+rank]==tuple(s['shape']) and d[16:16+rank]==tuple(s['strides'])
 assert d[72]==t and d[29]==d[83]==0 and d[4:6]==(0,0)
 st,v=d[27:29]
 assert o.row(1,h)[1][2:6]==(st,v,0,z) and o.row(3,v)[1][2]==0
 _,nbytes,rb=o.storage(st); assert nbytes==z and rb==physical(s)
 return st,v
def same_oracle(label,n,s,raw,ph,mode):
 x=OR[label,mode][n]
 assert x['logical_sha256']==sha(raw) and x['physical_sha256']==sha(ph)
 assert x['meta'][1:3]==[s['shape'],s['strides']] and x['meta'][4:9]==[0,0,len(raw),False,False]
def snap(d,a):
 b=zlib.decompress((d/'owners'/(a['sha256']+'.zlib')).read_bytes())
 assert sha(b)==a['sha256'] and len(b)==a['bytes'];return Owner(b)
def rone(R,i,op):
 st,res=resp(R[i-1]);assert st==0 and len(res)==1 and res[0][0]==op
 return res[0][1:]
def observed(Q,R,z,s):
 h,st,v=(z[k] for k in ('handle','storage','version')); j=z['meta_frame'];k=z['read_frame']
 tag,w=DT[s['dtype']];rank=len(s['shape']);n=math.prod(s['shape'])
 assert req(Q[j-1])[0][:2]==(13,h) and req(Q[k-1])[0][:4]==(98,h,0,n)
 m=q(rone(R,j,13)[1],16)
 assert m[:3]==(rank,n,0) and m[3:3+rank]==tuple(s['shape']) and m[7:7+rank]==tuple(s['strides'])
 assert m[11:]==(st,v,0,0,n*w)
 c,b=rone(R,k,98);assert c==n and b==bytes.fromhex(s['raw_le'])
 return h,st,v
metrics={}
for lane in ('combined','ms','la','nz'):
 d=D/lane; rep=json.loads((d/'report.json').read_bytes()); assert rep['no_torch'] and rep['lane']==lane
 Q=list(frames(d/'requests.frames.bin',b'HTOWNR01')); R=list(frames(d/'responses.frames.bin',b'HTOWNS01'))
 assert len(Q)==len(R)==rep['frames'] and len(rep['cases'])==len(f['cases'])==36
 nobs=0; changed_order=0; checked_words=0; untouched=0; checks_by_opcode={}
 for exp,rec in zip(f['cases'],rep['cases'],strict=True):
  assert exp['id']==rec['id']; spec=exp['inputs']|exp['expect']
  graph=[x for x in exp['composed_reference'] if lane=='combined' or x[0].split('_')[0]=={'ms':'masked','la':'logic','nz':'nonzero'}.get(lane)]
  assert len(graph)==len(rec['outputs'])
  alive={}
  for name,val in rec['factories'].items():
   s=spec[name];tag,w=DT[s['dtype']];n=math.prod(s['shape']);h,st,v=observed(Q,R,val['observation'],s)
   setup=val['setup']; fr=Q[setup-1]
   assert req(fr)==[(97,tag,96,0,0,0,0,0)]
   assert q(fr,8,96)==(len(s['shape']),0,*s['shape'],*[0]*(4-len(s['shape'])),160,n*w)
   assert fr[160:]==bytes.fromhex(s['raw_le']) and rone(R,setup,97)[0]==h
   alive[name]=(h,st,v)
  for (name,op,*args),obs in zip(graph,rec['outputs'],strict=True):
   assert [obs['name'],obs['operator'],obs['args']]==[name,op,args]
   fr=obs['frame'];cmd=[op,*[alive[k][0] for k in args]]
   assert req(Q[fr-1])==[tuple(cmd+[0]*(8-len(cmd)))]
   a=snap(d,obs['pre']); b=snap(d,obs['post'])
   for k,(h,st,v) in alive.items():
    assert tensor(a,h,spec[k])==tensor(b,h,spec[k])==(st,v)
    untouched+=1
   s=spec[name];h,st,v=observed(Q,R,obs['observation'],s)
   assert rone(R,fr,op)[0]==h and tensor(b,h,s)==(st,v)
   assert all(all(j!=k for j,k in zip((h,st,v),z,strict=True)) for z in alive.values())
   alive[name]=(h,st,v)
   raw=bytes.fromhex(s['raw_le']); ph=physical(s)
   same_oracle(rec['id'],name,s,raw,ph,'composed' if lane=='combined' else 'isolated')
   changed_order+=op==135 and raw!=ph;checked_words+=math.prod(s['shape'])
   nobs+=1;checks_by_opcode[op]=checks_by_opcode.get(op,0)+1
  assert set(r['name'] for r in rec['drops'])==set(alive)
  for ob in rec['drops']:
   h=alive[ob['name']][0];idx=ob['frame']
   assert req(Q[idx-1])[0][:2]==(41,h);rone(R,idx,41)
 metrics[lane]=dict(cases=36,ordinary_outputs=nobs,untouched_live_snapshots=untouched,words=checked_words,rank2_different_raw_order=changed_order,frames=len(Q),checks_by_opcode=checks_by_opcode,image=rep['image'],request_sha256=sha((d/'requests.frames.bin').read_bytes()),response_sha256=sha((d/'responses.frames.bin').read_bytes()))
assert [metrics[k]['ordinary_outputs'] for k in ('combined','ms','la','nz')]==[129,36,36,57]
assert metrics['combined']['rank2_different_raw_order']==21
print(json.dumps(dict(status='PASS',scope='reviewed ordinary v2 numerical fixture only',metrics=metrics),sort_keys=True,indent=2))
