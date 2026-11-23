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
d,f,l=map(Path,sys.argv[1:])
assert sha(f.read_bytes())=='898b3b2ccb70729d124709d608063e5eec7446161fbe36896319e9c27fecc6f1'
assert sha((l/'receipt.json').read_bytes())=='70015a27b3ec8353b1582413990e4588b6b175323889fc6f8eb438c028699a96'
fixed=(l/'all-f32le.bin').read_bytes();assert sha(fixed)=='e03f2b34c2155e7b827b5aaeaa130abc3426d573e1a9b329e05a411397e282b0'
p=json.loads(f.read_bytes());L={e['label']:e for e in json.loads((l/'receipt.json').read_bytes())['rows']}
Q=list(frames(d/'requests.frames.bin',b'HTOWNR01'));R=list(frames(d/'responses.frames.bin',b'HTOWNS01'))
ops=[req(v)[0][0] for v in Q]
assert len(Q)==len(R)==140 and all(resp(r)[0]==0 for r in R)
assert [o for o in ops if o in (2,4,7,9,11,32)]==[32,2,2,4,7,11,9,11,32,2,2,4]
assert ops.count(97)==3 and ops.count(41)==11 and 91 not in ops
def pkt(i,want,expected_ret=None):
 cmd,=req(Q[i-1]);assert cmd==tuple(want)+(0,)*(8-len(want))
 _,result=resp(R[i-1]);op,v,b=result[0];assert len(result)==1 and op==want[0]
 if expected_ret is not None:assert v==expected_ret
 return v,b
ledger=json.loads((d/'ledger.json').read_bytes())
live={};ever={};exp={};counts=dict(owner_snapshots=0,live_tensors=0,grad_owner_records=0);grad_ref=[]
def bytes_for(label,e):
 z=L[label];raw=fixed[z['offset']:z['offset']+z['length']]
 assert raw.hex()==e['raw_f32le_hex'] and sha(raw)==z['logical_sha256']==z['physical_sha256']
 return raw
for i,(n,e) in enumerate(p['input_tensors'].items(),1):
 h,_=pkt(i,(97,0,96));s=e['shape'];b=bytes_for(n,e)
 assert q(Q[i-1],8,96)==(len(s),int(e['requires_grad']),*s,*([0]*(4-len(s))),160,len(b)) and Q[i-1][160:]==b
 live[n]=h;exp[n]=(n,e)
def inspect(row):
 digest=row['owner'];b=zlib.decompress((d/'owners'/(digest+'.zlib')).read_bytes());assert sha(b)==digest
 o=Owner(b);counts['owner_snapshots']+=1
 assert row['observations'].keys()==live.keys();cur=[]
 for n,v in row['observations'].items():
  h=live[n];lab,e=exp[n];b=bytes_for(lab,e);nd=o.node(h)
  n0=len(b)//4;s=e['shape'];st=e['strides'];rk=len(s);rq=int(e['requires_grad'])
  assert nd[:2]==(rk,n0) and nd[4]==rq and nd[12:12+rk]==tuple(s) and nd[16:16+rk]==tuple(st)
  assert nd[29]==nd[72]==nd[83]==0
  so,vo=nd[27:29];assert all(x!=0 for x in (so,vo))
  assert o.row(1,h)[1][2:6]==(so,vo,0,len(b)) and o.storage(so)[1:]==(len(b),b)
  assert o.row(3,vo)[1][2]==e['version']
  meta=q(pkt(v['meta'],(13,h))[1],16)
  assert meta[:3]==(rk,n0,rq) and meta[3:3+rk]==tuple(s) and meta[7:7+rk]==tuple(st)
  assert meta[11:]==(so,vo,e['version'],0,len(b))
  assert pkt(v['raw'],(98,h,0,n0),n0)[1]==b and (v['handle'],v['storage'],v['version'])==(h,so,vo)
  if n in ever:assert ever[n]==(h,so,vo)
  ever[n]=(h,so,vo);cur.append((h,so,vo))
  if n in ('x','target') or n=='weight' and row['kind']=='input':assert nd[45]==nd[48]==nd[49]==0
 assert all(len({x[i] for x in cur})==len(cur) for i in range(3))
 counts['live_tensors']+=len(cur)
 if row['kind'] in ('backward','sgd'):
  n=o.node(live['weight']);i=n[48]
  assert n[45]!=0 and i>0 and n[49]>0
  page=q(o.b,1,4096+(i>>8)*8)[0];assert page>>63
  at=o.start+(page&MASK)+(i&255)*672
  gd=q(o.b,84,at);assert gd[40]==n[49]
  gh,gs,gv=gd[24],gd[27],gd[28]
  assert o.node(gh)==gd and gd[:2]==(2,2) and gd[4:6]==(0,0) and gd[12:14]==(2,1) and gd[16:18]==(1,1)
  assert gd[29]==gd[72]==gd[83]==0 and o.row(3,gv)[1][2]==0
  want=bytes_for('weight_gradient',p['backward']['expected_gradient'])
  assert o.storage(gs)[1:]==(8,want) and pkt(row['gradient_frame'],(11,live['weight']),2)[1]==want
  assert all((gh,gs,gv)[k] not in {z[k] for z in cur} for k in range(3))
  grad_ref.append((gh,gs,gv));counts['grad_owner_records']+=1
assert ledger[0]['kind']=='input'
inspect(ledger[0])
chain={s['name']:s for s in p['before_update']+p['after_update']}
for a in ledger[1:]:
 k=a['kind'];fr=a['frame']
 if k in ('before','after'):
  n=a['stage'];s=chain[n]
  cmd=[live[v] if isinstance(v,str) else v for v in s['command']]
  assert a['command']==cmd
  live[n]=pkt(fr,cmd)[0];exp[n]=(n,s['expected']);inspect(a)
 elif k=='backward':
  pkt(fr,(7,live['loss']));inspect(a)
 elif k=='sgd':
  assert set(live)==set(p['input_tensors'])
  pkt(fr,(9,live['weight'],0x3e000000))
  exp['weight']=('one_sgd_weight_update',p['update']['weight_after']);inspect(a)
 elif k.startswith('release-'):
  v=a['stage']
  pkt(fr,(41,live.pop(v)))
 else:raise AssertionError(k)
assert not live and len(grad_ref)==2 and grad_ref[0]==grad_ref[1]
assert counts==dict(owner_snapshots=11,live_tensors=57,grad_owner_records=2)
print(json.dumps(dict(status='PASS_INDEPENDENT_WIRE_AND_OWNER',frames=len(Q),counts=counts,
 requests_sha256=sha((d/'requests.frames.bin').read_bytes()),responses_sha256=sha((d/'responses.frames.bin').read_bytes()),
 exact_torch_words=len(fixed)//4,one_backward=True,one_sgd=True,old_graph_released_before_step=True),sort_keys=True))
