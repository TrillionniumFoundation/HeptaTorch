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
base,f,oracle=map(Path,sys.argv[1:])
assert sha(f.read_bytes())=='cf40ccecf20b5778aa5fe8a79ab5a3401dd9feb841b095e9563ffc62326a2946'
assert sha((oracle/'receipt.json').read_bytes())=='907d07066493f982e3802a870d4a3bb8110ce85e44363611abfec3ab30ecb058'
p=json.loads(f.read_bytes());locked=json.loads((oracle/'receipt.json').read_bytes())
fixed={a['label']:a for a in p['cases']}; answers={a['label']:a for a in locked['cases']}
raw=(oracle/'selected-i64le.bin').read_bytes(); assert sha(raw)=='229ef28d7c1ef5c30ae0b6598b0d9ced519a59e4598d11459fb428de78fc4f8c'
def ports(c,a):
 s=a['sha256'];b=zlib.decompress((c/'owners'/(s+'.zlib')).read_bytes());assert len(b)==a['length'] and sha(b)==s
 return b
def nt(o,h,t,shape,strides,raw):
 d=o.node(h);s,v=d[27:29];dim=len(shape)
 assert d[:2]==(dim,math.prod(shape)) and d[12:12+dim]==tuple(shape) and d[16:16+dim]==tuple(strides)
 assert d[72]==t and d[29]==d[83]==0 and d[4:6]==(0,0)
 assert o.row(1,h)[1][2:6]==(s,v,0,len(raw)) and o.storage(s)[1:]==(len(raw),raw) and o.row(3,v)[1][2]==0
 return s,v,d
def verify(c):
 R=json.loads((c/'report.json').read_bytes());t=json.loads((c/'transport.json').read_bytes())
 assert R['no_torch'] and R['elf']==t['executable_sha256'] and t['exit_code']==0
 u=list(frames(c/'requests.frames.bin',b'HTOWNR01'));v=list(frames(c/'responses.frames.bin',b'HTOWNS01'))
 assert len(u)==len(v)==R['native_frames']
 Q=list(map(req,u)); S=list(map(resp,v)); positives=words=0; stats={}; seen=set(); zero=0; exact_ctrl={}; setup={}
 def reads(m,h,tcount,width,expected):
  i=m['meta_frame']-1;assert Q[i]==[(13,h,0,0,0,0,0,0)] and S[i][0]==0
  op,tok,packed=S[i][1][0]; assert (op,tok)==(13,h)
  md=q(packed,16); assert md[1]==tcount and md[13:]==(0,0,tcount*width)
  rawdata=bytearray();j=0
  for frame in m['chunks']:
   k=frame-1;op,ht,at,n,*pad=Q[k][0]
   assert (op,ht,at)==(98,h,j) and pad==[0]*4 and S[k][0]==0
   it=S[k][1][0]; assert it[:2]==(98,n) and len(it[2])==n*width
   j+=n;rawdata+=it[2]
  assert j==tcount and rawdata==expected
  return md
 for a in R['rows']:
  i=a['frame']-1; l=a.get('label'); before=ports(c,a['pre']);after=ports(c,a['post'])
  if a['kind']=='setup_only':
   c0,=Q[i];assert c0[:3]==(97,1,96) and S[i]==(3,[]) and before==after
   assert sha(u[i])==a['q_sha256'] and sha(u[i][160:])==fixed[l]['input_raw_sha256']
   assert q(u[i],2,96)==(1,0) and q(u[i],1,112)[0]==262145
   setup[l]=a['status'];seen.add(l);continue
  if a['kind']=='refusal':
   assert sha(u[i])==a['q_sha256'] and S[i]==(a['status'],[]) and before==after and a['pre']['ptr']==a['post']['ptr']
   stats[l]=a['status']
   if l in fixed:
    n=fixed[l]['input_count'];cmd,=Q[i];raw_in=Owner(before).storage(Owner(before).node(cmd[1])[27])[2]
    assert cmd[0]==135 and cmd[2:]==(0,)*6 and sha(raw_in)==fixed[l]['input_raw_sha256']
    nt(Owner(before),cmd[1],1,[n],[1],raw_in);seen.add(l)
   if l=='actual-offset-Bool-view':
    cmd,=Q[i];o=Owner(before);x=o.node(cmd[1]); assert x[72]==x[29]==x[83]==1
    exact_ctrl[l]=dict(byte_offset=1,view_flag=1)
   continue
  if a['kind']=='same-frame-seeded':
   assert q(before,4,128)==q(after,4,128)==(2,3,0,0)
   assert len(Q[i])==3 and [cc[0] for cc in Q[i]]==[135,13,98] and S[i][0]==0
   o=Owner(after);h=a['out_handle'];x=a['in_handle'];keys=struct.pack('<3q',0,3,5)
   assert nt(o,h,2,[3,1],[1,3],keys)[2][1]==3 and S[i][1][2][2]==keys
   exact_ctrl['same-frame-seeded']=True;continue
  assert a['kind']=='positive' and R['mode']=='new'
  F=fixed[l];e=answers[l];n=F['input_count'];k=F['expected_torch']['shape'][0];x,y=a['handles']; seen.add(l)
  prior=Owner(before); now=Owner(after); xn=prior.node(x); b=prior.storage(xn[27])[2];assert sha(b)==F['input_raw_sha256'] and len(b)==n
  expected=b''.join(struct.pack('<q',j) for j,bit in enumerate(b) if bit==1); assert set(b)<={0,1}
  at=e['offset'];assert expected==raw[at:at+e['length']] and sha(expected)==F['expected_torch']['raw_i64le_sha256']
  x0=nt(prior,x,1,[n],[1],b);assert nt(now,x,1,[n],[1],b)==x0
  y0=nt(now,y,2,[k,1],[1,k],expected);assert x!=y and x0[0]!=y0[0] and x0[1]!=y0[1]
  assert before[128:160]==after[128:160] and Q[i]==[(135,x,0,0,0,0,0,0)] and S[i]==(0,[(135,y,b'')])
  assert reads(a['reads_before'],x,n,1,b)==reads(a['reads_after'],x,n,1,b)
  meta=reads(a['output'],y,k,8,expected)
  assert meta[:5]==(2,k,0,k,1) and meta[7:9]==(1,k) and meta[11:13]==y0[:2]
  if a['order']=='input-first': assert reads(a['survive'],y,k,8,expected)==meta
  else: reads(a['survive'],x,n,1,b)
  wf=a['weak_frame']-1; w=S[wf][1][0][1]; assert Q[wf][0][:2]==(42,y)
  we=a['weak_expired']-1; assert Q[we][0][:2]==(43,w) and S[we]==(6,[]) and Q[we+1][0][:2]==(44,w)
  positives+=1;words+=k;zero+=not k
 if R['mode']=='old':
  assert seen=={l for l,f0 in fixed.items() if f0['hypothesized_candidate_status']==0}
  assert len(stats)==17 and set(stats.values())=={4}
  assert (c/'error.json').read_text().strip()=="AssertionError(('factory', 3, 1, [262145], 112))"
  assert Q[-1][0][:3]==(97,1,96) and S[-1]==(3,[])
 else:
  assert seen==set(fixed) and positives==17 and words==263307 and zero==4 and setup=={'overcap-K0':3,'overcap-K2':3}
  guard={g['label']:g['local_status'] for g in p['guards'] if not g.get('setup_only')}
  assert all(stats.get(k)==v for k,v in guard.items()) and len(stats)==26
 return dict(mode=R['mode'],frames=len(Q),positives=positives,words=words,zeros=zero,refusals=stats,controls=exact_ctrl,setup_only=setup)
print(json.dumps({k:verify(base/k) for k in ('old','new')},sort_keys=True,indent=2))
