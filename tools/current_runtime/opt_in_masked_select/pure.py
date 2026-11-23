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
ELF_MAIN='ef82340209edc4dfb8d7ed0ecbc65e144fe26078cb29a0ba327798533476e86e'
FIX='6fdb5598642cc07dfa53571d4324b37e5b7ca27a400f4dacadd10dd419195c74'
ORACLE='f792543d2d5f97dd840482c9369aeff4e4f5433abfe93257db8035463e3b4d7c'
F32=(0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,
 0x7f800000,0xff800000,0x7fc12345,0xffc23456,0x7f812345,0xff823456,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff)
def compact(a):
 shape=a['input_shape']; cfg=a['recipe']; seed=cfg['seed']; n=math.prod(shape); kind=cfg['mask']; parts=[]; candidates=[]; flags=[]
 for i,xyz in enumerate(itertools.product(*(range(k) for k in shape))):
  j=0
  for dim,k in zip(shape,xyz,strict=True):j=j*dim+k
  assert j==i
  w=F32[(i+seed)%len(F32)] if cfg['source']=='special' else 0x3f000000+(i*13+seed)%(2**23)
  raw=struct.pack('<I',w)
  phase,rem=divmod(i,6)
  keep={'all':True,'none':False,'even':i&1==0,'odd':bool(i&1),'edge':i in {0,n-1},
   'interior':rem in {1,4,5},'late':i==n-1}[kind]
  candidates.append(raw);flags.append(int(keep))
  if keep:parts.append(raw)
 result=b''.join(parts)
 assert sha(result)==a['expected']['raw_u32le_sha256'] and len(result)==4*a['expected']['numel']
 return b''.join(candidates),bytes(flags),result
def inspect(p,fixture,measured,cand_pin):
 lr=json.loads((p/'report.json').read_bytes()); assert lr['no_torch']
 halted=p/'error.json'; assert not halted.exists() or (lr['mode']=='candidate' and halted.read_text().strip()=="AssertionError((7, [111], 696))")
 ledger=lr['rows']; mode=lr['mode']; pin=ELF_MAIN if mode=='baseline' else cand_pin
 transport=json.loads((p/'transport.json').read_bytes());assert lr['elf']==pin and transport['executable_sha256']==pin and transport['exit_code']==0
 raw_q=list(frames(p/'requests.frames.bin',b'HTOWNR01')); Q=list(map(req,raw_q)); R=list(map(resp,frames(p/'responses.frames.bin',b'HTOWNS01')))
 assert len(Q)==len(R)==lr['native_frames']==transport['frames']
 if halted.exists():assert len(Q)==696 and Q[-1][0][0]==111 and Q[-1][0][3]==0 and R[-1]==(7,[])
 official=(measured/'selected-u32le.bin').read_bytes()
 assert sha(official)=='b879c03d455f7362128c78950858a3f7c220d8dbe51265b48745fe32463812cf'
 lock=(measured/'receipt.json').read_bytes();assert sha(lock)==ORACLE
 O={x['label']:x for x in json.loads(lock)['rows']}
 fl=fixture.read_bytes();assert sha(fl)==FIX
 F={a['label']:a for a in json.loads(fl)['positive']}
 def snapshot(e):
  v=zlib.decompress((p/'owners'/(e['sha256']+'.zlib')).read_bytes());assert len(v)==e['length'] and sha(v)==e['sha256']
  return v
 def node(b,h,t,shape,data):
  owner=Owner(b);e=owner.node(h);k=owner.row(1,h)[1];ss,vs=e[27:29]
  assert e[72]==t and e[4:6]==(0,0) and e[29]==e[83]==0
  assert e[0]==len(shape) and e[1]==math.prod(shape) and e[12:12+e[0]]==tuple(shape)
  assert k[2:4]==(ss,vs) and k[4:6]==(0,len(data))
  base=owner.row(2,ss)[1][2]
  assert e[2]==(1<<63)+base and owner.storage(ss)[1:]==(len(data),data)
  return e,ss,vs,owner.row(3,vs)[1][2]
 def metadata(f,h,expected,width):
  j=f['meta_frame']-1;assert Q[j]==[(13,h,0,0,0,0,0,0)] and R[j][0]==0
  rec,=R[j][1]; md=q(rec[2],16); assert rec[:2]==(13,h)
  cs=[];n=len(expected)//width;jj=0
  for j0 in f['chunks']:
   j=j0-1;c=min(32768,n-jj);assert Q[j]==[(98,h,jj,c,0,0,0,0)]
   s,rr=R[j];a,=rr; assert not s and a[:2]==(98,c) and len(a[2])==c*width
   cs.append(a[2]);jj+=c
  assert jj==n and b''.join(cs)==expected and md[1]==n
  return md
 cases=words=rejects=0;  zero_cases=0; exact_caps=0
 seen=set(); controls={}
 for rec in ledger:
  k=rec['kind'];l=rec.get('label');j=rec['frame']-1
  if k=='refusal':
   status,records=R[j];assert (status,records)==(rec['status'],[]) and sha(raw_q[j])==rec['q_sha256']
   pre=snapshot(rec['pre']);post=snapshot(rec['post']);assert pre==post and rec['pre']['ptr']==rec['post']['ptr'];rejects+=1
   if mode=='baseline':
    assert status==4 and l in F and l not in seen;seen.add(l)
    f=F[l]; sx,sm,sout=compact(f);cmd,=Q[j];assert cmd[0]==130 and cmd[3:]==(0,)*5
    node(pre,cmd[1],0,f['input_shape'],sx);node(pre,cmd[2],1,f['input_shape'],sm)
   else:
    controls[l]=status
    if l.startswith('overcap-'):
     cmd,=Q[j]; assert cmd[0]==130 and status==3
     n=262656;shape=[513,512]; sx=b''.join(struct.pack('<I',0x3f000000+13*i) for i in range(n))
     sm=(b'\x01'+b'\x00'*(n-2)+b'\x01') if l.endswith('sparse') else b'\x00'*n
     node(pre,cmd[1],0,shape,sx);node(pre,cmd[2],1,shape,sm)
    if l=='suffix-rollback': assert [x[0] for x in Q[j]]==[130,65500] and status==4
    if l in ('rank0','actual-grad','positive-view'):
     cmd,=Q[j];z=Owner(pre).node(cmd[1]);assert cmd[0]==130 and status==7
     assert {'rank0':z[0]==0,'actual-grad':z[4]==1,'positive-view':z[29]==4 and z[83]==1}[l]
   continue
  if k=='seeded_same_frame':
   p0=snapshot(rec['pre']);p1=snapshot(rec['post'])
   assert q(p0,4,128)==q(p1,4,128)==(2,3,0,0)
   assert [x[0] for x in Q[j]]==[130,13,98] and R[j][1][2][2].hex()==rec['result_hex']
   assert q(R[j][1][1][2],16)[13]==0
   continue
  assert k=='positive' and l in F and l not in seen;seen.add(l)
  f=F[l]; s=f['input_shape'];sx,sm,sout=compact(f);K=len(sout)//4
  ol=O[l]; a=ol['raw_offset'];assert official[a:a+ol['raw_byte_count']]==sout
  assert sha(sout)==ol['raw_sha256'];hx,hm,ho=rec['handles'];assert Q[j]==[(130,hx,hm,0,0,0,0,0)] and R[j]==(0,[(130,ho,b'')])
  pre=snapshot(rec['pre']);post=snapshot(rec['post'])
  in_before=[node(pre,hx,0,s,sx),node(pre,hm,1,s,sm)]
  in_after=[node(post,hx,0,s,sx),node(post,hm,1,s,sm)];assert in_before==in_after
  assert metadata(rec['reads_before'][0],hx,sx,4)==metadata(rec['reads_after'][0],hx,sx,4)
  assert metadata(rec['reads_before'][1],hm,sm,1)==metadata(rec['reads_after'][1],hm,sm,1)
  out,st,ver,verval=node(post,ho,0,[K],sout);assert out[16]==1 and verval==0
  assert st not in {x[1] for x in in_after} and ver not in {x[2] for x in in_after} and ho not in (hx,hm)
  assert metadata(rec['output'],ho,sout,4)[:8]==(1,K,0,K,0,0,0,1)
  i=rec['weak_frame']-1; assert Q[i]==[(42,ho,0,0,0,0,0,0)]
  ws,wr=R[i]; (wu,w,empty),=wr; assert ws==0 and wu==42 and not empty
  wk,=[e for e in ledger if e['kind']=='refusal' and e['label']=='weak-'+l]
  i=wk['frame']-1; assert rec['weak']==wk['frame']
  assert Q[i]==[(43,w,0,0,0,0,0,0)] and R[i]==(6,[])
  assert Q[i+1]==[(44,w,0,0,0,0,0,0)] and R[i+1][0]==0
  if rec['order']=='inputs-first':metadata(rec['survive'],ho,sout,4)
  else:metadata(rec['survive'],hx,sx,4)
  cases+=1;words+=K;zero_cases+=K==0;exact_caps+=f['count']==262144
 assert len(seen)==(0 if mode=='guards' else len(F))
 if mode=='guards':
  assert controls=={'suffix-rollback':4,'shape-mismatch':7,'broadcast':7,'dtype':7,'rank0':7,'positive-view':7,'actual-grad':7,
    'stale-x':1,'stale-mask':1,'padding':7,'overcap-sparse':3,'overcap-zero':3}
 if mode=='candidate':assert (cases,words,zero_cases,exact_caps)==(28,364415,6,2)
 return dict(cases=cases,words=words,zeros=zero_cases,exact_caps=exact_caps,rollbacks=rejects,frames=len(Q),controls=controls,
  requests_sha256=sha((p/'requests.frames.bin').read_bytes()),responses_sha256=sha((p/'responses.frames.bin').read_bytes()))
if __name__=='__main__':
 f,o,pin,*captures=sys.argv[1:]
 print(json.dumps({Path(p).name:inspect(Path(p),Path(f),Path(o),pin) for p in captures},sort_keys=True,indent=2))
