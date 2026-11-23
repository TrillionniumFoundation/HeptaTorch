from pathlib import Path
import hashlib,math,json,struct,sys,zlib
H,protocol,image,fixture,measured,out,mode,pinfile=map(Path,sys.argv[1:]);mode=mode.name
sha=lambda b:hashlib.sha256(b).hexdigest()
for n,x in [('test_runtime_candidate.py','40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'),('test_component.py','97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c')]:
 assert sha((H/n).read_bytes())==x
assert sha(fixture.read_bytes())=='cf40ccecf20b5778aa5fe8a79ab5a3401dd9feb841b095e9563ffc62326a2946'
assert sha((measured/'receipt.json').read_bytes())=='907d07066493f982e3802a870d4a3bb8110ce85e44363611abfec3ab30ecb058'
official=(measured/'selected-i64le.bin').read_bytes()
assert sha(official)=='229ef28d7c1ef5c30ae0b6598b0d9ced519a59e4598d11459fb428de78fc4f8c'
rows=json.loads(fixture.read_bytes())['cases']
lock={q['label']:q for q in json.loads((measured/'receipt.json').read_bytes())['cases']}
sys.path.insert(0,str(H))
from test_runtime_candidate import factory,load_wire
assert mode in ('old','new')
pin=pinfile.read_text().strip()
if mode=='old': assert pin=='ef82340209edc4dfb8d7ed0ecbc65e144fe26078cb29a0ba327798533476e86e'
assert sha(image.read_bytes())==pin and 'torch' not in sys.modules
W=load_wire(protocol);S=W.Session(image,directory=out);(out/'owners').mkdir()
ledger=[]
def run(*cmd):
 s,r=W.parse(S.exchange(W.packet(cmd)));assert s==0 and [p['opcode'] for p in r]==[c[0] for c in cmd],(s,[c[0] for c in cmd],S.frames)
 return r
def one(*cmd):return run(cmd)[0]['value']
def make(tag,shape,data):
 req=b''.join(factory(tag,shape,data));assert len(req)<=1<<20
 st,r=W.parse(S.exchange(req));assert st==0 and len(r)==1 and r[0]['opcode']==97,('factory',st,tag,shape,S.frames)
 return r[0]['value']
def drop(*ids):
 for h in dict.fromkeys(ids):one(41,h)
def view(h,sizes,strides,off):
 p=[0]*(4-len(sizes));blob=struct.pack('<10Q',len(sizes),*sizes,*p,*strides,*p,off)
 return one(111,h,W.Payload(blob),len(blob))
def snap():
 q=S.owner();db=q['bytes']; h=sha(db); f=out/'owners'/(h+'.zlib')
 if not f.exists():f.write_bytes(zlib.compress(db,6))
 else:assert zlib.decompress(f.read_bytes())==db
 return dict(sha256=h,length=len(db),ptr=q['pointer'])
def raw_owner(x):return zlib.decompress((out/'owners'/(x['sha256']+'.zlib')).read_bytes())
def read(h,n,expected):
 row,=run((13,h)); m=W.meta(row); fs=S.frames; outp=b''; chunks=[]
 for a in (range(0,n,32768) if n else [0]):
  c=min(32768,n-a); rb,=run((98,h,a,c));assert rb['value']==c
  outp+=rb['payload'];chunks.append(S.frames)
 assert outp==expected,(h,n,sha(outp),sha(expected),S.frames)
 return dict(meta_frame=fs,chunks=chunks),m
def refus(l,status,*cmd):
 q=W.packet(cmd); a=snap(); s,r=W.parse(S.exchange(q));j=S.frames; z=snap()
 assert s==status and not r and a==z,(l,s,status,j,a,z)
 ledger.append(dict(kind='refusal',label=l,status=s,frame=j,pre=a,post=z,q_sha256=sha(q)))
 return j
def nz(x):return (135,x)
CAP=262144
def sha(b):return hashlib.sha256(b).hexdigest()
def source(n,r):
 k=r['kind']
 if k=='all':return b'\1'*n
 if k=='none':return bytes(n)
 if k=='points':
  hits=tuple(r['positions']);assert len(set(hits))==len(hits) and all(0<=i<n for i in hits)
  b=bytearray(n)
  for i in hits:b[i]=1
  return bytes(b)
 if k=='even':return bytes(1-i%2 for i in range(n))
 if k=='pattern':return bytes(int((17*i+3)%31 in (0,3,6,7,17,29)) for i in range(n))
 raise AssertionError(k)
def out_expected(a):
 o=lock[a['label']];at=o['offset'];return official[at:at+o['length']]
def equals(meta,n,width,shape,stride):
 assert meta['rank']==len(shape) and meta['numel']==n and meta['shape'][:len(shape)]==shape
 assert meta['strides'][:len(shape)]==stride and meta['version']==meta['storage_offset']==meta['requires_grad']==0
 assert meta['storage_bytes']==n*width
try:
 for i,a in enumerate(rows):
  n=a['input_count']; b=source(n,a['recipe']);assert sha(b)==a['input_raw_sha256']
  if a['hypothesized_candidate_status']!=3:continue
  if a['hypothesized_candidate_status']==3:
   q=b''.join(factory(1,[n],b));before=snap();fc,r=W.parse(S.exchange(q));frame=S.frames;after=snap()
   assert fc==3 and not r and before==after
   ledger.append(dict(kind='setup_only',label=a['label'],status=fc,frame=frame,pre=before,post=after,q_sha256=sha(q)))
   continue
  hx=make(1,[n],b);br,md=read(hx,n,b);equals(md,n,1,[n],[1])
  status=4 if mode=='old' else a['hypothesized_candidate_status']
  if status:
   refus(a['label'],status,nz(hx));drop(hx);continue
  before=snap(); out_h=one(*nz(hx));f=S.frames;after=snap()
  k=a['expected_torch']['shape'][0];out_bytes=out_expected(a)
  rr,om=read(out_h,k,out_bytes);equals(om,k,8,[k,1],[1,k])
  assert out_h!=hx and om['storage_id']!=md['storage_id'] and om['version_id']!=md['version_id']
  rb,checkm=read(hx,n,b);assert checkm==md
  assert raw_owner(before)[128:160]==raw_owner(after)[128:160]
  weak=one(42,out_h); wf=S.frames
  if i&1:
   drop(out_h); surv,_=read(hx,n,b);drop(hx);order='out-first'
  else:
   drop(hx);surv,_=read(out_h,k,out_bytes);drop(out_h);order='input-first'
  end=refus('weak-'+a['label'],6,(43,weak));one(44,weak)
  ledger.append(dict(kind='positive',label=a['label'],handles=[hx,out_h],pre=before,post=after,frame=f,
   reads_before=br,reads_after=rb,output=rr,weak_frame=wf,weak_expired=end,survive=surv,order=order))
 if mode=='new':
  for l,tag,dims,data in [('scalar-false',1,[],[0]),('scalar-true',1,[],[1]),('rank2-control',1,[2,3],[1,0,1,0,1,0]),
                           ('f32-truth-control',0,[5],[0,0x80000000,0x7fc12345,0xbf800000,1]),('i64-truth-control',2,[5],[0,1<<63,1,(1<<63)-1,0])]:
   x=make(tag,dims,data);refus(l,7,nz(x));drop(x)
  mb=make(1,[7],[0,1,0,1,0,1,1]);mv=view(mb,[5],[1],1)
  assert W.meta(run((13,mv))[0])['storage_offset']==1;refus('actual-offset-Bool-view',7,nz(mv));drop(mv,mb)
  dead=make(1,[1],[1]);drop(dead);refus('stale-source',1,nz(dead))
  x=make(1,[6],[1,0,0,1,0,1]);refus('padding',7,(135,x,1))
  refus('valid-nonzero-followed-by-unknown',4,nz(x),(65500,))
  a=make(5,[2,1],[0x3c00,0x4000]);b=make(5,[1,3],[0x4000,0x4200,0x4400]);seeded=one(2,3,a,b)
  before=snap();assert struct.unpack_from('<4Q',raw_owner(before),128)==(2,3,0,0)
  r=run(nz(x),(13,W.reference(1)),(98,W.reference(1),0,3));sf=S.frames;after=snap()
  equals(W.meta(r[1]),3,8,[3,1],[1,3])
  want=struct.pack('<3q',0,3,5);assert r[2]['payload']==want and raw_owner(before)[128:160]==raw_owner(after)[128:160]
  ledger.append(dict(kind='same-frame-seeded',frame=sf,pre=before,post=after,in_handle=x,out_handle=r[0]['value'],raw_sha256=sha(want)))
  drop(x,a,b,seeded,r[0]['value'])
except BaseException as e:
 (out/'error.json').write_text(repr(e)+'\n');raise
finally:
 (out/'report.json').write_text(json.dumps(dict(mode=mode,elf=pin,native_frames=S.frames,rows=ledger,no_torch='torch' not in sys.modules),indent=2,sort_keys=True)+'\n')
 S.finish()
