from pathlib import Path
import hashlib,math,json,struct,sys,zlib
H,protocol,image,fixture,measured,out,mode,pinfile=map(Path,sys.argv[1:]);mode=mode.name
sha=lambda b:hashlib.sha256(b).hexdigest()
for n,x in [('test_runtime_candidate.py','40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'),('test_component.py','97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c')]:
 assert sha((H/n).read_bytes())==x
assert sha(fixture.read_bytes())=='4f782478f42b9c3639ba3bfed921a8a932c8bb3dff2772096f00e312ae916f18'
assert sha((measured/'receipt.json').read_bytes())=='91e2341973c2e6552f82fa1fa522a8150f45479fc89f98d132ec7ce1a5079d5b'
official=(measured/'raw.bin').read_bytes()
assert sha(official)=='622feda4edf6e5510b7cdb05bc52f71c9584c149b12738e7713ef019f2f8972d'
rows=json.loads(fixture.read_bytes())['cases']
lock={q['label']:q for q in json.loads((measured/'receipt.json').read_bytes())['rows']}
sys.path.insert(0,str(H))
from test_runtime_candidate import factory,load_wire
assert mode in ('old','new')
pin=pinfile.read_text().strip()
if mode=='old': assert pin=='75401b00356b8941960fae2a2207c0e046fb0da36cbcef7de26bef41ec4a3deb'
assert sha(image.read_bytes())==pin and 'torch' not in sys.modules
W=load_wire(protocol);S=W.Session(image,directory=out);(out/'owners').mkdir()
ledger=[]
PAT=[0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x7f800000,0xff800000,0x7fc12345,0xffc23456,0x7f812345,0xff823456,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff]
def choose(i,n,m):
 return (m=='all' or m=='even' and i%2==0 or m=='odd' and i%2==1 or m=='edge' and i in (0,n-1) or m=='interior' and i%6 in (1,4,5) or m=='late' and i==n-1)
def recipe(n,s,m,seed):
 return ([PAT[(i+seed)%len(PAT)] if s=='special' else 0x3f000000+(13*i+seed)%(1<<23) for i in range(n)],
   [int(choose(i,n,m)) for i in range(n)])
def run(*cmd):
 s,r=W.parse(S.exchange(W.packet(cmd)));assert s==0 and [p['opcode'] for p in r]==[c[0] for c in cmd],(s,[c[0] for c in cmd],S.frames)
 return r
def one(*cmd):return run(cmd)[0]['value']
def make(tag,shape,words):
 req=b''.join(factory(tag,shape,words))
 if len(req)<=1<<20:
  s,r=W.parse(S.exchange(req));assert s==0 and len(r)==1 and r[0]['opcode']==97,(s,shape,S.frames)
  return r[0]['value']
 assert tag==0 and len(shape)==2
 R,C=shape
 seed=words[0]-0x3f000000
 assert all(w==0x3f000000+seed+13*i for i,w in enumerate(words))
 def fword(x):return struct.unpack('<I',struct.pack('<f',x))[0]
 row=[0x3f000000+seed+13*C*j for j in range(R)]
 col=[fword((13*j)/2**24) for j in range(C)]
 x=make(0,[R,1],row);y=make(0,[1,C],col)
 z=one(2,1,x,y)
 drop(x,y)
 return z
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
def masksel(x,m):return (130,x,m)
def words(n,data):return struct.pack(f'<{n}I',*data)
def expected(a):
 z=lock[a['label']];p=z['offset'];return official[p:p+z['length']]
def inputs(a):
 n=a['value_count']; kind=a['source']['kind']; seed=a['source']['seed']
 v=list(range(n)) if kind=='archive-exact' else recipe(n,kind,'none',seed)[0]
 if 'mask_formula' in a:
  m=bytearray(a['mask_count']);m[a['mask_formula']['index']]=1
 else: m=bytes.fromhex(a['mask_raw_u8_hex'])
 return v,m
def verify(h,n,x,dims):
 w,md=read(h,n,x)
 assert md['rank']==len(dims) and md['shape'][:len(dims)]==dims and md['version']==0 and md['requires_grad']==0 and md['storage_offset']==0,(dims,md)
 return w,md
def run_positive(ix,a,hx,hm,xx,mm,xb,mb,mx,mt):
 pre=snap();ho=one(*masksel(hx,hm));fr=S.frames;post=snap()
 k=a['mathematical_selected'];aout,meta=read(ho,k,expected(a))
 assert meta['rank']==1 and meta['shape'][0]==k and meta['strides'][0]==1 and meta['version']==0 and meta['requires_grad']==0 and meta['storage_offset']==0 and meta['storage_bytes']==4*k,(a['label'],meta)
 assert ho not in (hx,hm) and meta['storage_id'] not in (mx['storage_id'],mt['storage_id']) and meta['version_id'] not in (mx['version_id'],mt['version_id'])
 xa,ma=read(hx,a['value_count'],xx); m2,ma2=read(hm,a['mask_count'],mm);assert ma==mx and ma2==mt
 assert raw_owner(pre)[128:160]==raw_owner(post)[128:160]
 w=one(42,ho); wf=S.frames
 if ix%2==0:
  drop(hm,hx); survive,_=read(ho,k,expected(a));drop(ho); order='inputs-first'
 else:
  drop(ho); survive,_=read(hx,a['value_count'],xx);drop(hm,hx);order='result-first'
 weak=refus('weak-'+a['label'],6,(43,w));one(44,w)
 ledger.append(dict(kind='positive',label=a['label'],handles=[hx,hm,ho],reads_before=[xb,mb],reads_after=[xa,m2],output=aout,
   frame=fr,pre=pre,post=post,weak_frame=wf,weak=weak,survive=survive,order=order))
try:
 for i,a in enumerate(rows):
  xd=a['x_shape'];md=a['mask_shape'];n=a['value_count'];nm=a['mask_count'];v,m=inputs(a);xx=words(n,v);mm=bytes(m)
  hx=make(0,xd,v);hm=make(1,md,m)
  xb,mx=verify(hx,n,xx,xd);mb,mt=verify(hm,nm,mm,md)
  status=a['old_hypothesized_status'] if mode=='old' else a['new_hypothesized_status']
  if status:refus(a['label'],status,masksel(hx,hm));drop(hx,hm)
  else:run_positive(i,a,hx,hm,xx,mm,xb,mb,mx,mt)
 if mode=='new':
  for label,xs,ms in [('incompatible-zero-target',[2,0,4],[1,2,4]),('mask-zero-would-change-values',[2,1,4],[1,0,4]),('equal-count-transpose',[2,3],[3,2]),('symmetric-expansion-not-admitted',[2,1,4],[1,3,4]),('higher-mask-rank-not-admitted',[3,4],[1,3,4]),('scalar-value-not-admitted',[],[])]:
   x=make(0,xs,list(range(math.prod(xs))));m=make(1,ms,[1]*math.prod(ms))
   refus(label,7,masksel(x,m));drop(x,m)
  x=make(0,[2,3],list(range(6)))
  m=make(1,[1,3],[0,1,1]);mi=make(2,[1,3],[0,1,1]);refus('dtype-i64-mask',7,masksel(x,mi))
  gx=make(0,[2,3],list(range(6)));one(90,gx,1);assert W.meta(run((13,gx))[0])['requires_grad']==1
  refus('actual-grad-value',7,masksel(gx,m))
  mb=make(1,[4],[0,1,0,1]);mv=view(mb,[1,3],[3,1],1)
  assert W.meta(run((13,mv))[0])['storage_offset']==1;refus('actual-view-mask',7,masksel(x,mv))
  death=make(1,[],[1]);drop(death);refus('stale-mask',1,masksel(x,death));refus('nonzero-padding',7,(130,x,m,1))
  refus('valid-broadcast-then-unknown',4,masksel(x,m),(65500,))
  a=make(5,[2,1],[0x3c00,0x4000]);b=make(5,[1,3],[0x4000,0x4200,0x4400]);seeded=one(2,3,a,b)
  pre=snap(); assert struct.unpack_from('<4Q',raw_owner(pre),128)==(2,3,0,0)
  new=run(masksel(x,m),(13,W.reference(1)),(98,W.reference(1),0,4));fr=S.frames;post=snap()
  assert new[2]['payload']==words(4,[1,2,4,5]) and raw_owner(pre)[128:160]==raw_owner(post)[128:160]
  ledger.append(dict(kind='scratch',frame=fr,pre=pre,post=post,handles=[x,m,new[0]['value']]))
  drop(x,m,mi,gx,mv,mb,a,b,seeded,new[0]['value'])
except BaseException as e:
 (out/'error.json').write_text(repr(e)+'\n');raise
finally:
 (out/'report.json').write_text(json.dumps(dict(mode=mode,elf=pin,native_frames=S.frames,rows=ledger,no_torch='torch' not in sys.modules),indent=2,sort_keys=True)+'\n')
 S.finish()
