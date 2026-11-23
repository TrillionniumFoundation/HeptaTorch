from pathlib import Path
import hashlib,math,json,struct,sys,zlib
H,protocol,image,fixture,measured,out,mode=map(Path,sys.argv[1:]);mode=mode.name
sha=lambda b:hashlib.sha256(b).hexdigest()
for n,x in [('test_runtime_candidate.py','40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'),('test_component.py','97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c')]:
 assert sha((H/n).read_bytes())==x
assert sha(fixture.read_bytes())=='6fdb5598642cc07dfa53571d4324b37e5b7ca27a400f4dacadd10dd419195c74'
assert sha((measured/'receipt.json').read_bytes())=='f792543d2d5f97dd840482c9369aeff4e4f5433abfe93257db8035463e3b4d7c'
official=(measured/'selected-u32le.bin').read_bytes()
assert sha(official)=='b879c03d455f7362128c78950858a3f7c220d8dbe51265b48745fe32463812cf'
rows=json.loads(fixture.read_bytes())['positive']
lock={q['label']:q for q in json.loads((measured/'receipt.json').read_bytes())['rows']}
sys.path.insert(0,str(H))
from test_runtime_candidate import factory,load_wire
assert mode in ('baseline','candidate','guards')
pin='ef82340209edc4dfb8d7ed0ecbc65e144fe26078cb29a0ba327798533476e86e' if mode=='baseline' else '75401b00356b8941960fae2a2207c0e046fb0da36cbcef7de26bef41ec4a3deb'
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
 z=lock[a['label']];p=z['raw_offset'];return official[p:p+z['raw_byte_count']]
def do_read(k,a,w):
 h=make(0,a['input_shape'],w); return h,words(a['count'],w)
try:
 for i,a in enumerate(rows if mode!='guards' else []):
  n=a['count'];K=a['expected']['numel']; dims=a['input_shape'];sp=a['recipe']; v,m=recipe(n,sp['source'],sp['mask'],sp['seed'])
  hx,xx=do_read(i,a,v); hm=make(1,dims,m); mm=bytes(m)
  xb,mx=read(hx,n,xx);mb,mm0=read(hm,n,mm)
  assert tuple(mx['shape'][:len(dims)])==tuple(mm0['shape'][:len(dims)])==tuple(dims)
  if mode=='baseline':
   refus(a['label'],4,masksel(hx,hm));drop(hx,hm);continue
  pre=snap();ho=one(*masksel(hx,hm));fr=S.frames;post=snap()
  aout,meta=read(ho,K,expected(a))
  assert meta['rank']==1 and meta['shape'][0]==K and meta['strides'][0]==1 and meta['version']==0 and meta['requires_grad']==0 and meta['storage_offset']==0 and meta['storage_bytes']==4*K,(a['label'],meta)
  assert ho not in (hx,hm) and meta['storage_id'] not in (mx['storage_id'],mm0['storage_id']) and meta['version_id'] not in (mx['version_id'],mm0['version_id'])
  xa,ma=read(hx,n,xx); ma2,mas=read(hm,n,mm);assert ma==mx and mas==mm0
  w=one(42,ho); wf=S.frames
  if i%2==0:
   drop(hm,hx); survive,_=read(ho,K,expected(a));drop(ho); order='inputs-first'
  else:
   drop(ho); survive,_=read(hx,n,xx);drop(hm,hx);order='result-first'
  weak=refus('weak-'+a['label'],6,(43,w));one(44,w)
  ledger.append(dict(kind='positive',label=a['label'],handles=[hx,hm,ho],reads_before=[xb,mb],reads_after=[xa,ma2],output=aout,
    frame=fr,pre=pre,post=post,weak_frame=wf,weak=weak,survive=survive,order=order))
 if mode!='baseline':
  x=make(0,[2,3],[*range(6)]);m=make(1,[2,3],[1,0,1,0,1,0]);want=words(3,[0,2,4])
  a=make(5,[2,1],[0x3c00,0x4000]);b=make(5,[1,3],[0x4000,0x4200,0x4400])
  half=one(2,3,a,b); seed_read,_=read(half,6,bytes.fromhex('004000420044004400460048'))
  before=snap(); assert struct.unpack_from('<4Q',raw_owner(before),128)==(2,3,0,0)
  replay=run(masksel(x,m),(13,W.reference(1)),(98,W.reference(1),0,3));sf=S.frames
  after=snap(); assert raw_owner(before)[128:160]==raw_owner(after)[128:160] and replay[2]['payload']==want
  ledger.append(dict(kind='seeded_same_frame',frame=sf,pre=before,post=after,input=[x,m],out=replay[0]['value'],seed_read=seed_read,result_hex=want.hex()))
  drop(replay[0]['value'],half,a,b)
  refus('suffix-rollback',4,masksel(x,m),(65500,))
  mm=make(1,[3,2],[1]*6);mb=make(1,[1,3],[1]*3);mi=make(2,[2,3],[1]*6)
  sx=make(0,[],[0]);sm=make(1,[],[1])
  refus('shape-mismatch',7,masksel(x,mm));refus('broadcast',7,masksel(x,mb))
  refus('dtype',7,masksel(x,mi));refus('rank0',7,masksel(sx,sm))
  xb=make(0,[7],[*range(7)]);xv=view(xb,[2,3],[3,1],1)
  assert W.meta(run((13,xv))[0])['storage_offset']==4
  refus('positive-view',7,masksel(xv,m))
  gx=make(0,[2,3],[*range(6)]);one(90,gx,1)
  assert W.meta(run((13,gx))[0])['requires_grad']==1
  refus('actual-grad',7,masksel(gx,m))
  dead=make(0,[1],[1]);drop(dead)
  refus('stale-x',1,masksel(dead,m));refus('stale-mask',1,masksel(x,dead))
  refus('padding',7,(130,x,m,1))
  capx=make(0,[513,512],recipe(262656,'ordinal','edge',0)[0])
  for lbl,sel in [('sparse','edge'),('zero','none')]:
   capm=make(1,[513,512],recipe(262656,'ordinal',sel,0)[1])
   cap_fo=refus('overcap-'+lbl,3,masksel(capx,capm));drop(capm)
  drop(capx,x,m,mm,mb,mi,sx,sm,xb,xv,gx)
except BaseException as e:
 (out/'error.json').write_text(repr(e)+'\n');raise
finally:
 (out/'report.json').write_text(json.dumps(dict(mode=mode,elf=pin,native_frames=S.frames,rows=ledger,no_torch='torch' not in sys.modules),indent=2,sort_keys=True)+'\n')
 S.finish()
