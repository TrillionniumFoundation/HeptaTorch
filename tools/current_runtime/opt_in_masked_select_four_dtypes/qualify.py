from pathlib import Path
import hashlib,math,json,struct,sys,zlib
H,protocol,image,fixture,measured,out,mode,pinfile=map(Path,sys.argv[1:]);mode=mode.name
sha=lambda b:hashlib.sha256(b).hexdigest()
for n,x in [('test_runtime_candidate.py','40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'),('test_component.py','97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c')]:
 assert sha((H/n).read_bytes())==x
assert sha(fixture.read_bytes())=='f1be3964d390da1450e3a418e8e3dbd17fd3f1f6353cb63046b13d73133b409e'
assert sha((measured/'receipt.json').read_bytes())=='2eed544ec6db34343024bd8e983d0433798b18c7f7397b13871956349a8d6d7a'
official=(measured/'selected.bin').read_bytes()
assert sha(official)=='be7ce5fb0b7a2ed2d5218ee9eb3f70a88603dc6e2dd21ac831eaa3e00029579b'
rows=json.loads(fixture.read_bytes())['cases']
lock={q['label']:q for q in json.loads((measured/'receipt.json').read_bytes())['rows']}
sys.path.insert(0,str(H))
from test_runtime_candidate import factory,load_wire
assert mode in ('old','new')
pin=pinfile.read_text().strip()
if mode=='old': assert pin=='5130bc446381049134363953a687656e7bbfe9bfe6fb87c331383e3acaf27cd8'
assert sha(image.read_bytes())==pin and 'torch' not in sys.modules
W=load_wire(protocol);S=W.Session(image,directory=out);(out/'owners').mkdir()
ledger=[]
def run(*cmd):
 s,r=W.parse(S.exchange(W.packet(cmd)));assert s==0 and [p['opcode'] for p in r]==[c[0] for c in cmd],(s,[c[0] for c in cmd],S.frames)
 return r
def one(*cmd):return run(cmd)[0]['value']
def make(tag,shape,data):
 rawdata=[w&((1<<64)-1) for w in data] if tag==2 else data
 req=b''.join(factory(tag,shape,rawdata))
 if len(req)<=1<<20:
  st,r=W.parse(S.exchange(req));assert st==0 and len(r)==1 and r[0]['opcode']==97,('factory',st,tag,shape,S.frames)
  return r[0]['value']
 assert tag==4 and len(shape)==2 and len(data)>0
 rs,cs=shape; row=[data[i*cs] for i in range(rs)]; col=[floatword(13*j) for j in range(cs)]
 # Both component words and final full source are read from actual public native roots.
 x=make(4,[rs,1],row);y=make(4,[1,cs],col)
 xr,_=verify(x,rs,rawwords(row,8),[rs,1]);yr,_=verify(y,cs,rawwords(col,8),[1,cs])
 out_h=one(2,1,x,y); k=S.frames; drop(x,y)
 ledger.append(dict(kind='large_setup',handle=out_h,frame=k,row=xr,col=yr,shape=shape))
 return out_h
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
def floatword(i):return int.from_bytes(struct.pack('<d',i),'little')
def rawwords(w,sz):return b''.join(z.to_bytes(sz,'little') for z in w)
def expected(a):
 z=lock[a['label']];p=z['file_offset'];return official[p:p+z['byte_len']]
def inputs(a):
 n=a['value_count'];nm=a['mask_count']
 return ([word(i,a['value_shape'],a['source'],a['dtype']) for i in range(n)],bytes(mask(i,a['mask'],nm) for i in range(nm)))
def verify(h,n,x,dims):
 w,md=read(h,n,x)
 assert md['rank']==len(dims) and md['shape'][:len(dims)]==dims and md['version']==0 and md['requires_grad']==0 and md['storage_offset']==0,(dims,md)
 return w,md
def run_positive(ix,a,hx,hm,xx,mm,xb,mb,mx,mt):
 pre=snap();ho=one(*masksel(hx,hm));fr=S.frames;post=snap()
 k=a['expected']['words'];aout,meta=read(ho,k,expected(a))
 assert meta['rank']==1 and meta['shape'][0]==k and meta['strides'][0]==1 and meta['version']==0 and meta['requires_grad']==0 and meta['storage_offset']==0 and meta['storage_bytes']==a['byte_width']*k,(a['label'],meta)
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
  xd=a['value_shape'];md=a['mask_shape'];n=a['value_count'];nm=a['mask_count'];v,m=inputs(a);xx=b''.join(v);mm=bytes(m);assert sha(xx)==a['value_sha256'] and sha(mm)==a['mask_sha256']
  hx=make(a['dtype_tag'],xd,[int.from_bytes(w,'little') for w in v]);hm=make(1,md,m)
  xb,mx=verify(hx,n,xx,xd);mb,mt=verify(hm,nm,mm,md)
  status=a['old_status'] if mode=='old' else a['new_hypothesized_status']
  if status:refus(a['label'],status,masksel(hx,hm));drop(hx,hm)
  else:run_positive(i,a,hx,hm,xx,mm,xb,mb,mx,mt)
 if mode=='new':
  for g in json.loads(fixture.read_bytes())['guards']:
   label=g['label'];s=g['status'];xd=g.get('value_shape',[2,3]);md=g.get('mask_shape',[2,3])
   dtype=g.get('dtype','F64');mtype=g.get('mask','Bool');mapping={**T,'I64':(2,8,tuple(range(18)))}
   nt,w,pat=mapping[dtype]; mt,mw,mp=mapping[mtype];n=math.prod(xd);nm=math.prod(md)
   if label=='integer-mask-still-forbidden':md=[1,3];nm=3
   vals=[pat[i%len(pat)] for i in range(n)];bits=[mp[i%len(mp)] for i in range(nm)]
   x=make(nt,xd,vals);m=make(mt,md,bits);holds=[x,m]
   if label=='actual-offset-value-view':
    base=make(nt,[7],[pat[i] for i in range(7)]);holds+=[base]
    x=view(base,xd,[3,1],1);holds+=[x]
    assert W.meta(run((13,x))[0])['storage_offset']==w
   if label=='actual-offset-mask-view':
    base=make(mt,[7],[mp[i] for i in range(7)]);holds+=[base]
    m=view(base,md,[3,1],1);holds+=[m]
    assert W.meta(run((13,m))[0])['storage_offset']==mw
   if label=='actual-grad-value':one(90,x,1);assert W.meta(run((13,x))[0])['requires_grad']==1
   if label=='stale-capability':
    dead=make(1,[1],[1]);drop(dead);m=dead
   q=[masksel(x,m)]
   if label=='padding':q=[(130,x,m,1)]
   if label=='valid-then-unknown':q.append((65500,))
   refug=refus(label,s,*q);drop(*reversed(holds))
except BaseException as e:
 (out/'error.json').write_text(repr(e)+'\n');raise
finally:
 (out/'report.json').write_text(json.dumps(dict(mode=mode,elf=pin,native_frames=S.frames,rows=ledger,no_torch='torch' not in sys.modules),indent=2,sort_keys=True)+'\n')
 S.finish()
