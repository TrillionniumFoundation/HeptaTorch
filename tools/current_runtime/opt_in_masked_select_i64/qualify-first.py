from pathlib import Path
import hashlib,math,json,struct,sys,zlib
H,protocol,image,fixture,measured,out,mode,pinfile=map(Path,sys.argv[1:]);mode=mode.name
sha=lambda b:hashlib.sha256(b).hexdigest()
for n,x in [('test_runtime_candidate.py','40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'),('test_component.py','97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c')]:
 assert sha((H/n).read_bytes())==x
assert sha(fixture.read_bytes())=='81aad2d489f77adbba131bf703456ca5e7fb04ad9876197376307bc128d3cc19'
assert sha((measured/'receipt.json').read_bytes())=='2532cf089c2ce7b4140c7fd0ce0a187de39d8f9472b3a7e64f08164427bd0800'
official=(measured/'selected-i64le.bin').read_bytes()
assert sha(official)=='160d36a596e3d04b8e21b96491a649013fdb4e354eaee1a8215f213fb81277db'
rows=json.loads(fixture.read_bytes())['cases']
lock={q['label']:q for q in json.loads((measured/'receipt.json').read_bytes())['rows']}
sys.path.insert(0,str(H))
from test_runtime_candidate import factory,load_wire
assert mode in ('old','new')
pin=pinfile.read_text().strip()
if mode=='old': assert pin=='458e9cffa53dbb77a4474a992775aa62b64dd6c25b314c0cfd0e1ab8fbb7494f'
assert sha(image.read_bytes())==pin and 'torch' not in sys.modules
W=load_wire(protocol);S=W.Session(image,directory=out);(out/'owners').mkdir()
ledger=[]
LOW=-(1<<63); HIGH=(1<<63)-1
PAT=(LOW,LOW+1,LOW+0xffffffff,LOW+(1<<40),-(1<<62),-(1<<53)-1,-(1<<32)-1,-(1<<32),-1,
 0,1,0x7fffffff,0x80000000,0xffffffff,1<<32,(1<<53)+1,(1<<62)+0x5a,0x5555555555555555,HIGH-1,HIGH)
def rowvalue(row,seed):
 q=row*8192+seed
 return (LOW+q,-((1<<60)+q),(1<<61)+q,HIGH-(1<<24)+q)[row%4]
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
 assert tag==2 and len(shape)==2 and len(data)>0
 R,C=shape; row=[data[i*C] for i in range(R)]; col=[13*j for j in range(C)]
 assert all(v==row[i//C]+col[i%C] for i,v in enumerate(data))
 x=make(2,[R,1],row);y=make(2,[1,C],col)
 xcheck,_=verify(x,R,words(R,row),[R,1]);ycheck,_=verify(y,C,words(C,col),[1,C])
 out_h=one(103,1,x,y); k=S.frames; drop(x,y)
 ledger.append(dict(kind='large_setup',handle=out_h,frame=k,row=xcheck,col=ycheck,shape=shape))
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
def words(n,data):return struct.pack(f'<{n}q',*data)
def expected(a):
 z=lock[a['label']];p=z['offset'];return official[p:p+z['length']]
def inputs(a):
 n=a['value_count']; kind=a['source']['kind']; seed=a['source']['seed']
 if kind=='signed_pattern':v=[PAT[(i+seed)%len(PAT)] for i in range(n)]
 else:
  assert kind=='i64_add_cartesian'
  nr,nc=a['value_shape'];v=[rowvalue(r,seed)+13*c for r in range(nr) for c in range(nc)]
 s=a['mask']; k=s['kind'];nm=a['mask_count']
 if k=='literal':m=bytes(s['bits'])
 elif k=='strided':m=bytes(int(i%s['period'] in s['remainders']) for i in range(nm))
 elif k=='edges':m=bytes(int(i in (0,nm-1)) for i in range(nm))
 else:
  assert k in ('all','none');m=bytes([int(k=='all')])*nm
 return v,m
def verify(h,n,x,dims):
 w,md=read(h,n,x)
 assert md['rank']==len(dims) and md['shape'][:len(dims)]==dims and md['version']==0 and md['requires_grad']==0 and md['storage_offset']==0,(dims,md)
 return w,md
def run_positive(ix,a,hx,hm,xx,mm,xb,mb,mx,mt):
 pre=snap();ho=one(*masksel(hx,hm));fr=S.frames;post=snap()
 k=a['expected']['metadata']['shape'][0];aout,meta=read(ho,k,expected(a))
 assert meta['rank']==1 and meta['shape'][0]==k and meta['strides'][0]==1 and meta['version']==0 and meta['requires_grad']==0 and meta['storage_offset']==0 and meta['storage_bytes']==8*k,(a['label'],meta)
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
  xd=a['value_shape'];md=a['mask_shape'];n=a['value_count'];nm=a['mask_count'];v,m=inputs(a);xx=words(n,v);mm=bytes(m);assert sha(xx)==a['pre_source_sha256'] and sha(mm)==a['pre_mask_sha256']
  hx=make(2,xd,v);hm=make(1,md,m)
  xb,mx=verify(hx,n,xx,xd);mb,mt=verify(hm,nm,mm,md)
  status=a['old_status'] if mode=='old' else a['new_hypothesized_status']
  if status:refus(a['label'],status,masksel(hx,hm));drop(hx,hm)
  else:run_positive(i,a,hx,hm,xx,mm,xb,mb,mx,mt)
  if mode=='new':
  for label,xs,ms in [('incompatible-empty',[2,0,4],[1,2,4]),('mask-zero-changes-value-shape',[2,1,4],[1,0,4]),('equal-count-transpose',[2,3],[3,2]),('symmetric-value-expansion',[2,1,4],[1,3,4]),('higher-rank-mask',[3,4],[1,3,4]),('rank0-value',[],[])]:
   x=make(2,xs,list(range(math.prod(xs))));m=make(1,ms,[1]*math.prod(ms))
   refus(label,7,masksel(x,m));drop(x,m)
  xv=make(2,[2,3],list(range(6)));m=make(1,[1,3],[1,0,1]);mi=make(2,[1,3],[1,0,1]);f=make(4,[2,3],[0]*6)
  refus('nonbool-value-used-as-mask',7,masksel(xv,mi))
  refus('other-value-F64-remains-unadmitted',7,masksel(f,m))
  vm=make(1,[4],[0,1,0,1]);mview=view(vm,[1,3],[3,1],1)
  assert W.meta(run((13,mview))[0])['storage_offset']==1;refus('actual-Bool-mask-offset-view',7,masksel(xv,mview))
  vv=make(2,[7],list(range(7)));vview=view(vv,[2,3],[3,1],1)
  assert W.meta(run((13,vview))[0])['storage_offset']==8;refus('actual-I64-value-offset-view',7,masksel(vview,m))
  dead=make(1,[1],[1]);drop(dead)
  refus('stale-input-or-mask',1,masksel(xv,dead))
  refus('nonzero-padding',7,(130,xv,m,1))
  refus('valid-I64-compaction-then-unknown',4,masksel(xv,m),(65500,))
  drop(xv,m,mi,f,vm,mview,vv,vview)
except BaseException as e:
 (out/'error.json').write_text(repr(e)+'\n');raise
finally:
 (out/'report.json').write_text(json.dumps(dict(mode=mode,elf=pin,native_frames=S.frames,rows=ledger,no_torch='torch' not in sys.modules),indent=2,sort_keys=True)+'\n')
 S.finish()
