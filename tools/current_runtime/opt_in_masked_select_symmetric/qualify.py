from pathlib import Path
import hashlib,math,json,struct,sys,zlib
H,protocol,image,fixture,measured,out,mode,pinfile=map(Path,sys.argv[1:]);mode=mode.name
sha=lambda b:hashlib.sha256(b).hexdigest()
for n,x in [('test_runtime_candidate.py','40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'),('test_component.py','97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c')]:
 assert sha((H/n).read_bytes())==x
assert sha(fixture.read_bytes())=='d7b8af35d5288fd870c998e0564c3da7c6efcaf88c5723e438cd6e4999493cd1'
assert sha((measured/'receipt.json').read_bytes())=='c4d14601915388372b30c921b2022b8b669ced9368d90bc2cbf1d7c12566c448'
official=(measured/'selected.bin').read_bytes()
assert sha(official)=='0f7dbe529db92d62ebbcac9fa67d9c2875775d58b4493ce5ad6af36633416948'
rows=json.loads(fixture.read_bytes())['cases']
lock={q['label']:q for q in json.loads((measured/'receipt.json').read_bytes())['rows']}
sys.path.insert(0,str(H))
from test_runtime_candidate import factory,load_wire
assert mode in ('old','new')
pin=pinfile.read_text().strip()
if mode=='old': assert pin=='23ee4cc96d5f000c48dd128fbf65088a1b58b87dd72cd22add6cf0209144ad74'
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
def masksel(x,m):return (130,x,m)
CAP=262144
T={
'F64':(4,8,(0,0x8000000000000000,1,0x8000000000000001,0x000fffffffffffff,0x800fffffffffffff,0x0010000000000000,
 0x8010000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000001abc,0xfff8000000002abc,
 0x7ff0000000001357,0xfff0000000002468,0x3ff0000000000000,0xbff0000000000000,0x7fefffffffffffff,0xffefffffffffffff)),
'F16':(5,2,(0,0x8000,1,0x8001,0x03ff,0x83ff,0x0400,0x8400,0x7c00,0xfc00,0x7e35,0xfe57,0x7c35,0xfc57,0x3c00,0xbc00,0x7bff,0xfbff)),
'BF16':(6,2,(0,0x8000,1,0x8001,0x007f,0x807f,0x0080,0x8080,0x7f80,0xff80,0x7fc5,0xffd7,0x7f85,0xff97,0x3f80,0xbf80,0x7f7f,0xff7f)),
'Bool':(1,1,(0,1,1,0,0,1,0,1,1,1,0,0,1,0,1,0,0,1))}
T['F32']=(0,4,(0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x7f800000,0xff800000,
 0x7fc12345,0xffc23456,0x7f812345,0xff823456,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff))
T['I64']=(2,8,tuple(x&((1<<64)-1) for x in (-(1<<63),-(1<<63)+1,-(1<<62),-(1<<53)-1,-(1<<32)-1,-1,0,1,
 0x7fffffff,0x80000000,0xffffffff,1<<32,(1<<53)+1,(1<<62)+0x5a,0x5555555555555555,(1<<63)-2,(1<<63)-1)))
bytag={v[0]:d for d,v in T.items()}
def points(shape):
 return itertools.product(*(range(n) for n in shape))
def merge(*shapes):
 r=max(map(len,shapes));a,b=[(1,)*(r-len(s))+tuple(s) for s in shapes]
 out=[]
 for y,z in zip(a,b,strict=True): 
  if y==z or z==1:out.append(y)
  elif y==1:out.append(z)
  else:return None
 return tuple(out)
def key(c,shape):
 return tuple(0 if d==1 else k for k,d in zip(c[len(c)-len(shape):],shape,strict=True))
def words(d,shape,seed):
 _,w,p=T[d]; return b''.join(p[(i+seed)%len(p)].to_bytes(w,'little') for i in range(math.prod(shape)))
def flags(m,style):
 count=math.prod(m)
 if style=='all':return b'\1'*count
 if style=='none':return bytes(count)
 if style=='first':return bytes([1])+bytes(count-1)
 if style=='order':return bytes(int((i*7+2)%11 in (0,1,2,3,7,9)) for i in range(count))
 raise AssertionError(style)
def expected(a):
 z=lock[a['label']];p=z['at'];return official[p:p+z['length']]
def inputs(a):
 r=a['recipe']; dt=a['dtype'];w=a['width'];x=a['value_shape'];m=a['mask_shape']
 if 'operands' in r:b,c=map(bytes.fromhex,r['operands'])
 else:b=words(dt,x,r['seed']);c=flags(m,r['mask_style'])
 return [b[i:i+w] for i in range(0,len(b),w)],c
def verify(h,n,x,dims):
 w,md=read(h,n,x)
 assert md['rank']==len(dims) and md['shape'][:len(dims)]==dims and md['version']==0 and md['requires_grad']==0 and md['storage_offset']==0,(dims,md)
 return w,md
def run_positive(ix,a,hx,hm,xx,mm,xb,mb,mx,mt):
 pre=snap();ho=one(*masksel(hx,hm));fr=S.frames;post=snap()
 k=a['expected']['words'];aout,meta=read(ho,k,expected(a))
 assert meta['rank']==1 and meta['shape'][0]==k and meta['strides'][0]==1 and meta['version']==0 and meta['requires_grad']==0 and meta['storage_offset']==0 and meta['storage_bytes']==a['width']*k,(a['label'],meta)
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
  hx=make(a['tag'],xd,[int.from_bytes(w,'little') for w in v]);hm=make(1,md,m)
  xb,mx=verify(hx,n,xx,xd);mb,mt=verify(hm,nm,mm,md)
  status=a['old_hypothesized_status'] if mode=='old' else a['new_hypothesized_status']
  if status:refus(a['label'],status,masksel(hx,hm));drop(hx,hm)
  else:run_positive(i,a,hx,hm,xx,mm,xb,mb,mx,mt)
 if mode=='new':
  guards=json.loads(fixture.read_bytes())['guards']; box={}
  for g in guards:
   l=g['label']; xd=g.get('value',[2,3]);md=g.get('mask',[2,3])
   if l=='rank5-mask':
    base=bytearray(b''.join(factory(1,md[-4:],[1]*math.prod(md))))
    struct.pack_into('<Q',base,96,5);q=bytes(base)
    pre=snap();t,r=W.parse(S.exchange(q));post=snap()
    assert t and pre==post and not r
    ledger.append(dict(kind='setup_only',label=l,frame=S.frames,status=t,request=sha(q),pre=pre,post=post));continue
   if l=='private-tag3-value-stays-forbidden':
    q=b''.join(factory(3,[1],[0]));pre=snap();t,r=W.parse(S.exchange(q));post=snap()
    assert t and pre==post and not r
    ledger.append(dict(kind='setup_only',label=l,frame=S.frames,status=t,request=sha(q),pre=pre,post=post));continue
   if l=='existing-oversize-exact-or-oneway-still-cap':
    # The public Bool value factory can materialize 262656 bytes plus wire header without Cat or arithmetic.
    x=make(1,[513,512],[0]*262656);m=make(1,[513,512],[0]*262656);sm=make(1,[],[0])
    verify(x,262656,bytes(262656),[513,512]);verify(m,262656,bytes(262656),[513,512]);verify(sm,1,b'\0',[])
    refus(l+'-exact',3,masksel(x,m));refus(l+'-scalar-mask',3,masksel(x,sm));drop(x,m,sm);continue
   x=make(0,xd,[0x3f800000+i for i in range(math.prod(xd))]);m=make(1,md,[1]*math.prod(md));holds=[x,m]
   if l=='nonbool-mask-stays-forbidden':m=make(2,[2,3],[1]*6);holds+=[m]
   if l=='actual-view-mask':
    mb=make(1,[4],[0,1,0,1]);m=view(mb,[1,3],[3,1],1);holds+=[m,mb]
    assert W.meta(run((13,m))[0])['storage_offset']==1
   if l=='actual-grad-f32':one(90,x,1);assert W.meta(run((13,x))[0])['requires_grad']==1
   if l=='stale':
    dead=make(1,[1],[1]);drop(dead);m=dead
   q=[masksel(x,m)]
   if l=='padding':q=[(130,x,m,1)]
   if l=='compaction-then-unknown':
    nx=make(0,[2,1,4],[0x3f800000+i for i in range(8)]);nm=make(1,[1,3,1],[1,0,1]);holds+=[nx,nm]
    q=[masksel(nx,nm),(65500,)]
   refus(l,g['status'],*q);drop(*reversed(holds))
except BaseException as e:
 (out/'error.json').write_text(repr(e)+'\n');raise
finally:
 (out/'report.json').write_text(json.dumps(dict(mode=mode,elf=pin,native_frames=S.frames,rows=ledger,no_torch='torch' not in sys.modules),indent=2,sort_keys=True)+'\n')
 S.finish()
