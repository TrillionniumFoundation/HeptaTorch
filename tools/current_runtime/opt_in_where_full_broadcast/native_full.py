"""Native-only observer for published baseline and candidate; separate from offline reader."""
from pathlib import Path
import hashlib,math,json,struct,sys,zlib,os,resource
H,protocol,image,fixture,out,mode=map(Path,sys.argv[1:]); mode=mode.name
def sha(b):return hashlib.sha256(b).hexdigest()
for n,d in [('test_runtime_candidate.py','40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'),('test_component.py','97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c')]:
 assert sha((H/n).read_bytes())==d
sys.path.insert(0,str(H))
from test_runtime_candidate import factory,load_wire
assert 'torch' not in sys.modules
pin={'baseline':'112fa9623c8db21169ccb09dffe99e22fa40a44c80fc2419912edcf3c3be91a7','candidate':'2b9c240e764c081409a803dc7598a65317c623cf00e5a6981450c5a881300a6b'}[mode]
assert sha(image.read_bytes())==pin
W=load_wire(protocol); S=W.Session(image,directory=out);(out/'owners').mkdir()
raw_fixture=fixture.read_bytes(); rows=json.loads(raw_fixture)['rows']
ledger=[]
def packet(*cmd):return W.parse(S.exchange(W.packet(cmd)))
def run(*cmd):
 st,rr=packet(*cmd);assert not st and [r['opcode'] for r in rr]==[x[0] for x in cmd],(st,cmd,rr)
 return rr
def make(dtype,shape,words):
 st,rr=W.parse(S.exchange(b''.join(factory(dtype,shape,words))))
 assert not st and len(rr)==1 and rr[0]['opcode']==97,(st,dtype,shape)
 return rr[0]['value']
def snap():
 d=S.owner();raw=d['bytes']; h=sha(raw); p=out/'owners'/(h+'.zlib')
 if not p.exists():p.write_bytes(zlib.compress(raw,6))
 else:assert zlib.decompress(p.read_bytes())==raw
 return dict(sha256=h,length=len(raw),ptr=d['pointer'])
def refusal(name,expected,*cmd):
 req=W.packet(cmd); pre=snap(); st,ret=W.parse(S.exchange(req)); f=S.frames;post=snap()
 ledger.append(dict(kind='refusal',label=name,frame=f,pre=pre,post=post,expected=expected,status=st,request_sha256=sha(req)))
 assert st==expected and not ret and pre==post,(name,st,ret,expected,pre,post)
 return f
def read(tid,count,expected):
 m,r=run((13,tid),(98,tid,0,count));assert r['value']==count and r['payload'].hex()==expected,(r['payload'].hex(),expected)
 return S.frames,W.meta(m)
def where(m,x,y):return (129,m,x,y)
def make_row(f):
 d=f['dtype'];m=make(1,f['mask_shape'],f['mask']);x=make(d,f['left_shape'],f['left']);y=make(d,f['right_shape'],f['right']);return m,x,y
def drop(*tids):return [run((41,t))[0] for t in dict.fromkeys(tids)]
def view(tid,shape,strides,offset=0):
 pad=[0]*(4-len(shape));return run((111,tid,W.Payload(struct.pack('<10Q',len(shape),*shape,*pad,*strides,*pad,offset)),80))[0]['value']
try:
 for i,f in enumerate(rows):
  n=math.prod(f['shape']); m,x,y=make_row(f)
  before_reads=[read(h,c,f[k])[0] for h,c,k in zip((m,x,y),(len(f['mask']),len(f['left']),len(f['right'])),('mask_hex','left_hex','right_hex'))]
  if mode=='baseline':
   fno=refusal('original-'+f['label'],7,where(m,x,y))
   drop(m,x,y)
   continue
  pre=snap()
  result=run(where(m,x,y))[0]['value'];call=S.frames; post=snap()
  fo,meta=read(result,n,f['result_hex'])
  assert meta['rank']==len(f['shape']) and tuple(meta['shape'][:meta['rank']])==tuple(f['shape']) and meta['numel']==n
  assert meta['version']==0 and meta['requires_grad']==0 and meta['storage_offset']==0 and meta['storage_bytes']==n*f['width']
  assert tuple(meta['strides'][:meta['rank']])==tuple(f['strides']),(f['label'],meta)
  after_reads=[read(h,c,f[k])[0] for h,c,k in zip((m,x,y),(len(f['mask']),len(f['left']),len(f['right'])),('mask_hex','left_hex','right_hex'))]
  wk=run((42,result))[0]['value']; weak_frame=S.frames
  if i%2==0:
   drop(m,x,y); survivor,_=read(result,n,f['result_hex']);drop(result);order='inputs-first'
  else:
   drop(result);survivor,_=read(x,len(f['left']),f['left_hex']);drop(m,x,y);order='result-first'
  expired=refusal('weak-expired-'+f['label'],6,(43,wk));run((44,wk))
  ledger.append(dict(kind='positive',label=f['label'],handles=[m,x,y,result],before_reads=before_reads,
    after_reads=after_reads,read_result=fo,call=call,pre=pre,post=post,order=order,survivor=survivor,
    weak=wk,weak_frame=weak_frame,expired=expired))
 # Established typed half multiply physically writes the previous bshape workspace.
 a=make(5,[2,1],[0x3c00,0x4000]);b=make(5,[1,3],[0x4000,0x4200,0x4400])
 ar,=run((2,3,a,b)); arithmetic=S.frames
 assert tuple(read(ar['value'],6,'004000420044004400460048')[1]['shape'][:2])==(2,3)
 scratch_seed=snap()
 assert struct.unpack_from('<2Q',zlib.decompress((out/'owners'/(scratch_seed['sha256']+'.zlib')).read_bytes()),128)==(2,3)
 f=next(z for z in rows if z['label']=='three-distinct-axis-contributions')
 m,x,y=make_row(f); n=math.prod(f['shape'])
 if mode=='baseline':
  refusal('scratch-baseline-refusal',7,where(m,x,y))
  ledger.append(dict(kind='scratch-baseline',label=f['label'],arithmetic=arithmetic,seed=scratch_seed))
 else:
  prior=snap()
  zrow,=run(where(m,x,y));where_frame=S.frames; after=snap()
  pbytes=zlib.decompress((out/'owners'/(prior['sha256']+'.zlib')).read_bytes())
  abytes=zlib.decompress((out/'owners'/(after['sha256']+'.zlib')).read_bytes())
  seed_bytes=zlib.decompress((out/'owners'/(scratch_seed['sha256']+'.zlib')).read_bytes())
  assert pbytes[128:160]==abytes[128:160]==seed_bytes[128:160]
  nz,_=read(zrow['value'],n,f['result_hex'])
  ledger.append(dict(kind='scratch-matched',label=f['label'],arithmetic=arithmetic,seed=scratch_seed,pre=prior,post=after,
    frame=where_frame,out_handle=zrow['value'],read=nz))
  refusal('end-of-frame-unknown',4,where(m,x,y),(65500,))
  outrow, mr, rr=run(where(m,x,y),(13,W.reference(1)),(98,W.reference(1),0,n))
  assert rr['payload'].hex()==f['result_hex'];assert W.meta(mr)['version']==0
  ledger.append(dict(kind='same-frame',label=f['label'],frame=S.frames,result=outrow['value'],inputs=[m,x,y]))
  drop(zrow['value'],outrow['value'])
  bbig=make(1,[513,1],[0]*513);rbig=make(1,[1,512],[1]*512);bc=make(1,[],[1])
  refusal('capacity-over-limit',3,where(bc,bbig,rbig))
  em=make(1,[0,1],[]);ex=make(0,[0,3],[]);ey=make(0,[2,1],[1,1])
  refusal('zero-vs-nonsingleton',7,where(em,ex,ey))
  early_m=make(1,[0,0,1],[]);early_x=make(0,[0,0,3],[]);early_y=make(0,[1,2,1],[1,1])
  refusal('zero-conflict-after-compatible-empty-axis',7,where(early_m,early_x,early_y))
  drop(early_m,early_x,early_y)
  vx=make(0,[4,3],list(range(12)));vy=make(0,[3,4],list(range(12)))
  refusal('equal-count-incompatible-values',7,where(bc,vx,vy))
  gg=make(0,[3,4],list(range(12)));run((90,gg,1));gr,=run((13,gg));assert W.meta(gr)['requires_grad']==1
  refusal('requires-grad',7,where(bc,gg,vy))
  mixt=make(2,[1,4],list(range(4)));refusal('mixed-value-dtype',7,where(bc,vy,mixt))
  numm=make(0,[],[0]);refusal('numeric-scalar-condition',7,where(numm,vy,vy))
  bx=make(0,[13],[0x3f800000]+list(range(12)));xoff=view(bx,[3,4],[4,1],1)
  rr,=run((13,xoff));assert W.meta(rr)['storage_offset']==4
  refusal('view-x',7,where(bc,xoff,vy));refusal('view-y',7,where(bc,vy,xoff))
  dead=make(0,[],[0]);drop(dead)
  refusal('stale-2',1,where(bc,dead,vy));refusal('stale-3',1,where(bc,vy,dead))
  refusal('pad-4',7,(129,bc,vy,vy,1))
  drop(bbig,rbig,bc,em,ex,ey,vx,vy,gg,mixt,numm,bx,xoff)
 drop(m,x,y,a,b,ar['value'])
except BaseException as e:
 (out/'error.json').write_text(json.dumps({'error':repr(e)}))
 raise
finally:
 assert 'torch' not in sys.modules
 (out/'report.json').write_text(json.dumps(dict(mode=mode,elf=pin,fixture_sha256=sha(raw_fixture),
  rows=ledger,native_frames=S.frames,no_torch=True),sort_keys=True,indent=2)+'\n')
 S.finish()
