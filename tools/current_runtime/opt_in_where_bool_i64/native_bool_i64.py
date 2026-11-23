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
pin={'baseline':'2b9c240e764c081409a803dc7598a65317c623cf00e5a6981450c5a881300a6b','candidate':'7737917d5fff95de844231a7c0c422c96b7834dbfa2e8862bb3ab4b67a37072d'}[mode]
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
 d=f['dtype'];m=make(1,f['mask_shape'],f['mask']);x=make(f['left_dtype'],f['left_shape'],f['left']);y=make(f['right_dtype'],f['right_shape'],f['right']);return m,x,y
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
 if mode=='candidate':
  f=next(z for z in rows if z['label']=='three-shape-bool-left')
  m,x,y=make_row(f); n=math.prod(f['shape'])
  refusal('end-of-frame-unknown',4,where(m,x,y),(65500,))
  outrow,mr,rr=run(where(m,x,y),(13,W.reference(1)),(98,W.reference(1),0,n))
  assert rr['payload'].hex()==f['result_hex'] and W.meta(mr)['version']==0
  ledger.append(dict(kind='same-frame',label=f['label'],frame=S.frames,result=outrow['value'],inputs=[m,x,y]))
  drop(outrow['value'],m,x,y)
  c=make(1,[],[1]);one=make(1,[2,4],[0,1]*4);two=make(2,[2,4],[2**63,2**63-1,2**64-1,2**53+1]*2)
  mixdenials=[(0,two,'f32-i64'),(5,two,'f16-i64'),(6,one,'bf16-bool'),(4,one,'f64-bool')]
  for tag,rhs,label in mixdenials:
   extra=make(tag,[2,4],list(range(8)))
   refusal(label+'-forward',7,where(c,extra,rhs))
   refusal(label+'-reverse',7,where(c,rhs,extra));drop(extra)
  xbig=make(1,[513,1],[0]*513);ybig=make(2,[1,512],[2**63-1]*512)
  refusal('capacity-over-limit',3,where(c,xbig,ybig))
  xwrong=make(1,[4,2],[0,1]*4)
  refusal('equal-count-incompatible-values',7,where(c,xwrong,two))
  numeric=make(2,[],[1]);refusal('numeric-scalar-condition',7,where(numeric,one,two))
  br=make(1,[9],[1]+[0,1]*4);ir=make(2,[9],[0]+[2**63,2**63-1,2**64-1,2**53+1]*2)
  bv=view(br,[2,4],[4,1],1);iv=view(ir,[2,4],[4,1],1)
  for kind,h,other,offset in [('bool',bv,two,1),('i64',iv,one,8)]:
   r,=run((13,h));assert W.meta(r)['storage_offset']==offset
   refusal('view-'+kind+'-left',7,where(c,h,other))
   refusal('view-'+kind+'-right',7,where(c,other,h))
  dead=make(2,[],[0]);drop(dead)
  refusal('stale-2',1,where(c,dead,one));refusal('stale-3',1,where(c,one,dead))
  refusal('pad-4',7,(129,c,one,two,1))
  drop(c,one,two,xbig,ybig,xwrong,numeric,bv,iv,br,ir)
except BaseException as e:
 (out/'error.json').write_text(json.dumps({'error':repr(e)}))
 raise
finally:
 assert 'torch' not in sys.modules
 (out/'report.json').write_text(json.dumps(dict(mode=mode,elf=pin,fixture_sha256=sha(raw_fixture),
  rows=ledger,native_frames=S.frames,no_torch=True),sort_keys=True,indent=2)+'\n')
 S.finish()
