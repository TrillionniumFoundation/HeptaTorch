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
pin={'baseline':'c07d07fadfdfd095bb88620abddcf22fa7765ec501ed055b834073ad1d7735a0','candidate':'ad50b624a30c03cbc9b7c0d3ab71fff4e9edd7f7fce014de0e88f1f6282af90b'}[mode]
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
  f=next(z for z in rows if z['label']=='three-independently-determined-axes')
  m,x,y=make_row(f); n=math.prod(f['shape'])
  refusal('end-of-frame-unknown',4,where(m,x,y),(65500,))
  outrow,mr,rr=run(where(m,x,y),(13,W.reference(1)),(98,W.reference(1),0,n))
  assert rr['payload'].hex()==f['result_hex'] and W.meta(mr)['version']==0
  ledger.append(dict(kind='same-frame',label=f['label'],frame=S.frames,result=outrow['value'],inputs=[m,x,y]))
  drop(outrow['value'],m,x,y)
  c=make(1,[],[1]); vals={t:make(t,[2,4],list(range(8))) for t in (0,2,4,5,6)}
  for src,other in ((2,0),(0,2)):
   scalar=make(src,[],[0])
   refusal(f'rank0-{src}-left',7,where(c,scalar,vals[other]))
   refusal(f'rank0-{src}-right',7,where(c,vals[other],scalar))
   drop(scalar)
  scali64=make(2,[],[0]);scal64=make(0,[],[0])
  refusal('rank0-both-forward',7,where(c,scali64,scal64));refusal('rank0-both-reverse',7,where(c,scal64,scali64))
  for t,u,label in [(5,2,'f16-i64'),(6,2,'bf16-i64')]:
   refusal(label+'-forward',7,where(c,vals[t],vals[u]))
   refusal(label+'-reverse',7,where(c,vals[u],vals[t]))
  xbig=make(2,[513,1],[0]*513);ybig=make(0,[1,512],[0]*512)
  refusal('capacity-over-limit',3,where(c,xbig,ybig))
  xwrong=make(2,[4,2],list(range(8)));refusal('equal-count-incompatible-values',7,where(c,xwrong,vals[0]))
  refusal('numeric-scalar-condition',7,where(scali64,vals[2],vals[0]))
  gg=make(0,[2,4],list(range(8))); gst,en=packet((90,gg,1))
  if gst:
   assert gst==5 and not en
   ledger.append(dict(kind='setup-refused',label='f32-grad-mark-setup',frame=S.frames,status=gst,handle=gg))
  else:
   r,=run((13,gg));assert W.meta(r)['requires_grad']==1
   refusal('requires-grad-value',7,where(c,vals[2],gg))
  for kind,tag,other in [('i64',2,0),('f32',0,2)]:
   base=make(tag,[9],list(range(9)));v=view(base,[2,4],[4,1],1)
   r,=run((13,v));assert W.meta(r)['storage_offset']==(8 if tag==2 else 4)
   refusal('view-'+kind+'-left',7,where(c,v,vals[other]))
   refusal('view-'+kind+'-right',7,where(c,vals[other],v));drop(v,base)
  dead=make(2,[2,4],list(range(8)));drop(dead)
  refusal('stale-2',1,where(c,dead,vals[0]));refusal('stale-3',1,where(c,vals[0],dead))
  refusal('pad-4',7,(129,c,vals[2],vals[0],1))
  drop(c,xbig,ybig,xwrong,gg,scali64,scal64,*vals.values())
except BaseException as e:
 (out/'error.json').write_text(json.dumps({'error':repr(e)}))
 raise
finally:
 assert 'torch' not in sys.modules
 (out/'report.json').write_text(json.dumps(dict(mode=mode,elf=pin,fixture_sha256=sha(raw_fixture),
  rows=ledger,native_frames=S.frames,no_torch=True),sort_keys=True,indent=2)+'\n')
 S.finish()
