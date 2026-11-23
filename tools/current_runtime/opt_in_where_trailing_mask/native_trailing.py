"""Native-only observer for published baseline and candidate; separate from offline reader."""
from pathlib import Path
import hashlib,json,struct,sys,zlib,os,resource
H,protocol,image,fixture,out,mode=map(Path,sys.argv[1:]); mode=mode.name
def sha(b):return hashlib.sha256(b).hexdigest()
for n,d in [('test_runtime_candidate.py','40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'),('test_component.py','97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c')]:
 assert sha((H/n).read_bytes())==d
sys.path.insert(0,str(H))
from test_runtime_candidate import factory,load_wire
assert 'torch' not in sys.modules
pin={'baseline':'bee246ad3bd93c3128cedd5ecde4357599abcc032009055ce3e6a926147e0f6a','candidate':'6584192398a9abf7dfa886d0f9c820e90a441d6683b8ad1148284432204d6f14'}[mode]
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
 d=f['dtype'];s=f['shape'];m=make(1,f['mask_shape'],f['mask']);x=make(d,s,f['left']);y=x if f['same'] else make(d,s,f['right']); return m,x,y
def drop(*tids):return [run((41,t))[0] for t in dict.fromkeys(tids)]
def view(tid,shape,strides,offset=0):
 pad=[0]*(4-len(shape));return run((111,tid,W.Payload(struct.pack('<10Q',len(shape),*shape,*pad,*strides,*pad,offset)),80))[0]['value']
try:
 for i,f in enumerate(rows):
  n=len(f['left']); m,x,y=make_row(f)
  before_reads=[read(h,c,f[k])[0] for h,c,k in zip((m,x,y),(len(f['mask']),n,n),('mask_hex','left_hex','right_hex'))]
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
  after_reads=[read(h,c,f[k])[0] for h,c,k in zip((m,x,y),(len(f['mask']),n,n),('mask_hex','left_hex','right_hex'))]
  wk=run((42,result))[0]['value']; weak_frame=S.frames
  if i%2==0:
   drop(m,x,y); survivor,_=read(result,n,f['result_hex']);drop(result);order='inputs-first'
  else:
   drop(result);survivor,_=read(x,n,f['left_hex']);drop(m,x,y);order='result-first'
  expired=refusal('weak-expired-'+f['label'],6,(43,wk));run((44,wk))
  ledger.append(dict(kind='positive',label=f['label'],handles=[m,x,y,result],before_reads=before_reads,
    after_reads=after_reads,read_result=fo,call=call,pre=pre,post=post,order=order,survivor=survivor,
    weak=wk,weak_frame=weak_frame,expired=expired))
 if mode=='candidate':
  f=rows[0]; m,x,y=make_row(f); n=len(f['left']); d=f['dtype']; shape=f['shape']
  for i in range(4,8):
   c=list(where(m,x,y)) + [0]*4;c[i]=1
   refusal(f'pad-{i}',7,tuple(c))
  stale=make(d,shape,f['left']);drop(stale)
  for i in range(1,4):
   c=list(where(m,x,y));c[i]=stale
   refusal(f'stale-{i}',1,tuple(c))
  wrong=make(d,[4,3],f['left'])
  broad=make(d,[1,4],f['left'][:4])
  single=make(1,[1],[1]);single2=make(1,[1,1],[1]);leading=make(1,[1,4],[0,1,0,1]);bad_last=make(1,[3,2],[0,1]*3)
  mask_zero_mismatch=make(1,[2,1],[0,1]);empty_x=make(0,[3,0],[]);empty_y=make(0,[3,0],[])
  numeric=make(0,[],[0x80000000]);bad_type=make(2,shape,list(range(n)))
  refusal('numeric-scalar-condition',7,where(numeric,x,y))
  refusal('wrong-values-shape',7,where(m,wrong,y))
  refusal('broadcast-values',7,where(m,x,broad))
  refusal('mixed-value-dtype',7,where(m,x,bad_type))
  refusal('rank1-count1-mask',7,where(single,x,y))
  refusal('rank2-count1-mask',7,where(single2,x,y))
  refusal('leading-singleton-mask',7,where(leading,x,y))
  refusal('non-singleton-last',7,where(bad_last,x,y))
  refusal('zero-numel-wrong-prefix',7,where(mask_zero_mismatch,empty_x,empty_y))
  bv=make(1,[13],[0,1]*6+[0]); xv=make(0,[13],[0x3f800000]+f['left'])
  vm=view(bv,shape,[4,1],1); vx=view(xv,shape,[4,1],1)
  for name,hh,expected in [('mask',vm,1),('value',vx,4)]:
   mv,=run((13,hh));assert W.meta(mv)['storage_offset']==expected,(name,W.meta(mv))
  refusal('view-condition',7,where(vm,x,y));refusal('view-x',7,where(m,vx,y));refusal('view-y',7,where(m,x,vx))
  gg=make(d,shape,f['left']);run((90,gg,1));gmeta,_=run((13,gg),(98,gg,0,n));assert W.meta(gmeta)['requires_grad']==1
  refusal('requires-grad',7,where(m,gg,y))
  refusal('end-of-frame-unknown',4,where(m,x,y),(65500,))
  outrow, mr, rr=run(where(m,x,y),(13,W.reference(1)),(98,W.reference(1),0,n))
  assert rr['payload'].hex()==f['result_hex'];assert W.meta(mr)['version']==0
  ledger.append(dict(kind='same-frame',label=f['label'],frame=S.frames,result=outrow['value'],inputs=[m,x,y]))
  drop(outrow['value'],vm,vx,bv,xv,m,x,y,wrong,broad,bad_type,gg,single,single2,leading,bad_last,mask_zero_mismatch,empty_x,empty_y,numeric)
except BaseException as e:
 (out/'error.json').write_text(json.dumps({'error':repr(e)}))
 raise
finally:
 assert 'torch' not in sys.modules
 (out/'report.json').write_text(json.dumps(dict(mode=mode,elf=pin,fixture_sha256=sha(raw_fixture),
  rows=ledger,native_frames=S.frames,no_torch=True),sort_keys=True,indent=2)+'\n')
 S.finish()
