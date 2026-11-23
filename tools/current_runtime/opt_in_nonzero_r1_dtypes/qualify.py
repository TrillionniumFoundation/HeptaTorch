"""Ordinary valid CPU numerical qualification, no Torch import."""
from pathlib import Path
import hashlib,json,struct,sys,zlib
H,protocol,image,fixture,measured,out,mode,pinfile=map(Path,sys.argv[1:]);mode=mode.name
H,protocol,image,fixture,measured,out=map(Path.resolve,(H,protocol,image,fixture,measured,out))
sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha((H/'test_runtime_candidate.py').read_bytes())=='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'
assert sha((H/'test_component.py').read_bytes())=='97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c'
assert sha(fixture.read_bytes())=='e75921cb1b5366eaccbb4b5dc4eab5213bda2263c1ee53b4b77955b8e7716797'
assert sha((measured/'receipt.json').read_bytes())=='bc79206830c8a2030bc4be249f3b597f15929e6bffd795e5efb11358d0bd0545'
raw=(measured/'selected-i64le.bin').read_bytes()
assert sha(raw)=='f2fae0be8f205190a1832233159cd8b0c383762c6daea8257860a25202ddc97f'
oracle=fixture.with_name('freeze.py').read_bytes()
assert sha(oracle)=='95a7a36e15f4ae6f7fddac7d08e2d759f07b2875288e65668c60e51743c79c40'
lib={};exec(compile(oracle.split(b'\nraw=lambda ',1)[0],'frozen-prereg-functions','exec'),lib)
cases=[x for x in json.loads(fixture.read_bytes())['cases'] if x['admitted']]
results={x['label']:x for x in json.loads((measured/'receipt.json').read_bytes())['cases']}
pin=pinfile.read_text().strip();assert sha(image.read_bytes())==pin and mode in ('old','new')
if mode=='old':assert pin=='95aadc064a5a16a188988b699b81dda287443b0e7172fa163dde15db3857c6c7'
sys.path.insert(0,str(H))
from test_runtime_candidate import factory,load_wire
assert 'torch' not in sys.modules
W=load_wire(protocol);S=W.Session(image,directory=out);(out/'owners').mkdir()
def read(h,n,bs):
 row=one(13,h);f=S.frames;m=W.meta(row)
 data=bytearray();frames=[]
 for at in (range(0,n,32768) if n else (0,)):
  cnt=min(n-at,32768);v=one(98,h,at,cnt)
  assert v['value']==cnt,(S.frames,h,n,cnt,v['value'])
  data.extend(v['payload']);frames.append(S.frames)
 assert data==bs,(S.frames,h,len(data),sha(data),sha(bs))
 return dict(meta_frame=f,raw_frames=frames),m
def one(*cmd):return exchange(W.packet((cmd,)),cmd[0])
def exchange(q,op):
 st,rr=W.parse(S.exchange(q))
 assert st==0 and len(rr)==1 and rr[0]['opcode']==op, (st,op,S.frames)
 return rr[0]
def snapshot():
 x=S.owner();b=x['bytes'];digest=sha(b);p=out/'owners'/(digest+'.zlib')
 if not p.exists():p.write_bytes(zlib.compress(b,4))
 else:assert zlib.decompress(p.read_bytes())==b
 return dict(sha256=digest,bytes=len(b))
def md(m,n,w,shape,strides):
 assert m['rank']==len(shape) and m['numel']==n and m['shape'][:len(shape)]==shape and m['strides'][:len(shape)]==strides
 assert m['version']==m['storage_offset']==m['requires_grad']==0 and m['storage_bytes']==n*w
def factory_bytes(tag,n,data,w):
 # Literal raw op97 shape header + already generated frozen bytes.
 assert len(data)==n*w
 wargs=struct.pack('<8Q',97,tag,96,0,0,0,0,0)
 cfg=struct.pack('<8Q',1,0,n,0,0,0,160,len(data))
 return b'HTORCH03'+struct.pack('<3Q',1,160+len(data),0)+wargs+cfg+data
def make(tag,n,b,w,c):
 q=factory_bytes(tag,n,b,w)
 if len(q)<=1<<20:
  return exchange(q,97)['value'],[S.frames],[]
 # Existing public forced copies of EXACT representable 0/1. Not a shape trick.
 # Outside this fallback the IEEE raw patterns always use the literal factory.
 assert n==lib['CAP'] and tag in (0,2,4)
 half=lib['source']('float16',n,c['recipe'] if c['recipe']['kind']=='point' else
   dict(kind='uniform',word=0 if c['recipe']['word']==0 else 0x3c00))
 ha=exchange(factory_bytes(5,n,half,2),97)['value']; fs=[S.frames]; retire=[ha]
 _,hm=read(ha,n,half);md(hm,n,2,[n],[1])
 h=one(109,ha,0,0)['value'];fs.append(S.frames)
 assert h!=ha
 if tag!=0:
  f32=lib['source']('float32',n,c['recipe'] if c['recipe']['kind']=='point' else
    dict(kind='uniform',word=0 if c['recipe']['word']==0 else 0x3f800000))
  _,fm=read(h,n,f32);md(fm,n,4,[n],[1])
  assert fm['version_id']!=hm['version_id'] and fm['storage_id']!=hm['storage_id']
  retire.append(h)
  h=one(109,h,tag,0)['value'];fs.append(S.frames)
  assert h not in retire
 return h,fs,retire
events=[]
try:
 for idx,c in enumerate(cases):
  tag,w,*_=lib['TYPES'][c['input_dtype']]; n,=c['input_shape'];label=c['label']
  b=lib['source'](c['input_dtype'],n,c['recipe']);assert sha(b)==c['input_raw_sha256']
  hx,fs,retire=make(tag,n,b,w,c)
  ir,im=read(hx,n,b);md(im,n,w,[n],[1])
  for h in retire:one(41,h)
  before=snapshot(); req=W.packet(((135,hx),))
  st,rr=W.parse(S.exchange(req));fr=S.frames
  after=snapshot(); event=dict(label=label,frame=fr,input_handle=hx,input_read=ir,status=st,
      pre=before,post=after,setup_frames=fs,setup_literal=(len(fs)==1))
  if mode=='old' and tag!=1:
   assert st==7 and not rr and before==after,(label,st,fr)
   one(41,hx)
  else:
   assert st==0 and len(rr)==1 and rr[0]['opcode']==135,(label,st,fr)
   hy=rr[0]['value'];m=results[label]; want=raw[m['offset']:m['offset']+m['length']];k=c['expected_torch']['shape'][0]
   rr1,om=read(hy,k,want);md(om,k,8,[k,1],[1,k])
   assert hy!=hx and im['storage_id']!=om['storage_id'] and im['version_id']!=om['version_id']
   ia,am=read(hx,n,b);assert am==im
   if idx%2==0:
    one(41,hx);sr,sm=read(hy,k,want);assert sm==om;one(41,hy);order='input-first'
   else:
    one(41,hy);sr,sm=read(hx,n,b);assert sm==im;one(41,hx);order='output-first'
   event.update(output_handle=hy,output_read=rr1,input_reread=ia,survival_read=sr,order=order)
  events.append(event)
except BaseException as e:
 (out/'error.json').write_text(repr(e)+'\n');raise
finally:
 (out/'report.json').write_text(json.dumps(dict(elf=pin,mode=mode,frames=S.frames,rows=events,no_torch=True),sort_keys=True,indent=2)+'\n')
 S.finish()
assert len(events)==50
print(mode,len(events),S.frames,flush=True)
