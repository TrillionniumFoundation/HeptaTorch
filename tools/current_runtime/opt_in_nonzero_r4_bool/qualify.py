"""Ordinary valid CPU numerical qualification, no Torch import."""
from pathlib import Path
import hashlib,json,struct,sys,zlib
H,protocol,image,fixture,measured,out,mode,pinfile,scope=map(Path,sys.argv[1:]);mode=mode.name
assert scope.name in ("modest","cap");optional=scope.name=="cap"
H,protocol,image,fixture,measured,out=map(Path.resolve,(H,protocol,image,fixture,measured,out))
sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha((H/'test_runtime_candidate.py').read_bytes())=='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'
assert sha((H/'test_component.py').read_bytes())=='97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c'
assert sha(fixture.read_bytes())=='a79b5753ed6ca6f2972edd4629aafb299789eeae76e431e6f1c4aa8370dbd876'
assert sha((measured/'receipt.json').read_bytes())=='a2bdcb6ab3ecd336271fc79a9b07b5ab818853b50a4f558da0a17e447a1794c6'
raw=(measured/'logical-i64le.bin').read_bytes()
assert sha(raw)=='3f3cbd5e8b17421d908d83933867a6b6f5b6adf853e6b2b4ae11596b5a45ffed'
oracle=fixture.with_name('freeze.py').read_bytes()
assert sha(oracle)=='633609b6ae50de3bd11ecc08820dcd172319b100adb17b2dc5bda72e79593d2e'
lib={};exec(compile(oracle.split(b'\nZ=dict(',1)[0],'frozen-prereg-functions','exec'),lib)
cases=[x for x in json.loads(fixture.read_bytes())['cases'] if x['optional_in_cap_reference']==optional]
results={x['label']:x for x in json.loads((measured/'receipt.json').read_bytes())['cases']}
pin=pinfile.read_text().strip();assert sha(image.read_bytes())==pin and mode in ('old','new')
if mode=='old':assert pin=='db28aa0c8f2ca4f59ccf7a8d52eb30a324d8d1464fdd7ac7a3cf920df5353772'
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
def factory_bytes(shape,data):
 a,b,c,d=shape;assert len(data)==a*b*c*d and len(data)+160<=1<<20
 wargs=struct.pack('<8Q',97,1,96,0,0,0,0,0)
 cfg=struct.pack('<8Q',4,0,a,b,c,d,160,len(data))
 return b'HTORCH03'+struct.pack('<3Q',1,160+len(data),0)+wargs+cfg+data
events=[]
try:
 for idx,c in enumerate(cases):
  shape=c['input_shape'];n=c['input_count'];label=c['label']
  b=lib['source'](shape,c['recipe']);assert sha(b)==c['input_sha256']
  hx=exchange(factory_bytes(shape,b),97)['value'];setup=S.frames
  ir,im=read(hx,n,b);md(im,n,1,shape,c['input_strides'])
  before=snapshot();req=W.packet(((135,hx),))
  st,rr=W.parse(S.exchange(req));fr=S.frames
  after=snapshot();event=dict(label=label,frame=fr,input_handle=hx,input_read=ir,status=st,pre=before,post=after,setup_frame=setup)
  if mode=='old':
   assert st==7 and not rr and before==after,(label,st,fr)
   one(41,hx)
  else:
   assert st==0 and len(rr)==1 and rr[0]['opcode']==135,(label,st,fr)
   hy=rr[0]['value'];m=results[label];w=raw[m['logical_offset']:m['logical_offset']+m['bytes']];k=c['k']
   o,om=read(hy,4*k,w);md(om,4*k,8,[k,4],[1,k])
   assert hy!=hx and im['storage_id']!=om['storage_id'] and im['version_id']!=om['version_id']
   ia,am=read(hx,n,b);assert am==im
   if idx%2==0:
    one(41,hx);sr,sm=read(hy,4*k,w);assert sm==om;one(41,hy);order='input-first'
   else:
    one(41,hy);sr,sm=read(hx,n,b);assert sm==im;one(41,hx);order='output-first'
   event.update(output_handle=hy,output_read=o,input_reread=ia,survival_read=sr,order=order)
  events.append(event)
except BaseException as e:
 (out/'error.json').write_text(repr(e)+'\n');raise
finally:
 (out/'report.json').write_text(json.dumps(dict(elf=pin,mode=mode,frames=S.frames,rows=events,no_torch=True),sort_keys=True,indent=2)+'\n')
 S.finish()
assert len(events)==(4 if optional else 20)
print(mode,len(events),S.frames,flush=True)
