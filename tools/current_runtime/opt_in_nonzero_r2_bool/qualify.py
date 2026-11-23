"""Ordinary valid CPU numerical qualification, no Torch import."""
from pathlib import Path
import hashlib,json,struct,sys,zlib
H,protocol,image,fixture,measured,out,mode,pinfile=map(Path,sys.argv[1:]);mode=mode.name
H,protocol,image,fixture,measured,out=map(Path.resolve,(H,protocol,image,fixture,measured,out))
sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha((H/'test_runtime_candidate.py').read_bytes())=='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'
assert sha((H/'test_component.py').read_bytes())=='97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c'
assert sha(fixture.read_bytes())=='f7e77378482fc0ad9cf3896bd45d899b3cc04180f7a067debea1b6da0f4f08a9'
assert sha((measured/'receipt.json').read_bytes())=='ac0954acf0b7548e9fad5f5b717a655bb5f029fe885fc6adf72820be1c343ce5'
raw=(measured/'logical-i64le.bin').read_bytes()
assert sha(raw)=='5fbebc2a1eecd954fb530ba9a4a8d49873698dcbbba9ae372ec4a0d82ac938ad'
oracle=fixture.with_name('freeze.py').read_bytes()
assert sha(oracle)=='7eff0c4d7c3ad07157fc327bcf74205459a0cbba3e63e4e5d8a44605ba2884d6'
lib={};exec(compile(oracle.split(b'\nA=dict(',1)[0],'frozen-prereg-functions','exec'),lib)
cases=[x for x in json.loads(fixture.read_bytes())['cases'] if x['candidate_policy']=='admit']
results={x['label']:x for x in json.loads((measured/'receipt.json').read_bytes())['cases']}
pin=pinfile.read_text().strip();assert sha(image.read_bytes())==pin and mode in ('old','new')
if mode=='old':assert pin=='f952bf025e63f6545df424dff55e0d5a64cea196fc06a14d53ea76b5a37a9487'
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
 rows,cols=shape;assert len(data)==rows*cols and len(data)+160<=1<<20
 wargs=struct.pack('<8Q',97,1,96,0,0,0,0,0)
 cfg=struct.pack('<8Q',2,0,rows,cols,0,0,160,len(data))
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
   o,om=read(hy,2*k,w);md(om,2*k,8,[k,2],[1,k])
   assert hy!=hx and im['storage_id']!=om['storage_id'] and im['version_id']!=om['version_id']
   ia,am=read(hx,n,b);assert am==im
   if idx%2==0:
    one(41,hx);sr,sm=read(hy,2*k,w);assert sm==om;one(41,hy);order='input-first'
   else:
    one(41,hy);sr,sm=read(hx,n,b);assert sm==im;one(41,hx);order='output-first'
   event.update(output_handle=hy,output_read=o,input_reread=ia,survival_read=sr,order=order)
  events.append(event)
except BaseException as e:
 (out/'error.json').write_text(repr(e)+'\n');raise
finally:
 (out/'report.json').write_text(json.dumps(dict(elf=pin,mode=mode,frames=S.frames,rows=events,no_torch=True),sort_keys=True,indent=2)+'\n')
 S.finish()
assert len(events)==23
print(mode,len(events),S.frames,flush=True)
