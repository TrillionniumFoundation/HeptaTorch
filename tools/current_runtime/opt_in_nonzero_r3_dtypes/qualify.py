"""Ordinary valid CPU numerical qualification, no Torch import."""
from pathlib import Path
import hashlib,json,struct,sys,zlib
H,protocol,image,fixture,measured,out,mode,pinfile=map(Path,sys.argv[1:]);mode=mode.name
H,protocol,image,fixture,measured,out=map(Path.resolve,(H,protocol,image,fixture,measured,out))
sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha((H/'test_runtime_candidate.py').read_bytes())=='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'
assert sha((H/'test_component.py').read_bytes())=='97b9c476a0a0f501c72c0dd2540a69ba41d5cd000fd4e3bf8d10a651ccc9e12c'
assert sha(fixture.read_bytes())=='dcba5ad5292fab78bb6d9a93d36602dffaf504fbef5ec28eeb9a6d61e4166fb2'
assert sha((measured/'receipt.json').read_bytes())=='46c2e1a8a8d0d6808c314a99c038397462f921e860fec6bc9c8e5a3dca8f1ab4'
raw=(measured/'logical-i64le.bin').read_bytes()
assert sha(raw)=='984faf2a202c102fb33736ae3959cd7c1379a131c44c237a91f2f46a20bcafb0'
oracle=fixture.with_name('freeze.py').read_bytes()
assert sha(oracle)=='3922308c46ad9f850bd858eecd00979c1e5f49498927aa41ebf9f899d192dd59'
lib={};exec(compile(oracle.split(b'\ncases=[]',1)[0],'frozen-prereg-functions','exec'),lib)
cases=[x for x in json.loads(fixture.read_bytes())['cases']]
results={x['label']:x for x in json.loads((measured/'receipt.json').read_bytes())['cases']}
pin=pinfile.read_text().strip();assert sha(image.read_bytes())==pin and mode in ('old','new')
if mode=='old':assert pin=='7ebdbbae563577e908e01c8375a6047770db12781eaebbf5255a35fcbe8f1605'
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
def factory_bytes(tag,shape,data,width):
 planes,rows,cols=shape;assert len(data)==planes*rows*cols*width and len(data)+160<=1<<20
 wargs=struct.pack('<8Q',97,tag,96,0,0,0,0,0)
 cfg=struct.pack('<8Q',3,0,planes,rows,cols,0,160,len(data))
 return b'HTORCH03'+struct.pack('<3Q',1,160+len(data),0)+wargs+cfg+data
events=[]
try:
 for idx,c in enumerate(cases):
  shape=c['input_shape'];n=c['input_count'];label=c['label']
  dt=c['input_dtype'];tag=c['input_tag'];width=lib['SPECS'][dt][1]//8
  b=lib['source'](dt,shape,c['recipe']);assert sha(b)==c['input_raw_sha256']
  hx=exchange(factory_bytes(tag,shape,b,width),97)['value'];setup=S.frames
  ir,im=read(hx,n,b);md(im,n,width,shape,c['input_strides'])
  before=snapshot();req=W.packet(((135,hx),))
  st,rr=W.parse(S.exchange(req));fr=S.frames
  after=snapshot();event=dict(label=label,frame=fr,input_handle=hx,input_read=ir,status=st,pre=before,post=after,setup_frame=setup)
  if mode=='old' and tag!=1:
   assert st==7 and not rr and before==after,(label,st,fr)
   one(41,hx)
  else:
   assert st==0 and len(rr)==1 and rr[0]['opcode']==135,(label,st,fr)
   hy=rr[0]['value'];m=results[label];w=raw[m['logical_offset']:m['logical_offset']+m['bytes']];k=c['k']
   o,om=read(hy,3*k,w);md(om,3*k,8,[k,3],[1,k])
   assert hy!=hx and im['storage_id']!=om['storage_id'] and im['version_id']!=om['version_id']
   ia,am=read(hx,n,b);assert am==im
   if idx%2==0:
    one(41,hx);sr,sm=read(hy,3*k,w);assert sm==om;one(41,hy);order='input-first'
   else:
    one(41,hy);sr,sm=read(hx,n,b);assert sm==im;one(41,hx);order='output-first'
   event.update(output_handle=hy,output_read=o,input_reread=ia,survival_read=sr,order=order)
  events.append(event)
except BaseException as e:
 (out/'error.json').write_text(repr(e)+'\n');raise
finally:
 (out/'report.json').write_text(json.dumps(dict(elf=pin,mode=mode,frames=S.frames,rows=events,no_torch=True),sort_keys=True,indent=2)+'\n')
 S.finish()
assert len(events)==63
print(mode,len(events),S.frames,flush=True)
