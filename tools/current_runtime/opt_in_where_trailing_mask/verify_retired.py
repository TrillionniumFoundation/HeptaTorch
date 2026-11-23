"""Read original scalar refusal unchanged and exactly one excluded successor replay frame."""
from pathlib import Path
import hashlib,importlib.util,json,sys,zlib
old,retained,new,fixture=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('raw_where',Path(__file__).with_name('pure_trailing.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
js=lambda p:json.loads(p.read_bytes())
rec,=[x for x in js(old/'report.json')['rows'] if x['label']=='trailing-singleton-mask']
i=rec['frame']-1
witness=next(x for x in js(fixture)['rows'] if x['label']=='retired-scalar-control-f32')
for name,tag in (('requests',b'HTOWNR01'),('responses',b'HTOWNS01')):
 a=list(r.frames(old/(name+'.frames.bin'),tag));b=list(r.frames(retained/(name+'.frames.bin'),tag))
 assert b==a[:i]+a[i+1:] and len(a)==416
 if name=='responses':assert r.resp(a[i])==(7,[])
 else:
  (cmd,)=r.req(a[i]);assert cmd[0]==129 and cmd[4:]==(0,0,0,0)
  def snap(d,side):
   record=d[side]
   raw=zlib.decompress((old/'owners'/(record['sha256']+'.zlib')).read_bytes())
   assert r.sha(raw)==record['sha256'] and len(raw)==record['length'];return raw
  pre=snap(rec,'pre');assert pre==snap(rec,'post')
  ow=r.Owner(pre)
  for h,shape,field in zip(cmd[1:4],(witness['mask_shape'],witness['shape'],witness['shape']),('mask_hex','left_hex','right_hex')):
   ds=ow.node(h);assert ds[12:12+len(shape)]==tuple(shape)
   assert ow.storage(ds[27])[2].hex()==witness[field]
r.inspect(new,fixture)
assert len([v for v in js(new/'report.json')['rows'] if v['kind']=='positive' and v['label']==witness['label']])==1
print(json.dumps(dict(original_scalar_frames=416,retained_scalar_frames=415,retired_exact_status=7,
 original_refusal_frame=rec['frame'],original_owner_sha256=rec['pre']['sha256'],other_frames_byte_identical=True,
 successor_coordinate_witness=witness['label']),indent=2))
