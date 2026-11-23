"""Keep exact four prior refusals; prove same historical words are now admitted."""
from pathlib import Path
import importlib.util,json,sys,zlib
old,retained,new,fixture=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('raw_where',Path(__file__).with_name('pure_bool_float.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
js=lambda p:json.loads(p.read_bytes())
known={f['label']:f for f in js(fixture)['rows']}
mapping={f'{name}-bool-{order}':f'historical-{tag}-{side}-left'
 for name,tag in [('bf16',6),('f64',4)] for order,side in [('forward','float'),('reverse','bool')]}
def snapshot(root,z):
 b=zlib.decompress((root/'owners'/(z['sha256']+'.zlib')).read_bytes())
 assert r.sha(b)==z['sha256'] and len(b)==z['length'];return b
gone={e['frame']-1:e for e in js(old/'report.json')['rows'] if e['label'] in mapping}
assert len(gone)==4
old_q=list(r.frames(old/'requests.frames.bin',b'HTOWNR01'))
old_s=list(r.frames(old/'responses.frames.bin',b'HTOWNS01'))
assert len(old_q)==len(old_s)==289
new_q=list(r.frames(retained/'requests.frames.bin',b'HTOWNR01'))
new_s=list(r.frames(retained/'responses.frames.bin',b'HTOWNS01'))
assert new_q==[b for i,b in enumerate(old_q) if i not in gone]
assert new_s==[b for i,b in enumerate(old_s) if i not in gone]
seen=set();out=[]
for i,rec in sorted(gone.items()):
 row=known[mapping[rec['label']]];seen.add(row['label'])
 assert r.resp(old_s[i])==(7,[])
 cmd,=r.req(old_q[i]);assert cmd[0]==129 and cmd[4:]==(0,0,0,0)
 pre=snapshot(old,rec['pre']);assert pre==snapshot(old,rec['post'])
 own=r.Owner(pre)
 for h,shape,field,tag in zip(cmd[1:4],(row['mask_shape'],row['left_shape'],row['right_shape']),
       ('mask_hex','left_hex','right_hex'),(1,row['left_dtype'],row['right_dtype'])):
  ds=own.node(h)
  assert ds[0]==len(shape) and ds[12:12+len(shape)]==tuple(shape) and ds[72]==tag
  assert own.storage(ds[27])[2].hex()==row[field]
 out.append(dict(frame=i+1,label=rec['label'],old_status=7,owner_sha256=rec['pre']['sha256'],successor_witness=row['label']))
r.inspect(new,fixture)
positive={e['label'] for e in js(new/'report.json')['rows'] if e['kind']=='positive'}
assert seen==set(mapping.values()) and seen<=positive
print(json.dumps(dict(original_refusal_frames=4,distinct_successor_witnesses=4,original_frames=289,retained_frames=285,
 other_frames_byte_identical=True,retired=out),sort_keys=True,indent=2))
