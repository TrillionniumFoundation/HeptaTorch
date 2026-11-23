"""Verify the six archived original refusals and byte-identical unaffected records."""
from pathlib import Path
import importlib.util,json,sys,zlib
old_trailing,old_scalar,old_where,retained,new,fixture=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('raw_where',Path(__file__).with_name('pure_general.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
js=lambda p:json.loads(p.read_bytes())
known={f['label']:f for f in js(fixture)['rows']}
map_labels={'rank1-count1-mask':'historical-rank1-one','rank2-count1-mask':'historical-rank2-one',
 'leading-singleton-mask':'historical-leading-one','broadcast-only':'historical-first-where-broadcast-only'}
sources={'trailing':(old_trailing,255,['rank1-count1-mask','rank2-count1-mask','leading-singleton-mask']),
 'scalar':(old_scalar,415,['rank1-count1-mask','rank2-count1-mask']),
 'where':(old_where,487,['broadcast-only'])}
seen=set(); out={}
for k,(old,total,labels) in sources.items():
 gone={e['frame']-1:e for e in js(old/'report.json')['rows'] if e['label'] in labels}
 assert len(gone)==len(labels)
 old_q=list(r.frames(old/'requests.frames.bin',b'HTOWNR01'))
 old_s=list(r.frames(old/'responses.frames.bin',b'HTOWNS01'))
 new_q=list(r.frames(retained/k/'requests.frames.bin',b'HTOWNR01'))
 new_s=list(r.frames(retained/k/'responses.frames.bin',b'HTOWNS01'))
 assert len(old_q)==len(old_s)==total
 assert new_q==[b for i,b in enumerate(old_q) if i not in gone]
 assert new_s==[b for i,b in enumerate(old_s) if i not in gone]
 proved=[]
 for i,rec in sorted(gone.items()):
  row=known[map_labels[rec['label']]];seen.add(row['label'])
  assert r.resp(old_s[i])==(7,[])
  cmd,=r.req(old_q[i]); assert cmd[0]==129 and cmd[4:]==(0,0,0,0)
  def snapshot(side):
   z=rec[side];b=zlib.decompress((old/'owners'/(z['sha256']+'.zlib')).read_bytes())
   assert r.sha(b)==z['sha256'] and len(b)==z['length'];return b
  pre=snapshot('pre');assert pre==snapshot('post')
  own=r.Owner(pre)
  for h,shape,field in zip(cmd[1:4],(row['mask_shape'],row['shape'],row['shape']),('mask_hex','left_hex','right_hex')):
   ds=own.node(h); assert ds[12:12+len(shape)]==tuple(shape)
   assert own.storage(ds[27])[2].hex()==row[field]
  proved.append(dict(frame=i+1,label=rec['label'],old_status=7,owner_sha256=rec['pre']['sha256'],successor_witness=row['label']))
 out[k]=dict(original_frames=total,retained_frames=len(new_q),other_frames_byte_identical=True,retired=proved)
r.inspect(new,fixture)
positive={e['label'] for e in js(new/'report.json')['rows'] if e['kind']=='positive'}
assert seen==set(map_labels.values()) and seen<=positive
print(json.dumps(dict(original_refusal_frames=6,distinct_successor_witnesses=4,replays=out),sort_keys=True,indent=2))
