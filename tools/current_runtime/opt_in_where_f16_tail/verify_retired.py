"""Verify archived original raw refusals and exact stream removal only."""
from pathlib import Path
import importlib.util,json,sys,zlib
old_bf16,old_wide,old_bool_float,retained,new,fixture=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('raw_where',Path(__file__).with_name('pure_f16_tail.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
js=lambda p:json.loads(p.read_bytes())
known={f['label']:f for f in js(fixture)['rows']}
bf_labels={'f16-bf16-forward':'historical-6-f16-left','f16-bf16-reverse':'historical-6-f16-right'}
all_labels={**bf_labels,'f16-f32-forward':'historical-0-f16-left','f16-f32-reverse':'historical-0-f16-right'}
sources={'bf16_f32':(old_bf16,5206,all_labels),
 'wide':(old_wide,5206,bf_labels),'bool_float':(old_bool_float,1204,bf_labels)}
def snapshot(root,z):
 b=zlib.decompress((root/'owners'/(z['sha256']+'.zlib')).read_bytes())
 assert r.sha(b)==z['sha256'] and len(b)==z['length'];return b
seen=set(); out={}; total_removed=0
for k,(old,total,labels) in sources.items():
 gone={e['frame']-1:e for e in js(old/'report.json')['rows'] if e['label'] in labels}
 assert len(gone)==len(labels);total_removed+=len(gone)
 old_q=list(r.frames(old/'requests.frames.bin',b'HTOWNR01'))
 old_s=list(r.frames(old/'responses.frames.bin',b'HTOWNS01'))
 new_q=list(r.frames(retained/k/'requests.frames.bin',b'HTOWNR01'))
 new_s=list(r.frames(retained/k/'responses.frames.bin',b'HTOWNS01'))
 assert len(old_q)==len(old_s)==total
 assert new_q==[b for i,b in enumerate(old_q) if i not in gone]
 assert new_s==[b for i,b in enumerate(old_s) if i not in gone]
 proved=[]
 for i,rec in sorted(gone.items()):
  row=known[labels[rec['label']]];seen.add(row['label'])
  assert r.resp(old_s[i])==(7,[])
  cmd,=r.req(old_q[i]);assert cmd[0]==129 and cmd[4:]==(0,0,0,0)
  pre=snapshot(old,rec['pre']);assert pre==snapshot(old,rec['post'])
  own=r.Owner(pre)
  for h,shape,field,tag in zip(cmd[1:4],(row['mask_shape'],row['left_shape'],row['right_shape']),
     ('mask_hex','left_hex','right_hex'),(1,row['left_dtype'],row['right_dtype'])):
   ds=own.node(h);assert ds[0]==len(shape) and ds[12:12+len(shape)]==tuple(shape) and ds[72]==tag
   assert own.storage(ds[27])[2].hex()==row[field]
  proved.append(dict(frame=i+1,label=rec['label'],old_status=7,owner_sha256=rec['pre']['sha256'],successor_witness=row['label']))
 out[k]=dict(original_frames=total,retained_frames=len(new_q),other_frames_byte_identical=True,retired=proved)
r.inspect(new,fixture)
positive={e['label'] for e in js(new/'report.json')['rows'] if e['kind']=='positive'}
assert total_removed==8 and len(seen)==4 and seen<=positive
print(json.dumps(dict(original_refusal_frames=8,distinct_successor_witnesses=4,replays=out),sort_keys=True,indent=2))
