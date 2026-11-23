"""Raw exact reclassification plus matched bilateral scratch witnesses."""
from pathlib import Path
import importlib.util,json,sys,zlib
old_general,old_trailing,old_scalar,retained,baseline,new,fixture=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('raw_where',Path(__file__).with_name('pure_full.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
js=lambda p:json.loads(p.read_bytes())
known={f['label']:f for f in js(fixture)['rows']}
sources={'general':(old_general,290,{'broadcast-values':'historical-general-broadcast-value','higher-mask-rank':'historical-higher-mask-rank'}),
 'trailing':(old_trailing,252,{'broadcast-values':'historical-trailing-broadcast-value'}),
 'scalar':(old_scalar,413,{'broadcast-values':'historical-scalar-broadcast-value'})}
def snapshot(root,z):
 b=zlib.decompress((root/'owners'/(z['sha256']+'.zlib')).read_bytes())
 assert r.sha(b)==z['sha256'] and len(b)==z['length'];return b
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
  row=known[labels[rec['label']]];seen.add(row['label'])
  assert r.resp(old_s[i])==(7,[])
  cmd,=r.req(old_q[i]); assert cmd[0]==129 and cmd[4:]==(0,0,0,0)
  pre=snapshot(old,rec['pre']);assert pre==snapshot(old,rec['post'])
  own=r.Owner(pre)
  for h,shape,field in zip(cmd[1:4],(row['mask_shape'],row['left_shape'],row['right_shape']),('mask_hex','left_hex','right_hex')):
   ds=own.node(h); assert ds[12:12+len(shape)]==tuple(shape) and ds[0]==len(shape)
   assert own.storage(ds[27])[2].hex()==row[field]
  proved.append(dict(frame=i+1,label=rec['label'],old_status=7,owner_sha256=rec['pre']['sha256'],successor_witness=row['label']))
 out[k]=dict(original_frames=total,retained_frames=len(new_q),other_frames_byte_identical=True,retired=proved)
r.inspect(new,fixture)
positive={e['label'] for e in js(new/'report.json')['rows'] if e['kind']=='positive'}
assert len(seen)==4 and seen<=positive
def scratch(path,kind):
 e,=[x for x in js(path/'report.json')['rows'] if x['kind']==kind]
 a=list(r.frames(path/'requests.frames.bin',b'HTOWNR01'))[e['arithmetic']-1]
 b=list(r.frames(path/'responses.frames.bin',b'HTOWNS01'))[e['arithmetic']-1]
 return a,b,snapshot(path,e['seed'])[128:160]
rq,rs,ws=scratch(baseline,'scratch-baseline');uq,us,uscratch=scratch(new,'scratch-matched')
assert rq==uq and rs==us and ws==uscratch and r.q(ws,2)==(2,3)
print(json.dumps(dict(original_refusal_frames=4,distinct_successor_witnesses=4,replays=out,
 arithmetic_seed_frame_exact=True,arithmetic_seed_qwords=list(r.q(ws,4))),sort_keys=True,indent=2))
