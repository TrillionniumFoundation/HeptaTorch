"""No-Torch physical replay of unchanged predecessor frames, checked by pure raw parsing."""
from pathlib import Path
import hashlib,importlib.util,json,os,resource,sys,zlib
sys.path.insert(0,str(Path(__file__).parent))
from pure_half_f64 import frames,req,resp,Owner,sha
H,protocol,elf,previous,fixture,out=map(Path,sys.argv[1:])
sys.path.insert(0,str(H))
from test_runtime_candidate import load_wire
W=load_wire(protocol)
assert 'torch' not in sys.modules
assert sha(elf.read_bytes())=='c07d07fadfdfd095bb88620abddcf22fa7765ec501ed055b834073ad1d7735a0'
known={r['label']:r for r in json.loads(fixture.read_bytes())['rows']}
wanted={f'f64-{k}-{order}':f'historical-{t}-{int(order=="reverse")}' for t,k in ((5,'f16'),(6,'bf16')) for order in ('forward','reverse')}
totals={'i64_f64':(5168,295,4),'f16_tail':(870,80,4),'bf16_f32':(5202,292,2),'wide':(5204,294,2),'bool_float':(1202,76,0),'bool_i64':(285,27,0),'full':(296,25,0),'general':(288,30,0),'trailing':(251,27,0),'scalar':(412,34,0),'where':(486,39,0),'layout':(1492,175,0),'bf16-ctor':(80,20,0),'f16-ctor':(80,20,0),'retained79':(876,82,0)}
summary={}
out.mkdir()
for run,(expected_n,expected_rollback,expected_retire) in totals.items():
 p=previous/'candidate' if run=='i64_f64' else previous/'retain'/run
 qs=list(frames(p/'requests.frames.bin',b'HTOWNR01'));rs=list(frames(p/'responses.frames.bin',b'HTOWNS01'));assert len(qs)==len(rs)==expected_n
 rows=json.loads((p/'report.json').read_bytes())
 labels={r['frame']-1:r for r in rows.get('rows',[]) if r.get('kind') in ('refusal','pressure')}
 # Original owner records for older layout/metadata experiments used case-specific schemas.
 # Independently derive the complete rejected frame set directly from the frozen wire.
 rejected={i for i,v in enumerate(rs) if resp(v)[0]}
 gone={i:e for i,e in labels.items() if i in rejected and e.get('label') in wanted}
 assert len(gone)==expected_retire
 s=W.Session(elf,directory=out/run)
 checks=0;cap_witness=[];retired=[];pressure=None;limits=None
 if run=='where':
  pressure,=([i,e] for i,e in labels.items() if e['kind']=='pressure')
 try:
  for i,(q,r) in enumerate(zip(qs,rs,strict=True)):
   rec=labels.get(i)
   if pressure and i==pressure[0]+1-pressure[1]['attempt']:
    limits=resource.prlimit(s.pid,resource.RLIMIT_AS)
    own0=s.owner(); page=os.sysconf('SC_PAGE_SIZE')
    vmem=int(Path(f'/proc/{s.pid}/statm').read_bytes().split()[0])*page
    cap=vmem+((own0['length']+16+page-1)//page)*page+65536
    recorded=pressure[1]['limit_bytes']
    print('pressure archive cap',recorded,'candidate-relative cap',cap,flush=True)
    assert abs(recorded-cap)<=page*8
    cap=recorded
    resource.prlimit(s.pid,resource.RLIMIT_AS,(cap,limits[1]))
    assert resource.prlimit(s.pid,resource.RLIMIT_AS)==(cap,limits[1])
   if pressure and i==pressure[0]+1:
    resource.prlimit(s.pid,resource.RLIMIT_AS,limits)
    assert resource.prlimit(s.pid,resource.RLIMIT_AS)==limits
    limits=None
   if i in gone:
    t=gone[i]; f=known[wanted[t['label']]];cmd,=req(q)
    old=zlib.decompress((p/'owners'/(t['pre']['sha256']+'.zlib')).read_bytes())
    post=zlib.decompress((p/'owners'/(t['post']['sha256']+'.zlib')).read_bytes())
    assert resp(r)==(7,[]) and cmd[0]==129 and cmd[4:]==(0,0,0,0)
    assert old==post and sha(old)==t['pre']['sha256']==t['post']['sha256'] and sha(q)==t['request_sha256']
    assert s.owner()['bytes']==old, (run,i+1,'historical setup')
    own=Owner(old)
    for h,tag,k in zip(cmd[1:4],(1,f['left_dtype'],f['right_dtype']),('mask','left','right'),strict=True):
     d=own.node(h);assert d[72]==tag and list(d[12:12+d[0]])==f[k+'_shape']
     assert own.storage(d[27])[2].hex()==f[k+'_hex']
    dest=out/run/'historical';dest.mkdir(exist_ok=True)
    prefix=dest/f'{i+1}'
    prefix.with_suffix('.request.bin').write_bytes(q);prefix.with_suffix('.response.bin').write_bytes(r)
    prefix.with_suffix('.owner.zlib').write_bytes(zlib.compress(old,6))
    retired.append(dict(frame=i+1,label=t['label'],positive=f['label'],status=7,owner_sha256=sha(old),request_sha256=sha(q),response_sha256=sha(r)))
    continue
   if i in rejected:before=s.owner()
   assert s.exchange(q)==r,(run,i+1,'wire difference')
   if i in rejected:
    after=s.owner();assert before==after,(run,i+1,'owner rollback')
    if rec and isinstance(rec.get('pre'),dict) and 'sha256' in rec['pre']:
     assert sha(before['bytes'])==rec['pre']['sha256']==rec['post']['sha256'],(run,i+1,'frozen owner')
    if rec and rec.get('label','').startswith('profile-cap-'):
     assert resp(r)==(7,[]);cap_witness.append(rec['label'])
    checks+=1
 finally:
  if limits is not None:resource.prlimit(s.pid,resource.RLIMIT_AS,limits)
  s.finish()
 assert s.frames==expected_n-expected_retire
 assert checks>=expected_rollback-expected_retire,(run,checks,expected_rollback)
 emit=out/run
 assert list(frames(emit/'requests.frames.bin',b'HTOWNR01'))==[q for i,q in enumerate(qs) if i not in gone]
 assert list(frames(emit/'responses.frames.bin',b'HTOWNS01'))==[r for i,r in enumerate(rs) if i not in gone]
 summary[run]=dict(previous_frames=expected_n,nonretired_frames=s.frames,prior_comparable_rollbacks=expected_rollback-expected_retire,all_observed_refusals_owner_unchanged=checks,
   request_sha256=sha((emit/'requests.frames.bin').read_bytes()),response_sha256=sha((emit/'responses.frames.bin').read_bytes()),retired=retired,cap_witness=cap_witness)
 print(run,s.frames,checks,flush=True)
assert 'torch' not in sys.modules
assert len(summary['f16_tail']['cap_witness'])==8
(out/'replay.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
