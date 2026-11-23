"""Replay current-main original wire, including all still unsupported masked op130 requests."""
from pathlib import Path
import json,sys
D=Path(__file__).parent;sys.path.insert(0,str(D))
from read_old import frames,req,resp,sha
H,W,ELF,OLD,OUT=map(Path,sys.argv[1:])
assert sha((H/'test_runtime_candidate.py').read_bytes())=='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'
sys.path.insert(0,str(H));from test_runtime_candidate import load_wire
W=load_wire(W);assert 'torch' not in sys.modules
p=json.loads((D/'physical-pins.json').read_bytes());assert sha(ELF.read_bytes())==p['candidate']
orig=json.loads((OLD/'transport.json').read_bytes());assert orig['executable_sha256']==p['baseline']=='ef82340209edc4dfb8d7ed0ecbc65e144fe26078cb29a0ba327798533476e86e'
qf=OLD/'requests.frames.bin';rf=OLD/'responses.frames.bin'
assert sha(qf.read_bytes())=='e072990b7a315f537aec4c35fcd9dbe771f2c0ac5fa9568095094eeadd1b3b3a'
assert sha(rf.read_bytes())=='e03bb7617c24ad7cde9a44f1843c4f91776c6f1a9517ee320ceb66d6a9f3de4a'
a=list(frames(qf,b'HTOWNR01'));b=list(frames(rf,b'HTOWNS01'));assert len(a)==len(b)==290
ss=W.Session(ELF,directory=OUT)
failures=0;masked=0
try:
 for i,(q,r) in enumerate(zip(a,b,strict=True),1):
  c,=req(q);assert c[0]!=131
  if c[0]==130:assert resp(r)==(4,[]);masked+=1
  s=resp(r)[0]
  if s: prior=ss.owner()
  assert ss.exchange(q)==r,(i,c[0],s,sha(q))
  if s: assert ss.owner()==prior;failures+=1
finally:ss.finish()
assert ss.frames==290 and masked==28
assert qf.read_bytes()==(OUT/qf.name).read_bytes() and rf.read_bytes()==(OUT/rf.name).read_bytes()
assert 'torch' not in sys.modules
out=dict(main_elf=orig['executable_sha256'],frame_count=ss.frames,all_original_frames_replayed=True,retained_unsupported_masked_opcode_130=masked,
 exact_owner_rollbacks=failures,request_sha256=sha(qf.read_bytes()),reply_sha256=sha(rf.read_bytes()),candidate_elf=p['candidate'])
(OUT/'replay.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
print(json.dumps(out))
