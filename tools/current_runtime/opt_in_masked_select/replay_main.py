"""Replay exact prior main setup bytes; independent no-Torch framing, no new data expected."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from pure import frames,req,resp,sha
import json
H,W,ELF,OLD,OUT=map(Path,sys.argv[1:])
assert sha((H/'test_runtime_candidate.py').read_bytes())=='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'
sys.path.insert(0,str(H));from test_runtime_candidate import load_wire
W=load_wire(W)
assert 'torch' not in sys.modules
old=json.loads((OLD/'transport.json').read_bytes());assert old['executable_sha256']=='ef82340209edc4dfb8d7ed0ecbc65e144fe26078cb29a0ba327798533476e86e'
qf=OLD/'requests.frames.bin';rf=OLD/'responses.frames.bin'
assert sha(qf.read_bytes())=='e072990b7a315f537aec4c35fcd9dbe771f2c0ac5fa9568095094eeadd1b3b3a'
assert sha(rf.read_bytes())=='e03bb7617c24ad7cde9a44f1843c4f91776c6f1a9517ee320ceb66d6a9f3de4a'
Q=list(frames(qf,b'HTOWNR01'));R=list(frames(rf,b'HTOWNS01'));assert len(Q)==len(R)==290
skip=[];comparable=[]
ss=W.Session(ELF,directory=OUT)
try:
 for i,(q,r) in enumerate(zip(Q,R,strict=True),1):
  cmd,=req(q)
  if cmd[0]==130:
   assert resp(r)==(4,[]);skip.append(i);continue
  assert ss.exchange(q)==r,('main comparable replay',i,cmd[0])
  comparable.append((q,r))
finally:ss.finish()
assert len(skip)==28 and ss.frames==262
assert list(frames(OUT/'requests.frames.bin',b'HTOWNR01'))==[q for q,r in comparable]
assert list(frames(OUT/'responses.frames.bin',b'HTOWNS01'))==[r for q,r in comparable]
out=dict(exact_main_sha256=old['executable_sha256'],predecessor_stream_frames=290,replayed_frames=262,skipped_own_new_operations=skip,
 requests_sha256=sha((OUT/'requests.frames.bin').read_bytes()),responses_sha256=sha((OUT/'responses.frames.bin').read_bytes()),candidate_native_sha256=sha(ELF.read_bytes()))
(OUT/'replay.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out))
