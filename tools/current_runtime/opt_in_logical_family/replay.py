"""Reproduce all qualified AND and main frames; verify old op132 never dispatches."""
import json,sys
from pathlib import Path
D=Path(__file__).parent;sys.path.insert(0,str(D))
from read_old import frames,req,resp,sha
H,W,ELF,OLD,OUT=map(Path,sys.argv[1:])
assert sha((H/'test_runtime_candidate.py').read_bytes())=='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'
sys.path.insert(0,str(H));from test_runtime_candidate import load_wire
W=load_wire(W);assert 'torch' not in sys.modules
p=json.loads((D/'physical-pins.json').read_bytes());assert sha(ELF.read_bytes())==p['candidate']
orig=json.loads((D/'old-132.json').read_bytes());assert orig['op']==132 and orig['old_status']==4 and orig['frame']==2490
runs=[('and',2494,'db360c3e0b071ff3993d331d61276bbdf6437d3afecb2619b9a402e73080713f',
 '456fb911c1b01d4f780cb8c55aa032c420141754218a09685c552cafb3ae3feb','candidate',182,1),
 ('main',290,'e072990b7a315f537aec4c35fcd9dbe771f2c0ac5fa9568095094eeadd1b3b3a',
 'e03bb7617c24ad7cde9a44f1843c4f91776c6f1a9517ee320ceb66d6a9f3de4a','main-replay',28,0)]
summary={};OUT.mkdir()
for name,n,hq,hr,dst,expected_roll,expected_132 in runs:
 d=OLD/dst;qr=d/'requests.frames.bin';rp=d/'responses.frames.bin'
 assert sha(qr.read_bytes())==hq and sha(rp.read_bytes())==hr
 a=list(frames(qr,b'HTOWNR01'));b=list(frames(rp,b'HTOWNS01'));assert len(a)==len(b)==n
 out=OUT/name; ss=W.Session(ELF,directory=out);roll=neighbors=0
 try:
  for i,(q,r) in enumerate(zip(a,b,strict=True),1):
   c=req(q);row_ops=[j[0] for j in c]
   assert not any(k in (133,134) for k in row_ops)
   status=resp(r)[0]
   if status:prior=ss.owner()
   if 132 in row_ops:
    assert row_ops==[132] and name=='and' and i==orig['frame'] and status==4
    assert sha(q)==orig['sha_request'] and sha(r)==orig['sha_reply'] and sha(prior['bytes'])==orig['sha_owner']
    neighbors+=1
   assert ss.exchange(q)==r,(name,i,row_ops,status,sha(q))
   if status: assert ss.owner()==prior;roll+=1
 finally:ss.finish()
 assert roll==expected_roll and neighbors==expected_132 and ss.frames==n
 assert qr.read_bytes()==(out/qr.name).read_bytes() and rp.read_bytes()==(out/rp.name).read_bytes()
 summary[name]=dict(exact_frames=n,exact_owner_rollbacks=roll,request_sha256=hq,response_sha256=hr,prior_op132_status4_count=neighbors)
assert 'torch' not in sys.modules
(OUT/'receipt.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
print(json.dumps(summary))
