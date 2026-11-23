"""Exact historical raw replay, with only nine independently measured owner-bound retirements."""
import json,sys,zlib
from pathlib import Path
D=Path(__file__).parent;sys.path.insert(0,str(D))
from read_old import frames,req,resp,sha,Owner
H,W,ELF,AND,FAM,OUT=map(Path,sys.argv[1:])
assert sha((H/'test_runtime_candidate.py').read_bytes())=='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'
sys.path.insert(0,str(H));from test_runtime_candidate import load_wire
W=load_wire(W);assert 'torch' not in sys.modules
p=json.loads((D/'physical-pins.json').read_bytes());assert sha(ELF.read_bytes())==p['candidate']
s=(D/'suite.json').read_bytes();assert sha(s)==p['suite'];fixture=json.loads(s)
precheck=json.loads((D/'pure.json').read_bytes())
assert precheck['candidate']['archival_positive']==9 and precheck['baseline']['prior_status7']==405
old=fixture['old'];known={(x['origin'],x['request_sha256'],x['response_sha256']):x for x in old}
prior132=json.loads((FAM/'old-132.json').read_bytes())
runs=[
 ('and',AND/'candidate',2494,'db360c3e0b071ff3993d331d61276bbdf6437d3afecb2619b9a402e73080713f',
  '456fb911c1b01d4f780cb8c55aa032c420141754218a09685c552cafb3ae3feb',182,3,1,1),
 ('or-xor',FAM/'candidate',4990,'59cad47e38954c9b85ddef249f77f6505a51533264e1f9337dc756689e456e2d',
  'bf0f296efc2097f913c9831ad6dc42ed37672be7d93e60c37c59494b66bf9f68',366,6,2,2),
 ('main',AND/'main-replay',290,'e072990b7a315f537aec4c35fcd9dbe771f2c0ac5fa9568095094eeadd1b3b3a',
  'e03bb7617c24ad7cde9a44f1843c4f91776c6f1a9517ee320ceb66d6a9f3de4a',28,0,0,0)]
expected130={'and':1,'or-xor':2,'main':28}; summary={};OUT.mkdir();width={0:4,1:1,2:8,4:8,5:2,6:2}
def witness(w,owner,q0,r0):
 j=w['frame'];d=D/'exact-old'
 assert (d/f'{j}.request.bin').read_bytes()==q0 and (d/f'{j}.response.bin').read_bytes()==r0
 saved=(d/f'{j}.owner.zlib').read_bytes()
 assert sha(saved)==w['owner_compressed_sha256'] and zlib.decompress(saved)==owner and sha(owner)==w['owner_sha256']
 t=Owner(owner);c,=req(q0);assert c[0]==w['op'] and c[3:]==(0,)*5
 for h,x in zip(c[1:3],w['operands'],strict=True):
  n=t.node(h);r=len(x['shape'])
  assert n[72]==x['dtype'] and n[:2]==(r,len(x['words'])) and n[12:12+r]==tuple(x['shape']) and n[16:16+r]==tuple(x['strides'])
  assert t.storage(n[27])[2].hex()==x['raw_hex']
for name,source,count,hq,hr,oldroll,removed,keep_inc,n132 in runs:
 qp=source/'requests.frames.bin';rp=source/'responses.frames.bin'
 assert sha(qp.read_bytes())==hq and sha(rp.read_bytes())==hr
 qs=list(frames(qp,b'HTOWNR01'));rs=list(frames(rp,b'HTOWNS01'));assert len(qs)==len(rs)==count
 out=OUT/name;ss=W.Session(ELF,directory=out);drop=[];ret=[];roll=0;unsupported=0;mask=0
 try:
  for j,(q0,r0) in enumerate(zip(qs,rs,strict=True),1):
   status=resp(r0)[0]; commands=req(q0);ids=[i[0] for i in commands]
   if status:prior=ss.owner()
   if 132 in ids:
    assert ids==[132] and status==4
    if name=='and':assert j==prior132['frame'] and sha(q0)==prior132['sha_request'] and sha(r0)==prior132['sha_reply'] and sha(prior['bytes'])==prior132['sha_owner']
    unsupported+=1
   if 130 in ids:assert ids==[130] and status==4;mask+=1
   if (row:=known.get((name,sha(q0),sha(r0)))) is not None:
    assert row['frame']==j and status==7
    witness(row,prior['bytes'],q0,r0)
    record=dict(frame=j,request_sha256=row['request_sha256'],reply_sha256=row['response_sha256'],owner_sha256=row['owner_sha256'])
    if row['after_status']==0:drop.append(record);continue
    assert row['after_status']==7;ret.append(record)
   z=ss.exchange(q0)
   assert z==r0,(name,j,ids,status,resp(z)[0],sha(q0))
   if status:assert ss.owner()==prior;roll+=1
 finally:ss.finish()
 assert len(drop)==removed and len(ret)==keep_inc and unsupported==n132 and mask==expected130[name]
 assert roll==oldroll-removed and ss.frames==count-removed
 skip={z['frame'] for z in drop}
 qf=out/'requests.frames.bin';rf=out/'responses.frames.bin'
 assert list(frames(qf,b'HTOWNR01'))==[q for j,q in enumerate(qs,1) if j not in skip]
 assert list(frames(rf,b'HTOWNS01'))==[r for j,r in enumerate(rs,1) if j not in skip]
 summary[name]=dict(physical_frames=ss.frames,exact_rollbacks=roll,retired_exact_wire=drop,retained_exact_incompat=ret,
  op132_unsupported=unsupported,op130_unsupported=mask,q_sha=sha(qf.read_bytes()),r_sha=sha(rf.read_bytes()))
 print(name,ss.frames,roll,len(drop),len(ret),flush=True)
assert sum(len(x['retired_exact_wire']) for x in summary.values())==9 and sum(len(x['retained_exact_incompat']) for x in summary.values())==3
assert 'torch' not in sys.modules
(OUT/'receipt.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
