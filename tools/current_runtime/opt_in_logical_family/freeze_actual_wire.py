"""Record the only potential newly admitted exact AND neighbor refusal."""
from pathlib import Path
import json,sys,zlib
D=Path(__file__).parent;P=Path('/tmp/hepta-logical-and-prereg');sys.path.insert(0,str(P))
from read_old import Owner,frames,req,resp,sha
s=P/'candidate';r=json.loads((s/'report.json').read_bytes())
qb=(s/'requests.frames.bin').read_bytes();rb=(s/'responses.frames.bin').read_bytes()
assert sha(qb)=='db360c3e0b071ff3993d331d61276bbdf6437d3afecb2619b9a402e73080713f'
assert sha(rb)=='456fb911c1b01d4f780cb8c55aa032c420141754218a09685c552cafb3ae3feb'
Q=list(frames(s/'requests.frames.bin',b'HTOWNR01'));R=list(frames(s/'responses.frames.bin',b'HTOWNS01'))
hit=[]
for j,(q,b) in enumerate(zip(Q,R,strict=True),1):
 for command in req(q):
  if command[0] in (132,133):hit.append((j,q,b,command))
(j,q,b,c),=hit
e,=[e for e in r['rows'] if e.get('label')=='neighbor-132'];assert e['frame']==j==2490
assert sha(q)==e['request_sha256'] and resp(b)==(4,[]) and e['pre']==e['post'] and c[:1]==(132,) and c[1]==c[2] and c[3:]==(0,)*5
m=e['pre'];comp=(s/'owners'/(m['sha256']+'.zlib')).read_bytes()
r=zlib.decompress(comp);assert len(r)==m['length'] and sha(r)==m['sha256']
o=Owner(r);n=o.node(c[1]);assert n[:2]==(1,1) and n[12]==n[16]==1 and n[72]==0 and n[4]==n[29]==n[83]==0
blob=o.storage(n[27])[2];assert blob.hex()=='0000803f'
for fn,payload in [('old-132.request.bin',q),('old-132.reply.bin',b),('old-132.owner.zlib',comp)]: (D/fn).write_bytes(payload)
out=dict(frame=j,op=132,old_status=4,owner_rollback=True,sha_request=sha(q),sha_reply=sha(b),sha_owner=sha(r),
  sha_owner_compressed=sha(comp),source_stream_frames=len(Q),op133_absent=True,words=[1065353216],dtype=0,shape=[1],strides=[1],same_handle=True,
  expected_if_or_raw='01',expected_if_xor_raw='00',action='if selected op132 rerun exact operand witness and explicitly retire only this original status4 invocation, preserving every original op131 frame')
(D/'old-132.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
print(json.dumps(out))
