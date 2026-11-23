"""Prove the additional four rows are exact retained wires with frozen original operands."""
from pathlib import Path
import json,sys,zlib
D=Path(__file__).parent
sys.path.insert(0,str(D))
from read_old import Owner,frames,req,resp,sha
before=Path('/tmp/hepta-where-zero-reduced-f32-prereg/retain-final/zero_wide')
stopped=D/'retain/zero_wide'
def pairs(p):
 return list(zip(frames(p/'requests.frames.bin',b'HTOWNR01'),frames(p/'responses.frames.bin',b'HTOWNS01'),strict=True))
orig=pairs(before);partial=pairs(stopped)
assert orig[:-1]!=partial
assert len(partial)==3366 and orig[:3365]==partial[:3365]
assert orig[3365][0]==partial[3365][0] and resp(orig[3365][1])==(7,[]) and resp(partial[3365][1])[0]==0
p=json.loads((D/'exact-wide-old.json').read_bytes());ledger=[]
for e in p['witnesses']:
 j=e['frame'];r=e['retained_frame']; d=D/'exact-wide-old'
 q,s,o=((d/f'{j}.{ext}').read_bytes() for ext in ('request.bin','response.bin','owner.zlib'))
 matches=[k for k,(a,b) in enumerate(orig,1) if (q,s)==(a,b)]
 assert matches==[r] and sha(q)==e['request_sha256'] and sha(s)==e['response_sha256'] and resp(s)==(7,[])
 raw=zlib.decompress(o);assert sha(o)==e['compressed_owner_sha256'] and sha(raw)==e['owner_sha256'] and len(raw)==e['owner_length']
 node=Owner(raw);one,=req(q)
 for h,arg in zip(one[1:4],e['new_prediction']['args'],strict=True):
  tag,shape,words=arg;d0=node.node(h);w={0:4,1:1,4:8}[tag]
  assert d0[72]==tag and list(d0[12:12+d0[0]])==shape
  assert node.storage(d0[27])[2]==b''.join(x.to_bytes(w,'little') for x in words)
 ledger.append(dict(original=j,retained=r,request_sha256=sha(q),response_sha256=sha(s),owner_sha256=sha(raw),older_duplicate_operand_frame=e['exact_operands_duplicate_frame']))
print(json.dumps(dict(stopped_first_uncatalogued=dict(frame=3366,old_status=7,observed_status=0,observed_reply_sha256=sha(partial[3365][1])),
    matched=ledger,scope='Four original exact predecessor archived pairs in zero_wide retained stream'),indent=2))
