"""Conservatively re-encode exact locked BF16/F32 only rank-one profile rows."""
from pathlib import Path
import collections,hashlib,json,sys
inp,out=map(Path,sys.argv[1:]);raw=inp.read_bytes()
H=lambda b:hashlib.sha256(b).hexdigest()
assert H(raw)=='77406a184aae0d9d3e2507a7e62b8107600348288947c0311220d20e8f3146e4'
a=json.loads(raw)
assert len(a['rows'])==792 and len(a['mismatches'])==494
width={0:4,6:2};raw_bits=lambda seq,w:b''.join(t.to_bytes(w,'little') for t in seq)
rows=[]
for r in a['rows']:
 if r['pair']!='0+6':continue
 assert r['mismatches']==0 and r['actual']==r['expected']
 z=r['n'];x,y=r['left_dtype'],r['right_dtype']
 assert {x,y}=={0,6}
 seqs=[r['mask'],r['left'],r['right'],r['actual']]
 assert all(len(v)==z for v in seqs)
 rows.append(dict(label=r['label'],dtype=0,width=4,left_dtype=x,right_dtype=y,shape=[z],mask_shape=[z],left_shape=[z],
  right_shape=[z],strides=[1],mask=r['mask'],left=r['left'],right=r['right'],
  **{k+'_hex':raw_bits(v,w).hex() for k,v,w in zip(('mask','left','right','result'),seqs,(1,width[x],width[y],4))}))
assert len(rows)==264 and sum(len(r['mask']) for r in rows)==8660
out.write_text(json.dumps(dict(rows=rows,oracle=a['oracle'],original_profile_sha256=H(raw)),sort_keys=True,indent=2)+'\n')
