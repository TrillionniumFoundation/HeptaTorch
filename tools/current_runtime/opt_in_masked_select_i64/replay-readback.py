"""Read captured reply pairs independently; no runtime, producer, or PyTorch."""
import hashlib,json,struct,sys
from pathlib import Path
ref,new,receipt=map(Path,sys.argv[1:])
p=json.loads(receipt.read_bytes()); assert p['no_torch'] and p['byte_exact'] and p['retired']==0
assert len(p['rows'])==10 and p['frames']==2385
def parse(b,k):
 cur=0
 while cur<len(b):
  ident,n=struct.unpack_from('<8sQ',b,cur);cur+=16
  assert ident==k and 32<=n<=1<<20 and cur+n<=len(b)
  yield b[cur:cur+n];cur+=n
 assert cur==len(b)
for r in p['rows']:
 g=r['group']
 for typ,magic in [('requests',b'HTOWNR01'),('responses',b'HTOWNS01')]:
  fn=typ+'.frames.bin';a=(ref/g/fn).read_bytes();b=(new/g.replace('/','-')/fn).read_bytes()
  assert hashlib.sha256(a).hexdigest()==r[typ+'_sha256']
  xs=list(parse(a,magic));ys=list(parse(b,magic))
  assert len(xs)==len(ys)==r['frames'] and xs==ys and a==b
print(json.dumps(dict(readback='PASS',frames=2385,retired=0),sort_keys=True))
