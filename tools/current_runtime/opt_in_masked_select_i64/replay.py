"""Exact #67 F32 and prior comparable frame replay; no Torch."""
from pathlib import Path
import hashlib,json,struct,sys
sha=lambda x:hashlib.sha256(x).hexdigest()
h,pp,new,prior,out,pin=map(Path,sys.argv[1:])
assert sha(new.read_bytes())==pin.read_text().strip()
assert sha((h/'test_runtime_candidate.py').read_bytes())=='40690655b06038e74d8629f0f30a1d6c61ee49e8537f2ee4919edf6b9a0e9226'
sys.path.insert(0,str(h));from test_runtime_candidate import load_wire
W=load_wire(pp);assert 'torch' not in sys.modules
pins={
'new':('76464ee4df5aeeb351ac10f59b15cb850a76408dbd829b575e8a9252cdc00136','e48ad88e9dc1a4c07347358a5a92de524dbfa97b5e1d275c4a9d49c2456ef336',630),
'replay/candidate':('b0d58b7dedece4ddf00626cb7f616fe00a897311b2cca353602d9e6fa20fac32','69219654d820523b6cb8f1ec7bf9abec725b56ab7f0a3d9b4a121b7e3e0962ae',695),
'replay/guards':('a9ab8807e83c9df17e0c2e352b252bb1913d8ccf8e95ddc26669945cb472e29c','6edf6bf23fe1d1f65148a7852a9f89f8dce1d6d8417057525a9f9de365b2c37a',56),
'replay/replay-main':('cd85438f0e5dc3d59bce7d0cb160188f3b17e72c53fdb8511414b87a8e996ef0','4e569546ca8b506c5830d240f46d1288059ac0bb9c7d002bf4a323b1650e455d',262),
'replay/mirror-sparse-clean':('cfea2e8a26ae8cfc393266f87d73bd58670472f109c87ac179c7c56d68d40323','e19d4f368e20b9727f3b956730b1753963212cce0d04361015c8a166c551e3c8',7),
'replay/mirror-sparse-reject':('24465b4d34b81d615956001df35ca444544a99ac8d3e73d6a3f9ed7b56f5234d','244dd75753cc33c5ef2caaed8a90527be1c8933ab1eee7b9cc90b69cb9bda59f',8),
'replay/mirror-zero-clean':('e2050025b2eedc78bdc671914aaafdba5cb4e5820a3e7681e32feedb88561e61','e19d4f368e20b9727f3b956730b1753963212cce0d04361015c8a166c551e3c8',7),
'replay/mirror-zero-reject':('c6a98ab25be01ee8e99372865d784d6c4b85922f490de87f4bdf2ddfc89e1fa0','244dd75753cc33c5ef2caaed8a90527be1c8933ab1eee7b9cc90b69cb9bda59f',8),
'replay/witness-guards-new':('c6ab05dbc16337c13d6b6eef869acebf90c5aa80d635e864e7850aa53cd91acb','11f9550b055e87d737dee699874feba365fc84878f440b42179febf5fd752e3a',20),
'replay/witness-candidate-new':('1144d13f513e5498bab82df1a7252fd8f1616d6dad4ed1be155a6a09345d5b00','0f438915511746ffc668d871e05bc93b707effc0bd9de270c451965a84ebfa39',692)}
def frames(b,tag):
 i=0
 while i<len(b):
  got,n=struct.unpack_from('<8sQ',b,i);assert got==tag and 32<=n<=1<<20 and i+16+n<=len(b)
  i+=16;yield b[i:i+n];i+=n
 assert i==len(b)
out.mkdir();rows=[]
for group,(x,y,n) in pins.items():
 path=prior/group
 q=(path/'requests.frames.bin').read_bytes();r=(path/'responses.frames.bin').read_bytes()
 assert sha(q)==x and sha(r)==y
 dest=out/group.replace('/','-')
 s=W.Session(new,directory=dest)
 try:
  for i,(a,b) in enumerate(zip(frames(q,b'HTOWNR01'),frames(r,b'HTOWNS01'),strict=True),1):
   got=s.exchange(a); assert got==b,(group,i,sha(got),sha(b))
 finally:s.finish()
 assert s.frames==n and (dest/'requests.frames.bin').read_bytes()==q and (dest/'responses.frames.bin').read_bytes()==r
 rows.append(dict(group=group,frames=n,requests_sha256=x,responses_sha256=y))
print(json.dumps(dict(frames=sum(x['frames'] for x in rows),rows=rows,byte_exact=True,retired=0,new_sha256=sha(new.read_bytes()),no_torch='torch' not in sys.modules),sort_keys=True,indent=2))
