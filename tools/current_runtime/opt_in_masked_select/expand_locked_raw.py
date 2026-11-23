"""Expand the exact measured PyTorch raw words."""
from pathlib import Path
import base64,hashlib,lzma,sys
p,t=map(Path,sys.argv[1:]);data=lzma.decompress(base64.b85decode(p.read_bytes().strip()))
assert len(data)==1457660 and hashlib.sha256(data).hexdigest()=='b879c03d455f7362128c78950858a3f7c220d8dbe51265b48745fe32463812cf'
with t.open('xb') as z:z.write(data)
