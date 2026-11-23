"""Restore exact official fixture without transporting the long targeted profile uncompressed."""
import base64,hashlib,sys,zlib
from pathlib import Path
inp,out=map(Path,sys.argv[1:])
packed=base64.b85decode(inp.read_bytes().strip())
raw=zlib.decompress(packed)
assert hashlib.sha256(raw).hexdigest()=='5338bec52c5c8516f9f929420a51925b74911ef5e5386d11b0582364dea0dae8'
with out.open('xb') as f:f.write(raw)
