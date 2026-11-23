"""Restore exact official fixture without transporting the long targeted profile uncompressed."""
import base64,hashlib,sys,zlib
from pathlib import Path
inp,out=map(Path,sys.argv[1:])
packed=base64.b85decode(inp.read_bytes().strip())
raw=zlib.decompress(packed)
assert hashlib.sha256(raw).hexdigest()=='438ed9759eddadfd468f8dbd37600004836b711752921c58e160ff030fee3782'
with out.open('xb') as f:f.write(raw)
