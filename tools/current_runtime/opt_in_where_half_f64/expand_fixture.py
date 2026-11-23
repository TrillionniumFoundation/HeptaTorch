"""Restore exact official fixture without transporting the long targeted profile uncompressed."""
import base64,hashlib,sys,zlib
from pathlib import Path
inp,out=map(Path,sys.argv[1:])
packed=base64.b85decode(inp.read_bytes().strip())
raw=zlib.decompress(packed)
assert hashlib.sha256(raw).hexdigest()=='4ab287cdcdde42d702321f01a34f7c5390e098d02d4f6d9c049ae7682405fdb1'
with out.open('xb') as f:f.write(raw)
