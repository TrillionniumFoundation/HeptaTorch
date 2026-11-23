"""Restore exact official fixture without transporting the long targeted profile uncompressed."""
import base64,hashlib,sys,zlib
from pathlib import Path
inp,out=map(Path,sys.argv[1:])
packed=base64.b85decode(inp.read_bytes().strip())
raw=zlib.decompress(packed)
assert hashlib.sha256(raw).hexdigest()=='98dfc6aca1bc0d6faa7008c818be8b63511092963cf203692a15cce73b00d498'
with out.open('xb') as f:f.write(raw)
