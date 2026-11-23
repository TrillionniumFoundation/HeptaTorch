"""Restore exact official fixture without transporting the long Bool output uncompressed."""
import base64,hashlib,sys,zlib
from pathlib import Path
inp,out=map(Path,sys.argv[1:])
packed=base64.b85decode(inp.read_bytes().strip())
raw=zlib.decompress(packed)
assert hashlib.sha256(raw).hexdigest()=='428f1dcae21b4db465ed9a2ade30b3449a85f0bbbb145e6d036ff0d49fb4f229'
with out.open('xb') as f:f.write(raw)
