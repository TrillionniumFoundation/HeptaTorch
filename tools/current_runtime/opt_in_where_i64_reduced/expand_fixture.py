"""Expand and pin the frozen fixtures in isolated stdlib Python."""
from pathlib import Path
import base64,hashlib,sys,zlib
src,dst=map(Path,sys.argv[1:])
pins={'prereg.json': '57b06e065d8ac10bf92f8d4133d1f525be91cd2f864bbc4d068bd8260a044502', 'suite.json': 'ef1f8673289c68d7b86cadc3104c60e3098709485b068b702cb0fd29d5f89264'}
data=zlib.decompress(base64.b85decode(src.read_bytes().strip()))
assert hashlib.sha256(data).hexdigest()==pins[dst.name]
with dst.open('xb') as f:f.write(data)
