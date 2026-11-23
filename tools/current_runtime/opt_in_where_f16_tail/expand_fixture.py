"""Expand byte-exact official fixture or held-out compact records."""
import base64,hashlib,sys,zlib
from pathlib import Path
kind,infile= sys.argv[1],Path(sys.argv[2]);outfile=Path(sys.argv[3])
sha={'heldout':'0190d75767f242a51f53c46ab68c0fda0700f251109d18fc7175a04652e6254a',
 'fixture':'ca062e21f8e960d40a6d6794e9ad78634b0a57e65fc21f8cfb847dbcf73c42dd'}[kind]
raw=zlib.decompress(base64.b85decode(infile.read_bytes().strip()))
assert hashlib.sha256(raw).hexdigest()==sha
with outfile.open('xb') as f:f.write(raw)
