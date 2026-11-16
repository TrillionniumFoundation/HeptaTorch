"""Emit a byte-exact compressed report envelope within CI log transport limits."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('report', type=Path)
args = parser.parse_args()
raw = args.report.read_bytes()
assert len(raw) <= 64 * 1024 * 1024, 'report exceeds this transport profile'
json.loads(raw)  # Validate without reserializing the original report bytes.
compressed = zlib.compress(raw, level=9)
assert zlib.decompress(compressed) == raw
encoded = base64.b64encode(compressed).decode('ascii')
print(json.dumps({'schema': 'heptatorch.ci-report-envelope.v1',
                  'path': str(args.report), 'bytes': len(raw),
                  'sha256': hashlib.sha256(raw).hexdigest(),
                  'encoding': 'zlib-base64',
                  'base64_blocks': [encoded[i:i+4096] for i in range(0, len(encoded), 4096)]},
                 indent=2))
