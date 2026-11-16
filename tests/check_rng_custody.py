"""Verify the frozen native RNG body map and the original licensed sources."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
sha = lambda raw: hashlib.sha256(raw).hexdigest()
source = (ROOT/'tools/rng-cpu/rng.hepta').read_bytes()
profile_raw = (ROOT/'tools/rng-cpu/profile.json').read_bytes()
assert sha(source) == 'c1de63f5f046bfb20aa4d4f65cbffa6217a38a36c0dc63b05f08c7ddc7612032'
assert sha(profile_raw) == '827b0ffd1529f9a04c91240c55c39716d72c0843fe3f2d90b8d446bade5d5e6f'
profile = json.loads(profile_raw)
extract = lambda raw: {m[2].decode(): m[1] for m in re.finditer(rb'^(fn (\w+)\([^\n]*\n.*?^end\n)', raw, re.M|re.S)}
bodies = extract(source)
assert len(bodies) == profile['functions'] == 18
assert {name: sha(body) for name, body in bodies.items()} == profile['function_body_sha256']
assert len(profile['reused_functions']) == 8
for name, row in profile['reused_functions'].items():
    assert bodies[name] == extract((ROOT/row['source']).read_bytes())[name]
    assert sha(bodies[name]) == row['body_sha256']
sources = json.loads((ROOT/'tools/rng-cpu/upstream-sources.json').read_bytes())
assert sources['upstream_commit'] == '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
assert len(sources['sources']) == 6
for row in sources['sources']:
    raw = (ROOT/row['path']).read_bytes()
    assert sha(raw) == row['sha256'] and len(raw) == row['bytes']
engine = (ROOT/'third_party/pytorch/rng/aten/src/ATen/core/MT19937RNGEngine.h').read_text()
start = engine.index(' * Copyright notice:'); end = engine.index('\n */', start)
notice = '\n'.join(line[3:] if line.startswith(' * ') else '' if line == ' *' else line
                   for line in engine[start:end].splitlines())
assert notice in (ROOT/'tools/rng-cpu/THIRD_PARTY_NOTICES.txt').read_text()
assert sha((ROOT/'tests/rng_cpu_oracle.py').read_bytes()) == '5eef7d3dd12cf1c3816a36c1429ef584cc70cbfffa423c1a15e354b19a3ec239'
print(json.dumps({'schema': 'heptatorch.native-rng-source-custody.v1', 'status': 'PASS',
                  'source_sha256': sha(source), 'functions_verified': 18,
                  'reused_function_bodies_identical': 8, 'primary_source_files': 6,
                  'original_mt19937_notice_retained': True,
                  'whole_pytorch_rewrite_complete': False}))
