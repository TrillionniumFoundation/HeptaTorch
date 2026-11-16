#!/usr/bin/env python3
"""Compile and pack the new session linker entirely through native HIA tools."""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--generated', type=Path, required=True)
parser.add_argument('--compiler', type=Path, required=True)
parser.add_argument('--borrowed-tools', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
generated = args.generated.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
compiler = args.compiler.resolve()
old = args.borrowed_tools.resolve()
sha = lambda raw: hashlib.sha256(raw).hexdigest()
assert sha(compiler.read_bytes()) == '0ccdb524ea327ab290b2e70feaac49a4789abb2fee6da8746a095f435c17af73'
assert sha((old / 'module-compiler').read_bytes()) == 'e250d93bc0b2a5ff878d2188c95780abf7e7ebaf186e4531b136fa2735b2af59'
assert sha((old / 'linker-packer').read_bytes()) == '98f0181dd509ffeb09679c6602751c86c2b972e22c5f1b9c3fd26a1e677f972b'
producer_snapshots = {path: path.read_bytes() for path in
                      (compiler, old / 'module-compiler', old / 'linker-packer')}
receipt = json.loads((generated / 'authoring.json').read_text())
source = (generated / 'owned-session-linker.hepta').read_bytes()
assert sha(source) == receipt['source_sha256']


def native(executable, request, path):
    proc = subprocess.run([str(executable)], input=request, capture_output=True, env={'PATH': ''}, timeout=30)
    path.with_suffix(path.suffix + '.stderr').write_bytes(proc.stderr)
    assert proc.returncode == 0 and not proc.stderr and proc.stdout, (path, proc.returncode, proc.stderr)
    path.write_bytes(proc.stdout)
    return proc.stdout


image = native(compiler, source, output / 'linker.image')
baseline = native(old / 'module-compiler', source, output / 'linker.baseline-image')
assert image == baseline
request = b'HIAFLT01' + struct.pack('<Q', len(image)) + image
executable = native(old / 'linker-packer', request, output / 'owned-session-linker')
(output / 'owned-session-linker').chmod(0o700)
assert (generated / 'owned-session-linker.hepta').read_bytes() == source
for path, snapshot in producer_snapshots.items():
    assert path.read_bytes() == snapshot, str(path)
report = {'schema': 'heptatorch.owned-session-linker-native-build.v1', 'status': 'PASS',
          'authoring_receipt_sha256': sha((generated / 'authoring.json').read_bytes()),
          'source_sha256': sha(source), 'image_sha256': sha(image),
          'executable_sha256': sha(executable), 'compiler_sha256': sha(compiler.read_bytes()),
          'baseline_compiler_sha256': sha((old / 'module-compiler').read_bytes()),
          'native_packer_sha256': sha((old / 'linker-packer').read_bytes()),
          'compiler_images_identical': True, 'startup_prefix_sha256': receipt['startup_prefix_sha256'],
          'producer_bytes_verified_before_and_after': True,
          'request_magic': receipt['request_magic'], 'qualification_only': receipt['qualification_only'],
          'target_libc_dependency': False, 'host_numerical_runtime': False}
(output / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
