"""Qualify the new compiler against the current real HeptaTorch application.

Host code only transports source/object bytes and authors the same complete
source partition already admitted by the frozen borrowed module toolchain.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import struct
import subprocess

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-build', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--repo-root', type=Path, required=True)
parser.add_argument('--baseline-compiler', type=Path, required=True)
args = parser.parse_args()
REPO = args.repo_root.resolve()
O = args.output.resolve()
O.mkdir(parents=True, exist_ok=False)
C = args.native_build.resolve() / 'owned-module-compiler'
L = args.native_build.resolve() / 'linker-abi2-w1m'
OLD = args.baseline_compiler.resolve()
sha = lambda raw: hashlib.sha256(raw).hexdigest()
pins = {
    'compiler': '0ccdb524ea327ab290b2e70feaac49a4789abb2fee6da8746a095f435c17af73',
    'linker': '23eed229e329fb5b7e2116012f74f210d5de4acd2b4f8475e364890748da86ee',
    'baseline_compiler': 'c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6',
}
for name, path in [('compiler', C), ('linker', L), ('baseline_compiler', OLD)]:
    assert sha(path.read_bytes()) == pins[name]


def native(exe, source, path):
    proc = subprocess.run([str(exe)], input=source, capture_output=True, env={'PATH': ''}, timeout=30)
    path.with_suffix(path.suffix + '.stderr').write_bytes(proc.stderr)
    assert proc.returncode == 0 and not proc.stderr and proc.stdout, (str(exe), proc.returncode)
    path.write_bytes(proc.stdout)
    return proc.stdout


rows = []
for name, source_folder in [('heptatorch', 'src'), ('unhex', 'tools')]:
    source = REPO / source_folder / (name + '.hepta')
    raw = source.read_bytes()
    (O / source.name).write_bytes(raw)
    baseline = native(OLD, raw, O / (name + '.baseline'))
    candidate = native(C, raw, O / (name + '.owned'))
    assert candidate == baseline and source.read_bytes() == raw
    rows.append({'source': source_folder + '/' + source.name, 'source_sha256': sha(raw),
                 'ordinary_executable_sha256': sha(candidate), 'ordinary_byte_identical': True})

splitter = REPO / 'tools/modules/split_runtime.py'
spec = importlib.util.spec_from_file_location('source_partition_only', splitter)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.split(O / 'heptatorch.hepta', O / 'units')
objects = []
for name in ('ht_runtime', 'ht_kernel'):
    source = (O / 'units' / (name + '.hepta')).read_bytes()
    objects.append(native(C, source, O / (name + '.o')))
entry = b'entry'
request = b'HIAOWN02' + struct.pack('<3Q', len(objects), len(entry), 0) + entry
request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
(O / 'link-request.bin').write_bytes(request)
linked = native(L, request, O / 'heptatorch-linked')
assert sha(linked) == 'cf78556c3c512227ffe123635b76e5a6af97697606e8ba3535c56f5101f4f80d'
assert linked == (REPO / 'build/heptatorch').read_bytes()
(O / 'heptatorch-linked').chmod(0o700)
report = {'schema': 'heptatorch.owned-profile-real-runtime-equivalence.v1', 'status': 'PASS',
          **{key + '_sha256': value for key, value in pins.items()},
          'source_organizer_sha256': sha(splitter.read_bytes()), 'ordinary_mode': rows,
          'linked_executable_sha256': sha(linked), 'linked_matches_frozen_borrowed_executable': True,
          'linked_objects_sha256': [sha(obj) for obj in objects], 'link_request_sha256': sha(request),
          'additional_fixture_execution_claim': False,
          'reason': 'The new linked executable is byte-identical to the already qualified complete runtime.'}
(O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
