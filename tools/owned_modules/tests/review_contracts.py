"""Replay existing object attacks and check the new three-result ABI2 fields.

This test mutates real native object files. It does not assert that a linker
authenticates arbitrary code whose metadata was forged consistently.
"""
from pathlib import Path
import hashlib
import json
import runpy
import struct
import subprocess
import sys

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-build', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--repo-root', type=Path, required=True)
args = parser.parse_args()
O = args.output.resolve()
O.mkdir(parents=True, exist_ok=False)
C = args.native_build.resolve() / 'owned-module-compiler'
L = args.native_build.resolve() / 'linker-abi2-w1m'
sha = lambda raw: hashlib.sha256(raw).hexdigest()
BASE = args.repo_root.resolve() / 'tools/modules/tests/contracts.py'
original = BASE.read_bytes()
assert sha(original) == '9a53bf0b38781c12345dc0c28c39170bde3e657694ba96e5a3f519c0065f7fbe'
adapted = original
changes = [(b"b'HIABLK01'", b"b'HIAOWN02'"),
           (b'(8,[0,2])', b'(8,[0,1])'),
           (b'(32,[1,16,(1<<64)-1])', b'(32,[17,16,(1<<64)-1])'),
           (b'heptatorch.modules-independent-contract-review.v1',
            b'heptatorch.owned-modules-existing-contract-review.v1')]
for before, after in changes:
    assert adapted.count(before) == 1
    adapted = adapted.replace(before, after)
script = O / 'existing_contracts.py'
script.write_bytes(adapted)
result = subprocess.run([sys.executable, '-I', '-S', '-B', str(script), '--compiler', str(C),
                         '--linker', str(L), '--output', str(O / 'existing')],
                        capture_output=True, timeout=120)
(O / 'existing.stdout').write_bytes(result.stdout)
(O / 'existing.stderr').write_bytes(result.stderr)
assert result.returncode == 0 and not result.stderr, (result.returncode, result.stderr)
existing = json.loads((O / 'existing/report.json').read_text())
assert existing['cases'] == 204 and existing['status'] == 'PASS'
helpers = runpy.run_path(str(script), run_name='test_utilities_only')
frame, layout, u64, patched = [helpers[key] for key in ('frame', 'layout', 'u64', 'patched')]

CALLER = b'''hia-asm.v1 hepta.hia.program.v8 owner_contract_caller
exports(entry)
fn make(n:usize) -> (buffer,chain,buffer) effects(alloc,trap) external()
fn discard(b:buffer) -> () effects(alloc,trap) external()
fn entry(s:readbuf,o:mutbuf,w:mutbuf) -> (u32) effects(alloc,read,write,trap) entry(b0)
block b0(s:readbuf,o:mutbuf,w:mutbuf)
  n:usize = const() {"value":8}
  left:buffer,middle:chain,right:buffer = call(n) {"function":"make"}
  _ = call(left) {"function":"discard"}
  _ = owner.drop(middle)
  _ = owner.drop(right)
  z:usize = const() {"value":0}
  value:u8 = const() {"value":93}
  _ = buffer.store(o,z,value)
  count:u32 = const() {"value":1}
  ret(count)
end
'''
PROVIDER = b'''hia-asm.v1 hepta.hia.program.v8 owner_contract_provider
exports(make,discard)
fn make(n:usize) -> (buffer,chain,buffer) effects(alloc,trap) entry(b0)
block b0(n:usize)
  left:buffer = buffer.new(n)
  middle:chain = chain.empty()
  right:buffer = buffer.new(n)
  ret(left,middle,right)
end
fn discard(b:buffer) -> () effects(alloc,trap) entry(b0)
block b0(b:buffer)
  _ = owner.drop(b)
  ret()
end
'''


def run(executable, request):
    return subprocess.run([str(executable)], input=request, capture_output=True,
                          env={'PATH': ''}, timeout=10)


objects = []
for name, raw in [('caller', CALLER), ('provider', PROVIDER)]:
    (O / (name + '.hepta')).write_bytes(raw)
    result = run(C, raw)
    assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF'), name
    objects.append(result.stdout)
    (O / (name + '.o')).write_bytes(result.stdout)
valid = frame(objects)
(O / 'positive.request').write_bytes(valid)
result = run(L, valid)
assert result.returncode == 0 and not result.stderr and result.stdout
executable = O / 'positive'
executable.write_bytes(result.stdout)
executable.chmod(0o700)
actual = run(executable, b'')
assert actual.returncode == 0 and actual.stdout == b'\x5d' and not actual.stderr
rows = [{'case': 'three-owner-results-and-zero-result-owner-consumer', 'status': 'PASS',
         'kind': 'native-execution', 'executable_sha256': sha(result.stdout), 'stdout_hex': '5d'}]


def corrupt(number, name, offset, value):
    changed = objects.copy()
    changed[number] = patched(changed[number], offset, value)
    request = frame(changed)
    result = run(L, request)
    assert result.returncode == 2 and not result.stdout and not result.stderr, (name, result.returncode)
    (O / (name + '.request')).write_bytes(request)
    rows.append({'case': name, 'kind': 'native-metadata-refusal', 'status': 'PASS',
                 'request_sha256': sha(request), 'exit': 2})


for number, obj in enumerate(objects):
    _, sections = layout(obj)
    meta = sections[7][0]
    for record in range(u64(obj, meta + 16)):
        at = meta + 64 + record * 256
        count = u64(obj, at + 24)
        if count == 3:
            for slot, offset in enumerate((40, 176, 184)):
                for value in (0, 6, 7, 17, (1 << 64) - 1):
                    corrupt(number, f'o{number}-r{record}-result{slot}-invalid-{value}', at + offset, value)
                current = u64(obj, at + offset)
                for value in (3, 8, 9, 16):
                    if value != current:
                        corrupt(number, f'o{number}-r{record}-result{slot}-mismatch-{value}', at + offset, value)
            for value in (0, 1, 2, 4, (1 << 64) - 1):
                corrupt(number, f'o{number}-r{record}-result-count-{value}', at + 24, value)
        if count == 0:
            for offset in (40, 176, 184):
                corrupt(number, f'o{number}-r{record}-unused-result-{offset}', at + offset, 8)
            for value in (1, 3, 4):
                corrupt(number, f'o{number}-r{record}-zero-result-count-{value}', at + 24, value)
        for offset in range(192, 256, 8):
            corrupt(number, f'o{number}-r{record}-reserved-{offset}', at + offset, 1)

report = {'schema': 'heptatorch.owned-module-contract-review.v1', 'status': 'PASS',
          'compiler_sha256': sha(C.read_bytes()), 'linker_sha256': sha(L.read_bytes()),
          'original_review_source_sha256': sha(original), 'adapted_existing_review_sha256': sha(adapted),
          'existing_review_cases': existing['cases'], 'abi2_additional_cases': len(rows),
          'existing_review_changes': [[a.decode(), b.decode()] for a, b in changes],
          'cases': existing['cases'] + len(rows), 'rows': rows,
          'limitation': 'Pinned compiler and build custody authenticate code. Native linker verifies the declared contract.'}
(O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, indent=2))
