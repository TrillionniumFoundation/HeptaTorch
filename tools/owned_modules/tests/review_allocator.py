from pathlib import Path
import hashlib
import json
import os
import struct
import subprocess
import sys

import argparse
parser = argparse.ArgumentParser(description='Observe actual HIA-emitted allocator bytes, including ABI and error paths.')
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--compiler-adaptation', type=Path, required=True)
parser.add_argument('--allocation-object', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
R = Path(__file__).resolve().parent
O = args.output.resolve()
O.mkdir(parents=True, exist_ok=False)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
receipt = json.loads(args.compiler_adaptation.read_text())
obj = args.allocation_object.read_bytes()
table = struct.unpack_from('<Q', obj, 40)[0]
text_at, text_length = struct.unpack_from('<QQ', obj, table + 64 + 24)
code = obj[text_at:text_at + text_length]
inputs = {}
for name in ('calloc', 'free'):
    data = bytes.fromhex(receipt['allocator_helpers'][name]['hex'])
    assert sha(data) == receipt['allocator_helpers'][name]['sha256']
    assert code.count(data) == 1, name
    (O / (name + '.bin')).write_bytes(data)
    inputs[name] = {'sha256': sha(data), 'bytes': len(data), 'object_text_offset': code.index(data)}

# Independent test instrumentation uses the pinned upstream encoder only at
# authoring time. The observer loads no HIA authoring or compiler modules.
upstream = args.source_root.resolve()
contract = (upstream / 'docs/HIA_SELFHOST_CORE_SUPPORT.json').read_bytes()
assert sha(contract) == '8cbf1595eeaef6280621120dd407066ae4f992ff9f9129f3404f8c36965a304c'
machine = upstream / 'hepta/hia_machine_x64.py'
assert sha(machine.read_bytes()) == json.loads(contract)['source_identity']['files']['hepta/hia_machine_x64.py']
closure = receipt['upstream_source_files_verified_before_and_after']
for relative, expected in closure.items():
    assert sha((upstream / relative).read_bytes()) == expected, relative
sys.path.insert(0, str(upstream))
from hepta.hia_machine_x64 import Code

c = Code()
registers = ['rbx', 'rbp', 'r12', 'r13', 'r14', 'r15']
sentinels = [0x1122334455667788 + index for index in range(6)]
for register in registers:
    c.push(register)
c.alu_imm('sub', 'rsp', 40)  # entry 8 mod16 -48 -40 = aligned before call
c.store('rsp', 0, 'rdi')
c.store('rsp', 8, 'rsi')
c.lea('rax', 'rsp', 0)
c.store('rsp', 24, 'rax')
for register, sentinel in zip(registers, sentinels):
    c.mov_imm(register, sentinel)


def check_preserved():
    for register, sentinel in zip(registers, sentinels):
        c.mov_imm('rcx', sentinel)
        c.cmp(register, 'rcx')
        c.jcc('ne', 'failed')
    c.lea('rax', 'rsp', 0)
    c.load('rcx', 'rsp', 24)
    c.cmp('rax', 'rcx')
    c.jcc('ne', 'failed')


c.load('rax', 'rsp', 0)
c.mov_imm('rdi', 3)
c.mov_imm('rsi', 17)
c.emit(bytes.fromhex('ffd0'))
c.test('rax', 'rax')
c.jcc('e', 'failed')
c.store('rsp', 32, 'rax')
check_preserved()
c.load('rdi', 'rsp', 32)
c.load('rax', 'rsp', 8)
c.emit(bytes.fromhex('ffd0'))
check_preserved()
c.mov_imm('rax', 1)
c.jmp('done')
c.label('failed')
c.mov_imm('rax', 0)
c.label('done')
c.alu_imm('add', 'rsp', 40)
for register in reversed(registers):
    c.pop(register)
c.ret()
probe, relocs = c.finalize()
assert not relocs
(O / 'abi_probe.bin').write_bytes(probe)
inputs['abi_probe'] = {'sha256': sha(probe), 'bytes': len(probe)}
(O / 'inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
observer = R / 'observe_allocator.py'
positive = subprocess.run([sys.executable, '-I', '-S', '-B', str(observer), str(O), 'ordinary'],
                          capture_output=True, timeout=10)
(O / 'ordinary.stdout').write_bytes(positive.stdout)
(O / 'ordinary.stderr').write_bytes(positive.stderr)
assert positive.returncode == 0 and not positive.stderr, (positive.returncode, positive.stderr)
observed = json.loads(positive.stdout)
assert observed['status'] == 'PASS' and observed['cases'] == 12
error = subprocess.run([sys.executable, '-I', '-S', '-B', str(observer), str(O), 'free-error'],
                       capture_output=True, timeout=10)
(O / 'free-error.stdout').write_bytes(error.stdout)
(O / 'free-error.stderr').write_bytes(error.stderr)
assert error.returncode == -4 and not error.stdout and not error.stderr, (error.returncode, error.stdout, error.stderr)
report = {'schema': 'heptatorch.native-owned-allocator-backend-review.v1', 'status': 'PASS',
          'source_sha256': receipt['source_sha256'], 'native_object_sha256': sha(obj),
          'observer_sha256': sha(observer.read_bytes()), 'backend_and_probe': inputs,
          'cases': 13, 'ordinary': observed,
          'real_munmap_EINVAL_is_declared_UD2': True,
          'error_probe_is_private_header_corruption_not_admitted_hia_source': True,
          'allocator_runtime': 'native HIA-emitted Linux syscalls; no libc allocator',
          'foreign_owner_pointer_api_qualified': False}
for relative, expected in closure.items():
    assert sha((upstream / relative).read_bytes()) == expected, relative
for module in tuple(sys.modules.values()):
    filename = getattr(module, '__file__', None)
    if not filename:
        continue
    try:
        relative = Path(filename).resolve().relative_to(upstream).as_posix()
    except ValueError:
        continue
    assert closure.get(relative) == sha(Path(filename).read_bytes()), relative
(O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key not in ('ordinary', 'backend_and_probe')}, indent=2))
