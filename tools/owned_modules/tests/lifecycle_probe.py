from pathlib import Path
import hashlib
import json
import struct
import subprocess

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-build', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=False)
COMP = args.native_build.resolve() / 'owned-module-compiler'
LINK = args.native_build.resolve() / 'linker-abi2-w1m'
sha = lambda raw: hashlib.sha256(raw).hexdigest()
sources = {
    'allocation': '''hia-asm.v1 hepta.hia.program.v8 allocation
exports(create)
fn create(n:usize) -> (buffer,bool) effects(alloc,read,write,trap) entry(b0)
block b0(n:usize)
  b:buffer,ok:bool = buffer.try_new(n)
  ret(b,ok)
end
''',
    'transfer': '''hia-asm.v1 hepta.hia.program.v8 transfer
exports(mutate)
fn mutate(b:buffer) -> (buffer) effects(alloc,read,write,trap) entry(b0)
block b0(b:buffer)
  child:mutbuf = borrow.mut(b)
  value:u8 = const() {"value":173}
  _ = buffer.fill(child,value)
  _ = loan.end(child)
  moved:buffer = owner.move(b)
  ret(moved)
end
''',
    'caller': '''hia-asm.v1 hepta.hia.program.v8 caller
exports(entry)
fn create(n:usize) -> (buffer,bool) effects(alloc,read,write,trap) external()
fn mutate(b:buffer) -> (buffer) effects(alloc,read,write,trap) external()
fn entry(s:readbuf,o:mutbuf,w:mutbuf) -> (u32) effects(alloc,read,write,trap) entry(b0)
block b0(s:readbuf,o:mutbuf,w:mutbuf)
  n:usize = const() {"value":2097169}
  owner:buffer,ok:bool = call(n) {"function":"create"}
  z:usize = const() {"value":0}
  initial:readbuf = borrow.read(owner)
  zero:u8 = buffer.load(initial,z)
  _ = loan.end(initial)
  moved:buffer = call(owner) {"function":"mutate"}
  read:readbuf = borrow.read(moved)
  first:u8 = buffer.load(read,z)
  one:usize = const() {"value":1}
  last_index:usize = int.sub.wrap(n,one)
  last:u8 = buffer.load(read,last_index)
  _ = loan.end(read)
  _ = owner.drop(moved)
  _ = buffer.store(o,z,zero)
  _ = buffer.store(o,one,first)
  two:usize = const() {"value":2}
  three:usize = const() {"value":3}
  yes:u8 = const() {"value":1}
  no:u8 = const() {"value":0}
  success:u8 = select(ok,yes,no)
  _ = buffer.store(o,two,last)
  _ = buffer.store(o,three,success)
  count:u32 = const() {"value":4}
  ret(count)
end
''',
}
objects = []
for name in ('caller', 'allocation', 'transfer'):
    raw = sources[name].encode()
    (OUT / (name + '.hepta')).write_bytes(raw)
    result = subprocess.run([str(COMP)], input=raw, capture_output=True, env={'PATH': ''}, timeout=30)
    (OUT / (name + '.compile.stderr')).write_bytes(result.stderr)
    assert result.returncode == 0 and not result.stderr, (name, result.returncode, result.stderr)
    (OUT / (name + '.o')).write_bytes(result.stdout)
    objects.append(result.stdout)
entry = b'entry'
request = b'HIAOWN02' + struct.pack('<3Q', len(objects), len(entry), 0) + entry
request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
(OUT / 'request.bin').write_bytes(request)
linked = subprocess.run([str(LINK)], input=request, capture_output=True, env={'PATH': ''}, timeout=30)
(OUT / 'link.stderr').write_bytes(linked.stderr)
assert linked.returncode == 0 and not linked.stderr, (linked.returncode, linked.stderr)
exe = OUT / 'application'
exe.write_bytes(linked.stdout)
exe.chmod(0o700)
result = subprocess.run([str(exe)], input=b'', capture_output=True, env={'PATH': ''}, timeout=30)
(OUT / 'actual.bin').write_bytes(result.stdout)
(OUT / 'execute.stderr').write_bytes(result.stderr)
expected = bytes([0, 173, 173, 1])
assert result.returncode == 0 and not result.stderr and result.stdout == expected, (result.returncode, result.stderr, result.stdout.hex())
report = {'status': 'PASS', 'case': 'three-unit-2MiB-allocation-borrow-mutate-move-return-read-free',
          'compiler_sha256': sha(COMP.read_bytes()), 'linker_sha256': sha(LINK.read_bytes()),
          'objects_sha256': [sha(obj) for obj in objects], 'executable_sha256': sha(exe.read_bytes()),
          'allocation_bytes': 2097169, 'application_workspace_bytes': 1048576,
          'expected_hex': expected.hex(), 'observed_hex': result.stdout.hex(),
          'native_allocation_backend': 'Linux mmap/munmap with hidden length header',
          'libc_runtime_dependency': False, 'host_numerical_runtime': False, 'runtime_path': 'empty'}
(OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
