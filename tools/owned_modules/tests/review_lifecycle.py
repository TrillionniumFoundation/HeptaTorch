"""Run source-level ownership and real syscall allocation qualification."""
from pathlib import Path
import hashlib
import json
import re
import resource
import runpy
import struct
import subprocess

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-build', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--borrowed-tools', type=Path, required=True)
args = parser.parse_args()
BUILD = args.native_build.resolve()
O = args.output.resolve()
O.mkdir(parents=True, exist_ok=False)
C = BUILD / 'owned-module-compiler'
L = BUILD / 'linker-abi2-w1m'
OLD = args.borrowed_tools.resolve()
sha = lambda raw: hashlib.sha256(raw).hexdigest()
PEER = Path(__file__).resolve().parent / 'owner_fixtures.py'
peer = runpy.run_path(str(PEER), run_name='fixture_authoring_only')
rows = []


def limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (64 << 20, 64 << 20))


def compile_source(name, source, expected=0, compiler=C):
    raw = source.encode() if isinstance(source, str) else source
    (O / (name + '.hepta')).write_bytes(raw)
    result = subprocess.run([str(compiler)], input=raw, capture_output=True, env={'PATH': ''}, timeout=30)
    (O / (name + '.compile.stdout')).write_bytes(result.stdout)
    (O / (name + '.compile.stderr')).write_bytes(result.stderr)
    assert result.returncode == expected and not result.stderr, (name, result.returncode, result.stderr)
    if expected:
        assert not result.stdout
        rows.append({'case': name, 'kind': 'native-source-refusal', 'status': 'PASS',
                     'source_sha256': sha(raw), 'compiler_sha256': sha(compiler.read_bytes()), 'exit': expected})
    else:
        (O / (name + '.o')).write_bytes(result.stdout)
    return result.stdout


def frame(objects, magic=b'HIAOWN02', entry=b'entry'):
    return magic + struct.pack('<3Q', len(objects), len(entry), 0) + entry + b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)


def link_objects(name, objects, expected=0, linker=L, magic=b'HIAOWN02'):
    request = frame(objects, magic)
    (O / (name + '.request')).write_bytes(request)
    result = subprocess.run([str(linker)], input=request, capture_output=True, env={'PATH': ''}, timeout=30)
    (O / (name + '.link.stdout')).write_bytes(result.stdout)
    (O / (name + '.link.stderr')).write_bytes(result.stderr)
    assert result.returncode == expected and not result.stderr, (name, result.returncode, result.stderr)
    if expected:
        assert not result.stdout
        rows.append({'case': name, 'kind': 'native-link-refusal', 'status': 'PASS',
                     'request_sha256': sha(request), 'linker_sha256': sha(linker.read_bytes()), 'exit': expected})
        return None
    exe = O / (name + '.native')
    exe.write_bytes(result.stdout)
    exe.chmod(0o700)
    return exe


def execute(name, executable, expected=b'', exit_code=0):
    result = subprocess.run([str(executable)], input=b'', capture_output=True,
                            env={'PATH': ''}, timeout=20, preexec_fn=limits)
    (O / (name + '.execute.stdout')).write_bytes(result.stdout)
    (O / (name + '.execute.stderr')).write_bytes(result.stderr)
    assert result.returncode == exit_code and not result.stderr and result.stdout == expected, (
        name, result.returncode, result.stderr, result.stdout.hex(), expected.hex())
    rows.append({'case': name, 'kind': 'native-execution', 'status': 'PASS', 'exit': result.returncode,
                 'executable_sha256': sha(executable.read_bytes()), 'expected_hex': expected.hex(),
                 'observed_hex': result.stdout.hex(), 'address_space_limit': 64 << 20})


def wrapper(source):
    assert source.startswith('hia-asm.v1 hepta.hia.program.v8 selfhost_elf\n')
    raw = source.split('\n', 1)[1]
    body = '''fn entry(s:readbuf,o:mutbuf,w:mutbuf) -> (u32) effects(alloc,read,write,trap) entry(b0)
block b0(s:readbuf,o:mutbuf,w:mutbuf)
  marker:u8 = const() {"value":165}
  _ = buffer.fill(o,marker)
  result:u32 = call(s,o,w) {"function":"f0"}
  wide:u64 = int.cast(result) {"mode":"wrap"}
'''
    for i in range(4):
        body += f'  shift{i}:u64 = const() {{"value":{8*i}}}\n  part{i}:u64 = int.shr.checked(wide,shift{i})\n'
        body += f'  byte{i}:u8 = int.cast(part{i}) {{"mode":"wrap"}}\n  at{i}:usize = const() {{"value":{16+i}}}\n  _ = buffer.store(o,at{i},byte{i})\n'
    body += '  count:u32 = const() {"value":20}\n  ret(count)\nend\n'
    return 'hia-asm.v1 hepta.hia.program.v8 owner_probe\nexports(entry)\n' + body + raw


alloc = peer['alloc']
cases = [
    ('new-empty', alloc(0), 0, bytes([165])*16),
    ('new-2MiB-plus-17', alloc(2097169), 2097169, bytes(8)+bytes([165])*8),
    ('try-empty-success', alloc(0, True), 0, bytes([165])*16),
    ('try-high-bit-empty-failure', alloc(1 << 63, True), 0xffffffff, bytes([165])*16),
    ('try-all-bits-set-empty-failure', alloc((1 << 64)-1, True), 0xffffffff, bytes([165])*16),
    ('try-native-kernel-ENOMEM', alloc(1 << 62, True), 0xffffffff, bytes([165])*16),
    ('try-header-near-signed-limit', alloc((1 << 63)-8, True), 0xffffffff, bytes([165])*16),
    ('move-borrow-copy-drop', peer['MOVE'], 17, bytes([173])*8+bytes([165])*8),
    ('owner-tuple-private-return', peer['RETURN'], 8, bytes(8)+bytes([165])*8),
    ('three-private-owner-slots', peer['THREE'], 16, bytes([165])*16),
    ('owner-call-consumes', peer['CONSUME'], 8, bytes([165])*16),
    ('owner-cfg-join', peer['CFG'], 8, bytes(8)+bytes([165])*8),
    ('chain-push-pop', peer['CHAIN'], 8, bytes([93])*8+bytes([165])*8),
]
for name, source, result, expected in cases:
    obj = compile_source(name, wrapper(source))
    execute(name, link_objects(name, [obj]), expected + struct.pack('<I', result))
for name, size in [('mandatory-high-bit-trap', 1 << 63), ('mandatory-kernel-ENOMEM-trap', 1 << 62)]:
    obj = compile_source(name, wrapper(alloc(size)))
    execute(name, link_objects(name, [obj]), exit_code=-4)

base = alloc(8, True)
bad = {
    'missing-alloc-effect': base.replace('effects(alloc,read,write,trap)', 'effects(read,write,trap)'),
    'leaked-owner': base.replace('  _ = owner.drop(v4)\n', ''),
    'double-drop': base.replace('  _ = owner.drop(v4)\n', '  _ = owner.drop(v4)\n  _ = owner.drop(v4)\n'),
    'drop-with-live-loan': base.replace('  _ = loan.end(v7)\n', ''),
    'move-with-live-loan': peer['MOVE'].replace('  v5:buffer = owner.move(v4)', '  v30:readbuf = borrow.read(v4)\n  v5:buffer = owner.move(v4)'),
    'use-after-move': peer['MOVE'].replace('borrow.mut(v5)', 'borrow.mut(v4)'),
    'use-after-owner-call': peer['CONSUME'].replace('  ret(v5)', '  v6:usize = buffer.len(v4)\n  ret(v5)'),
    'return-with-live-child': peer['RETURN'].replace('  ret(v1,v2)', '  v3:readbuf = borrow.read(v1)\n  ret(v1,v2)'),
    'pop-with-live-head': peer['CHAIN'].replace('  v11:chain,v12:buffer,v13:bool = chain.pop(v8)', '  v30:readbuf = chain.head_read(v8)\n  v11:chain,v12:buffer,v13:bool = chain.pop(v8)'),
}
for name, source in bad.items():
    compile_source(name, wrapper(source), 2)
compile_source('ordinary-flat-owned-refused', base.replace('selfhost_elf', 'selfhost'), 2)
compile_source('old-borrowed-profile-still-refuses-owned', wrapper(base), 2, OLD / 'module-compiler')

drop = '''hia-asm.v1 hepta.hia.program.v8 effect_probe
exports(discard)
fn discard(b:buffer) -> () effects(alloc) entry(b0)
block b0(b:buffer)
  _ = owner.drop(b)
  ret()
end
'''
compile_source('drop-requires-declared-trap', drop, 2)
compile_source('drop-with-declared-trap', drop.replace('effects(alloc)', 'effects(alloc,trap)'))
pop = '''hia-asm.v1 hepta.hia.program.v8 effect_probe
exports(pop)
fn pop(c:chain) -> (chain,buffer,bool) effects(alloc) entry(b0)
block b0(c:chain)
  rest:chain,b:buffer,ok:bool = chain.pop(c)
  ret(rest,b,ok)
end
'''
compile_source('pop-requires-declared-trap', pop, 2)
compile_source('pop-with-declared-trap', pop.replace('effects(alloc)', 'effects(alloc,trap)'))
optional = '''hia-asm.v1 hepta.hia.program.v8 optional_probe
exports(make)
fn make(n:usize) -> (buffer,bool) effects(alloc) entry(b0)
block b0(n:usize)
  b:buffer,ok:bool = buffer.try_new(n)
  ret(b,ok)
end
'''
compile_source('try-new-retains-nontrapping-alloc-contract', optional)
escape = '''hia-asm.v1 hepta.hia.program.v8 escape_probe
exports(escape)
fn escape(b:buffer) -> (readbuf) effects(alloc,read,write,trap) entry(b0) return_loans(["b"])
block b0(b:buffer)
  loan:readbuf = borrow.read(b)
  ret(loan)
end
'''
compile_source('borrowed-escape-from-owner-refused', escape, 2)

# A native free that only retired descriptors without releasing memory would
# exceed this child's 64 MiB address-space limit after a handful of iterations.
loop = '''hia-asm.v1 hepta.hia.program.v8 release_loop
exports(entry)
fn entry(s:readbuf,o:mutbuf,w:mutbuf) -> (u32) effects(alloc,read,write,trap) entry(b0)
block b0(s:readbuf,o:mutbuf,w:mutbuf)
  initial:u64 = const() {"value":0}
  jump b1(initial,o)
block b1(i:u64,out:mutbuf)
  size:usize = const() {"value":8388608}
  b:buffer = buffer.new(size)
  loan:mutbuf = borrow.mut(b)
  z:usize = const() {"value":0}
  value:u8 = const() {"value":173}
  _ = buffer.store(loan,z,value)
  _ = loan.end(loan)
  _ = owner.drop(b)
  one:u64 = const() {"value":1}
  next:u64 = int.add.wrap(i,one)
  limit:u64 = const() {"value":256}
  repeat:bool = compare.lt(next,limit)
  branch repeat b1(next,out) b2(next,out)
block b2(total:u64,out2:mutbuf)
  z2:usize = const() {"value":0}
  byte:u8 = const() {"value":173}
  _ = buffer.store(out2,z2,byte)
  count:u32 = const() {"value":1}
  ret(count)
end
'''
obj = compile_source('256-real-8MiB-release-iterations', loop)
execute('256-real-8MiB-release-iterations', link_objects('256-real-8MiB-release-iterations', [obj]), b'\xad')

# Old and new allocation/interface profiles cannot be mixed implicitly.
simple = '''hia-asm.v1 hepta.hia.program.v8 simple
exports(entry)
fn entry(s:readbuf,o:mutbuf,w:mutbuf) -> (u32) effects(read,write,trap) entry(b0)
block b0(s:readbuf,o:mutbuf,w:mutbuf)
  result:u32 = const() {"value":0}
  ret(result)
end
'''
old_obj = compile_source('abi1-simple', simple, compiler=OLD / 'module-compiler')
new_obj = compile_source('abi2-simple', simple)
link_objects('abi1-object-refused-by-abi2', [old_obj], 2)
link_objects('abi2-object-refused-by-abi1', [new_obj], 2, OLD / 'module-linker', b'HIABLK01')
link_objects('compiler-request-magic-refused-by-runtime-linker', [new_obj], 2, magic=b'HIAOWNC2')
link_objects('runtime-request-magic-refused-by-compiler-linker', [new_obj], 2, BUILD / 'linker-abi2-w2m')

report = {'schema': 'heptatorch.owned-resource-lifecycle-review.v1', 'status': 'PASS',
          'compiler_sha256': sha(C.read_bytes()), 'linker_sha256': sha(L.read_bytes()),
          'source_fixture_author_sha256': sha(PEER.read_bytes()), 'cases': len(rows), 'rows': rows,
          'runtime_path': 'empty', 'libc_target_dependency': False, 'host_owner_vm': False,
          'new_effect_contract': 'owner.drop and chain.pop require alloc+trap; buffer.try_new remains alloc-only',
          'tested_address_space_limit': 64 << 20}
(O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, indent=2))
