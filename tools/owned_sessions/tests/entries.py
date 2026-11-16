"""Native positive/negative admission for the new owned session entry pair."""
from pathlib import Path
import hashlib
import json
import os
import selectors
import struct
import subprocess

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--compiler', type=Path, required=True)
parser.add_argument('--linker', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
O = args.output.resolve()
O.mkdir(parents=True, exist_ok=False)
C = args.compiler.resolve()
L = args.linker.resolve()
sha = lambda raw: hashlib.sha256(raw).hexdigest()
assert sha(C.read_bytes()) == '0ccdb524ea327ab290b2e70feaac49a4789abb2fee6da8746a095f435c17af73'
assert sha(L.read_bytes()) == '55a2b0a762c8db574a65d0d0e6a18e3cedc668c01d20080a11c26cf1ccb601b3'

STEP = b'''hia-asm.v1 hepta.hia.program.v8 session_entry
exports(step)
fn make(n:usize) -> (buffer) effects(alloc,trap) external()
fn bump(state:buffer) -> (buffer,u8) effects(read,write,trap) external()
fn step(s:readbuf,o:mutbuf,state:buffer) -> (buffer,u32) effects(alloc,read,write,trap) entry(b0)
block b0(s:readbuf,o:mutbuf,state:buffer)
  length:usize = buffer.len(state)
  zero:usize = const() {"value":0}
  empty:bool = compare.eq(length,zero)
  branch empty fresh(o,state) existing(o,state)
block fresh(of:mutbuf,old:buffer)
  _ = owner.drop(old)
  size:usize = const() {"value":8}
  made:buffer = call(size) {"function":"make"}
  jump apply(of,made)
block existing(oe:mutbuf,kept:buffer)
  jump apply(oe,kept)
block apply(out:mutbuf,current:buffer)
  next:buffer,value:u8 = call(current) {"function":"bump"}
  zbyte:u8 = const() {"value":0}
  _ = buffer.fill(out,zbyte)
  z:usize = const() {"value":0}
  _ = buffer.store(out,z,value)
  count:u32 = const() {"value":32}
  ret(next,count)
end
'''
OWNERS = b'''hia-asm.v1 hepta.hia.program.v8 session_owners
exports(make,bump,close)
fn make(n:usize) -> (buffer) effects(alloc,trap) entry(b0)
block b0(n:usize)
  state:buffer = buffer.new(n)
  ret(state)
end
fn bump(state:buffer) -> (buffer,u8) effects(read,write,trap) entry(b0)
block b0(state:buffer)
  loan:mutbuf = borrow.mut(state)
  z:usize = const() {"value":0}
  prior:u8 = buffer.load(loan,z)
  one:u8 = const() {"value":1}
  next:u8 = int.add.wrap(prior,one)
  _ = buffer.store(loan,z,next)
  _ = loan.end(loan)
  ret(state,next)
end
fn close(state:buffer) -> () effects(alloc,trap) entry(b0)
block b0(state:buffer)
  _ = owner.drop(state)
  ret()
end
'''


def run(exe, raw):
    return subprocess.run([str(exe)], input=raw, capture_output=True, env={'PATH': ''}, timeout=10)


def compile(name, raw):
    (O / (name + '.hepta')).write_bytes(raw)
    result = run(C, raw)
    assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF'), (name, result)
    (O / (name + '.o')).write_bytes(result.stdout)
    return result.stdout


def frame(objects, name=b'step', magic=b'HIAOWNS1'):
    return magic + struct.pack('<3Q', len(objects), len(name), 0) + name + b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)


def u64(raw, offset):
    return struct.unpack_from('<Q', raw, offset)[0]


def signatures(raw):
    table = u64(raw, 40)
    symtab = u64(raw, table + 128 + 24)
    strings = u64(raw, table + 192 + 24)
    metadata = u64(raw, table + 448 + 24)
    result = {}
    for record in range(u64(raw, metadata + 16)):
        at = metadata + 64 + record * 256
        ordinal = u64(raw, at)
        string = strings + struct.unpack_from('<I', raw, symtab + ordinal * 24)[0]
        end = raw.index(0, string)
        result[raw[string:end].decode()] = at
    return result


objects = [compile('step', STEP), compile('owners', OWNERS)]
rows = []
for reverse in (False, True):
    request = frame(objects[::-1] if reverse else objects)
    linked = run(L, request)
    assert linked.returncode == 0 and not linked.stderr and linked.stdout
    executable = O / ('application-reversed' if reverse else 'application')
    executable.write_bytes(linked.stdout)
    executable.chmod(0o700)
    (O / ('link-reversed.request' if reverse else 'link.request')).write_bytes(request)
    child = subprocess.Popen([str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={'PATH': ''})

    def exact(count):
        result = bytearray()
        selector = selectors.DefaultSelector()
        selector.register(child.stdout, selectors.EVENT_READ)
        try:
            while len(result) < count:
                assert selector.select(3), ('response not live before EOF', reverse)
                part = os.read(child.stdout.fileno(), count - len(result))
                assert part, ('unexpected EOF', child.poll(), child.stderr.read())
                result.extend(part)
        finally:
            selector.close()
        return bytes(result)

    observed = []
    for expected, body in enumerate((b'first', b'', b'third'), 1):
        child.stdin.write(b'HTOWNR01' + struct.pack('<Q', len(body)) + body)
        child.stdin.flush()
        header = exact(16)
        assert header == b'HTOWNS01' + struct.pack('<Q', 32)
        output = exact(32)
        assert output == bytes([expected]) + bytes(31), (reverse, expected, output.hex())
        observed.append(output.hex())
    child.stdin.close()
    assert child.wait(timeout=5) == 0 and not child.stdout.read() and not child.stderr.read()
    rows.append({'case': 'live-owner-retained-three-frames' + ('-reversed-objects' if reverse else ''),
                 'kind': 'native-execution', 'status': 'PASS', 'responses_hex': observed,
                 'reply_before_next_frame_and_stdin_eof': True,
                 'executable_sha256': sha(linked.stdout), 'request_sha256': sha(request)})


def refuse(name, request):
    result = run(L, request)
    (O / (name + '.request')).write_bytes(request)
    (O / (name + '.stdout')).write_bytes(result.stdout)
    (O / (name + '.stderr')).write_bytes(result.stderr)
    assert result.returncode == 2 and not result.stdout and not result.stderr, (name, result.returncode)
    rows.append({'case': name, 'kind': 'native-link-refusal', 'status': 'PASS',
                 'request_sha256': sha(request), 'exit': result.returncode})


for entry in (b'entry', b'close', b'Step', b'step\0', b'', b'steps', b'make'):
    refuse('wrong-entry-' + entry.hex(), frame(objects, name=entry))
for magic in (b'HIAOWN02', b'HIAOWNC2', b'HIABLK01', b'HIAOWNT1'):
    refuse('wrong-profile-' + magic.decode(), frame(objects, magic=magic))
no_close = compile('close-private', OWNERS.replace(b'exports(make,bump,close)', b'exports(make,bump)'))
refuse('close-must-be-public-definition', frame([objects[0], no_close]))
renamed = compile('close-renamed', OWNERS.replace(b'close', b'finish'))
refuse('close-exact-name', frame([objects[0], renamed]))
refuse('duplicate-owner-provider', frame([objects[0], objects[1], objects[1]]))
refuse('no-owner-provider', frame([objects[0]]))

changes = {
    'step': [(16, [0, 1, 2, 4, 17, (1 << 64) - 1]), (24, [0, 1, 3, 4]),
             (32, [0, 1, 7, 9, 11, 14, 16]), (40, [0, 3, 6, 7, 9, 16]),
             (176, [0, 1, 2, 8, 9, 16]), (184, [3, 8]),
             (48, [1, 7, 8]), (56, [1, 6, 8]), (64, [1, 6, 7, 9])],
    'close': [(16, [0, 2, 17]), (24, [1, 2, 3]), (32, [0, 1, 8, 10, 11, 15, 16]),
              (48, [1, 6, 7, 9]), (40, [3, 8]), (176, [3, 8]), (184, [3, 8])],
}
for number, name in ((0, 'step'), (1, 'close')):
    at = signatures(objects[number])[name]
    for offset, values in changes[name]:
        for value in values:
            altered = bytearray(objects[number])
            struct.pack_into('<Q', altered, at + offset, value)
            current = objects.copy()
            current[number] = bytes(altered)
            refuse(name + '-field-' + str(offset) + '-value-' + str(value), frame(current))
    for offset in range(192, 256, 8):
        altered = bytearray(objects[number])
        struct.pack_into('<Q', altered, at + offset, 1)
        current = objects.copy()
        current[number] = bytes(altered)
        refuse(name + '-reserved-' + str(offset), frame(current))

valid = frame(objects)
for cut in (0, 7, 31, 32, 35, len(valid) - 1):
    refuse('truncated-' + str(cut), valid[:cut])
report = {'schema': 'heptatorch.owned-session-native-entry-review.v1', 'status': 'PASS',
          'compiler_sha256': sha(C.read_bytes()), 'linker_sha256': sha(L.read_bytes()),
          'reviewer_sha256': sha(Path(__file__).read_bytes()), 'cases': len(rows),
          'object_sha256': [sha(obj) for obj in objects], 'rows': rows,
          'owner_resource_operations_are_real_native_hia': True,
          'host_owner_runtime': False, 'production_storage_service_claim': False}
(O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, indent=2))
