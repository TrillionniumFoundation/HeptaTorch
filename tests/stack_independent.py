"""Root-owned ABI3 admission witness; no candidate implementation imports."""
from pathlib import Path
import hashlib
import argparse
import json
import resource
import struct
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
OUT = args.output.resolve(); OUT.mkdir(parents=True, exist_ok=False)
NATIVE = args.native.resolve()
sha = lambda raw: hashlib.sha256(raw).hexdigest()
C = NATIVE/'stack-module-compiler'; L = NATIVE/'stack-linker-runtime'
assert sha(C.read_bytes()) == 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
assert sha(L.read_bytes()) == '0c5176e3aa968631ae592b887c5dc3cf1e8e3f48f73593f076e9d7965ffacfd7'
PROVIDER = b'''hia-asm.v1 hepta.hia.program.v8 root_review_provider
exports(increment)
fn increment(x:u8) -> (u8) effects() entry(b0)
block b0(x:u8)
  one:u8 = const() {"value":1}
  answer:u8 = int.add.wrap(x,one)
  ret(answer)
end
'''
CONSUMER = b'''hia-asm.v1 hepta.hia.program.v8 root_review_consumer
exports(entry)
fn increment(x:u8) -> (u8) effects() external()
fn entry(s:readbuf,o:mutbuf,w:mutbuf) -> (u32) effects(read,write,trap) entry(b0)
block b0(s:readbuf,o:mutbuf,w:mutbuf)
  zero:usize = const() {"value":0}
  value:u8 = const() {"value":41}
  remote:u8 = call(value) {"function":"increment"}
  answer:u8 = call(remote) {"function":"private_increment"}
  _ = buffer.store(o,zero,answer)
  count:u32 = const() {"value":1}
  ret(count)
end
fn private_increment(x:u8) -> (u8) effects() entry(b0)
block b0(x:u8)
  one:u8 = const() {"value":1}
  answer:u8 = int.add.wrap(x,one)
  ret(answer)
end
'''


def invoke(tool, raw):
    return subprocess.run([str(tool)], input=raw, capture_output=True,
                          env={'PATH': ''}, timeout=30)


def compile(name, source):
    (OUT/(name+'.hepta')).write_bytes(source)
    proc = invoke(C, source)
    assert proc.returncode == 0 and not proc.stderr and proc.stdout.startswith(b'\x7fELF')
    (OUT/(name+'.o')).write_bytes(proc.stdout)
    return proc.stdout


def sections(raw):
    table = struct.unpack_from('<Q', raw, 40)[0]
    width, count, name_index = struct.unpack_from('<3H', raw, 58)
    assert width == 64 and count == 9
    headers = [struct.unpack_from('<IIQQQQIIQQ', raw, table+i*width) for i in range(count)]
    names = raw[headers[name_index][4]:headers[name_index][4]+headers[name_index][5]]
    return {names[h[0]:names.index(0, h[0])].decode(): h for h in headers}


def limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_STACK, (32768, 32768))


rows = []
provider, consumer = compile('provider', PROVIDER), compile('consumer', CONSUMER)
s = sections(consumer); stack = s['.hia.stack'][4]; text_at, text_size = s['.text'][4:6]
header = struct.unpack_from('<8Q', consumer, stack)
assert consumer[stack:stack+8] == b'HIASTK03' and header[1:7] == (1, 3, 2, 64, 1, 40)
matrix = stack+64+3*64
assert consumer[matrix:matrix+3] == b'\x00\x05\x00'


def case(name, changed, accept, order=False):
    objects = [provider, changed] if order else [changed, provider]
    raw = b'HIASTKR1'+struct.pack('<3Q', 2, 5, 0)+b'entry'
    raw += b''.join(struct.pack('<Q', len(o))+o for o in objects)
    (OUT/(name+'.request')).write_bytes(raw)
    proc = invoke(L, raw)
    (OUT/(name+'.link.stdout')).write_bytes(proc.stdout)
    (OUT/(name+'.link.stderr')).write_bytes(proc.stderr)
    assert not proc.stderr
    row = {'case': name, 'request_sha256': sha(raw), 'exit': proc.returncode,
           'output_bytes': len(proc.stdout), 'output_sha256': sha(proc.stdout),
           'changed_object_bytes': sum(a != b for a, b in zip(consumer, changed)),
           'machine_text_unchanged': consumer[text_at:text_at+text_size] == changed[text_at:text_at+text_size]}
    if accept:
        assert proc.returncode == 0 and proc.stdout.startswith(b'\x7fELF')
        path = OUT/name; path.write_bytes(proc.stdout); path.chmod(0o500)
        result = subprocess.run([str(path)], input=b'', capture_output=True,
                                env={'PATH': ''}, preexec_fn=limits, timeout=30)
        assert result.returncode == 0 and result.stdout == b'+' and not result.stderr
        row.update({'execution_exit': 0, 'exact_output_hex': result.stdout.hex(), 'RLIMIT_STACK_bytes': 32768})
    else:
        assert proc.returncode == 2 and not proc.stdout
    rows.append(row | {'status': 'PASS'})


case('baseline-both-real-calls', consumer, True)
case('baseline-provider-first', consumer, True, True)
for name, offset, replacement in (
        ('deleted-real-import-edge', matrix+1, 4),
        ('deleted-real-private-edge', matrix+1, 1),
        ('invented-private-import-edge', matrix+2, 1),
        ('invented-import-private-edge', matrix, 4),
        ('deleted-both-real-edges', matrix+1, 0)):
    bad = bytearray(consumer); bad[offset] = replacement
    case(name, bytes(bad), False)
bad = bytearray(consumer)
declared_frame = stack+64+64+24
frame = struct.unpack_from('<Q', bad, declared_frame)[0]
struct.pack_into('<Q', bad, declared_frame, frame-16)
case('understated-actual-prologue-frame', bytes(bad), False)
case('restored-real-matrix', consumer, True)
report = {'schema': 'heptatorch.abi3-independent-root-review.v1', 'status': 'PASS',
          'compiler_sha256': sha(C.read_bytes()), 'linker_sha256': sha(L.read_bytes()),
          'script_sha256': sha(Path(__file__).read_bytes()), 'cases': rows,
          'object_source_hashes': {'consumer': sha(CONSUMER), 'provider': sha(PROVIDER)},
          'scope': 'Actual external/private calls remain byte-identical while adversarial metadata changes; fresh native producer and linker execution under 32 KiB process stack. No claim of arbitrary x86 effect verification.',
          'whole_pytorch_rewrite_complete': False}
(OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
