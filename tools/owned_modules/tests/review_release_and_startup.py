from pathlib import Path
import hashlib
import json
import resource
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
OLD = args.borrowed_tools.resolve() / 'module-compiler'
sha = lambda raw: hashlib.sha256(raw).hexdigest()
rows = []


def limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    resource.setrlimit(resource.RLIMIT_AS, (64 << 20, 64 << 20))


def observe(name, source, expected, *, abi=2, capacity=1048576, input_bytes=b'', expected_exit=0):
    raw = source.encode()
    (O / (name + '.hepta')).write_bytes(raw)
    compiler = C if abi == 2 else OLD
    obj = subprocess.run([str(compiler)], input=raw, capture_output=True, env={'PATH': ''}, timeout=30)
    assert obj.returncode == 0 and not obj.stderr, (name, 'compile', obj.returncode, obj.stderr)
    (O / (name + '.o')).write_bytes(obj.stdout)
    label = 'abi' + str(abi) + '-w' + str(capacity // 1048576) + 'm'
    linker = BUILD / ('linker-' + label)
    magic = {(1, 2097152): b'HIACMP01', (2, 2097152): b'HIAOWNC2', (2, 1048576): b'HIAOWN02'}[(abi, capacity)]
    request = magic + struct.pack('<3Q', 1, 5, 0) + b'entry' + struct.pack('<Q', len(obj.stdout)) + obj.stdout
    (O / (name + '.request')).write_bytes(request)
    linked = subprocess.run([str(linker)], input=request, capture_output=True, env={'PATH': ''}, timeout=30)
    assert linked.returncode == 0 and not linked.stderr, (name, 'link', linked.returncode, linked.stderr)
    exe = O / (name + '.native')
    exe.write_bytes(linked.stdout)
    exe.chmod(0o700)
    result = subprocess.run([str(exe)], input=input_bytes, capture_output=True, env={'PATH': ''}, timeout=20, preexec_fn=limits)
    (O / (name + '.stdout')).write_bytes(result.stdout)
    (O / (name + '.stderr')).write_bytes(result.stderr)
    assert result.returncode == expected_exit and not result.stderr and result.stdout == expected, (
        name, result.returncode, result.stderr, result.stdout.hex(), expected.hex())
    rows.append({'case': name, 'status': 'PASS', 'source_sha256': sha(raw),
                 'compiler_sha256': sha(compiler.read_bytes()), 'linker_sha256': sha(linker.read_bytes()),
                 'executable_sha256': sha(linked.stdout), 'interface_abi': abi,
                 'request_magic': magic.decode(), 'entry_workspace_bytes': capacity,
                 'input_hex': input_bytes.hex(), 'observed_hex': result.stdout.hex(),
                 'exit': result.returncode, 'address_space_limit': 64 << 20})


def chain_loop(nodes, payload):
    return '''hia-asm.v1 hepta.hia.program.v8 chain_release
exports(entry)
fn entry(s:readbuf,o:mutbuf,w:mutbuf) -> (u32) effects(alloc,read,write,trap) entry(b0)
block b0(s:readbuf,o:mutbuf,w:mutbuf)
  initial:u64 = const() {"value":0}
  jump b1(initial,o)
block b1(cycle:u64,out:mutbuf)
  empty:chain = chain.empty()
  initial_count:u64 = const() {"value":0}
  jump b2(cycle,initial_count,empty,out)
block b2(cycle2:u64,count:u64,c:chain,out2:mutbuf)
  size:usize = const() {"value":PAYLOAD}
  b:buffer = buffer.new(size)
  next:chain,left:buffer,ok:bool = chain.try_push(c,b)
  _ = owner.drop(left)
  one:u64 = const() {"value":1}
  next_count:u64 = int.add.wrap(count,one)
  branch ok b3(cycle2,next_count,next,out2) b6(next,out2)
block b3(cycle3:u64,count3:u64,c3:chain,out3:mutbuf)
  limit:u64 = const() {"value":NODES}
  more:bool = compare.lt(count3,limit)
  branch more b2(cycle3,count3,c3,out3) b4(cycle3,c3,out3)
block b4(cycle4:u64,c4:chain,out4:mutbuf)
  _ = owner.drop(c4)
  increment:u64 = const() {"value":1}
  next_cycle:u64 = int.add.wrap(cycle4,increment)
  cycles:u64 = const() {"value":8}
  repeat:bool = compare.lt(next_cycle,cycles)
  branch repeat b1(next_cycle,out4) b5(out4)
block b5(out5:mutbuf)
  zero:usize = const() {"value":0}
  passed:u8 = const() {"value":173}
  _ = buffer.store(out5,zero,passed)
  length:u32 = const() {"value":1}
  ret(length)
block b6(c6:chain,out6:mutbuf)
  _ = owner.drop(c6)
  zero6:usize = const() {"value":0}
  failed:u8 = const() {"value":0}
  _ = buffer.store(out6,zero6,failed)
  length6:u32 = const() {"value":1}
  ret(length6)
end
'''.replace('PAYLOAD', str(payload)).replace('NODES', str(nodes))


observe('chain-payload-reclamation-8x24MiB-under64MiB', chain_loop(24, 1 << 20), b'\xad')
observe('chain-node-reclamation-8x4096-under64MiB', chain_loop(4096, 0), b'\xad')


def startup_probe(index=2097151):
    body = '''hia-asm.v1 hepta.hia.program.v8 startup_probe
exports(entry)
fn entry(s:readbuf,o:mutbuf,w:mutbuf) -> (u32) effects(read,write,trap) entry(b0)
block b0(s:readbuf,o:mutbuf,w:mutbuf)
  z:usize = const() {"value":0}
  mark:u8 = const() {"value":85}
  _ = buffer.store(o,z,mark)
  start:u8 = const() {"value":161}
  _ = buffer.store(w,z,start)
  endpoint:usize = const() {"value":INDEX}
  end:u8 = const() {"value":178}
  _ = buffer.store(w,endpoint,end)
  original_output:u8 = buffer.load(o,z)
  original_input:u8 = buffer.load(s,z)
  first:u8 = buffer.load(w,z)
  last:u8 = buffer.load(w,endpoint)
  one:usize = const() {"value":1}
  two:usize = const() {"value":2}
  three:usize = const() {"value":3}
  _ = buffer.store(o,z,original_output)
  _ = buffer.store(o,one,original_input)
  _ = buffer.store(o,two,first)
  _ = buffer.store(o,three,last)
  length:usize = buffer.len(w)
  wide:u64 = int.cast(length) {"mode":"wrap"}
'''.replace('INDEX', str(index))
    for i in range(8):
        body += f'  shift{i}:u64 = const() {{"value":{8*i}}}\n  part{i}:u64 = int.shr.checked(wide,shift{i})\n'
        body += f'  byte{i}:u8 = int.cast(part{i}) {{"mode":"wrap"}}\n  at{i}:usize = const() {{"value":{4+i}}}\n  _ = buffer.store(o,at{i},byte{i})\n'
    return body + '  count:u32 = const() {"value":12}\n  ret(count)\nend\n'


expected = b'\x55\x5a\xa1\xb2' + struct.pack('<Q', 2097152)
for abi in (1, 2):
    observe('abi' + str(abi) + '-compiler-workspace-end-and-disjoint-buffers', startup_probe(), expected,
            abi=abi, capacity=2097152, input_bytes=b'\x5a')
    observe('abi' + str(abi) + '-compiler-workspace-one-past-end-traps', startup_probe(2097152), b'',
            abi=abi, capacity=2097152, input_bytes=b'\x5a', expected_exit=-4)
observe('runtime-workspace-remains-exactly-1MiB', startup_probe(1048575),
        b'\x55\x5a\xa1\xb2' + struct.pack('<Q', 1048576), input_bytes=b'\x5a')
observe('runtime-does-not-silently-gain-compiler-workspace', startup_probe(), b'', input_bytes=b'\x5a', expected_exit=-4)
report = {'schema': 'heptatorch.owned-release-and-startup-review.v1', 'status': 'PASS',
          'cases': len(rows), 'rows': rows, 'native_execution': True, 'libc_target_dependency': False,
          'qualification_scope': 'Actual mapping reclamation under RLIMIT_AS; two explicit compiler startup identities, buffer separation and exact bounds.'}
(O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, indent=2))
