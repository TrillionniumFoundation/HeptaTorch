#!/usr/bin/env python3
"""Execute the real HIA probability helpers against exact integer rounding.

The extra entry is qualification-only, linked to byte-identical production
objects. No replacement helper, numerical stub or production opcode is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import struct
import subprocess

SHA = lambda raw: hashlib.sha256(raw).hexdigest()
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = '0c5176e3aa968631ae592b887c5dc3cf1e8e3f48f73593f076e9d7965ffacfd7'
ENTRY = '''fn probability_probe(s:readbuf,o:mutbuf,w:mutbuf) -> (u32) effects(read,write,trap) entry(start)
block start(s:readbuf,o:mutbuf,w:mutbuf)
  n:usize = buffer.len(s)
  cap:usize = buffer.len(o)
  eight:usize = const() {"value":8}
  thirtytwo:usize = const() {"value":32}
  count:usize = int.div.checked(n,eight)
  room:usize = int.div.checked(cap,thirtytwo)
  fits:bool = compare.le(count,room)
  branch fits begin(s,o,count) reject()
block reject()
  zero:u32 = const() {"value":0}
  ret(zero)
block begin(s0:readbuf,o0:mutbuf,c0:usize)
  count0:u64 = int.cast(c0) {"mode":"wrap"}
  index0:u64 = const() {"value":0}
  jump loop(s0,o0,count0,index0)
block loop(s1:readbuf,o1:mutbuf,count1:u64,index1:u64)
  more:bool = compare.lt(index1,count1)
  branch more read(s1,o1,count1,index1) done(o1,count1)
block read(s2:readbuf,o2:mutbuf,count2:u64,index2:u64)
  eight2:u64 = const() {"value":8}
  off2:u64 = int.mul.wrap(index2,eight2)
  p2:u64 = call(s2,off2) {"function":"getr"}
  valid2:u64 = call(p2) {"function":"dist_prob64_valid"}
  zero2:u64 = const() {"value":0}
  validbool2:bool = compare.ne(valid2,zero2)
  branch validbool2 compute(s2,o2,count2,index2,p2,valid2) write(s2,o2,count2,index2,valid2,zero2,zero2,zero2)
block compute(s3:readbuf,o3:mutbuf,count3:u64,index3:u64,p3:u64,valid3:u64)
  q3:u64 = call(p3) {"function":"dist_complement64"}
  zero3:u64 = const() {"value":0}
  positive3:bool = compare.ne(q3,zero3)
  branch positive3 scale(s3,o3,count3,index3,valid3,q3) write(s3,o3,count3,index3,valid3,q3,zero3,zero3)
block scale(s4:readbuf,o4:mutbuf,count4:u64,index4:u64,valid4:u64,q4:u64)
  f4:u64 = call(q4) {"function":"dist_probability_to32"}
  one4:u64 = const() {"value":1065353216}
  scale4:u64 = call(one4,f4) {"function":"divf"}
  jump write(s4,o4,count4,index4,valid4,q4,f4,scale4)
block write(s5:readbuf,o5:mutbuf,count5:u64,index5:u64,valid5:u64,q5:u64,f5:u64,scale5:u64)
  thirtytwo5:u64 = const() {"value":32}
  at5:u64 = int.mul.wrap(index5,thirtytwo5)
  write_valid5:u64 = call(o5,at5,valid5) {"function":"putw"}
  eight5:u64 = const() {"value":8}
  atq5:u64 = int.add.wrap(at5,eight5)
  write_q5:u64 = call(o5,atq5,q5) {"function":"putw"}
  atf5:u64 = int.add.wrap(atq5,eight5)
  write_f5:u64 = call(o5,atf5,f5) {"function":"putw"}
  ats5:u64 = int.add.wrap(atf5,eight5)
  write_scale5:u64 = call(o5,ats5,scale5) {"function":"putw"}
  one5:u64 = const() {"value":1}
  next5:u64 = int.add.wrap(index5,one5)
  jump loop(s5,o5,count5,next5)
block done(o6:mutbuf,count6:u64)
  thirtytwo6:u64 = const() {"value":32}
  size6:u64 = int.mul.wrap(count6,thirtytwo6)
  result6:u32 = int.cast(size6) {"mode":"wrap"}
  ret(result6)
end
'''


def rounded_quotient(n, d):
    q, remainder = divmod(n, d)
    return q + (2 * remainder > d or (2 * remainder == d and q & 1))


def rational_bits(n, d, precision, bias):
    if not n:
        return 0
    exponent = n.bit_length() - d.bit_length()
    if (n < d << exponent) if exponent >= 0 else (n << -exponent < d):
        exponent -= 1
    shift = precision - 1 - exponent
    m = rounded_quotient(n << shift, d) if shift >= 0 else rounded_quotient(n, d << -shift)
    if m == 1 << precision:
        exponent += 1
        m >>= 1
    assert 1 <= exponent + bias < 2 * bias + 1
    return ((exponent + bias) << (precision - 1)) | (m - (1 << (precision - 1)))


def expected(bits):
    magnitude = bits & ((1 << 63) - 1)
    if magnitude > 0x3ff0000000000000 or (bits >> 63 and magnitude):
        return (0, 0, 0, 0)
    e = magnitude >> 52
    m = magnitude & ((1 << 52) - 1)
    if e:
        m |= 1 << 52
    power = (e - 1075) if e else -1074
    denominator = 1 << -power
    q = rational_bits(denominator - m, denominator, 53, 1023)
    if not q:
        return (1, 0, 0, 0)
    qe, qm = (q >> 52) - 1075, (q & ((1 << 52) - 1)) | (1 << 52)
    f = rational_bits(qm, 1 << -qe, 24, 127)
    fe, fm = (f >> 23) - 150, (f & ((1 << 23) - 1)) | (1 << 23)
    scale = rational_bits(1 << -fe, fm, 24, 127)
    return (1, q, f, scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'stack-build', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    build_dir = a.build.resolve()
    build_raw = (build_dir / 'build.json').read_bytes()
    build = json.loads(build_raw)
    compiler = a.stack_build.resolve() / 'native/stack-module-compiler'
    linker = a.stack_build.resolve() / 'native/stack-linker-runtime'
    assert SHA(compiler.read_bytes()) == COMPILER and SHA(linker.read_bytes()) == LINKER
    inputs = {compiler: compiler.read_bytes(), linker: linker.read_bytes(), build_dir / 'build.json': build_raw,
              Path(__file__).resolve(): Path(__file__).read_bytes()}
    core = (build_dir / 'runtime.hepta').read_bytes()
    assert SHA(core) == build['runtime_source_sha256']
    inputs[build_dir / 'runtime.hepta'] = core
    function = re.compile(rb'^(fn (\w+)\([^\n]* entry\(\w+\)\n.*?^end\n)', re.M | re.S)
    bodies = {name.decode(): body for body, name in function.findall(core)}
    source = b'hia-asm.v1 hepta.hia.program.v8 distribution_probability_probe\nexports(probability_probe)\n'
    for name in ('getr', 'putw', 'dist_prob64_valid', 'dist_complement64', 'dist_probability_to32', 'divf'):
        source += re.sub(rb' entry\(\w+\)$', b' external()', bodies[name].split(b'\n', 1)[0]) + b'\n'
    source += ENTRY.encode()
    (out / 'entry.hepta').write_bytes(source)
    result = subprocess.run([str(compiler)], input=source, capture_output=True, env={'PATH': ''}, timeout=30)
    (out / 'entry.o').write_bytes(result.stdout)
    (out / 'compile.stderr').write_bytes(result.stderr)
    assert result.returncode == 0 and not result.stderr, (result.returncode, result.stderr)
    objects = []
    for unit in build['units']:
        if unit['name'] == 'owners':
            continue
        path = build_dir / (unit['name'] + '.o')
        obj = path.read_bytes()
        assert SHA(obj) == unit['object_sha256']
        inputs[path] = obj
        objects.append(obj)
    objects.append(result.stdout)
    symbol = b'probability_probe'
    request = b'HIASTKR1' + struct.pack('<3Q', len(objects), len(symbol), 0) + symbol
    request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
    (out / 'link.request').write_bytes(request)
    result = subprocess.run([str(linker)], input=request, capture_output=True, env={'PATH': ''}, timeout=60)
    (out / 'link.stderr').write_bytes(result.stderr)
    assert result.returncode == 0 and not result.stderr, (result.returncode, result.stderr)
    exe = out / 'probability-probe'
    exe.write_bytes(result.stdout)
    exe.chmod(0o500)
    candidates = {0, 1, 1 << 63, 0x3ff0000000000000, 0x3ff0000000000001,
                  0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000000, 0x7ff0000000000001}
    for exponent in range(1024):
        for fraction in (0, 1, (1 << 51) - 1, 1 << 51, (1 << 52) - 2, (1 << 52) - 1):
            value = (exponent << 52) | fraction
            candidates.add(value)
            candidates.add(value | (1 << 63))
    for base in (0x3c90000000000000, 0x3fe0000000000000, 0x3ff0000000000000):
        candidates.update(range(base - 2048, base + 2048))
    rng = random.Random(0x4245524e)
    candidates.update(rng.randrange(0x3ff0000000000001) for _ in range(65536))
    candidates.update((0x3fe << 52) | rng.getrandbits(52) for _ in range(16384))
    candidates = sorted(candidates)
    rows = []
    corpus = b''.join(struct.pack('<Q', value) for value in candidates)
    (out / 'inputs.bin').write_bytes(corpus)
    observations = bytearray()
    for start in range(0, len(candidates), 30000):
        part = candidates[start:start + 30000]
        result = subprocess.run([str(exe)], input=corpus[start * 8:(start + len(part)) * 8],
                                capture_output=True, env={'PATH': ''}, timeout=60)
        assert result.returncode == 0 and not result.stderr and len(result.stdout) == len(part) * 32
        observations.extend(result.stdout)
        for index, value in enumerate(part):
            actual = struct.unpack_from('<4Q', result.stdout, index * 32)
            answer = expected(value)
            assert actual == answer, (hex(value), actual, answer)
        rows.append({'start': start, 'count': len(part), 'returncode': result.returncode,
                     'stdout_sha256': SHA(result.stdout)})
    (out / 'observations.bin').write_bytes(observations)
    for path, data in inputs.items():
        assert path.read_bytes() == data, path
    report = {'schema': 'heptatorch.distribution-probability-integer-qualification.v1', 'status': 'PASS',
              'cases': len(candidates), 'native_u64_results': 4 * len(candidates),
              'expected_method': 'independent arbitrary-precision exact rational nearest-even quotient, binary64 complement then binary32 divisor then binary32 reciprocal',
              'probe_executable_sha256': SHA(exe.read_bytes()), 'probe_source_sha256': SHA(source),
              'runtime_source_sha256': build['runtime_source_sha256'],
              'production_executable_sha256': build['executable_sha256'],
              'production_objects_byte_identical': True,
              'qualification_only_extra_entry': True, 'input_custody_after': True,
              'inputs': {str(path): SHA(data) for path, data in inputs.items()}, 'batches': rows,
              'corpus_sha256': SHA(corpus), 'observations_sha256': SHA(observations)}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'cases', 'native_u64_results', 'probe_executable_sha256')}, indent=2))


if __name__ == '__main__':
    main()
