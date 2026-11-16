"""Execute real bool mutation and the exact scalar-byte HIA helper.

This is a native ownership/byte-contract control. Torch detach26 supplies a
separate independent replay; this script does not synthesize Torch results.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import subprocess
import traceback


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def create_frame(shape, raw):
    return (b'HTORCH03' + struct.pack('<3Q', 1, 160 + len(raw), 0)
            + struct.pack('<8Q', 97, 1, 96, 0, 0, 0, 0, 0)
            + struct.pack('<8Q', len(shape), 0, *shape, *([0] * (4-len(shape))), 160, len(raw)) + raw)


ENTRY = '''fn entry(v0:readbuf,v1:mutbuf,v2:mutbuf) -> (u32) effects(read,write,trap) entry(b0)
block b0(v0:readbuf,v1:mutbuf,v2:mutbuf)
  v3:usize = buffer.len(v0)
  v4:usize = buffer.len(v1)
  v5:bool = compare.gt(v3,v4)
  branch v5 b1() b2(v0,v1,v3)
block b1()
  v6:u32 = const() {"value":4294967295}
  ret(v6)
block b2(v7:readbuf,v8:mutbuf,v9:usize)
  v10:usize = const() {"value":0}
  jump b3(v7,v8,v9,v10)
block b3(v11:readbuf,v12:mutbuf,v13:usize,v14:usize)
  v15:bool = compare.lt(v14,v13)
  branch v15 b4(v11,v12,v13,v14) b5(v13)
block b4(v16:readbuf,v17:mutbuf,v18:usize,v19:usize)
  v20:u8 = buffer.load(v16,v19)
  v21:u64 = int.cast(v20) {"mode":"wrap"}
  v22:u64 = call(v21) {"function":"bool_logical_not_byte"}
  v23:u8 = int.cast(v22) {"mode":"wrap"}
  _ = buffer.store(v17,v19,v23)
  v24:usize = const() {"value":1}
  v25:usize = int.add.wrap(v19,v24)
  jump b3(v16,v17,v18,v25)
block b5(v26:usize)
  v27:u32 = int.cast(v26) {"mode":"wrap"}
  ret(v27)
end
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'protocol', 'flat-compiler', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {p.resolve(): p.read_bytes() for p in (a.binary, a.source, a.protocol, a.flat_compiler, Path(__file__))}
    (out/'executed-test.py').write_bytes(inputs[Path(__file__).resolve()])
    report = {'schema': 'heptatorch.bool-logical-not-native-control.v1', 'status': 'RUNNING',
              'source_sha256': sha(inputs[a.source.resolve()]), 'native_sha256': sha(inputs[a.binary.resolve()]),
              'inputs_sha256': {str(p): sha(raw) for p, raw in inputs.items()},
              'native_executed': True, 'Torch_executed_here': False, 'rollback_bytes': 0, 'checks': []}
    s = None
    try:
        assert sha(inputs[a.flat_compiler.resolve()]) == 'c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6'
        pattern = rb'^(fn bool_logical_not_byte\([^\n]*\n.*?^end\n)'
        helper = re.findall(pattern, inputs[a.source.resolve()], re.M | re.S)
        assert len(helper) == 1
        unit = b'hia-asm.v1 hepta.hia.program.v8 selfhost_exec\n' + ENTRY.encode() + helper[0]
        (out/'exact-helper.hepta').write_bytes(unit)
        compiled = subprocess.run([str(a.flat_compiler.resolve())], input=unit, capture_output=True, env={'PATH': ''}, timeout=120)
        (out/'compiler.stderr').write_bytes(compiled.stderr)
        assert compiled.returncode == 0 and not compiled.stderr and compiled.stdout.startswith(b'\x7fELF')
        exe = out/'exact-helper'
        exe.write_bytes(compiled.stdout)
        exe.chmod(0o500)
        request = bytes(range(256))
        expected = b'\x01' + bytes(255)
        result = subprocess.run([str(exe)], input=request, capture_output=True, env={'PATH': ''}, timeout=20)
        for name, raw in [('request.bin', request), ('expected.bin', expected), ('response.bin', result.stdout), ('execution.stderr', result.stderr)]:
            (out/name).write_bytes(raw)
        assert result.returncode == 0 and not result.stderr and result.stdout == expected
        report['scalar_byte_helper'] = {'status': 'PASS', 'input_bytes': 256, 'exact_output_bytes': 256,
            'helper_hia_sha256': sha(helper[0]), 'harness_native_sha256': sha(compiled.stdout),
            'scope': 'Exact production scalar helper maps any nonzero byte to zero; public constructor still admits only canonical bool bytes.'}
        spec = importlib.util.spec_from_file_location('logical_not_wire', a.protocol)
        w = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(w)
        s = w.Session(a.binary.resolve(), directory=out/'raw')
        def one(*args):
            return s.run([args])[0]
        def check(name, got, expected):
            assert got == expected, (name, got, expected)
            report['checks'].append(name)
        def meta(t):
            return w.meta(one(13, t))
        def raw(t, count):
            return one(98, t, 0, count)['payload']
        def create(shape, data):
            status, rows = w.parse(s.exchange(create_frame(shape, data)))
            assert status == 0 and len(rows) == 1
            return rows[0]['value']
        def error(commands, expected):
            before = s.owner()
            status, records = w.parse(s.exchange(w.packet(commands)))
            after = s.owner()
            check('refusal status', status, expected)
            check('no published records', records, [])
            check('rollback owner identity', (after['pointer'], after['length']), (before['pointer'], before['length']))
            check('rollback complete bytes', sha(after['bytes']), sha(before['bytes']))
            report['rollback_bytes'] += before['length']
        generator = one(60, 1729)['value']
        state = one(63, generator)['payload']
        data = bytes([0, 1, 1, 0, 1, 0])
        x = create([2, 3], data)
        alias = one(19, x)['value']
        view = one(6, x, 0, 1)['value']
        saved = one(46, view)['value']
        before = meta(x)
        check('returns same real Tensor', one(102, view)['value'], view)
        check('one-byte storage inverted', raw(x, 6), bytes([1, 0, 0, 1, 0, 1]))
        check('transpose logical order', raw(view, 6), bytes([1, 1, 0, 0, 0, 1]))
        check('alias reads same bytes', raw(alias, 6), raw(x, 6))
        after = meta(x)
        for key in ('storage_id', 'version_id', 'storage_bytes', 'storage_offset'):
            check('stable ' + key, after[key], before[key])
        check('single shared Version increment', after['version'], before['version'] + 1)
        error([(47, saved)], 6)
        check('second inverse same Tensor', one(102, alias)['value'], alias)
        check('second inverse restores data', raw(x, 6), data)
        check('second inverse shared Version', meta(view)['version'], before['version'] + 2)
        error([(102, x), (999,)], 4)
        for command, status in [((102, x, 1), 7), ((102, x, 0, 1), 7), ((102, x, 0, 0, 0, 1), 7),
                                ((102, generator), 1)]:
            error([command], status)
        f32 = one(*w.full([6], 1))['value']
        error([(102, f32)], 7)
        scalar = create([], b'\x00')
        check('scalar byte false to true', one(102, scalar)['value'], scalar)
        check('scalar value', raw(scalar, 1), b'\x01')
        empty = create([2, 0, 3], b'')
        empty_alias = one(19, empty)['value']
        before_empty = meta(empty)
        one(102, empty_alias)
        check('empty still bumps actual Version', meta(empty)['version'], before_empty['version'] + 1)
        check('empty uses zero-byte storage', meta(empty)['storage_bytes'], 0)
        cfg = struct.pack('<10Q', 2, 3, 2, 0, 0, 0, 1, 0, 0, 0)
        expanded = one(69, x, w.Payload(cfg), 80)['value']
        error([(102, expanded)], 2)
        cfg = struct.pack('<10Q', 2, 2, 2, 0, 0, 3, 1, 0, 0, 0)
        hole_view = one(69, x, w.Payload(cfg), 80)['value']
        one(102, hole_view)
        check('strided holes untouched', raw(x, 6), bytes([1, 0, 1, 1, 0, 0]))
        # A mask is an actual bool Tensor retained by SavedVariable.
        p = one(*w.full([6], .25, 1))['value']
        nd = one(99, p, generator, 0x3fe0000000000000, 1)
        y, mask = nd['value'], struct.unpack('<Q', nd['payload'])[0]
        old_mask = raw(mask, 6)
        current_state = one(63, generator)['payload']
        one(102, mask)
        check('Saved actual bool mask inverted', raw(mask, 6), bytes(1 if b == 0 else 0 for b in old_mask))
        check('logical operation no RNG consumption', one(63, generator)['payload'], current_state)
        seed = one(*w.full([6], 1))['value']
        check('Saved mutation fails real backward', one(58, y, seed, 0)['value'], 6)
        error([(96, y, 1)], 6)
        check('unreached leaf gradient stays None', one(59, p)['value'], 0)
        check('full object audit', one(53)['value'], 1)
        for t in (view, alias, x, f32, scalar, empty_alias, empty, expanded, hole_view, y, mask, seed, p):
            one(41, t)
        one(48, saved)
        one(66, generator)
        check('final audit', one(53)['value'], 1)
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
    finally:
        if s is not None:
            report['frames'] = s.frames
            try:
                report['transport'] = s.finish()
            except BaseException as error:
                report.update(status='FAIL', transport_error=repr(error))
        assert all(p.read_bytes() == raw for p, raw in inputs.items())
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'frames', 'rollback_bytes', 'error')}))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
