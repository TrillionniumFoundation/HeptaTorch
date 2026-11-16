"""Real RLIMIT_AS rejection, committed-owner preservation, and native recovery.

No allocator response is mocked. The child executes only the immutable native
image. Host code supplies packets and reads its own child's committed memory.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import traceback


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'protocol', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {p.resolve(): p.read_bytes() for p in (a.binary, a.source, a.protocol, Path(__file__))}
    (out/'executed-test.py').write_bytes(inputs[Path(__file__).resolve()])
    spec = importlib.util.spec_from_file_location('dtype_allocation_wire', a.protocol)
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    address_limit = 32 << 20
    s = w.Session(a.binary.resolve(), directory=out/'raw', address_limit=address_limit)
    report = {'schema': 'heptatorch.dtype-allocation-failure-native-controls.v1', 'status': 'RUNNING',
        'source_sha256': sha(inputs[a.source.resolve()]), 'native_sha256': sha(inputs[a.binary.resolve()]),
        'inputs_sha256': {str(p): sha(raw) for p, raw in inputs.items()},
        'native_executed': True, 'Torch_executed_here': False, 'RLIMIT_AS_bytes': address_limit,
        'steps': [], 'rollback_bytes': 0, 'checks': 0}
    fillers = []
    def one(*args):
        return s.run([args])[0]
    def check(name, got, expected):
        assert got == expected, (name, got, expected)
        report['checks'] += 1
    def stats():
        return struct.unpack('<12Q', one(54)['payload'])
    def filler():
        size = 262144
        config = struct.pack('<8Q', 2, 0, 2, 131072, 0, 0, 160, size)
        request = b'HTORCH03'+struct.pack('<3Q', 1, 160+size, 0)+struct.pack('<8Q', 97, 1, 96, 0, 0, 0, 0, 0)+config+bytes(size)
        status, records = w.parse(s.exchange(request))
        assert status == 0 and len(records) == 1, ('setup filler admission', status)
        fillers.append(records[0]['value'])
    def constrained_failure(name, commands):
        rng = one(63, gen)['payload']
        before = s.owner()
        maps = Path(f'/proc/{s.pid}/maps').read_bytes()
        status, records = w.parse(s.exchange(w.packet(commands)))
        after = s.owner()
        maps_after = Path(f'/proc/{s.pid}/maps').read_bytes()
        (out/(name+'-owner-before.bin')).write_bytes(before['bytes'])
        (out/(name+'-owner-after.bin')).write_bytes(after['bytes'])
        (out/(name+'-maps-before.txt')).write_bytes(maps)
        (out/(name+'-maps-after.txt')).write_bytes(maps_after)
        (out/(name+'-rng-before.bin')).write_bytes(rng)
        after_rng = one(63, gen)['payload']
        (out/(name+'-rng-after.bin')).write_bytes(after_rng)
        event = {'name': name, 'status': 'RUNNING', 'frame_status': status,
            'owner_bytes': before['length'], 'before_sha256': sha(before['bytes']),
            'after_sha256': sha(after['bytes']), 'maps_before_sha256': sha(maps),
            'maps_after_sha256': sha(maps_after), 'rng_before_sha256': sha(rng), 'rng_after_sha256': sha(after_rng)}
        report['steps'].append(event)
        check(name+' real capacity refusal', status, 3)
        check(name+' no partial public records', records, [])
        check(name+' original owner pointer/length', (after['pointer'], after['length']), (before['pointer'], before['length']))
        check(name+' complete owner unchanged', sha(after['bytes']), sha(before['bytes']))
        check(name+' mapping set unchanged', maps_after, maps)
        check(name+' full RNG unchanged', after_rng, rng)
        check(name+' committed invariant audit', one(53)['value'], 1)
        report['rollback_bytes'] += before['length']
        event['status'] = 'PASS'
    try:
        count = 65536
        x = one(*w.full([2, 32768], .25, 1))['value']
        seed = one(*w.full([2, 32768], 1))['value']
        gen = one(60, 1729)['value']
        while True:
            current = stats()
            if current[1] == (8 << 20) and current[4] < 400000:
                break
            assert current[1] <= (8 << 20) and len(fillers) < 40, ('setup capacity', current)
            filler()
        report['setup_stats'] = list(current)
        report['setup_filler_count'] = len(fillers)
        (out/'child-status-before-failure.txt').write_bytes(Path(f'/proc/{s.pid}/status').read_bytes())
        constrained_failure('forward-allocation', [(99, x, gen, 0x3fe0000000000000, 1)])
        for _ in range(4):
            one(41, fillers.pop())
        nd = one(99, x, gen, 0x3fe0000000000000, 1)
        y, mask = nd['value'], struct.unpack('<Q', nd['payload'])[0]
        check('forward recovered with same input', w.meta(one(13, x))['version'], 0)
        check('forward recovered without larger owner pool', stats()[1], 8 << 20)
        mask_raw = one(98, mask, 0, count)['payload']
        check('mask physical extent is one byte', w.meta(one(13, mask))['storage_bytes'], count)
        expected_values = b''.join(struct.pack('<I', 0 if bit == 0 else 0x3f000000) for bit in mask_raw)
        check('recovered forward reads actual byte mask', one(39, y, 0, count)['payload'], expected_values)
        # Refill remaining holes so gradient admission must request pool growth.
        while stats()[4] >= 350000:
            filler()
        constrained_failure('backward-allocation', [(58, y, seed, 0)])
        check('failed admission leaves gradient None', one(59, x)['value'], 0)
        check('failed admission keeps real Saved mask', one(96, y, 1)['value'], mask)
        for tensor in fillers:
            one(41, tensor)
        fillers.clear()
        record = one(58, y, seed, 0)
        check('gradient recovery succeeds', record['value'], 0)
        expected_gradient = b''.join(struct.pack('<I', 0 if bit == 0 else 0x40000000) for bit in mask_raw)
        check('recovered gradient uses real mask and scale', one(40, x, 0, count)['payload'], expected_gradient)
        check('real gradient materialized', one(59, x)['value'], 1)
        check('default recovery released Saved', one(58, y, seed, 0)['value'], 10)
        check('recovery audit', one(53)['value'], 1)
        one(70, x)
        for tensor in (y, mask, seed, x):
            one(41, tensor)
        one(66, gen)
        check('all resources released audit', one(53)['value'], 1)
        report.update(status='PASS', forward_elements=count, gradient_elements=count)
    except BaseException as error:
        report.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
    finally:
        report['frames'] = s.frames
        try:
            report['transport'] = s.finish()
        except BaseException as error:
            report.update(status='FAIL', transport_error=repr(error))
        assert all(p.read_bytes() == raw for p, raw in inputs.items())
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'frames', 'rollback_bytes', 'checks', 'error')}))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
