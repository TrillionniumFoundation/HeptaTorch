#!/usr/bin/env python3
"""Actual joint nullable-gradient, RNG, backward and SGD owner transactions."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import sys
import traceback

sys.dont_write_bytecode = True
SHA = lambda raw: hashlib.sha256(raw).hexdigest()
IMAGE = '5baa9cd5e5625f0e2324941486982b714dcdedfe897f5c16686a3ac8882ce242'
MANIFEST = '5f0833b00e8f3068da6b60308f9941b7319c77793d10f90df5d779f520e9437b'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('executable', 'rng-package', 'cpp-reference', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    image = a.executable.resolve()
    assert SHA(image.read_bytes()) == IMAGE
    package = a.rng_package.resolve()
    manifest = (package / 'source-pins.sha256').read_bytes()
    assert SHA(manifest) == MANIFEST
    files = {}
    for line in manifest.decode().splitlines():
        pin, name = line.split('  ', 1)
        files[package / name] = (package / name).read_bytes()
        assert SHA(files[package / name]) == pin

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    wire = load('nullable_rng_wire', package / 'protocol.py')
    mt = load('nullable_rng_reference', package / 'reference.py')
    fp = load('nullable_rng_fp', package / 'reference_fp.py')
    pack = lambda values: struct.pack('<' + 'I' * len(values), *values)
    report = {'schema': 'heptatorch.tensor-rng-nullable-joint.v1', 'status': 'RUNNING',
              'executable_sha256': IMAGE, 'native_executed': True, 'Torch_executed_here': False,
              'single_native_process_per_case': True, 'cases': [], 'frames': 0,
              'exact_words': 0, 'full_generator_states': 0, 'rollback_bytes': 0,
              'committed_node_invariant_checks': 0}

    def save():
        (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def nodes(s):
        raw = s.owner()['bytes']
        q = lambda at: struct.unpack_from('<Q', raw, at)[0]
        assert raw[64:72] == b'HTRNGN01' and raw[139264:139272] == b'HTORNGN1'
        base = 139264 + 144 + 488 * q(139264 + 16)
        result = {}
        for i in range(1, q(8)):
            page = q(4096 + (i >> 8) * 8)
            assert page >> 63
            at = base + (page & ((1 << 63) - 1)) + (i & 255) * 384
            if not q(at + 304):
                continue
            assert q(at + 336) == q(at + 368) == q(at + 376) == 0
            hold, address, presence, op = q(at + 248), q(at + 24), q(at + 360), q(at + 40)
            if op or not presence:
                assert hold == address == 0
            else:
                assert presence == 1 and hold and address >> 63
            result[q(at + 192)] = {'hold': hold, 'address': address, 'presence': presence}
        report['committed_node_invariant_checks'] += len(result)
        return result

    def state(s, generator, model):
        actual = s.run([(63, generator)])[0]['payload']
        assert actual == model.state()
        report['full_generator_states'] += 1

    def values(record, expected):
        assert record['payload'] == pack(expected)
        report['exact_words'] += len(expected)

    def case(name, body):
        s = wire.Session(image, directory=a.output / name)
        row = {'name': name, 'status': 'RUNNING'}
        report['cases'].append(row)
        save()
        try:
            details = body(s)
            s.run([(53,)])
            nodes(s)
            s.finish()
            row.update(status='PASS', frames=s.frames, **details)
            report['frames'] += s.frames
        except BaseException as error:
            row.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
            report['status'] = 'FAIL'
            save()
            if s.poll() is None:
                os.kill(s.pid, signal.SIGKILL)
            raise
        finally:
            save()

    def training(code):
        def execute(s):
            t, g = [r['value'] for r in s.run([wire.full((8,), 0, 1), (60, 1729)])]
            model = mt.Reference(a.cpp_reference.resolve(), 1729)
            old_hold = 0
            checks = []
            for iteration in range(12):
                commands = [(70, t), (59, t), (68, t, g, 0, wire.bits(1), 1), (2, 3, t, t),
                            (4, 7, wire.reference(4)), (code, wire.reference(5), 0, 0),
                            (9, t, wire.bits(.125)), (39, t, 0, 8), (40, t, 0, 8),
                            (13, t), (63, g), (33, 1), (59, t), (53,)]
                r = s.run(commands)
                assert r[1]['value'] == 0 and r[12]['value'] == 1
                expected, gradients = [], []
                for _ in range(8):
                    x = fp.uniform(model.word(), 0, wire.bits(1))
                    m, exponent = fp.parts(x)
                    grad = fp.rne((m * 2, exponent))
                    gm, ge = fp.parts(grad)
                    dm, de = fp.parts(fp.rne((gm, ge - 3)))
                    updated = fp.rne(fp.add((m, exponent), (-dm, de)))
                    expected.append(updated)
                    gradients.append(grad)
                values(r[7], expected)
                values(r[8], gradients)
                assert wire.meta(r[9])['version'] == 2 * (iteration + 1)
                assert r[10]['payload'] == model.state()
                report['full_generator_states'] += 1
                current = nodes(s)[t]
                assert current['presence'] == 1 and current['hold'] != old_hold
                old_hold = current['hold']
                if code == 58:
                    assert r[5]['value'] == 0 and len(r[5]['payload']) == 32
                checks.append({'update': iteration + 1, 'gradient_hold': old_hold,
                               'parameter_bits': expected, 'gradient_bits': gradients})
            s.run([(70, t), (66, g)])
            assert nodes(s)[t]['presence'] == 0
            return {'backward_code': code, 'updates': 12, 'genuine_new_gradient_hold_each_update': True,
                    'checks': checks}
        return execute

    case('nullable-uniform-backward57-SGD-reset-training', training(57))
    case('nullable-uniform-backward58-SGD-reset-training', training(58))

    def partial(s):
        x, c, marker, g = [r['value'] for r in s.run([
            wire.full((), 2, 1), wire.full((), 3), wire.full((4,), 0), (60, 31)])]
        model = mt.Reference(a.cpp_reference.resolve(), 31)
        y = s.run([(2, 3, x, c)])[0]['value']
        s.run([(58, y, 0, 1)])
        assert nodes(s)[x]['presence'] == 1
        s.run([(12, c, 0, wire.bits(5))])
        packet = [(70, x), (68, marker, g, 0, wire.bits(1))]
        packet += [(60, 700 + i) for i in range(100)]
        packet += [(58, y, 0, 0), (68, marker, g, 0, wire.bits(1))]
        before = s.owner()
        status, r = wire.parse(s.exchange(wire.packet(packet)))
        assert status == 0 and len(r) == 103 and r[-1]['opcode'] == 58 and r[-1]['value'] == 6
        assert s.owner()['length'] > before['length']
        assert nodes(s)[x]['presence'] == 0
        expected = [fp.uniform(model.word(), 0, wire.bits(1)) for _ in range(4)]
        values(s.run([(39, marker, 0, 4)])[0], expected)
        state(s, g, model)
        generators = [record['value'] for record in r[2:102]]
        for i in (0, 49, 99):
            state(s, generators[i], mt.Reference(a.cpp_reference.resolve(), 700 + i))
        s.run([(2, 3, x, c), (58, wire.reference(1), 0, 0)])
        values(s.run([(40, x, 0, 1)])[0], [wire.bits(5)])
        report['rollback_bytes'] += s.error([(70, x), (68, marker, g, 0, wire.bits(1)),
                                             (57, y, 0, 0), (68, marker, g, 0, wire.bits(1))], 6)
        assert nodes(s)[x]['presence'] == 1
        state(s, g, model)
        values(s.run([(39, marker, 0, 4)])[0], expected)
        report['rollback_bytes'] += s.error([(70, x), (68, marker, g, 0, wire.bits(1)), (999,)], 4)
        state(s, g, model)
        assert nodes(s)[x]['presence'] == 1
        s.run([(66, generator) for generator in generators] + [(70, x), (33, 3)])
        assert nodes(s)[x]['presence'] == 0
        state(s, g, model)
        return {'created_generators_during_growth_replay': 100, 'runtime_error': 6,
                'prefix_draws_committed': 4, 'suffix_draws_committed': 0,
                'nullable_clear_committed': True, 'whole_owner_rollback_controls': 2,
                'all_private_Generator_Holds_survive_gradient_release': True}

    case('nullable-clear-uniform-growth-partial-commit-and-atomic-refusal', partial)

    def empty(s):
        t, seed, g = [r['value'] for r in s.run([wire.full((0, 4), 0, 1), wire.full((0, 4), 1), (60, 19)])]
        model = mt.Reference(a.cpp_reference.resolve(), 19)
        holds = []
        for i in range(3):
            r = s.run([(70, t), (59, t), (68, t, g, 0, wire.bits(1), 1), (58, t, seed, 0),
                       (59, t), (13, t), (63, g)])
            assert r[1]['value'] == 0 and r[4]['value'] == 1
            assert wire.meta(r[5])['version'] == i + 1 and r[6]['payload'] == model.state()
            report['full_generator_states'] += 1
            hold = nodes(s)[t]['hold']
            assert hold and hold not in holds
            holds.append(hold)
        s.run([(70, t), (66, g)])
        assert nodes(s)[t]['presence'] == 0
        return {'empty_fills': 3, 'RNG_words_consumed': 0, 'versions_incremented': 3,
                'distinct_zero_length_gradient_Holds': holds}

    case('empty-uniform-does-not-consume-and-null-gradient-rematerializes', empty)
    for path, original in files.items():
        assert path.read_bytes() == original
    assert SHA(image.read_bytes()) == IMAGE
    report['status'] = 'PASS'
    save()
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}))


if __name__ == '__main__':
    main()
