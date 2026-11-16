#!/usr/bin/env python3
"""Native distribution bounds, owned transactions and existing-backward composition."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import traceback

SHA = lambda raw: hashlib.sha256(raw).hexdigest()
PROTOCOL = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
D64 = lambda value: struct.unpack('<Q', struct.pack('<d', value))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('executable', 'build-report', 'protocol', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {p.resolve(): p.read_bytes() for p in (a.executable, a.build_report, a.protocol, Path(__file__))}
    build = json.loads(inputs[a.build_report.resolve()])
    assert SHA(inputs[a.executable.resolve()]) == build['executable_sha256']
    assert SHA(inputs[a.protocol.resolve()]) == PROTOCOL
    spec = importlib.util.spec_from_file_location('distribution_controls_wire', a.protocol.resolve())
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    report = {'schema': 'heptatorch.cpu-distributions-native-controls.v1', 'status': 'RUNNING',
              'runtime_source_sha256': build['runtime_source_sha256'], 'executable_sha256': build['executable_sha256'],
              'cases': [], 'frames_before_EOF': 0, 'rollback_bytes': 0, 'full_state_comparisons': 0,
              'exact_value_comparisons': 0, 'pytorch_executed_here': False}

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def state(session, generator):
        return session.run([(63, generator)])[0]['payload']

    def same_state(session, a, b):
        assert state(session, a) == state(session, b)
        report['full_state_comparisons'] += 1

    def rollback(session, commands, code):
        report['rollback_bytes'] += session.error(commands, code)

    def words(session, tensor, count, gradient=False):
        data = session.run([(40 if gradient else 39, tensor, 0, count)])[0]['payload']
        return list(struct.unpack('<' + 'I' * count, data))

    def expected_values(session, tensor, expected, gradient=False):
        assert words(session, tensor, len(expected), gradient) == expected
        report['exact_value_comparisons'] += len(expected)

    def case(name, body):
        row = {'name': name, 'status': 'RUNNING'}
        report['cases'].append(row)
        session = wire.Session(a.executable.resolve(), directory=out / name)
        try:
            details = body(session)
            session.run([(53,)])
            raw = session.owner()['bytes']
            assert raw[64:72] == b'HTDIST01'
            assert raw[139264:139272] == b'HTODIST1'
            session.finish()
            row.update(status='PASS', frames=session.frames, **details)
            report['frames_before_EOF'] += session.frames
        except BaseException as error:
            row.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
            try:
                session.finish()
            except BaseException as cleanup:
                row['cleanup_error'] = repr(cleanup)
            report['status'] = 'FAIL'
            save()
            raise
        finally:
            save()

    def admission(session):
        x, p, g = [r['value'] for r in session.run([wire.full((4,), .5), wire.full((4,), .5), (60, 23)])]
        malformed = [(80, x, g, D64(-.1), 1), (80, x, g, 0x7ff0000000000001, 1),
                     (80, x, g, D64(.5), 2), (81, x, g, p, 2), (82, p, g, 2),
                     (83, x, g, D64(.5), 2, 1), (83, x, g, D64(.5), 1, 2),
                     (84, x, g, D64(.5), 2, 1), (85, x, 1)]
        for opcode in range(80, 86):
            base = [opcode, x, g, D64(.5), 1, 0, 0, 0]
            for field in (6, 7):
                command = base.copy()
                command[field] = 1
                malformed.append(tuple(command))
        for command in malformed:
            rollback(session, [command], 7)
        for command in [(80, 0, g, 0), (80, x, x, 0), (81, x, g, g), (82, g, g), (85, g)]:
            rollback(session, [command], 1)
        for opcode in (86, 87, 88, 89):
            rollback(session, [(opcode,)], 4)
        rollback(session, [(80, x, g, D64(.5), 1), (999,), (83, x, g, D64(.5), 1, 1)], 4)
        session.run([(66, g)])
        replacement = session.run([(60, 23)])[0]['value']
        assert replacement != g
        rollback(session, [(80, x, g, D64(.5), 1)], 1)
        session.run([(80, x, replacement, D64(.5), 1), (81, x, replacement, p, 1)])
        return {'malformed_reserved_and_scalar_requests': len(malformed), 'wrong_or_stale_handles': 6,
                'unsupported_reserved_opcodes': 4, 'full_frame_failure_after_random_prefix': True}

    case('admission-and-stale-generator', admission)

    def size_bound(session):
        x, empty, too_large, g = [r['value'] for r in session.run(
            [wire.full((1000, 1000), 7), wire.full((0, 3), 0), wire.full((101, 9901), 7), (60, 71)])]
        control = session.run([(65, g)])[0]['value']
        for code, args in ((80, (D64(0.), 1)), (81, (x, 1)), (83, (D64(.5), 1, 1)), (84, (D64(.5), 1, 1))):
            rollback(session, [(code, too_large, g, *args)], 3)
        # Boundary execution consumes two raw32 words per scalar Bernoulli value,
        # including probability zero; the established raw advance control is split
        # at its own honest one-million bound.
        old = session.owner()
        session.run([(80, x, g, D64(0.), 1)])
        session.run([(67, control, 1000000), (67, control, 1000000)])
        same_state(session, g, control)
        assert wire.meta(session.run([(13, x)])[0])['version'] == 1
        for start in range(0, 1000000, 125000):
            payload = session.run([(39, x, start, 125000)])[0]['payload']
            assert payload == bytes(500000)
            report['exact_value_comparisons'] += 125000
        state_before = state(session, g)
        session.run([(80, empty, g, D64(1.), 1), (81, empty, g, x, 1)])
        assert state(session, g) == state_before
        assert wire.meta(session.run([(13, empty)])[0])['version'] == 2
        same = session.run([(83, empty, g, D64(.5), 1, 1)])[0]['value']
        assert same == empty and state(session, g) == state_before
        report['full_state_comparisons'] += 3
        assert not session.mapped(old['pointer'])
        return {'exact_maximum_elements': 1000000, 'max_plus_one_refused_before_draw': 4,
                'two_million_raw32_draws_compared': True, 'empty_probability_shape_bypass': True,
                'empty_Bernoulli_Version_two_dropouts_identity': True}

    case('one-million-bound-and-empty-no-consumption', size_bound)

    def composition(session):
        x, c, target, g = [r['value'] for r in session.run(
            [wire.full((), 2, 1), wire.full((), 3), wire.full((4,), .5), (60, 19)])]
        control = session.run([(65, g)])[0]['value']
        y = session.run([(2, 3, x, c)])[0]['value']
        session.run([(12, c, 0, wire.bits(5))])
        status, rows = wire.parse(session.exchange(wire.packet(
            [(80, target, g, D64(1.), 1), (58, y, 0, 0), (80, target, g, D64(0.), 1), (60, 999)])))
        assert status == 0 and len(rows) == 2 and rows[-1]['value'] == 6
        assert struct.unpack_from('<Q', rows[-1]['payload'])[0] == 6
        session.run([(67, control, 8)])
        same_state(session, g, control)
        expected_values(session, target, [1065353216] * 4)
        assert wire.meta(session.run([(13, target)])[0])['version'] == 1
        rollback(session, [(80, target, g, D64(0.), 1), (57, y, 0, 0), (80, target, g, D64(1.), 1)], 6)
        same_state(session, g, control)
        session.run([(33, 3)])
        same_state(session, g, control)
        # Existing per-node backward also traverses the new exact zero derivative.
        probability = session.run([wire.full((4,), .5, 1)])[0]['value']
        result = session.run([(82, probability, g, 0)])[0]['value']
        nan_seed = session.run([wire.tensor((4,), [-0., float('nan'), float('inf'), -1.])])[0]['value']
        session.run([(12, probability, 0, 0x7fc12345)])
        before = state(session, g)
        session.run([(58, result, nan_seed, 0)])
        expected_values(session, probability, [0] * 4, True)
        assert state(session, g) == before
        report['full_state_comparisons'] += 1
        session.run([(70, probability), (57, result, nan_seed, 0)])
        expected_values(session, probability, [0] * 4, True)
        return {'partial58_commits_random_prefix_and_stops_suffix': True,
                'atomic57_preserves_both_Tensor_and_Generator': True,
                'generator_survives_reset': True, 'both_backwards_execute_Bernoulli_zero_derivative': True,
                'Bernoulli_derivative_has_no_saved_probability_value': True}

    case('backward58-prefix-and57-rollback', composition)

    def ieee(session):
        raw = [0x7f800000, 0xff800000, 0x7fc12345, 0, 0x80000000, 0xc0000000, 0x40000000]
        payload = wire.Payload(struct.pack('<7I', *raw), 4)
        x, g = [r['value'] for r in session.run([(1, 1, 0, 7, 0, 0, 0, payload), (60, 99)])]
        before = state(session, g)
        y = session.run([(83, x, g, D64(1.), 1, 1)])[0]['value']
        values = words(session, y, 7)
        assert all((value & 0x7fffffff) > 0x7f800000 for value in values[:3])
        assert values[3:] == [0, 0x80000000, 0x80000000, 0]
        report['exact_value_comparisons'] += 4
        assert state(session, g) == before
        report['full_state_comparisons'] += 1
        assert session.run([(83, x, g, 0, 1, 0)])[0]['value'] == x
        assert words(session, x, 7) == raw
        report['exact_value_comparisons'] += 7
        return {'p1_uses_real_IEEE_multiplication': True, 'p0_identity_preserves_NaN_payload': True,
                'p1_NaN_payload_not_claimed_exact': True}

    case('dropout-IEEE-specials', ieee)
    for path, data in inputs.items():
        assert path.read_bytes() == data, path
    report.update(status='PASS', case_count=len(report['cases']), input_custody_after=True,
                  inputs={str(path): SHA(data) for path, data in inputs.items()})
    save()
    print(json.dumps({k: report[k] for k in ('status', 'case_count', 'frames_before_EOF', 'rollback_bytes',
                                           'full_state_comparisons', 'exact_value_comparisons')}, indent=2))


if __name__ == '__main__':
    main()
