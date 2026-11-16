#!/usr/bin/env python3
"""Real HIA owned-entry capacity/commit gates for the I64 profile.

The unchanged HIA probe creates genuine owned buffers and reconstructs only
observed, relative-offset state bytes. No native pointer is imported, no owner
descriptor is copied, and the production entry and eleven units are unchanged.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import traceback

from resource_support import (COMPILER, IMAGE, LINKER, PROBE, SOURCE, Custody,
                              State, create, integer_raw, load, sha, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'stack-build', 'probe-source', 'protocol', 'reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--expected-source-sha256', default=SOURCE)
    p.add_argument('--expected-native-sha256', default=IMAGE)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    custody = Custody(out)
    report = {'schema': 'heptatorch.i64-independent-owned-entry.v1', 'status': 'RUNNING',
              'source_sha256': a.expected_source_sha256, 'native_sha256': a.expected_native_sha256,
              'probe_source_sha256': PROBE, 'compiler_sha256': COMPILER,
              'runtime_linker_sha256': LINKER, 'cases': [], 'native_probe_processes': 0,
              'interactive_frames': 0, 'complete_owner_bytes_compared': 0,
              'Torch_executed_here': False, 'foreign_pointer_import': False,
              'owner_descriptor_memcpy': False, 'whole_pytorch_gate': 'HOLD',
              'pytorch_api_denominator': 2590, 'full_pytorch_rewrite_complete': False}

    def save():
        write_json(out/'report.json', report)

    try:
        custody.bind(Path(__file__), copy='executed-test.py')
        wire, image, build, observed = load(custody, a.build, a.protocol, a.reference,
                                          a.expected_source_sha256, a.expected_native_sha256)
        compiler = a.stack_build.resolve()/'native/stack-module-compiler'
        linker = a.stack_build.resolve()/'native/stack-linker-runtime'
        custody.bind(compiler, COMPILER)
        custody.bind(linker, LINKER)
        probe_source = custody.bind(a.probe_source, PROBE, 'entry_probe.hepta')
        result = subprocess.run([str(compiler)], input=probe_source, capture_output=True,
                                env={'PATH': ''}, timeout=60)
        (out/'entry-probe.o').write_bytes(result.stdout)
        (out/'compile.stderr').write_bytes(result.stderr)
        write_json(out/'compile.exit.json', {'exit_code': result.returncode})
        assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
        objects = [custody.bind(a.build/(unit['name']+'.o'), unit['object_sha256']) for unit in build['units']]
        objects.append(result.stdout)
        request = b'HIASTKR1'+struct.pack('<3Q', len(objects), 5, 0)+b'probe'
        request += b''.join(struct.pack('<Q', len(obj))+obj for obj in objects)
        (out/'link.request').write_bytes(request)
        linked = subprocess.run([str(linker)], input=request, capture_output=True,
                                env={'PATH': ''}, timeout=60)
        (out/'link.stderr').write_bytes(linked.stderr)
        write_json(out/'link.exit.json', {'exit_code': linked.returncode})
        assert linked.returncode == 0 and not linked.stderr and linked.stdout.startswith(b'\x7fELF')
        probe = out/'owned-entry-probe'
        probe.write_bytes(linked.stdout)
        probe.chmod(0o500)
        report.update(linked_modules=len(objects), probe_image_sha256=sha(linked.stdout))
        assert len(objects) == 12
        reference = observed['rows'][46]
        assert reference['index'] == 46 and reference['name'] == 'dropout-dense-true-0.5'
        original_rng = bytes.fromhex(observed['states'][reference['state_before']]['hex'])
        advanced_rng = bytes.fromhex(observed['states'][reference['state_after']]['hex'])

        def invoke(state, commands, capacity):
            packet = wire.packet(commands)
            raw = struct.pack('<3Q', len(state), len(packet), capacity)+state+packet
            assert len(raw) <= 1 << 20
            index = report['native_probe_processes']
            report['native_probe_processes'] += 1
            process = subprocess.run([str(probe)], input=raw, capture_output=True, env={'PATH': ''}, timeout=30)
            name = 'probe-'+str(index).zfill(3)
            (out/(name+'.input')).write_bytes(raw)
            (out/(name+'.stdout')).write_bytes(process.stdout)
            (out/(name+'.stderr')).write_bytes(process.stderr)
            write_json(out/(name+'.exit.json'), {'exit_code': process.returncode})
            assert process.returncode == 0 and not process.stderr
            length, size, cap = struct.unpack_from('<3Q', process.stdout)
            assert cap == capacity and len(process.stdout) == 24+capacity+size
            return length, process.stdout[24:24+capacity], process.stdout[24+capacity:]

        def setup(session, empty=False):
            if empty:
                base = create(wire, session, 2, [2, 0, 3])
                target = base
            else:
                base = create(wire, session, 2, [2, 3],
                    bytes.fromhex(reference['observation']['before']['x']['storage_bytes_hex']))
                target = session.run([(6, base, 0, 1)])[0]['value']
            gen = session.run([(60, 1729)])[0]['value']
            return {'base': base, 'target': target, 'gen': gen}

        def run_case(name, prepare, commands_for, expected_length, verify, late=False):
            row = {'name': name, 'status': 'RUNNING', 'expected_length': expected_length,
                   'late_error_stops_suffix': late}
            report['cases'].append(row)
            session = wire.Session(image, directory=out/name)
            try:
                handles = prepare(session)
                before = session.owner()['bytes']
                assert State(before).rng(handles['gen']) == original_rng
                commands = commands_for(handles)
                response = session.exchange(wire.packet(commands))
                status, records = wire.parse(response)
                after = session.owner()['bytes']
                assert status == 0 and len(response) == expected_length and after != before
                verify(session, handles, records, State(before), State(after))
                capacities = sorted({0, 1, 31, 32, 33, 63, 79, 80, expected_length-1})
                capacities = [cap for cap in capacities if cap < expected_length]
                observations = []
                for capacity in capacities:
                    length, output, restored = invoke(before, commands, capacity)
                    assert restored == before, (name, capacity, 'short output changed complete owner')
                    assert State(restored).rng(handles['gen']) == original_rng
                    report['complete_owner_bytes_compared'] += len(before)
                    if capacity < 32:
                        assert length == 0 and output == bytes(capacity)
                    else:
                        assert length == 32 and wire.parse(output[:length]) == (3, [])
                    observations.append({'capacity': capacity, 'returned_length': length,
                                         'complete_owner_sha256': sha(restored), 'full_RNG_unchanged': True})
                length, output, restored = invoke(before, commands, expected_length)
                assert length == expected_length and output == response and restored == after, (name, 'exact output capacity')
                report['complete_owner_bytes_compared'] += len(after)
                # In the explicit late-error contract, a suffix is not executed.
                # Ordinary successful commands instead reach unknown opcode999
                # and the owned step must return the entire original owner.
                length, output, restored = invoke(before, commands+[(999,)], expected_length+32)
                if late:
                    assert length == expected_length and output[:length] == response and restored == after
                else:
                    assert length == 32 and wire.parse(output[:length]) == (4, []) and restored == before
                report['complete_owner_bytes_compared'] += len(restored)
                assert session.run([(53,)])[0]['value'] == 1
                row.update(status='PASS', short_capacities=observations, exact_reply_and_full_state_match=True,
                           ordinary_suffix_behavior='skipped after explicit late error' if late else 'full owner rollback',
                           before_owner_sha256=sha(before), after_owner_sha256=sha(after),
                           interactive_response_sha256=sha(response), before_owner_bytes=len(before),
                           after_owner_bytes=len(after))
            except BaseException as error:
                row.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
                raise
            finally:
                report['interactive_frames'] += session.frames
                try:
                    session.finish()
                    row['EOF_exit_code'] = session.exit_code
                except BaseException as error:
                    row.update(status='FAIL', transport_error=repr(error))
                    raise
                save()

        def verify_late(session, h, records, before, after):
            assert len(records) == 1 and records[0]['opcode'] == 105
            assert records[0]['value'] == 0 and records[0]['payload'] == struct.pack('<2Q', 12, 0)
            assert after.rng(h['gen']) == advanced_rng and after.counts() == before.counts()
            assert after.next == before.next+2
            assert after.free() == before.free()
            for node in range(before.next, after.next):
                assert after.node(node)[304//8] == 0 and after.node(node)[208//8] == 0

        run_case('late_cast_output_guard_precedes_RNG', setup,
                 lambda h: [(105, h['target'], h['gen'], 0x3fe0000000000000, 1)],
                 80, verify_late, late=True)

        def verify_false(session, h, records, before, after):
            assert len(records) == 1
            row = records[0]
            error, mask = struct.unpack('<2Q', row['payload'])
            assert error == 0 and row['value'] and mask
            assert after.rng(h['gen']) == original_rng
            assert struct.unpack('<4Q', session.run([(96, mask)])[0]['payload'])[:2] == (1, 1)
            assert session.run([(98, mask, 0, 6)])[0]['payload'] == bytes([1])*6
            assert session.run([(98, row['value'], 0, 6)])[0]['payload'] == bytes.fromhex(
                reference['observation']['before']['x']['logical_bytes_hex'])

        run_case('False_output_mask_ownership', setup,
                 lambda h: [(105, h['target'], h['gen'], 0x7ff8000000000001, 0)],
                 80, verify_false)

        def verify_empty(session, h, records, before, after):
            row = records[0]
            error, mask = struct.unpack('<2Q', row['payload'])
            assert error == 0 and row['value'] == h['target'] and mask
            assert after.rng(h['gen']) == original_rng
            assert struct.unpack('<4Q', session.run([(96, mask)])[0]['payload'])[:2] == (2, 8)
            assert session.run([(98, mask, 0, 0)])[0]['payload'] == b''

        run_case('empty_NaN_preserves_RNG_and_I64_mask', lambda s: setup(s, True),
                 lambda h: [(105, h['target'], h['gen'], 0x7ff8000000000001, 1)],
                 80, verify_empty)

        def promotion_setup(session):
            h = setup(session)
            h['f'] = create(wire, session, 0, [3, 2], struct.pack('<f', .5)*6)
            return h

        def verify_promotion(session, h, records, before, after):
            assert len(records) == 1 and records[0]['value'] != 0 and records[0]['payload'] == b''
            assert after.rng(h['gen']) == original_rng and after.next == before.next+2
            assert after.node(before.next)[304//8] == 0
            assert after.tensor_node(records[0]['value'])[0] == before.next+1

        run_case('CPU_promotion_temporary_output_guard', promotion_setup,
                 lambda h: [(103, 1, h['target'], h['f'])], 64, verify_promotion)

        def prefix_setup(session):
            h = setup(session)
            h['sentinel'] = create(wire, session, 2, [1], integer_raw([9]))
            return h

        def verify_prefix(session, h, records, before, after):
            assert [r['opcode'] for r in records] == [106, 105]
            verify_late(session, h, records[1:], before, after)
            assert session.run([(98, h['sentinel'], 0, 1)])[0]['payload'] == integer_raw([10])
            assert wire.meta(session.run([(13, h['sentinel'])])[0])['version'] == 1

        run_case('prefix_guard_rollback_and_explicit_late_suffix_stop', prefix_setup,
                 lambda h: [(106, h['sentinel'], 1),
                     (105, h['target'], h['gen'], 0x3fe0000000000000, 1),
                     (106, h['sentinel'], 100), (999,)], 112, verify_prefix, late=True)
        assert len(report['cases']) == 5 and all(row['status'] == 'PASS' for row in report['cases'])
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
    try:
        report['inputs'] = custody.finish()
        report['source_custody_after'] = True
    except BaseException as error:
        report.update(status='FAIL', custody_error=repr(error))
    report['case_count'] = len(report['cases'])
    report['passed'] = sum(row['status'] == 'PASS' for row in report['cases'])
    save()
    print(json.dumps({k: report.get(k) for k in ('status', 'case_count', 'passed', 'native_probe_processes', 'interactive_frames', 'error')}))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
