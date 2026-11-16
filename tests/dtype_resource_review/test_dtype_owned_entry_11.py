#!/usr/bin/env python3
"""Bounded-output qualification of real owned dtype03 step/return calls.

The immutable HIA probe allocates genuine buffers and copies valid, observed
relative-offset state bytes into them. It never imports pointers or duplicates
an ownership descriptor. Exact full returned state is compared with execution
of the same commands in the real interactive target. This is an ABI/atomicity
test, not a Torch numerical oracle or admission of arbitrary foreign state.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import traceback

COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = '0c5176e3aa968631ae592b887c5dc3cf1e8e3f48f73593f076e9d7965ffacfd7'
PROBE = '54f81a65652e2c2ba9eb3f04d608e89b71c92c5ad33ff7061e9a03b89f45f9ea'
PROTOCOL = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
CLIENT = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
SOURCE = '6ffe509afce28ff15705b4c7b269eaef8f8f89f9d8b2486d4d9fa45b6be66ef3'
IMAGE = '830f0b4ef8d6ad8c6a105be56d383628b45c84e3e43905fee6812c7c52d35a51'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'stack-build', 'probe-source', 'protocol', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    report = {'schema': 'heptatorch.dtype03-independent-owned-entry.v1', 'status': 'RUNNING',
              'source_sha256': SOURCE, 'native_sha256': IMAGE, 'probe_source_sha256': PROBE,
              'compiler_sha256': COMPILER, 'runtime_linker_sha256': LINKER,
              'cases': [], 'native_probe_processes': 0, 'interactive_frames': 0,
              'complete_owner_bytes_compared': 0, 'qualification_owned_byte_copy_only': True,
              'foreign_pointer_import': False, 'owner_descriptor_memcpy': False,
              'numerical_oracle_claim': False, 'Torch_executed_here': False,
              'whole_pytorch_gate': 'HOLD', 'pytorch_api_denominator': 2590,
              'full_pytorch_rewrite_complete': False}

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def bind(path, expected=None):
        path = path.resolve()
        raw = path.read_bytes()
        if expected:
            assert sha(raw) == expected, (path, sha(raw), expected)
        inputs[path] = raw
        return raw

    try:
        bind(Path(__file__))
        compiler = a.stack_build.resolve() / 'native/stack-module-compiler'
        linker = a.stack_build.resolve() / 'native/stack-linker-runtime'
        bind(compiler, COMPILER)
        bind(linker, LINKER)
        probe_source = bind(a.probe_source, PROBE)
        bind(a.protocol, PROTOCOL)
        bind(a.protocol.with_name('owner_client.py'), CLIENT)
        source = a.build.resolve() / 'runtime.hepta'
        image = a.build.resolve() / 'tensor-dtype-dropout-session'
        bind(source, SOURCE)
        bind(image, IMAGE)
        build = json.loads(bind(a.build.resolve() / 'build.json'))
        assert build['status'] == 'PASS' and build['source_sha256'] == SOURCE and build['native_sha256'] == IMAGE
        assert build['descriptor_bytes'] == 640 and build['state_format'] == 'HTDTYP03'
        spec = importlib.util.spec_from_file_location('dtype_boundary_wire', a.protocol)
        wire = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wire)
        result = subprocess.run([str(compiler)], input=probe_source, capture_output=True,
                                env={'PATH': ''}, timeout=60)
        (out / 'entry-probe.hepta').write_bytes(probe_source)
        (out / 'entry-probe.o').write_bytes(result.stdout)
        (out / 'compile.stderr').write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
        objects = []
        for unit in build['units']:
            assert unit['compiler_exit'] == 0
            bind(a.build.resolve() / (unit['name'] + '.hepta'), unit['source_sha256'])
            raw = bind(a.build.resolve() / (unit['name'] + '.o'), unit['object_sha256'])
            objects.append(raw)
        assert len(objects) == 11
        objects.append(result.stdout)
        request = b'HIASTKR1' + struct.pack('<3Q', len(objects), 5, 0) + b'probe'
        request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
        (out / 'link.request').write_bytes(request)
        linked = subprocess.run([str(linker)], input=request, capture_output=True, env={'PATH': ''}, timeout=60)
        (out / 'link.stderr').write_bytes(linked.stderr)
        (out / 'link.exit.json').write_text(json.dumps({'exit_code': linked.returncode}) + '\n')
        assert linked.returncode == 0 and not linked.stderr and linked.stdout.startswith(b'\x7fELF')
        probe = out / 'owned-entry-probe'
        probe.write_bytes(linked.stdout)
        probe.chmod(0o500)
        report.update(probe_image_sha256=sha(linked.stdout), linked_modules=len(objects))
        save()

        def invoke(state, commands, capacity):
            body = wire.packet(commands)
            raw = struct.pack('<3Q', len(state), len(body), capacity) + state + body
            assert len(raw) <= 1 << 20
            index = report['native_probe_processes']
            report['native_probe_processes'] += 1
            execution = subprocess.run([str(probe)], input=raw, capture_output=True, env={'PATH': ''}, timeout=30)
            name = f'probe-{index:03d}'
            (out / (name + '.input')).write_bytes(raw)
            (out / (name + '.stdout')).write_bytes(execution.stdout)
            (out / (name + '.stderr')).write_bytes(execution.stderr)
            assert execution.returncode == 0 and not execution.stderr, (index, execution.returncode)
            assert len(execution.stdout) >= 24
            length, owner_size, returned_capacity = struct.unpack_from('<3Q', execution.stdout)
            assert returned_capacity == capacity and len(execution.stdout) == 24 + capacity + owner_size
            return length, execution.stdout[24:24 + capacity], execution.stdout[24 + capacity:]

        def case(name, setup, make_commands, expected_length, verify):
            row = {'name': name, 'status': 'RUNNING', 'expected_length': expected_length}
            report['cases'].append(row)
            save()
            session = wire.Session(image, directory=out / name)
            try:
                handles = setup(session)
                before = session.owner()['bytes']
                assert before[64:72] == b'HTDTYP03' and before[139264:139272] == b'HTODTY03'
                commands = make_commands(handles)
                response = session.exchange(wire.packet(commands))
                status, records = wire.parse(response)
                after = session.owner()['bytes']
                assert status == 0 and len(response) == expected_length and len(records) == 1
                assert after != before, 'The control must exercise a real state transition'
                verify(session, handles, records[0])
                capacities = sorted({0, 1, 31, 32, 33, 63, expected_length - 1})
                for capacity in capacities:
                    if capacity >= expected_length:
                        continue
                    length, output, restored = invoke(before, commands, capacity)
                    assert restored == before, (name, capacity, 'short output changed actual owner')
                    report['complete_owner_bytes_compared'] += len(before)
                    if capacity < 32:
                        assert length == 0 and output == bytes(capacity), (name, capacity, length)
                    else:
                        assert length == 32 and wire.parse(output[:length]) == (3, []), (name, capacity, length)
                length, output, restored = invoke(before, commands, expected_length)
                assert length == expected_length and output == response and restored == after, (name, 'exact capacity mismatch')
                report['complete_owner_bytes_compared'] += len(after)
                # Same owner and command, then a genuine ordinary admission error:
                # all allocations, Version increments, RNG consumption and graph
                # identity changes must roll back together.
                length, output, restored = invoke(before, commands + [(999,)], expected_length + 32)
                assert length == 32 and wire.parse(output[:length]) == (4, []) and restored == before
                report['complete_owner_bytes_compared'] += len(before)
                assert session.run([(53,)])[0]['value'] == 1
                row.update(status='PASS', insufficient_capacities=[n for n in capacities if n < expected_length],
                    exact_capacity_response_and_complete_state_match=True, later_unknown_command_full_owner_rollback=True,
                    original_owner_bytes=len(before), committed_owner_bytes=len(after),
                    original_owner_sha256=sha(before), committed_owner_sha256=sha(after),
                    interactive_response_sha256=sha(response))
            except BaseException as error:
                row.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
                raise
            finally:
                report['interactive_frames'] += session.frames
                try:
                    session.finish()
                    row['transport_exit_code'] = session.exit_code
                except BaseException as error:
                    row.update(status='FAIL', transport_error=repr(error))
                    raise
                save()

        def nonleaf(session, empty=False, retained=False):
            shape = (2, 0, 3) if empty else (2, 3)
            x = session.run([wire.full(shape, .25, 1)])[0]['value']
            scalar = session.run([wire.full((), 2)])[0]['value']
            q = session.run([(2, 3, x, scalar)])[0]['value']
            session.run([(41, scalar)])
            result = {'x': x, 'q': q}
            if retained:
                session.run([(93, q)])
                seed = session.run([wire.full(shape, 1)])[0]['value']
                assert session.run([(58, q, seed, 1)])[0]['value'] == 0
                g = session.run([(91, q)])[0]['value']
                result.update(g=g, meta=wire.meta(session.run([(13, q)])[0]),
                              typed=session.run([(96, q)])[0]['payload'])
            return result

        def retained(session):
            return nonleaf(session, retained=True)

        def check_detach(session, h, record):
            assert record['value'] == h['q'] and record['payload'] == b''
            meta = wire.meta(session.run([(13, h['q'])])[0])
            for key in ('storage_id', 'version_id', 'version'):
                assert meta[key] == h['meta'][key]
            assert meta['requires_grad'] == 0
            assert session.run([(94, h['q'])])[0]['value'] == 1
            assert session.run([(91, h['q'])])[0]['value'] == h['g']
            info = session.run([(96, h['q'])])[0]['payload']
            assert info[:24] == h['typed'][:24] and struct.unpack('<4Q', info)[3] == 1

        case('detach_retained_nonleaf_with_existing_G', retained, lambda h: [(101, h['q'])], 64, check_detach)

        def boolean(session, empty=False, transpose=False):
            shape, raw = ([2, 0, 3], b'') if empty else ([2, 3], bytes((0, 1, 1, 0, 1, 0)))
            config = struct.pack('<8Q', len(shape), 0, *shape, *([0] * (4 - len(shape))), 160, len(raw))
            body = b'HTORCH03' + struct.pack('<3Q', 1, 160 + len(raw), 0) + struct.pack('<8Q', 97, 1, 96, 0, 0, 0, 0, 0) + config + raw
            status, rows = wire.parse(session.exchange(body))
            assert status == 0 and len(rows) == 1
            base = rows[0]['value']
            target = session.run([(6, base, 0, 1)])[0]['value'] if transpose else base
            meta = wire.meta(session.run([(13, target)])[0])
            before_raw = session.run([(98, target, 0, meta['numel'])])[0]['payload']
            return {'base': base, 'q': target, 'meta': meta, 'raw': before_raw}

        def check_not(session, h, record):
            assert record['value'] == h['q'] and record['payload'] == b''
            meta = wire.meta(session.run([(13, h['q'])])[0])
            assert meta['version'] == h['meta']['version'] + 1
            for key in ('storage_id', 'version_id', 'shape', 'strides', 'storage_offset', 'storage_bytes'):
                assert meta[key] == h['meta'][key]
            after_raw = session.run([(98, h['q'], 0, meta['numel'])])[0]['payload']
            assert after_raw == bytes(int(v == 0) for v in h['raw'])
            assert wire.meta(session.run([(13, h['base'])])[0])['version'] == meta['version']

        case('logical_not_real_bool_transpose', lambda s: boolean(s, transpose=True), lambda h: [(102, h['q'])], 64, check_not)
        case('logical_not_empty_bool_Version_increment', lambda s: boolean(s, empty=True), lambda h: [(102, h['q'])], 64, check_not)

        def dropout(session, empty):
            h = nonleaf(session, empty=empty)
            h['gen'] = session.run([(60, 1729)])[0]['value']
            h['empty'] = empty
            return h

        def check_dropout(session, h, record):
            mask = struct.unpack('<Q', record['payload'])[0]
            assert record['value'] != 0 and mask != 0
            info = struct.unpack('<4Q', session.run([(96, mask)])[0]['payload'])
            meta = wire.meta(session.run([(13, mask)])[0])
            if h['empty']:
                assert record['value'] == h['q'] and info[:2] == (0, 4) and meta['numel'] == meta['storage_bytes'] == 0
            else:
                assert record['value'] != h['q'] and info[:2] == (1, 1) and meta['storage_bytes'] == meta['numel'] == 6
            assert meta['version'] == 0

        case('native_dropout_nonempty_bool_mask_and_Generator', lambda s: dropout(s, False),
             lambda h: [(99, h['q'], h['gen'], 0x3fe0000000000000, 1)], 72, check_dropout)
        case('native_dropout_empty_same_Tensor_history_and_float_mask', lambda s: dropout(s, True),
             lambda h: [(99, h['q'], h['gen'], 0x3fe0000000000000, 1)], 72, check_dropout)
        report.update(status='PASS', case_count=5, passed=5, failed=0)
    except BaseException as error:
        report.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
    finally:
        report['all_source_custody_unchanged'] = all(path.read_bytes() == raw for path, raw in inputs.items())
        if not report['all_source_custody_unchanged']:
            report['status'] = 'FAIL'
        report['inputs'] = {str(path): {'bytes': len(raw), 'sha256': sha(raw)} for path, raw in inputs.items()}
        (out / 'executed-test.py').write_bytes(Path(__file__).read_bytes())
        save()
    print(json.dumps({key: report.get(key) for key in ('status', 'case_count', 'native_probe_processes', 'interactive_frames',
                                                     'complete_owner_bytes_compared', 'error')}))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
