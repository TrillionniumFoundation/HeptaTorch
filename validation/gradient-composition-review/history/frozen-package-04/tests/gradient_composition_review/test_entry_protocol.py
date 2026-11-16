#!/usr/bin/env python3
"""Exercise real owned step/return at exact output boundaries for public gradients.

The frozen HIA probe allocates real owners, calls the checked native step, then
returns bytes and drops its actual owners. It never imports foreign pointers.
Interactive execution supplies the complete-state comparison for this ABI test;
this is not a numerical oracle or an arbitrary owner-state authentication claim.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess

SHA = lambda data: hashlib.sha256(data).hexdigest()
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = '0c5176e3aa968631ae592b887c5dc3cf1e8e3f48f73593f076e9d7965ffacfd7'
PROBE = '54f81a65652e2c2ba9eb3f04d608e89b71c92c5ad33ff7061e9a03b89f45f9ea'
PROTOCOL = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'compiler', 'runtime-linker', 'probe-source', 'protocol', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('expected-source', 'expected-image', 'image-name'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--include-adam-distributions', action='store_true')
    a = parser.parse_args()
    a.build = a.build.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    image = a.build / a.image_name
    source = a.build / 'runtime.hepta'
    inputs = {path.resolve(): path.read_bytes() for path in (
        Path(__file__), a.compiler, a.runtime_linker, a.probe_source, a.protocol,
        source, image, a.build / 'build.json')}
    for path, expected in [(a.compiler, COMPILER), (a.runtime_linker, LINKER),
                           (a.probe_source, PROBE), (a.protocol, PROTOCOL),
                           (source, a.expected_source), (image, a.expected_image)]:
        assert SHA(inputs[path.resolve()]) == expected, path
    build = json.loads(inputs[(a.build / 'build.json').resolve()])
    assert build['status'] == 'PASS'
    spec = importlib.util.spec_from_file_location('gradient_boundary_wire', a.protocol)
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    compile_result = subprocess.run([str(a.compiler.resolve())], input=inputs[a.probe_source.resolve()],
                                    capture_output=True, env={'PATH': ''}, timeout=60)
    (a.output / 'probe.o').write_bytes(compile_result.stdout)
    (a.output / 'compile.stderr').write_bytes(compile_result.stderr)
    assert compile_result.returncode == 0 and not compile_result.stderr
    objects = []
    for unit in build['units']:
        path = a.build / (unit['name'] + '.o')
        raw = path.read_bytes()
        assert SHA(raw) == unit['object_sha256']
        inputs[path] = raw
        objects.append(raw)
    objects.append(compile_result.stdout)
    request = b'HIASTKR1' + struct.pack('<3Q', len(objects), 5, 0) + b'probe'
    request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
    (a.output / 'link.request').write_bytes(request)
    link_result = subprocess.run([str(a.runtime_linker.resolve())], input=request,
                                 capture_output=True, env={'PATH': ''}, timeout=60)
    (a.output / 'link.stderr').write_bytes(link_result.stderr)
    assert link_result.returncode == 0 and not link_result.stderr
    probe = a.output / 'owned-entry-probe'
    probe.write_bytes(link_result.stdout)
    probe.chmod(0o500)
    report = {'schema': 'heptatorch.gradient-composition.independent-owned-entry.v1',
              'status': 'RUNNING', 'source_sha256': a.expected_source, 'image_sha256': a.expected_image,
              'probe_source_sha256': PROBE, 'probe_image_sha256': SHA(link_result.stdout),
              'compiler_sha256': COMPILER, 'runtime_linker_sha256': LINKER,
              'cases': [], 'native_probe_processes': 0, 'interactive_frames': 0,
              'complete_owner_bytes_compared': 0, 'qualification_owned_byte_copy_only': True,
              'foreign_pointer_import': False, 'numerical_oracle_claim': False}

    def save():
        (a.output / 'report.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')

    def invoke(state, commands, capacity):
        body = wire.packet(commands)
        raw = struct.pack('<3Q', len(state), len(body), capacity) + state + body
        assert len(raw) <= 1 << 20
        index = report['native_probe_processes']
        report['native_probe_processes'] += 1
        run = subprocess.run([str(probe.resolve())], input=raw, capture_output=True, env={'PATH': ''}, timeout=30)
        (a.output / (str(index) + '.input')).write_bytes(raw)
        (a.output / (str(index) + '.stdout')).write_bytes(run.stdout)
        (a.output / (str(index) + '.stderr')).write_bytes(run.stderr)
        assert run.returncode == 0 and not run.stderr, (index, run.returncode)
        length, owner_size, returned_capacity = struct.unpack_from('<3Q', run.stdout)
        assert returned_capacity == capacity and len(run.stdout) == 24 + capacity + owner_size
        return length, run.stdout[24:24 + capacity], run.stdout[24 + capacity:]

    def case(name, setup, make_commands, expected_length, expected_record_count, operation_error=0):
        row = {'name': name, 'status': 'RUNNING', 'expected_length': expected_length}
        report['cases'].append(row)
        save()
        session = wire.Session(image, directory=a.output / name)
        try:
            handles = setup(session)
            before = session.owner()['bytes']
            commands = make_commands(handles)
            response = session.exchange(wire.packet(commands))
            status, records = wire.parse(response)
            after = session.owner()['bytes']
            assert status == 0 and len(response) == expected_length and len(records) == expected_record_count
            if operation_error:
                assert records[-1]['value'] == operation_error
            assert after != before, 'this boundary case must exercise real state mutation'
            capacities = sorted({0, 31, 32, 63, expected_length - 1})
            for capacity in capacities:
                length, output, restored = invoke(before, commands, capacity)
                assert restored == before
                report['complete_owner_bytes_compared'] += len(before)
                if capacity < 32:
                    assert length == 0 and output == bytes(capacity)
                else:
                    assert length == 32 and wire.parse(output[:length]) == (3, [])
            length, output, restored = invoke(before, commands, expected_length)
            assert length == expected_length and output == response and restored == after
            report['complete_owner_bytes_compared'] += len(after)
            # A successful ordinary command cannot make a subsequent unknown
            # command non-atomic. The explicit error path already stops suffixes.
            if not operation_error:
                length, output, restored = invoke(before, commands + [(999,)], expected_length + 32)
                assert length == 32 and wire.parse(output[:length]) == (4, []) and restored == before
                report['complete_owner_bytes_compared'] += len(before)
            session.run([(53,)])
            session.finish()
            report['interactive_frames'] += session.frames
            row.update(status='PASS', insufficient_capacities=capacities,
                       exact_capacity_matches_real_interactive_owner=True,
                       explicit_prefix_error=operation_error,
                       original_owner_bytes=len(before), committed_owner_bytes=len(after),
                       interactive_response_sha256=SHA(response))
            save()
        except BaseException as error:
            row.update(status='FAIL', error=repr(error))
            report['status'] = 'FAIL'
            save()
            raise

    def base(session):
        p, g = [row['value'] for row in session.run([
            wire.tensor((4,), [1., 2., 3., 4.], 1), wire.tensor((4,), [2., 2., 2., 2.])])]
        return {'p': p, 'g': g}

    case('assign-real-gradient-and-read-result', base,
         lambda h: [(92, h['p'], h['g']), (91, h['p'])], 96, 2)

    def revoked(session):
        handles = base(session)
        session.run([(92, handles['p'], handles['g']), (41, handles['g'])])
        return handles

    case('republish-still-owned-gradient', revoked, lambda h: [(91, h['p'])], 64, 1)
    case('clear-binding-releases-last-private-gradient-root', revoked, lambda h: [(92, h['p'], 0)], 64, 1)

    if a.include_adam_distributions:
        def adam(session):
            h = base(session)
            h['rng'] = session.run([(60, 71)])[0]['value']
            config = struct.pack('<8Q', wire.bits(.125), 0, 0, 0, 0, 0, 0, 0)
            groups = struct.pack('<12Q', 1, 1, 0, *([0] * 8), h['p'])
            h['opt'] = session.run([(71, 1, wire.Payload(config), wire.Payload(groups))])[0]['value']
            return h
        case('assign-gradient-Adam-step-late-dropout-prefix', adam,
             lambda h: [(92, h['p'], h['g']), (72, h['opt']),
                        (87, h['p'], h['rng'], 0x3fe0000000000000, 1, 0), (999,)],
             160, 3, operation_error=5)
    for path, raw in inputs.items():
        assert path.read_bytes() == raw, path
    report.update(status='PASS', input_custody_after=True,
                  input_sha256={str(path): SHA(raw) for path, raw in inputs.items()},
                  all_pytorch_complete=False)
    save()
    print(json.dumps({key: report[key] for key in ('status', 'native_probe_processes',
                                                  'interactive_frames', 'complete_owner_bytes_compared')}))


if __name__ == '__main__':
    main()
