#!/usr/bin/env python3
"""Compile real old/new HIA units into one checked owned handoff process.

Only symbol names are changed, with an exact byte inverse. No Python pointer is
passed to HIA and no owner descriptor is synthesized or copied by this runner.
The unchanged HIA controller moves the actual owner between checked step calls.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
NEW_SOURCE = '666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b'
NEW_IMAGE = '7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5'
OLD_SOURCE = '1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae'
OLD_IMAGE = '6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe'
WIRE = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
OBSERVER = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
CONTROLLER = '4093b8a1a4ce1b19d6943f457649355c950cfba50743ac7afe8a8396437cc7ee'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def rename(raw, prefix, reverse=False):
    def symbol(name):
        if reverse:
            assert name.startswith(prefix)
            return name[len(prefix):]
        return prefix + name
    raw = re.sub(rb'^(hia-asm\.v1 hepta\.hia\.program\.v8 )(\w+)$',
                 lambda m: m[1] + symbol(m[2]), raw, flags=re.M)
    raw = re.sub(rb'^(fn )(\w+)(\()', lambda m: m[1] + symbol(m[2]) + m[3], raw, flags=re.M)
    raw = re.sub(rb'^exports\(([^)]*)\)$', lambda m: b'exports(' +
                 b','.join(symbol(n) for n in m[1].split(b',')) + b')', raw, flags=re.M)
    return re.sub(rb'(\{"function":")(\w+)("\})', lambda m: m[1] + symbol(m[2]) + m[3], raw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'previous-build', 'stack-build', 'protocol', 'controller', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def bind(path, expected=None):
        path = path.resolve()
        raw = path.read_bytes()
        if expected:
            assert sha(raw) == expected, (path, sha(raw), expected)
        inputs[path] = raw
        return raw

    bind(Path(__file__))
    compiler = a.stack_build.resolve() / 'native/stack-module-compiler'
    linker = a.stack_build.resolve() / 'native/stack-linker-session'
    bind(compiler, COMPILER)
    bind(linker, LINKER)
    bind(a.protocol, WIRE)
    bind(a.protocol.with_name('owner_client.py'), OBSERVER)
    controller = bind(a.controller, CONTROLLER)
    spec = importlib.util.spec_from_file_location('retained_owned_handoff_wire', a.protocol)
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    report = {'schema': 'heptatorch.retained-gradient-owned-handoff.v1', 'status': 'RUNNING',
              'source_sha256': NEW_SOURCE, 'native_sha256': NEW_IMAGE,
              'previous_source_sha256': OLD_SOURCE, 'previous_native_sha256': OLD_IMAGE,
              'compiler_sha256': COMPILER, 'linker_sha256': LINKER,
              'controller_sha256': CONTROLLER, 'test_sha256': sha(Path(__file__).read_bytes()),
              'raw_pointer_import': False, 'owner_descriptor_memcpy': False,
              'full_rewrite_complete': False, 'pytorch_api_denominator': 2590,
              'units': [], 'cases': [], 'frames': []}

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def compile_one(name, raw):
        (out / (name + '.hepta')).write_bytes(raw)
        result = subprocess.run([str(compiler)], input=raw, capture_output=True,
                                env={'PATH': ''}, timeout=60)
        (out / (name + '.o')).write_bytes(result.stdout)
        (out / (name + '.stderr')).write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
        return result.stdout

    objects = []
    for label, directory, source_pin, image_name, image_pin in (
        ('old', a.previous_build.resolve(), OLD_SOURCE, 'tensor-gradient-adam-distributions-session', OLD_IMAGE),
        ('new', a.build.resolve(), NEW_SOURCE, 'tensor-retained-session', NEW_IMAGE),
    ):
        bind(directory / 'runtime.hepta', source_pin)
        bind(directory / image_name, image_pin)
        receipt = json.loads(bind(directory / 'build.json'))
        assert receipt['status'] == 'PASS' and receipt['source_sha256'] == source_pin
        assert receipt['native_sha256'] == image_pin and receipt['module_count'] == len(receipt['units']) == 9
        for row in receipt['units']:
            raw = bind(directory / (row['name'] + '.hepta'), row['source_sha256'])
            bind(directory / (row['name'] + '.o'), row['object_sha256'])
            prefix = (label + '_').encode()
            transformed = rename(raw, prefix)
            assert rename(transformed, prefix, reverse=True) == raw
            renamed_name = label + '_' + row['name']
            obj = compile_one(renamed_name, transformed)
            objects.append(obj)
            report['units'].append({'role': label, 'original_name': row['name'],
                                    'original_source_sha256': sha(raw),
                                    'original_object_sha256': row['object_sha256'],
                                    'renamed_source_sha256': sha(transformed),
                                    'renamed_object_sha256': sha(obj), 'inverse_byte_identical': True})
        save()
    objects.append(compile_one('controller', controller))
    request = b'HIASTKS1' + struct.pack('<3Q', len(objects), 4, 0) + b'step'
    request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
    (out / 'link.request').write_bytes(request)
    result = subprocess.run([str(linker)], input=request, capture_output=True, env={'PATH': ''}, timeout=90)
    (out / 'link.stderr').write_bytes(result.stderr)
    assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
    image = out / 'handoff-session'
    image.write_bytes(result.stdout)
    image.chmod(0o500)
    report.update(linked_modules=len(objects), qualification_image_sha256=sha(result.stdout))
    save()
    session = wire.Session(image, directory=out / 'transport')
    rollback_bytes = 0

    def route(which, commands=(), error=0):
        nonlocal rollback_bytes
        before = session.owner() if error else None
        request = struct.pack('<Q', which) + (wire.packet(commands) if which < 2 else b'')
        reply = session.exchange(request)
        status, rows = wire.parse(reply)
        assert status == error, (which, status, error)
        if error:
            assert not rows and session.owner() == before
            rollback_bytes += before['length']
        report['frames'].append({'route': which, 'status': status, 'request_sha256': sha(request),
                                 'response_sha256': sha(reply), 'record_count': len(rows)})
        return rows

    # Previous05 state has actual Saved, Generator and public gradient objects.
    rows = route(0, [wire.full((3,), 7, 1), (46, wire.reference(1)), (60, 19),
                     (63, wire.reference(3)), wire.full((3,), 1),
                     (58, wire.reference(1), wire.reference(5), 1),
                     (91, wire.reference(1)), (53,)])
    tensor, saved, generator, old_gradient = (rows[i]['value'] for i in (0, 1, 2, 6))
    generator_state = rows[3]['payload']
    assert len(generator_state) == 5056 and rows[-1]['value'] == 1
    previous_owner = session.owner()
    assert previous_owner['bytes'][64:72] == b'HTGAD002'
    route(1, [(93, tensor), (95, tensor, wire.bits(999)), (61, generator, 99)], error=7)
    got = route(0, [(39, tensor, 0, 3), (49, saved, 0, 3), (63, generator),
                    (39, old_gradient, 0, 3), (53,)])
    assert got[0]['payload'] == got[1]['payload'] == struct.pack('<3f', 7, 7, 7)
    assert got[2]['payload'] == generator_state and got[3]['payload'] == struct.pack('<3f', 1, 1, 1)
    assert got[-1]['value'] == 1
    old_pointer = session.owner()['pointer']
    route(2)
    assert session.owner()['length'] == 0 and not session.mapped(old_pointer)
    report['cases'].append({'name': 'old05-to-retained01-rejection-and-recovery', 'status': 'PASS',
                            'owner_pointer_length_and_bytes_unchanged': True,
                            'Saved_Generator_and_gradient_objects_preserved': True,
                            'actual_closed_owner_mapping_absent': True})
    save()

    # New01 state contains an enabled retained nonleaf and a real materialized G.
    rows = route(1, [wire.full((3,), .5, 1), wire.full((), 2),
                     (2, 3, wire.reference(1), wire.reference(2)), (93, wire.reference(3)),
                     wire.full((3,), 1), (58, wire.reference(3), wire.reference(5), 1),
                     (91, wire.reference(3)), (60, 19), (63, wire.reference(8)), (53,)])
    tensor, seed, gradient, generator = (rows[i]['value'] for i in (2, 4, 6, 7))
    generator_state = rows[8]['payload']
    assert rows[3]['value'] == rows[-1]['value'] == 1
    new_owner = session.owner()
    assert new_owner['bytes'][64:72] == b'HTRET001'
    route(0, [(95, tensor, wire.bits(999)), (33, 0)], error=7)
    got = route(1, [(94, tensor), (91, tensor), (39, tensor, 0, 3), (39, gradient, 0, 3),
                    (63, generator), (13, gradient), (53,)])
    assert got[0]['value'] == 1 and got[1]['value'] == gradient
    assert got[2]['payload'] == got[3]['payload'] == struct.pack('<3f', 1, 1, 1)
    assert got[4]['payload'] == generator_state and wire.meta(got[5])['version'] == 0 and got[-1]['value'] == 1
    got = route(1, [(58, tensor, seed, 1), (91, tensor), (94, tensor), (53,)])
    new_gradient = got[1]['value']
    assert got[0]['value'] == 0 and got[2]['value'] == got[3]['value'] == 1 and new_gradient != gradient
    got = route(1, [(39, new_gradient, 0, 3), (39, gradient, 0, 3), (13, new_gradient)])
    assert got[0]['payload'] == struct.pack('<3f', 2, 2, 2)
    assert got[1]['payload'] == struct.pack('<3f', 1, 1, 1) and wire.meta(got[2])['version'] == 0
    new_pointer = session.owner()['pointer']
    route(3)
    assert session.owner()['length'] == 0 and not session.mapped(new_pointer)
    report['cases'].append({'name': 'retained01-to-old05-rejection-and-recovery', 'status': 'PASS',
                            'owner_pointer_length_and_bytes_unchanged': True,
                            'retained_flag_and_materialized_gradient_preserved': True,
                            'returned_owner_can_execute_next_prehook': True,
                            'external_old_gradient_unchanged': True,
                            'actual_closed_owner_mapping_absent': True})
    report['transport'] = session.finish()
    assert all(path.read_bytes() == raw for path, raw in inputs.items())
    report.update(status='PASS', case_count=2, passed=2, failed=0, frame_count=session.frames,
                  whole_owner_rollback_bytes=rollback_bytes, all_source_custody_unchanged=True,
                  inputs_sha256={str(path): sha(raw) for path, raw in inputs.items()})
    save()
    (out / 'executed-test.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: value for key, value in report.items() if key not in ('units', 'frames', 'inputs_sha256')}, indent=2))


if __name__ == '__main__':
    main()
