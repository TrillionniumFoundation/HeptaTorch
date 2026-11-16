#!/usr/bin/env python3
"""Real linear-owner handoff between frozen retained446 and dtype03 runtimes.

Both complete native programs are compiled with reversible symbol prefixes and
linked with the already qualified HIA ownership controller. No pointer enters
the wire and no host code manufactures/copies an owner descriptor. This is a
format-isolation and resource test; no new Torch numerical comparison is claimed.
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

COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
OLD_SOURCE = '666230f05cc6942470006d1f184135321d8cbc26a6c6ba0bf7e3cf9857fe515b'
OLD_IMAGE = '7568c0a4ec40d82d3a0b46d1e8a27499dd96e8fe3fd15433523131a196da89b5'
NEW_SOURCE = '6ffe509afce28ff15705b4c7b269eaef8f8f89f9d8b2486d4d9fa45b6be66ef3'
NEW_IMAGE = '830f0b4ef8d6ad8c6a105be56d383628b45c84e3e43905fee6812c7c52d35a51'
WIRE = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
OBSERVER = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
CONTROLLER = '4093b8a1a4ce1b19d6943f457649355c950cfba50743ac7afe8a8396437cc7ee'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def rename(raw, prefix, reverse=False):
    # Same symbol-only transformation used by the frozen 428/446 controller
    # qualification. The complete original unit must be recovered byte for byte.
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
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'previous-build', 'stack-build', 'protocol', 'controller', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    session = None
    report = {'schema': 'heptatorch.dtype03-independent-owned-handoff.v1', 'status': 'RUNNING',
              'source_sha256': NEW_SOURCE, 'native_sha256': NEW_IMAGE,
              'previous_source_sha256': OLD_SOURCE, 'previous_native_sha256': OLD_IMAGE,
              'compiler_sha256': COMPILER, 'linker_sha256': LINKER,
              'controller_sha256': CONTROLLER, 'test_sha256': sha(Path(__file__).read_bytes()),
              'raw_pointer_import': False, 'owner_descriptor_memcpy': False,
              'Torch_executed_here': False, 'full_rewrite_complete': False,
              'pytorch_api_denominator': 2590, 'whole_pytorch_gate': 'HOLD',
              'units': [], 'cases': [], 'frames': [], 'rollback_bytes': 0}

    def bind(path, expected=None):
        path = path.resolve()
        raw = path.read_bytes()
        if expected:
            assert sha(raw) == expected, (path, sha(raw), expected)
        inputs[path] = raw
        return raw

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def compile_one(name, raw, compiler):
        (out / (name + '.hepta')).write_bytes(raw)
        result = subprocess.run([str(compiler)], input=raw, capture_output=True,
                                env={'PATH': ''}, timeout=60)
        (out / (name + '.o')).write_bytes(result.stdout)
        (out / (name + '.stderr')).write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF'), (name, result.returncode)
        return result.stdout

    try:
        bind(Path(__file__))
        compiler = a.stack_build.resolve() / 'native/stack-module-compiler'
        linker = a.stack_build.resolve() / 'native/stack-linker-session'
        bind(compiler, COMPILER)
        bind(linker, LINKER)
        bind(a.protocol, WIRE)
        bind(a.protocol.with_name('owner_client.py'), OBSERVER)
        controller = bind(a.controller, CONTROLLER)
        spec = importlib.util.spec_from_file_location('dtype_owned_handoff_wire', a.protocol)
        wire = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wire)
        objects = []
        for label, directory, source_pin, image_name, image_pin, count in (
            ('old', a.previous_build.resolve(), OLD_SOURCE, 'tensor-retained-session', OLD_IMAGE, 9),
            ('new', a.build.resolve(), NEW_SOURCE, 'tensor-dtype-dropout-session', NEW_IMAGE, 11),
        ):
            bind(directory / 'runtime.hepta', source_pin)
            bind(directory / image_name, image_pin)
            receipt = json.loads(bind(directory / 'build.json'))
            assert receipt['status'] == 'PASS' and receipt['source_sha256'] == source_pin
            assert receipt['native_sha256'] == image_pin and len(receipt['units']) == count
            assert receipt['compiler_sha256'] == COMPILER and receipt['linker_sha256'] == LINKER
            assert receipt['link_exit'] == 0
            for row in receipt['units']:
                assert row['compiler_exit'] == 0
                raw = bind(directory / (row['name'] + '.hepta'), row['source_sha256'])
                bind(directory / (row['name'] + '.o'), row['object_sha256'])
                prefix = (label + '_').encode()
                transformed = rename(raw, prefix)
                assert rename(transformed, prefix, reverse=True) == raw
                name = label + '_' + row['name']
                obj = compile_one(name, transformed, compiler)
                objects.append(obj)
                report['units'].append({'role': label, 'original_name': row['name'],
                    'original_source_sha256': sha(raw), 'original_object_sha256': row['object_sha256'],
                    'renamed_source_sha256': sha(transformed), 'renamed_object_sha256': sha(obj),
                    'inverse_byte_identical': True, 'compiler_exit': 0})
                save()
        objects.append(compile_one('controller', controller, compiler))
        request = b'HIASTKS1' + struct.pack('<3Q', len(objects), 4, 0) + b'step'
        request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
        (out / 'link.request').write_bytes(request)
        result = subprocess.run([str(linker)], input=request, capture_output=True, env={'PATH': ''}, timeout=90)
        (out / 'link.stderr').write_bytes(result.stderr)
        (out / 'link.exit.json').write_text(json.dumps({'exit_code': result.returncode}) + '\n')
        assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
        image = out / 'handoff-session'
        image.write_bytes(result.stdout)
        image.chmod(0o500)
        report.update(linked_modules=len(objects), qualification_image_sha256=sha(result.stdout))
        save()
        session = wire.Session(image, directory=out / 'transport')

        def route(which, commands=(), error=0):
            before = session.owner() if error else None
            request = struct.pack('<Q', which) + (wire.packet(commands) if which < 2 else b'')
            reply = session.exchange(request)
            status, rows = wire.parse(reply)
            number = len(report['frames'])
            item = {'route': which, 'status': status, 'request_sha256': sha(request),
                    'response_sha256': sha(reply), 'record_count': len(rows)}
            report['frames'].append(item)
            assert status == error, (which, status, error)
            if error:
                after = session.owner()
                (out / f'frame-{number}-owner-before.bin').write_bytes(before['bytes'])
                (out / f'frame-{number}-owner-after.bin').write_bytes(after['bytes'])
                item.update(owner_pointer_before=before['pointer'], owner_pointer_after=after['pointer'],
                            owner_length=before['length'], owner_sha256=sha(before['bytes']))
                assert not rows and after == before
                report['rollback_bytes'] += before['length']
            save()
            return rows

        # Actual retained446 producer state: Saved, full Generator, and public G.
        rows = route(0, [wire.full((3,), 7, 1), (46, wire.reference(1)), (60, 19),
                         (63, wire.reference(3)), wire.full((3,), 1),
                         (58, wire.reference(1), wire.reference(5), 1),
                         (91, wire.reference(1)), (53,)])
        tensor, saved, generator, gradient = (rows[i]['value'] for i in (0, 1, 2, 6))
        generator_state = rows[3]['payload']
        assert len(generator_state) == 5056 and rows[-1]['value'] == 1
        assert session.owner()['bytes'][64:72] == b'HTRET001'
        route(1, [(101, tensor), (95, tensor, wire.bits(999)), (61, generator, 99)], error=7)
        got = route(0, [(39, tensor, 0, 3), (49, saved, 0, 3), (63, generator),
                        (39, gradient, 0, 3), (53,)])
        assert got[0]['payload'] == got[1]['payload'] == struct.pack('<3f', 7, 7, 7)
        assert got[2]['payload'] == generator_state and got[3]['payload'] == struct.pack('<3f', 1, 1, 1)
        assert got[-1]['value'] == 1
        old_pointer = session.owner()['pointer']
        route(2)
        assert session.owner()['length'] == 0 and not session.mapped(old_pointer)
        report['cases'].append({'name': 'retained446-to-dtype03-rejection-and-recovery', 'status': 'PASS',
            'owner_pointer_length_and_bytes_unchanged': True, 'Saved_Generator_and_gradient_objects_preserved': True,
            'actual_close_unmapped_owner_before_process_exit': True})
        save()

        # New dtype03 state really contains a one-byte bool mask, its public
        # Saved/Weak roles, full Generator and a retained nonleaf's real G.
        rows = route(1, [wire.full((3,), .5, 1), wire.full((), 2),
                         (2, 3, wire.reference(1), wire.reference(2)), (93, wire.reference(3)),
                         wire.full((3,), 1), (58, wire.reference(3), wire.reference(5), 1),
                         (91, wire.reference(3)), (60, 19),
                         (99, wire.reference(3), wire.reference(8), 0x3fe0000000000000, 1), (53,)])
        tensor, seed, gradient, generator = (rows[i]['value'] for i in (2, 4, 6, 7))
        dropout = rows[8]['value']
        mask = struct.unpack('<Q', rows[8]['payload'])[0]
        assert rows[3]['value'] == rows[-1]['value'] == 1
        state = session.owner()['bytes']
        assert state[64:72] == b'HTDTYP03' and state[139264:139272] == b'HTODTY03'
        got = route(1, [(96, mask), (98, mask, 0, 3), (13, mask), (46, mask), (42, mask), (63, generator)])
        mask_info, mask_raw, mask_meta = got[0], got[1], got[2]
        mask_saved, mask_weak, generator_state = got[3]['value'], got[4]['value'], got[5]['payload']
        assert struct.unpack('<4Q', mask_info['payload'])[:2] == (1, 1)
        assert wire.meta(mask_meta)['storage_bytes'] == len(mask_raw['payload']) == 3
        assert set(mask_raw['payload']) <= {0, 1}
        route(0, [(102, mask), (95, tensor, wire.bits(999)), (33, 0)], error=7)
        got = route(1, [(94, tensor), (91, tensor), (39, tensor, 0, 3), (39, gradient, 0, 3),
                        (63, generator), (96, mask), (98, mask, 0, 3), (13, mask),
                        (56, mask_saved), (47, mask_saved), (43, mask_weak), (53,)])
        assert got[0]['value'] == 1 and got[1]['value'] == gradient
        assert got[2]['payload'] == got[3]['payload'] == struct.pack('<3f', 1, 1, 1)
        assert got[4]['payload'] == generator_state
        assert got[5]['payload'] == mask_info['payload'] and got[6]['payload'] == mask_raw['payload']
        assert got[7]['payload'] == mask_meta['payload'] and got[-1]['value'] == 1
        unpacked, hold = got[8]['value'], got[10]['value']
        assert unpacked != 0 and hold != 0
        unpacked_rows = route(1, [(98, unpacked, 0, 3), (96, unpacked), (13, unpacked),
                                  (41, unpacked), (45, hold)])
        assert unpacked_rows[0]['payload'] == mask_raw['payload']
        assert struct.unpack('<4Q', unpacked_rows[1]['payload'])[:2] == (1, 1)
        assert wire.meta(unpacked_rows[2])['storage_id'] == wire.meta(mask_meta)['storage_id']
        assert wire.meta(unpacked_rows[2])['version_id'] == wire.meta(mask_meta)['version_id']
        got = route(1, [(58, tensor, seed, 1), (91, tensor), (94, tensor), (53,)])
        new_gradient = got[1]['value']
        assert got[0]['value'] == 0 and got[2]['value'] == got[3]['value'] == 1 and new_gradient != gradient
        got = route(1, [(39, new_gradient, 0, 3), (39, gradient, 0, 3), (13, new_gradient), (96, dropout, 1)])
        assert got[0]['payload'] == struct.pack('<3f', 2, 2, 2)
        assert got[1]['payload'] == struct.pack('<3f', 1, 1, 1) and wire.meta(got[2])['version'] == 0
        assert got[3]['value'] == mask
        pointer = session.owner()['pointer']
        route(3)
        assert session.owner()['length'] == 0 and not session.mapped(pointer)
        report['cases'].append({'name': 'dtype03-to-retained446-rejection-and-recovery', 'status': 'PASS',
            'owner_pointer_length_and_bytes_unchanged': True, 'actual_bool_mask_Saved_Weak_Tensor_roles_preserved': True,
            'retained_flag_and_materialized_gradient_preserved': True, 'full_Generator_state_preserved': True,
            'returned_owner_executes_next_prehook': True, 'old_public_gradient_unchanged': True,
            'actual_close_unmapped_owner_before_process_exit': True})
        report.update(status='PASS', case_count=2, passed=2, failed=0)
    except BaseException as error:
        report.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
    finally:
        if session is not None:
            report['frame_count'] = session.frames
            try:
                session.finish()
                report['transport_exit_code'] = session.exit_code
            except BaseException as error:
                report.update(status='FAIL', transport_error=repr(error))
        report['all_source_custody_unchanged'] = all(path.read_bytes() == raw for path, raw in inputs.items())
        if not report['all_source_custody_unchanged']:
            report['status'] = 'FAIL'
        report['inputs'] = {str(path): {'bytes': len(raw), 'sha256': sha(raw)} for path, raw in inputs.items()}
        (out / 'executed-test.py').write_bytes(Path(__file__).read_bytes())
        save()
    print(json.dumps({key: report.get(key) for key in ('status', 'case_count', 'passed', 'failed', 'linked_modules',
                                                     'frame_count', 'rollback_bytes', 'error')}))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
