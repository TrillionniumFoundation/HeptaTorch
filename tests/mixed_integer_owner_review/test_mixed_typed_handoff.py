#!/usr/bin/env python3
"""Real linear owned handoff between immutable I64-07 and mixed Saved-T-10.

Both complete programs are symbol-prefixed with exact inverses and compiled by
the real HIA checker. The unchanged HIA controller consumes/returns each owner
and calls the matching close. The host never copies an owner descriptor.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import struct
import subprocess
import traceback

import resource_support as support
from resource_support import Custody, State, create_packet, dump_raw, integer_raw, sha, write_json

OLD_SOURCE = 'adf6cae67666273aa6fb77989080ca3b1e3632188d2d7956a6d8b71808787e62'
OLD_IMAGE = 'b7324639bade3312981d1cbf9c2491b9fcff04fb6c3f808e9c5e1a7c2ca88f08'
NEW_SOURCE = '5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49'
NEW_IMAGE = '69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6'
SESSION_LINKER = 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
CONTROLLER = '4093b8a1a4ce1b19d6943f457649355c950cfba50743ac7afe8a8396437cc7ee'
SUPPORT = '07248452cef312f264ccc4279479765542084c61217916eb364cf4ba3cdc4d88'


def rename(raw, prefix, reverse=False):
    def symbol(name):
        if reverse:
            assert name.startswith(prefix)
            return name[len(prefix):]
        return prefix+name
    raw = re.sub(rb'^(hia-asm\.v1 hepta\.hia\.program\.v8 )(\w+)$',
                 lambda m: m[1]+symbol(m[2]), raw, flags=re.M)
    raw = re.sub(rb'^(fn )(\w+)(\()', lambda m: m[1]+symbol(m[2])+m[3], raw, flags=re.M)
    raw = re.sub(rb'^exports\(([^)]*)\)$', lambda m: b'exports('+b','.join(symbol(n) for n in m[1].split(b','))+b')', raw, flags=re.M)
    return re.sub(rb'(\{"function":")(\w+)("\})', lambda m: m[1]+symbol(m[2])+m[3], raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'previous-build', 'stack-build', 'protocol', 'controller', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    custody = Custody(out)
    session = None
    report = {'schema': 'heptatorch.i64-independent-owned-profile-handoff.v1', 'status': 'RUNNING',
        'source_sha256': NEW_SOURCE, 'native_sha256': NEW_IMAGE,
        'previous_source_sha256': OLD_SOURCE, 'previous_native_sha256': OLD_IMAGE,
        'compiler_sha256': support.COMPILER, 'linker_sha256': SESSION_LINKER,
        'controller_sha256': CONTROLLER, 'units': [], 'frames': [], 'cases': [],
        'raw_pointer_import': False, 'owner_descriptor_memcpy': False,
        'Torch_executed_here': False, 'whole_pytorch_gate': 'HOLD',
        'pytorch_api_denominator': 2590, 'full_pytorch_rewrite_complete': False,
        'rollback_bytes': 0}

    def save():
        write_json(out/'report.json', report)

    def compile_one(name, raw, compiler):
        (out/(name+'.hepta')).write_bytes(raw)
        result = subprocess.run([str(compiler)], input=raw, capture_output=True, env={'PATH': ''}, timeout=60)
        (out/(name+'.o')).write_bytes(result.stdout)
        (out/(name+'.stderr')).write_bytes(result.stderr)
        write_json(out/(name+'.exit.json'), {'exit_code': result.returncode})
        assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
        return result.stdout

    try:
        custody.bind(Path(__file__), copy='executed-test.py')
        custody.bind(Path(support.__file__), SUPPORT, 'resource_support.py')
        compiler = a.stack_build.resolve()/'native/stack-module-compiler'
        linker = a.stack_build.resolve()/'native/stack-linker-session'
        custody.bind(compiler, support.COMPILER)
        custody.bind(linker, SESSION_LINKER)
        custody.bind(a.protocol, support.PROTOCOL, 'wire/protocol.py')
        custody.bind(a.protocol.with_name('owner_client.py'), support.CLIENT, 'wire/owner_client.py')
        controller = custody.bind(a.controller, CONTROLLER, 'state_handoff.hepta')
        spec = importlib.util.spec_from_file_location('integer_handoff_wire', a.protocol)
        wire = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wire)
        objects = []
        for label, directory, source, image_name, image in (
            ('old', a.previous_build.resolve(), OLD_SOURCE, 'tensor-integer-session', OLD_IMAGE),
            ('new', a.build.resolve(), NEW_SOURCE, 'tensor-mixed-autograd-session', NEW_IMAGE)):
            custody.bind(directory/'runtime.hepta', source, label+'/runtime.hepta')
            custody.bind(directory/image_name, image, label+'/'+image_name)
            b = json.loads(custody.bind(directory/'build.json', copy=label+'/build.json'))
            assert b['status'] == 'PASS' and b['source_sha256'] == source and b['native_sha256'] == image
            assert b['compiler_sha256'] == support.COMPILER and b['linker_sha256'] == SESSION_LINKER
            assert b['link_exit'] == 0 and len(b['units']) == 11
            for unit in b['units']:
                assert unit['compiler_exit'] == 0
                name = unit['name']
                raw = custody.bind(directory/(name+'.hepta'), unit['source_sha256'], label+'/'+name+'.hepta')
                custody.bind(directory/(name+'.o'), unit['object_sha256'], label+'/'+name+'.o')
                prefix = (label+'_').encode()
                transformed = rename(raw, prefix)
                assert rename(transformed, prefix, reverse=True) == raw
                obj = compile_one(label+'_'+name, transformed, compiler)
                objects.append(obj)
                report['units'].append({'role': label, 'name': name,
                    'original_source_sha256': sha(raw), 'original_object_sha256': unit['object_sha256'],
                    'renamed_source_sha256': sha(transformed), 'renamed_object_sha256': sha(obj),
                    'compiler_exit': 0, 'inverse_byte_identical': True})
                save()
        objects.append(compile_one('controller', controller, compiler))
        request = b'HIASTKS1'+struct.pack('<3Q', len(objects), 4, 0)+b'step'
        request += b''.join(struct.pack('<Q', len(obj))+obj for obj in objects)
        (out/'link.request').write_bytes(request)
        result = subprocess.run([str(linker)], input=request, capture_output=True, env={'PATH': ''}, timeout=90)
        (out/'link.stderr').write_bytes(result.stderr)
        write_json(out/'link.exit.json', {'exit_code': result.returncode})
        assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
        image = out/'handoff-session'
        image.write_bytes(result.stdout)
        image.chmod(0o500)
        assert len(objects) == 23
        report.update(linked_modules=23, qualification_image_sha256=sha(result.stdout))
        session = wire.Session(image, directory=out/'transport')

        def route(which, commands=(), expected=0, packet=None):
            before = session.owner() if expected else None
            request = struct.pack('<Q', which)+(wire.packet(commands) if packet is None and which < 2 else packet or b'')
            response = session.exchange(request)
            status, rows = wire.parse(response)
            index = len(report['frames'])
            event = {'route': which, 'status': status, 'expected_status': expected,
                     'request_sha256': sha(request), 'response_sha256': sha(response), 'records': len(rows)}
            report['frames'].append(event)
            assert status == expected, (index, status, expected)
            if expected:
                after = session.owner()
                event['before'] = dump_raw(out, 'frame-'+str(index)+'-before', before['bytes'])
                event['after'] = dump_raw(out, 'frame-'+str(index)+'-after', after['bytes'])
                event['pointer_before'] = before['pointer']
                event['pointer_after'] = after['pointer']
                assert not rows and after == before
                report['rollback_bytes'] += before['length']
            save()
            return rows

        # Old07 state genuinely contains captured AD, an actual bool Saved mask,
        # a public gradient, and a full5056-byte Generator.
        rows = route(0, [wire.full([3], .5, 1), (46, wire.reference(1)), (60, 1729),
                        wire.full([3], 1), (58, wire.reference(1), wire.reference(4), 1),
                        (91, wire.reference(1)),
                        (99, wire.reference(1), wire.reference(3), 0x3fe0000000000000, 1), (53,)])
        tensor, saved, gen, gradient = (rows[i]['value'] for i in (0, 1, 2, 5))
        mask = struct.unpack('<Q', rows[6]['payload'])[0]
        assert rows[-1]['value'] == 1
        old_got = route(0, [(39, tensor, 0, 3), (39, gradient, 0, 3), (63, gen),
                            (98, mask, 0, 3), (13, mask), (96, mask), (49, saved, 0, 3)])
        assert session.owner()['bytes'][64:72] == b'HTINT002'
        assert struct.unpack('<4Q', old_got[5]['payload'])[:2] == (1, 1)
        route(1, [(95, tensor, wire.bits(999)), (61, gen, 99), (106, tensor, 1)], expected=7)
        old_again = route(0, [(39, tensor, 0, 3), (39, gradient, 0, 3), (63, gen),
                             (98, mask, 0, 3), (13, mask), (96, mask), (49, saved, 0, 3), (53,)])
        assert old_again[:-1] == old_got and old_again[-1]['value'] == 1
        assert old_got[0]['payload'] == old_got[6]['payload'] == struct.pack('<3f', .5, .5, .5)
        assert old_got[1]['payload'] == struct.pack('<3f', 1, 1, 1) and len(old_got[2]['payload']) == 5056
        pointer = session.owner()['pointer']
        route(2)
        assert session.owner()['length'] == 0 and not session.mapped(pointer)
        report['cases'].append({'name': 'integer07_to_mixed10_reject_return_recover_close', 'status': 'PASS',
            'complete_owner_pointer_length_bytes_preserved': True,
            'F32_gradient_Saved_bool_mask_full_Generator_preserved': True,
            'matching_native_close_unmapped_owner_while_process_alive': True})
        save()

        base = route(1, packet=create_packet(2, [3], integer_raw([3, 5, 7])))[0]['value']
        config = struct.pack('<10Q', 1, 2, 0, 0, 0, 1, 0, 0, 0, 1)
        rows = route(1, [(69, base, wire.Payload(config), 80), (46, wire.reference(1)),
                        (42, base), (60, 1729), wire.full([2], .5, 1),
                        (103, 3, wire.reference(1), wire.reference(5)), (53,)])
        view, saved, weak, gen, f, y = (rows[i]['value'] for i in range(6))
        assert rows[-1]['value'] == 1
        initial = State(session.owner()['bytes'])
        base_node, base_desc = initial.tensor_node(base)
        route(1, [(41, base)])
        state = State(session.owner()['bytes'])
        desc = state.node(base_node)
        assert desc[200//8] == 0 and desc[600//8] == desc[208//8] == 1
        assert state.tensor_node(view)[1][584//8] != 0
        # Both base ownership and the new Saved-original-T role are real.
        vi, vd = state.tensor_node(view)
        yi, yd = state.tensor_node(y)
        assert yd[584//8] > 65535 and vd[600//8] == 1 and vd[208//8] == 2
        hold_row = state.row(yd[584//8], 6)
        pairs = struct.unpack('<4Q', state.storage_bytes(hold_row[2]))
        assert pairs == (vi, vd[320//8], 0, 0)
        assert route(1, [(107, y, 0)])[0]['value'] == view
        expected = route(1, [(96, view), (98, view, 0, 2), (13, view), (63, gen), (98, y, 0, 2)])
        assert struct.unpack('<4Q', expected[0]['payload'])[:2] == (2, 8)
        assert expected[1]['payload'] == integer_raw([5, 7]) and expected[4]['payload'] == struct.pack('<2f', 2.5, 3.5)
        route(0, [(95, f, wire.bits(999)), (61, gen, 99), (33, 0)], expected=7)
        got = route(1, [(96, view), (98, view, 0, 2), (13, view), (63, gen), (98, y, 0, 2),
                        (56, saved), (43, weak), (53,)])
        assert got[:5] == expected and got[-1]['value'] == 1
        unpacked, hold = got[5]['value'], got[6]['value']
        got = route(1, [(98, unpacked, 0, 2), (96, unpacked), (13, unpacked), (41, unpacked), (45, hold),
                        (106, view, 1), (98, view, 0, 2), (13, view), (53,)])
        assert got[0]['payload'] == integer_raw([5, 7])
        assert struct.unpack('<4Q', got[1]['payload'])[:2] == (2, 8)
        assert wire.meta(got[2])['storage_id'] == wire.meta(expected[2])['storage_id']
        assert got[6]['payload'] == integer_raw([6, 8])
        assert wire.meta(got[7])['version'] == wire.meta(expected[2])['version']+1 and got[8]['value'] == 1
        pointer = session.owner()['pointer']
        route(3)
        assert session.owner()['length'] == 0 and not session.mapped(pointer)
        report['cases'].append({'name': 'mixed10_to_integer07_reject_return_recover_close', 'status': 'PASS',
            'complete_owner_pointer_length_bytes_preserved': True,
            'actual_I64_8byte_Saved_Weak_ultimate_base_and_promoted_F32_preserved': True,
            'actual_original_Tensor_Saved_private_Hold_preserved': True,
            'full_Generator_preserved': True, 'real_scalar_mutation_executes_after_recovery': True,
            'matching_native_close_unmapped_owner_while_process_alive': True})
        report.update(status='PASS', case_count=2, passed=2, failed=0)
    except BaseException as error:
        report.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
    finally:
        if session is not None:
            report['frame_count'] = session.frames
            try:
                session.finish()
                report['EOF_exit_code'] = session.exit_code
            except BaseException as error:
                report.update(status='FAIL', transport_error=repr(error))
        try:
            report['inputs'] = custody.finish()
            report['source_custody_after'] = True
        except BaseException as error:
            report.update(status='FAIL', custody_error=repr(error))
        save()
    print(json.dumps({k: report.get(k) for k in ('status', 'case_count', 'passed', 'linked_modules', 'frame_count', 'rollback_bytes', 'error')}))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
