#!/usr/bin/env python3
"""Six direct unsupported Half operations plus six separate failed FE transactions.

This is a separate protocol-boundary control, not a Torch numerical oracle and
not a replacement or edit of the original Half47/239 qualification.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys
import traceback
import types

sys.dont_write_bytecode = True
BRIDGE = 'f3b19a8dcc1a004323e033f414e801a52f12b0f0d556e6cbc005169d26fdef9e'
MASK = (1 << 64) - 1


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    with path.open('xb') as f:
        f.write((json.dumps(value, indent=2, sort_keys=True) + '\n').encode())


def frames(raw, magic):
    rows, at = [], 0
    while at < len(raw):
        assert at + 16 <= len(raw)
        head, size = struct.unpack_from('<8sQ', raw, at)
        assert head == magic and at + 16 + size <= len(raw)
        rows.append(raw[at + 16:at + 16 + size])
        at += 16 + size
    assert at == len(raw)
    return rows


def bind_original_method(inputs, function):
    """Bind the actual method code to its full pinned physical source."""
    path = Path(function.__globals__['__file__']).resolve()
    raw = inputs.bind(path)
    compiled = compile(raw, str(path), 'exec')
    found = []
    def visit(code):
        for value in code.co_consts:
            if isinstance(value, types.CodeType):
                if (value.co_qualname == function.__code__.co_qualname
                        and value.co_firstlineno == function.__code__.co_firstlineno):
                    found.append(value)
                visit(value)
    visit(compiled)
    assert len(found) == 1 and function.__code__ == found[0]
    assert function.__code__.co_filename == str(path)
    return dict(source_path=str(path), source_sha256=sha(raw),
                qualname=function.__code__.co_qualname, first_line=function.__code__.co_firstlineno,
                actual_method_code_equals_recompiled_original=True)


def reject_failed_half_fe_transaction(driver, wire, body, dtype, opcode):
    """Dedicated failed-frame scope; never alter the old single-sum-frame rule."""
    assert dtype in (5, 6) and opcode in (115, 118, 119)
    expected_raw = struct.pack('<2H', 0x3c00 if dtype == 5 else 0x3f80, 0x8000)
    expected = b'HTORCH03' + struct.pack('<3Q', 2, 224 + len(expected_raw), 0)
    expected += struct.pack('<8Q', 97, dtype, 160, 0, 0, 0, 0, 0)
    expected += struct.pack('<8Q', opcode, wire.reference(1), MASK, 0, 0, 1, 0, 0)
    expected += struct.pack('<8Q', 1, 0, 2, 0, 0, 0, 224, len(expected_raw)) + expected_raw
    assert body == expected
    commands = [list(struct.unpack_from('<8Q', body, at)) for at in (32, 96)]
    label = 'same-frame-factory-FE-refusal'
    maps = Path('/proc/%d/maps' % driver.n.s.pid)
    maps_before = maps.read_bytes()
    before, before_ref, before_raw = driver.n.capture(label + '/before')
    row = dict(index=len(driver.n.boundaries), label=label, before=before_ref,
               request_bytes=len(body), request_sha256=sha(body), commands=commands,
               frame_admission='new-explicit-failed-Half-FE-transaction-only')
    try:
        response = driver.n.s.exchange(body)
        row.update(response_bytes=len(response), response_sha256=sha(response))
        after, after_ref, after_raw = driver.n.capture(label + '/after')
        row['after'] = after_ref
        status, records = wire.parse(response)
        row.update(status=status, records=len(records))
        driver.operations.append(dict(label=label, request_sha256=sha(body), actual_status=status,
            expected_status=7, record_count=len(records), frame_admission=row['frame_admission']))
        driver.check(label + '/actual-frame-status', status, 7)
        driver.check(label + '/empty-records', records, [])
        driver.check(label + '/full-owner-byte-rollback', after_raw, before_raw)
        driver.check(label + '/same-owner-pointer', after_ref['pointer'], before_ref['pointer'])
        driver.check(label + '/same-owner-length', after_ref['length'], before_ref['length'])
        maps_after = maps.read_bytes()
        (driver.directory / (label + '.maps-before')).write_bytes(maps_before)
        (driver.directory / (label + '.maps-after')).write_bytes(maps_after)
        driver.check(label + '/all-mappings-unchanged', maps_after, maps_before)
        row['owner_check'] = dict(original_full_snapshot_predicates_before_and_after=True,
            actual_complete_owner_bytes_equal=True, actual_owner_pointer_and_length_equal=True,
            old_single_sum_frame_admission_invoked=False, intermediate_success_state_inferred=False)
    except BaseException as error:
        row['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        driver.n.boundaries.append(row)
        driver.n.journal.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
        driver.n.journal.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge-directory', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--build-directory', type=Path, required=True)
    parser.add_argument('--expected-build-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not sys.flags.optimize
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    raw = (args.bridge_directory / 'bridge.py').read_bytes()
    assert sha(raw) == BRIDGE
    bridge = types.ModuleType('current736_half6_original_bridge')
    bridge.__file__ = str((args.bridge_directory / 'bridge.py').resolve())
    exec(compile(raw, bridge.__file__, 'exec'), bridge.__dict__)
    report = {'schema': 'heptatorch.current736.half-direct6-and-FE6-refusals.v1', 'status': 'FAIL',
              'source_sha256': bridge.SOURCE, 'native_sha256': bridge.IMAGE,
              'actual_build_sha256': args.expected_build_sha256,
              'required_direct_cases': 6, 'required_FE_transaction_cases': 6, 'required_refusal_frames': 12,
              'cases': [], 'failures': [],
              'native_executed': False, 'Torch_executed': False, 'compiler_executed': False,
              'original47_and239_modified_or_rerun': False, 'full_project_status': 'HOLD'}
    bound = None
    try:
        with bridge.original_modules() as (originals, old_run, startup):
            bound = originals['common'].Inputs()
            bound.bind(Path(__file__).resolve())
            bound.bind(Path(sys.executable).resolve())
            adapter = bridge.Adapter(originals, old_run, startup)
            report['reader_manifest_sha256'] = adapter.bind_package(bound)
            report['source_admission'] = adapter.candidate_admission(args.repository, bound)
            report['cold_admission'] = adapter.cold_admission(args.build_directory, bridge.IMAGE,
                args.expected_build_sha256, args.repository, bound)
            owner = adapter.make_owner(args.repository, bridge.IMAGE, bound)
            wire, native = originals['native'].WIRE, originals['native'].Native
            reject = originals['controls'].reject
            report['original_method_bindings'] = {
                'Native.capture': bind_original_method(bound, native.capture),
                'Native.exchange': bind_original_method(bound, native.exchange),
                'owner.verify_snapshot': bind_original_method(bound, owner.verify_snapshot),
                'transport.exchange': bind_original_method(bound, originals['native'].CapturingSession.exchange),
                'transport.finish': bind_original_method(bound, originals['native'].CapturingSession.finish),
                'controls.reject': bind_original_method(bound, reject),
            }
            class Driver:
                check = old_run.Driver.check
                rng_unchanged = old_run.Driver.rng_unchanged
            for group in ('direct', 'FE-transaction'):
                for dtype, operation, opcode in [(d, op, code) for d in (5, 6) for op, code in
                                                 (('any', 118), ('all', 119), ('sum-Bool', 115))]:
                    directory = out / ('%s-dtype%d-%s' % (group, dtype, operation))
                    row = {'name': directory.name, 'group': group, 'dtype': dtype, 'opcode': opcode,
                           'expected_status': 7, 'status': 'FAIL', 'native_executed': None}
                    driver = Driver()
                    driver.checks, driver.operations = [], []
                    driver.directory, driver.n = directory, None
                    try:
                        driver.n = native(args.build_directory / 'heptatorch', args.build_directory / 'runtime.hepta',
                                          owner, directory, bridge.IMAGE)
                        row['native_executed'] = True
                        report['native_executed'] = True
                        driver.gen = driver.n.one(60, 1729)['value']
                        driver.rng = driver.n.one(63, driver.gen)['payload']
                        assert len(driver.rng) == 5056
                        input_raw = struct.pack('<2H', 0x3c00 if dtype == 5 else 0x3f80, 0x8000)
                        token = driver.n.create(dtype, [2], input_raw)
                        before = driver.n.tensor(token)
                        assert before['logical'] == input_raw and before['storage'] == input_raw
                        if group == 'direct':
                            command = (opcode, token, MASK, 0, 0, 1, 0, 0)
                            reject(driver, 'existing-public-token-refusal', wire.packet([command]), 7, mappings=True)
                        else:
                            config = struct.pack('<8Q', 1, 0, 2, 0, 0, 0, 224, len(input_raw))
                            body = b'HTORCH03' + struct.pack('<3Q', 2, 224 + len(input_raw), 0)
                            body += struct.pack('<8Q', 97, dtype, 160, 0, 0, 0, 0, 0)
                            body += struct.pack('<8Q', opcode, wire.reference(1), MASK, 0, 0, 1, 0, 0)
                            body += config + input_raw
                            reject_failed_half_fe_transaction(driver, wire, body, dtype, opcode)
                        driver.rng_unchanged('after-rejected-transaction')
                        after = driver.n.tensor(token)
                        for key in ('logical', 'storage', 'shape', 'strides', 'version', 'tensor_identity',
                                    'storage_identity', 'native_version_identity', 'requires_grad'):
                            driver.check('existing-input-preserved/' + key, after[key], before[key])
                        driver.n.one(41, token)
                        driver.rng_unchanged('after-input-release')
                        driver.n.one(66, driver.gen)
                        driver.check('final-native-audit', driver.n.one(53)['value'], 1)
                        snapshot, endpoint, _ = driver.n.capture('all-public-resources-released')
                        driver.check('all-public-Tensor-slots-released', sum(r[1] for r in snapshot['tables']['1']), 0)
                        driver.check('all-generators-released', sum(r[1] for r in snapshot['tables']['7']), 0)
                        row.update(status='PASS', original_input_sha256=sha(input_raw),
                                   complete_RNG_sha256=sha(driver.rng), cleanup_owner=endpoint)
                    except BaseException as error:
                        row['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
                    finally:
                        if driver.n is not None:
                            try:
                                row['transport'] = driver.n.finish()
                            except BaseException as error:
                                row['status'] = 'FAIL'
                                row['finish_error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
                            row['frames'] = driver.n.s.frames
                        transport_path = directory / 'raw/transport.json'
                        if transport_path.exists():
                            row['transport'] = json.loads(transport_path.read_bytes())
                            if row['transport'].get('pid'):
                                row['native_executed'] = True
                                report['native_executed'] = True
                        row['checks'], row['refusals'] = driver.checks, driver.operations
                        if directory.exists():
                            dump(directory / 'control-report.json', row)
                        report['cases'].append(row)
                        if row['status'] != 'PASS':
                            report['failures'].append({'case': row['name'], 'error': row.get('error'), 'finish_error': row.get('finish_error')})
            # Read actual streams and complete owner evidence, without sending
            # another native command or regenerating any expected result.
            refusal_frames = 0
            for row in report['cases']:
                if row['status'] != 'PASS':
                    continue
                directory = out / row['name']
                transport = row['transport']
                assert transport['exit_code'] == 0 and transport['stdout_EOF'] and transport['stderr_EOF']
                assert not transport['forced_cleanup'] and not transport['errors']
                assert transport['stderr_bytes'] == transport['extra_stdout_bytes'] == 0
                assert transport['actual_proc_exe_sha256'] == transport['selected_native_sha256'] == bridge.IMAGE
                requests_raw = (directory / 'raw/requests.frames.bin').read_bytes()
                responses_raw = (directory / 'raw/responses.frames.bin').read_bytes()
                assert sha(requests_raw) == transport['request_stream_sha256']
                assert sha(responses_raw) == transport['response_stream_sha256']
                requests = frames(requests_raw, b'HTOWNR01')
                responses = frames(responses_raw, b'HTOWNS01')
                assert len(requests) == len(responses) == row['frames'] == transport['frames']
                journal = [json.loads(line) for line in (directory / 'owner-frames.jsonl').read_bytes().splitlines()]
                assert len(journal) == len(requests)
                blobs = json.loads((directory / 'owner-blob-manifest.json').read_bytes())
                rejected = []
                for request, response, observation in zip(requests, responses, journal):
                    assert observation['request_sha256'] == sha(request) and observation['response_sha256'] == sha(response)
                    status, records = wire.parse(response)
                    if status == 0:
                        continue
                    assert status == 7 and records == []
                    rejected.append(observation)
                    owners = []
                    for endpoint in (observation['before'], observation['after']):
                        meta = blobs[endpoint['sha256']]
                        packed = (directory / meta['path']).read_bytes()
                        assert sha(packed) == meta['compressed_sha256'] and len(packed) == meta['compressed_bytes']
                        raw = gzip.decompress(packed)
                        assert len(raw) == endpoint['length'] == meta['bytes'] and sha(raw) == endpoint['sha256']
                        owner.verify_snapshot(owner.decode_owner(raw, endpoint['pointer']))
                        owners.append(raw)
                    assert owners[0] == owners[1]
                    assert observation['before']['pointer'] == observation['after']['pointer']
                    refusal_frames += 1
                assert len(rejected) == 1
                if row['group'] == 'direct':
                    assert len(rejected[0]['commands']) == 1 and rejected[0]['commands'][0][0] == row['opcode']
                    assert rejected[0]['commands'][0][2:] == [MASK, 0, 0, 1, 0, 0]
                    assert 'frame_admission' not in rejected[0]
                else:
                    assert len(rejected[0]['commands']) == 2 and rejected[0]['commands'][0][0] == 97
                    assert rejected[0]['commands'][0][1] == row['dtype']
                    assert rejected[0]['commands'][1] == [row['opcode'], wire.reference(1), MASK, 0, 0, 1, 0, 0]
                    assert rejected[0]['frame_admission'] == 'new-explicit-failed-Half-FE-transaction-only'
            report['readback_refusal_frames'] = refusal_frames
            report['all_inputs_unchanged'] = bound.unchanged()
            assert len(report['cases']) == 12
            assert [sum(r['group'] == g for r in report['cases']) for g in ('direct', 'FE-transaction')] == [6, 6]
            assert not report['failures'] and refusal_frames == 12
            assert len({row['transport']['pid'] for row in report['cases']}) == 12
            report.update(status='PASS_DIRECT6_AND_FE6_UNSUPPORTED_HALF_CONTROLS', native_children=12,
                          frames=sum(row['frames'] for row in report['cases']),
                          checks=sum(len(row['checks']) for row in report['cases']),
                          complete_owner_byte_rollback_readback=True, original_full_snapshot_predicates_rechecked=True,
                          original_frame_predicates_for_direct6=True, dedicated_failed_FE_frame_scope_for_FE6=True,
                          actual_stdin_transactions_and_both_EOF=True)
    except BaseException as error:
        report['failures'].append({'stage': 'admission-or-physical-readback', 'type': type(error).__name__,
                                   'message': str(error), 'traceback': traceback.format_exc()})
    finally:
        if bound is not None:
            try:
                report['all_inputs_unchanged'] = bound.unchanged()
            except BaseException as error:
                report['status'] = 'FAIL'
                report['failures'].append({'stage': 'input-recheck', 'error': repr(error)})
            report['inputs'] = bound.rows
        dump(out / 'report.json', report)
    print(json.dumps({key: report.get(key) for key in ('status', 'native_children', 'frames', 'checks', 'readback_refusal_frames', 'native_executed')}))
    return 0 if report['status'] == 'PASS_DIRECT6_AND_FE6_UNSUPPORTED_HALF_CONTROLS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
