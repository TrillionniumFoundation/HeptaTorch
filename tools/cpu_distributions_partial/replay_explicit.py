#!/usr/bin/env python3
"""Compare native distributions with the frozen actual PyTorch observation.

The observation stays immutable. Commands86/87 expose explicit prefix commit;
unrepresented dtypes/native_dropout remain open, never counted PASS.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import traceback

SHA = lambda raw: hashlib.sha256(raw).hexdigest()
REFERENCE = 'c624f5c4762b1c71e7053aba9af55aaa3d8bbaae5aa100b0a6df5f44f58da467'
PROTOCOL = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'protocol', 'executable', 'build-report', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_inputs = {p.resolve(): p.read_bytes() for p in (a.reference, a.protocol, a.executable, a.build_report, Path(__file__))}
    assert SHA(source_inputs[a.reference.resolve()]) == REFERENCE
    assert SHA(source_inputs[a.protocol.resolve()]) == PROTOCOL
    build = json.loads(source_inputs[a.build_report.resolve()])
    assert build['status'] == 'PASS' and SHA(source_inputs[a.executable.resolve()]) == build['executable_sha256']
    reference = json.loads(source_inputs[a.reference.resolve()])
    assert reference['status'] == 'OBSERVED' and reference['cases'] == 183 and reference['probe_error_count'] == 0
    assert len(reference['oracle_provenance']['checks']) == 11 and all(reference['oracle_provenance']['checks'].values())
    states = {}
    for digest, record in reference['states'].items():
        data = bytes.fromhex(record['hex'])
        assert len(data) == 5056 and SHA(data) == digest
        states[digest] = data
    spec = importlib.util.spec_from_file_location('distribution_test_protocol', a.protocol.resolve())
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    rows, exact_words, state_checks, frames, rollback_bytes = [], 0, 0, 0, 0
    report = {'schema': 'heptatorch.cpu-distributions.explicit-partial-actual-replay.v1', 'status': 'RUNNING',
              'reference_sha256': REFERENCE, 'reference_cases': 183,
              'runtime_source_sha256': build['runtime_source_sha256'],
              'executable_sha256': build['executable_sha256'], 'rows': rows,
              'actual_reference_provenance_checks': reference['oracle_provenance']['checks'],
              'all_rng_complete': False, 'full_pytorch_rewrite_complete': False,
              'scope': 'float32 tensors; double scalar probabilities; explicit owned Generator; normal-gradient-mode inplace on requires-grad tensors remains restricted',
              'lifecycle_mutation_mapping': 'post-forward input/mask mutation uses one checked native element write to advance the shared Version; this does not claim a complete add_(1) implementation',
              'error_contract': 'explicit commands86/87 commit operation-error prefixes and stop suffixes; original atomic80-85 are separately replayed unchanged',
              'response_mapping': '86/87 result is operation error and 32B payload is error,Tensor,writes,raw32; successful Tensor result is decoded from that payload',
              'original_reference_unchanged': True}

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def contiguous(shape):
        result, stride = [0] * len(shape), 1
        for i in range(len(shape) - 1, -1, -1):
            result[i] = stride
            stride *= max(shape[i], 1)
        return result

    def command_tensor(snapshot):
        shape, words = snapshot['shape'], snapshot['words']
        return (1, len(shape), int(snapshot.get('requires_grad', False)), *shape,
                *([0] * (4 - len(shape))), wire.Payload(struct.pack('<' + 'I' * len(words), *words), 4))

    def view(session, base, snapshot):
        shape, strides = snapshot['shape'], snapshot['strides']
        data = struct.pack('<10Q', len(shape), *shape, *([0] * (4 - len(shape))),
                           *strides, *([0] * (4 - len(strides))), snapshot['storage_offset'])
        return session.run([(69, base, wire.Payload(data), 80)])[0]['value']

    def create(session, snapshot, physical=None):
        assert snapshot['dtype'] == 'torch.float32'
        if physical is None or (snapshot['storage_offset'] == 0 and snapshot['strides'] == contiguous(snapshot['shape'])
                                and len(physical) == len(snapshot['words'])):
            result = session.run([command_tensor(snapshot)])[0]['value']
            return result, result
        storage = dict(snapshot, shape=[len(physical)], strides=[1], storage_offset=0, words=physical)
        base = session.run([command_tensor(storage)])[0]['value']
        return view(session, base, snapshot), base

    def read_words(session, tensor, count, gradient=False):
        payload = session.run([(40 if gradient else 39, tensor, 0, count)])[0]['payload']
        return list(struct.unpack('<' + 'I' * count, payload))

    def equal(actual, expected, field):
        nonlocal exact_words
        if field.endswith('words'):
            exact_words += len(expected)
        def brief(value):
            if isinstance(value, bytes):
                return {'bytes': len(value), 'sha256': SHA(value)}
            if isinstance(value, dict):
                return {key: brief(part) for key, part in value.items()}
            return value[:25] if isinstance(value, list) else value
        assert actual == expected, {'field': field, 'actual': brief(actual), 'expected': brief(expected)}

    def snapshot_check(session, tensor, expected, label):
        metadata = wire.meta(session.run([(13, tensor)])[0])
        assert metadata['storage_offset'] % 4 == 0
        for field, actual in [('shape', metadata['shape'][:metadata['rank']]),
                              ('strides', metadata['strides'][:metadata['rank']]),
                              ('storage_offset', metadata['storage_offset'] // 4),
                              ('version', metadata['version']), ('requires_grad', bool(metadata['requires_grad']))]:
            equal(actual, expected[field], label + '.' + field)
        equal(read_words(session, tensor, len(expected['words'])), expected['words'], label + '.words')
        return metadata

    save()
    for index, expected in enumerate(reference['rows']):
        row = {'index': index, 'name': expected['name'], 'family': expected['family']}
        rows.append(row)
        family = expected['family']
        probability = expected.get('probabilities', expected.get('probabilities_before'))
        if family == 'native-dropout-contrast' or (probability and probability['dtype'] != 'torch.float32'):
            row.update(status='OPEN_UNREPRESENTED', reason='native_dropout bool-mask interface is distinct' if family == 'native-dropout-contrast'
                       else 'probability Tensor dtype is outside this float32 Tensor profile')
            save()
            continue
        session = None
        try:
            session = wire.Session(a.executable.resolve(), directory=out / ('case-' + str(index)))
            x, backing = create(session, expected['input_before'], expected['physical_before'])
            g = session.run([(60, 0)])[0]['value']
            session.run([(64, g, wire.Payload(states[expected['state_before']]), 5056)])
            no_grad = int(not expected['input_before']['requires_grad'])
            if family.startswith('dropout'):
                pbits = expected.get('probability_double_bits', struct.unpack('<Q', struct.pack('<d', .3))[0])
                inplace = expected.get('inplace', family == 'dropout-inplace-grad-leaf')
                command = (87 if inplace else 83, x, g, pbits, int(expected.get('training', True)), no_grad)
            elif family == 'bernoulli-functional-autograd':
                command = (82, x, g, 0)
            elif probability is not None:
                if family == 'tensor-alias':
                    p = x if expected['name'] in ('tensor-alias-self', 'tensor-alias-repeat-self') else view(session, backing, probability)
                else:
                    p, _ = create(session, probability)
                command = (86, x, g, p, no_grad)
            else:
                command = (80, x, g, expected['probability_double_bits'], no_grad)
            owner_before = session.owner()
            status, records = wire.parse(session.exchange(wire.packet([command])))
            row['native_frame_status'] = status
            explicit_error = False
            if command[0] in (86,87) and status == 0:
                assert len(records) == 1 and len(records[0]['payload']) == 32
                operation_error, returned_tensor, writes, consumed = struct.unpack('<4Q',records[0]['payload'])
                assert operation_error == records[0]['value'] and returned_tensor == x
                assert consumed in expected['matching_raw32_uniform_counts'], (consumed,expected['matching_raw32_uniform_counts'])
                row['explicit_progress'] = {'error':operation_error,'completed_writes':writes,'consumed_raw32':consumed}
                if operation_error:
                    explicit_error = True
                    status = operation_error
                    records = []
                else:
                    records[0] = dict(records[0],value=returned_tensor,payload=b'')
            row['operation_status'] = status
            if not expected['accepted']:
                assert status != 0 and not records
                message = expected['error']['message']
                expected_error = 5 if 'leaf Variable' in message else 2 if 'single memory location' in message else 7
                assert status == expected_error, (status,expected_error,message)
                if not explicit_error:
                    equal(session.owner(), owner_before, 'whole_owner_rollback')
                    rollback_bytes += owner_before['length']
                native_state = session.run([(63,g)])[0]['payload']
                state_checks += 1
                if explicit_error:
                    equal(native_state,states[expected['state_after']],'committed_error_complete_MT_state')
                    equal(read_words(session,backing,len(expected['physical_after'])),expected['physical_after'],'committed_error.physical.words')
                    snapshot_check(session,x,expected['input_after'],'committed_error.input')
                    session.run([(53,)])
                    row['status'] = 'PASS'
                else:
                    equal(native_state, states[expected['state_before']], 'native_error_keeps_complete_state')
                    equal(read_words(session, backing, len(expected['physical_before'])), expected['physical_before'], 'error.physical.words')
                    assert expected['state_unchanged'], 'unexpected early admission for observed late error'
                    snapshot_check(session, x, expected['input_after'], 'error.input')
                    row['status'] = 'PASS'
            else:
                assert status == 0 and len(records) == 1, (status, records)
                result = records[0]['value']
                before_meta = snapshot_check(session, x, expected['input_after'], 'input')
                result_meta = snapshot_check(session, result, expected['result'], 'result')
                equal(result == x, expected['result_is_input'], 'result_identity')
                equal(result_meta['storage_id'] == before_meta['storage_id'], expected['result_aliases_input_storage'], 'result_storage_alias')
                equal(read_words(session, backing, len(expected['physical_after'])), expected['physical_after'], 'physical.words')
                native_state = session.run([(63, g)])[0]['payload']
                state_checks += 1
                equal(native_state, states[expected['state_after']], 'complete_MT_state')
                saved = expected.get('saved_other_before', expected.get('saved_scale_words', expected.get('saved_before_mutation')))
                if saved is not None:
                    mask = session.run([(85, result)])[0]['value']
                    snapshot_check(session, mask, saved, 'saved_mask')
                if family in ('dropout-probability-bound', 'dropout-autograd', 'bernoulli-functional-autograd'):
                    shape = expected['result']['shape']
                    if family == 'dropout-autograd':
                        seed_words = [2147483648, 1065353216, 3212836864, 1073741824, 3225419776, 0]
                    elif family == 'bernoulli-functional-autograd':
                        seed_words = [2147483648, 2143289344, 2139095040, 3212836864]
                    else:
                        seed_words = [1065353216] * len(expected['result']['words'])
                    seed = session.run([command_tensor({'shape': shape, 'words': seed_words, 'requires_grad': False})])[0]['value']
                    session.run([(57, result, seed, 1 if family == 'dropout-autograd' else 0)])
                    gradient = expected.get('first_gradient', expected.get('gradient'))
                    equal(read_words(session, x, len(gradient['words']), True), gradient['words'], 'gradient.words')
                    if family == 'dropout-autograd':
                        session.run([(70, x)])
                        # One checked no-grad write is sufficient for the observed
                        # input-Version mutation; the derivative saves only noise.
                        session.run([(12, x, 0, 1065353216)])
                        session.run([(57, result, seed, 0)])
                        equal(read_words(session, x, len(gradient['words']), True),
                              expected['gradient_after_input_mutation']['words'], 'gradient_after_input_mutation.words')
                        if expected['backward_after_release']['accepted']:
                            session.run([(57, result, seed, 0)])
                        else:
                            rollback_bytes += session.error([(57, result, seed, 0)], 10)
                    equal(session.run([(63, g)])[0]['payload'], states[expected.get('state_after_backward', expected['state_after'])], 'backward_MT_state')
                    state_checks += 1
                if family == 'dropout-saved-mask-mutation':
                    assert saved is not None
                    session.run([(12, mask, 0, 1065353216)])
                    seed = session.run([command_tensor({'shape': expected['result']['shape'],
                                                       'words': [1065353216] * len(expected['result']['words'])})])[0]['value']
                    rollback_bytes += session.error([(57, result, seed, 0)], 6)
                    equal(session.run([(59, x)])[0]['value'], 0, 'gradient_remains_None')
                    equal(session.run([(63, g)])[0]['payload'], states[expected['state_after_backward']], 'failed_backward_MT_state')
                    state_checks += 1
                session.run([(53,)])
                row['status'] = 'PASS'
            frames += session.frames
            row['frames'] = session.frames
            session.finish()
            session = None
        except BaseException as error:
            row.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
            if session is not None:
                frames += session.frames
                try:
                    session.finish()
                except BaseException as cleanup:
                    row['cleanup_error'] = repr(cleanup)
        save()
    for path, data in source_inputs.items():
        assert path.read_bytes() == data, path
    counts = {status: sum(row['status'] == status for row in rows) for status in sorted({row['status'] for row in rows})}
    report.update(status='FAIL' if counts.get('FAIL', 0) else 'PARTIAL', counts=counts,
                  frames_before_EOF=frames, exact_float32_words=exact_words, complete_state_comparisons=state_checks,
                  rollback_bytes_compared=rollback_bytes, input_custody_after=True,
                  inputs={str(p): SHA(data) for p, data in source_inputs.items()})
    save()
    print(json.dumps({k: report[k] for k in ('status', 'counts', 'frames_before_EOF', 'exact_float32_words',
                                           'complete_state_comparisons', 'rollback_bytes_compared')}, indent=2))
    raise SystemExit(1 if counts.get('FAIL', 0) else 0)


if __name__ == '__main__':
    main()
