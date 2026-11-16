#!/usr/bin/env python3
"""Strictly replay the two actual mixed-zero/nonzero incoming observations.

Exact comparisons cover all logical values, Version counters and recorded
layouts; opaque Tensor/Storage equality relations; and all Generator bytes.
Intermediate forward-layout differences are also failures, retained separately
for diagnosis and never used to alter the gradient expectation.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import signal
import struct
import traceback


WIRE_SHA = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
CLIENT_SHA = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
PROBE_SHA = '3f8c8c6b3d8bb86cdbfe574ab92ac2c43f6817e3b6cc32b1f4e48957cd8816ae'
SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
NAMES = ('multiply-created-first-bernoulli-created-last',
         'bernoulli-created-first-multiply-created-last')
EXPECTED = [struct.unpack('<I', struct.pack('<f', i))[0] for i in range(1, 7)]
FORWARD_INTERMEDIATES = {'multiply_output', 'bernoulli_output', 'output'}
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'source', 'protocol', 'observations', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('expected-native-sha256', 'expected-source-sha256', 'expected-observations-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pins = {args.binary: args.expected_native_sha256, args.source: args.expected_source_sha256,
            args.protocol: WIRE_SHA, args.protocol.with_name('owner_client.py'): CLIENT_SHA,
            args.observations: args.expected_observations_sha256}
    before = {str(path.resolve()): sha(path.read_bytes()) for path in pins}
    for path, expected in pins.items():
        assert before[str(path.resolve())] == expected, str(path)
    script = Path(__file__).read_bytes()
    (output / 'executed-test.py').write_bytes(script)
    reference_bytes = args.observations.read_bytes()
    (output / 'actual-observations.json').write_bytes(reference_bytes)
    reference = json.loads(reference_bytes)
    assert reference['status'] == 'OBSERVED' and reference['cases'] == 2 and reference['probe_errors'] == 0
    assert reference['probe_sha256'] == PROBE_SHA and reference['native_executed'] is False
    assert reference['reference_support_sha256'] == SUPPORT_SHA and reference['lock_sha256'] == LOCK_SHA
    assert reference['case_plan'] == list(NAMES)
    assert reference['mathematical_expected_gradient_words'] == EXPECTED
    provenance = reference['oracle_provenance']
    assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11
    assert all(value is True for value in provenance['checks'].values())
    assert all(reference['order_comparison'].values())
    spec = importlib.util.spec_from_file_location('actual_zero_merge_wire', args.protocol)
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    report = {'schema': 'heptatorch.independent-actual-zero-merge-layout-replay.v1', 'status': 'RUNNING',
              'native_source_sha256': args.expected_source_sha256,
              'native_executable_sha256': args.expected_native_sha256,
              'actual_reference_sha256': args.expected_observations_sha256,
              'actual_probe_sha256': PROBE_SHA, 'test_sha256': sha(script),
              'source_custody_before': before, 'native_executed': True, 'Torch_executed_here': False,
              'production_host_numerics': False,
              'strict_scope': ['all logical binary32 words', 'all absolute Version counters',
                  'all recorded layouts and storage extents, including leaf, seed and actual gradient',
                  'all recorded Tensor/Storage identity equality relations within each snapshot',
                  'identity stability for each Tensor across the successful backward',
                  'every byte of the three Generator snapshots'],
              'separately_reported': ['forward intermediate strides/storage extent mismatches remain strict failures',
                  'native public metadata does not expose PyTorch grad_fn_type or is_leaf'],
              'cases': [], 'frames': 0, 'exact_assertions': 0, 'raw_words_compared': 0,
              'full_generator_bytes_compared': 0, 'forward_layout_differences': [],
              'full_PyTorch_rewrite_complete': False, 'pytorch_api_denominator': 2590}

    for ref_row in reference['rows']:
        assert ref_row['status'] == 'OBSERVED' and ref_row['case'] in NAMES
        ref = ref_row['observation']
        assert ref['mathematical_expected_gradient_words'] == EXPECTED and all(ref['checks'].values())
        row = {'case': ref_row['case'], 'status': 'RUNNING', 'comparisons': [],
               'exact_assertions': 0, 'raw_words_compared': 0,
               'full_generator_bytes_compared': 0, 'snapshots': {}}
        report['cases'].append(row)
        session = wire.Session(args.binary, directory=output / ref_row['case'] / 'transport')
        one = lambda *command: session.run([command])[0]

        def equal(actual, expected, label):
            row['comparisons'].append({'label': label, 'status': 'PASS' if actual == expected else 'FAIL',
                                       'actual': actual, 'expected': expected})
            row['exact_assertions'] += 1

        def rng(generator, expected, phase):
            raw = one(63, generator)['payload']
            expected_bytes = bytes.fromhex(expected['complete_state_hex'])
            assert len(expected_bytes) == expected['bytes'] and sha(expected_bytes) == expected['sha256']
            equal(raw.hex(), expected['complete_state_hex'], phase + ' every Generator byte')
            row['full_generator_bytes_compared'] += len(raw)

        def describe(target):
            value = wire.meta(one(13, target))
            rank = value['rank']
            raw = one(39, target, 0, value['numel'])['payload']
            return {'shape': value['shape'][:rank], 'strides': value['strides'][:rank],
                    'storage_offset_bytes': value['storage_offset'], 'storage_bytes': value['storage_bytes'],
                    'version': value['version'], 'requires_grad': bool(value['requires_grad']),
                    'tensor_identity': target, 'storage_identity': value['storage_id'],
                    'native_version_identity': value['version_id'],
                    'binary32_words': list(struct.unpack('<' + 'I' * value['numel'], raw))}

        def snapshot(tensors, expected, phase):
            gradient = one(91, tensors['parameter'])['value']
            equal(one(59, tensors['parameter'])['value'], int(expected['gradient'] is not None),
                  phase + ' explicit gradient presence')
            current = {name: describe(target) for name, target in tensors.items()}
            current['gradient'] = None if gradient == 0 else describe(gradient)
            row['snapshots'][phase] = current
            for name, actual in current.items():
                wanted = expected[name]
                if actual is None or wanted is None:
                    equal(actual is None, wanted is None, phase + ' ' + name + ' None/defined')
                    continue
                for field in ('shape', 'requires_grad', 'version', 'binary32_words'):
                    equal(actual[field], wanted[field], phase + ' ' + name + ' ' + field)
                row['raw_words_compared'] += len(actual['binary32_words'])
                for field in ('strides', 'storage_offset_bytes', 'storage_bytes'):
                    equal(actual[field], wanted[field], phase + ' ' + name + ' ' + field)
                    if name in FORWARD_INTERMEDIATES and actual[field] != wanted[field]:
                        report['forward_layout_differences'].append({'case': ref_row['case'], 'phase': phase,
                            'tensor': name, 'field': field, 'native': actual[field], 'Torch': wanted[field],
                            'status': 'FAIL_FORWARD_METADATA'})
            live = [name for name, value in current.items() if value is not None and expected[name] is not None]
            for left, right in itertools.combinations(live, 2):
                for field in ('tensor_identity', 'storage_identity'):
                    equal(current[left][field] == current[right][field],
                          expected[left][field] == expected[right][field],
                          phase + ' ' + field + ' equality relation ' + left + '/' + right)
            return current

        try:
            parameter_base = one(*wire.full((2, 3), .5))['value']
            parameter = one(6, parameter_base, 0, 1)['value']
            one(90, parameter, 1)
            coefficient = one(*wire.tensor((3, 2), [1, 2, 3, 4, 5, 6]))['value']
            generator = one(60, 17)['value']
            rng(generator, ref['generator_before_forward'], 'before forward')
            if ref_row['case'] == NAMES[0]:
                multiplied = one(2, 3, parameter, coefficient)['value']
                bernoulli = one(82, parameter, generator, 0)['value']
            else:
                bernoulli = one(82, parameter, generator, 0)['value']
                multiplied = one(2, 3, parameter, coefficient)['value']
            joined = one(2, 1, multiplied, bernoulli)['value']
            rng(generator, ref['generator_after_forward'], 'after forward')
            seed_base = one(*wire.full((2, 3), 1))['value']
            seed = one(6, seed_base, 0, 1)['value']
            tensors = {'parameter_base': parameter_base, 'parameter': parameter, 'coefficient': coefficient,
                       'multiply_output': multiplied, 'bernoulli_output': bernoulli, 'output': joined,
                       'seed_base': seed_base, 'seed': seed}
            first = snapshot(tensors, ref['before_backward'], 'before_backward')
            one(57, joined, seed, 0)
            last = snapshot(tensors, ref['after_backward'], 'after_backward')
            rng(generator, ref['generator_after_backward'], 'after backward')
            for name in tensors:
                for field in ('tensor_identity', 'storage_identity'):
                    equal(first[name][field] == last[name][field],
                          ref['before_backward'][name][field] == ref['after_backward'][name][field],
                          name + ' ' + field + ' stability across backward')
            if last['gradient'] is not None:
                equal(last['gradient']['binary32_words'], EXPECTED, 'independent exact mathematical logical derivative')
            equal(one(53)['value'], 1, 'native whole-owner audit')
            session.finish()
            row['status'] = 'PASS' if all(item['status'] == 'PASS' for item in row['comparisons']) else 'FAIL'
        except BaseException as error:
            row.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
            if session.poll() is None:
                os.kill(session.pid, signal.SIGKILL)
            try:
                session.finish(expected=-signal.SIGKILL)
            except BaseException as cleanup:
                row['cleanup_error'] = repr(cleanup)
        row['frames'] = session.frames
        report['frames'] += row['frames']
        for field in ('exact_assertions', 'raw_words_compared', 'full_generator_bytes_compared'):
            report[field] += row[field]
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    after = {str(path.resolve()): sha(path.read_bytes()) for path in pins}
    assert before == after and Path(__file__).read_bytes() == script
    report.update(status='PASS' if all(row['status'] == 'PASS' for row in report['cases']) else 'FAIL',
                  case_count=len(report['cases']), failed_cases=sum(row['status'] != 'PASS' for row in report['cases']),
                  forward_layout_difference_count=len(report['forward_layout_differences']),
                  source_custody_after=after)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'case_count', 'failed_cases', 'frames',
                    'exact_assertions', 'raw_words_compared', 'full_generator_bytes_compared',
                    'forward_layout_difference_count')}))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
