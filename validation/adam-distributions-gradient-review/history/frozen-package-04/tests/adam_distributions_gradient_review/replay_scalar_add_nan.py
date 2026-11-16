#!/usr/bin/env python3
"""Replay the fixed actual Torch Scalar-add observations with exact raw words.

Only the pinned wire/own-child transport is imported. Payloads enter native HIA
as little-endian u32 words, so the test does not quiet signaling NaNs in Python.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import struct
import traceback


WIRE_SHA = '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
CLIENT_SHA = 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
PROBE_SHA = 'bd4660bf02d32e175d8c9802c09f2dcc93b4f589d06749ae43645544fb76936a'
SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
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
    observed_bytes = args.observations.read_bytes()
    (output / 'actual-observations.json').write_bytes(observed_bytes)
    observed = json.loads(observed_bytes)
    assert observed['status'] == 'OBSERVED' and observed['cases'] == 12
    assert observed['variants'] == 48 and observed['probe_errors'] == 0
    assert observed['probe_sha256'] == PROBE_SHA
    assert observed['reference_support_sha256'] == SUPPORT_SHA and observed['lock_sha256'] == LOCK_SHA
    assert observed['native_executed'] is False
    provenance = observed['oracle_provenance']
    assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11
    assert all(value is True for value in provenance['checks'].values())
    spec = importlib.util.spec_from_file_location('scalar_add_nan_wire', args.protocol)
    wire = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wire)
    report = {'schema': 'heptatorch.independent-actual-scalar-add-nan-replay.v1', 'status': 'RUNNING',
              'native_source_sha256': args.expected_source_sha256,
              'native_executable_sha256': args.expected_native_sha256,
              'actual_reference_sha256': args.expected_observations_sha256,
              'actual_probe_sha256': PROBE_SHA, 'test_sha256': sha(script),
              'source_custody_before': before, 'native_executed': True, 'Torch_executed_here': False,
              'production_host_numerics': False,
              'comparison_policy': 'Exact raw binary32 including NaN payload/sign/quiet bits. No normalization or numerical tolerance.',
              'scalar_admission_policy': observed['scalar_admission_policy'],
              'cases': [], 'frames': 0, 'exact_assertions': 0, 'raw_words_compared': 0,
              'full_PyTorch_rewrite_complete': False, 'pytorch_api_denominator': 2590}

    for ref_case in observed['rows']:
        assert ref_case['status'] == 'OBSERVED'
        case = {'case': ref_case['case'], 'status': 'RUNNING', 'variants': []}
        report['cases'].append(case)
        for ref in ref_case['variants']:
            name = ref['variant']
            row = {'variant': name, 'status': 'RUNNING', 'frames': 0,
                   'exact_assertions': 0, 'raw_words_compared': 0, 'comparisons': []}
            case['variants'].append(row)
            session = wire.Session(args.binary, directory=output / ref_case['case'] / name / 'transport')
            one = lambda *command: session.run([command])[0]

            def equal(actual, expected, label):
                success = actual == expected
                row['comparisons'].append({'label': label, 'status': 'PASS' if success else 'FAIL',
                                           'actual': actual, 'expected': expected})
                row['exact_assertions'] += 1
                return success

            def metadata(target, expected, phase):
                current = wire.meta(one(13, target))
                rank = current['rank']
                fields = {'shape': current['shape'][:rank], 'strides': current['strides'][:rank],
                          'storage_offset_bytes': current['storage_offset'],
                          'storage_bytes': current['storage_bytes'], 'version': current['version'],
                          'requires_grad': bool(current['requires_grad'])}
                for field, value in fields.items():
                    equal(value, expected[field], phase + ' ' + field)
                value_record = one(39, target, 0, current['numel'])
                actual_words = list(struct.unpack('<' + 'I' * current['numel'], value_record['payload']))
                equal(actual_words, expected['binary32_words'], phase + ' exact binary32 words')
                row['raw_words_compared'] += current['numel']
                return current

            try:
                shape = (2, 3) if name == 'transpose-3-by-2' else tuple(ref['before']['shape'])
                count = len(ref['before']['binary32_words'])
                raw = struct.pack('<I', ref['input_word']) * count
                requires = 0 if name == 'transpose-3-by-2' else 1
                target = one(1, len(shape), requires, *shape, *([0] * (4 - len(shape))),
                             wire.Payload(raw, 4))['value']
                if name == 'transpose-3-by-2':
                    target = one(6, target, 0, 1)['value']
                    one(90, target, 1)
                first = metadata(target, ref['before'], 'before')
                result = one(95, target, ref['scalar_admitted_binary32_word'])
                equal(result['value'], target, 'inplace return preserves the exact public Tensor token')
                last = metadata(target, ref['after'], 'after')
                equal([last['storage_id'], last['version_id']],
                      [first['storage_id'], first['version_id']], 'actual Storage and Version identities remain unchanged')
                equal(one(59, target)['value'], 0, 'no gradient object created by no-grad mutation')
                equal(one(53)['value'], 1, 'native owner audit')
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
            report['exact_assertions'] += row['exact_assertions']
            report['raw_words_compared'] += row['raw_words_compared']
        case['status'] = 'PASS' if all(row['status'] == 'PASS' for row in case['variants']) else 'FAIL'
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    after = {str(path.resolve()): sha(path.read_bytes()) for path in pins}
    assert before == after and Path(__file__).read_bytes() == script
    report.update(status='PASS' if all(row['status'] == 'PASS' for row in report['cases']) else 'FAIL',
                  case_count=len(report['cases']), variant_count=sum(len(row['variants']) for row in report['cases']),
                  source_custody_after=after,
                  failed_cases=sum(row['status'] != 'PASS' for row in report['cases']),
                  failed_variants=sum(item['status'] != 'PASS' for row in report['cases'] for item in row['variants']))
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'case_count', 'variant_count', 'frames',
                    'exact_assertions', 'raw_words_compared', 'failed_cases', 'failed_variants')}))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
