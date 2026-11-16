#!/usr/bin/env python3
"""Strictly compare captured native layouts with the fixed actual Torch report."""
import argparse
import hashlib
import json
from pathlib import Path


SHA = lambda raw: hashlib.sha256(raw).hexdigest()
REFERENCE = '2de8fbd82d10fc16b0640684f4b36ae958a5a0455773401f82092210967137c8'
PROBE = '568ece8996db71f5d1a16b6dfb71efd963a258f457e73cfdc738f7d60e226f89'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'native-observations', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('expected-source', 'expected-image'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    paths = [args.reference.resolve(), args.native_observations.resolve(), Path(__file__).resolve()]
    custody = {path: path.read_bytes() for path in paths}
    assert SHA(custody[paths[0]]) == REFERENCE
    reference, native = [json.loads(custody[path]) for path in paths[:2]]
    assert reference['schema'] == 'heptatorch.pinned-torch-extreme-stride-observations.v1'
    assert reference['probe_sha256'] == PROBE and reference['status'] == 'OBSERVED'
    assert reference['expected_cases'] == reference['cases'] == 8
    assert reference['probe_errors'] == 0 and reference['torch_executed'] is True
    provenance = reference['oracle_provenance']['checks']
    assert len(provenance) == 11 and all(value is True for value in provenance.values())
    assert all(reference['source_files_unchanged'].values())
    assert native['schema'] == 'heptatorch.native-extreme-stride-observations.v1'
    assert native['status'] == 'OBSERVED_NOT_TORCH_QUALIFIED'
    assert native['source_sha256'] == args.expected_source
    assert native['executable_sha256'] == args.expected_image
    assert native['reference_probe_sha256'] == PROBE and native['cases'] == 8
    assert native['native_executed'] is True and native['input_custody_after'] is True
    plan = reference['case_plan']
    assert [row['case'] for row in reference['rows']] == [row['case'] for row in plan]
    assert [row['case'] for row in native['rows']] == [row['case'] for row in plan]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name, path in [('actual-reference.json', paths[0]), ('native-observations.json', paths[1]),
                       ('executed-test.py', paths[2])]:
        (out / name).write_bytes(custody[path])
    rows = []
    for index, (spec, expected, observed) in enumerate(zip(plan, reference['rows'], native['rows'])):
        assert observed['stride'] == spec['stride']
        assert observed['reverse_inputs'] == expected['reverse_inputs'] == spec['reverse_inputs']
        assert expected['requested_shape'] == [1, 2, 3]
        assert expected['requested_strides'] == [spec['stride'], 1, 2]
        assert expected['status'] == observed['status'] == 'OBSERVED'
        assert expected['view_accepted'] and expected['operation_accepted']
        assert observed['audit'] == 1
        comparisons = []

        def compare(path, a, b):
            comparisons.append({'field': path, 'expected': a, 'actual': b, 'equal': a == b})

        for field in ('view_accepted', 'operation_accepted'):
            compare(field, expected[field], observed.get(field))
        if observed['view_accepted'] and observed.get('operation_accepted'):
            for snap in ('left_before', 'right_before', 'left_after', 'right_after', 'output'):
                for field in ('shape', 'strides', 'storage_offset_bytes', 'storage_bytes',
                              'version', 'requires_grad', 'binary32_words'):
                    compare(snap + '.' + field, expected[snap][field], observed[snap][field])
            assert set(expected['checks']) == set(observed['checks'])
            for field in sorted(expected['checks']):
                compare('checks.' + field, expected['checks'][field], observed['checks'][field])
        differences = [row for row in comparisons if not row['equal']]
        rows.append({'index': index, 'case': spec['case'], 'status': 'FAIL' if differences else 'PASS',
                     'comparisons': comparisons, 'differences': differences,
                     'frames': observed['frames']})
    for path, raw in custody.items():
        assert path.read_bytes() == raw
    failures = sum(row['status'] == 'FAIL' for row in rows)
    report = {'schema': 'heptatorch.extreme-stride.actual-reference-comparison.v1',
              'status': 'FAIL' if failures else 'PASS', 'cases': len(rows),
              'passing_cases': len(rows) - failures, 'failing_cases': failures,
              'source_sha256': args.expected_source, 'executable_sha256': args.expected_image,
              'reference_sha256': REFERENCE, 'reference_probe_sha256': PROBE,
              'native_observations_sha256': SHA(custody[paths[1]]),
              'test_sha256': SHA(custody[paths[2]]),
              'field_comparisons': sum(len(row['comparisons']) for row in rows),
              'different_fields': sum(len(row['differences']) for row in rows),
              'frames_previously_executed': native['frames'],
              'native_reexecuted_here': False, 'torch_reexecuted_here': False,
              'identity_comparison': 'Opaque identity values are compared only through exact observed relations, not process-specific numeric IDs.',
              'comparison_policy': 'Exact admission, shape, all element strides, offsets, storage bytes, Version, requires_grad, every raw binary32 word, and identity relations. No tolerance and no expected failure count.',
              'oracle_provenance_checks': provenance, 'rows': rows,
              'source_custody_after': True, 'full_rewrite_complete': False}
    (out / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'cases', 'passing_cases',
                                                  'failing_cases', 'different_fields')}))
    return int(failures != 0)


if __name__ == '__main__':
    raise SystemExit(main())
