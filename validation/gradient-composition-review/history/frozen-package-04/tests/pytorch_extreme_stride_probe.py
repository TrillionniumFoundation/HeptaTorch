#!/usr/bin/env python3
"""Observe functional binary output layout with extreme size-one axis strides.

Every backing storage contains only six float32 elements. The large stride
belongs exclusively to an axis of size one. This probe observes admission and
real output metadata; it installs no predicted wrapped-stride expectation and
does not execute the native target.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import runpy
import sys
import traceback


SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
SOURCE_PIN = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
CASES = (
    ('largest-positive-byte-stride', (1 << 61) - 1, False),
    ('byte-sign-boundary', 1 << 61, False),
    ('byte-sign-boundary-reversed', 1 << 61, True),
    ('above-byte-sign-boundary', (1 << 61) + 1, False),
    ('byte-wrap-zero', 1 << 62, False),
    ('byte-wrap-zero-reversed', 1 << 62, True),
    ('below-byte-wrap-zero', (1 << 62) - 1, False),
    ('largest-signed-element-stride', (1 << 63) - 1, False),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def execute(torch):
    def info(tensor):
        assert tensor.dtype == torch.float32 and tensor.device.type == 'cpu'
        words = [int(word) & 0xffffffff for word in
                 tensor.detach().contiguous().reshape(-1).view(torch.int32).tolist()]
        return {'shape': list(tensor.shape), 'strides': list(tensor.stride()),
                'storage_offset_bytes': tensor.storage_offset() * tensor.element_size(),
                'storage_bytes': tensor.untyped_storage().nbytes(),
                'version': int(tensor._version), 'requires_grad': bool(tensor.requires_grad),
                'binary32_words': words, 'tensor_identity': str(tensor._cdata),
                'storage_identity': str(tensor.untyped_storage()._cdata)}

    rows = []
    with torch.no_grad():
        for name, stride, reverse in CASES:
            row = {'case': name, 'requested_shape': [1, 2, 3],
                   'requested_strides': [stride, 1, 2], 'reverse_inputs': reverse,
                   'operation': 'add', 'backing_elements': 6}
            try:
                backing = torch.tensor([1., 2., 3., 4., 5., 6.], dtype=torch.float32)
                other = torch.tensor([10., 11., 12., 13., 14., 15.], dtype=torch.float32).reshape(1, 2, 3)
                try:
                    view = backing.as_strided((1, 2, 3), (stride, 1, 2), 0)
                except (RuntimeError, OverflowError) as error:
                    row.update(status='OBSERVED', view_accepted=False,
                               exception_stage='as_strided', exception_type=type(error).__name__,
                               exception_message=str(error))
                    rows.append(row)
                    continue
                left, right = (other, view) if reverse else (view, other)
                before_left, before_right = info(left), info(right)
                assert info(view)['strides'] == [stride, 1, 2]
                row.update(view_accepted=True, left_before=before_left, right_before=before_right)
                try:
                    output = torch.add(left, right)
                except (RuntimeError, OverflowError) as error:
                    row.update(status='OBSERVED', operation_accepted=False,
                               exception_stage='add', exception_type=type(error).__name__,
                               exception_message=str(error),
                               left_after=info(left), right_after=info(right))
                    rows.append(row)
                    continue
                left_after, right_after, output_info = info(left), info(right), info(output)
                checks = {'left_unchanged': before_left == left_after,
                          'right_unchanged': before_right == right_after,
                          'fresh_output_tensor': output is not left and output is not right,
                          'fresh_output_storage': output_info['storage_identity'] not in
                              (before_left['storage_identity'], before_right['storage_identity']),
                          'same_output_shape': output_info['shape'] == [1, 2, 3],
                          'new_output_version_zero': output_info['version'] == 0}
                row.update(status='OBSERVED' if all(checks.values()) else 'PROBE_ERROR',
                           operation_accepted=True, left_after=left_after, right_after=right_after,
                           output=output_info, checks=checks)
            except Exception as error:
                row.update(status='PROBE_ERROR', exception_type=type(error).__name__,
                           exception_message=str(error), traceback=traceback.format_exc())
            rows.append(row)
            gc.collect()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'lock', 'repo', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    for name in ('reference', 'lock', 'repo', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    assert digest(args.reference) == SUPPORT_SHA and digest(args.lock) == LOCK_SHA
    assert len(CASES) == 8 and len({case[0] for case in CASES}) == 8
    base = {'schema': 'heptatorch.pinned-torch-extreme-stride-observations.v1',
            'probe_sha256': digest(__file__), 'reference_support_sha256': SUPPORT_SHA,
            'lock_sha256': LOCK_SHA, 'reference_source_commit': SOURCE_PIN,
            'expected_cases': 8, 'case_plan': [
                {'case': name, 'stride': stride, 'reverse_inputs': reverse}
                for name, stride, reverse in CASES],
            'native_executed': False, 'native_values_compared': 0,
            'comparison_policy': 'Observe exact accepted shape/element strides, raw binary32 words, output Version and fresh-storage relations; never pre-normalize or predict byte-stride overflow.',
            'not_covered': ['other dtypes or devices', 'all compiler or ISA dispatches',
                            'negative element strides', 'arbitrary high-rank layouts',
                            'native equality is not established by this observation-only probe'],
            'full_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    if args.prepare_only:
        emit(args.output, dict(base, status='PREPARED_NOT_RUN', cases=0, torch_executed=False))
        print(json.dumps({'status': 'PREPARED_NOT_RUN', 'expected_cases': 8,
                          'probe_sha256': base['probe_sha256']}))
        return 0
    provenance_path = args.output.with_name(args.output.stem + '-provenance.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    old_argv, old_cwd = sys.argv, Path.cwd()
    try:
        os.chdir(args.repo)
        sys.argv = [str(args.reference), '/dev/null', str(provenance_path), str(args.lock)]
        try:
            support = runpy.run_path(str(args.reference), run_name='extreme_stride_probe_support')
        except (SystemExit, ModuleNotFoundError) as error:
            observed = json.loads(provenance_path.read_bytes()) if provenance_path.exists() else None
            emit(args.output, dict(base, status='NOT_RUN', cases=0, torch_executed=False,
                 reason='Pinned reference did not admit this environment',
                 initialization_exception=type(error).__name__, initialization_message=str(error),
                 support_observation=observed))
            return 2
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
    provenance = support['PROVENANCE']
    assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11
    assert all(value is True for value in provenance['checks'].values())
    rows = execute(support['torch'])
    errors = sum(row['status'] != 'OBSERVED' for row in rows)
    report = dict(base, status='PROBE_ERROR' if errors else 'OBSERVED', cases=len(rows),
                  torch_executed=True, probe_errors=errors, rows=rows,
                  oracle_provenance=provenance,
                  source_files_unchanged={'probe': digest(__file__) == base['probe_sha256'],
                      'support': digest(args.reference) == SUPPORT_SHA,
                      'lock': digest(args.lock) == LOCK_SHA})
    assert len(rows) == 8 and all(report['source_files_unchanged'].values())
    emit(args.output, report)
    print(json.dumps({'status': report['status'], 'cases': len(rows),
                      'probe_errors': errors, 'probe_sha256': base['probe_sha256']}))
    return int(errors != 0)


if __name__ == '__main__':
    raise SystemExit(main())
