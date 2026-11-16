#!/usr/bin/env python3
"""Observe exact CPU float32 Tensor.add_(Scalar) NaN and invalid-sum bits.

Twelve fixed input pairs each use a scalar Tensor, a short vector, a 64-element
vector, and a transposed Tensor. These distinguish scalar/tail and vector paths.
No NaN payload is normalized and no native output supplies a reference value.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import runpy
import struct
import sys
import traceback


SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
SOURCE_PIN = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
CASES = (
    ('positive-quiet-self-finite-scalar', 0x7fc00001, 0x3f000000),
    ('negative-quiet-self-finite-scalar', 0xffc12345, 0xbf000000),
    ('positive-signaling-self-finite-scalar', 0x7f812345, 0x3f000000),
    ('negative-signaling-self-finite-scalar', 0xff812345, 0x3f000000),
    ('finite-self-positive-quiet-scalar', 0x3f800000, 0x7fc23456),
    ('finite-self-negative-quiet-scalar', 0x3f800000, 0xffc34567),
    ('distinct-quiet-self-quiet-scalar', 0x7fc00001, 0x7fc23456),
    ('signaling-self-negative-quiet-scalar', 0x7f812345, 0xffc34567),
    ('quiet-self-signaling-scalar-conversion', 0x7fc00001, 0x7f812345),
    ('positive-infinity-negative-infinity', 0x7f800000, 0xff800000),
    ('negative-infinity-positive-infinity', 0xff800000, 0x7f800000),
    ('finite-exact-cancellation-control', 0xbf800000, 0x3f800000),
)
VARIANTS = ('scalar', 'short-vector', 'vector-64', 'transpose-3-by-2')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def execute(torch):
    def words(tensor):
        return [int(word) & 0xffffffff for word in
                tensor.detach().contiguous().reshape(-1).view(torch.int32).tolist()]

    def info(tensor):
        assert tensor.dtype == torch.float32 and tensor.device.type == 'cpu'
        return {'shape': list(tensor.shape), 'strides': list(tensor.stride()),
                'storage_offset_bytes': tensor.storage_offset() * tensor.element_size(),
                'storage_bytes': tensor.untyped_storage().nbytes(),
                'version': int(tensor._version), 'requires_grad': bool(tensor.requires_grad),
                'is_leaf': bool(tensor.is_leaf), 'binary32_words': words(tensor),
                'tensor_identity': str(tensor._cdata),
                'storage_identity': str(tensor.untyped_storage()._cdata)}

    def run_variant(name, input_word, scalar_word):
        count = 1 if name == 'scalar' else 64 if name == 'vector-64' else 6
        signed_word = input_word if input_word < 0x80000000 else input_word - 0x100000000
        integer_storage = torch.tensor([signed_word] * count, dtype=torch.int32)
        tensor = integer_storage.view(torch.float32)
        if name == 'scalar':
            tensor = tensor.reshape(())
        elif name == 'transpose-3-by-2':
            tensor = tensor.reshape(2, 3).transpose(0, 1)
        tensor.requires_grad_(True)

        # Python Scalar admission can quiet an sNaN when converting f32 -> f64
        # -> f32. Record both requested and admitted words rather than claiming
        # that Python preserved a signaling Scalar representation.
        scalar = struct.unpack('<f', struct.pack('<I', scalar_word))[0]
        python_bits = struct.unpack('<Q', struct.pack('<d', scalar))[0]
        admitted = words(torch.tensor(scalar, dtype=torch.float32))[0]
        before = info(tensor)
        with torch.no_grad():
            result = tensor.add_(scalar)
        after = info(tensor)
        assert before['binary32_words'] == [input_word] * count
        checks = {
            'returned_same_python_tensor': result is tensor,
            'tensor_identity_unchanged': before['tensor_identity'] == after['tensor_identity'],
            'storage_identity_unchanged': before['storage_identity'] == after['storage_identity'],
            'layout_unchanged': all(before[key] == after[key] for key in
                                    ('shape', 'strides', 'storage_offset_bytes', 'storage_bytes')),
            'version_increased_once': after['version'] == before['version'] + 1,
            'leaf_and_requires_grad_unchanged': before['is_leaf'] is True and after['is_leaf'] is True
                and before['requires_grad'] is True and after['requires_grad'] is True,
        }
        return {'variant': name, 'input_word': input_word,
                'scalar_requested_binary32_word': scalar_word,
                'scalar_python_binary64_word': python_bits,
                'scalar_admitted_binary32_word': admitted,
                'scalar_conversion_changed_word': scalar_word != admitted,
                'before': before, 'after': after, 'checks': checks}

    rows = []
    with torch.autograd.set_multithreading_enabled(False):
        for name, input_word, scalar_word in CASES:
            row = {'case': name, 'input_word': input_word,
                   'scalar_requested_binary32_word': scalar_word, 'variants': []}
            try:
                for variant in VARIANTS:
                    row['variants'].append(run_variant(variant, input_word, scalar_word))
                row['status'] = 'OBSERVED' if all(all(item['checks'].values()) for item in row['variants']) else 'PROBE_ERROR'
                # Deliberately observe, rather than assume, payload propagation
                # equivalence between vector, scalar and noncontiguous paths.
                row['variant_output_word_sets'] = {
                    item['variant']: sorted(set(item['after']['binary32_words'])) for item in row['variants']}
            except Exception as error:
                row.update(status='PROBE_ERROR', exception_type=type(error).__name__,
                           message=str(error), traceback=traceback.format_exc())
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
    base = {'schema': 'heptatorch.pinned-torch-scalar-add-nan-observations.v1',
            'probe_sha256': digest(__file__), 'reference_support_sha256': SUPPORT_SHA,
            'lock_sha256': LOCK_SHA, 'reference_source_commit': SOURCE_PIN,
            'expected_cases': 12, 'expected_variants': 48,
            'case_plan': [{'case': name, 'input_word': a, 'scalar_requested_binary32_word': b}
                          for name, a, b in CASES], 'variant_plan': list(VARIANTS),
            'native_executed': False, 'native_values_compared': 0,
            'comparison_policy': 'Every raw binary32 word including NaN sign, signaling/quiet bit and payload; absolute Version and opaque Tensor/Storage identity relations. Never normalize NaNs.',
            'scalar_admission_policy': 'Observe requested f32 word, Python f64 Scalar word and admitted f32 word. Native comparison must use admitted f32 bits for the Python Scalar contract.',
            'identity_policy': 'Equal Version counters do not establish shared Version identity.',
            'not_covered': ['other dtypes or devices', 'all compiler or ISA dispatches',
                            'Tensor.add_ autograd semantics outside no_grad',
                            'native equality is not established by this observation-only probe'],
            'full_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    assert len(CASES) == 12 and len(VARIANTS) == 4
    if args.prepare_only:
        emit(args.output, dict(base, status='PREPARED_NOT_RUN', cases=0, torch_executed=False))
        print(json.dumps({'status': 'PREPARED_NOT_RUN', 'expected_cases': 12,
                          'expected_variants': 48, 'probe_sha256': base['probe_sha256']}))
        return 0
    provenance_path = args.output.with_name(args.output.stem + '-provenance.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    old_argv, old_cwd = sys.argv, Path.cwd()
    try:
        os.chdir(args.repo)
        sys.argv = [str(args.reference), '/dev/null', str(provenance_path), str(args.lock)]
        try:
            support = runpy.run_path(str(args.reference), run_name='scalar_add_nan_probe_support')
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
        variants=sum(len(row['variants']) for row in rows), torch_executed=True,
        probe_errors=errors, rows=rows, oracle_provenance=provenance,
        source_files_unchanged={'probe': digest(__file__) == base['probe_sha256'],
            'support': digest(args.reference) == SUPPORT_SHA, 'lock': digest(args.lock) == LOCK_SHA})
    assert all(report['source_files_unchanged'].values())
    emit(args.output, report)
    print(json.dumps({'status': report['status'], 'cases': len(rows), 'variants': report['variants'],
                      'probe_errors': errors, 'probe_sha256': base['probe_sha256']}))
    return int(errors != 0)


if __name__ == '__main__':
    raise SystemExit(main())
