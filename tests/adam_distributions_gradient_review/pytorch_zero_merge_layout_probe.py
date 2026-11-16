#!/usr/bin/env python3
"""Observe two creation orders of a mixed Bernoulli/nonzero gradient merge.

The mathematical derivative of (p * coefficient) + Bernoulli(p), with an
all-one seed, is exactly coefficient in Tensor logical order.  This independent
reference-only probe does not use a native result to construct expectations.
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
CASES = ('multiply-created-first-bernoulli-created-last',
         'bernoulli-created-first-multiply-created-last')
COEFFICIENT = (1, 2, 3, 4, 5, 6)
EXPECTED_WORDS = [struct.unpack('<I', struct.pack('<f', value))[0]
                  for value in COEFFICIENT]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def execute(torch):
    def info(tensor):
        assert tensor.dtype == torch.float32 and tensor.device.type == 'cpu'
        words = tensor.detach().contiguous().reshape(-1).view(torch.int32).tolist()
        return {'shape': list(tensor.shape), 'strides': list(tensor.stride()),
                'storage_offset_bytes': tensor.storage_offset() * tensor.element_size(),
                'storage_bytes': tensor.untyped_storage().nbytes(),
                'requires_grad': bool(tensor.requires_grad), 'is_leaf': bool(tensor.is_leaf),
                'grad_fn_type': None if tensor.grad_fn is None else type(tensor.grad_fn).__name__,
                'version': int(tensor._version),
                'binary32_words': [int(word) & 0xffffffff for word in words],
                'tensor_identity': str(tensor._cdata),
                'storage_identity': str(tensor.untyped_storage()._cdata)}

    def rng_info(generator):
        state = bytes(generator.get_state().tolist())
        return {'bytes': len(state), 'sha256': hashlib.sha256(state).hexdigest(),
                'complete_state_hex': state.hex()}

    def run_case(name):
        parameter_base = torch.full((2, 3), .5, dtype=torch.float32)
        parameter = parameter_base.transpose(0, 1)
        parameter.requires_grad_(True)
        coefficient = torch.tensor(COEFFICIENT, dtype=torch.float32).reshape(3, 2)
        generator = torch.Generator(device='cpu').manual_seed(17)
        state_before_forward = rng_info(generator)
        leaf_before_forward = info(parameter)

        if name == CASES[0]:
            multiplied = parameter * coefficient
            bernoulli = torch.bernoulli(parameter, generator=generator)
            branch_creation_order = ['multiply', 'bernoulli']
        else:
            bernoulli = torch.bernoulli(parameter, generator=generator)
            multiplied = parameter * coefficient
            branch_creation_order = ['bernoulli', 'multiply']
        output = multiplied + bernoulli
        state_after_forward = rng_info(generator)
        seed_base = torch.full((2, 3), 1., dtype=torch.float32)
        seed = seed_base.transpose(0, 1)

        def snapshot():
            return {'parameter_base': info(parameter_base), 'parameter': info(parameter),
                    'coefficient': info(coefficient), 'multiply_output': info(multiplied),
                    'bernoulli_output': info(bernoulli), 'output': info(output),
                    'seed_base': info(seed_base), 'seed': info(seed),
                    'gradient': None if parameter.grad is None else info(parameter.grad)}

        before = snapshot()
        output.backward(seed)
        after = snapshot()
        state_after_backward = rng_info(generator)
        gradient = after['gradient']
        checks = {
            'parameter_is_leaf': before['parameter']['is_leaf'] is True,
            'parameter_layout_is_3_by_2_stride_1_3':
                before['parameter']['shape'] == [3, 2] and before['parameter']['strides'] == [1, 3],
            'seed_layout_is_3_by_2_stride_1_3':
                before['seed']['shape'] == [3, 2] and before['seed']['strides'] == [1, 3],
            'gradient_was_none': before['gradient'] is None,
            'gradient_is_defined': gradient is not None,
            'logical_gradient_words_equal_mathematical_coefficient':
                gradient is not None and gradient['binary32_words'] == EXPECTED_WORDS,
            'backward_does_not_consume_generator': state_after_forward == state_after_backward,
            'parameter_value_and_version_unchanged': before['parameter'] == after['parameter'],
            'seed_value_and_version_unchanged': before['seed'] == after['seed'],
        }
        aliases = None if gradient is None else {
            'gradient_is_parameter_tensor': gradient['tensor_identity'] == after['parameter']['tensor_identity'],
            'gradient_is_seed_tensor': gradient['tensor_identity'] == after['seed']['tensor_identity'],
            'gradient_shares_parameter_storage': gradient['storage_identity'] == after['parameter']['storage_identity'],
            'gradient_shares_seed_storage': gradient['storage_identity'] == after['seed']['storage_identity'],
            'parameter_shares_base_storage': after['parameter']['storage_identity'] == after['parameter_base']['storage_identity'],
            'seed_shares_base_storage': after['seed']['storage_identity'] == after['seed_base']['storage_identity'],
        }
        return {'branch_creation_order': branch_creation_order,
                'leaf_before_forward': leaf_before_forward,
                'before_backward': before, 'after_backward': after,
                'mathematical_expected_gradient_words': EXPECTED_WORDS,
                'gradient_identity_relations': aliases,
                'generator_before_forward': state_before_forward,
                'generator_after_forward': state_after_forward,
                'generator_after_backward': state_after_backward, 'checks': checks}

    rows = []
    with torch.autograd.set_multithreading_enabled(False):
        for name in CASES:
            row = {'case': name}
            try:
                observation = run_case(name)
                row.update(status='OBSERVED' if all(observation['checks'].values()) else 'PROBE_ERROR',
                           observation=observation)
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
    base = {'schema': 'heptatorch.pinned-torch-zero-nonzero-merge-layout-observations.v1',
            'probe_sha256': digest(__file__), 'reference_support_sha256': SUPPORT_SHA,
            'lock_sha256': LOCK_SHA, 'reference_source_commit': SOURCE_PIN,
            'expected_cases': 2, 'case_plan': list(CASES),
            'mathematical_expected_gradient_words': EXPECTED_WORDS,
            'expected_derivation': 'Each logical element has derivative coefficient + zero from Bernoulli, multiplied by an all-one seed.',
            'native_executed': False, 'native_values_compared': 0,
            'identity_policy': 'Tensor and Storage identities are opaque observations within each process. Version counters are recorded; equal counters do not establish shared Version identity.',
            'comparison_policy': 'Exact binary32 words in logical order; record shape, strides, offset, Storage size, opaque identity relations, absolute Version counters, and all Generator state bytes.',
            'not_covered': ['gradient hooks', 'higher-order derivatives', 'non-CPU or non-float32 tensors',
                            'native equality is not established by this observation-only probe'],
            'full_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    assert len(CASES) == 2
    if args.prepare_only:
        emit(args.output, dict(base, status='PREPARED_NOT_RUN', cases=0, torch_executed=False))
        print(json.dumps({'status': 'PREPARED_NOT_RUN', 'expected_cases': 2,
                          'probe_sha256': base['probe_sha256']}))
        return 0
    provenance_path = args.output.with_name(args.output.stem + '-provenance.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    old_argv, old_cwd = sys.argv, Path.cwd()
    try:
        os.chdir(args.repo)
        sys.argv = [str(args.reference), '/dev/null', str(provenance_path), str(args.lock)]
        try:
            support = runpy.run_path(str(args.reference), run_name='zero_merge_layout_probe_support')
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
    order_comparison = None
    if not errors:
        left, right = (row['observation'] for row in rows)
        order_comparison = {
            'logical_gradient_words_equal': left['after_backward']['gradient']['binary32_words'] == right['after_backward']['gradient']['binary32_words'],
            'forward_generator_states_equal': left['generator_after_forward'] == right['generator_after_forward'],
            'backward_generator_states_equal': left['generator_after_backward'] == right['generator_after_backward'],
        }
        if not all(order_comparison.values()):
            errors += 1
    report = dict(base, status='PROBE_ERROR' if errors else 'OBSERVED', cases=len(rows),
        torch_executed=True, probe_errors=errors, rows=rows, order_comparison=order_comparison,
        oracle_provenance=provenance,
        source_files_unchanged={'probe': digest(__file__) == base['probe_sha256'],
            'support': digest(args.reference) == SUPPORT_SHA, 'lock': digest(args.lock) == LOCK_SHA})
    assert all(report['source_files_unchanged'].values())
    emit(args.output, report)
    print(json.dumps({'status': report['status'], 'cases': len(rows), 'probe_errors': errors,
                      'probe_sha256': base['probe_sha256']}))
    return int(errors != 0)


if __name__ == '__main__':
    raise SystemExit(main())
