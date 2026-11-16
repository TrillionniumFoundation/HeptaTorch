"""Observe Bernoulli backward's real gradient objects, layouts and aliases.

Eight fixed cases supplement the frozen 183 distribution and 25/12 gradient
observations. This is reference-only: no native result is an expected value.
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
import warnings

SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
SOURCE_FRAMEWORK_SHA = 'cc0392888f0db9f7026c3f24e0787845b60e43b5e18d8c63cdcb8c720d51acd5'
PIN = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
CASES = (
    'contiguous-parameter-contiguous-seed',
    'contiguous-parameter-transpose-seed',
    'transpose-parameter-transpose-seed',
    'transpose-parameter-contiguous-seed',
    'non-dense-parameter-transpose-seed',
    'empty-contiguous-parameter-transpose-seed',
    'empty-transpose-parameter-contiguous-seed',
    'special-word-contiguous-seed',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def execute(torch):
    from torch.multiprocessing.reductions import StorageWeakRef

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

    def make_case(name):
        empty = name.startswith('empty-')
        transpose_parameter = name.startswith('transpose-') or name.startswith('empty-transpose-')
        if empty:
            p = torch.empty((3, 0) if transpose_parameter else (0, 3), dtype=torch.float32)
        elif name.startswith('non-dense-'):
            p = torch.full((2, 6), .5, dtype=torch.float32)[:, ::2]
        else:
            p = torch.full((3, 2) if transpose_parameter else (2, 3), .5, dtype=torch.float32)
        if transpose_parameter:
            p = p.t()
        p.requires_grad_(True)
        transpose_seed = name.endswith('transpose-seed')
        if empty:
            seed = torch.empty((3, 0) if transpose_seed else (0, 3), dtype=torch.float32)
        elif name == 'special-word-contiguous-seed':
            seed = torch.tensor([-2147483648, 2139095040, -8388608, 2143289345, 1065353216, 1],
                                dtype=torch.int32).view(torch.float32).reshape(2, 3)
        else:
            seed = torch.tensor([[-0., 2.], [3., 4.], [5., 6.]] if transpose_seed else
                                [[-0., 2., 3.], [4., 5., 6.]], dtype=torch.float32)
        if transpose_seed:
            seed = seed.t()
        assert p.shape == seed.shape
        return p, seed

    def run_case(name):
        p, seed = make_case(name)
        generator = torch.Generator(device='cpu').manual_seed(12345)
        state_before = bytes(generator.get_state().tolist()).hex()
        output = torch.bernoulli(p, generator=generator)
        state_after_forward = bytes(generator.get_state().tolist()).hex()
        held = {'seed': seed}
        events = []

        def snapshot():
            return {'parameter': info(p), 'output': info(output),
                    'gradient': None if p.grad is None else info(p.grad),
                    'held_Tensors': {key: info(value) for key, value in held.items()}}

        def action(label, operation):
            row = {'action': label, 'before': snapshot()}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    operation()
                    row['outcome'] = 'SUCCEEDED'
                except Exception as error:
                    row.update(outcome='RAISED', exception_type=type(error).__name__, message=str(error))
            row['warnings'] = [{'category': item.category.__name__, 'message': str(item.message)} for item in caught]
            row['after'] = snapshot()
            events.append(row)

        def mutate(key, scalar):
            with torch.no_grad():
                held[key].add_(scalar)

        action('first Bernoulli backward with held seed', lambda: output.backward(seed))
        assert p.grad is not None
        held['old_gradient'] = p.grad
        held['old_alias'] = p.grad.detach()
        old_weak = StorageWeakRef(held['old_gradient'].untyped_storage())
        action('seed add_(0.5)', lambda: mutate('seed', .5))
        action('old gradient add_(1)', lambda: mutate('old_gradient', 1.))
        action('clear parameter.grad', lambda: setattr(p, 'grad', None))
        action('old alias add_(2) after clear', lambda: mutate('old_alias', 2.))
        action('second Bernoulli backward materializes a new gradient', lambda: output.backward(seed))
        assert p.grad is not None
        held['new_gradient'] = p.grad
        new_weak = StorageWeakRef(held['new_gradient'].untyped_storage())
        action('old alias add_(4) after new gradient materialization', lambda: mutate('old_alias', 4.))
        old_before_drop = old_weak.expired()
        del held['old_gradient']
        gc.collect()
        old_alias_only = {'storage_expired': old_weak.expired(), 'snapshot': snapshot()}
        del held['old_alias']
        gc.collect()
        old_after_drop = old_weak.expired()
        action('clear new gradient binding while publicly held', lambda: setattr(p, 'grad', None))
        new_before_drop = new_weak.expired()
        del held['new_gradient']
        gc.collect()
        return {'events': events, 'old_storage_expired_before_drops': old_before_drop,
                'old_alias_only': old_alias_only, 'old_storage_expired_after_final_alias_drop': old_after_drop,
                'new_storage_expired_before_public_drop': new_before_drop,
                'new_storage_expired_after_public_drop': new_weak.expired(), 'final': snapshot(),
                'generator_state_before_forward_hex': state_before,
                'generator_state_after_forward_hex': state_after_forward,
                'generator_state_after_all_backward_hex': bytes(generator.get_state().tolist()).hex()}

    rows = []
    with torch.autograd.set_multithreading_enabled(False):
        for name in CASES:
            row = {'case': name}
            try:
                row.update(status='OBSERVED', observation=run_case(name))
            except Exception as error:
                row.update(status='PROBE_ERROR', exception_type=type(error).__name__, message=str(error),
                           traceback=traceback.format_exc())
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
    base = {'schema': 'heptatorch.pinned-torch-bernoulli-gradient-layout-observations.v1',
            'probe_sha256': digest(__file__), 'source_framework_sha256': SOURCE_FRAMEWORK_SHA,
            'source_framework_marker_meaning': 'Metadata/provenance framework origin only; this new probe has its own source identity.',
            'reference_support_sha256': SUPPORT_SHA, 'lock_sha256': LOCK_SHA,
            'reference_source_commit': PIN, 'expected_cases': 8, 'case_plan': list(CASES),
            'native_executed': False, 'native_values_compared': 0,
            'future_comparison_policy': 'Exact words, shape/strides/offset, actual Version changes, opaque Tensor and Storage relations, StorageWeakRef lifetime, and complete Generator state. No native output is substituted for an observed reference.',
            'identity_policy': 'Opaque identities are compared only as relations within one process. Equal Version values do not establish shared Version identity; separate seed/gradient mutations are recorded.',
            'not_covered': ['gradient hooks', 'create_graph or higher-order derivatives', 'non-CPU or non-float32 tensors',
                            'arbitrary overlapping layouts', 'native equality is not established by this observation-only probe'],
            'full_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    assert len(CASES) == 8
    if args.prepare_only:
        emit(args.output, dict(base, status='PREPARED_NOT_RUN', cases=0))
        print(json.dumps({'status': 'PREPARED_NOT_RUN', 'expected_cases': 8, 'probe_sha256': base['probe_sha256']}))
        return 0
    provenance_path = args.output.with_name(args.output.stem + '-provenance.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    old_argv, old_cwd = sys.argv, Path.cwd()
    try:
        os.chdir(args.repo)
        sys.argv = [str(args.reference), '/dev/null', str(provenance_path), str(args.lock)]
        try:
            support = runpy.run_path(str(args.reference), run_name='bernoulli_gradient_layout_probe_support')
        except (SystemExit, ModuleNotFoundError) as error:
            observed = json.loads(provenance_path.read_bytes()) if provenance_path.exists() else None
            emit(args.output, dict(base, status='NOT_RUN', cases=0,
                reason='Pinned reference did not admit this environment',
                initialization_exception=type(error).__name__, initialization_message=str(error),
                support_observation=observed))
            return 2
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
    provenance = support['PROVENANCE']
    assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11 and all(value is True for value in provenance['checks'].values())
    rows = execute(support['torch'])
    errors = sum(row['status'] != 'OBSERVED' for row in rows)
    report = dict(base, status='PROBE_ERROR' if errors else 'OBSERVED', cases=len(rows),
        probe_errors=errors, rows=rows, oracle_provenance=provenance,
        source_files_unchanged={'probe': digest(__file__) == base['probe_sha256'],
            'support': digest(args.reference) == SUPPORT_SHA, 'lock': digest(args.lock) == LOCK_SHA})
    assert all(report['source_files_unchanged'].values())
    emit(args.output, report)
    print(json.dumps({'status': report['status'], 'cases': len(rows), 'probe_errors': errors,
                      'probe_sha256': base['probe_sha256']}))
    return int(errors != 0)


if __name__ == '__main__':
    raise SystemExit(main())
