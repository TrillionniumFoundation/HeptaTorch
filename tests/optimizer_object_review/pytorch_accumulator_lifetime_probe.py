"""Observe real Torch ownership of captured AccumulateGrad edges after flag changes.

This is a separate three-case reference probe. It does not modify the earlier
six requires_grad recipes or the 24 Optimizer-object observations.
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
import weakref

SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
PIN = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
CASE_NAMES = (
    'captured-leaf-accumulator-retains-disabled-publicly-dropped-leaf-with-None',
    'captured-leaf-accumulator-retains-disabled-publicly-dropped-leaf-with-existing-grad',
    'nonrequiring-forward-input-enabled-later-has-Saved-storage-but-no-accumulator',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def execute(torch):
    def info(tensor):
        words = tensor.detach().contiguous().reshape(-1).view(torch.int32).tolist()
        return {'shape': list(tensor.shape), 'strides': list(tensor.stride()),
                'requires_grad': bool(tensor.requires_grad), 'version': int(tensor._version),
                'binary32_words': [int(word) & 0xffffffff for word in words],
                'tensor_identity': str(tensor._cdata),
                'storage_identity': str(tensor.untyped_storage()._cdata)}

    class WeakStorage:
        def __init__(self, tensor):
            self.token = tensor.untyped_storage()._weak_ref()
        def expired(self):
            return bool(torch.UntypedStorage._expired(self.token))
        def close(self):
            torch.UntypedStorage._free_weak_ref(self.token)

    def observe(tensor_weak, storage_weak, gradient_weak=None):
        gc.collect()
        tensor = tensor_weak()
        result = {'python_Tensor_alive': tensor is not None,
                  'parameter_Storage_expired': storage_weak.expired()}
        if gradient_weak is not None:
            result['original_gradient_Storage_expired'] = gradient_weak.expired()
        if tensor is not None:
            result['parameter'] = info(tensor)
            result['gradient'] = None if tensor.grad is None else info(tensor.grad)
        return result

    def backward(root, retain=False):
        try:
            root.backward(retain_graph=retain)
            return {'outcome': 'SUCCEEDED'}
        except Exception as error:
            return {'outcome': 'RAISED', 'exception_type': type(error).__name__, 'message': str(error)}

    def captured(initial_gradient):
        p = torch.tensor([1., -2.], dtype=torch.float32, requires_grad=True)
        if initial_gradient:
            p.backward(torch.tensor([3., 4.], dtype=torch.float32))
        coefficient = torch.tensor(2., dtype=torch.float32)
        root = (p*coefficient).sum()
        p.requires_grad_(False)
        tensor_weak, storage_weak = weakref.ref(p), WeakStorage(p)
        gradient_weak = WeakStorage(p.grad) if p.grad is not None else None
        try:
            rows = [{'phase': 'False after forward before public drop',
                     'observation': observe(tensor_weak, storage_weak, gradient_weak)}]
            del p
            rows.append({'phase': 'after public drop with root live',
                         'observation': observe(tensor_weak, storage_weak, gradient_weak)})
            result = backward(root, True)
            rows.append({'phase': 'after retained backward while accumulation disabled', 'backward': result,
                         'observation': observe(tensor_weak, storage_weak, gradient_weak)})
            result = backward(root, False)
            rows.append({'phase': 'after default backward with graph root still live', 'backward': result,
                         'observation': observe(tensor_weak, storage_weak, gradient_weak)})
            del root
            rows.append({'phase': 'after final graph-root drop',
                         'observation': observe(tensor_weak, storage_weak, gradient_weak)})
            return {'initial_gradient_defined': initial_gradient, 'events': rows}
        finally:
            storage_weak.close()
            if gradient_weak is not None:
                gradient_weak.close()

    def absent_edge():
        p = torch.tensor([1., -2.], dtype=torch.float32)
        q = torch.tensor([3., 4.], dtype=torch.float32, requires_grad=True)
        root = (p*q).sum()
        p.requires_grad_(True)
        tensor_weak, storage_weak = weakref.ref(p), WeakStorage(p)
        try:
            rows = [{'phase': 'True after nonrequiring forward before public drop',
                     'observation': observe(tensor_weak, storage_weak)}]
            del p
            rows.append({'phase': 'after public drop while saved input is needed',
                         'observation': observe(tensor_weak, storage_weak)})
            result = backward(root)
            rows.append({'phase': 'after default backward with root still live', 'backward': result,
                         'observation': observe(tensor_weak, storage_weak),
                         'q': info(q), 'q_gradient': None if q.grad is None else info(q.grad)})
            del root
            rows.append({'phase': 'after graph-root drop',
                         'observation': observe(tensor_weak, storage_weak)})
            return {'events': rows}
        finally:
            storage_weak.close()

    recipes = (lambda: captured(False), lambda: captured(True), absent_edge)
    rows = []
    with torch.autograd.set_multithreading_enabled(False):
        for name, recipe in zip(CASE_NAMES, recipes):
            row = {'case': name}
            try:
                row.update(status='OBSERVED', observation=recipe())
            except Exception as error:
                row.update(status='PROBE_ERROR', exception_type=type(error).__name__,
                           message=str(error), traceback=traceback.format_exc())
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'lock', 'repo', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    for name in ('reference', 'lock', 'repo', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    assert digest(args.reference) == SUPPORT_SHA and digest(args.lock) == LOCK_SHA
    base = {'schema': 'heptatorch.pinned-torch-captured-accumulator-lifetime-observations.v1',
            'probe_sha256': digest(__file__), 'reference_support_sha256': SUPPORT_SHA,
            'lock_sha256': LOCK_SHA, 'reference_source_commit': PIN,
            'expected_cases': len(CASE_NAMES), 'case_plan': list(CASE_NAMES),
            'native_executed': False, 'native_values_compared': 0,
            'predeclared_future_native_policy': 'Exact Storage weak expiration, None presence and raw binary32 words; opaque object identities compared only as relationships.',
            'scope': 'Captured AccumulateGrad ownership versus saved nonrequiring input Storage after changing leaf flags and dropping public roots.',
            'not_covered': 'Python wrapper weakref identity is observed, but the native wire has no Python-wrapper object; do not claim that API is implemented.',
            'full_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    if args.prepare_only:
        emit(args.output, dict(base, status='PREPARED_NOT_RUN', cases=0))
        print(json.dumps({'status': 'PREPARED_NOT_RUN', 'expected_cases': len(CASE_NAMES), 'probe_sha256': base['probe_sha256']}))
        return 0
    provenance_path = args.output.with_name(args.output.stem+'-provenance.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous_argv, previous_cwd = sys.argv, Path.cwd()
    try:
        os.chdir(args.repo)
        sys.argv = [str(args.reference), '/dev/null', str(provenance_path), str(args.lock)]
        try:
            support = runpy.run_path(str(args.reference), run_name='captured_accumulator_lifetime_probe_support')
        except (SystemExit, ModuleNotFoundError) as error:
            observed = json.loads(provenance_path.read_bytes()) if provenance_path.exists() else None
            emit(args.output, dict(base, status='NOT_RUN', cases=0,
                reason='Pinned reference did not admit this local environment',
                initialization_exception=type(error).__name__, initialization_message=str(error), support_observation=observed))
            return 2
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
    provenance = support['PROVENANCE']
    assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11
    assert all(value is True for value in provenance['checks'].values())
    rows = execute(support['torch'])
    errors = sum(row['status'] != 'OBSERVED' for row in rows)
    report = dict(base, status='PROBE_ERROR' if errors else 'OBSERVED', cases=len(rows), probe_errors=errors,
                  rows=rows, oracle_provenance=provenance,
                  source_files_unchanged={'probe': digest(__file__) == base['probe_sha256'],
                                         'support': digest(args.reference) == SUPPORT_SHA, 'lock': digest(args.lock) == LOCK_SHA})
    assert all(report['source_files_unchanged'].values())
    emit(args.output, report)
    print(json.dumps({'status': report['status'], 'cases': len(rows), 'probe_errors': errors, 'probe_sha256': base['probe_sha256']}))
    return int(errors != 0)


if __name__ == '__main__':
    raise SystemExit(main())
