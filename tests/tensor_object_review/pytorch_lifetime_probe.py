"""Observe pinned PyTorch storage/version/graph lifetime; never run target code.

The unchanged public oracle module supplies its eleven provenance checks only.
Every numerical result below is evaluated by the installed, verified PyTorch.
Raw native pointer values from the private weak-storage API are never reported.
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
PIN = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
CASE_NAMES = (
    'view-detach-clone-data-storage-and-version-identities',
    'empty-transpose-clone-preserve-format',
    'add-input-mutation',
    'clone-input-mutation',
    'mul-required-input-mutation-when-only-that-input-needs-grad',
    'mul-nongrad-coefficient-mutation',
    'relu-input-mutation',
    'relu-output-mutation',
    'gather-input-mutation',
    'avg-pool-input-mutation',
    'index-add-source-mutation',
    'scatter-add-source-mutation',
    'nonleaf-storage-not-needed-by-later-derivative',
    'saved-output-storage-retain-and-default-backward',
    'leaf-storage-retained-by-accumulate-grad',
    'nongrad-coefficient-storage-retained-only-by-saved-value',
    'matmul-mm-mutated-unsaved-left',
    'matmul-mm-mutated-saved-right',
    'matmul-folded-view-mutated-saved-left',
    'matmul-folded-copy-mutated-original-left',
    'matmul-broadcast-reshape-copy-mutated-original-left',
    'matmul-mm-noncontiguous-mutated-saved-left',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def execute(torch, F):
    rows = []

    def snapshot(t):
        return {'shape': list(t.shape), 'strides': list(t.stride()),
                'offset_elements': t.storage_offset(), 'version': t._version,
                'requires_grad': t.requires_grad, 'is_leaf': t.is_leaf,
                'values': t.detach().reshape(-1).tolist()}

    class WeakStorage:
        def __init__(self, tensor):
            self.token = tensor.untyped_storage()._weak_ref()

        def expired(self):
            gc.collect()
            return bool(torch.UntypedStorage._expired(self.token))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            torch.UntypedStorage._free_weak_ref(self.token)

    def observe(name, fn):
        try:
            rows.append({'case': name, 'status': 'OBSERVED', 'observation': fn()})
        except Exception as exc:
            rows.append({'case': name, 'status': 'PROBE_ERROR',
                         'exception_type': type(exc).__name__, 'message': str(exc),
                         'traceback': traceback.format_exc()})

    def aliases():
        a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
        transposed, detached, data = a.transpose(0, 1), a.detach(), a.data
        copied = transposed.clone()
        before = {name: snapshot(t) for name, t in
                  [('base', a), ('transpose', transposed), ('detach', detached),
                   ('data', data), ('clone', copied)]}
        sharing = {name: bool(t.untyped_storage()._cdata == a.untyped_storage()._cdata)
                   for name, t in [('transpose', transposed), ('detach', detached),
                                   ('data', data), ('clone', copied)]}
        detached.add_(10.)
        after_detach = {name: t._version for name, t in
                        [('base', a), ('transpose', transposed), ('detach', detached),
                         ('data', data), ('clone', copied)]}
        data.add_(100.)
        after_data = {name: snapshot(t) for name, t in
                      [('base', a), ('transpose', transposed), ('detach', detached),
                       ('data', data), ('clone', copied)]}
        return {'before': before, 'shares_base_storage': sharing,
                'after_detach_mutation_versions': after_detach,
                'after_data_mutation': after_data}

    observe(CASE_NAMES[0], aliases)

    def empty_clone():
        a = torch.empty((0, 2, 3), requires_grad=True).transpose(1, 2)
        b = a.clone()
        return {'input': snapshot(a), 'clone': snapshot(b),
                'shares_storage': bool(a.untyped_storage()._cdata == b.untyped_storage()._cdata)}

    observe(CASE_NAMES[1], empty_clone)

    def mutation(builder):
        leaves, output, changed, configuration = builder()
        before = {'leaves': [snapshot(t) for t in leaves],
                  'output': snapshot(output), 'changed': snapshot(changed)}
        with torch.no_grad():
            changed.add_(4.)
        exception = None
        try:
            output.sum().backward()
            outcome = 'BACKWARD_SUCCEEDED'
        except Exception as exc:
            outcome = 'BACKWARD_RAISED'
            exception = {'type': type(exc).__name__, 'message': str(exc)}
        return {'configuration': configuration, 'before': before,
                'after_changed': snapshot(changed), 'outcome': outcome,
                'exception': exception,
                'gradients': [None if t.grad is None else t.grad.detach().tolist()
                              for t in leaves]}

    def unary_builder(kind, change_output=False):
        x = torch.tensor([-2., 1., 3.], requires_grad=True)
        if kind == 'add': y = x + 2.
        elif kind == 'clone': y = x.clone()
        elif kind == 'relu': y = torch.relu(x)
        else: raise AssertionError(kind)
        return [x], y, y if change_output else x, {'operator': kind}

    def mul_builder(change_coefficient):
        x = torch.tensor([1., 2., 3.], requires_grad=True)
        coefficient = torch.tensor([2., -1., 4.])
        return [x, coefficient], x * coefficient, coefficient if change_coefficient else x, {
            'operator': 'mul', 'requires': [True, False]}

    def gather_builder():
        x = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
        index = torch.tensor([[2, 0], [1, 1]], dtype=torch.int64)
        return [x], x.gather(1, index), x, {'operator': 'gather', 'index': index.tolist()}

    def pool_builder():
        x = torch.tensor([[[[1., 2.], [3., 4.]]]], requires_grad=True)
        return [x], F.avg_pool2d(x, 2), x, {'operator': 'avg_pool2d', 'kernel': 2}

    def indexed_builder(scatter):
        x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        source = torch.tensor([[0.5, 1.], [2., -1.]], requires_grad=True)
        if scatter:
            index = torch.tensor([[1, 0], [1, 1]], dtype=torch.int64)
            y = x.scatter_add(0, index, source)
        else:
            index = torch.tensor([1, 0], dtype=torch.int64)
            y = x.index_add(0, index, source, alpha=0.5)
        return [x, source], y, source, {'operator': 'scatter_add' if scatter else 'index_add',
                                       'index': index.tolist()}

    builders = [lambda: unary_builder('add'), lambda: unary_builder('clone'),
                lambda: mul_builder(False), lambda: mul_builder(True),
                lambda: unary_builder('relu'), lambda: unary_builder('relu', True),
                gather_builder, pool_builder, lambda: indexed_builder(False),
                lambda: indexed_builder(True)]
    for name, builder in zip(CASE_NAMES[2:12], builders, strict=True):
        observe(name, lambda builder=builder: mutation(builder))

    def nonleaf_lifetime():
        x = torch.tensor([1., 2., 3.], requires_grad=True)
        intermediate = x * 2.
        output = (intermediate + 1.).sum()
        with WeakStorage(intermediate) as weak:
            alive_while_public = not weak.expired()
            del intermediate
            expired_after_public_drop = weak.expired()
            output.backward()
            return {'alive_while_public': alive_while_public,
                    'expired_after_public_drop': expired_after_public_drop,
                    'expired_after_backward': weak.expired(), 'gradient': x.grad.tolist()}

    def saved_lifetime():
        x = torch.tensor([-1., 0., 1.], requires_grad=True)
        intermediate = x.tanh()
        output = (intermediate * intermediate).sum()
        with WeakStorage(intermediate) as weak:
            del intermediate
            after_public_drop = weak.expired()
            output.backward(retain_graph=True)
            after_retained = weak.expired()
            output.backward()
            after_default = weak.expired()
            return {'expired_after_public_drop': after_public_drop,
                    'expired_after_retained_backward': after_retained,
                    'expired_after_default_backward': after_default,
                    'gradient_after_two_backwards': x.grad.tolist()}

    def leaf_lifetime():
        x = torch.tensor([1., 2., 3.], requires_grad=True)
        output = (x + 1.).sum()
        with WeakStorage(x) as weak:
            del x
            after_public_drop = weak.expired()
            output.backward()
            after_backward = weak.expired()
            del output
            return {'expired_after_public_drop': after_public_drop,
                    'expired_after_default_backward_with_root_alive': after_backward,
                    'expired_after_root_drop': weak.expired()}

    def coefficient_lifetime():
        x = torch.tensor([1., 2., 3.], requires_grad=True)
        coefficient = torch.tensor([2., 3., 4.])
        output = (x * coefficient).sum()
        with WeakStorage(coefficient) as weak:
            del coefficient
            after_public_drop = weak.expired()
            output.backward()
            after_backward = weak.expired()
            return {'expired_after_public_drop': after_public_drop,
                    'expired_after_default_backward_with_root_alive': after_backward,
                    'gradient': x.grad.tolist()}

    for name, fn in zip(CASE_NAMES[12:16],
                        [nonleaf_lifetime, saved_lifetime, leaf_lifetime, coefficient_lifetime], strict=True):
        observe(name, fn)

    def matmul_builder(mode):
        if mode in (0, 1):
            a = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
            b = torch.tensor([[0.5, -1.], [2., 1.]])
            changed = a if mode == 0 else b
        elif mode in (2, 3):
            a = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
            if mode == 3: a = a.transpose(0, 1)
            b = torch.tensor([[1., 2.], [3., 4.], [-1., 1.], [0.5, -0.5]], requires_grad=True)
            changed = a
        elif mode == 4:
            a = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4)
            b = torch.arange(16, dtype=torch.float32).reshape(1, 2, 4, 2).requires_grad_()
            changed = a
        elif mode == 5:
            a = torch.arange(6, dtype=torch.float32).reshape(3, 2).transpose(0, 1)
            b = torch.tensor([[1., -1.], [2., 3.], [4., 0.5]], requires_grad=True)
            changed = a
        else:
            raise AssertionError(mode)
        y = a @ b
        return [a, b], y, changed, {'operator': 'matmul', 'mode': mode,
                                  'grad_fn': type(y.grad_fn).__name__}

    for mode, name in enumerate(CASE_NAMES[16:]):
        observe(name, lambda mode=mode: mutation(lambda: matmul_builder(mode)))
    assert tuple(row['case'] for row in rows) == CASE_NAMES
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    for name in ('reference', 'lock', 'repo', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    if digest(args.reference) != SUPPORT_SHA:
        raise SystemExit('Refusing changed provenance/support module')
    base = {'schema': 'heptatorch.pinned-torch-lifetime-observations.v1',
            'probe_sha256': digest(__file__), 'support_sha256': SUPPORT_SHA,
            'lock_sha256': digest(args.lock), 'upstream_commit': PIN,
            'expected_cases': len(CASE_NAMES), 'case_names': list(CASE_NAMES),
            'target_execution': False, 'native_values_compared': 0,
            'scope': 'Pinned real PyTorch observations only; no native compatibility claim.'}
    if args.prepare_only:
        write(args.output, dict(base, status='PREPARED_NOT_RUN', cases=0))
        return
    provenance_output = args.output.with_name(args.output.stem + '-provenance.json')
    cwd, argv = Path.cwd(), sys.argv[:]
    try:
        os.chdir(args.repo)
        sys.argv = [str(args.reference), '/dev/null', str(provenance_output), str(args.lock)]
        try:
            scope = runpy.run_path(str(args.reference), run_name='pinned_lifetime_support')
        except SystemExit as exc:
            observed = json.loads(provenance_output.read_text()) if provenance_output.exists() else None
            write(args.output, dict(base, status='NOT_RUN', cases=0,
                                    reason='Pinned support did not admit this environment',
                                    support_exit=exc.code, support_observation=observed))
            raise SystemExit(2)
    finally:
        sys.argv = argv
        os.chdir(cwd)
    assert scope['PROVENANCE']['status'] == 'PASS'
    assert len(scope['PROVENANCE']['checks']) == 11
    rows = execute(scope['torch'], scope['F'])
    errors = sum(row['status'] != 'OBSERVED' for row in rows)
    report = dict(base, status='OBSERVED' if not errors else 'PROBE_ERROR',
                  cases=len(rows), probe_errors=errors, rows=rows,
                  oracle_provenance=scope['PROVENANCE'])
    write(args.output, report)
    print(json.dumps({'status': report['status'], 'cases': len(rows), 'probe_errors': errors}))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
