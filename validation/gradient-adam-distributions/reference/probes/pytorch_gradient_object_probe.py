"""Observe pinned Torch public .grad Tensor identity, assignment and lifetime.

Ten fixed groups exercise actual autograd, Tensor.grad and Optimizer.zero_grad.
This probe has no native executable input. It does not replace the existing
Optimizer case 20, gradient-clear, or leaf-flag observations.
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
PIN = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
CASE_NAMES = (
    'first-grad-Tensor-direct-vs-shared-graph-materialization',
    'first-grad-layout-dense-transpose-and-holey-leaf',
    'grad-reassignment-retains-new-Tensor-and-old-external-alias',
    'assigned-noncontiguous-gradient-keeps-layout-through-backward',
    'assigned-shared-Storage-gradient-alias-mutates-parameter-Version',
    'zero-False-detaches-requiring-leaf-gradient-in-place',
    'zero-False-nonleaf-gradient-versus-view-gradient',
    'grad-setter-rejections-and-None-deletion',
    'scalar-and-empty-grad-Tensor-identity-Version-and-lifetime',
    'partial-backward-materializes-only-ready-leaf-grad-object',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def execute(torch):
    def info(tensor):
        assert tensor.dtype == torch.float32 and tensor.device.type == 'cpu'
        words = tensor.detach().contiguous().reshape(-1).view(torch.int32).tolist()
        return {'shape': list(tensor.shape), 'strides': list(tensor.stride()),
                'storage_offset_bytes': tensor.storage_offset()*tensor.element_size(),
                'storage_bytes': tensor.untyped_storage().nbytes(),
                'requires_grad': bool(tensor.requires_grad), 'is_leaf': bool(tensor.is_leaf),
                'grad_fn_type': None if tensor.grad_fn is None else type(tensor.grad_fn).__name__,
                'version': int(tensor._version), 'dtype': str(tensor.dtype), 'device': str(tensor.device),
                'binary32_words': [int(word) & 0xffffffff for word in words],
                'tensor_identity': str(tensor._cdata),
                'storage_identity': str(tensor.untyped_storage()._cdata)}

    class WeakStorage:
        def __init__(self, tensor):
            self.token = tensor.untyped_storage()._weak_ref()
        def expired(self):
            gc.collect()
            return bool(torch.UntypedStorage._expired(self.token))
        def close(self):
            torch.UntypedStorage._free_weak_ref(self.token)

    def make(values, requires=False):
        return torch.tensor(values, dtype=torch.float32, requires_grad=requires)

    def sgd(params):
        return torch.optim.SGD(params, lr=.125, foreach=False, fused=False)

    def snapshot(params, held=None):
        return {'parameters': {name: info(p) for name, p in params.items()},
                'gradients': {name: None if p.grad is None else info(p.grad) for name, p in params.items()},
                'held_Tensors': {name: info(value) for name, value in (held or {}).items()}}

    def action(label, function, params, held=None):
        row = {'action': label, 'before': snapshot(params, held)}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                result = function()
                row['outcome'] = 'SUCCEEDED'
                if isinstance(result, torch.Tensor):
                    row['returned_Tensor_identity'] = str(result._cdata)
            except Exception as error:
                row.update(outcome='RAISED', exception_type=type(error).__name__, message=str(error))
        row['warnings'] = [{'category': item.category.__name__, 'message': str(item.message)} for item in caught]
        row['after'] = snapshot(params, held)
        return row

    def first_materialization():
        p = make([1., -2.], True); seed = make([-0., 2.]); params = {'p': p}
        initial = snapshot(params, {'seed': seed})
        none_reads = [p.grad is None for _ in range(3)]
        direct = [action('first direct leaf backward with externally held seed', lambda: p.backward(seed), params, {'seed': seed})]
        held = p.grad
        getters = {'same_Tensor_on_repeated_get': p.grad is held and p.grad is held,
                   'same_Tensor_as_seed': held is seed,
                   'same_Storage_as_seed': held.untyped_storage()._cdata == seed.untyped_storage()._cdata}
        direct.append(action('second direct backward accumulates into held grad', lambda: p.backward(make([3., 4.])), params, {'gradient': held, 'seed': seed}))
        q = make([1., -2.], True); root = (q*q).sum(); qparams = {'q': q}
        generated = [action('first shared-parent graph backward', lambda: root.backward(retain_graph=True), qparams)]
        graph_grad = q.grad
        generated.append(action('second graph backward retains existing grad identity', root.backward, qparams, {'gradient': graph_grad}))
        return {'initial': initial, 'None_reads': none_reads, 'direct_actions': direct, 'getter_relationships': getters,
                'graph_actions': generated, 'graph_gradient_same_Tensor': q.grad is graph_grad}

    def first_layout():
        rows = []
        for name in ('dense-transpose', 'holey-offset'):
            if name == 'dense-transpose':
                base = make([[1., -2., 3.], [.5, 2., -1.]])
                p = base.t().requires_grad_()
                seed = make([[1., 2.], [3., -1.], [-2., -3.]])
            else:
                base = make([99., 1., 99., -2.])
                p = base.as_strided((2,), (2,), 1).requires_grad_()
                seed = make([2., -1.])
            params = {'p': p}
            events = [action('first materialization from held seed', lambda: p.backward(seed), params, {'base': base, 'seed': seed})]
            grad = p.grad
            events.append(action('second accumulation preserves existing grad', lambda: p.backward(seed), params, {'base': base, 'seed': seed, 'gradient': grad}))
            rows.append({'layout': name, 'parameter_non_overlapping_and_dense_contract': name == 'dense-transpose',
                         'events': events, 'getter_is_held_gradient': p.grad is grad,
                         'same_Storage_as_seed': grad.untyped_storage()._cdata == seed.untyped_storage()._cdata})
        return {'layouts': rows}

    def reassignment():
        p = make([1., -2.], True); old, new = make([-0., 2.]), make([3., 4.]); params = {'p': p}
        old_weak, new_weak = WeakStorage(old), WeakStorage(new)
        try:
            events = [action('assign first real Tensor object', lambda: setattr(p, 'grad', old), params, {'old': old, 'new': new}),
                      action('replace binding with another real Tensor object', lambda: setattr(p, 'grad', new), params, {'old': old, 'new': new}),
                      action('external old alias mutation leaves current binding independent', lambda: old.add_(.5), params, {'old': old, 'new': new}),
                      action('same-object reassignment does not mutate it', lambda: setattr(p, 'grad', new), params, {'old': old, 'new': new}),
                      action('clear binding retains both external aliases', lambda: setattr(p, 'grad', None), params, {'old': old, 'new': new})]
            before_drop = {'old_expired': old_weak.expired(), 'new_expired': new_weak.expired()}
            del old, new
            return {'events': events, 'before_external_drop': before_drop,
                    'after_external_drop': {'old_expired': old_weak.expired(), 'new_expired': new_weak.expired()},
                    'parameter_after': info(p), 'grad_is_None': p.grad is None}
        finally:
            old_weak.close(); new_weak.close()

    def assigned_layout():
        p = make([[1., -2., 3.], [.5, 2., -1.]], True)
        base = make([[1., 2.], [3., 4.], [5., 6.]])
        gradient = base.t(); params = {'p': p}; held = {'gradient': gradient, 'base': base}
        events = [action('assign noncontiguous same-shape Tensor', lambda: setattr(p, 'grad', gradient), params, held),
                  action('backward adds in place without enforcing new layout', lambda: p.backward(make([[1., 1., 1.], [1., 1., 1.]])), params, held),
                  action('zero False preserves noncontiguous Tensor identity', lambda: sgd([p]).zero_grad(set_to_none=False), params, held)]
        return {'events': events, 'getter_is_assigned_Tensor': p.grad is gradient}

    def storage_alias():
        p = make([1., -2.], True); gradient = p.detach(); params = {'p': p}; held = {'gradient': gradient}
        opt = sgd([p])
        events = [action('distinct Tensor sharing parameter Storage may be assigned', lambda: setattr(p, 'grad', gradient), params, held),
                  action('backward mutation of aliased grad also changes parameter', lambda: p.backward(make([2., -1.])), params, held),
                  action('SGD reads the actual aliased current gradient', opt.step, params, held),
                  action('zero False changes the shared Storage', lambda: opt.zero_grad(set_to_none=False), params, held),
                  action('zero True only clears the binding', lambda: opt.zero_grad(set_to_none=True), params, held)]
        return {'events': events, 'same_Tensor': gradient is p,
                'same_Storage': gradient.untyped_storage()._cdata == p.untyped_storage()._cdata}

    def requiring_leaf():
        p = make([1., -2.], True); gradient = make([2., -1.], True); params = {'p': p}; held = {'gradient': gradient}
        opt = sgd([p])
        events = [action('assign requiring leaf gradient', lambda: setattr(p, 'grad', gradient), params, held),
                  action('zero False disables grad tracking on same gradient Tensor', lambda: opt.zero_grad(set_to_none=False), params, held),
                  action('clear true retains external Tensor', lambda: opt.zero_grad(set_to_none=True), params, held),
                  action('None optimizer step skips parameter', opt.step, params, held)]
        return {'events': events}

    def nonleaf_and_view():
        rows = []
        for name in ('ordinary-nonleaf', 'differentiable-view'):
            p = make([1., -2.], True); base = make([2., -1.], True)
            gradient = base*2. if name == 'ordinary-nonleaf' else base.view(2)
            params = {'p': p}; held = {'base': base, 'gradient': gradient}; opt = sgd([p])
            events = [action('assign '+name+' as actual grad', lambda: setattr(p, 'grad', gradient), params, held),
                      action('zero False applies actual detach_ rule', lambda: opt.zero_grad(set_to_none=False), params, held),
                      action('zero True still clears binding', lambda: opt.zero_grad(set_to_none=True), params, held)]
            rows.append({'source_kind': name, 'events': events})
        return {'variants': rows}

    def rejected_setters():
        p = make([1., -2.], True); valid = make([3., 4.]); p.grad = valid
        params = {'p': p}; held = {'valid_gradient': valid}; events = []
        invalids = (
            ('Python list', lambda: [1., 2.]),
            ('Tensor is its own grad', lambda: p),
            ('same numel different rank', lambda: make([[1., 2.]])),
            ('short shape', lambda: make([1.])),
            ('empty shape mismatch', lambda: make([])),
            ('wrong dtype under default grad_dtype', lambda: torch.tensor([1., 2.], dtype=torch.float64)),
            ('wrong device type', lambda: torch.empty((2,), dtype=torch.float32, device='meta')),
        )
        for name, factory in invalids:
            value = factory()
            events.append(action(name, lambda value=value: setattr(p, 'grad', value), params, held))
        events.append(action('None is accepted', lambda: setattr(p, 'grad', None), params, held))
        events.append(action('reassign valid object', lambda: setattr(p, 'grad', valid), params, held))
        events.append(action('delete property invokes actual setter contract', lambda: delattr(p, 'grad'), params, held))
        return {'events': events}

    def scalar_empty():
        rows = []
        for name in ('scalar-negative-zero', 'empty-rank2'):
            p = make(2., True) if name == 'scalar-negative-zero' else torch.empty((0, 3), dtype=torch.float32, requires_grad=True)
            seed = make(-0.) if name == 'scalar-negative-zero' else torch.empty((0, 3), dtype=torch.float32)
            params = {'p': p}; opt = sgd([p])
            events = [action('first backward produces actual grad Tensor', lambda: p.backward(seed), params, {'seed': seed})]
            gradient = p.grad; weak = WeakStorage(gradient)
            try:
                events.append(action('zero False mutates same scalar or empty Tensor', lambda: opt.zero_grad(set_to_none=False), params, {'gradient': gradient, 'seed': seed}))
                events.append(action('zero True preserves external old grad', lambda: opt.zero_grad(set_to_none=True), params, {'gradient': gradient, 'seed': seed}))
                old_alive = not weak.expired()
                events.append(action('new first gradient has independent binding', lambda: p.backward(seed), params, {'gradient': gradient, 'seed': seed}))
                same = {'Tensor': p.grad is gradient, 'Storage': p.grad.untyped_storage()._cdata == gradient.untyped_storage()._cdata}
                del gradient
                rows.append({'shape_kind': name, 'events': events, 'old_storage_alive_after_clear': old_alive,
                             'new_equals_old': same, 'old_storage_expired_after_external_drop': weak.expired()})
            finally:
                weak.close()
        return {'variants': rows}

    def partial_ready():
        bad, good = make([1., -2.], True), make([3., 4.], True)
        coefficient = make(2.); bad_branch = bad*coefficient; good_branch = good*good
        root = (bad_branch+good_branch).sum(); coefficient.add_(1.)
        params = {'bad': bad, 'good': good}
        events = [action('later-created branch reaches leaf before saved-version failure', root.backward, params)]
        gradient = good.grad
        assert gradient is not None
        weak = WeakStorage(gradient)
        try:
            events.append(action('zero True clears ready binding and skips pending None', lambda: sgd([bad, good]).zero_grad(set_to_none=True), params, {'ready_gradient': gradient}))
            alive = not weak.expired()
            del gradient
            return {'events': events, 'external_ready_gradient_survived_clear': alive,
                    'ready_gradient_Storage_expired_after_external_drop': weak.expired()}
        finally:
            weak.close()

    recipes = (first_materialization, first_layout, reassignment, assigned_layout, storage_alias,
               requiring_leaf, nonleaf_and_view, rejected_setters, scalar_empty, partial_ready)
    rows = []
    with torch.autograd.set_multithreading_enabled(False):
        for name, recipe in zip(CASE_NAMES, recipes):
            row = {'case': name}
            try:
                row.update(status='OBSERVED', observation=recipe())
            except Exception as error:
                row.update(status='PROBE_ERROR', exception_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'lock', 'repo', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    a = parser.parse_args()
    for name in ('reference', 'lock', 'repo', 'output'):
        setattr(a, name, getattr(a, name).resolve())
    assert digest(a.reference) == SUPPORT_SHA and digest(a.lock) == LOCK_SHA
    base = {'schema': 'heptatorch.pinned-torch-public-gradient-object-observations.v1',
            'probe_sha256': digest(__file__), 'reference_support_sha256': SUPPORT_SHA,
            'lock_sha256': LOCK_SHA, 'reference_source_commit': PIN,
            'expected_groups': len(CASE_NAMES), 'case_plan': list(CASE_NAMES),
            'native_executed': False, 'native_values_compared': 0,
            'predeclared_future_native_policy': 'Exact unsigned binary32 words, shape/stride/offset/size metadata, None presence, parameter/gradient Versions, opaque Tensor/Storage identity relationships, and Storage weak expiration. No floating tolerance for these dyadic fixtures.',
            'scope': 'Public gradient Tensor binding, first materialization and accumulation, assignment/reassignment guards, external aliases, zero_grad detachment, partial readiness and scalar/empty layouts.',
            'not_covered': ['higher-order/create_graph', 'sparse/complex/fused/foreach gradient implementations', 'custom grad_dtype', 'gradient-reference cycles', 'Python-wrapper weakref implementation', 'arbitrary devices/dtypes beyond recorded rejection observations'],
            'full_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    if a.prepare_only:
        emit(a.output, dict(base, status='PREPARED_NOT_RUN', cases=0))
        print(json.dumps({'status': 'PREPARED_NOT_RUN', 'expected_groups': len(CASE_NAMES), 'probe_sha256': base['probe_sha256']}))
        return 0
    provenance_path = a.output.with_name(a.output.stem+'-provenance.json')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    previous_argv, previous_cwd = sys.argv, Path.cwd()
    try:
        os.chdir(a.repo)
        sys.argv = [str(a.reference), '/dev/null', str(provenance_path), str(a.lock)]
        try:
            support = runpy.run_path(str(a.reference), run_name='public_gradient_object_probe_support')
        except (SystemExit, ModuleNotFoundError) as error:
            observed = json.loads(provenance_path.read_bytes()) if provenance_path.exists() else None
            emit(a.output, dict(base, status='NOT_RUN', cases=0, reason='Pinned reference did not admit this local environment',
                initialization_exception=type(error).__name__, initialization_message=str(error), support_observation=observed))
            return 2
    finally:
        sys.argv = previous_argv; os.chdir(previous_cwd)
    provenance = support['PROVENANCE']
    assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11 and all(v is True for v in provenance['checks'].values())
    rows = execute(support['torch']); errors = sum(row['status'] != 'OBSERVED' for row in rows)
    report = dict(base, status='PROBE_ERROR' if errors else 'OBSERVED', cases=len(rows), probe_errors=errors, rows=rows,
        oracle_provenance=provenance, source_files_unchanged={'probe': digest(__file__) == base['probe_sha256'],
            'support': digest(a.reference) == SUPPORT_SHA, 'lock': digest(a.lock) == LOCK_SHA})
    assert all(report['source_files_unchanged'].values())
    emit(a.output, report)
    print(json.dumps({'status': report['status'], 'cases': len(rows), 'probe_errors': errors, 'probe_sha256': base['probe_sha256']}))
    return int(errors != 0)


if __name__ == '__main__':
    raise SystemExit(main())
