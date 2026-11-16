#!/usr/bin/env python3
"""Observe retained nonleaf gradients, optimizer membership and weak lifetime.

Only the pinned real PyTorch reference executes this probe. No native program,
native-derived expected values, or custom gradient hook is used. Explicitly
held gradients/seed tensors are part of the fixed recipe.
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
import weakref


SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
SOURCE_PIN = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
ORIGINAL_OPTIMIZER_REFERENCE_SHA = '7ad0a2e718300a33c91a9212f14ada3c0c7997be7fc77f64bb557c2ecd858160'
PLAN = (
    ('admission-leaf-noop-and-idempotence', 1),
    ('retained-gradient-replacement-clear-and-old-saved-alias', 1),
    ('transpose-incoming-and-assigned-gradient-layout', 1),
    ('empty-retained-gradient-replacement', 1),
    ('retained-nonleaf-owned-optimizer-state', 3),
    ('retained-hook-before-own-saved-version-failure', 1),
    ('retained-hook-before-own-released-saved-failure', 1),
    ('weak-retention-versus-optimizer-strong-root', 3),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def execute(torch):
    def tensor(values, requires_grad=False):
        return torch.tensor(values, dtype=torch.float32, device='cpu', requires_grad=requires_grad)

    def info(value):
        assert value.dtype == torch.float32 and value.device.type == 'cpu'
        words = value.detach().contiguous().reshape(-1).view(torch.int32).tolist()
        return {
            'shape': list(value.shape), 'strides': list(value.stride()),
            'storage_offset_bytes': value.storage_offset() * value.element_size(),
            'storage_bytes': value.untyped_storage().nbytes(),
            'dtype': str(value.dtype), 'requires_grad': bool(value.requires_grad),
            'is_leaf': bool(value.is_leaf), 'retains_grad': bool(value.retains_grad),
            'grad_fn_type': None if value.grad_fn is None else type(value.grad_fn).__name__,
            'version': int(value._version),
            'binary32_words': [int(word) & 0xffffffff for word in words],
            'tensor_identity': str(value._cdata),
            'storage_identity': str(value.untyped_storage()._cdata),
        }

    def warning_rows(observed):
        return [{'category': w.category.__name__, 'message': str(w.message)} for w in observed]

    def capture(named, optimizer=None):
        result = {'tensors': {}, 'gradients': {}, 'gradient_read_warnings': {}}
        for name, value in named.items():
            result['tensors'][name] = info(value)
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter('always')
                grad = value.grad
                result['gradients'][name] = None if grad is None else info(grad)
            result['gradient_read_warnings'][name] = warning_rows(observed)
        if optimizer is not None:
            result['optimizer_type'] = type(optimizer).__name__
            result['optimizer_state_entries'] = len(optimizer.state)
            result['optimizer_groups'] = []
            result['optimizer_state'] = []
            for group in optimizer.param_groups:
                config = {key: value for key, value in group.items()
                          if key != 'params' and isinstance(value, (bool, int, float, str, type(None)))}
                if 'betas' in group:
                    config['betas'] = list(group['betas'])
                result['optimizer_groups'].append({
                    'parameter_identities': [str(p._cdata) for p in group['params']],
                    'config': config})
                for p in group['params']:
                    fields = optimizer.state.get(p)
                    result['optimizer_state'].append({
                        'parameter_identity': str(p._cdata),
                        'entry_exists': fields is not None,
                        'fields': {} if fields is None else {
                            key: info(value) if isinstance(value, torch.Tensor) else value
                            for key, value in fields.items()}})
        return result

    def invoke(label, fn):
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            try:
                value = fn()
                row = {'action': label, 'outcome': 'SUCCEEDED',
                       'return_type': type(value).__name__}
            except Exception as error:
                value = None
                row = {'action': label, 'outcome': 'RAISED',
                       'exception_type': type(error).__name__, 'message': str(error)}
        row['warnings'] = warning_rows(observed)
        return value, row

    def action(label, fn, named, optimizer=None):
        before = capture(named, optimizer)
        value, result = invoke(label, fn)
        del value
        result['before'] = before
        result['after'] = capture(named, optimizer)
        return result

    def optimizer(kind, parameter):
        common = {'lr': .125, 'weight_decay': .25, 'foreach': False,
                  'fused': False, 'differentiable': False}
        if kind == 'SGD':
            return torch.optim.SGD([parameter], momentum=.5, dampening=0., **common)
        return getattr(torch.optim, kind)([parameter], betas=(.5, .75), eps=.01, **common)

    class WeakStorage:
        def __init__(self, value):
            self.handle = value.untyped_storage()._weak_ref()
        def expired(self):
            return bool(torch.UntypedStorage._expired(self.handle))
        def close(self):
            if self.handle is not None:
                torch.UntypedStorage._free_weak_ref(self.handle)
                self.handle = None

    def admission():
        x = tensor([1., -2.], True)
        q = x * 2.
        inert = tensor([3.])
        named = {'leaf': x, 'nonleaf': q, 'inert': inert}
        events = [action('leaf retain is a no-op', x.retain_grad, named),
                  action('requires-false retain request', inert.retain_grad, named)]
        unexpected, outcome = invoke('nonleaf optimizer before retain',
                                     lambda: optimizer('SGD', q))
        del unexpected
        events.append(outcome)
        events.append(action('first nonleaf retain', q.retain_grad, named))
        events.append(action('repeated nonleaf retain', q.retain_grad, named))
        opt, outcome = invoke('nonleaf optimizer after retain', lambda: optimizer('SGD', q))
        events.append(outcome)
        return {'events': events, 'after': capture(named, opt)}

    def replacement():
        x = tensor([1., -2.], True)
        q = x * 2.
        q.retain_grad()
        seed1, seed2, seed3 = tensor([-0., 2.]), tensor([1., -2.]), tensor([-0., -0.])
        named = {'x': x, 'q': q, 'seed1': seed1, 'seed2': seed2, 'seed3': seed3}
        events = [action('first retained backward', lambda: q.backward(seed1, retain_graph=True), named)]
        old1 = q.grad
        z = tensor([1., 1.], True)
        saved_old_gradient_root = (old1 * z).sum()
        named = dict(named, held_gradient1=old1, saved_guard_parameter=z)
        events.append(action('second retained backward', lambda: q.backward(seed2, retain_graph=True), named))
        old2 = q.grad
        named = dict(named, held_gradient2=old2)
        events.append(action('backward through saved old gradient', saved_old_gradient_root.backward, named))
        events.append(action('clear retained gradient reference', lambda: setattr(q, 'grad', None), named))
        events.append(action('first materialization again including negative zero', lambda: q.backward(seed3), named))
        return {'events': events, 'after': capture(named)}

    def strided():
        x = tensor([[1., 2., 3.], [4., 5., 6.]], True)
        q = x.transpose(0, 1)
        q.retain_grad()
        seed = tensor([[1., 3., 5.], [2., 4., 6.]]).transpose(0, 1)
        named = {'x': x, 'q': q, 'held_strided_seed': seed}
        events = [action('first retain clones contiguous despite strided seed',
                         lambda: q.backward(seed, retain_graph=True), named)]
        old = q.grad
        assigned = tensor([[.25, .75, 1.25], [.5, 1., 1.5]]).transpose(0, 1)
        named = dict(named, held_first_gradient=old, assigned_gradient=assigned)
        events.append(action('assign actual strided gradient', lambda: setattr(q, 'grad', assigned), named))
        events.append(action('later retain uses out-of-place gradient addition', lambda: q.backward(seed), named))
        return {'events': events, 'after': capture(named)}

    def empty():
        x = torch.empty((2, 0, 3), dtype=torch.float32, requires_grad=True)
        q = x.transpose(0, 2)
        q.retain_grad()
        seed = torch.empty_strided(tuple(q.shape), (100, 17, 1), dtype=torch.float32)
        named = {'x': x, 'q': q, 'held_empty_seed': seed}
        events = [action('first empty retained backward', lambda: q.backward(seed, retain_graph=True), named)]
        old = q.grad
        named = dict(named, held_empty_gradient=old)
        events.append(action('second empty retained backward', lambda: q.backward(seed), named))
        return {'events': events, 'after': capture(named)}

    def optimized():
        variants = []
        for kind in ('SGD', 'Adam', 'AdamW'):
            x = tensor([1., -2.], True)
            q = x * 2.
            q.retain_grad()
            opt = optimizer(kind, q)
            seed1, seed2 = tensor([2., -1.]), tensor([-.5, .25])
            named = {'x': x, 'q': q, 'seed1': seed1, 'seed2': seed2}
            events = [action('skip before any retained gradient', opt.step, named, opt),
                      action('materialize retained gradient', lambda: q.backward(seed1, retain_graph=True), named, opt),
                      action('first owned-state step', opt.step, named, opt)]
            held = q.grad
            named = dict(named, held_gradient=held)
            events.append(action('optimizer zero False', lambda: opt.zero_grad(set_to_none=False), named, opt))
            events.append(action('defined zero gradient is consumed', opt.step, named, opt))
            events.append(action('optimizer zero True', lambda: opt.zero_grad(set_to_none=True), named, opt))
            events.append(action('None preserves existing parameter and state', opt.step, named, opt))
            events.append(action('rebuild retained gradient after parameter mutation', lambda: q.backward(seed2), named, opt))
            events.append(action('resume with existing owned state', opt.step, named, opt))
            variants.append({'variant': kind, 'events': events, 'after': capture(named, opt)})
        return {'variants': variants}

    def own_version_failure():
        x, c = tensor([1., -2.], True), tensor(2.)
        q = x * c
        q.retain_grad()
        seed = tensor([1., 1.])
        named = {'x': x, 'coefficient': c, 'q': q, 'seed': seed}
        with torch.no_grad():
            c.add_(1.)
        events = [action('retained prehook before own version failure', lambda: q.backward(seed), named)]
        old = q.grad
        named = dict(named, held_gradient_after_failure=old)
        events.append(action('retry still reaches retained prehook', lambda: q.backward(seed), named))
        return {'events': events, 'after': capture(named)}

    def own_released_failure():
        x = tensor([1., -2.], True)
        q = x * x
        q.retain_grad()
        seed = tensor([2., -.5])
        named = {'x': x, 'q': q, 'seed': seed}
        events = [action('successful default releases Saved tensors', lambda: q.backward(seed), named)]
        old = q.grad
        named = dict(named, held_gradient_before_retry=old)
        events.append(action('retained prehook before own released-Saved failure', lambda: q.backward(seed), named))
        return {'events': events, 'after': capture(named)}

    def lifetimes():
        variants = []
        for materialize in (False, True):
            x = tensor([1., -2.], True)
            q = x * 2.
            q.retain_grad()
            seed = tensor([2., -1.])
            held = None
            if materialize:
                q.backward(seed, retain_graph=True)
                held = q.grad
            child = q + 1.
            weak_q = WeakStorage(q)
            weak_g = None if held is None else WeakStorage(held)
            py_q = weakref.ref(q)
            before = capture({'x': x, 'q': q, 'child': child, 'seed': seed})
            del q
            gc.collect()
            dropped = {'q_storage_expired': weak_q.expired(), 'python_q_wrapper_alive': py_q() is not None,
                       'held_gradient_storage_expired': None if weak_g is None else weak_g.expired()}
            _, outcome = invoke('backward after public nonleaf is gone', lambda: child.backward(seed))
            named = {'x': x, 'child': child, 'seed': seed}
            if held is not None:
                named['held_old_gradient'] = held
            after = capture(named)
            variants.append({'variant': 'materialized-and-held' if materialize else 'never-materialized',
                             'before_drop': before, 'after_drop': dropped, 'backward': outcome,
                             'after_backward': after, 'q_storage_expired_after_backward': weak_q.expired()})
            weak_q.close()
            if weak_g is not None:
                weak_g.close()

        x = tensor([1., -2.], True)
        q = x * 2.
        q.retain_grad()
        q.backward(tensor([2., -1.]), retain_graph=True)
        held = q.grad
        opt = optimizer('SGD', q)
        weak_x, weak_q, weak_g, weak_xg = (WeakStorage(x), WeakStorage(q), WeakStorage(held), WeakStorage(x.grad))
        py_x, py_q = weakref.ref(x), weakref.ref(q)
        before = capture({'x': x, 'q': q, 'held_gradient': held}, opt)
        del x, q
        gc.collect()
        def weak_snapshot():
            return {'x_storage_expired': weak_x.expired(), 'q_storage_expired': weak_q.expired(),
                    'x_gradient_storage_expired': weak_xg.expired(), 'held_gradient_storage_expired': weak_g.expired(),
                    'python_x_wrapper_alive': py_x() is not None, 'python_q_wrapper_alive': py_q() is not None}
        after_public_drop = weak_snapshot()
        _, stepped = invoke('optimizer reaches privately retained nonleaf', opt.step)
        private = capture({'private_parameter': opt.param_groups[0]['params'][0], 'held_gradient': held}, opt)
        del opt
        gc.collect()
        after_optimizer_drop = weak_snapshot()
        held_info = info(held)
        del held
        gc.collect()
        after_gradient_drop = weak_snapshot()
        variants.append({'variant': 'optimizer-is-the-only-strong-parameter-root',
                         'before_drop': before, 'after_public_drop': after_public_drop,
                         'private_step': stepped, 'after_private_step': private,
                         'after_optimizer_drop': after_optimizer_drop, 'held_gradient_after_optimizer_drop': held_info,
                         'after_final_gradient_drop': after_gradient_drop})
        for w in (weak_x, weak_q, weak_g, weak_xg):
            w.close()
        return {'variants': variants}

    functions = (admission, replacement, strided, empty, optimized,
                 own_version_failure, own_released_failure, lifetimes)
    rows = []
    with torch.autograd.set_multithreading_enabled(False):
        for (name, variants), fn in zip(PLAN, functions, strict=True):
            row = {'case': name, 'planned_variants': variants}
            try:
                observation = fn()
                assert len(observation.get('variants', [observation])) == variants
                row.update(status='OBSERVED', observation=observation)
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
    before = {'probe': digest(__file__), 'reference': digest(args.reference), 'lock': digest(args.lock)}
    base = {
        'schema': 'heptatorch.pinned-retained-nonleaf-observations.v1',
        'probe_sha256': before['probe'], 'reference_support_sha256': SUPPORT_SHA,
        'lock_sha256': LOCK_SHA, 'reference_source_commit': SOURCE_PIN,
        'original_optimizer_24_reference_sha256': ORIGINAL_OPTIMIZER_REFERENCE_SHA,
        'expected_cases': 8, 'expected_variants': 12,
        'case_plan': [{'case': name, 'variants': count} for name, count in PLAN],
        'native_executed': False, 'native_values_compared': 0,
        'identity_policy': 'Opaque Tensor and Storage identities and equality relations are recorded. Equal Version counts do not assert shared Version identity. Python wrapper Weak observations are separate from Storage Weak lifetime.',
        'comparison_policy': 'Record exact logical binary32 words, None, layouts, Version counters, opaque identities and entire optimizer state. Future Adam numerical comparison must retain the existing declared tolerance; metadata/identity/Version/None stay exact.',
        'observer_policy': 'No custom autograd hooks are registered. Explicit held seeds, gradients and the Saved old-gradient guard are fixed inputs. Snapshots retain primitive data only.',
        'scope': 'Eight CPU binary32 first-order families needed to complete the retained-nonleaf Optimizer boundary and its ownership/exception behavior.',
        'not_covered': ['custom hooks', 'create_graph/higher-order derivatives', 'sparse gradients',
                        'non-CPU or non-float32 tensors', 'inplace graph rebase/detach_ of the nonleaf parameter',
                        'native equality is not established by this reference-only observation'],
        'full_rewrite_complete': False, 'pytorch_api_denominator': 2590,
    }
    if args.prepare_only:
        emit(args.output, dict(base, status='PREPARED_NOT_RUN', cases=0, torch_executed=False))
        print(json.dumps({'status': 'PREPARED_NOT_RUN', 'expected_cases': 8,
                          'expected_variants': 12, 'probe_sha256': before['probe']}))
        return 0
    provenance_path = args.output.with_name(args.output.stem + '-provenance.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    old_argv, old_cwd = sys.argv, Path.cwd()
    try:
        os.chdir(args.repo)
        sys.argv = [str(args.reference), '/dev/null', str(provenance_path), str(args.lock)]
        try:
            support = runpy.run_path(str(args.reference), run_name='retained_nonleaf_probe_support')
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
    unchanged = before == {'probe': digest(__file__), 'reference': digest(args.reference), 'lock': digest(args.lock)}
    assert unchanged
    result = dict(base, status='OBSERVED' if errors == 0 else 'PROBE_ERROR',
                  cases=len(rows), probe_errors=errors, rows=rows, oracle_provenance=provenance,
                  torch_executed=True, source_files_unchanged=unchanged)
    emit(args.output, result)
    print(json.dumps({'status': result['status'], 'cases': len(rows), 'probe_errors': errors,
                      'probe_sha256': before['probe']}))
    return 0 if errors == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
