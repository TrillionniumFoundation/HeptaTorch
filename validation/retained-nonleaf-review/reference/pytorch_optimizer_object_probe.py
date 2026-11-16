"""Observe actual pinned Torch Optimizer object ownership and lazy state.

This file has no native executable input and performs no target implementation.
All state, arithmetic, exceptions and ownership observations come from real Torch.
The fixed recipes are independent of the HIA implementation and its results.
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
PIN = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
SGD_VARIANTS = {
    'momentum': {'momentum': .5},
    'dampening': {'momentum': .5, 'dampening': .25},
    'nesterov': {'momentum': .5, 'nesterov': True},
    'weight-decay': {'momentum': .5, 'weight_decay': .25},
}
CASE_NAMES = (
    'SGD-plain-None-zero-None',
    'SGD-momentum-None-first-zero-None',
) + tuple('SGD-'+name+'-first-and-two-updates' for name in SGD_VARIANTS) + (
    'SGD-empty-defined-gradient',
    'SGD-strided-parameter-state-layout',
    'SGD-ordered-groups-and-skipped-parameter',
    'SGD-add-group-after-state-created',
    'SGD-across-group-duplicate-rejections',
    'SGD-within-group-duplicate-first-and-second-step',
    'SGD-empty-top-level-and-empty-group',
    'SGD-constructor-and-group-override-boundaries',
    'SGD-parameter-retained-after-public-and-graph-drop',
    'SGD-state-alias-and-saved-survive-optimizer-drop',
    'SGD-state-mutation-invalidates-saved',
    'SGD-two-optimizers-one-parameter',
    'SGD-distinct-parameter-identities-sharing-storage',
    'external-gradient-alias-survives-zero-grad-True',
    'Adam-lazy-step-and-state-None-skip',
    'AdamW-lazy-step-and-state-None-skip',
    'Adam-within-group-duplicate-first-and-second-step',
    'Adam-real-state-Tensor-lifetime-and-external-mutation',
)
assert len(CASE_NAMES) == len(set(CASE_NAMES)) == 24


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
                'dtype': str(tensor.dtype), 'requires_grad': bool(tensor.requires_grad),
                'is_leaf': bool(tensor.is_leaf), 'version': int(tensor._version),
                'binary32_words': [int(word) & 0xffffffff for word in words],
                'values': tensor.detach().reshape(-1).tolist(),
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

    def make(values):
        return torch.tensor(values, dtype=torch.float32, requires_grad=True)

    def backward(parameter, values):
        parameter.backward(torch.tensor(values, dtype=torch.float32).reshape(parameter.shape))

    def sgd(params, **overrides):
        config = dict(lr=.125, momentum=0., dampening=0., weight_decay=0.,
                      nesterov=False, maximize=False, foreach=False, fused=False)
        config.update(overrides)
        return torch.optim.SGD(params, **config)

    def adam(params, decoupled=False):
        constructor = torch.optim.AdamW if decoupled else torch.optim.Adam
        return constructor(params, lr=.125, betas=(.5, .75), eps=.01,
                           weight_decay=.25, foreach=False, fused=False)

    def field(value):
        if isinstance(value, torch.Tensor):
            return {'tensor': info(value)}
        if isinstance(value, (str, bool, int, float, type(None))):
            return value
        if isinstance(value, (tuple, list)):
            return [field(item) for item in value]
        raise TypeError(('unsupported observed state/config value', type(value).__name__))

    def snapshot(params, optimizer=None):
        result = {'parameters': {name: info(tensor) for name, tensor in params.items()},
                  'gradients': {name: None if tensor.grad is None else info(tensor.grad)
                                for name, tensor in params.items()}}
        if optimizer is not None:
            result['optimizer_state'] = {}
            for name, tensor in params.items():
                existing = optimizer.state.get(tensor, {})
                result['optimizer_state'][name] = {'entry_exists': tensor in optimizer.state,
                    'fields': {key: field(value) for key, value in existing.items()}}
            result['state_entries'] = len(optimizer.state)
            result['groups'] = [
                {'parameters': [str(tensor._cdata) for tensor in group['params']],
                 'config': {key: field(value) for key, value in group.items() if key != 'params'}}
                for group in optimizer.param_groups]
        return result

    def invoke(label, function):
        row = {'action': label}
        result = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                result = function()
                row['outcome'] = 'SUCCEEDED'
                if result is not None:
                    row['return_type'] = type(result).__name__
            except Exception as error:
                row.update(outcome='RAISED', exception_type=type(error).__name__, message=str(error))
        row['warnings'] = [{'category': item.category.__name__, 'message': str(item.message)} for item in caught]
        return result, row

    def action(label, function, params, optimizer=None):
        before = snapshot(params, optimizer)
        _, result = invoke(label, function)
        result.update(before=before, after=snapshot(params, optimizer))
        return result

    def lazy_sgd(momentum):
        p = make([1., -2.]); params = {'p': p}
        opt = sgd([p], momentum=momentum, weight_decay=.25)
        actions = [action('initial None skips all state creation', opt.step, params, opt),
            action('define an exact zero gradient', lambda: backward(p, [0., 0.]), params, opt),
            action('first present-zero step', opt.step, params, opt),
            action('clear True after first participation', lambda: opt.zero_grad(set_to_none=True), params, opt),
            action('existing state unchanged while None', opt.step, params, opt),
            action('second present gradient', lambda: backward(p, [2., -1.]), params, opt),
            action('resume existing state', opt.step, params, opt)]
        return {'algorithm': 'SGD', 'config': {'momentum': momentum, 'weight_decay': .25}, 'actions': actions}

    def momentum_variant(name):
        p = make([1., -2., .5]); params = {'p': p}
        opt = sgd([p], **SGD_VARIANTS[name]); actions = []
        for index, values in enumerate(([2., -1., .5], [-1., 2., -.5], [0., 0., 0.])):
            if index:
                actions.append(action('clear True before next input', lambda: opt.zero_grad(set_to_none=True), params, opt))
            actions.append(action('autograd seed '+str(index+1), lambda v=values: backward(p, v), params, opt))
            actions.append(action('present-gradient step '+str(index+1), opt.step, params, opt))
        return {'algorithm': 'SGD', 'config': SGD_VARIANTS[name], 'actions': actions}

    def empty_sgd():
        p = make([]); params = {'p': p}; opt = sgd([p], momentum=.5, weight_decay=.25)
        actions = [action('None empty parameter step', opt.step, params, opt),
            action('define empty gradient', lambda: backward(p, []), params, opt),
            action('first empty participation', opt.step, params, opt),
            action('second empty participation', opt.step, params, opt),
            action('clear empty gradient True', lambda: opt.zero_grad(set_to_none=True), params, opt),
            action('empty None skips existing state', opt.step, params, opt)]
        return {'actions': actions}

    def strided_sgd():
        p = torch.tensor([[1., -2., 3.], [.5, 2., -1.]], dtype=torch.float32).t().requires_grad_()
        params = {'p': p}; opt = sgd([p], momentum=.5, dampening=.25)
        actions = [action('noncontiguous leaf seed', lambda: backward(p, [1., 2., 3., -1., -2., -3.]), params, opt),
                   action('first clone preserves admissible layout', opt.step, params, opt),
                   action('second update preserves state identity and layout', opt.step, params, opt)]
        return {'actions': actions}

    def ordered_groups():
        p, q = make([1., -2.]), make([.5, 3.]); params = {'p': p, 'q': q}
        opt = sgd([{'params': [p], 'lr': .125, 'momentum': .5, 'weight_decay': .25},
                   {'params': [q], 'lr': .25, 'momentum': .75, 'dampening': .25}])
        actions = [action('only first parameter acquires grad', lambda: backward(p, [2., -1.]), params, opt),
                   action('first group active and second absent', opt.step, params, opt),
                   action('clear both True', lambda: opt.zero_grad(set_to_none=True), params, opt),
                   action('only second parameter acquires grad', lambda: backward(q, [-1., 2.]), params, opt),
                   action('second group first state creation', opt.step, params, opt),
                   action('clear both again', lambda: opt.zero_grad(set_to_none=True), params, opt),
                   action('first leaf next seed', lambda: backward(p, [1., 1.]), params, opt),
                   action('second leaf next seed', lambda: backward(q, [.5, -.5]), params, opt),
                   action('both groups update in order', opt.step, params, opt)]
        return {'actions': actions}

    def late_group():
        p, q = make([1., -2.]), make([.5, 3.]); params = {'p': p, 'q': q}
        opt = sgd([p], momentum=.5)
        actions = [action('seed original parameter', lambda: backward(p, [2., -1.]), params, opt),
                   action('original state created', opt.step, params, opt),
                   action('append distinct group without state creation',
                          lambda: opt.add_param_group({'params': [q], 'lr': .25, 'momentum': .75}), params, opt),
                   action('clear all groups True', lambda: opt.zero_grad(set_to_none=True), params, opt),
                   action('seed newly added parameter', lambda: backward(q, [-1., 2.]), params, opt),
                   action('new state first step without resetting original state', opt.step, params, opt)]
        return {'actions': actions}

    def across_groups():
        p, q = make([1., -2.]), make([.5, 3.]); params = {'p': p, 'q': q}
        _, construction = invoke('constructor repeats Tensor across groups',
                                 lambda: sgd([{'params': [p]}, {'params': [p]}], momentum=.5))
        opt = sgd([p], momentum=.5)
        actions = [action('existing member cannot be appended again',
                          lambda: opt.add_param_group({'params': [p]}), params, opt),
                   action('mixed new and existing member rejected',
                          lambda: opt.add_param_group({'params': [q, p]}), params, opt),
                   action('distinct group succeeds after failures',
                          lambda: opt.add_param_group({'params': [q]}), params, opt)]
        return {'construction': construction, 'actions': actions}

    def duplicate_sgd():
        p = make([1., -2.]); params = {'p': p}
        opt, construction = invoke('duplicate occurrences inside one group',
                                   lambda: sgd([p, p], momentum=.5, dampening=.25, weight_decay=.25))
        assert opt is not None
        actions = [action('one shared parameter gradient', lambda: backward(p, [2., -1.]), params, opt),
                   action('first duplicate step plans two absent buffers', opt.step, params, opt),
                   action('second duplicate step shares existing buffer', opt.step, params, opt)]
        return {'construction': construction, 'actions': actions}

    def empty_groups():
        _, empty_top = invoke('empty top-level parameter iterable', lambda: sgd([]))
        opt, empty_group = invoke('one explicitly empty group', lambda: sgd([{'params': []}], momentum=.5))
        assert opt is not None
        actions = [action('empty group step', opt.step, {}, opt),
                   action('append another empty group', lambda: opt.add_param_group({'params': []}), {}, opt),
                   action('empty groups clear True', lambda: opt.zero_grad(set_to_none=True), {}, opt)]
        return {'empty_top_level': empty_top, 'explicit_empty_group': empty_group, 'actions': actions}

    def construction_boundaries():
        p = make([1., -2.]); outcomes = []
        for label, kwargs in (
            ('negative default learning rate', {'lr': -.125}),
            ('negative default momentum', {'momentum': -.5}),
            ('negative default weight decay', {'weight_decay': -.25}),
            ('Nesterov with zero momentum', {'nesterov': True}),
            ('Nesterov with nonzero dampening', {'nesterov': True, 'momentum': .5, 'dampening': .25})):
            _, outcome = invoke(label, lambda options=kwargs: sgd([p], **options))
            outcomes.append(outcome)
        q = (p*2.)
        _, no_retain = invoke('nonleaf without retain_grad', lambda: sgd([q]))
        q.retain_grad()
        retained, with_retain = invoke('nonleaf with retain_grad', lambda: sgd([q]))
        inert = torch.tensor([3.], dtype=torch.float32)
        _, nonrequiring = invoke('nonrequiring leaf is a valid member', lambda: sgd([inert]))
        opt, override = invoke('negative group override with valid constructor defaults',
                               lambda: sgd([{'params': [p], 'lr': -.125}]))
        actions = []
        if opt is not None:
            actions = [action('gradient before overridden step', lambda: backward(p, [2., -1.]), {'p': p}, opt),
                       action('step with observed group override', opt.step, {'p': p}, opt)]
        return {'default_constructor_boundaries': outcomes, 'nonleaf': no_retain,
                'retained_nonleaf': with_retain, 'nonrequiring_leaf': nonrequiring,
                'group_override': override, 'actions': actions}

    def parameter_lifetime():
        p = make([1., -2.]); opt = sgd([p], momentum=.5)
        loss = (p*p).sum(); loss.backward()
        tensor_weak, storage_weak = weakref.ref(p), WeakStorage(p)
        before = snapshot({'p': p}, opt)
        try:
            del p, loss
            after_external_drop = {'tensor_alive': tensor_weak() is not None,
                                   'storage_expired': storage_weak.expired()}
            retained = opt.param_groups[0]['params'][0]
            events = [action('step through private optimizer parameter reference', opt.step, {'p': retained}, opt)]
            after = snapshot({'p': retained}, opt)
            del retained, opt
            gc.collect()
            after_optimizer_drop = {'tensor_alive': tensor_weak() is not None,
                                    'storage_expired': storage_weak.expired()}
            return {'before': before, 'after_public_and_graph_drop': after_external_drop,
                    'actions': events, 'after': after, 'after_optimizer_drop': after_optimizer_drop}
        finally:
            storage_weak.close()

    def state_alias_lifetime():
        p = make([1., -2.]); opt = sgd([p], momentum=.5)
        backward(p, [2., -1.]); opt.step()
        buffer = opt.state[p]['momentum_buffer']; alias = buffer.detach()
        q = make([3., 4.]); root = (q*buffer).sum()
        weak = WeakStorage(buffer)
        before = {'state': info(buffer), 'alias': info(alias), 'q': info(q)}
        try:
            del buffer, opt, p
            after_optimizer_drop = weak.expired()
            call = action('Saved state survives owner drop', root.backward, {'q': q})
            with_alias_after_backward = weak.expired()
            before_write = info(alias)
            alias.add_(.5)
            after_write = info(alias)
            del alias
            after_alias_drop = weak.expired()
            return {'before': before, 'storage_expired_after_optimizer_drop': after_optimizer_drop,
                    'backward': call, 'storage_expired_with_alias_after_backward': with_alias_after_backward,
                    'alias_before_write': before_write, 'alias_after_write': after_write,
                    'storage_expired_after_last_alias_drop': after_alias_drop}
        finally:
            weak.close()

    def state_saved_version():
        p, q = make([1., -2.]), make([3., 4.]); opt = sgd([p], momentum=.5)
        backward(p, [2., -1.]); opt.step()
        buffer = opt.state[p]['momentum_buffer']; root = (q*buffer).sum()
        before = info(buffer)
        update = action('optimizer mutates the actual Saved state object', opt.step, {'p': p}, opt)
        after = info(buffer)
        failure = action('Saved checks shared state Version', root.backward, {'q': q})
        return {'state_before_update': before, 'optimizer_update': update,
                'state_after_update': after, 'backward': failure}

    def two_optimizers():
        p = make([1., -2.]); params = {'p': p}
        first, second = sgd([p], momentum=.5), sgd([p], lr=.25, momentum=.75, weight_decay=.25)
        backward(p, [2., -1.]); actions = []
        for label, optimizer in (('first optimizer creates its state', first),
                                  ('second optimizer creates independent state', second),
                                  ('first optimizer updates its original state', first)):
            before = {'first': snapshot(params, first), 'second': snapshot(params, second)}
            _, outcome = invoke(label, optimizer.step)
            actions.append(dict(outcome, before=before,
                                after={'first': snapshot(params, first), 'second': snapshot(params, second)}))
        a, b = first.state[p]['momentum_buffer'], second.state[p]['momentum_buffer']
        return {'actions': actions, 'same_state_Tensor': a is b,
                'same_state_Storage': a.untyped_storage()._cdata == b.untyped_storage()._cdata}

    def aliased_parameters():
        p = make([1., -2.]); q = p.detach().requires_grad_(True)
        params = {'p': p, 'q': q}; opt = sgd([p, q], momentum=.5, weight_decay=.25)
        actions = [action('first Tensor gradient', lambda: backward(p, [2., -1.]), params, opt),
                   action('distinct aliased Tensor gradient', lambda: backward(q, [-1., 2.]), params, opt),
                   action('distinct state entries despite shared parameter Storage', opt.step, params, opt)]
        a, b = opt.state[p]['momentum_buffer'], opt.state[q]['momentum_buffer']
        return {'same_parameter_Tensor': p is q,
                'same_parameter_Storage': p.untyped_storage()._cdata == q.untyped_storage()._cdata,
                'same_state_Tensor': a is b,
                'same_state_Storage': a.untyped_storage()._cdata == b.untyped_storage()._cdata,
                'actions': actions}

    def external_grad():
        p = make([1., -2.]); opt = sgd([p]); backward(p, [-0., 2.])
        held = p.grad; weak = WeakStorage(held)
        before = {'parameter': info(p), 'gradient': info(held)}
        try:
            opt.zero_grad(set_to_none=True)
            after_true = {'parameter': info(p), 'grad_is_None': p.grad is None,
                          'held': info(held), 'old_storage_expired': weak.expired()}
            held.add_(.5)
            held_mutated = info(held)
            backward(p, [3., 4.])
            after_rebuild = {'new_gradient': info(p.grad), 'held': info(held),
                             'same_grad_Tensor': p.grad is held,
                             'same_grad_Storage': p.grad.untyped_storage()._cdata == held.untyped_storage()._cdata}
            opt.zero_grad(set_to_none=False)
            after_false = {'current_gradient': info(p.grad), 'held': info(held)}
            del held
            after_external_drop = weak.expired()
            return {'before': before, 'after_true': after_true,
                    'old_external_gradient_after_mutation': held_mutated,
                    'after_rebuild': after_rebuild, 'after_false': after_false,
                    'old_storage_expired_after_external_drop': after_external_drop}
        finally:
            weak.close()

    def lazy_adam(decoupled):
        p = make([1., -2.]); params = {'p': p}; opt = adam([p], decoupled)
        actions = [action('initial None does not create step or moments', opt.step, params, opt),
                   action('defined zero gradient', lambda: backward(p, [0., 0.]), params, opt),
                   action('first actual step initializes real state Tensors', opt.step, params, opt),
                   action('clear True preserves state', lambda: opt.zero_grad(set_to_none=True), params, opt),
                   action('None preserves step Tensor value Version and identity', opt.step, params, opt),
                   action('resume gradient', lambda: backward(p, [2., -1.]), params, opt),
                   action('resume increments existing state', opt.step, params, opt),
                   action('False zeros same gradient', lambda: opt.zero_grad(set_to_none=False), params, opt),
                   action('defined zero still participates', opt.step, params, opt)]
        return {'algorithm': 'AdamW' if decoupled else 'Adam', 'actions': actions}

    def duplicate_adam():
        p = make([1., -2.]); params = {'p': p}
        opt, construction = invoke('Adam duplicate occurrences inside one group', lambda: adam([p, p]))
        assert opt is not None
        actions = [action('shared gradient', lambda: backward(p, [2., -1.]), params, opt),
                   action('first group plans one shared step and moment identity', opt.step, params, opt),
                   action('second duplicate step keeps the same state objects', opt.step, params, opt)]
        return {'construction': construction, 'actions': actions}

    def adam_state_lifetime():
        p = make([1., -2.]); opt = adam([p]); backward(p, [2., -1.]); opt.step()
        step, avg = opt.state[p]['step'], opt.state[p]['exp_avg']
        step_weak, avg_weak = WeakStorage(step), WeakStorage(avg)
        square_weak = WeakStorage(opt.state[p]['exp_avg_sq'])
        q = make([3., 4.]); root = (q*avg).sum()
        before = {'step': info(step), 'exp_avg': info(avg), 'optimizer': snapshot({'p': p}, opt)}
        try:
            del opt, p
            after_optimizer_drop = {'step_expired': step_weak.expired(),
                                    'exp_avg_expired': avg_weak.expired(),
                                    'exp_avg_sq_expired': square_weak.expired()}
            step.add_(.5); avg.add_(.5)
            mutated = {'step': info(step), 'exp_avg': info(avg)}
            backward_result = action('Saved moment sees external mutation after optimizer drop', root.backward, {'q': q})
            del root, step, avg
            after_all_drop = {'step_expired': step_weak.expired(), 'exp_avg_expired': avg_weak.expired()}
            return {'before': before, 'after_optimizer_drop': after_optimizer_drop,
                    'after_external_state_mutation': mutated, 'backward': backward_result,
                    'after_all_external_and_graph_drop': after_all_drop}
        finally:
            step_weak.close(); avg_weak.close(); square_weak.close()

    recipes = [lambda: lazy_sgd(0.), lambda: lazy_sgd(.5)]
    recipes += [lambda key=key: momentum_variant(key) for key in SGD_VARIANTS]
    recipes += [empty_sgd, strided_sgd, ordered_groups, late_group, across_groups,
                duplicate_sgd, empty_groups, construction_boundaries, parameter_lifetime,
                state_alias_lifetime, state_saved_version, two_optimizers, aliased_parameters,
                external_grad, lambda: lazy_adam(False), lambda: lazy_adam(True),
                duplicate_adam, adam_state_lifetime]
    assert len(recipes) == len(CASE_NAMES)
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
    lock = json.loads(args.lock.read_bytes())
    assert PIN in json.dumps(lock)
    base = {'schema': 'heptatorch.pinned-torch-optimizer-object-observations.v1',
            'probe_sha256': digest(__file__), 'reference_support_sha256': SUPPORT_SHA,
            'lock_sha256': LOCK_SHA, 'reference_source_commit': PIN,
            'expected_cases': len(CASE_NAMES), 'case_plan': list(CASE_NAMES),
            'native_executed': False, 'native_values_compared': 0,
            'reference_encoding': 'Unsigned binary32 words are authoritative. Opaque Tensor/Storage identity strings are only compared as relationships within one trajectory, never to native pointer values.',
            'predeclared_future_native_policy': {
                'presence_identity_relationships_Versions_state_creation_and_step_words': 'exact',
                'SGD_dyadic_recipe_values_and_gradient_words': 'exact binary32 words',
                'Adam_parameter_and_moment_values': {'rtol': 4e-5, 'atol': 5e-6,
                    'origin': 'Unchanged frozen gradient-presence replay policy; raw words remain recorded.'},
                'target_comparison_executed_by_this_probe': False},
            'engine_mode': 'CPU binary32, first-order, autograd multithreading disabled, foreach=False, fused=False.',
            'scope': 'Lazy SGD/Adam state, parameter-group ordering and rejection boundaries, duplicate occurrences, actual parameter/state/gradient Tensor ownership and shared Version observations.',
            'not_covered': ['native allocation-failure qualification', 'all optimizer algorithms',
                            'closures/hooks/schedulers', 'state_dict/load_state_dict',
                            'fused/foreach/capturable/differentiable optimizers',
                            'general Python mutable object compatibility'],
            'full_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    if args.prepare_only:
        emit(args.output, dict(base, status='PREPARED_NOT_RUN', cases=0))
        print(json.dumps({'status': 'PREPARED_NOT_RUN', 'expected_cases': len(CASE_NAMES),
                          'probe_sha256': base['probe_sha256']}))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = args.output.with_name(args.output.stem+'-provenance.json')
    previous_argv, previous_cwd = sys.argv, Path.cwd()
    try:
        os.chdir(args.repo)
        sys.argv = [str(args.reference), '/dev/null', str(provenance_path), str(args.lock)]
        try:
            support = runpy.run_path(str(args.reference), run_name='optimizer_object_probe_support')
        except (SystemExit, ModuleNotFoundError) as error:
            observed = json.loads(provenance_path.read_bytes()) if provenance_path.exists() else None
            emit(args.output, dict(base, status='NOT_RUN', cases=0,
                reason='Pinned reference did not admit this local environment',
                initialization_exception=type(error).__name__, initialization_message=str(error),
                support_observation=observed))
            return 2
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
    provenance = support['PROVENANCE']
    assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11
    assert all(value is True for value in provenance['checks'].values())
    rows = execute(support['torch'])
    assert tuple(row['case'] for row in rows) == CASE_NAMES
    errors = sum(row['status'] != 'OBSERVED' for row in rows)
    report = dict(base, status='PROBE_ERROR' if errors else 'OBSERVED', cases=len(rows),
                  probe_errors=errors, rows=rows, oracle_provenance=provenance,
                  source_files_unchanged={'probe': digest(__file__) == base['probe_sha256'],
                                          'support': digest(args.reference) == SUPPORT_SHA,
                                          'lock': digest(args.lock) == LOCK_SHA})
    assert all(report['source_files_unchanged'].values())
    emit(args.output, report)
    print(json.dumps({'status': report['status'], 'cases': len(rows), 'probe_errors': errors,
                      'probe_sha256': base['probe_sha256']}))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
