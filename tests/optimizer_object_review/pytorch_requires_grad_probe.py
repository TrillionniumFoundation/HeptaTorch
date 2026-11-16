"""Observe leaf requires_grad_ transitions and actual autograd/Optimizer edges.

Only the pinned real Torch reference is executed. No native target or inferred
expected values enter these six fixed recipes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import runpy
import sys
import traceback

SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
PIN = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
CASE_NAMES = (
    'leaf-flags-preserve-identity-data-Version-and-None',
    'False-preserves-existing-gradient-and-optimizer-consumes-it',
    'existing-graph-AccumulateGrad-observes-current-leaf-flag',
    'False-at-forward-True-later-does-not-create-an-edge',
    'False-at-backward-does-not-suppress-Saved-version-error',
    'distinct-alias-leaves-have-independent-flags-and-gradients',
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
                'dtype': str(tensor.dtype), 'requires_grad': bool(tensor.requires_grad),
                'is_leaf': bool(tensor.is_leaf), 'version': int(tensor._version),
                'binary32_words': [int(word) & 0xffffffff for word in words],
                'values': tensor.detach().reshape(-1).tolist(),
                'tensor_identity': str(tensor._cdata),
                'storage_identity': str(tensor.untyped_storage()._cdata),
                'storage_offset_bytes': tensor.storage_offset()*tensor.element_size(),
                'storage_bytes': tensor.untyped_storage().nbytes()}

    def snapshot(params, optimizer=None):
        result = {'parameters': {name: info(tensor) for name, tensor in params.items()},
                  'gradients': {name: None if tensor.grad is None else info(tensor.grad)
                                for name, tensor in params.items()}}
        if optimizer is not None:
            result['optimizer_state'] = {
                name: {'entry_exists': tensor in optimizer.state,
                       'fields': {field: {'tensor': info(value)} if isinstance(value, torch.Tensor) else value
                                  for field, value in optimizer.state.get(tensor, {}).items()}}
                for name, tensor in params.items()}
        return result

    def action(label, function, params, optimizer=None):
        result = {'action': label, 'before': snapshot(params, optimizer)}
        try:
            returned = function()
            result['outcome'] = 'SUCCEEDED'
            if isinstance(returned, torch.Tensor):
                result['returned_Tensor_identity'] = str(returned._cdata)
        except Exception as error:
            result.update(outcome='RAISED', exception_type=type(error).__name__, message=str(error))
        result['after'] = snapshot(params, optimizer)
        return result

    def make(values, requires=True):
        return torch.tensor(values, dtype=torch.float32, requires_grad=requires)

    def seed(values):
        return torch.tensor(values, dtype=torch.float32)

    def flags_only():
        p = make([1., -2.], False); params = {'p': p}; actions = []
        for flag in (False, True, True, False, True):
            actions.append(action('requires_grad_('+str(flag)+')',
                                  lambda flag=flag: p.requires_grad_(flag), params))
        return {'actions': actions}

    def existing_gradient():
        p = make([1., -2.]); params = {'p': p}
        opt = torch.optim.SGD([p], lr=.125, momentum=.5, weight_decay=.25,
                              foreach=False, fused=False)
        actions = [action('initial defined gradient', lambda: p.backward(seed([2., -1.])), params, opt),
                   action('False retains old gradient', lambda: p.requires_grad_(False), params, opt),
                   action('Optimizer consumes present gradient despite requires False', opt.step, params, opt),
                   action('False zero_grad zeros the existing gradient', lambda: opt.zero_grad(set_to_none=False), params, opt),
                   action('present zero still updates while requires False', opt.step, params, opt),
                   action('True zero_grad clears while requires False', lambda: opt.zero_grad(set_to_none=True), params, opt),
                   action('None skips while requires False', opt.step, params, opt),
                   action('True preserves identity without defining a gradient', lambda: p.requires_grad_(True), params, opt),
                   action('fresh first negative-zero gradient', lambda: p.backward(seed([-0., -0.])), params, opt)]
        return {'actions': actions}

    def existing_graph():
        p = make([1., -2.]); coefficient = make(2., False)
        root = (p*coefficient).sum(); params = {'p': p, 'coefficient': coefficient}
        actions = [action('False after forward', lambda: p.requires_grad_(False), params),
                   action('retained backward with leaf flag False', lambda: root.backward(retain_graph=True), params),
                   action('True before next retained backward', lambda: p.requires_grad_(True), params),
                   action('retained backward with leaf flag True', lambda: root.backward(retain_graph=True), params),
                   action('False while a gradient already exists', lambda: p.requires_grad_(False), params),
                   action('default backward releases Saved while accumulation is disabled', root.backward, params),
                   action('True after graph release', lambda: p.requires_grad_(True), params),
                   action('spent graph still errors', root.backward, params)]
        return {'actions': actions}

    def absent_forward_edge():
        p, q = make([1., -2.], False), make([3., 4.])
        root = (p*q).sum(); params = {'p': p, 'q': q}
        actions = [action('enable formerly nonrequiring forward input', lambda: p.requires_grad_(True), params),
                   action('backward uses the edges captured at forward', root.backward, params)]
        return {'actions': actions}

    def saved_version_still_checked():
        p = make([1., -2.]); coefficient = make(2., False)
        root = (p*coefficient).sum(); params = {'p': p, 'coefficient': coefficient}
        actions = [action('disable leaf accumulation after forward', lambda: p.requires_grad_(False), params),
                   action('mutate saved scalar coefficient', lambda: coefficient.add_(1.), params),
                   action('Saved version validation still executes', root.backward, params)]
        return {'actions': actions}

    def alias_flags():
        p = make([1., -2.]); q = p.detach(); params = {'p': p, 'q': q}
        root = (p*p).sum()
        actions = [action('enable distinct detached leaf', lambda: q.requires_grad_(True), params),
                   action('accumulate only detached leaf', lambda: q.backward(seed([3., 4.])), params),
                   action('disable original leaf independently', lambda: p.requires_grad_(False), params),
                   action('disable alias but retain its gradient', lambda: q.requires_grad_(False), params),
                   action('reenable original', lambda: p.requires_grad_(True), params),
                   action('reenable alias', lambda: q.requires_grad_(True), params),
                   action('original retained graph accumulates only original leaf', root.backward, params)]
        return {'same_parameter_Storage': p.untyped_storage()._cdata == q.untyped_storage()._cdata,
                'same_parameter_Tensor': p is q, 'actions': actions}

    recipes = (flags_only, existing_gradient, existing_graph, absent_forward_edge,
               saved_version_still_checked, alias_flags)
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
    base = {'schema': 'heptatorch.pinned-torch-leaf-requires-grad-observations.v1',
            'probe_sha256': digest(__file__), 'reference_support_sha256': SUPPORT_SHA,
            'lock_sha256': LOCK_SHA, 'reference_source_commit': PIN,
            'expected_cases': len(CASE_NAMES), 'case_plan': list(CASE_NAMES),
            'native_executed': False, 'native_values_compared': 0,
            'reference_encoding': 'Unsigned binary32 words and opaque identity relationships; never compare host addresses to native handles.',
            'predeclared_future_native_policy': 'Exact gradient and dyadic SGD words, leaf flags, parameter/available state Versions, None presence and existing/absent graph-edge behavior. Public grad Tensor identity/Version remains open until exposed.',
            'scope': 'Leaf requires_grad_ changes, retained actual gradients, SGD participation, graph edges and Saved validation. Not nonleaf retain_grad or differentiable optimizer support.',
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
            support = runpy.run_path(str(args.reference), run_name='leaf_requires_grad_probe_support')
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
    unchanged = {'probe': digest(__file__) == base['probe_sha256'],
                 'support': digest(args.reference) == SUPPORT_SHA, 'lock': digest(args.lock) == LOCK_SHA}
    assert all(unchanged.values())
    report = dict(base, status='PROBE_ERROR' if errors else 'OBSERVED', cases=len(rows),
                  probe_errors=errors, rows=rows, oracle_provenance=provenance,
                  source_files_unchanged=unchanged)
    emit(args.output, report)
    print(json.dumps({'status': report['status'], 'cases': len(rows), 'probe_errors': errors,
                      'probe_sha256': base['probe_sha256']}))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
