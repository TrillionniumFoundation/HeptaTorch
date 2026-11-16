"""Observe pinned Torch gradient clearing and state preservation, without a target.

The ten recipes are frozen before any native comparison. References come only
from real Torch/autograd/optim; raw binary32 words distinguish signed zeros.
No current or future native result supplies an expected value here.
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
PIN = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
KINDS = ('SGD-momentum', 'Adam', 'AdamW')
CASE_NAMES = (
    'already-None-true-and-false-are-noops',
    'clear-true-then-first-negative-zero-gradient',
    'clear-false-retains-defined-zero-gradient',
    'empty-gradient-clear-true-and-rebuild',
    'retained-graph-after-gradient-clear',
    'partial-independent-ready-leaf-then-clear',
    'partial-shared-pending-leaf-clear-is-noop',
) + tuple('existing-state-None-skip-and-resume-' + kind for kind in KINDS)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def execute(torch):
    def info(t):
        words = t.detach().contiguous().view(torch.int32).reshape(-1).tolist()
        return {'shape': list(t.shape), 'strides': list(t.stride()), 'dtype': str(t.dtype),
                'requires_grad': bool(t.requires_grad), 'version': int(t._version),
                'values': t.detach().reshape(-1).tolist(),
                'binary32_words': [int(word) & 0xffffffff for word in words],
                'tensor_identity': str(t._cdata), 'storage_identity': str(t.untyped_storage()._cdata)}

    def snapshot(params, opt=None):
        row = {'parameters': {name: info(t) for name, t in params.items()},
               'gradients': {name: None if t.grad is None else info(t.grad) for name, t in params.items()}}
        if opt is not None:
            row['optimizer_state'] = {}
            for name, t in params.items():
                state = opt.state.get(t, {})
                row['optimizer_state'][name] = {'entry_exists': t in opt.state,
                    'fields': {key: {'tensor': info(value)} if isinstance(value, torch.Tensor) else value
                               for key, value in state.items()}}
        return row

    def action(label, fn, params, opt=None):
        row = {'action': label, 'before': snapshot(params, opt)}
        try:
            fn()
            row['outcome'] = 'SUCCEEDED'
        except Exception as error:
            row.update(outcome='RAISED', exception_type=type(error).__name__, message=str(error))
        row['after'] = snapshot(params, opt)
        return row

    def optimizer(kind, params):
        if kind == 'SGD-momentum':
            config = {'lr': .125, 'momentum': .5, 'dampening': .25, 'weight_decay': .25,
                      'nesterov': False, 'foreach': False, 'fused': False}
            return torch.optim.SGD(list(params.values()), **config), config
        config = {'lr': .125, 'betas': (.5, .75), 'eps': .01, 'weight_decay': .25,
                  'amsgrad': False, 'foreach': False, 'fused': False}
        cls = torch.optim.Adam if kind == 'Adam' else torch.optim.AdamW
        return cls(list(params.values()), **config), config

    def holder(params):
        return torch.optim.SGD(list(params.values()), lr=.125, foreach=False, fused=False)

    def make(values):
        return torch.tensor(values, dtype=torch.float32, requires_grad=True)

    def seed(values):
        return torch.tensor(values, dtype=torch.float32)

    def none_noop():
        p = make([1., -2.]); params = {'p': p}; opt = holder(params)
        rows = [action('default zero_grad() on None', opt.zero_grad, params, opt),
                action('explicit zero_grad(True) on None', lambda: opt.zero_grad(set_to_none=True), params, opt),
                action('zero_grad(False) on None', lambda: opt.zero_grad(set_to_none=False), params, opt)]
        return {'actions': rows}

    def true_rebuild():
        p = make([1., -2.]); params = {'p': p}; opt = holder(params)
        rows = [action('first defined gradient', lambda: p.backward(seed([2., -3.])), params, opt),
                action('clear true removes gradient', lambda: opt.zero_grad(set_to_none=True), params, opt),
                action('second clear true remains None', lambda: opt.zero_grad(set_to_none=True), params, opt),
                action('new first gradient preserves negative zero', lambda: p.backward(seed([-0., -0.])), params, opt)]
        return {'explicit_seeds': [[2., -3.], [-0., -0.]], 'actions': rows}

    def false_retains():
        p = make([1., -2.]); params = {'p': p}; opt = holder(params)
        rows = [action('first negative-zero gradient', lambda: p.backward(seed([-0., -0.])), params, opt),
                action('clear false keeps gradient object', lambda: opt.zero_grad(set_to_none=False), params, opt),
                action('accumulate negative zero into existing positive zero', lambda: p.backward(seed([-0., -0.])), params, opt),
                action('clear true now removes object', lambda: opt.zero_grad(set_to_none=True), params, opt),
                action('new first negative-zero gradient after true', lambda: p.backward(seed([-0., -0.])), params, opt)]
        return {'explicit_seeds': [[-0., -0.]] * 3, 'actions': rows}

    def empty():
        p = make([]); params = {'p': p}; opt = holder(params)
        rows = [action('empty gradient initially None', lambda: opt.zero_grad(set_to_none=True), params, opt),
                action('backward defines empty gradient', lambda: p.backward(seed([])), params, opt),
                action('false retains defined empty gradient', lambda: opt.zero_grad(set_to_none=False), params, opt),
                action('true removes empty gradient', lambda: opt.zero_grad(set_to_none=True), params, opt),
                action('backward redefines empty gradient', lambda: p.backward(seed([])), params, opt)]
        return {'explicit_seeds': [[], []], 'actions': rows}

    def retained():
        p = make([1., -2., 3.]); params = {'p': p}; opt = holder(params)
        square = p * p; loss = square.sum()
        rows = [action('first retained backward', lambda: loss.backward(retain_graph=True), params, opt),
                action('clear true while graph remains', lambda: opt.zero_grad(set_to_none=True), params, opt),
                action('second backward creates fresh gradient and releases graph', loss.backward, params, opt),
                action('clear true after release', lambda: opt.zero_grad(set_to_none=True), params, opt),
                action('spent graph errors without defining gradient', loss.backward, params, opt)]
        return {'graph': ['p*p', 'sum'], 'actions': rows}

    def partial_independent():
        bad = make([1., 2., 3.]); good = make([2., -1., .5]); coefficient = seed(2.)
        params = {'bad': bad, 'good': good}; opt = holder(params)
        a = bad * coefficient; b = good * good; loss = (a + b).sum()
        with torch.no_grad():
            coefficient.add_(1.)
        rows = [action('failure after independent good leaf becomes ready', loss.backward, params, opt),
                action('clear true clears ready leaf and leaves pending leaf None', lambda: opt.zero_grad(set_to_none=True), params, opt),
                action('fresh first negative-zero gradient only for good leaf', lambda: good.backward(seed([-0., -0., -0.])), params, opt),
                action('clear false preserves bad None and good defined zero', lambda: opt.zero_grad(set_to_none=False), params, opt)]
        return {'graph_creation_order': ['bad*coefficient', 'good*good', 'add', 'sum'],
                'coefficient': {'initial': 2., 'mutated_to': 3., 'mutation_count': 1}, 'actions': rows}

    def partial_shared():
        p = make([1., 2., 3.]); coefficient = seed(2.); params = {'p': p}; opt = holder(params)
        bad = p * coefficient; good = p * p; loss = (bad + good).sum()
        with torch.no_grad():
            coefficient.add_(1.)
        rows = [action('retained failure leaves shared leaf pending', lambda: loss.backward(retain_graph=True), params, opt),
                action('clear true on pending None leaf', lambda: opt.zero_grad(set_to_none=True), params, opt),
                action('fresh good root can define gradient', lambda: good.sum().backward(), params, opt),
                action('clear true removes newly defined gradient', lambda: opt.zero_grad(set_to_none=True), params, opt)]
        return {'graph_creation_order': ['p*coefficient', 'p*p', 'add', 'sum'],
                'coefficient': {'initial': 2., 'mutated_to': 3., 'mutation_count': 1}, 'actions': rows}

    def state_skip(kind):
        p = make([1., -2., .5]); params = {'p': p}; opt, config = optimizer(kind, params)
        rows = [action('first defined gradient', lambda: p.backward(seed([1., -2., .5])), params, opt),
                action('first step creates state lazily', opt.step, params, opt),
                action('clear true removes gradient and preserves state', lambda: opt.zero_grad(set_to_none=True), params, opt),
                action('step with None preserves existing state and parameter', opt.step, params, opt),
                action('false on None does not create gradient', lambda: opt.zero_grad(set_to_none=False), params, opt),
                action('backward redefines negative-zero gradient', lambda: p.backward(seed([-0., -0., -0.])), params, opt),
                action('step resumes existing state', opt.step, params, opt),
                action('false preserves defined gradient and state', lambda: opt.zero_grad(set_to_none=False), params, opt),
                action('third present-gradient step reuses state', opt.step, params, opt)]
        return {'optimizer': kind, 'config': config, 'explicit_seeds': [[1., -2., .5], [-0., -0., -0.]], 'actions': rows}

    recipes = [none_noop, true_rebuild, false_retains, empty, retained, partial_independent, partial_shared]
    recipes += [lambda kind=kind: state_skip(kind) for kind in KINDS]
    rows = []
    with torch.autograd.set_multithreading_enabled(False):
        for name, recipe in zip(CASE_NAMES, recipes, strict=True):
            try:
                rows.append({'case': name, 'status': 'OBSERVED', 'observation': recipe()})
            except Exception as error:
                rows.append({'case': name, 'status': 'PROBE_ERROR', 'exception_type': type(error).__name__,
                             'message': str(error), 'traceback': traceback.format_exc()})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'lock', 'repo', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    for name in ('reference', 'lock', 'repo', 'output'):
        setattr(a, name, getattr(a, name).resolve())
    assert digest(a.reference) == SUPPORT_SHA, 'Changed reference support'
    base = {'schema': 'heptatorch.pinned-torch-grad-clear-observations.v1', 'probe_sha256': digest(__file__),
        'support_sha256': SUPPORT_SHA, 'lock_sha256': digest(a.lock), 'upstream_commit': PIN,
        'expected_cases': len(CASE_NAMES), 'case_names': list(CASE_NAMES), 'target_execution': False,
        'native_values_compared': 0, 'full_rewrite_complete': False, 'pytorch_api_denominator': 2590,
        'scope': 'Real Torch zero_grad(True/False), empty and partial-backward gradient clearing, retained graph reuse, and existing optimizer-state preservation while grad is None.',
        'reference_encoding': 'Unsigned binary32 words are authoritative. Tensor/Storage identities are opaque decimal strings; compare relationships, never native addresses.',
        'engine_mode': 'CPU binary32, first-order, multithreading disabled, foreach=False and fused=False.',
        'not_covered': ['external user-held .grad Tensor aliases', 'nonleaf retain_grad', 'native Optimizer object implementation', 'arbitrary optimizer groups and mutable state dictionaries']}
    if a.prepare_only:
        write(a.output, dict(base, status='PREPARED_NOT_RUN', cases=0))
        return 0
    a.output.parent.mkdir(parents=True, exist_ok=True)
    provenance = a.output.with_name(a.output.stem + '-provenance.json')
    saved_argv, saved_directory = sys.argv, Path.cwd()
    try:
        os.chdir(a.repo)
        sys.argv = [str(a.reference), '/dev/null', str(provenance), str(a.lock)]
        try:
            support = runpy.run_path(str(a.reference), run_name='pinned_grad_clear_support')
        except SystemExit as error:
            observed = json.loads(provenance.read_text()) if provenance.exists() else {'exit': error.code}
            write(a.output, dict(base, status='NOT_RUN', cases=0, reason='Pinned support did not admit this environment', support_observation=observed))
            return 2
    finally:
        sys.argv = saved_argv
        os.chdir(saved_directory)
    assert support['PROVENANCE']['status'] == 'PASS'
    assert len(support['PROVENANCE']['checks']) == 11 and all(support['PROVENANCE']['checks'].values())
    rows = execute(support['torch'])
    assert tuple(row['case'] for row in rows) == CASE_NAMES
    errors = sum(row['status'] != 'OBSERVED' for row in rows)
    report = dict(base, status='PROBE_ERROR' if errors else 'OBSERVED', cases=len(rows), probe_errors=errors,
        rows=rows, oracle_provenance=support['PROVENANCE'], probe_unchanged=digest(__file__) == base['probe_sha256'],
        reference_unchanged=digest(a.reference) == SUPPORT_SHA)
    write(a.output, report)
    print(json.dumps({'status': report['status'], 'cases': len(rows), 'probe_errors': errors}))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
