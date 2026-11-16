"""Observe pinned Torch incoming-gradient ownership and first materialization.

Sixteen fixed families, twenty-five variants. This is a new reference-only
probe; it does not alter the frozen public-gradient-object ten-group probe.
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
# Framework origin marker only; this new probe is not byte-identical to it.
SOURCE_FRAMEWORK_SHA = 'e69b849b3521ceae35a835f84b76eca8c472253acd2c6ec20869f7f7bc2e3e62'
PIN = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
FAMILIES = (
    ('direct-leaf-implicit-scalar', ('implicit',)),
    ('direct-leaf-held-versus-inline-seed', ('held', 'inline')),
    ('dense-transposed-leaf-seed-layout', ('matching', 'contiguous')),
    ('holey-offset-leaf-held-seed', ('holey',)),
    ('clone-backward-held-seed', ('clone',)),
    ('view-and-transpose-backward-held-seed', ('view', 'transpose')),
    ('single-mul-incoming-reference-controls', ('no-hook', 'info-only-hook', 'retained-incoming-hook')),
    ('mul-two-requiring-inputs', ('both',)),
    ('mul-same-parent-two-edges', ('same-parent',)),
    ('two-independent-mul-branches-fanout', ('independent',)),
    ('shared-intermediate-fanout-before-mul', ('shared',)),
    ('sum-backward-implicit-versus-held-scalar-seed', ('implicit', 'held')),
    ('neg-and-relu-new-gradient-results', ('neg', 'relu')),
    ('cat-repeated-parent-gradient-views', ('cat',)),
    ('gather-repeated-indices-reduction-gradient', ('gather',)),
    ('clear-recreate-and-grad-mode-context', ('enabled', 'no-grad', 'create-graph')),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def execute(torch):
    def make(values, requires=False):
        return torch.tensor(values, dtype=torch.float32, requires_grad=requires)

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

    def snapshot(params, output, seed, held):
        return {'parameters': {name: info(p) for name, p in params.items()},
                'gradients': {name: None if p.grad is None else info(p.grad) for name, p in params.items()},
                'output': info(output), 'held_seed': None if seed is None else info(seed),
                'held_Tensors': {name: info(value) for name, value in held.items()}}

    def relation(a, b):
        return {'same_Tensor': a._cdata == b._cdata,
                'same_Storage': a.untyped_storage()._cdata == b.untyped_storage()._cdata,
                'same_current_Version_value': int(a._version) == int(b._version)}

    def construct(family, variant):
        held = {}
        seed = None
        inline = False
        if family == 0:
            p = make(2., True)
            output = p
        elif family == 1:
            p = make([1., -2.], True)
            output = p
            inline = variant == 'inline'
            if not inline:
                seed = make([-0., 2.])
        elif family == 2:
            base = make([[1., -2., 3.], [.5, 2., -1.]])
            p = base.t().requires_grad_()
            held['parameter_base'] = base
            output = p
            if variant == 'matching':
                seed_base = make([[1., 3., -2.], [2., -1., -3.]])
                seed = seed_base.t()
                held['seed_base'] = seed_base
            else:
                seed = make([[1., 2.], [3., -1.], [-2., -3.]])
        elif family == 3:
            base = make([99., 1., 99., -2.])
            p = base.as_strided((2,), (2,), 1).requires_grad_()
            output = p
            seed = make([2., -1.])
            held['parameter_base'] = base
        else:
            p = make([1., -2.], True)
            if family == 4:
                output = p.clone()
                seed = make([-0., 2.])
            elif family == 5:
                if variant == 'view':
                    output = p.view(1, 2)
                    seed = make([[-0., 2.]])
                else:
                    p = make([[1., -2.], [.5, 2.]], True)
                    output = p.t()
                    seed = make([[1., 2.], [3., 4.]])
            elif family == 6:
                coefficient = make([2., 3.])
                output = (p*coefficient).sum()
                held['coefficient'] = coefficient
            elif family == 7:
                q = make([2., 3.], True)
                return {'p': p, 'q': q}, (p*q).sum(), None, held, False
            elif family == 8:
                output = (p*p).sum()
            elif family == 9:
                left = p*2.
                right = p*3.
                output = (left+right).sum()
                held.update(left=left, right=right)
            elif family == 10:
                shared = p*2.
                output = (shared+shared).sum()
                held['shared'] = shared
            elif family == 11:
                output = p.sum()
                if variant == 'held':
                    seed = make(2.)
            elif family == 12:
                output = -p if variant == 'neg' else torch.relu(p)
                seed = make([2., -1.])
            elif family == 13:
                output = torch.cat((p, p), dim=0)
                seed = make([1., 2., 3., 4.])
            elif family == 14:
                p = make([1., -2., 3.], True)
                index = torch.tensor([2, 0, 2], dtype=torch.int64)
                output = torch.gather(p, 0, index)
                seed = make([1., 2., 3.])
            else:
                assert family == 15
                output = (p*p).sum()
        return {'p': p}, output, seed, held, inline

    def observe(family, variant):
        params, output, seed, held, inline = construct(family, variant)
        incoming = []
        hook_observations = []
        handle = None
        if family == 6 and variant != 'no-hook':
            def hook(gradient):
                hook_observations.append(info(gradient))
                if variant == 'retained-incoming-hook':
                    incoming.append(gradient)
                return None
            handle = params['p'].register_hook(hook)
        row = {'variant': variant, 'initial': snapshot(params, output, seed, held),
               'inline_seed_recipe': [-0., 2.] if inline else None,
               'incoming_reference_control': variant if family == 6 else 'no hook',
               'ambient_grad_context': variant if family == 15 else 'enabled',
               'create_graph': family == 15 and variant == 'create-graph',
               'first_retain_graph': family == 15}
        events = []

        def backward_event(label, create_graph, retain_graph):
            event = {'action': label, 'before': snapshot(params, output, seed, held)}
            context = torch.no_grad() if family == 15 and variant == 'no-grad' else torch.enable_grad()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    with context:
                        event['ambient_grad_enabled_at_call'] = bool(torch.is_grad_enabled())
                        if inline:
                            output.backward(make([-0., 2.]), retain_graph=retain_graph, create_graph=create_graph)
                        else:
                            output.backward(seed, retain_graph=retain_graph, create_graph=create_graph)
                    event['outcome'] = 'SUCCEEDED'
                except Exception as error:
                    event.update(outcome='RAISED', exception_type=type(error).__name__, message=str(error))
            event['warnings'] = [{'category': warning.category.__name__, 'message': str(warning.message)} for warning in caught]
            event['after'] = snapshot(params, output, seed, held)
            events.append(event)

        try:
            backward_event('first backward', row['create_graph'], row['first_retain_graph'])
            row['hook_observations'] = hook_observations
            row['retained_incoming_after'] = [info(value) for value in incoming]
            relationships = {}
            for name, p in params.items():
                if p.grad is None:
                    relationships[name] = None
                else:
                    relationships[name] = {
                        'versus_held_seed': None if seed is None else relation(p.grad, seed),
                        'versus_retained_incoming': [relation(p.grad, value) for value in incoming],
                        'versus_held_Tensors': {key: relation(p.grad, value) for key, value in held.items()}}
            row['first_gradient_relationships'] = relationships
            if family == 15:
                old = params['p'].grad
                row['before_clear'] = snapshot(params, output, seed, held)
                params['p'].grad = None
                if old is not None:
                    held['old_gradient'] = old
                row['after_clear'] = snapshot(params, output, seed, held)
                backward_event('first-order backward after binding clear', False, False)
                new = params['p'].grad
                row['new_vs_old_gradient'] = None if new is None or old is None else relation(new, old)
            row['events'] = events
            return row
        finally:
            if handle is not None:
                handle.remove()
            # Break a create_graph reference cycle without modifying any snapshot
            # or Version. The observed clear/recreate operation is above.
            for p in params.values():
                p.grad = None
            incoming.clear()
            held.clear()
            gc.collect()

    rows = []
    with torch.autograd.set_multithreading_enabled(False):
        for family, (name, variants) in enumerate(FAMILIES):
            row = {'case': name}
            try:
                row.update(status='OBSERVED', observations=[observe(family, variant) for variant in variants])
            except Exception as error:
                row.update(status='PROBE_ERROR', exception_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
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
    base = {'schema': 'heptatorch.pinned-torch-gradient-materialization-observations.v1',
            'probe_sha256': digest(__file__), 'source_framework_sha256': SOURCE_FRAMEWORK_SHA,
            'source_framework_marker_meaning': 'Metadata/provenance framework origin only. This probe has its own source hash and is not byte-identical to the framework source.',
            'reference_support_sha256': SUPPORT_SHA, 'lock_sha256': LOCK_SHA,
            'reference_source_commit': PIN, 'expected_families': 16, 'expected_variants': 25,
            'case_plan': [{'family': name, 'variants': list(variants)} for name, variants in FAMILIES],
            'native_executed': False, 'native_values_compared': 0,
            'future_comparison_policy': 'Exact binary32 words for these fixed dyadic first-order recipes, metadata, None, Versions, opaque identity relations and actual incoming ownership controls. No tolerance or Version reset. create_graph and hooks are reference observations, not native API qualification.',
            'identity_policy': 'Opaque Tensor and Storage identifiers are meaningful only within this process. Equal current Version values alone do not establish shared Version identity.',
            'not_covered': ['arbitrary higher-order graph closure', 'sparse/complex/custom grad_dtype', 'Python wrapper weakref implementation', 'arbitrary allocator and use-count variations outside these explicit controls'],
            'full_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    assert len(FAMILIES) == 16 and sum(len(variants) for _, variants in FAMILIES) == 25
    if args.prepare_only:
        emit(args.output, dict(base, status='PREPARED_NOT_RUN', cases=0, variants=0))
        print(json.dumps({'status': 'PREPARED_NOT_RUN', 'expected_families': 16, 'expected_variants': 25, 'probe_sha256': base['probe_sha256']}))
        return 0
    provenance_path = args.output.with_name(args.output.stem+'-provenance.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    old_argv, old_cwd = sys.argv, Path.cwd()
    try:
        os.chdir(args.repo)
        sys.argv = [str(args.reference), '/dev/null', str(provenance_path), str(args.lock)]
        try:
            support = runpy.run_path(str(args.reference), run_name='gradient_materialization_probe_support')
        except (SystemExit, ModuleNotFoundError) as error:
            observed = json.loads(provenance_path.read_bytes()) if provenance_path.exists() else None
            emit(args.output, dict(base, status='NOT_RUN', cases=0, variants=0,
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
    variants = sum(len(row.get('observations', [])) for row in rows)
    report = dict(base, status='PROBE_ERROR' if errors else 'OBSERVED', cases=len(rows), variants=variants,
        probe_errors=errors, rows=rows, oracle_provenance=provenance,
        source_files_unchanged={'probe': digest(__file__) == base['probe_sha256'],
            'support': digest(args.reference) == SUPPORT_SHA, 'lock': digest(args.lock) == LOCK_SHA})
    assert all(report['source_files_unchanged'].values())
    emit(args.output, report)
    print(json.dumps({'status': report['status'], 'cases': len(rows), 'variants': variants,
                      'probe_errors': errors, 'probe_sha256': base['probe_sha256']}))
    return int(errors != 0)


if __name__ == '__main__':
    raise SystemExit(main())
