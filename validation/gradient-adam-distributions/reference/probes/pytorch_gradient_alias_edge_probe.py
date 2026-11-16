"""Observe actual gradient alias mutations and edge accumulation order.

Six fixed families and twelve variants supplement the immutable 16/25 probe.
The actual Version changes, identities, lifetimes and words are observations.
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
SOURCE_FRAMEWORK_SHA = '8706090d731cf705e68a2e474693d10448b1fa00371ca8ddd5db36163500bd69'
PIN = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
FAMILIES = (
    ('view-gradient-seed-and-gradient-mutations', ('seed-first', 'gradient-first')),
    ('equal-Version-values-with-distinct-Version-identities', ('clone', 'data-alias')),
    ('repeated-cat-edges-after-large-leaf-contribution', ('right-created-last', 'cat-created-last')),
    ('repeated-cat-edges-through-shared-mul-fanout', ('right-created-last', 'cat-created-last')),
    ('old-gradient-alias-across-clear-and-recreation', ('view', 'clone')),
    ('transpose-incoming-layout-and-Version-sharing', ('matching', 'nonmatching')),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def execute(torch):
    from torch.multiprocessing.reductions import StorageWeakRef

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
                'version': int(tensor._version),
                'binary32_words': [int(word) & 0xffffffff for word in words],
                'tensor_identity': str(tensor._cdata),
                'storage_identity': str(tensor.untyped_storage()._cdata)}

    def snapshot(p, held):
        return {'parameter': info(p),
                'gradient': None if p.grad is None else info(p.grad),
                'held_Tensors': {name: info(tensor) for name, tensor in held.items()}}

    def event(events, name, p, held, action):
        row = {'action': name, 'before': snapshot(p, held)}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                action()
                row['outcome'] = 'SUCCEEDED'
            except Exception as error:
                row.update(outcome='RAISED', exception_type=type(error).__name__, message=str(error))
        row['warnings'] = [{'category': item.category.__name__, 'message': str(item.message)} for item in caught]
        row['after'] = snapshot(p, held)
        events.append(row)

    def mutate(held, name, amount):
        with torch.no_grad():
            held[name].add_(amount)

    def shared_view(variant):
        p = make([1., -2.], True)
        output = p.view(1, 2)
        held = {'seed': make([[-0., 2.]])}
        events = []
        event(events, 'view backward with held seed', p, held, lambda: output.backward(held['seed']))
        held['gradient'] = p.grad
        held['alias'] = held['gradient'].detach()
        weak = StorageWeakRef(held['seed'].untyped_storage())
        order = ('seed', 'gradient') if variant == 'seed-first' else ('gradient', 'seed')
        for name in order:
            event(events, name+' add_(0.5)', p, held, lambda name=name: mutate(held, name, .5))
        event(events, 'clear parameter.grad binding', p, held, lambda: setattr(p, 'grad', None))
        event(events, 'held alias add_(1) after clear', p, held, lambda: mutate(held, 'alias', 1.))
        before = weak.expired()
        del held['seed']
        del held['gradient']
        gc.collect()
        alias_only = {'storage_expired': weak.expired(), 'snapshot': snapshot(p, held)}
        del held['alias']
        gc.collect()
        return {'events': events, 'storage_expired_before_drops': before,
                'alias_only': alias_only, 'storage_expired_after_final_alias_drop': weak.expired()}

    def equal_value_counterexample(variant):
        p = make([1., -2.], True)
        output = p.view(1, 2)
        held = {'seed': make([[1., 2.]])}
        output.backward(held['seed'])
        held['gradient'] = p.grad
        if variant == 'clone':
            held['other'] = held['gradient'].clone()
        else:
            held['other'] = held['gradient'].data
        weak = StorageWeakRef(held['gradient'].untyped_storage())
        initial = snapshot(p, held)
        events = []
        event(events, 'other add_(1)', p, held, lambda: mutate(held, 'other', 1.))
        event(events, 'gradient add_(2)', p, held, lambda: mutate(held, 'gradient', 2.))
        event(events, 'clear binding while other and gradient are held', p, held, lambda: setattr(p, 'grad', None))
        event(events, 'other add_(0.5) after clear', p, held, lambda: mutate(held, 'other', .5))
        del held['seed']
        del held['gradient']
        gc.collect()
        other_only = {'storage_expired': weak.expired(), 'snapshot': snapshot(p, held)}
        del held['other']
        gc.collect()
        return {'initial': initial, 'events': events, 'other_only': other_only,
                'storage_expired_after_all_drops': weak.expired(),
                'note': 'Equal initial Version values are observed only. Separate mutations test whether those values advance together.'}

    def edge_order(variant, shared):
        p = make([1.], True)
        held = {}
        if shared:
            target = p*1.
            held['shared'] = target
        else:
            target = p
        def cat_branch():
            return torch.cat((target, target)).sum()
        def mul_branch():
            return (target*16777216.).sum()
        if variant == 'right-created-last':
            left = cat_branch()
            right = mul_branch()
            created = ['cat', 'mul']
        else:
            right = mul_branch()
            left = cat_branch()
            created = ['mul', 'cat']
        held.update(cat_branch=left, mul_branch=right)
        output = left+right
        events = []
        event(events, 'backward with repeated cat edges and 2^24 contribution',
              p, held, lambda: output.backward())
        return {'created_branch_order': created, 'shared_intermediate': shared,
                'events': events, 'parameter_gradient': None if p.grad is None else info(p.grad),
                'prediction_scope': 'No predicted word or scheduling order is substituted for the observed result.'}

    def clear_recreate(variant):
        p = make([1., -2.], True)
        output = p.view(1, 2) if variant == 'view' else p.clone()
        held = {'seed': make([[1., 2.]]) if variant == 'view' else make([1., 2.])}
        events = []
        event(events, 'first backward', p, held, lambda: output.backward(held['seed']))
        held['old_gradient'] = p.grad
        held['old_alias'] = held['old_gradient'].detach()
        weak = StorageWeakRef(held['old_gradient'].untyped_storage())
        event(events, 'clear first gradient binding', p, held, lambda: setattr(p, 'grad', None))
        event(events, 'seed add_(0.5) with old alias held', p, held, lambda: mutate(held, 'seed', .5))
        event(events, 'second backward recreates gradient binding', p, held, lambda: output.backward(held['seed']))
        held['new_gradient'] = p.grad
        event(events, 'old alias add_(1) after recreation', p, held, lambda: mutate(held, 'old_alias', 1.))
        del held['old_gradient']
        del held['old_alias']
        del held['seed']
        gc.collect()
        new_only = {'storage_expired': weak.expired(), 'snapshot': snapshot(p, held)}
        event(events, 'clear second binding while new gradient held', p, held, lambda: setattr(p, 'grad', None))
        del held['new_gradient']
        gc.collect()
        return {'events': events, 'new_gradient_only': new_only,
                'storage_expired_after_all_gradient_and_seed_drops': weak.expired()}

    def transpose_alias(variant):
        p = make([[1., -2.], [.5, 2.]], True)
        output = p.t()
        held = {'seed_base': make([[1., 2.], [3., 4.]])}
        held['seed'] = held['seed_base'].t() if variant == 'matching' else held['seed_base']
        events = []
        event(events, 'transpose backward', p, held, lambda: output.backward(held['seed']))
        held['gradient'] = p.grad
        weak = StorageWeakRef(held['gradient'].untyped_storage())
        event(events, 'seed add_(0.5)', p, held, lambda: mutate(held, 'seed', .5))
        event(events, 'gradient add_(1)', p, held, lambda: mutate(held, 'gradient', 1.))
        event(events, 'clear binding and retain gradient externally', p, held, lambda: setattr(p, 'grad', None))
        del held['seed']
        del held['seed_base']
        gc.collect()
        gradient_only = {'storage_expired': weak.expired(), 'snapshot': snapshot(p, held)}
        del held['gradient']
        gc.collect()
        return {'events': events, 'gradient_only': gradient_only,
                'storage_expired_after_final_gradient_drop': weak.expired()}

    rows = []
    functions = (shared_view, equal_value_counterexample,
                 lambda variant: edge_order(variant, False),
                 lambda variant: edge_order(variant, True),
                 clear_recreate, transpose_alias)
    with torch.autograd.set_multithreading_enabled(False):
        for (name, variants), function in zip(FAMILIES, functions, strict=True):
            row = {'case': name}
            try:
                row.update(status='OBSERVED', observations=[
                    {'variant': variant, 'observation': function(variant)} for variant in variants])
            except Exception as error:
                row.update(status='PROBE_ERROR', exception_type=type(error).__name__,
                           message=str(error), traceback=traceback.format_exc())
            rows.append(row)
            gc.collect()
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
    base = {'schema': 'heptatorch.pinned-torch-gradient-alias-edge-observations.v1',
            'probe_sha256': digest(__file__), 'source_framework_sha256': SOURCE_FRAMEWORK_SHA,
            'source_framework_marker_meaning': 'Metadata/provenance framework origin only. This probe has its own source hash and is not byte-identical to the framework source.',
            'reference_support_sha256': SUPPORT_SHA, 'lock_sha256': LOCK_SHA,
            'reference_source_commit': PIN, 'expected_families': 6, 'expected_variants': 12,
            'case_plan': [{'family': name, 'variants': list(variants)} for name, variants in FAMILIES],
            'native_executed': False, 'native_values_compared': 0,
            'future_comparison_policy': 'Exact binary32 words, layouts, actual Version changes under separate seed and gradient mutations, opaque Tensor/Storage relations, StorageWeakRef lifetime and per-edge accumulation. No tolerance, Version reset or expected result substitution. Native execution count remains zero.',
            'identity_policy': 'Opaque Tensor and Storage identifiers are meaningful only within this process. Equal current Version values alone do not establish shared Version identity.',
            'not_covered': ['arbitrary higher-order graph closure', 'sparse/complex/custom grad_dtype', 'Python wrapper weakref implementation', 'arbitrary allocator and use-count variations outside these explicit controls', 'native equality is not established by this reference-only probe'],
            'full_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    assert len(FAMILIES) == 6 and sum(len(variants) for _, variants in FAMILIES) == 12
    if args.prepare_only:
        emit(args.output, dict(base, status='PREPARED_NOT_RUN', cases=0, variants=0))
        print(json.dumps({'status': 'PREPARED_NOT_RUN', 'expected_families': 6, 'expected_variants': 12, 'probe_sha256': base['probe_sha256']}))
        return 0
    provenance_path = args.output.with_name(args.output.stem+'-provenance.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    old_argv, old_cwd = sys.argv, Path.cwd()
    try:
        os.chdir(args.repo)
        sys.argv = [str(args.reference), '/dev/null', str(provenance_path), str(args.lock)]
        try:
            support = runpy.run_path(str(args.reference), run_name='gradient_alias_edge_probe_support')
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
