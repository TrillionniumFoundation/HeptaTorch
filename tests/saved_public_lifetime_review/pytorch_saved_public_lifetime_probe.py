#!/usr/bin/env python3
"""Observe held Saved Tensors across backward and graph destruction on fixed Torch.

No native program runs. No numerical result, acceptance, Version, lifetime or
identity expectation is supplied by this probe. All outcomes are actual Torch.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import runpy
import struct
import sys
import traceback
import warnings

SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
PARENT_PROBE_SHA = 'cf3017df475fcb64b4eca771476b8433588e8d2f379b3cc3cb05a9addbd3d165'
SOURCE_REF = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
CHECKS = {'python_matches', 'platform_matches', 'version_backend_matches',
    'git_matches_oracle_build_pin', 'distribution_version_matches', 'direct_url_matches',
    'direct_url_archive_sha256_matches', 'pip_report_url_matches',
    'pip_report_archive_sha256_matches', 'pip_report_version_matches', 'pip_report_requested'}
SHA = lambda b: hashlib.sha256(b).hexdigest()


def plan():
    rows = []
    for release in ('backward_default', 'drop_graph'):
        for operation in ('mul', 'div'):
            for side in ('left', 'right'):
                rows.append(dict(name=f'mixed-{operation}-{side}-{release}',
                    family='mixed', operation=operation, float_side=side,
                    layout='dense', release=release))
        rows.append(dict(name=f'integer-view-mul-{release}', family='mixed',
            operation='mul', float_side='left', layout='integer_holes', release=release))
        for operation in ('mul', 'div'):
            rows.append(dict(name=f'float-self-{operation}-{release}', family='self',
                operation=operation, float_side='both', layout='dense', release=release))
    assert len(rows) == 14 and len({x['name'] for x in rows}) == 14
    return [dict(row, index=i) for i, row in enumerate(rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'lock', 'repo', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    a = parser.parse_args()
    assert Path.cwd().resolve() == a.repo.resolve()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    assert not a.output.exists()
    inputs = {p.resolve(): p.read_bytes() for p in (Path(__file__), a.reference, a.lock)}
    assert SHA(inputs[a.reference.resolve()]) == SUPPORT_SHA
    assert SHA(inputs[a.lock.resolve()]) == LOCK_SHA
    common = dict(schema='heptatorch.saved-public-lifetime-observations.v1',
        probe_sha256=SHA(inputs[Path(__file__).resolve()]), parent_probe_sha256=PARENT_PROBE_SHA,
        primary_source_ref=SOURCE_REF, expected_cases=14, case_plan=plan(),
        native_executed=False, native_values_compared=0, native_qualified=False,
        full_pytorch_rewrite_complete=False, pytorch_api_denominator=2590,
        comparison_policy='Observation only; no result or lifetime expectations are injected.',
        identity_scope='TensorImpl/StorageImpl string identities, pairwise live relations, explicit Storage weak reference.',
        alias_scope='Self operations obtain both Saved slots and drop both held aliases together; no independent native-handle multiplicity is assumed.',
        autograd_scope='CPU F32/int64 or F32 self mul/div; default grad mode and create_graph=False.')
    if a.prepare_only:
        report = dict(common, status='PREPARED_NOT_RUN', torch_executed=False, cases=0, rows=[])
        a.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(status=report['status'], cases=0, probe_sha256=common['probe_sha256'])))
        return 0
    old_argv = sys.argv
    try:
        sys.argv = [str(a.reference), str(a.repo/'reference-only-no-native-execution'),
                    str(a.output.with_suffix('.provenance.json')), str(a.lock)]
        context = runpy.run_path(str(a.reference), run_name='saved_public_lifetime_provenance_only')
    finally:
        sys.argv = old_argv
    t, provenance = context['torch'], context['PROVENANCE']
    assert provenance['status'] == 'PASS' and set(provenance['checks']) == CHECKS
    assert all(v is True for v in provenance['checks'].values())
    states = {}

    def rng():
        raw = bytes(t.get_rng_state().tolist())
        assert len(raw) == 5056
        key = SHA(raw)
        states.setdefault(key, {'bytes': len(raw), 'hex': raw.hex()})
        return key

    def tensor(value):
        if value is None:
            return None
        raw = bytes(value.detach().contiguous().reshape(-1).view(t.uint8).tolist())
        fn = value.grad_fn
        return {'tensor_identity': str(value._cdata), 'storage_identity': str(value.untyped_storage()._cdata),
                'dtype': str(value.dtype), 'element_size': value.element_size(), 'numel': value.numel(),
                'shape': list(value.shape), 'strides': list(value.stride()),
                'storage_bytes': value.untyped_storage().nbytes(),
                'storage_offset_elements': value.storage_offset(),
                'storage_offset_bytes': value.storage_offset() * value.element_size(),
                'version': value._version, 'requires_grad': value.requires_grad, 'is_leaf': value.is_leaf,
                'is_view': value._is_view(),
                'base_tensor_identity': None if value._base is None else str(value._base._cdata),
                'grad_fn': None if fn is None else type(fn).__name__,
                'next_functions': [] if fn is None else [None if edge[0] is None else type(edge[0]).__name__ for edge in fn.next_functions],
                'logical_bytes_hex': raw.hex()}

    def capture(operation):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                result = operation()
                event = {'accepted': True, 'outcome': 'SUCCEEDED', 'error': None}
            except Exception as error:
                result = None
                event = {'accepted': False, 'outcome': 'RAISED',
                         'error': {'type': type(error).__name__, 'message': str(error)}}
        event['warnings'] = [{'type': type(w.message).__name__, 'message': str(w.message)} for w in caught]
        return result, event

    def saved(value, live):
        fn = value.grad_fn
        if fn is None:
            return {'grad_fn': None, 'attributes': {}}
        attributes = {}
        for name in sorted(n for n in dir(fn) if n.startswith('_saved_')):
            item, event = capture(lambda name=name: getattr(fn, name))
            if event['accepted']:
                if isinstance(item, t.Tensor):
                    event['tensor'] = tensor(item)
                    event['live_Tensor_matches'] = [key for key, obj in live.items() if obj._cdata == item._cdata]
                    event['live_Storage_matches'] = [key for key, obj in live.items()
                                                     if obj.untyped_storage()._cdata == item.untyped_storage()._cdata]
                elif item is None or isinstance(item, (bool, int, str)):
                    event['value'] = item
                elif isinstance(item, float):
                    event['double_bits'] = struct.unpack('<Q', struct.pack('<d', item))[0]
                else:
                    event['diagnostic_type'] = type(item).__name__
                    event['diagnostic_repr'] = repr(item)
            attributes[name] = event
            del item
        return {'grad_fn': type(fn).__name__, 'attributes': attributes}

    def snapshot(live, weak):
        result = dict(tensors={name: tensor(value) for name, value in live.items()},
            gradients={}, saved={name: saved(live[name], live) for name in ('y',) if name in live},
            rng_state=rng(), original_Storage_expired=bool(t.UntypedStorage._expired(weak)))
        for name in ('x', 'held', 'held_self', 'held_other'):
            if name in live:
                value, event = capture(lambda name=name: live[name].grad)
                event['tensor'] = tensor(value)
                result['gradients'][name] = event
                del value
        return result

    def observe(spec):
        live = {'x': t.tensor([1.5, -2., 4., .5, 8., -4.], dtype=t.float32).reshape(2, 3).detach().requires_grad_(True)}
        original_name = 'x' if spec['family'] == 'self' else 'i'
        if spec['family'] != 'self':
            if spec['layout'] == 'integer_holes':
                live['i'] = t.tensor(list(range(2, 26)), dtype=t.int64).as_strided((2, 3), (9, 2), 1)
            else:
                live['i'] = t.tensor([2, 3, 5, 7, 16777217, -16777219], dtype=t.int64).reshape(2, 3).detach()
        weak = live[original_name].untyped_storage()._weak_ref()
        row = dict(spec, status='RUNNING', input_before=snapshot(live, weak), events=[])
        try:
            def apply():
                if spec['family'] == 'self':
                    return t.mul(live['x'], live['x']) if spec['operation'] == 'mul' else t.div(live['x'], live['x'])
                if spec['float_side'] == 'left':
                    return t.mul(live['x'], live['i']) if spec['operation'] == 'mul' else t.div(live['x'], live['i'])
                return t.mul(live['i'], live['x']) if spec['operation'] == 'mul' else t.div(live['i'], live['x'])
            value, event = capture(apply)
            row['forward'] = event
            if not event['accepted']:
                row.update(status='OBSERVED', after_forward=snapshot(live, weak))
                return row
            live['y'] = value
            del value
            live['seed'] = t.tensor([-0., 1., -1., .5, 2., -2.], dtype=t.float32).reshape(2, 3).detach()
            row['after_forward'] = snapshot(live, weak)
            del live[original_name]
            gc.collect()
            row['events'].append(dict(event='drop_original_public', target=original_name, after=snapshot(live, weak)))
            if spec['family'] == 'self':
                requests = [('held_self', '_saved_self'), ('held_other', '_saved_other')]
            else:
                requests = [('held', '_saved_other' if spec['float_side'] == 'left' else '_saved_self')]
            obtained = []
            for name, attribute in requests:
                value, event = capture(lambda attribute=attribute: getattr(live['y'].grad_fn, attribute))
                if event['accepted'] and isinstance(value, t.Tensor):
                    live[name] = value
                    obtained.append(name)
                    event['tensor'] = tensor(value)
                del value
                row['events'].append(dict(event='hold_saved_public', name=name, attribute=attribute,
                    result=event, after=snapshot(live, weak)))
            if spec['release'] == 'backward_default':
                value, event = capture(lambda: t.autograd.backward(live['y'], live['seed'], retain_graph=False, create_graph=False))
                del value
                gc.collect()
                row['events'].append(dict(event='backward_default', result=event, after=snapshot(live, weak)))
            else:
                del live['y']
                gc.collect()
                row['events'].append(dict(event='drop_graph', after=snapshot(live, weak)))
            for name in obtained:
                del live[name]
            gc.collect()
            row['events'].append(dict(event='drop_all_held_saved', names=obtained, after=snapshot(live, weak)))
            if 'y' in live:
                del live['y']
                gc.collect()
                row['events'].append(dict(event='drop_graph_after_backward', after=snapshot(live, weak)))
            row['status'] = 'OBSERVED'
            row['state_after_all'] = rng()
            return row
        finally:
            live.clear()
            gc.collect()
            t.UntypedStorage._free_weak_ref(weak)

    rows, errors = [], []
    with t.random.fork_rng(devices=[]):
        for spec in plan():
            try:
                t.manual_seed(1729)
                rows.append(observe(spec))
            except BaseException as error:
                row = dict(spec, status='PROBE_ERROR', error=repr(error), traceback=traceback.format_exc())
                rows.append(row)
                errors.append(row)
    assert all(p.read_bytes() == raw for p, raw in inputs.items())
    report = dict(common, status='OBSERVED' if not errors else 'PROBE_ERROR', torch_executed=True,
        oracle_provenance=provenance, build_configuration=t.__config__.show(),
        cpu_capability=t.backends.cpu.get_cpu_capability(), cases=len(rows), rows=rows,
        probe_errors=errors, probe_error_count=len(errors), rng_states=states,
        source_files_unchanged=True, inputs_sha256={str(p): SHA(raw) for p, raw in inputs.items()})
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in ('status', 'cases', 'probe_sha256')}))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
