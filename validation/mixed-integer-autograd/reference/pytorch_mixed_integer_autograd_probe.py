#!/usr/bin/env python3
"""Observe mixed float32/int64 autograd on the fixed real CPU Torch build.

Reference only. Every acceptance, value, layout, saved object, version, release
and partial-gradient outcome comes from Torch. No native program is executed.
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
FRAMEWORK_SHA = '16105b790db5ff63145aac0bb03afc66003952ae5f048b4b393a885302d4ec56'
SOURCE_REF = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
SOURCE_PINS = {
    'tools/autograd/derivatives.yaml': '4a4d1ae1ddea605f57594de991da2c92e89ae11d6aa6edf348107589b548082c',
    'tools/autograd/gen_variable_type.py': '66bb855965a3829800523189eb137b15b3470d75fbffbb48b286cf7e37e78982',
    'torch/csrc/autograd/FunctionsManual.cpp': 'a8603d6309822a3f93fa4fcb5ada05e2f0f7e29a1b96db64fe6f59ce0c730558',
    'torch/csrc/autograd/saved_variable.cpp': '3051d8fc2ac1a4c2715f0b567ab18d6d46b429edead6dc688aaa35efe65f04dd',
    'aten/src/ATen/TensorIterator.cpp': 'a71a7ec9ba0532bf1380aa8ebc92975e4acb31414cd4b8eeeba650afe9e4490c',
    'aten/src/ATen/native/TensorConversions.cpp': '81b7019d5e7a31ba1f72d8d50a635f1dd637b827f75000fa85dfaf34bb5b9c3f',
}
CHECKS = {
    'python_matches', 'platform_matches', 'version_backend_matches',
    'git_matches_oracle_build_pin', 'distribution_version_matches',
    'direct_url_matches', 'direct_url_archive_sha256_matches',
    'pip_report_url_matches', 'pip_report_archive_sha256_matches',
    'pip_report_version_matches', 'pip_report_requested',
}
SHA = lambda b: hashlib.sha256(b).hexdigest()


def plan():
    rows = []
    for family in ('baseline', 'integer_mutation'):
        for operation in ('add', 'sub', 'mul', 'div'):
            for side in ('left', 'right'):
                rows.append({'name': f'{family}-{operation}-float-{side}', 'family': family,
                             'operation': operation, 'float_side': side, 'layout': 'dense'})
    for family in ('float_mutation', 'broadcast'):
        for operation in ('mul', 'div'):
            for side in ('left', 'right'):
                rows.append({'name': f'{family}-{operation}-float-{side}', 'family': family,
                             'operation': operation, 'float_side': side,
                             'layout': 'broadcast' if family == 'broadcast' else 'dense'})
    rows.extend([
        {'name': 'layout-integer-holes-mul', 'family': 'layout', 'operation': 'mul', 'float_side': 'left', 'layout': 'integer_holes'},
        {'name': 'layout-transpose-float-div', 'family': 'layout', 'operation': 'div', 'float_side': 'right', 'layout': 'float_transpose'},
        {'name': 'empty-mul-float-left', 'family': 'empty', 'operation': 'mul', 'float_side': 'left', 'layout': 'empty'},
        {'name': 'empty-div-float-right', 'family': 'empty', 'operation': 'div', 'float_side': 'right', 'layout': 'empty'},
        {'name': 'integer-lifetime-add', 'family': 'lifetime', 'operation': 'add', 'float_side': 'left', 'layout': 'dense'},
        {'name': 'integer-lifetime-mul', 'family': 'lifetime', 'operation': 'mul', 'float_side': 'left', 'layout': 'dense'},
        {'name': 'integer-lifetime-div', 'family': 'lifetime', 'operation': 'div', 'float_side': 'right', 'layout': 'dense'},
        {'name': 'integer-lifetime-two-mul-roots', 'family': 'shared_lifetime', 'operation': 'mul', 'float_side': 'left', 'layout': 'dense'},
        {'name': 'data-mutation-mul-float-left', 'family': 'data_mutation', 'operation': 'mul', 'float_side': 'left', 'layout': 'dense'},
        {'name': 'data-mutation-div-float-right', 'family': 'data_mutation', 'operation': 'div', 'float_side': 'right', 'layout': 'dense'},
        {'name': 'partial-bad-created-first', 'family': 'partial', 'operation': 'mul', 'float_side': 'left', 'layout': 'dense', 'creation_order': ['bad', 'good']},
        {'name': 'partial-good-created-first', 'family': 'partial', 'operation': 'mul', 'float_side': 'left', 'layout': 'dense', 'creation_order': ['good', 'bad']},
    ])
    assert len(rows) == 36 and len({r['name'] for r in rows}) == 36
    return [dict(row, index=i) for i, row in enumerate(rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'lock', 'repo', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    a = parser.parse_args()
    assert Path.cwd().resolve() == a.repo.resolve()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    assert not a.output.exists()
    inputs = {p.resolve(): p.read_bytes() for p in (Path(__file__), a.reference, a.lock)}
    assert SHA(inputs[a.reference.resolve()]) == SUPPORT_SHA
    assert SHA(inputs[a.lock.resolve()]) == LOCK_SHA
    common = {
        'schema': 'heptatorch.mixed-integer-autograd-observations.v1',
        'probe_sha256': SHA(inputs[Path(__file__).resolve()]),
        'source_framework_sha256': FRAMEWORK_SHA,
        'source_framework_meaning': 'Provenance/identity observation framework ancestry, not a claim that the two source files are equal.',
        'primary_source_ref': SOURCE_REF, 'primary_source_pins': SOURCE_PINS,
        'expected_cases': 36, 'case_plan': plan(),
        'native_executed': False, 'native_values_compared': 0, 'native_qualified': False,
        'full_pytorch_rewrite_complete': False, 'pytorch_api_denominator': 2590,
        'comparison_policy': 'OBSERVATION ONLY. No expected values, statuses, version counters, saved identities or release outcomes are inserted.',
        'identity_scope': 'TensorImpl and StorageImpl strings plus pairwise live relations. Storage weak references are explicit. No Python-wrapper weakref claim.',
        'autograd_scope': 'Default grad mode, create_graph=False, float32 differentiable operand and int64 nondifferentiable operand; add/sub alpha=1.',
        'held_seed_scope': 'Each explicit seed is held by a named local across backward and observation; this is part of the fixed recipe.',
    }
    if a.prepare_only:
        report = {**common, 'status': 'PREPARED_NOT_RUN', 'torch_executed': False, 'cases': 0, 'rows': []}
    else:
        old_argv = sys.argv
        try:
            sys.argv = [str(a.reference), str(a.repo / 'reference-only-no-native-execution'),
                        str(a.output.with_suffix('.provenance.json')), str(a.lock)]
            context = runpy.run_path(str(a.reference), run_name='mixed_integer_autograd_provenance_only')
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

        def snapshot(live, roots, weak=None):
            tensors = {name: tensor(value) for name, value in live.items()}
            gradients = {}
            for name in ('x', 'q'):
                if name in live:
                    value, event = capture(lambda name=name: live[name].grad)
                    event['tensor'] = tensor(value)
                    gradients[name] = event
                    del value
            result = {'tensors': tensors, 'gradients': gradients,
                      'saved': {name: saved(live[name], live) for name in roots if name in live},
                      'rng_state': rng()}
            if weak is not None:
                result['original_integer_Storage_expired'] = bool(t.UntypedStorage._expired(weak))
            return result

        def inputs_for(spec):
            floating = [1.5, -2., 4., .5, 8., -4.]
            integers = [2, 3, 5, 7, 16777217, -16777219]
            layout = spec['layout']
            if layout == 'broadcast':
                x = t.tensor([[1.5], [-2.]], dtype=t.float32)
                i = t.tensor([[2, 3, 16777217]], dtype=t.int64)
            elif layout == 'integer_holes':
                x = t.tensor(floating, dtype=t.float32).reshape(2, 3)
                base = t.tensor(list(range(2, 26)), dtype=t.int64)
                i = base.as_strided((2, 3), (9, 2), 1)
            elif layout == 'float_transpose':
                x = t.tensor(floating, dtype=t.float32).reshape(2, 3).transpose(0, 1)
                i = t.tensor(integers, dtype=t.int64).reshape(3, 2).detach()
            elif layout == 'empty':
                x = t.empty((0,), dtype=t.float32).as_strided((2, 0, 3), (17, 101, 2), 7)
                i = t.empty((0,), dtype=t.int64).as_strided((2, 0, 3), (17, 101, 2), 7)
            else:
                x = t.tensor(floating, dtype=t.float32).reshape(2, 3)
                i = t.tensor(integers, dtype=t.int64).reshape(2, 3).detach()
            return {'x': x.detach().requires_grad_(True), 'i': i}

        def apply(spec, live):
            left, right = (live['x'], live['i']) if spec['float_side'] == 'left' else (live['i'], live['x'])
            if spec['operation'] == 'add':
                return t.add(left, right, alpha=1)
            if spec['operation'] == 'sub':
                return t.sub(left, right, alpha=1)
            if spec['operation'] == 'mul':
                return t.mul(left, right)
            if spec['operation'] == 'div':
                return t.true_divide(left, right)
            raise AssertionError(spec['operation'])

        def seed_for(output):
            values = [-0., 1., -1., .5, 2., -2.]
            return t.tensor([values[k % len(values)] for k in range(output.numel())], dtype=t.float32).reshape(tuple(output.shape)).detach()

        def backward(live, root, seed, roots, retain, weak=None):
            _, event = capture(lambda: live[root].backward(live[seed], retain_graph=retain, create_graph=False))
            event.update(root=root, seed=seed, retain_graph=retain, after=snapshot(live, roots, weak))
            return event

        def mutate(live, name, value, roots, weak=None):
            def operation():
                with t.no_grad():
                    live[name].add_(value)
            _, event = capture(operation)
            event.update(target=name, scalar_add=value, after=snapshot(live, roots, weak))
            return event

        def observe(spec):
            live = inputs_for(spec)
            weak = live['i'].untyped_storage()._weak_ref()
            row = dict(spec, state_before=rng(), input_before=snapshot(live, [], weak), events=[])
            try:
                if spec['family'] == 'partial':
                    live['q'] = t.tensor([.5, 2., -1., 4., 3., -2.], dtype=t.float32).reshape(2, 3).detach().requires_grad_(True)
                    live['j'] = t.tensor([2., 3., 4., 5., 6., 7.], dtype=t.float32).reshape(2, 3).detach()
                    for name in spec['creation_order']:
                        live[name] = apply(spec, live) if name == 'bad' else t.mul(live['q'], live['j'])
                    live['y'] = t.add(live['good'], live['bad'])
                    roots = ['y', 'good', 'bad']
                    live['seed'] = seed_for(live['y'])
                    row['forward'] = {'accepted': True, 'outcome': 'SUCCEEDED', 'error': None}
                    row['after_forward'] = snapshot(live, roots, weak)
                    row['events'].append(dict(event='integer_mutation', **mutate(live, 'i', 1, roots, weak)))
                    for attempt in (1, 2):
                        row['events'].append(dict(event=f'partial_default_backward_{attempt}', **backward(live, 'y', 'seed', roots, False, weak)))
                else:
                    output, forward = capture(lambda: apply(spec, live))
                    row['forward'] = forward
                    if not forward['accepted']:
                        row.update(status='OBSERVED', after_forward=snapshot(live, [], weak), state_after_all=rng())
                        row['rng_unchanged'] = row['state_before'] == row['state_after_all']
                        return row
                    live['y'] = output
                    del output
                    roots = ['y']
                    if spec['family'] == 'shared_lifetime':
                        live['z'] = apply(spec, live)
                        roots.append('z')
                        live['seed_z'] = seed_for(live['z'])
                    live['seed'] = seed_for(live['y'])
                    row['after_forward'] = snapshot(live, roots, weak)
                    family = spec['family']
                    if family in ('lifetime', 'shared_lifetime'):
                        del live['i']
                        gc.collect()
                        row['events'].append({'event': 'drop_original_integer_local', 'after': snapshot(live, roots, weak)})
                        if family == 'lifetime':
                            row['events'].append(dict(event='backward_retain', **backward(live, 'y', 'seed', roots, True, weak)))
                        row['events'].append(dict(event='backward_default', **backward(live, 'y', 'seed', roots, False, weak)))
                        if family == 'shared_lifetime':
                            row['events'].append(dict(event='second_root_backward_default', **backward(live, 'z', 'seed_z', roots, False, weak)))
                        for name in tuple(roots):
                            del live[name]
                        gc.collect()
                        row['events'].append({'event': 'drop_all_output_roots', 'after': snapshot(live, [], weak)})
                    elif family in ('integer_mutation', 'float_mutation', 'data_mutation'):
                        target, increment = ('i', 1) if family == 'integer_mutation' else ('x', .25)
                        if family == 'data_mutation':
                            live['integer_data_alias'] = live['i'].data
                            row['events'].append({'event': 'create_integer_data_alias', 'after': snapshot(live, roots, weak)})
                            target, increment = 'integer_data_alias', 1
                        row['events'].append(dict(event='input_mutation', **mutate(live, target, increment, roots, weak)))
                        row['events'].append(dict(event='backward_default', **backward(live, 'y', 'seed', roots, False, weak)))
                        row['events'].append(dict(event='retry_default', **backward(live, 'y', 'seed', roots, False, weak)))
                    else:
                        for name, retain in (('backward_retain', True), ('backward_default', False), ('retry_default', False)):
                            row['events'].append(dict(event=name, **backward(live, 'y', 'seed', roots, retain, weak)))
                row.update(status='OBSERVED', state_after_all=rng())
                row['rng_unchanged'] = row['state_before'] == row['state_after_all']
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
                    event = dict(spec, status='PROBE_ERROR', error=repr(error), traceback=traceback.format_exc())
                    rows.append(event)
                    errors.append(event)
        report = {**common, 'status': 'OBSERVED' if not errors else 'PROBE_ERROR', 'torch_executed': True,
                  'oracle_provenance': provenance, 'build_configuration': t.__config__.show(),
                  'cpu_capability': t.backends.cpu.get_cpu_capability(), 'cases': len(rows), 'rows': rows,
                  'probe_errors': errors, 'probe_error_count': len(errors), 'rng_states': states}
    assert all(p.read_bytes() == raw for p, raw in inputs.items())
    report['source_files_unchanged'] = True
    report['inputs_sha256'] = {str(p): SHA(raw) for p, raw in inputs.items()}
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'cases', 'probe_sha256')}))
    return int(report['status'] == 'PROBE_ERROR')


if __name__ == '__main__':
    raise SystemExit(main())
