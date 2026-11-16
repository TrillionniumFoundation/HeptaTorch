#!/usr/bin/env python3
"""Observe fixed CPU I64/F32 promotion, copies, layouts and original mixed ops.

The explicit to(common_dtype) observations are separate calls. Their returned
Tensors are never inputs to the original mixed add/true_divide calls. There is
no native execution and no expected result or layout supplied by a model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import traceback
import warnings

SUPPORT_SHA256 = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA256 = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
PARENT54_SHA256 = '4c454e443c4147c740ea400cb1195bcbcb80e16e5b50debce54c068fa9ae712b'
SOURCE_REF = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
CHECKS = {'python_matches', 'platform_matches', 'version_backend_matches',
          'git_matches_oracle_build_pin', 'distribution_version_matches', 'direct_url_matches',
          'direct_url_archive_sha256_matches', 'pip_report_url_matches',
          'pip_report_archive_sha256_matches', 'pip_report_version_matches', 'pip_report_requested'}
I64_DATA = (1, 16777217, -16777217, 1099511693313, -1099511693313,
            9223372036854775807, -9223372036854775808,
            3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
            33, 35, 37, 39, 41, 43, 45, 47, 49, 51)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def plan():
    layouts = [
        ('holes-row', (2, 3), (9, 2), 1, (1, 2), 0),
        ('holes-column', (2, 3), (2, 7), 1, (3, 1), 0),
        ('overlap', (2, 3), (1, 1), 1, (3, 1), 0),
        ('broadcast-zero-stride', (2, 3), (0, 1), 1, (1, 2), 0),
        ('singleton-2p60-minus1', (1, 2, 3), ((1 << 60)-1, 1, 2), 0, (6, 3, 1), 0),
        ('singleton-2p60', (1, 2, 3), (1 << 60, 1, 2), 0, (6, 3, 1), 0),
        ('singleton-2p61', (1, 2, 3), (1 << 61, 1, 2), 0, (6, 3, 1), 0),
        ('singleton-2p62', (1, 2, 3), (1 << 62, 1, 2), 0, (6, 3, 1), 0),
        ('empty-irregular', (2, 0, 3), (17, 101, 2), 7, (23, 103, 3), 5),
        ('empty-huge', (1, 0, 3), (1 << 62, 101, 2), 7, (1 << 61, 103, 3), 5),
    ]
    rows = []
    for name, shape, istride, ioffset, fstride, foffset in layouts:
        for order in ('i64-first', 'f32-first'):
            rows.append({'name': name+'-'+order, 'layout': name, 'order': order,
                         'shape': list(shape),
                         'i64_strides_elements_decimal': [str(x) for x in istride],
                         'i64_offset_elements_decimal': str(ioffset),
                         'f32_strides_elements_decimal': [str(x) for x in fstride],
                         'f32_offset_elements_decimal': str(foffset),
                         'operations': ['i64.to(common_dtype)', 'f32.to(common_dtype)',
                                        'torch.add(original_lhs, original_rhs)',
                                        'torch.true_divide(original_lhs, original_rhs)']})
    assert len(rows) == len({row['name'] for row in rows}) == 20
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'lock', 'repo', 'parent-observations', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists(), 'Output is create-only; preserve previous observations'
    paths = [Path(__file__).resolve(), args.reference.resolve(), args.lock.resolve(),
             args.parent_observations.resolve()]
    inputs = {p: p.read_bytes() for p in paths}
    assert sha(inputs[args.reference.resolve()]) == SUPPORT_SHA256
    assert sha(inputs[args.lock.resolve()]) == LOCK_SHA256
    assert sha(inputs[args.parent_observations.resolve()]) == PARENT54_SHA256
    common = {'schema': 'heptatorch.pinned-int64-mixed-promotion-observation.v1',
              'probe_sha256': sha(inputs[Path(__file__).resolve()]),
              'primary_source_ref': SOURCE_REF, 'parent54_observation_sha256': PARENT54_SHA256,
              'expected_cases': 20, 'case_plan': plan(),
              'i64_root_values_decimal': [str(x) for x in I64_DATA],
              'f32_root_values_decimal': [str(i+0.25) for i in range(1, 33)],
              'native_executed': False, 'native_values_compared': 0,
              'pytorch_api_denominator': 2590, 'whole_pytorch_gate': 'HOLD',
              'full_pytorch_rewrite_complete': False,
              'comparison_policy': 'Actual observations only; no outcome is substituted from a model. Explicit to() observations do not replace original mixed calls.',
              'scope': 'Ten fixed original-input layouts in two operand orders. Strong roots intentionally preserve observed object identities. No weak/free, general promotion, all dtype, or universal layout claim.'}
    if args.prepare_only:
        report = {**common, 'status': 'PREPARED_NOT_RUN', 'torch_executed': False,
                  'cases': 0, 'rows': [],
                  'inputs_sha256': {str(p): sha(b) for p, b in inputs.items()}}
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'status': report['status'], 'expected_cases': 20,
                          'probe_sha256': common['probe_sha256']}))
        return 0

    assert Path.cwd().resolve() == args.repo.resolve(), 'Use the real locked repo and pip receipt'
    for name in ('compatibility/upstream-lock.json', 'build/pytorch-install.json'):
        p = (args.repo/name).resolve()
        inputs[p] = p.read_bytes()
    previous_argv = sys.argv
    try:
        sys.argv = [str(args.reference), str(args.repo/'reference-only-no-native-execution'),
                    str(args.output.with_suffix('.provenance.json')), str(args.lock)]
        context = runpy.run_path(str(args.reference), run_name='int64_promotion_provenance_only')
    finally:
        sys.argv = previous_argv
    torch, provenance = context['torch'], context['PROVENANCE']
    assert provenance['status'] == 'PASS' and set(provenance['checks']) == CHECKS
    assert all(v is True for v in provenance['checks'].values())
    states, rows, errors = {}, [], []

    def rng_state():
        raw = bytes(torch.get_rng_state().tolist())
        assert len(raw) == 5056
        key = sha(raw)
        states.setdefault(key, {'bytes': len(raw), 'hex': raw.hex()})
        return key

    def tensor_info(value):
        if value is None:
            return None
        assert value.device.type == 'cpu'
        logical_tensor = value.detach().contiguous().reshape(-1)
        logical = bytes(logical_tensor.view(torch.uint8).tolist())
        storage = value.untyped_storage()
        storage_raw = bytes(storage)
        base, node = value._base, value.grad_fn
        return {'python_identity': str(id(value)), 'tensor_identity': str(value._cdata),
                'storage_identity': str(storage._cdata), 'dtype': str(value.dtype),
                'element_size': int(value.element_size()), 'numel': int(value.numel()),
                'shape': list(value.shape),
                'strides_elements_decimal': [str(int(x)) for x in value.stride()],
                'storage_offset_elements_decimal': str(int(value.storage_offset())),
                'storage_bytes': int(storage.nbytes()), 'version': int(value._version),
                'requires_grad': bool(value.requires_grad), 'is_leaf': bool(value.is_leaf),
                'is_view': bool(value._is_view()), 'is_contiguous': bool(value.is_contiguous()),
                'base_tensor_identity': None if base is None else str(base._cdata),
                'grad_fn': None if node is None else type(node).__name__,
                'logical_bytes': len(logical), 'logical_bytes_hex': logical.hex(),
                'storage_bytes_hex': storage_raw.hex(),
                'logical_values_text': [str(x) for x in logical_tensor.tolist()]}

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

    def observe(spec):
        roots = {'i64_root': None, 'i64': None, 'f32_root': None, 'f32': None}
        held_results = {}
        constructions = []
        for name in ('i64', 'f32'):
            before = rng_state()
            if name == 'i64':
                factory = lambda: torch.tensor(I64_DATA, dtype=torch.int64)
            else:
                factory = lambda: torch.tensor([i+0.25 for i in range(1, 33)], dtype=torch.float32)
            root, event = capture(factory)
            roots[name+'_root'] = root
            after = rng_state()
            constructions.append({'name': name+'_root', 'operation': event,
                                  'state_before': before, 'state_after': after,
                                  'returned': tensor_info(root)})
            if root is None:
                constructions.append({'name': name+'_view', 'operation': {'outcome': 'SKIPPED_ROOT_UNAVAILABLE'}})
                continue
            stride = tuple(int(x) for x in spec[name+'_strides_elements_decimal'])
            offset = int(spec[name+'_offset_elements_decimal'])
            before = rng_state()
            view, event = capture(lambda: root.as_strided(tuple(spec['shape']), stride, offset))
            roots[name] = view
            after = rng_state()
            constructions.append({'name': name+'_view', 'operation': event,
                                  'state_before': before, 'state_after': after,
                                  'returned': tensor_info(view)})

        def snapshot():
            return {name: tensor_info(value) for name, value in roots.items()}

        constructed = snapshot()
        if roots['i64'] is None or roots['f32'] is None:
            return {'constructions': constructions, 'constructed': constructed,
                    'common_dtype': None, 'events': [
                        {'name': name, 'operation': {'outcome': 'SKIPPED_ORIGINAL_INPUT_UNAVAILABLE'}}
                        for name in spec['operations']], 'final_originals': snapshot()}
        lhs = roots['i64'] if spec['order'] == 'i64-first' else roots['f32']
        rhs = roots['f32'] if spec['order'] == 'i64-first' else roots['i64']
        state_before_type = rng_state()
        dtype, type_event = capture(lambda: torch.result_type(lhs, rhs))
        state_after_type = rng_state()
        events = []

        def measured(name, operation):
            before_observer = rng_state()
            before = snapshot()
            before_call = rng_state()
            result, event = capture(operation)
            after_call = rng_state()
            held_results[name] = result
            returned = tensor_info(result)
            after = snapshot()
            after_observer = rng_state()
            return {'name': name, 'before_originals': before, 'operation': event,
                    'returned': returned, 'after_originals': after,
                    'state_before_observer': before_observer, 'state_before_call': before_call,
                    'state_after_call': after_call, 'state_after_observer': after_observer,
                    'state_unchanged_by_operation': before_call == after_call,
                    'observer_state_unchanged_before': before_observer == before_call,
                    'observer_state_unchanged_after': after_call == after_observer,
                    'original_snapshots_equal': before == after}

        if dtype is None:
            events.extend({'name': name, 'operation': {'outcome': 'SKIPPED_COMMON_DTYPE_UNAVAILABLE'}}
                          for name in spec['operations'][:2])
        else:
            events.append(measured('i64.to(common_dtype)', lambda: roots['i64'].to(dtype=dtype)))
            events.append(measured('f32.to(common_dtype)', lambda: roots['f32'].to(dtype=dtype)))
        # These are the ORIGINAL inputs, independently of every explicit to() result.
        events.append(measured('torch.add(original_lhs, original_rhs)', lambda: torch.add(lhs, rhs)))
        events.append(measured('torch.true_divide(original_lhs, original_rhs)', lambda: torch.true_divide(lhs, rhs)))
        return {'constructions': constructions, 'constructed': constructed,
                'common_dtype': None if dtype is None else str(dtype),
                'result_type_operation': type_event,
                'state_before_result_type': state_before_type, 'state_after_result_type': state_after_type,
                'original_lhs_tensor_identity': str(lhs._cdata), 'original_rhs_tensor_identity': str(rhs._cdata),
                'events': events, 'final_originals': snapshot(),
                'held_result_identities': {name: None if value is None else str(value._cdata)
                                           for name, value in held_results.items()},
                'weak_lifetime_claimed': False,
                'identity_scope': 'Exact identities within one recipe only; roots and all returned observations are strongly held until recipe end.'}

    external_before = rng_state()
    with torch.random.fork_rng(devices=[]):
        for index, spec in enumerate(plan()):
            try:
                torch.manual_seed(1729)
                before = rng_state()
                observation = observe(spec)
                after = rng_state()
                rows.append({'index': index, **spec, 'status': 'OBSERVED', 'observation': observation,
                             'state_before': before, 'state_after': after, 'state_unchanged': before == after})
            except BaseException as error:
                record = {'index': index, **spec, 'error': repr(error), 'traceback': traceback.format_exc()}
                errors.append(record)
                rows.append({**record, 'status': 'PROBE_ERROR'})
    external_after = rng_state()
    assert all(p.read_bytes() == b for p, b in inputs.items())
    report = {**common, 'status': 'OBSERVED' if not errors else 'PROBE_ERROR',
              'torch_executed': True, 'oracle_provenance': provenance,
              'source_files_unchanged': True, 'cases': len(rows), 'rows': rows,
              'probe_error_count': len(errors), 'probe_errors': errors, 'states': states,
              'unique_complete_states': len(states),
              'state_before_recipe_fork': external_before, 'state_after_recipe_fork': external_after,
              'recipe_fork_restored_rng': external_before == external_after,
              'cpu_capability': torch.backends.cpu.get_cpu_capability(),
              'default_dtype': str(torch.get_default_dtype()), 'threads': torch.get_num_threads(),
              'inputs_sha256': {str(p): sha(b) for p, b in inputs.items()}}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    events = [event['operation'] for row in rows if row['status'] == 'OBSERVED'
              for event in row['observation']['events']]
    print(json.dumps({'status': report['status'], 'cases': len(rows), 'probe_error_count': len(errors),
                      'attempted_to_or_mixed_calls': sum('accepted' in e for e in events),
                      'successful_to_or_mixed_calls': sum(e.get('accepted') is True for e in events),
                      'skipped_to_or_mixed_calls': sum(e.get('outcome', '').startswith('SKIPPED') for e in events),
                      'output_bytes': args.output.stat().st_size, 'output_sha256': sha(args.output.read_bytes())}))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
