#!/usr/bin/env python3
"""Record actual int64 Tensor behavior before implementing its HIA successor.

This is an additive reference probe, not a numerical runtime or a native
qualification. It records actual values, aliases, layouts, Versions, errors
and complete CPU RNG states without substituting an expected outcome.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import traceback
import warnings

SUPPORT = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
UPSTREAM = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
CHECKS = {'python_matches', 'platform_matches', 'version_backend_matches',
          'git_matches_oracle_build_pin', 'distribution_version_matches', 'direct_url_matches',
          'direct_url_archive_sha256_matches', 'pip_report_url_matches',
          'pip_report_archive_sha256_matches', 'pip_report_version_matches', 'pip_report_requested'}
FACTORIES = ['scalar', 'edge-vector', 'transpose', 'clone-transpose', 'contiguous-view',
             'offset-view', 'empty-nonstandard', 'clone-empty', 'detach', 'data-alias',
             'fill-signed-min', 'construction-requires-grad']
OPS = ['add', 'sub', 'mul', 'bitwise_and', 'bitwise_or', 'bitwise_xor']
PROMOTIONS = ['i64-f32-add', 'f32-i64-mul', 'i64-bool-add', 'bool-i64-mul',
              'i64-true-divide', 'requires-grad-inplace']
PLAN = ([{'name': 'tensor-' + name, 'family': 'tensor', 'operation': name} for name in FACTORIES]
        + [{'name': 'binary-' + layout + '-' + op, 'family': 'binary', 'layout': layout, 'operation': op}
           for layout in ['dense', 'transpose', 'broadcast'] for op in OPS]
        + [{'name': 'inplace-transpose-' + op, 'family': 'inplace', 'operation': op} for op in OPS]
        + [{'name': 'promotion-' + name, 'family': 'promotion', 'operation': name} for name in PROMOTIONS]
        + [{'name': 'dropout-' + layout + '-' + str(train).lower() + '-' + str(probability),
            'family': 'dropout', 'layout': layout, 'train': train, 'probability': probability}
           for layout in ['dense', 'empty'] for train in [False, True] for probability in [0.0, 0.5, 1.0]])


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'lock', 'repo', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    assert len(PLAN) == len({row['name'] for row in PLAN}) == 54
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists(), 'Create-only reference output'
    inputs = {Path(__file__).resolve(): Path(__file__).read_bytes(),
              args.reference.resolve(): args.reference.read_bytes(), args.lock.resolve(): args.lock.read_bytes()}
    assert sha(inputs[args.reference.resolve()]) == SUPPORT
    assert sha(inputs[args.lock.resolve()]) == LOCK
    common = {'schema': 'heptatorch.integer-tensor-foundation-observation.v1',
              'probe_sha256': sha(inputs[Path(__file__).resolve()]), 'primary_source_ref': UPSTREAM,
              'expected_cases': len(PLAN), 'case_plan': PLAN, 'native_executed': False,
              'native_values_compared': 0, 'full_pytorch_rewrite_complete': False,
              'pytorch_api_denominator': 2590,
              'scope': 'Actual CPU int64 Tensor construction, aliases, views, clone, six binary and inplace operations, six promotion/admission cases and twelve native_dropout cases. No native qualification or full-schema claim.'}
    if args.prepare_only:
        report = {**common, 'status': 'PREPARED_NOT_RUN', 'torch_executed': False, 'cases': 0, 'rows': [],
                  'inputs_sha256': {str(path): sha(raw) for path, raw in inputs.items()}}
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'status': report['status'], 'expected_cases': len(PLAN), 'probe_sha256': common['probe_sha256']}))
        return 0
    assert Path.cwd().resolve() == args.repo.resolve()
    previous_argv = sys.argv
    try:
        sys.argv = [str(args.reference), str(args.repo / 'reference-only-no-native-execution'),
                    str(args.output.with_suffix('.provenance.json')), str(args.lock)]
        context = runpy.run_path(str(args.reference), run_name='integer_tensor_provenance_only')
    finally:
        sys.argv = previous_argv
    torch, provenance = context['torch'], context['PROVENANCE']
    assert provenance['status'] == 'PASS' and set(provenance['checks']) == CHECKS
    assert all(value is True for value in provenance['checks'].values())
    states, rows, errors = {}, [], []
    minimum, maximum = -(1 << 63), (1 << 63) - 1

    def state():
        raw = bytes(torch.get_rng_state().tolist())
        assert len(raw) == 5056
        key = sha(raw)
        states.setdefault(key, {'bytes': len(raw), 'hex': raw.hex()})
        return key

    def tensor(value):
        if value is None:
            return None
        assert value.device.type == 'cpu'
        logical = bytes(value.detach().contiguous().reshape(-1).view(torch.uint8).tolist())
        storage = bytes(value.untyped_storage())
        return {'dtype': str(value.dtype), 'element_size': value.element_size(),
                'shape': list(value.shape), 'strides': list(value.stride()), 'numel': value.numel(),
                'tensor_identity': str(value._cdata), 'storage_identity': str(value.untyped_storage()._cdata),
                'storage_bytes': value.untyped_storage().nbytes(), 'storage_offset_elements': value.storage_offset(),
                'version': value._version, 'requires_grad': value.requires_grad, 'is_leaf': value.is_leaf,
                'grad_fn': type(value.grad_fn).__name__ if value.grad_fn is not None else None,
                'logical_bytes': len(logical), 'logical_bytes_hex': logical.hex(),
                'storage_bytes_hex': storage.hex()}

    def snapshot(roots):
        return {name: tensor(value) for name, value in roots.items()}

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

    def edge_matrix():
        return torch.tensor([minimum, maximum, -1, 0, 1, 3], dtype=torch.int64).reshape(2, 3)

    def observe(spec):
        roots = {}
        family, operation = spec['family'], spec.get('operation')
        if family == 'tensor':
            if operation == 'scalar':
                perform = lambda: torch.tensor(minimum, dtype=torch.int64)
            elif operation == 'edge-vector':
                perform = lambda: torch.tensor([minimum, maximum, -1, 0, 1, 3], dtype=torch.int64)
            elif operation == 'construction-requires-grad':
                perform = lambda: torch.tensor([1, 2], dtype=torch.int64, requires_grad=True)
            elif operation in ('empty-nonstandard', 'clone-empty'):
                roots['base'] = torch.empty((0,), dtype=torch.int64)
                if operation == 'empty-nonstandard':
                    perform = lambda: roots['base'].as_strided((2, 0, 3), (17, 101, 2), 7)
                else:
                    roots['x'] = roots['base'].as_strided((2, 0, 3), (17, 101, 2), 7)
                    perform = lambda: roots['x'].clone()
            else:
                roots['base'] = edge_matrix()
                if operation in ('transpose', 'clone-transpose'):
                    if operation == 'transpose':
                        perform = lambda: roots['base'].t()
                    else:
                        roots['x'] = roots['base'].t()
                        perform = lambda: roots['x'].clone()
                elif operation == 'contiguous-view':
                    perform = lambda: roots['base'].view(3, 2)
                elif operation == 'offset-view':
                    perform = lambda: roots['base'].as_strided((2, 2), (3, 1), 1)
                elif operation == 'detach':
                    perform = lambda: roots['base'].detach()
                elif operation == 'data-alias':
                    perform = lambda: roots['base'].data
                elif operation == 'fill-signed-min':
                    roots['x'] = roots['base'].t()
                    perform = lambda: roots['x'].fill_(minimum)
                else:
                    raise AssertionError(('unknown tensor recipe', operation))
        elif family == 'binary':
            roots['base'] = edge_matrix()
            roots['other'] = torch.tensor([-1, 2, 5, 7, maximum, minimum], dtype=torch.int64).reshape(2, 3)
            if spec['layout'] == 'transpose':
                roots['x'], roots['y'] = roots['base'].t(), roots['other'].t()
            elif spec['layout'] == 'broadcast':
                roots['x'] = roots['base']
                roots['y'] = torch.tensor([minimum, 1, maximum], dtype=torch.int64)
            else:
                roots['x'], roots['y'] = roots['base'], roots['other']
            perform = lambda: getattr(torch, operation)(roots['x'], roots['y'])
        elif family == 'inplace':
            roots['base'] = edge_matrix()
            roots['x'] = roots['base'].t()
            roots['y'] = torch.tensor([[-1], [1], [3]], dtype=torch.int64)
            perform = lambda: getattr(roots['x'], operation + '_')(roots['y'])
        elif family == 'promotion':
            roots['x'] = edge_matrix()
            if operation == 'i64-f32-add':
                roots['y'] = torch.tensor([0.5, -2.0, 4.0], dtype=torch.float32)
                perform = lambda: torch.add(roots['x'], roots['y'])
            elif operation == 'f32-i64-mul':
                roots['y'] = torch.tensor([[0.5], [-2.0]], dtype=torch.float32)
                perform = lambda: torch.mul(roots['y'], roots['x'])
            elif operation in ('i64-bool-add', 'bool-i64-mul'):
                roots['y'] = torch.tensor([True, False, True], dtype=torch.bool)
                perform = (lambda: torch.add(roots['x'], roots['y'])) if operation == 'i64-bool-add' else (lambda: torch.mul(roots['y'], roots['x']))
            elif operation == 'i64-true-divide':
                roots['y'] = torch.tensor([2, 3, -1], dtype=torch.int64)
                perform = lambda: torch.div(roots['x'], roots['y'])
            elif operation == 'requires-grad-inplace':
                perform = lambda: roots['x'].requires_grad_(True)
            else:
                raise AssertionError(('unknown promotion recipe', operation))
        elif family == 'dropout':
            if spec['layout'] == 'empty':
                roots['base'] = torch.empty((0,), dtype=torch.int64)
                roots['x'] = roots['base'].as_strided((2, 0, 3), (17, 101, 2), 7)
            else:
                roots['x'] = edge_matrix().t()
            perform = lambda: torch.ops.aten.native_dropout.default(roots['x'], spec['probability'], spec['train'])
        else:
            raise AssertionError(('unknown family', family))
        before = snapshot(roots)
        before_state = state()
        result, event = capture(perform)
        after_state = state()
        result_values = list(result) if isinstance(result, (tuple, list)) else [result]
        return {'before': before, 'operation': event, 'after': snapshot(roots),
                'results': [tensor(value) for value in result_values],
                'state_before_operation': before_state, 'state_after_operation': after_state,
                'state_unchanged_by_operation': before_state == after_state,
                'strong_observer_roots': list(roots), 'weak_lifetime_claimed': False}

    with torch.random.fork_rng(devices=[]):
        for index, spec in enumerate(PLAN):
            try:
                torch.manual_seed(1729)
                before = state()
                observation = observe(spec)
                after = state()
                rows.append({'index': index, **spec, 'status': 'OBSERVED', 'observation': observation,
                             'state_before': before, 'state_after': after, 'state_unchanged': before == after})
            except BaseException as error:
                record = {'index': index, 'name': spec['name'], 'error': repr(error), 'traceback': traceback.format_exc()}
                errors.append(record)
                rows.append({**record, 'status': 'PROBE_ERROR'})
    assert all(path.read_bytes() == raw for path, raw in inputs.items())
    report = {**common, 'status': 'OBSERVED' if not errors else 'PROBE_ERROR', 'torch_executed': True,
              'oracle_provenance': provenance, 'source_files_unchanged': True, 'cases': len(rows), 'rows': rows,
              'probe_error_count': len(errors), 'probe_errors': errors, 'states': states,
              'unique_complete_states': len(states), 'cpu_capability': torch.backends.cpu.get_cpu_capability(),
              'threads': torch.get_num_threads(), 'inputs_sha256': {str(path): sha(raw) for path, raw in inputs.items()}}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'cases': len(rows), 'probe_error_count': len(errors),
                      'output_bytes': args.output.stat().st_size, 'output_sha256': sha(args.output.read_bytes())}))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
