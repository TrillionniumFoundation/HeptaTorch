#!/usr/bin/env python3
"""Observe actual CPU int64 inplace overlap, cast and error ordering.

No native implementation or expected outcomes enter this reference process.
All roots remain alive through before/after observations; this is not a weak
lifetime test. The Scalar and Tensor overloads are recorded separately.
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
          'git_matches_oracle_build_pin', 'distribution_version_matches',
          'direct_url_matches', 'direct_url_archive_sha256_matches',
          'pip_report_url_matches', 'pip_report_archive_sha256_matches',
          'pip_report_version_matches', 'pip_report_requested'}
OPS = ['add', 'sub', 'mul', 'bitwise_and', 'bitwise_or', 'bitwise_xor']
LAYOUTS = ['reverse-partial', 'disjoint-contiguous', 'interleaved-disjoint',
           'strided-partial', 'transpose-mismatch', 'same-internal-overlap',
           'independent-internal-overlap', 'zero-stride-target',
           'singleton-zero-stride', 'empty-zero-stride',
           'broadcast-output-mismatch', 'partial-overlap-broadcast',
           'overlap-with-scalar-Tensor', 'Scalar-on-overlap',
           'Scalar-on-zero-stride', 'Scalar-on-empty']
DTYPES = ['bool-add', 'bool-sub', 'bool-and', 'float-add', 'float-and',
          'float-divide', 'bool-shape-sub', 'float-shape-add',
          'zero-stride-float-add', 'empty-float-add', 'empty-bool-add',
          'empty-shared-i64-sub']
PLAN = ([{'name': form + '-' + op, 'form': form, 'op': op, 'family': 'shared'}
         for form in ('exact-alias', 'partial-shift') for op in OPS]
        + [{'name': name, 'form': name, 'op': 'add', 'family': 'layout'} for name in LAYOUTS]
        + [{'name': name, 'form': name, 'family': 'dtype'} for name in DTYPES])
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'lock', 'repo', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    assert len(PLAN) == len({r['name'] for r in PLAN}) == 40
    assert not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inputs = {q.resolve(): q.read_bytes() for q in (Path(__file__), args.reference, args.lock)}
    assert sha(inputs[args.reference.resolve()]) == SUPPORT
    assert sha(inputs[args.lock.resolve()]) == LOCK
    common = {'schema': 'heptatorch.int64-inplace-observations.v1',
              'probe_sha256': sha(inputs[Path(__file__).resolve()]),
              'primary_source_ref': UPSTREAM, 'expected_cases': 40, 'case_plan': PLAN,
              'native_executed': False, 'full_pytorch_rewrite_complete': False,
              'pytorch_api_denominator': 2590, 'weak_lifetime_claimed': False,
              'scope': 'Fixed CPU int64 inplace Tensor/Scalar calls, actual overlap acceptance, raw before/after Storage and values, identity, layout, Version, error text and full CPU RNG.'}
    if args.prepare_only:
        report = {**common, 'status': 'PREPARED_NOT_RUN', 'torch_executed': False, 'cases': 0, 'rows': []}
    else:
        assert Path.cwd().resolve() == args.repo.resolve()
        saved = sys.argv
        try:
            sys.argv = [str(args.reference), str(args.repo / 'reference-only-no-native-execution'),
                        str(args.output.with_suffix('.provenance.json')), str(args.lock)]
            context = runpy.run_path(str(args.reference), run_name='int64_inplace_reference_only')
        finally:
            sys.argv = saved
        t, prov = context['torch'], context['PROVENANCE']
        assert prov['status'] == 'PASS' and set(prov['checks']) == CHECKS
        assert all(v is True for v in prov['checks'].values())
        states = {}

        def state():
            raw = bytes(t.get_rng_state().tolist())
            assert len(raw) == 5056
            key = sha(raw)
            states.setdefault(key, {'bytes': len(raw), 'hex': raw.hex()})
            return key

        def tensor(x):
            if x is None:
                return None
            raw = bytes(x.detach().contiguous().reshape(-1).view(t.uint8).tolist())
            physical = bytes(x.untyped_storage())
            return {'dtype': str(x.dtype), 'element_size': x.element_size(),
                    'shape': list(x.shape), 'strides': list(x.stride()),
                    'numel': x.numel(), 'storage_offset_elements': x.storage_offset(),
                    'tensor_identity': str(x._cdata),
                    'storage_identity': str(x.untyped_storage()._cdata),
                    'base_tensor_identity': None if x._base is None else str(x._base._cdata),
                    'storage_bytes': x.untyped_storage().nbytes(), 'version': x._version,
                    'requires_grad': x.requires_grad, 'is_leaf': x.is_leaf,
                    'is_view': x._is_view(),
                    'grad_fn': None if x.grad_fn is None else type(x.grad_fn).__name__,
                    'logical_bytes': len(raw), 'logical_bytes_hex': raw.hex(),
                    'storage_bytes_hex': physical.hex()}

        def observe(spec):
            roots, construction = {}, []

            def create(name, values, dtype, shape=None):
                x = t.tensor(values, dtype=dtype)
                if shape is not None:
                    assert list(x.shape) == shape
                roots[name] = x
                construction.append({'kind': 'tensor', 'name': name, 'values': values,
                                     'dtype': str(dtype), 'returned': tensor(x)})
                return x

            def view(name, parent, shape, strides, offset):
                x = roots[parent].as_strided(tuple(shape), tuple(strides), offset)
                roots[name] = x
                construction.append({'kind': 'as_strided', 'name': name, 'parent': parent,
                                     'shape': shape, 'strides': strides, 'offset': offset,
                                     'returned': tensor(x)})
                return x

            create('base', [-(1 << 63), (1 << 63)-1, -1, 0, 1, 3,
                            7, -9, 11, 13, 17, 19, 23, 29, 31, 37], t.int64)
            form, op, scalar = spec['form'], spec.get('op', 'add'), None
            if form == 'exact-alias':
                view('x', 'base', [2, 3], [3, 1], 0)
                view('y', 'base', [2, 3], [3, 1], 0)
            elif form in ('partial-shift', 'reverse-partial'):
                xoff, yoff = (0, 1) if form == 'partial-shift' else (1, 0)
                view('x', 'base', [5], [1], xoff)
                view('y', 'base', [5], [1], yoff)
            elif form == 'disjoint-contiguous':
                view('x', 'base', [3], [1], 0); view('y', 'base', [3], [1], 6)
            elif form in ('interleaved-disjoint', 'strided-partial'):
                view('x', 'base', [4], [2], 0)
                view('y', 'base', [4], [2], 1 if form == 'interleaved-disjoint' else 2)
            elif form == 'transpose-mismatch':
                view('x', 'base', [2, 2], [2, 1], 0); view('y', 'base', [2, 2], [1, 2], 0)
            elif form in ('same-internal-overlap', 'independent-internal-overlap',
                          'overlap-with-scalar-Tensor', 'Scalar-on-overlap'):
                view('x', 'base', [2, 2], [1, 1], 0)
                if form == 'same-internal-overlap':
                    view('y', 'base', [2, 2], [1, 1], 0)
                elif form == 'independent-internal-overlap':
                    create('y', [[1, 2], [3, 4]], t.int64)
                elif form == 'overlap-with-scalar-Tensor':
                    create('y', 1, t.int64)
                else:
                    scalar = 1
            elif form in ('zero-stride-target', 'Scalar-on-zero-stride'):
                view('x', 'base', [2, 3], [0, 1], 0)
                if form == 'zero-stride-target': create('y', [1, 2, 3], t.int64)
                else: scalar = 1
            elif form == 'singleton-zero-stride':
                view('x', 'base', [1, 3], [0, 1], 0); create('y', [1, 2, 3], t.int64)
            elif form in ('empty-zero-stride', 'Scalar-on-empty'):
                view('x', 'base', [2, 0, 3], [0, 0, 1], 7)
                if form == 'empty-zero-stride': create('y', [1, 2, 3], t.int64)
                else: scalar = 1
            elif form == 'broadcast-output-mismatch':
                view('x', 'base', [1, 3], [3, 1], 0); create('y', [[1], [2]], t.int64)
            elif form == 'partial-overlap-broadcast':
                view('x', 'base', [2, 3], [3, 1], 0); view('y', 'base', [3], [1], 1)
            elif spec['family'] == 'dtype':
                empty = form.startswith('empty-')
                zero = form.startswith('zero-stride-')
                view('x', 'base', [2, 0, 3] if empty else [2, 3],
                     [17, 101, 2] if empty else ([0, 1] if zero else [3, 1]), 7 if empty else 0)
                if form == 'empty-shared-i64-sub':
                    view('y', 'base', [2, 0, 3], [17, 101, 2], 7); op = 'sub'
                else:
                    dtype = t.bool if 'bool' in form else t.float32
                    vals = [True, False, True] if dtype == t.bool else [0.5, -2.0, 4.0]
                    if '-shape-' in form: vals = vals[:2]
                    create('y', vals, dtype)
                    if form.endswith('-sub'): op = 'sub'
                    elif form.endswith('-and'): op = 'bitwise_and'
                    elif form.endswith('-divide'): op = 'true_divide'
            else:
                raise AssertionError(('unhandled layout', form))
            before = {name: tensor(x) for name, x in roots.items()}
            before_state = state()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    result = getattr(roots['x'], op + '_')(scalar if scalar is not None else roots['y'])
                    event = {'accepted': True, 'error': None}
                except Exception as error:
                    result = None
                    event = {'accepted': False, 'error': {'type': type(error).__name__, 'message': str(error)}}
            after_state = state()
            event['warnings'] = [{'type': type(w.message).__name__, 'message': str(w.message)} for w in caught]
            return {'constructions': construction, 'before': before, 'operation': event,
                    'op': op, 'overload': 'Scalar' if scalar is not None else 'Tensor',
                    'scalar': scalar, 'returned': tensor(result),
                    'after': {name: tensor(x) for name, x in roots.items()},
                    'state_before_operation': before_state, 'state_after_operation': after_state,
                    'all_strong_roots': list(roots), 'weak_lifetime_claimed': False}

        rows, errors = [], []
        with t.random.fork_rng(devices=[]):
            for index, spec in enumerate(PLAN):
                try:
                    t.manual_seed(1729); before = state()
                    observation = observe(spec)
                    rows.append({'index': index, **spec, 'status': 'OBSERVED', 'observation': observation,
                                 'state_before': before, 'state_after': state()})
                except BaseException as error:
                    row = {'index': index, **spec, 'status': 'PROBE_ERROR',
                           'error': repr(error), 'traceback': traceback.format_exc()}
                    rows.append(row); errors.append(row)
        report = {**common, 'status': 'OBSERVED' if not errors else 'PROBE_ERROR',
                  'torch_executed': True, 'cases': len(rows), 'rows': rows,
                  'probe_error_count': len(errors), 'probe_errors': errors,
                  'oracle_provenance': prov, 'states': states,
                  'threads': t.get_num_threads(), 'cpu_capability': t.backends.cpu.get_cpu_capability()}
    assert all(path.read_bytes() == raw for path, raw in inputs.items())
    report['source_files_unchanged'] = True
    report['inputs_sha256'] = {str(path): sha(raw) for path, raw in inputs.items()}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'cases': report['cases'],
                      'output_bytes': args.output.stat().st_size,
                      'output_sha256': sha(args.output.read_bytes())}))
    return int(report['status'] == 'PROBE_ERROR')


if __name__ == '__main__':
    raise SystemExit(main())
