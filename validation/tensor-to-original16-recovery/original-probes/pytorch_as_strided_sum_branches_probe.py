#!/usr/bin/env python3
"""Observe CPU sum branch order through true AsStrided backward.

Six additive recipes only. This file never imports the native producer or
computes an expected gradient. Earlier four/four/two references remain
unchanged. Source-selected scalar outer, vector inner, and vector outer
geometries use explicit binary32 seed words; actual output is only observed.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import runpy
import struct
import sys
import traceback
import warnings

SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
CHECKS = {
    'python_matches', 'platform_matches', 'version_backend_matches',
    'git_matches_oracle_build_pin', 'distribution_version_matches',
    'direct_url_matches', 'direct_url_archive_sha256_matches',
    'pip_report_url_matches', 'pip_report_archive_sha256_matches',
    'pip_report_version_matches', 'pip_report_requested',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def plan():
    layouts = (
        ('scalar_outer_groups4', [2, 4], [8, 4], [0, 1]),
        ('vector_inner_main_tail_scalar', [2, 3], [3, 81], [1, 0]),
        ('vector_outer_main_tail_scalar', [2, 81], [8, 81], [0, 1]),
    )
    result = []
    for branch, base_shape, shape, stride in layouts:
        rows, cols = shape
        for seed_kind in ('quiet_nan_payload', 'finite_cancellation'):
            words = [0] * (rows * cols)
            inner = branch.startswith('vector_inner')
            outputs = rows if inner else cols
            for output in range(outputs):
                def put(reduced, word):
                    index = output * cols + reduced if inner else reduced * cols + output
                    words[index] = word
                sign = 0x80000000 if output % 2 else 0
                if seed_kind == 'quiet_nan_payload':
                    if inner:
                        put(4, (0x7FC10000 | (output + 1)) ^ sign)
                        put(64, (0x7FC20000 | (output + 1)) ^ (sign ^ 0x80000000))
                        put(80, (0x7FC30000 | (output + 1)) ^ sign)
                    else:
                        put(1, (0x7FC10000 | (output + 1)) ^ sign)
                        put(4, (0x7FC20000 | (output + 1)) ^ (sign ^ 0x80000000))
                else:
                    unit = 0xBF800000 if output % 3 == 2 else 0x3F800000
                    extra = 0 if output % 3 == 0 else unit
                    put(0, 0x4C000000)
                    put(4 if inner else 1, unit)
                    put(64 if inner else 4, 0xCC000000)
                    put(80 if inner else 5, extra)
            result.append({'index': len(result), 'name': branch + '__' + seed_kind,
                           'source_inferred_branch': branch,
                           'base_shape': base_shape, 'shape': shape,
                           'stride': stride, 'offset': 0,
                           'seed_kind': seed_kind, 'seed_words': words,
                           'seed_hex': struct.pack('<' + 'I' * len(words), *words).hex()})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'lock', 'repo', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    assert Path.cwd().resolve() == args.repo.resolve()
    assert not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inputs = {p.resolve(): p.read_bytes() for p in (Path(__file__), args.reference, args.lock)}
    assert sha(inputs[args.reference.resolve()]) == SUPPORT_SHA
    assert sha(inputs[args.lock.resolve()]) == LOCK_SHA
    common = {
        'schema': 'heptatorch.as-strided-sum-branches-reference6.v1',
        'probe_sha256': sha(inputs[Path(__file__).resolve()]),
        'primary_source_ref': '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457',
        'case_plan': plan(), 'native_executed': False, 'native_qualified': False,
        'pytorch_api_denominator': 2590, 'pytorch_api_qualified': 0,
        'full_pytorch_rewrite_complete': False,
        'policy': 'OBSERVATION ONLY. Raw seed words are input, never a computed gradient expectation.',
        'scope': 'Three zero-stride view geometries select scalar-outer groups of four columns, vector-inner with main/vector-tail/scalar-tail, and vector-outer with main/vector-tail/scalar-tail on the fixed AVX512 CPU profile. Each uses explicit signed quiet-NaN or finite cancellation input. Branch names are source inferences, not instruction tracing. First-order backward only.',
        'getter_order': 'dtype/layout/storage/version/requires, is_leaf, grad_fn and immediate next-edge kinds/AccumulateGrad Tensor, base, retains_grad, then current grad. These getters may refresh lazy ViewInfo history.',
        'ownership_scope': 'Only live holds the explicitly named base/view/copy/seed. Snapshot functions return primitives and copied bytes; this probe makes no weak-reference or object-reclamation claim.',
    }
    if args.prepare_only:
        report = {**common, 'status': 'PREPARED_NOT_RUN', 'torch_executed': False,
                  'cases': 0, 'rows': []}
    else:
        argv = sys.argv
        try:
            sys.argv = [str(args.reference), str(args.repo / 'reference-only-no-native-execution'),
                        str(args.output.with_suffix('.provenance.json')), str(args.lock)]
            context = runpy.run_path(str(args.reference), run_name='as_strided_sum_branches_provenance_only')
        finally:
            sys.argv = argv
        torch, provenance = context['torch'], context['PROVENANCE']
        assert provenance['status'] == 'PASS' and set(provenance['checks']) == CHECKS
        assert all(value is True for value in provenance['checks'].values())
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.utils.deterministic.fill_uninitialized_memory is True
        assert torch.backends.cpu.get_cpu_capability() == 'AVX512'
        states, rows, errors = {}, [], []

        def rng():
            raw = bytes(torch.get_rng_state().tolist())
            assert len(raw) == 5056
            key = sha(raw)
            states.setdefault(key, {'bytes': len(raw), 'hex': raw.hex()})
            return key

        def node_info(fn):
            if fn is None:
                return None
            edges = []
            for child, output_nr in fn.next_functions:
                edge = {'kind': None if child is None else type(child).__name__,
                        'output_nr': output_nr}
                if child is not None and type(child).__name__ == 'AccumulateGrad':
                    variable = child.variable
                    edge['variable_tensor_identity'] = str(variable._cdata)
                    del variable
                edges.append(edge)
                del child
            return {'kind': type(fn).__name__, 'next_edges': edges}

        def tensor(x, include_grad=False):
            if x is None:
                return None
            storage = x.untyped_storage()
            raw = bytes(storage)
            shape, stride, offset = tuple(x.shape), tuple(x.stride()), x.storage_offset()
            width = x.element_size()
            logical = b''.join(raw[(offset + sum(c * s for c, s in zip(coords, stride))) * width:
                                  (offset + sum(c * s for c, s in zip(coords, stride))) * width + width]
                               for coords in itertools.product(*(range(d) for d in shape)))
            out = {
                'tensor_identity': str(x._cdata), 'storage_identity': str(storage._cdata),
                'dtype': str(x.dtype), 'width': width, 'shape': list(shape),
                'stride': list(stride), 'offset': offset, 'storage_bytes': storage.nbytes(),
                'version': x._version, 'requires_grad': x.requires_grad,
                'is_leaf': x.is_leaf,
                'storage_hex': raw.hex(), 'storage_sha256': sha(raw),
                'logical_hex': logical.hex(), 'logical_sha256': sha(logical),
                'binary_words': [int.from_bytes(logical[i:i + width], 'little')
                                 for i in range(0, len(logical), width)],
            }
            fn = x.grad_fn
            out['grad_fn'] = node_info(fn)
            del fn
            base = x._base
            out['base_tensor_identity'] = None if base is None else str(base._cdata)
            del base
            out['retains_grad'] = x.retains_grad
            if include_grad:
                gradient = x.grad
                out['grad'] = tensor(gradient)
                del gradient
            del storage
            return out

        original = {'rng': rng(), 'grad_enabled': torch.is_grad_enabled(),
                    'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                    'deterministic_fill': torch.utils.deterministic.fill_uninitialized_memory}

        def observe(spec, row):
            live = {}
            row.update(status='OBSERVED', events=[], snapshots=[], state_before=rng())

            def event(label, action, publish=None):
                observation = {'label': label, 'state_before': rng()}
                row['events'].append(observation)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    try:
                        value = action()
                        if publish is not None:
                            live[publish] = value
                        observation.update(accepted=True, error=None,
                                           result_tensor_identity=(str(value._cdata)
                                               if isinstance(value, torch.Tensor) else None))
                        del value
                    except Exception as error:
                        observation.update(accepted=False, result_tensor_identity=None,
                                           error={'type': type(error).__name__, 'message': str(error)})
                observation['warnings'] = [{'type': type(w.message).__name__, 'message': str(w.message)} for w in caught]
                observation['state_after'] = rng()
                return observation['accepted']

            def snapshot(label):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    values = {name: tensor(live[name], include_grad=name in ('base', 'view', 'copy'))
                              for name in ('base', 'view', 'copy', 'seed') if name in live}
                row['snapshots'].append({'label': label, 'tensors': values,
                    'warnings': [{'type': type(w.message).__name__, 'message': str(w.message)} for w in caught],
                    'rng': rng()})

            with torch.enable_grad():
                event('create_base', lambda: torch.tensor(
                      [[float(r * spec['base_shape'][1] + c) for c in range(spec['base_shape'][1])]
                       for r in range(spec['base_shape'][0])],
                      dtype=torch.float32, device='cpu', requires_grad=False), publish='base')
                event('create_true_view', lambda: live['base'].as_strided(spec['shape'], spec['stride'], spec['offset']), publish='view')
                snapshot('before_requires')
                event('base_requires_true', lambda: live['base'].requires_grad_(True))
                snapshot('derived_requires_before_version')

                def mutate_base():
                    with torch.no_grad():
                        return live['base'].add_(1.0)
                event('base_add_one', mutate_base)
                snapshot('after_version_and_ordered_getters')
                event('real_to_copy', lambda: torch.ops.aten._to_copy.default(live['view'],
                      dtype=torch.float32, memory_format=torch.preserve_format), publish='copy')
                event('retain_view_gradient', lambda: live['view'].retain_grad())

                def make_seed():
                    data = bytearray.fromhex(spec['seed_hex'])
                    return torch.frombuffer(data, dtype=torch.float32).clone().reshape(spec['shape'])
                event('create_raw_seed', make_seed, publish='seed')
                snapshot('before_backward')
                event('backward', lambda: live['copy'].backward(live['seed'], retain_graph=False, create_graph=False))
                snapshot('after_backward')
            row['state_after'] = rng()
            live.clear()

        with torch.random.fork_rng(devices=[]):
            for spec in plan():
                row = dict(spec)
                rows.append(row)
                try:
                    observe(spec, row)
                except BaseException as error:
                    row.update(status='PROBE_ERROR', error=repr(error), traceback=traceback.format_exc())
                    errors.append({'index': spec['index'], 'error': repr(error)})
        after = {'rng': rng(), 'grad_enabled': torch.is_grad_enabled(),
                 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                 'deterministic_fill': torch.utils.deterministic.fill_uninitialized_memory}
        assert after == original
        report = {**common, 'status': 'OBSERVED' if not errors else 'PROBE_ERROR',
                  'torch_executed': True, 'oracle_provenance': provenance,
                  'build_configuration': torch.__config__.show(),
                  'cpu_capability': torch.backends.cpu.get_cpu_capability(),
                  'environment_before': original, 'environment_after': after,
                  'environment_restored': True, 'cases': len(rows), 'rows': rows,
                  'states': states, 'probe_error_count': len(errors), 'probe_errors': errors,
                  'operation_totals': {'attempted': sum(len(row.get('events', [])) for row in rows),
                    'succeeded': sum(e['accepted'] is True for row in rows for e in row.get('events', [])),
                    'raised': sum(e['accepted'] is False for row in rows for e in row.get('events', []))}}
    assert all(path.read_bytes() == raw for path, raw in inputs.items())
    report['source_files_unchanged'] = True
    report['inputs_sha256'] = {str(path): sha(raw) for path, raw in inputs.items()}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'cases', 'native_executed', 'probe_sha256')}))
    return int(report.get('probe_error_count', 0) != 0)


if __name__ == '__main__':
    raise SystemExit(main())
