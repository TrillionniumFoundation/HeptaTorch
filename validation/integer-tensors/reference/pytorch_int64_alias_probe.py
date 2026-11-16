#!/usr/bin/env python3
"""Observe eight actual int64 alias/Version trajectories on pinned PyTorch.

Reference only: no native execution, expected-value model, or predicted alias
relationship. Every mutation is the written scalar overload, not a Tensor RHS.
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
PRIMARY_SOURCE_REF = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
CHECKS = {'python_matches', 'platform_matches', 'version_backend_matches',
          'git_matches_oracle_build_pin', 'distribution_version_matches', 'direct_url_matches',
          'direct_url_archive_sha256_matches', 'pip_report_url_matches',
          'pip_report_archive_sha256_matches', 'pip_report_version_matches', 'pip_report_requested'}
MINIMUM = -(1 << 63)
MAXIMUM = (1 << 63) - 1


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def plan():
    rows = []
    for construction in ('detach', 'data-alias', 'as-strided', 'clone-transpose'):
        for empty in (False, True):
            if construction == 'detach':
                expression = 'base.detach()'
            elif construction == 'data-alias':
                expression = 'base.data'
            elif construction == 'as-strided':
                expression = ('base.as_strided((2, 0, 3), (17, 101, 2), 7)' if empty else
                              'base.as_strided((2, 2), (3, 1), 1)')
            else:
                expression = 'base.transpose(0, 1).clone(memory_format=torch.preserve_format)'
            rows.append({'name': construction + ('-empty' if empty else '-six'),
                         'construction': construction, 'empty': empty,
                         'base_shape': [0, 3] if empty else [2, 3],
                         'alias_expression': expression,
                         'mutations': ['alias.fill_(INT64_MIN)', 'base.add_(1)']})
    assert len(rows) == len({r['name'] for r in rows}) == 8
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'lock', 'repo', 'parent-observations', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists(), 'Output is create-only; preserve prior observations'
    paths = [Path(__file__).resolve(), args.reference.resolve(), args.lock.resolve(),
             args.parent_observations.resolve()]
    inputs = {p: p.read_bytes() for p in paths}
    assert sha(inputs[args.reference.resolve()]) == SUPPORT_SHA256
    assert sha(inputs[args.lock.resolve()]) == LOCK_SHA256
    assert sha(inputs[args.parent_observations.resolve()]) == PARENT54_SHA256
    common = {'schema': 'heptatorch.pinned-int64-alias-version-observation.v1',
              'probe_sha256': sha(inputs[Path(__file__).resolve()]),
              'primary_source_ref': PRIMARY_SOURCE_REF,
              'parent54_observation_sha256': PARENT54_SHA256,
              'expected_cases': 8, 'case_plan': plan(),
              'comparison_policy': 'OBSERVATION ONLY: no success, error, alias, Version or numerical outcome is substituted from a model.',
              'native_executed': False, 'native_values_compared': 0,
              'pytorch_api_denominator': 2590, 'whole_pytorch_gate': 'HOLD',
              'full_pytorch_rewrite_complete': False,
              'scope': 'Four exact alias constructions with nonempty/empty int64 roots, followed by scalar fill_ and scalar add_. Strong references are held intentionally; no weak/free lifetime or universal dtype claim.'}
    if args.prepare_only:
        report = {**common, 'status': 'PREPARED_NOT_RUN', 'torch_executed': False,
                  'cases': 0, 'rows': [],
                  'inputs_sha256': {str(p): sha(b) for p, b in inputs.items()}}
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'status': report['status'], 'expected_cases': 8,
                          'probe_sha256': common['probe_sha256']}))
        return 0

    assert Path.cwd().resolve() == args.repo.resolve(), 'Use the real locked repo and installation receipt'
    for name in ('compatibility/upstream-lock.json', 'build/pytorch-install.json'):
        p = (args.repo / name).resolve()
        inputs[p] = p.read_bytes()
    previous_argv = sys.argv
    try:
        sys.argv = [str(args.reference), str(args.repo / 'reference-only-no-native-execution'),
                    str(args.output.with_suffix('.provenance.json')), str(args.lock)]
        context = runpy.run_path(str(args.reference), run_name='int64_alias_provenance_only')
    finally:
        sys.argv = previous_argv
    torch, provenance = context['torch'], context['PROVENANCE']
    assert provenance['status'] == 'PASS' and set(provenance['checks']) == CHECKS
    assert all(value is True for value in provenance['checks'].values())
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
        logical = bytes(value.detach().contiguous().reshape(-1).view(torch.uint8).tolist())
        storage = value.untyped_storage()
        storage_raw = bytes(storage)
        base = value._base
        node = value.grad_fn
        return {'python_identity': str(id(value)), 'tensor_identity': str(value._cdata),
                'storage_identity': str(storage._cdata), 'dtype': str(value.dtype),
                'element_size': value.element_size(), 'numel': value.numel(),
                'shape': list(value.shape), 'strides': list(value.stride()),
                'storage_offset_elements': value.storage_offset(), 'storage_bytes': storage.nbytes(),
                'version': int(value._version), 'requires_grad': bool(value.requires_grad),
                'is_leaf': bool(value.is_leaf), 'is_view': bool(value._is_view()),
                'base_tensor_identity': None if base is None else str(base._cdata),
                'grad_fn': None if node is None else type(node).__name__,
                'logical_bytes': len(logical), 'logical_bytes_hex': logical.hex(),
                'storage_bytes_hex': storage_raw.hex(),
                'signed_values_decimal': [str(int(x)) for x in value.detach().contiguous().reshape(-1).tolist()]}

    def snapshot(roots):
        return {name: tensor_info(value) for name, value in roots.items()}

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
        roots = {'base': None, 'alias': None}
        events = []
        before_base = rng_state()
        factory = (lambda: torch.empty((0, 3), dtype=torch.int64)) if spec['empty'] else (
            lambda: torch.tensor([[MINIMUM, MAXIMUM, -1], [0, 1, 3]], dtype=torch.int64))
        roots['base'], base_event = capture(factory)
        after_base = rng_state()
        initial_base = snapshot(roots)
        if roots['base'] is None:
            return {'base_construction': base_event, 'initial_base': initial_base,
                    'state_before_base': before_base, 'state_after_base': after_base,
                    'alias_construction': {'outcome': 'SKIPPED_BASE_UNAVAILABLE'},
                    'events': events, 'final': snapshot(roots)}
        base = roots['base']
        form = spec['construction']
        if form == 'detach':
            make_alias = lambda: base.detach()
        elif form == 'data-alias':
            make_alias = lambda: base.data
        elif form == 'as-strided':
            make_alias = (lambda: base.as_strided((2, 0, 3), (17, 101, 2), 7)) if spec['empty'] else (
                lambda: base.as_strided((2, 2), (3, 1), 1))
        elif form == 'clone-transpose':
            make_alias = lambda: base.transpose(0, 1).clone(memory_format=torch.preserve_format)
        else:
            raise AssertionError(form)
        before_alias = rng_state()
        roots['alias'], alias_event = capture(make_alias)
        after_alias = rng_state()
        constructed = snapshot(roots)

        def mutate(name, operation):
            state_before_observer = rng_state()
            before = snapshot(roots)
            state_before_call = rng_state()
            result, event = capture(operation)
            state_after_call = rng_state()
            returned = tensor_info(result)
            after = snapshot(roots)
            state_after_observer = rng_state()
            return {'name': name, 'before': before, 'operation': event,
                    'returned': returned, 'after': after,
                    'state_before_observer': state_before_observer,
                    'state_before_call': state_before_call,
                    'state_after_call': state_after_call,
                    'state_after_observer': state_after_observer,
                    'state_unchanged_by_operation': state_before_call == state_after_call,
                    'observer_state_unchanged_before': state_before_observer == state_before_call,
                    'observer_state_unchanged_after': state_after_call == state_after_observer}

        if roots['alias'] is not None:
            events.append(mutate('alias.fill_(INT64_MIN)', lambda: roots['alias'].fill_(MINIMUM)))
        else:
            events.append({'name': 'alias.fill_(INT64_MIN)', 'operation': {'outcome': 'SKIPPED_ALIAS_UNAVAILABLE'},
                           'before': snapshot(roots), 'after': snapshot(roots), 'state': rng_state()})
        events.append(mutate('base.add_(1)', lambda: roots['base'].add_(1)))
        return {'base_construction': base_event, 'initial_base': initial_base,
                'state_before_base': before_base, 'state_after_base': after_base,
                'alias_construction': alias_event, 'after_alias_construction': constructed,
                'state_before_alias': before_alias, 'state_after_alias': after_alias,
                'events': events, 'final': snapshot(roots),
                'strong_observer_roots': ['base', 'alias'], 'weak_lifetime_claimed': False,
                'identity_scope': 'Exact object identities within this recipe only; addresses may be reused between recipes.'}

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
              'cpu_capability': torch.backends.cpu.get_cpu_capability(), 'threads': torch.get_num_threads(),
              'inputs_sha256': {str(p): sha(b) for p, b in inputs.items()}}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'cases': len(rows), 'probe_error_count': len(errors),
                      'output_bytes': args.output.stat().st_size, 'output_sha256': sha(args.output.read_bytes())}))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
