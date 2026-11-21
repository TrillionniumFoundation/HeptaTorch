#!/usr/bin/env python3
"""Observe direct F64 view aliases using the exact locked real Torch wheel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import traceback
import weakref

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
from f64_view.cases import (CASES, CASE_COUNT, CASE_SHA256, CATEGORY_COUNTS, raw_input,
                            ALLOCATION_SPEC, ALLOCATION_SHA256, KNOWN_INCOMPATIBILITY_INDEX,
                            ORIGINAL_CASE_SHA256, CASE_INPUT_SHA256)
from numeric_predicates.reference import tensor, identity
import pinned_reference_custody as custody

SUPPORT = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
STATUS = 'OBSERVED_F64_VIEW_ORIGINAL188'
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def setup(torch, spec):
    raw = raw_input(spec)
    flat = torch.frombuffer(bytearray(raw), dtype=torch.float64).clone() if raw else torch.empty(0, dtype=torch.float64)
    return flat.reshape(spec['shape']).clone(memory_format=torch.contiguous_format)


def assign_bits(torch, value, index, word):
    assert 0 <= index < value.numel()
    coordinates = []
    for size in reversed(value.shape):
        coordinates.append(index % size)
        index //= size
    value.view(torch.int64)[tuple(reversed(coordinates))] = word if word < (1 << 63) else word - (1 << 64)


def error_matches(error, category, operation):
    message = str(error).lower()
    if type(error) is not RuntimeError:
        return False
    if category == 'storage':
        return 'storage' in message and ('bounds' in message or 'size' in message)
    if category == 'negative-stride':
        return 'negative' in message and 'stride' in message
    if category == 'negative-offset':
        return 'storage offset' in message and ('invalid' in message or 'negative' in message)
    if category == 'negative-size':
        # Full actual fixed-wheel errors from diagnostic 338b464966645d2d... .
        # These two exact requests remain refusals; the empty negative request
        # is separately retained as an observed incompatibility.
        errors = {
            ((-1,), (1,)): 'Storage size calculation overflowed with sizes=[-1] and strides=[1]',
            ((1, -1, 1, 1), (1, 1, 1, 1)):
                'Storage size calculation overflowed with sizes=[1, -1, 1, 1] and strides=[1, 1, 1, 1]',
        }
        if (set(operation) != {'operation', 'name', 'source', 'shape', 'strides', 'offset'}
                or operation['operation'] != 'view' or operation['name'] != 'view'
                or operation['source'] != 'root' or operation['offset'] != 0):
            return False
        expected = errors.get((tuple(operation['shape']), tuple(operation['strides'])))
        return expected is not None and str(error) == expected
    return False


def observe_known_incompatibility(torch, spec, out, index):
    assert index == KNOWN_INCOMPATIBILITY_INDEX and spec == CASES[index]
    root = setup(torch, spec['base'])
    before = tensor(torch, root, out)
    root_id, storage_id = root._cdata, root.untyped_storage()._cdata
    root_weak = weakref.ref(root)
    base_stage = dict(label='base', tensors={'root': before},
        aliases={'root': dict(shared_original_storage=root.untyped_storage()._cdata == storage_id,
                             original_root_base=root._base is None)},
        actual_Torch_Tensor_liveness={'root': root_weak() is not None})
    operation = spec['operations'][0]
    child = root.as_strided(operation['shape'], operation['strides'], operation['offset'])
    child_weak = weakref.ref(child)
    storage = bytes(child.untyped_storage())
    storage_path = out / 'blobs' / (sha(storage) + '.bin')
    if not storage_path.exists():
        storage_path.write_bytes(storage)
    actual = dict(dtype=str(child.dtype), shape=list(child.shape), strides=list(child.stride()),
        numel=child.numel(), element_size=child.element_size(), storage_offset=child.storage_offset(),
        storage_bytes=child.untyped_storage().nbytes(), version=child._version,
        requires_grad=child.requires_grad, is_leaf=child.is_leaf, is_view=child._is_view(),
        history_present=child.grad_fn is not None, retains_grad=child.retains_grad,
        actual_Torch_tolist=child.tolist(),
        storage=dict(path='blobs/' + storage_path.name, bytes=len(storage), sha256=sha(storage)),
        fresh_Tensor=child._cdata != root_id,
        shared_Storage=child.untyped_storage()._cdata == storage_id,
        original_root_base=child._base is not None and child._base._cdata == root_id)
    after = tensor(torch, root, out)
    assert before == after and actual['fresh_Tensor'] and actual['shared_Storage'] and actual['original_root_base']
    liveness = [dict(stage='root-and-view', root=root_weak() is not None, view=child_weak() is not None)]
    del root
    liveness.append(dict(stage='public-root-released', root=root_weak() is not None, view=child_weak() is not None))
    del child
    liveness.append(dict(stage='view-released', root=root_weak() is not None, view=child_weak() is not None))
    assert not liveness[-1]['root'] and not liveness[-1]['view']
    return dict(index=index, spec=spec, accepted=True,
        known_incompatibility=spec['known_incompatibility'], actual_Torch_operation_returned=True,
        stages=[base_stage],
        boundary_result=actual, original_input_after=after, original_inputs_unchanged_after_view=True,
        actual_Torch_lifetimes=liveness, all_Torch_Tensors_released=True,
        rng_unchanged=True, host_expected_values_computed=False)


def observe_case(torch, spec, out, index):
    if index == KNOWN_INCOMPATIBILITY_INDEX:
        return observe_known_incompatibility(torch, spec, out, index)
    values = {'root': setup(torch, spec['base'])}
    root_identity = values['root']._cdata
    storage_identity = values['root'].untyped_storage()._cdata
    weak = {'root': weakref.ref(values['root'])}

    def snapshot(label):
        tensors = {name: tensor(torch, value, out) for name, value in values.items()}
        aliases = {name: dict(shared_original_storage=value.untyped_storage()._cdata == storage_identity,
                              original_root_base=(value._base is None if name == 'root'
                                  else value._base is not None and value._base._cdata == root_identity))
                   for name, value in values.items()}
        assert all(row['shared_original_storage'] and row['original_root_base'] for row in aliases.values())
        assert len({value._version for value in values.values()}) <= 1
        return dict(label=label, tensors=tensors, aliases=aliases,
                    actual_Torch_Tensor_liveness={name: item() is not None for name, item in weak.items()})

    stages = [snapshot('base')]
    events = []
    for at, operation in enumerate(spec['operations']):
        name = operation['name']
        label = '%02d-%s-%s' % (at, operation['operation'], name)
        if operation['operation'] == 'view':
            before = {key: tensor(torch, value, out) for key, value in values.items()}
            try:
                result = values[operation['source']].as_strided(
                    operation['shape'], operation['strides'], operation['offset'])
            except RuntimeError as error:
                assert spec.get('expected_error') and error_matches(error, spec['expected_error'], operation), str(error)
                assert {key: tensor(torch, value, out) for key, value in values.items()} == before
                return dict(index=index, spec=spec, accepted=False, stages=stages,
                    error=dict(type=type(error).__name__, message=str(error)),
                    original_inputs_unchanged_after_error=True, events=events,
                    rng_unchanged=True, host_expected_values_computed=False)
            assert not spec.get('expected_error'), 'expected actual Torch geometry error was accepted'
            assert name not in values and result._cdata not in [value._cdata for value in values.values()]
            assert result._base is not None and result._base._cdata == root_identity
            assert result.untyped_storage()._cdata == storage_identity
            assert {key: tensor(torch, value, out) for key, value in values.items()} == before
            values[name] = result
            weak[name] = weakref.ref(result)
            del result
            events.append(dict(operation=operation, fresh_Tensor=True, shared_Storage=True,
                               original_root_base=True, prior_aliases_unchanged=True))
        elif operation['operation'] == 'write':
            before = {key: value._version for key, value in values.items()}
            assign_bits(torch, values[name], operation['index'], operation['word'])
            assert {key: value._version for key, value in values.items()} == {key: value + 1 for key, value in before.items()}
            events.append(dict(operation=operation, all_live_alias_versions_incremented_once=True))
        else:
            assert operation['operation'] == 'release' and name in values
            values.pop(name)
            events.append(dict(operation=operation, public_Tensor_reference_released=True))
        stages.append(snapshot(label))
    assert not values and all(item() is None for item in weak.values())
    return dict(index=index, spec=spec, accepted=True, stages=stages, events=events,
                all_Torch_Tensors_released=True, explicit_mutations_observed=True,
                rng_unchanged=True, host_expected_values_computed=False)


def observe_allocation(torch, out):
    """Actual Torch alias family for one fixed native allocation/recovery request."""
    spec = ALLOCATION_SPEC
    values = {'root': setup(torch, spec['base'])}
    root_identity = values['root']._cdata
    storage_identity = values['root'].untyped_storage()._cdata
    weak = {'root': weakref.ref(values['root'])}

    def snapshot(label):
        return dict(label=label, tensors={name: tensor(torch, value, out) for name, value in values.items()},
            aliases={name: dict(shared_original_storage=value.untyped_storage()._cdata == storage_identity,
                original_root_base=(value._base is None if name == 'root' else
                    value._base is not None and value._base._cdata == root_identity))
                for name, value in values.items()},
            actual_Torch_Tensor_liveness={name: item() is not None for name, item in weak.items()})

    stages = [snapshot('base')]
    for index in range(spec['views']):
        name = 'alias-%03d' % index
        value = values['root'].as_strided(spec['shape'], spec['strides'], spec['offset'])
        assert value._base is not None and value._base._cdata == root_identity
        assert value.untyped_storage()._cdata == storage_identity and value._version == 0
        assert value._cdata not in [item._cdata for item in values.values()]
        values[name] = value
        weak[name] = weakref.ref(value)
        del value
    stages.append(snapshot('all-views'))
    values.pop('root')
    stages.append(snapshot('release-root'))
    values.clear()
    stages.append(snapshot('release-all-views'))
    assert all(item() is None for item in weak.values())
    return dict(spec=spec, spec_sha256=ALLOCATION_SHA256, stages=stages,
                actual_Torch_views=spec['views'], all_Torch_Tensors_released=True,
                host_expected_values_computed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'blobs').mkdir()
    assert Path.cwd().resolve() == ROOT and not sys.flags.optimize
    files = [Path(__file__), ROOT / 'tests/f64_view/__init__.py', ROOT / 'tests/f64_view/cases.py',
             ROOT / 'tests/numeric_predicates/reference.py', ROOT / 'tests/numeric_predicates/cases.py',
             ROOT / 'tests/numeric_predicates/__init__.py', ROOT / 'tests/pytorch_oracle.py',
             ROOT / 'compatibility/pytorch-oracle-lock.json', ROOT / 'compatibility/upstream-lock.json',
             ROOT / 'build/pytorch-install.json', Path(sys.executable).resolve()]
    files += [ROOT / 'tests/pinned_reference_custody.py',
              ROOT / 'compatibility/pytorch-reference-requirements.txt',
              ROOT / 'build/pytorch-reference-support-install.json'] + custody.source_files(ROOT)
    from f64_view import family_reference
    files += family_reference.source_files()
    inputs = {str(path.resolve()): identity(path) for path in files}
    assert inputs[str(ROOT / 'tests/pytorch_oracle.py')]['sha256'] == SUPPORT
    assert inputs[str(ROOT / 'compatibility/pytorch-oracle-lock.json')]['sha256'] == LOCK
    inherited = os.environ.get('ATEN_CPU_CAPABILITY')
    capture = None
    report = dict(schema='heptatorch.f64-view.reference.v2', status='RUNNING', rows=[], failures=[],
        expected_cases=CASE_COUNT, case_sha256=CASE_SHA256, category_counts=CATEGORY_COUNTS,
        original_case_sha256=ORIGINAL_CASE_SHA256, case_input_sha256=CASE_INPUT_SHA256,
        inputs=inputs, Torch_executed=False, native_executed=False, compiler_executed=False,
        host_expected_values_computed=False, requested_cpu_capability='avx2',
        cpu_environment=None, inherited_cpu_capability=inherited,
        actual_process=dict(pid=os.getpid(), argv=list(sys.orig_argv), python=identity(sys.executable)),
        production_oracle_qualified=False, allow_unverified_local=False, reference_observations_only=True,
        historical_boundary_recipe_indices=[KNOWN_INCOMPATIBILITY_INDEX],
        historical_boundary_labels_are_not_current_native_status=True,
        full_project_status='HOLD', full_pytorch_rewrite_complete=False)
    try:
        # Capture the actual clean supervisor environment before setting either
        # capability/cache variable and before importing the pinned Torch wheel.
        capture = custody.begin(ROOT, out, __file__, False)
        report['cpu_environment'] = os.environ['ATEN_CPU_CAPABILITY']
        previous = sys.argv
        try:
            sys.argv = [str(ROOT / 'tests/pytorch_oracle.py'), str(out / 'NO_NATIVE_EXECUTION'),
                        str(out / 'oracle-provenance.json'), str(ROOT / 'compatibility/pytorch-oracle-lock.json')]
            context = runpy.run_path(str(ROOT / 'tests/pytorch_oracle.py'), run_name='f64_view_reference_only')
        finally:
            sys.argv = previous
        torch, provenance = context['torch'], context['PROVENANCE']
        assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11 and all(provenance['checks'].values())
        assert torch.backends.cpu.get_cpu_capability() == 'AVX2' and torch.get_num_threads() == 1
        custody.finish_capture(capture)
        report['reference_custody'] = custody.capture(ROOT, out, __file__, torch, provenance, False, capture)
        report['production_oracle_qualified'] = True
        report.update(Torch_executed=True, oracle_provenance=provenance, cpu_capability='AVX2', torch_threads=1)
        rng = bytes(torch.get_rng_state().tolist())
        for index, spec in enumerate(CASES):
            row = observe_case(torch, spec, out, index)
            assert bytes(torch.get_rng_state().tolist()) == rng
            report['rows'].append(row)
            (out / ('case-%03d.json' % index)).write_text(json.dumps(row, sort_keys=True, indent=2, allow_nan=False) + '\n')
        report['allocation_observation'] = observe_allocation(torch, out)
        assert bytes(torch.get_rng_state().tolist()) == rng
        (out / 'allocation-observation.json').write_text(
            json.dumps(report['allocation_observation'], sort_keys=True, indent=2, allow_nan=False) + '\n')
        report['families'] = family_reference.observe_all(torch, out / 'families', inputs)
        assert bytes(torch.get_rng_state().tolist()) == rng
        custody.finish_reference(capture)
        assert all(identity(path) == inputs[str(path.resolve())] for path in files)
        report.update(status=STATUS, cases=len(report['rows']), inputs_unchanged=True,
                      rng_unchanged=True, rng_sha256=sha(rng),
                      actual_Torch_acceptances=sum(row['accepted'] for row in report['rows']),
                      actual_Torch_geometry_errors=sum(not row['accepted'] for row in report['rows']),
                      known_incompatibility_recipes=[KNOWN_INCOMPATIBILITY_INDEX])
    except BaseException as error:
        if capture is not None:
            custody.finish_capture(capture)
        report.update(status='FAIL', production_oracle_qualified=False,
                      error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    (out / 'report.json').write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report.get(key) for key in ('status', 'cases', 'expected_cases', 'error')}), flush=True)
    return 0 if report['status'] == STATUS else 1


if __name__ == '__main__':
    raise SystemExit(main())
