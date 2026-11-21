"""Actual Torch family observations inside the strict original188 process.

No separate oracle process or local-qualification bypass is provided here.
The numerical/alias/lifetime observers retain their original function bodies.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import weakref

from . import reference as original
from .cases import CASES as ORIGINAL_CASES
from .family_cases import (EMPTY_CASES, EMPTY_CASE_SHA256, WIDE_CASES, WIDE_CASE_SHA256,
    OFFSET_CASES, OFFSET_CASE_SHA256, CHAINS, CHAIN_SHA256, WORD,
    EMPTY_ALLOCATION_SPEC, EMPTY_ALLOCATION_SHA256, HIGH_ALLOCATION_SPEC, HIGH_ALLOCATION_SHA256)

sha = lambda raw: hashlib.sha256(raw).hexdigest()
STATUS = 'OBSERVED_F64_VIEW_FAMILIES'


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def tensor(torch, value, folder):
    """Keep original metadata; use a real zero-length alias only for empty bytes."""
    if value.numel() != 0:
        return original.tensor(torch, value, folder)
    assert value.dtype == torch.float64 and value.numel() == 0
    before = (value._cdata, tuple(value.shape), tuple(value.stride()), value.storage_offset(), value._version,
              value.untyped_storage()._cdata, value.untyped_storage().nbytes(), value._base._cdata)
    logical = bytes(value.as_strided((0,), (1,), 0).view(torch.uint8).tolist())
    storage = bytes(value.untyped_storage())
    after = (value._cdata, tuple(value.shape), tuple(value.stride()), value.storage_offset(), value._version,
             value.untyped_storage()._cdata, value.untyped_storage().nbytes(), value._base._cdata)
    assert before == after and logical == b''
    def blob(raw):
        path = folder / 'blobs' / (sha(raw) + '.bin')
        if not path.exists():
            path.write_bytes(raw)
        return dict(path='blobs/' + path.name, bytes=len(raw), sha256=sha(raw))
    return dict(dtype=str(value.dtype), shape=list(value.shape), strides=list(value.stride()),
        numel=value.numel(), element_size=value.element_size(), storage_bytes=value.untyped_storage().nbytes(),
        storage_offset=value.storage_offset(), version=value._version, requires_grad=value.requires_grad,
        is_leaf=value.is_leaf, is_view=value._is_view(), history_present=value.grad_fn is not None,
        retains_grad=value.retains_grad, logical=blob(logical), storage=blob(storage))


def observe(torch, spec, output, index, *, original187=False):
    values = {'root': original.setup(torch, spec['base'])}
    root_id = values['root']._cdata
    storage_id = values['root'].untyped_storage()._cdata
    weak = {'root': weakref.ref(values['root'])}
    def stage(label):
        tensors = {name: tensor(torch, value, output) for name, value in values.items()}
        aliases = {name: dict(shared_original_storage=value.untyped_storage()._cdata == storage_id,
            original_root_base=(value._base is None if name == 'root' else
                value._base is not None and value._base._cdata == root_id)) for name, value in values.items()}
        assert all(row['shared_original_storage'] and row['original_root_base'] for row in aliases.values())
        assert len({value._version for value in values.values()}) <= 1
        return dict(label=label, tensors=tensors, aliases=aliases,
                    actual_Torch_Tensor_liveness={name: value() is not None for name, value in weak.items()})
    stages = [stage('base')]
    operation = spec['operations'][0]
    before = stages[0]['tensors']['root']
    try:
        view = values['root'].as_strided(operation['shape'], operation['strides'], operation['offset'])
    except (RuntimeError, OverflowError) as error:
        assert tensor(torch, values['root'], output) == before
        return dict(index=index, spec=spec, accepted=False, stages=stages,
                    error=dict(type=type(error).__name__, message=str(error)),
                    original_input_unchanged_after_error=True, host_expected_values_computed=False)
    assert view._cdata != root_id and view._base is not None and view._base._cdata == root_id
    assert view.untyped_storage()._cdata == storage_id and view.numel() == 0
    values['view'] = view
    weak['view'] = weakref.ref(view)
    del view
    assert tensor(torch, values['root'], output) == before
    stages.append(stage('view'))
    if original187:
        # Original case187 has exactly one operation. Cleanup observations are
        # supplemental and never rewrite that historical recipe or its digest.
        values.pop('root'); stages.append(stage('release-root'))
        values.pop('view'); stages.append(stage('release-view'))
        assert all(value() is None for value in weak.values())
        return dict(index=index, spec=spec, accepted=True, stages=stages,
                    original_recipe_unchanged=True, supplemental_cleanup_only=True,
                    all_Torch_Tensors_released=True, host_expected_values_computed=False)
    versions = {name: value._version for name, value in values.items()}
    values['view'].fill_(42.0)
    assert {name: value._version for name, value in values.items()} == {name: old+1 for name, old in versions.items()}
    assert tensor(torch, values['view'], output)['storage'] == before['storage']
    stages.append(stage('empty-fill'))
    values.pop('root'); stages.append(stage('release-root'))
    child = values['view'].as_strided([2], [1], 0)
    assert child._base is not None and child._base._cdata == root_id
    assert child._cdata not in (root_id, values['view']._cdata)
    values['child'] = child
    weak['child'] = weakref.ref(child)
    del child
    stages.append(stage('recreate-nonempty'))
    original.assign_bits(torch, values['child'], 0, WORD)
    stages.append(stage('child-write'))
    values.pop('view'); stages.append(stage('release-view'))
    values.pop('child'); stages.append(stage('release-child'))
    assert all(value() is None for value in weak.values())
    return dict(index=index, spec=spec, accepted=True, stages=stages,
                actual_empty_fill_shared_version_bump=True, original_root_retained_after_public_release=True,
                actual_nonempty_child_recreated_from_empty_view=True,
                all_Torch_Tensors_released=True, host_expected_values_computed=False)


def allocation(torch, output):
    source = inspect.getsource(original.observe_allocation)
    context = dict(original.__dict__, ALLOCATION_SPEC=EMPTY_ALLOCATION_SPEC,
                   ALLOCATION_SHA256=EMPTY_ALLOCATION_SHA256, tensor=tensor)
    exec(compile(source, '<original allocation observer with explicit empty family>', 'exec'), context)
    result = context['observe_allocation'](torch, output)
    result['original_observe_allocation_body_sha256'] = sha(source.encode())
    return result


def observe_chain(torch,spec,output,index):
    values={'root':original.setup(torch,spec['base'])}
    root_id=values['root']._cdata;storage_id=values['root'].untyped_storage()._cdata
    weak={'root':weakref.ref(values['root'])}
    stages=[]
    def stage(label):
        tensors={name:tensor(torch,value,output) for name,value in values.items()}
        aliases={name:dict(shared_original_storage=value.untyped_storage()._cdata==storage_id,
            original_root_base=(value._base is None if name=='root' else value._base is not None and value._base._cdata==root_id))
            for name,value in values.items()}
        assert all(row['shared_original_storage'] and row['original_root_base'] for row in aliases.values())
        assert len({value._version for value in values.values()})<=1
        stages.append(dict(label=label,tensors=tensors,aliases=aliases,
            actual_Torch_Tensor_liveness={name:value() is not None for name,value in weak.items()}))
    stage('base');source='root'
    for at,offset in enumerate(spec['offsets']):
        name='alias%d'%at
        value=values[source].as_strided(spec['shape'],spec['strides'],offset)
        assert value.numel()==0 and value.storage_offset()==offset
        values[name]=value;weak[name]=weakref.ref(value);del value
        stage('create-'+name)
        versions={n:v._version for n,v in values.items()}
        values[name].fill_(42.0)
        assert {n:v._version for n,v in values.items()}=={n:v+1 for n,v in versions.items()}
        stage('fill-'+name)
        if at==0:
            values.pop('root');stage('release-root')
        source=name
    value=values[source].as_strided([2],[1],0)
    values['child']=value;weak['child']=weakref.ref(value);del value
    stage('recreate-nonempty')
    original.assign_bits(torch,values['child'],0,WORD);stage('child-write')
    for at in (2,0,4,1,3):
        name='alias%d'%at;values.pop(name);stage('release-'+name)
    values.pop('child');stage('release-child')
    assert all(value() is None for value in weak.values())
    return dict(index=index,spec=spec,accepted=True,stages=stages,host_expected_values_computed=False,
        exact_high_and_ordinary_offsets_observed=True,shared_version_each_empty_fill=True,
        public_root_released_before_new_high_and_ordinary_aliases=True,
        nonempty_child_recreated_after_public_root_release=True,all_Torch_Tensors_released=True)


_FUNCTION_RELATIONS = [{'source': 'empty_reference_v3.py', 'name': 'tensor', 'original_sha256': 'ef360c61624027a8fd6a15c255d283f7d94663ff2c8f2ec94aba86f4d2e7b0b9', 'derived_sha256': 'ef360c61624027a8fd6a15c255d283f7d94663ff2c8f2ec94aba86f4d2e7b0b9', 'replacements': [], 'whole_function_inverse_exact': True}, {'source': 'empty_reference_v3.py', 'name': 'observe', 'original_sha256': '83c2ff73b469d41867779e997c7e7e9334663806d49a0b5b4a9021834f4f1b35', 'derived_sha256': '83c2ff73b469d41867779e997c7e7e9334663806d49a0b5b4a9021834f4f1b35', 'replacements': [], 'whole_function_inverse_exact': True}, {'source': 'empty_reference_v3.py', 'name': 'allocation', 'original_sha256': 'ee36607a81d31228d6903346851e085664738d29cde90c90a99f61720373c537', 'derived_sha256': 'ee36607a81d31228d6903346851e085664738d29cde90c90a99f61720373c537', 'replacements': [], 'whole_function_inverse_exact': True}, {'source': 'offset_reference_v3.py', 'name': 'observe_chain', 'original_sha256': 'f620834bb9e342675800c6bfd5e5dcaed47527766e24fa39356886307d50d3c5', 'derived_sha256': '56e67d92cea6832f9541323cb08353fad568ac0c5ed5045412898f7030511dd6', 'replacements': [{'before': 'empty.tensor(', 'after': 'tensor('}], 'whole_function_inverse_exact': True}]

def source_relations():
    result = []
    for row in _FUNCTION_RELATIONS:
        derived = inspect.getsource(globals()[row['name']])
        assert sha(derived.encode()) == row['derived_sha256']
        restored = derived
        for edit in reversed(row['replacements']):
            assert edit['after'] in restored
            restored = restored.replace(edit['after'], edit['before'])
        assert sha(restored.encode()) == row['original_sha256']
        result.append(dict(row, independently_reconstructed_original_sha256=sha(restored.encode())))
    return dict(functions=result, numerical_metadata_alias_and_lifetime_predicates_unchanged=True,
                original_allocation_observer_body_sha256=sha(inspect.getsource(original.observe_allocation).encode()),
                actual_empty_bytes_observed_without_cloning_signed_shapes=True)


def source_files():
    return [Path(__file__).resolve(), Path(__file__).with_name('family_cases.py').resolve()]


def observe_all(torch, output, inputs):
    """Observe every unchanged family under the caller's original strict custody."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'blobs').mkdir()
    source_inputs = {str(path): original.identity(path) for path in source_files()}
    assert all(inputs[path] == record for path, record in source_inputs.items())
    assert not sys.flags.optimize and torch.backends.cpu.get_cpu_capability() == 'AVX2' and torch.get_num_threads() == 1
    rng = bytes(torch.get_rng_state().tolist())
    result = dict(schema='heptatorch.f64-view.families-reference.v1', status='RUNNING',
        empty_rows=[], wide_rows=[], offset_rows=[], chains=[],
        expected_empty_cases=841, expected_wide_cases=20, expected_offset_cases=75, expected_chains=3,
        empty_case_sha256=EMPTY_CASE_SHA256, wide_case_sha256=WIDE_CASE_SHA256,
        offset_case_sha256=OFFSET_CASE_SHA256, chain_sha256=CHAIN_SHA256,
        empty_allocation_spec_sha256=EMPTY_ALLOCATION_SHA256, high_allocation_spec_sha256=HIGH_ALLOCATION_SHA256,
        original188_inputs_unchanged=True, historical_capacity_gap_labels_are_not_current_native_status=True,
        actual_process=dict(pid=os.getpid(), argv=list(sys.orig_argv), python=original.identity(sys.executable)),
        parent_strict_reference_required=True, independent_reference_process=False,
        reference_observations_only=True, host_expected_values_computed=False,
        all_empty_logical_reads_use_actual_zero_length_alias=True,
        no_empty_tensor_clone=True, no_element_iteration_for_empty_shapes=True,
        Torch_executed=True, native_executed=False, compiler_executed=False,
        cpu_capability='AVX2', torch_threads=1, actual_Torch_version=torch.__version__,
        observer_source_relations=source_relations(), inputs=source_inputs,
        full_project_status='HOLD', full_pytorch_rewrite_complete=False)
    result['original187'] = observe(torch, ORIGINAL_CASES[187], output, 187, original187=True)
    assert result['original187']['accepted'] is True
    save(output / 'original187.json', result['original187'])
    for prefix, catalog, key in (('empty', EMPTY_CASES, 'empty_rows'),
                                  ('wide', WIDE_CASES, 'wide_rows'), ('offset', OFFSET_CASES, 'offset_rows')):
        for index, spec in enumerate(catalog):
            row = observe(torch, spec, output, index)
            assert bytes(torch.get_rng_state().tolist()) == rng
            result[key].append(row)
            save(output / ('%s-%04d.json' % (prefix, index)), row)
    for index, spec in enumerate(CHAINS):
        row = observe_chain(torch, spec, output, index)
        result['chains'].append(row)
        save(output / ('chain-%02d.json' % index), row)
        assert bytes(torch.get_rng_state().tolist()) == rng
    result['empty_allocation'] = allocation(torch, output)
    save(output / 'empty-allocation.json', result['empty_allocation'])
    source = inspect.getsource(original.observe_allocation)
    context = dict(original.__dict__, ALLOCATION_SPEC=HIGH_ALLOCATION_SPEC,
                   ALLOCATION_SHA256=HIGH_ALLOCATION_SHA256, tensor=tensor)
    exec(compile(source, '<unchanged original allocation observer with literal high empty offset>', 'exec'), context)
    result['high_allocation'] = context['observe_allocation'](torch, output)
    result['original_allocation_observer_body_sha256'] = sha(source.encode())
    save(output / 'high-allocation.json', result['high_allocation'])
    f32root=torch.tensor([1.0,-0.0],dtype=torch.float32)
    f32view=f32root.as_strided([0],[1],0)
    f32=dict(root=original.tensor(torch,f32root,output),view=original.tensor(torch,f32view,output))
    assert f32view._base is f32root and f32view.untyped_storage()._cdata==f32root.untyped_storage()._cdata
    del f32view,f32root
    f64root=original.setup(torch,OFFSET_CASES[0]['base'])
    f64view=f64root.as_strided([2],[1],0)
    nonempty_fault_setup=dict(root=original.tensor(torch,f64root,output),view=original.tensor(torch,f64view,output))
    assert f64view._base is f64root and f64view.untyped_storage()._cdata==f64root.untyped_storage()._cdata
    del f64view,f64root
    result.update(f32_fault_setup=f32, nonempty_fault_setup=nonempty_fault_setup)
    save(output / 'fault-setups.json', dict(f32_fault_setup=f32, nonempty_fault_setup=nonempty_fault_setup))
    assert bytes(torch.get_rng_state().tolist()) == rng
    assert all(original.identity(path) == record == inputs[path] for path, record in source_inputs.items())
    result.update(status=STATUS, rng_unchanged=True, rng_sha256=sha(rng), all_bound_inputs_unchanged=True,
        empty_cases=len(result['empty_rows']), wide_cases=len(result['wide_rows']), offset_cases=len(result['offset_rows']),
        actual_empty_acceptances=sum(row['accepted'] for row in result['empty_rows']),
        actual_empty_errors=sum(not row['accepted'] for row in result['empty_rows']),
        actual_wide_acceptances=sum(row['accepted'] for row in result['wide_rows']),
        actual_wide_errors=sum(not row['accepted'] for row in result['wide_rows']),
        actual_offset_acceptances=sum(row['accepted'] for row in result['offset_rows']),
        actual_offset_errors=sum(not row['accepted'] for row in result['offset_rows']))
    save(output / 'report.json', result)
    raw = (output / 'report.json').read_bytes()
    return dict(path='families/report.json', bytes=len(raw), sha256=sha(raw))
