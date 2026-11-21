"""Read every family observation under the same complete strict188 custody.

Values are original Torch observations. This module does not execute Torch,
native code, or a compiler, and does not manufacture reference outputs.
"""
import hashlib
import inspect
import json
from pathlib import Path
import struct

from . import evidence, family_reference, reference
from .cases import CASES as ORIGINAL_CASES, raw_input
from .family_cases import (EMPTY_CASES, EMPTY_CASE_SHA256, WIDE_CASES, WIDE_CASE_SHA256,
    OFFSET_CASES, OFFSET_CASE_SHA256, CHAINS, CHAIN_SHA256,
    EMPTY_ALLOCATION_SPEC, EMPTY_ALLOCATION_SHA256, HIGH_ALLOCATION_SPEC, HIGH_ALLOCATION_SHA256)

require = evidence.require
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def _metadata(value, blob, shape, strides, offset, count, version, view, dtype='torch.float64', storage_bytes=None):
    fields = {'dtype', 'shape', 'strides', 'numel', 'element_size', 'storage_bytes', 'storage_offset',
              'version', 'requires_grad', 'is_leaf', 'is_view', 'history_present', 'retains_grad',
              'logical', 'storage'}
    width = 8 if dtype == 'torch.float64' else 4
    storage_bytes = 2 * width if storage_bytes is None else storage_bytes
    require(set(value) == fields and value['dtype'] == dtype
            and value['shape'] == list(shape) and value['strides'] == list(strides)
            and value['numel'] == count and value['element_size'] == width
            and value['storage_offset'] == offset and value['version'] == version
            and value['requires_grad'] is False and value['history_present'] is False
            and value['retains_grad'] is False and value['is_leaf'] is True and value['is_view'] is view,
            'exact independently observed shape/stride/offset/Version/no-AD metadata')
    logical, storage = blob(value['logical']), blob(value['storage'])
    require(value['storage_bytes'] == len(storage) == storage_bytes and len(logical) == count * width,
            'complete actual logical and original Storage bytes')
    return storage


def _stage(stage, label, public, known, geometries, version, blob):
    require(set(stage) == {'label', 'tensors', 'aliases', 'actual_Torch_Tensor_liveness'}
            and stage['label'] == label and set(stage['tensors']) == set(stage['aliases']) == set(public)
            and set(stage['actual_Torch_Tensor_liveness']) == set(known),
            'complete ordered actual Tensor/alias/weak-reference inventories')
    alive = set(public)
    if public:
        alive.add('root')
    require(stage['actual_Torch_Tensor_liveness'] == {name: name in alive for name in known}
            and all(type(value) is bool for value in stage['actual_Torch_Tensor_liveness'].values()),
            'public root retention and exact actual weak-reference lifetime of every alias')
    storages = []
    for name, value in stage['tensors'].items():
        require(stage['aliases'][name] == dict(shared_original_storage=True, original_root_base=True),
                'actual shared original Storage and original root-base relationship')
        shape, strides, offset, count = geometries[name]
        storages.append(_metadata(value, blob, shape, strides, offset, count, version, name != 'root',
                                  storage_bytes=geometries['root'][3] * 8))
    require(not storages or all(raw == storages[0] for raw in storages),
            'all live aliases expose the complete same actual Storage bytes')


def _single(row, spec, index, blob, original187=False):
    require(row['index'] == index and row['spec'] == spec and type(row['accepted']) is bool
            and row['host_expected_values_computed'] is False,
            'literal original family input, original index and actual observed outcome')
    stages = row['stages']
    base_count = len(raw_input(spec['base'])) // 8
    require(spec['base']['shape'] == [base_count], 'complete literal one-dimensional family root')
    geometry = {'root': (spec['base']['shape'], [1], 0, base_count)}
    _stage(stages[0], 'base', {'root'}, {'root'}, geometry, 0, blob)
    base = stages[0]['tensors']['root']
    require(blob(base['logical']) == blob(base['storage']) == raw_input(spec['base']),
            'the exact original signed-zero/F64 input words were observed')
    if not row['accepted']:
        error = row['error']
        require(not original187 and len(stages) == 1 and set(error) == {'type', 'message'}
                and error['type'] in ('RuntimeError', 'OverflowError')
                and isinstance(error['message'], str) and bool(error['message'])
                and row['original_input_unchanged_after_error'] is True,
                'actual retained complete Torch error and unchanged original input')
        return
    operation = spec['operations'][0]
    require(operation['operation'] == 'view' and operation['name'] == 'view' and operation['source'] == 'root'
            and len(operation['shape']) == len(operation['strides']) <= 4,
            'one exact original view operation precedes supplemental observations')
    geometry['view'] = (operation['shape'], operation['strides'], operation['offset'], 0)
    _stage(stages[1], 'view', {'root', 'view'}, {'root', 'view'}, geometry, 0, blob)
    require(stages[1]['tensors']['root'] == base and blob(stages[1]['tensors']['view']['logical']) == b'',
            'view creation leaves the actual root unchanged and observes genuinely empty bytes')
    if original187:
        require(len(stages) == 4 and row['original_recipe_unchanged'] is True
                and row['supplemental_cleanup_only'] is True and row['all_Torch_Tensors_released'] is True,
                'original187 input stays intact; only real cleanup observations are supplemental')
        _stage(stages[2], 'release-root', {'view'}, {'root', 'view'}, geometry, 0, blob)
        _stage(stages[3], 'release-view', set(), {'root', 'view'}, geometry, 0, blob)
        return
    require(len(stages) == 8 and row['actual_empty_fill_shared_version_bump'] is True
            and row['original_root_retained_after_public_release'] is True
            and row['actual_nonempty_child_recreated_from_empty_view'] is True
            and row['all_Torch_Tensors_released'] is True,
            'all actual empty mutation, root release, child creation/mutation and lifetimes observed')
    _stage(stages[2], 'empty-fill', {'root', 'view'}, {'root', 'view'}, geometry, 1, blob)
    require(blob(stages[2]['tensors']['view']['storage']) == blob(base['storage']),
            'actual empty fill changes Version while original Storage bytes remain unchanged')
    _stage(stages[3], 'release-root', {'view'}, {'root', 'view'}, geometry, 1, blob)
    geometry['child'] = ([2], [1], 0, 2)
    known = {'root', 'view', 'child'}
    _stage(stages[4], 'recreate-nonempty', {'view', 'child'}, known, geometry, 1, blob)
    _stage(stages[5], 'child-write', {'view', 'child'}, known, geometry, 2, blob)
    _stage(stages[6], 'release-view', {'child'}, known, geometry, 2, blob)
    _stage(stages[7], 'release-child', set(), known, geometry, 2, blob)


def _chain(row, spec, index, blob):
    require(row['index'] == index and row['spec'] == spec and row['accepted'] is True
            and row['host_expected_values_computed'] is False
            and all(row[key] is True for key in (
                'exact_high_and_ordinary_offsets_observed', 'shared_version_each_empty_fill',
                'public_root_released_before_new_high_and_ordinary_aliases',
                'nonempty_child_recreated_after_public_root_release', 'all_Torch_Tensors_released')),
            'complete actual alias chain, Version, original-root retention and lifetime observations')
    stages = row['stages']
    public, known = {'root'}, {'root'}
    geometry = {'root': ([2], [1], 0, 2)}
    at, version = 0, 0
    def consume(label):
        nonlocal at
        require(at < len(stages), 'each actual chain event has an observed stage')
        _stage(stages[at], label, public, known, geometry, version, blob)
        at += 1
    consume('base')
    base = stages[0]['tensors']['root']
    require(blob(base['logical']) == blob(base['storage']) == raw_input(spec['base']),
            'chain starts with the exact original input words')
    for index, offset in enumerate(spec['offsets']):
        name = 'alias%d' % index
        public.add(name)
        known.add(name)
        geometry[name] = (spec['shape'], spec['strides'], offset, 0)
        consume('create-' + name)
        version += 1
        consume('fill-' + name)
        require(blob(stages[at - 1]['tensors'][name]['storage']) == blob(base['storage']),
                'each actual empty fill preserves all original Storage bytes')
        if index == 0:
            public.remove('root')
            consume('release-root')
    public.add('child')
    known.add('child')
    geometry['child'] = ([2], [1], 0, 2)
    consume('recreate-nonempty')
    version += 1
    consume('child-write')
    for index in (2, 0, 4, 1, 3):
        name = 'alias%d' % index
        public.remove(name)
        consume('release-' + name)
    public.remove('child')
    consume('release-child')
    require(at == len(stages), 'every original chain observation consumed exactly once')


def _allocation(row, spec, digest, blob):
    require(row['spec'] == spec and row['spec_sha256'] == digest
            and row['actual_Torch_views'] == spec['views'] == 256
            and row['all_Torch_Tensors_released'] is True and row['host_expected_values_computed'] is False
            and len(row['stages']) == 4,
            'all original256 real aliases and complete allocation-family observations')
    names = {'alias-%03d' % index for index in range(spec['views'])}
    geometry = {name: (spec['shape'], spec['strides'], spec['offset'], 0) for name in names}
    geometry['root'] = ([2], [1], 0, 2)
    for stage, label, public, known in zip(row['stages'],
            ('base', 'all-views', 'release-root', 'release-all-views'),
            ({'root'}, names | {'root'}, names, set()),
            ({'root'}, names | {'root'}, names | {'root'}, names | {'root'})):
        _stage(stage, label, public, known, geometry, 0, blob)
    base = row['stages'][0]['tensors']['root']
    require(blob(base['logical']) == blob(base['storage']) == raw_input(spec['base']),
            'original256 allocation family starts with the complete literal input')


def _read_observations(directory, inputs):
    """Pure complete family data checks; strict qualification lives only below."""
    directory = Path(directory).resolve(strict=True)
    raw = inputs.bind(directory / 'report.json')
    report = json.loads(raw)
    require(report['schema'] == 'heptatorch.f64-view.families-reference.v1'
            and report['status'] == family_reference.STATUS
            and report['parent_strict_reference_required'] is True
            and report['independent_reference_process'] is False
            and report['reference_observations_only'] is True
            and report['host_expected_values_computed'] is False
            and report['Torch_executed'] is True and report['native_executed'] is False
            and report['compiler_executed'] is False and report['rng_unchanged'] is True
            and report['all_bound_inputs_unchanged'] is True
            and report['original188_inputs_unchanged'] is True
            and report['historical_capacity_gap_labels_are_not_current_native_status'] is True
            and report['all_empty_logical_reads_use_actual_zero_length_alias'] is True
            and report['no_empty_tensor_clone'] is True and report['no_element_iteration_for_empty_shapes'] is True
            and report['cpu_capability'] == 'AVX2' and report['torch_threads'] == 1
            and report['full_project_status'] == 'HOLD' and report['full_pytorch_rewrite_complete'] is False,
            'complete actually observed bounded families under the original strict process')
    require(report['observer_source_relations'] == family_reference.source_relations(),
            'every executed actual observer retains its exact function/source inverse')
    source_inputs = {str(path): reference.identity(path) for path in family_reference.source_files()}
    require(report['inputs'] == source_inputs, 'complete actual family observer source identities')
    for path, identity in source_inputs.items():
        require(len(inputs.bind(path, identity['sha256'])) == identity['bytes'], 'same complete actual family source input')
    used = set()
    def blob(record):
        require(set(record) == {'path', 'bytes', 'sha256'}
                and record['path'] == 'blobs/' + record['sha256'] + '.bin',
                'canonical actual complete observed family blob')
        value = inputs.bind(directory / record['path'], record['sha256'])
        require(len(value) == record['bytes'], 'all actual observed blob bytes')
        used.add(record['sha256'] + '.bin')
        return value
    expected_files = {'report.json', 'original187.json', 'empty-allocation.json',
                      'high-allocation.json', 'fault-setups.json', 'blobs'}
    for prefix, catalog, digest, acceptances, errors in (
            ('empty', EMPTY_CASES, EMPTY_CASE_SHA256, 606, 235),
            ('wide', WIDE_CASES, WIDE_CASE_SHA256, 13, 7),
            ('offset', OFFSET_CASES, OFFSET_CASE_SHA256, 55, 20)):
        rows = report[prefix + '_rows']
        require(report[prefix + '_cases'] == report['expected_' + prefix + '_cases'] == len(rows) == len(catalog)
                and report[prefix + '_case_sha256'] == digest
                and report['actual_' + prefix + '_acceptances'] == sum(row['accepted'] for row in rows) == acceptances
                and report['actual_' + prefix + '_errors'] == sum(not row['accepted'] for row in rows) == errors,
                'unchanged catalog, complete denominator and actual accepted/error census: ' + prefix)
        for index, (row, spec) in enumerate(zip(rows, catalog)):
            name = '%s-%04d.json' % (prefix, index)
            expected_files.add(name)
            require(row == json.loads(inputs.bind(directory / name)), 'complete original physical observed row: ' + name)
            _single(row, spec, index, blob)
    original187 = report['original187']
    require(original187 == json.loads(inputs.bind(directory / 'original187.json')),
            'literal separately retained original187 success observation')
    _single(original187, ORIGINAL_CASES[187], 187, blob, original187=True)
    require(report['chain_sha256'] == CHAIN_SHA256 and len(report['chains']) == report['expected_chains'] == len(CHAINS) == 3,
            'complete unchanged three literal high/ordinary alias chains')
    for index, (row, spec) in enumerate(zip(report['chains'], CHAINS)):
        name = 'chain-%02d.json' % index
        expected_files.add(name)
        require(row == json.loads(inputs.bind(directory / name)), 'complete physical observed alias chain')
        _chain(row, spec, index, blob)
    for prefix, spec, digest in (('empty', EMPTY_ALLOCATION_SPEC, EMPTY_ALLOCATION_SHA256),
                                 ('high', HIGH_ALLOCATION_SPEC, HIGH_ALLOCATION_SHA256)):
        row = report[prefix + '_allocation']
        require(row == json.loads(inputs.bind(directory / (prefix + '-allocation.json')))
                and report[prefix + '_allocation_spec_sha256'] == digest,
                'complete separately retained actual256 empty allocation observation')
        _allocation(row, spec, digest, blob)
    body_sha = sha(inspect.getsource(reference.observe_allocation).encode())
    require(report['original_allocation_observer_body_sha256'] == body_sha
            and report['empty_allocation']['original_observe_allocation_body_sha256'] == body_sha,
            'all actual allocation observations executed the complete original function body')
    setups = dict(f32_fault_setup=report['f32_fault_setup'], nonempty_fault_setup=report['nonempty_fault_setup'])
    require(setups == json.loads(inputs.bind(directory / 'fault-setups.json')), 'complete physical special fault setup observations')
    for key, dtype, width, empty in (('f32_fault_setup', 'torch.float32', 4, True),
                                      ('nonempty_fault_setup', 'torch.float64', 8, False)):
        row = report[key]
        require(set(row) == {'root', 'view'}, 'actual original root and special corruption target')
        root = _metadata(row['root'], blob, [2], [1], 0, 2, 0, False, dtype)
        view = _metadata(row['view'], blob, [0] if empty else [2], [1], 0, 0 if empty else 2, 0, True, dtype)
        raw_input_words = struct.pack('<II', 0x3f800000, 0x80000000) if width == 4 else raw_input(OFFSET_CASES[0]['base'])
        require(root == view == blob(row['root']['logical']) == raw_input_words,
                'complete literal F32/nonempty F64 diagnostic setup bytes and shared Storage')
    require({path.name for path in directory.iterdir()} == expected_files
            and {path.name for path in (directory / 'blobs').iterdir()} == used,
            'exact full family file/blob inventories with no omitted or additional observations')
    result = inputs.recheck()
    require(result is None or result == [], 'all actual family source and observed input bytes remain unchanged')
    return dict(report=report, report_sha256=sha(raw), blob=blob, root=directory)


def read_families(directory, execution, inputs):
    """Qualify families only after rereading the actual full original188 custody."""
    directory = Path(directory).resolve(strict=True)
    original = evidence.read_reference(directory, execution, inputs)
    require(original['strict_reference_qualified'] is True, 'full strict original188 reference is mandatory')
    parent = original['report']
    record = parent['families']
    require(set(record) == {'path', 'bytes', 'sha256'} and record['path'] == 'families/report.json',
            'canonical separately retained family report in the same actual reference process')
    raw = inputs.bind(directory / record['path'], record['sha256'])
    require(len(raw) == record['bytes'], 'complete same-process family report binding')
    result = _read_observations(directory / 'families', inputs)
    report = result['report']
    require(result['report_sha256'] == record['sha256']
            and report['actual_process'] == parent['actual_process']
            and report['rng_sha256'] == parent['rng_sha256']
            and report['actual_Torch_version'] == parent['oracle_provenance']['observed']['version']
            and all(parent['inputs'][path] == row for path, row in report['inputs'].items()),
            'actual same interpreter/process/RNG/oracle/observer inputs as the complete strict188 capture')
    baseline = parent['rows'][187]
    current = report['original187']['stages'][1]['tensors']['view']
    boundary = baseline['boundary_result']
    require(baseline['accepted'] is True and all(value == boundary[key] for key, value in current.items() if key != 'logical')
            and report['original187']['stages'][0]['tensors']['root'] == baseline['stages'][0]['tensors']['root'],
            'the original188 accepted boundary metadata is preserved by its explicit success/lifetime supplement')
    result.update(strict_reference_qualified=True, original_reference=original,
                  original_reference_sha256=original['report_sha256'], custody=original['custody'])
    return result
