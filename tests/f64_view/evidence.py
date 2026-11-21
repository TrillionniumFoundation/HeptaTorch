"""Read complete original F64 view observations without computing their values."""
import hashlib
import json
from pathlib import Path
import stat

from .cases import (CASES, CASE_COUNT, CASE_SHA256, CATEGORY_COUNTS,
                    ALLOCATION_SPEC, ALLOCATION_SHA256, KNOWN_INCOMPATIBILITY_INDEX,
                    ORIGINAL_CASE_SHA256, CASE_INPUT_SHA256)
from .reference import error_matches
import pinned_reference_custody as custody

ROOT = Path(__file__).resolve().parents[2]
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read_reference(directory, execution, inputs):
    directory, execution = Path(directory).resolve(strict=True), Path(execution).resolve(strict=True)
    report = json.loads(inputs.bind(directory / 'report.json'))
    process = json.loads(inputs.bind(execution / 'process.json'))
    require(report['schema'] == 'heptatorch.f64-view.reference.v2'
            and report['status'] == 'OBSERVED_F64_VIEW_ORIGINAL188'
            and report['cases'] == report['expected_cases'] == len(report['rows']) == CASE_COUNT
            and report['case_sha256'] == CASE_SHA256 and report['category_counts'] == CATEGORY_COUNTS
            and report['original_case_sha256'] == ORIGINAL_CASE_SHA256
            and report['case_input_sha256'] == CASE_INPUT_SHA256
            and report['actual_Torch_acceptances'] == 178 and report['actual_Torch_geometry_errors'] == 10
            and report['known_incompatibility_recipes'] == [KNOWN_INCOMPATIBILITY_INDEX]
            and report['failures'] == [] and report['Torch_executed'] is True
            and report['native_executed'] is False and report['compiler_executed'] is False
            and report['host_expected_values_computed'] is False and report['inputs_unchanged'] is True
            and report['rng_unchanged'] is True and report['cpu_capability'] == 'AVX2'
            and report['requested_cpu_capability'] == report['cpu_environment'] == 'avx2'
            and report['torch_threads'] == 1 and report['full_project_status'] == 'HOLD'
            and report['full_pytorch_rewrite_complete'] is False,
            'complete fixed-wheel F64 view observation and exact bounded scope')
    require(report['production_oracle_qualified'] is True and report['allow_unverified_local'] is False
            and report['reference_observations_only'] is True
            and report['historical_boundary_recipe_indices'] == [KNOWN_INCOMPATIBILITY_INDEX]
            and report['historical_boundary_labels_are_not_current_native_status'] is True,
            'original188 observes actual Torch without asserting a current native boundary')
    custody_proof = custody.read(ROOT, directory, execution, report, inputs.bind,
                                 'tests/f64_view/reference.py', allow=False)
    require(custody_proof['original_eleven_checks_recomputed_from_raw_inputs'] is True
            and custody_proof['original_check_names'] == list(custody.CHECK_NAMES)
            and set(custody_proof['original_checks']) == set(custody.CHECK_NAMES)
            and len(custody_proof['original_checks']) == 11
            and all(value is True for value in custody_proof['original_checks'].values())
            and custody_proof['original_checks'] == report['oracle_provenance']['checks']
            and custody_proof['actual_process_verified'] is True
            and custody_proof['actual_supervisor_intent_verified'] is True
            and custody_proof['actual_environment_transition_verified'] is True
            and custody_proof['original_reference_script_bytes_verified'] is True
            and custody_proof['full_installation_provenance_verified'] is True
            and custody_proof['complete_official_wheel_archive_verified'] is True
            and custody_proof['independent_official_wheel_member_anchor_verified'] is True
            and custody_proof['complete_installed_raw_snapshot_verified'] is True
            and custody_proof['all_actual_installed_inputs_unchanged'] is True
            and custody_proof['actual_mapped_torch_cpu_code_verified'] is True,
            'strict actual original wheel/install/mapped-code custody, never a report-only PASS')
    provenance = report['oracle_provenance']
    require(provenance['status'] == 'PASS' and len(provenance['checks']) == 11
            and all(value is True for value in provenance['checks'].values()),
            'all eleven original fixed-wheel checks')
    require(process['started'] is True and process['exit_code'] == 0
            and process['stdout_EOF'] is True and process['stderr_EOF'] is True
            and process['forced_cleanup'] is False and not process.get('timed_out')
            and not process.get('error') and not process.get('execution_error')
            and not process['cleanup_errors'] and not process['running_descendants_after_cleanup']
            and not process['terminated_process_groups'], 'actual clean reference process termination')
    argv = process['argv']
    require(argv[1:] == ['-B', str(ROOT / 'tests/f64_view/reference.py'), '--output', str(directory)]
            and report['actual_process']['argv'] == argv and report['actual_process']['pid'] == process['pid'],
            'actual complete reference argv and PID binding')
    interpreter = Path(argv[0]).resolve(strict=True)
    for channel in ('stdout', 'stderr'):
        raw = inputs.bind(execution / ('runner.' + channel), process[channel + '_sha256'])
        require(len(raw) == process[channel + '_bytes'], 'complete original reference stream')
    require(inputs.bind(execution / 'runner.stderr') == b'', 'empty original reference stderr')
    require(json.loads(inputs.bind(execution / 'runner.stdout')) == dict(
        status='OBSERVED_F64_VIEW_ORIGINAL188', cases=CASE_COUNT, expected_cases=CASE_COUNT, error=None),
        'complete original reference stdout')
    expected_files = {str(path.resolve()) for path in (
        ROOT / 'tests/f64_view/reference.py', ROOT / 'tests/f64_view/__init__.py',
        ROOT / 'tests/f64_view/cases.py', ROOT / 'tests/numeric_predicates/reference.py',
        ROOT / 'tests/numeric_predicates/cases.py', ROOT / 'tests/numeric_predicates/__init__.py',
        ROOT / 'tests/pytorch_oracle.py', ROOT / 'compatibility/pytorch-oracle-lock.json',
        ROOT / 'compatibility/upstream-lock.json', ROOT / 'build/pytorch-install.json', interpreter)}
    expected_files.update(str(path.resolve()) for path in [ROOT / 'tests/pinned_reference_custody.py',
        ROOT / 'compatibility/pytorch-reference-requirements.txt',
        ROOT / 'build/pytorch-reference-support-install.json'] + custody.source_files(ROOT))
    from . import family_reference
    expected_files.update(str(path.resolve()) for path in family_reference.source_files())
    require(set(report['inputs']) == expected_files, 'exact full original reference input inventory')
    for path, record in report['inputs'].items():
        raw = inputs.bind(path, record['sha256'])
        require(record == dict(path=str(Path(path).resolve()), bytes=len(raw), sha256=sha(raw),
                               mode=stat.S_IMODE(Path(path).stat().st_mode)),
                'original reference input bytes and mode')
    require(report['actual_process']['python'] == report['inputs'][str(interpreter)],
            'actual complete reference interpreter identity')
    family = report['families']
    require(set(family) == {'path', 'bytes', 'sha256'} and family['path'] == 'families/report.json',
            'independently retained family report uses the same strict actual process')
    require(len(inputs.bind(directory / family['path'], family['sha256'])) == family['bytes'],
            'same-process complete family report bytes; full family predicates remain a separate required reader')

    def blob(item):
        require(set(item) == {'path', 'bytes', 'sha256'}
                and item['path'] == 'blobs/' + item['sha256'] + '.bin', 'canonical observed blob identity')
        raw = inputs.bind(directory / item['path'], item['sha256'])
        require(len(raw) == item['bytes'], 'complete observed blob')
        return raw

    for index, (row, spec) in enumerate(zip(report['rows'], CASES)):
        require(row == json.loads(inputs.bind(directory / ('case-%03d.json' % index)))
                and row['index'] == index and row['spec'] == spec
                and row['accepted'] is (not bool(spec.get('expected_error')))
                and row['rng_unchanged'] is True and row['host_expected_values_computed'] is False,
                'literal input, order and independently retained operation outcome')
        require(row['stages'][0]['label'] == 'base' and set(row['stages'][0]['tensors']) == {'root'},
                'one actual base observation precedes every operation')
        known = {'root'}
        public = {'root'}
        stages = row['stages']
        if index == KNOWN_INCOMPATIBILITY_INDEX:
            require(row['accepted'] is True and spec['expected_Torch_acceptance'] is True
                    and row['known_incompatibility'] == spec['known_incompatibility'] == 'negative_dimension_on_empty_view'
                    and row['actual_Torch_operation_returned'] is True and len(stages) == 1
                    and row['original_input_after'] == stages[0]['tensors']['root']
                    and row['original_inputs_unchanged_after_view'] is True and row['all_Torch_Tensors_released'] is True,
                    'actual accepted negative empty shape is an explicit incompatibility, never a Torch error')
            value = row['boundary_result']
            require(value['dtype'] == 'torch.float64' and value['shape'] == [0, -1]
                    and value['strides'] == [1, 1] and value['numel'] == 0 and value['element_size'] == 8
                    and value['storage_offset'] == 0 and value['version'] == 0
                    and value['requires_grad'] is False and value['is_leaf'] is True and value['is_view'] is True
                    and value['history_present'] is False and value['retains_grad'] is False
                    and value['actual_Torch_tolist'] == [] and value['fresh_Tensor'] is True
                    and value['shared_Storage'] is True and value['original_root_base'] is True,
                    'complete actual Torch metadata and iteration for the unimplemented negative empty view')
            root = stages[0]['tensors']['root']
            require(blob(value['storage']) == blob(root['storage'])
                    and value['storage_bytes'] == root['storage_bytes'] == len(blob(value['storage']))
                    and len(blob(root['logical'])) == root['numel'] * 8,
                    'all actual boundary Storage and original root bytes')
            require(row['actual_Torch_lifetimes'] == [
                dict(stage='root-and-view', root=True, view=True),
                dict(stage='public-root-released', root=True, view=True),
                dict(stage='view-released', root=False, view=False)],
                'actual original-root retention and release for the known incompatible Torch view')
            require(stages[0]['aliases'] == {'root': dict(shared_original_storage=True, original_root_base=True)}
                    and stages[0]['actual_Torch_Tensor_liveness'] == {'root': True},
                    'complete actual base alias and lifetime before the boundary request')
            continue
        if row['accepted']:
            require(row['all_Torch_Tensors_released'] is True and row['explicit_mutations_observed'] is True
                    and len(stages) == len(spec['operations']) + 1
                    and len(row['events']) == len(spec['operations']),
                    'every actual view, mutation and release has an original observation')
        else:
            error = row['error']
            require(len(spec['operations']) == 1 and len(stages) == 1 and row['events'] == []
                    and error['type'] == 'RuntimeError'
                    and error_matches(RuntimeError(error['message']), spec['expected_error'], spec['operations'][0])
                    and row['original_inputs_unchanged_after_error'] is True,
                    'actual original Torch geometry refusal and unchanged complete input')
        for stage_index, stage in enumerate(stages):
            if stage_index:
                operation = spec['operations'][stage_index - 1]
                require(stage['label'] == '%02d-%s-%s' %
                        (stage_index - 1, operation['operation'], operation['name']),
                        'all stages retain the exact original operation order')
                event = row['events'][stage_index - 1]
                require(event['operation'] == operation, 'exact original observed event')
                if operation['operation'] == 'view':
                    require(operation['source'] in public and operation['name'] not in known
                            and all(event[key] is True for key in (
                                'fresh_Tensor', 'shared_Storage', 'original_root_base', 'prior_aliases_unchanged')),
                            'actual fresh alias to the original root and Storage')
                    known.add(operation['name'])
                    public.add(operation['name'])
                elif operation['operation'] == 'write':
                    require(operation['name'] in public and event['all_live_alias_versions_incremented_once'] is True,
                            'actual original shared-version mutation observation')
                else:
                    require(operation['operation'] == 'release' and operation['name'] in public
                            and event['public_Tensor_reference_released'] is True,
                            'actual original public Tensor reference release')
                    public.remove(operation['name'])
            require(set(stage['tensors']) == set(stage['aliases']) == public
                    and set(stage['actual_Torch_Tensor_liveness']) == known
                    and all(type(value) is bool for value in stage['actual_Torch_Tensor_liveness'].values())
                    and all(value['shared_original_storage'] is True and value['original_root_base'] is True
                            for value in stage['aliases'].values()),
                    'complete actual public Tensor, alias and weak-reference inventories')
            for value in stage['tensors'].values():
                require(value['dtype'] == 'torch.float64' and value['element_size'] == 8
                        and len(value['shape']) == len(value['strides']) <= 4
                        and value['requires_grad'] is False and value['history_present'] is False
                        and value['is_leaf'] is True and value['retains_grad'] is False,
                        'independently observed no-AD public F64 view metadata')
                require(len(blob(value['logical'])) == value['numel'] * 8
                        and len(blob(value['storage'])) == value['storage_bytes'],
                        'all independently observed logical and physical storage bytes')
        if row['accepted']:
            require(not public and not any(stages[-1]['actual_Torch_Tensor_liveness'].values()),
                    'all actual original Torch Tensor lifetimes ended')
    allocation = report['allocation_observation']
    require(allocation == json.loads(inputs.bind(directory / 'allocation-observation.json'))
            and allocation['spec'] == ALLOCATION_SPEC and allocation['spec_sha256'] == ALLOCATION_SHA256
            and allocation['actual_Torch_views'] == ALLOCATION_SPEC['views']
            and allocation['all_Torch_Tensors_released'] is True
            and allocation['host_expected_values_computed'] is False
            and [stage['label'] for stage in allocation['stages']] ==
                ['base', 'all-views', 'release-root', 'release-all-views'],
            'actual complete fixed alias-family allocation observation')
    names = {'alias-%03d' % index for index in range(ALLOCATION_SPEC['views'])}
    for stage, public, known in zip(allocation['stages'],
            ({'root'}, names | {'root'}, names, set()),
            ({'root'}, names | {'root'}, names | {'root'}, names | {'root'})):
        require(set(stage['tensors']) == set(stage['aliases']) == public
                and set(stage['actual_Torch_Tensor_liveness']) == known
                and all(type(value) is bool and value is bool(public)
                        for value in stage['actual_Torch_Tensor_liveness'].values())
                and all(row['shared_original_storage'] is True and row['original_root_base'] is True
                        for row in stage['aliases'].values()),
                'complete actual allocation alias/weak-reference lifetime observation')
        for value in stage['tensors'].values():
            require(value['dtype'] == 'torch.float64' and value['shape'] == [2]
                    and value['strides'] == [1] and value['numel'] == 2 and value['version'] == 0
                    and value['element_size'] == 8 and value['storage_offset'] == 0
                    and value['storage_bytes'] == len(blob(value['storage'])) == len(blob(value['logical'])) == 16,
                    'complete actual allocation result metadata and bytes')
    inputs.recheck()
    return dict(report=report, blob=blob, report_sha256=sha(inputs.bind(directory / 'report.json')),
                strict_reference_qualified=True, custody=custody_proof, root=directory, execution=execution)
