"""Read completed current-image reports without changing their original results."""
import hashlib
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import struct
from current_runtime_torch import FUNCTIONAL_SOURCE_SHA, FUNCTIONAL_SOURCE_BYTES, NUMPY_WARNING_SUFFIX

_profile_path = Path(__file__).resolve().parents[1] / "tools/current_runtime/qualification.py"
_profile_spec = importlib.util.spec_from_file_location("current_runtime_report_profile", _profile_path)
current_profile = importlib.util.module_from_spec(_profile_spec)
_profile_spec.loader.exec_module(current_profile)
SOURCE, IMAGE = current_profile.SOURCE, current_profile.IMAGE
FUNCTIONS, UNITS = current_profile.FUNCTIONS, current_profile.UNITS
F32_REFERENCE = 'c86b0a9384c788eaf7765480b022416075a0e3aa38455f48dcbd7f40476cecb0'


def require(condition, rule):
    if not condition:
        raise ValueError(rule)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def historical_reader(repository, inputs):
    directory = Path(repository) / 'tests/current_runtime_historical'
    def load(name, path):
        inputs.bind(path)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    profile = load('current_runtime_historical_profile', directory / 'profile.py')
    previous = sys.modules.get('profile')
    sys.modules['profile'] = profile
    try:
        return load('current_runtime_historical_report_reader', directory / 'run.py')
    finally:
        if previous is None:
            del sys.modules['profile']
        else:
            sys.modules['profile'] = previous


def complete_exit(receipt, expected_exit=0):
    require(receipt['started'] is True and receipt['exit_code'] == expected_exit
            and receipt['stdout_EOF'] is True and receipt['stderr_EOF'] is True
            and not receipt.get('forced_cleanup')
            and not receipt.get('timed_out') and not receipt.get('cleanup_errors')
            and not receipt.get('execution_error') and not receipt.get('error'),
            'complete actual runner exit and both stream EOFs')


def complete_runner(receipt, expected_exit=0):
    complete_exit(receipt, expected_exit)
    require(receipt['stderr_bytes'] == 0, 'actual runner stderr must be empty')


def current35_live_runner(receipt, directory, inputs):
    """Admit only empty stderr or the single observed fixed-wheel NumPy warning."""
    complete_exit(receipt)
    require(receipt.get('stderr_mirrored') is False, 'retain live-Torch stderr separately from the outer runner')
    observation_raw = inputs.bind(directory / 'torch-source.json')
    observation = json.loads(observation_raw)
    require(observation['schema'] == 'heptatorch.current-runtime.live-torch-source.v1'
            and observation['status'] == 'PASS_SOURCE_OBSERVATION_ONLY'
            and observation['before'] == observation['after'], 'complete actual fixed Torch source observation')
    require(isinstance(observation['cpu_capability'], str) and observation['cpu_capability']
            and observation['aten_cpu_capability_override'] is None,
            'record the actual CPU capability without overriding Torch dispatch')
    before = observation['before']
    require(before['sha256'] == FUNCTIONAL_SOURCE_SHA and before['bytes'] == FUNCTIONAL_SOURCE_BYTES
            and Path(before['path']).is_absolute()
            and Path(before['path']).as_posix().endswith('/torch/_subclasses/functional_tensor.py')
            and observation['actual_loaded']['torch._subclasses.functional_tensor'] == before['path']
            and observation['actual_loaded']['torch'] == observation['torch_init']
            and Path(observation['torch_init']).parent / '_subclasses/functional_tensor.py' == Path(before['path']),
            'warning is tied to the actually imported fixed Torch source')
    require(observation['python_executable'] == receipt['argv'][0]
            and str(Path(observation['python_executable']).resolve()) == observation['python_executable_resolved'],
            'source observation ran in the explicitly selected Torch interpreter')
    require(len(receipt['argv']) >= 7 and receipt['argv'][1] == '-B'
            and Path(receipt['argv'][2]).name == 'current_runtime_torch.py'
            and receipt['argv'][3:6] == ['--source-observation', str(directory / 'torch-source.json'), '--']
            and observation['target_argv'] == receipt['argv'][6:],
            'one exact unchanged reader invocation inside the observed interpreter')
    copy = observation['source_copy']
    require(copy['path'] == 'torch-functional_tensor.py' and copy['sha256'] == FUNCTIONAL_SOURCE_SHA
            and copy['bytes'] == FUNCTIONAL_SOURCE_BYTES and copy['mode'] == before['mode'],
            'complete original Torch warning-source evidence')
    raw_source = inputs.bind(directory / copy['path'], FUNCTIONAL_SOURCE_SHA)
    require(len(raw_source) == FUNCTIONAL_SOURCE_BYTES
            and ((directory / copy['path']).stat().st_mode & 0o7777) == before['mode'],
            'preserved Torch source bytes and mode')
    for stream in ('stdout', 'stderr'):
        raw = inputs.bind(directory / ('native_and_live_reference-execution/runner.' + stream),
                          receipt[stream + '_sha256'])
        require(len(raw) == receipt[stream + '_bytes'], 'complete actual live-Torch runner streams')
        if stream == 'stderr':
            expected = (before['path'] + NUMPY_WARNING_SUFFIX).encode()
            require(raw in (b'', expected), 'only one exact fixed-wheel NumPy-unavailable warning is admitted')
    return dict(source_observation=dict(path='torch-source.json', bytes=len(observation_raw),
                                      sha256=sha(observation_raw)),
                stderr_kind='EMPTY' if receipt['stderr_bytes'] == 0 else 'SINGLE_FIXED_TORCH_NUMPY_WARNING',
                stderr_bytes=receipt['stderr_bytes'], stderr_sha256=receipt['stderr_sha256'],
                original_stderr_retained=True, native_stderr_rule_unchanged=True,
                cpu_capability=observation['cpu_capability'], aten_cpu_capability_override=None,
                fixed_functional_source_sha256=FUNCTIONAL_SOURCE_SHA)


def qualify_historical(report, suite, repository, build, build_sha, directory, inputs):
    require(suite in ('bool27', 'integral24'), 'exact historical suite name')
    require(report['schema'] == 'heptatorch.701.original-readers-and-contract-delta.v1'
            and report['status'] == 'PASS_701_COMPARISONS_WITH_ORIGINAL_FAILURES_RETAINED'
            and report['suite'] == suite and report['error'] is None,
            'complete historical comparison and explicit contract-delta admission')
    require(report['source_sha256'] == SOURCE and report['native_sha256'] == IMAGE
            and report['actual_build_sha256'] == build_sha and report['native_executed'] is True
            and report['native_execution_requested'] is True and report['Torch_executed'] is False
            and report['compiler_executed'] is False, 'same actual build and execution')
    require(report['original_reports_modified'] is False and report['original_comparisons_modified'] is False
            and report['original_expected_refusals_modified'] is False
            and report['F32_numerical_semantics_qualified_here'] is False,
            'original refusals and comparisons remain unchanged')
    require(report['schema_closure'] is False and report['full_project_status'] == 'HOLD',
            'finite historical comparisons must preserve project status')
    complete_runner(report['original_runner'], expected_exit=1)
    for stream in ('stdout', 'stderr'):
        data = inputs.bind(directory / ('original-runner.' + stream), report['original_runner'][stream + '_sha256'])
        require(len(data) == report['original_runner'][stream + '_bytes'], 'actual original-runner streams')
    original = inputs.bind(directory / 'original/report.json', report['original_report']['sha256'])
    require(len(original) == report['original_report']['bytes']
            and report['original_report']['status'] == report['original_report_status'] == 'FAIL',
            'original FAIL report remains an exact separate artifact')
    inner = json.loads(original)
    reader = historical_reader(repository, inputs)
    config = dict(build_directory=str(build), expected_build_sha256=build_sha,
                  original_output=str(directory / 'original'))
    facts = (reader.qualify_bool if suite == 'bool27' else reader.qualify_integral)(inner, config)
    require(all(report[key] == value for key, value in facts.items()), 'repeat original complete comparison admission')
    children, cases, changed = (36, 27, [2, 3, 5, 6]) if suite == 'bool27' else (29, 24, [0, 1, 2, 3])
    require(report['explicitly_changed_control_indices'] == changed
            and report['original_cases_passed'] == cases, 'exact retained denominator and contract changes')
    transports = reader.native_transports(directory / 'original', children)
    require(report['transport'] == transports, 'all original raw frames and native child EOFs must remain complete')
    return dict(cases=cases, native_sessions=children, frames=transports['frames'],
                original_report_status='FAIL_RETAINED', original_dtype_passed=report['original_dtype_passed'],
                original_dtype_failed=report['original_dtype_failed'],
                explicitly_changed_control_indices=changed,
                events=260 if suite == 'integral24' else None,
                current_dtype_qualification='sum_f32_17/dtype_controls')


def qualify_f32(report, repository, build, build_sha, directory, inputs):
    require(report['schema'] == 'heptatorch.current-fusion.f32.v1'
            and report['status'] == 'PASS' and report['error'] is None,
            'new F32 comparison report must actually pass')
    require(report['source_sha256'] == SOURCE and report['native_sha256'] == IMAGE
            and report['reference_sha256'] == F32_REFERENCE
            and report['inputs'][str(build / 'build.json')]['sha256'] == build_sha,
            'new F32 source, image, actual build and frozen reference bindings')
    require(report['native_executed'] is True and report['Torch_executed'] is False
            and report['inputs_unchanged'] is True and report['build_functions'] == FUNCTIONS
            and report['cold_units'] == UNITS, 'new F32 actual execution and input custody')
    require(report['expected_cases'] == report['passed'] == 17 and report['failed'] == 0
            and [len(report[key]) for key in ('cases', 'axis_controls', 'dtype_controls',
                                            'entry_controls', 'allocation_controls')] == [17, 4, 7, 1, 2],
            'all seventeen cases and every control group are mandatory')
    require(report['expected_axis_controls'] == 4 and report['expected_dtype_controls'] == 7
            and report['expected_entry_checks'] == 10 and report['expected_allocation_controls'] == 2,
            'new F32 control denominator unchanged')
    rows = sum((report[key] for key in ('cases', 'axis_controls', 'dtype_controls',
                                      'entry_controls', 'allocation_controls')), [])
    require(len(rows) == 31 and report['children'] == 33 and
            all(row['status'] == 'PASS' and row['error'] is None and row['exit_code'] == 0
                and row['frames'] > 0 and row['checks']
                and all(check['passed'] is True for check in row['checks']) for row in rows),
            'all thirty-one successful current-contract sessions and original assertions must pass')
    original_row, extension_row = report['original_report'], report['extension_report']
    require(original_row['path'] == 'original/report.json' and extension_row['path'] == 'extension/report.json',
            'original F32 refusal results and new accepted controls remain separate')
    original_raw = inputs.bind(directory / original_row['path'], original_row['sha256'])
    extension_raw = inputs.bind(directory / extension_row['path'], extension_row['sha256'])
    require(len(original_raw) == original_row['bytes'] and len(extension_raw) == extension_row['bytes'],
            'complete original and extended F32 reports')
    original, extension = json.loads(original_raw), json.loads(extension_raw)
    require(original['status'] == 'FAIL' and original['error'] is None
            and original['children'] == 31 and original['passed'] == 17 and original['failed'] == 0
            and report['original_report_status'] == 'FAIL_RETAINED'
            and report['original_dtype_failed_indices'] == [5, 6]
            and report['original_dtype_controls'] == original['dtype_controls']
            and all(report[key] == original[key] for key in ('cases', 'axis_controls', 'entry_controls', 'allocation_controls')),
            'all original comparisons and the exact two old refusal failures are retained')
    require(extension['schema'] == 'heptatorch.bool-dtype-extension.native.v1'
            and extension['status'] == 'PASS' and extension['children'] == len(extension['cases']) == 2
            and extension['inputs_unchanged'] is True and extension['native_executed'] is True
            and extension['source_sha256'] == SOURCE and extension['native_sha256'] == IMAGE
            and extension['actual_build_sha256'] == build_sha
            and extension['reference']['actual_torch_cases'] == 2
            and extension['reference']['report_sha256'] == '7b38959d3e052a0105c0a5e17fe9688dc8bcda975501b388a805f35599a1537c'
            and report['new_Bool_dtype_reference'] == extension['reference']
            and report['dtype_controls'] == original['dtype_controls'][:5] + extension['cases'],
            'both new successes use their independently observed exact recipes and same image')
    for row in original['dtype_controls'][5:]:
        failed = [check for check in row['checks'] if check['passed'] is not True]
        require(row['status'] == 'FAIL' and row['exit_code'] == 0
                and row['error']['type'] == 'AssertionError' and 'EOF_failure' not in row['error']
                and failed == [dict(label='documented-current-dtype-boundary', actual=0, expected=7, passed=False)]
                and len(row['operations']) == 1
                and row['operations'][0]['normalized_status'] == row['operations'][0]['frame_status'] == 0,
                'only the original unsupported-dtype expectation changed')
    complete_runner(report['original_runner'], expected_exit=1)
    for stream in ('stdout', 'stderr'):
        raw = inputs.bind(directory / ('runner.' + stream), report['original_runner'][stream + '_sha256'])
        require(len(raw) == report['original_runner'][stream + '_bytes'], 'complete original worker streams')
    require(report['frames'] == original['frames'] + extension['frames']
            and report['frames'] == sum(row['frames'] for row in rows)
                + sum(row['frames'] for row in original['dtype_controls'][5:]),
            'every original success/failure frame and both new native control sessions')
    require([row['name'] for row in report['axis_controls']] ==
            ['duplicate_normalized_axis', 'positive_axis_out_of_range', 'negative_axis_out_of_range',
             'scalar_axis_out_of_range'], 'four original axis error cases')
    require([row['name'] for row in report['dtype_controls']] ==
            ['Bool_explicit_Bool', 'I64_explicit_Bool', 'Bool_explicit_F32', 'I64_explicit_F32',
             'F32_explicit_I64', 'F32_explicit_Bool', 'wrapped_F64_explicit_Bool'], 'all seven current dtype controls')
    require([row['operations'][0]['normalized_status'] for row in report['dtype_controls']]
            == [0, 0, 0, 0, 7, 0, 0], 'six independently observed successes and one preserved refusal')
    operations = report['entry_controls'][0]['operations']
    require([row['label'] for row in operations] ==
            ['invalid_public_capability', 'invalid_keepdim', 'unsupported_dtype_code',
             'reserved_word_six', 'reserved_word_seven', 'truncated_axis_payload',
             'None_nonzero_payload', 'empty_axes_nonzero_payload', 'axis_payload_before_payload_region',
             'stale-public-capability'] and [row['normalized_status'] for row in operations]
            == [1, 7, 7, 7, 7, 2, 7, 7, 2, 1], 'all ten entry/capability assertions')
    for stage, row in zip(('cast-storage', 'result-storage-after-cast'), report['allocation_controls']):
        require(row['name'] == 'integral-F32-' + stage + '-OOM-full-rollback-and-recovery',
                'two distinct actual allocation failure stages')
        operations = row['operations']
        witness = 'cast-only108-allocation-failure' if stage == 'cast-storage' else 'cast-only108-under-same-limit'
        require([op['label'] for op in operations] ==
                [witness, 'sum115-' + stage + '-allocation-failure', 'same-sum115-after-resource-recovery']
                and [op['normalized_status'] for op in operations]
                == ([3, 3, 0] if stage == 'cast-storage' else [0, 3, 0]),
                'actual cast witness, exact sum failure, and same-command recovery')
    cat_transition = report['source_transition']
    cat_path = Path(repository) / 'tools/current_runtime/cat_source_transition.py'
    cat_raw = inputs.bind(cat_path)
    cat_spec = importlib.util.spec_from_file_location('qualified_current_cat_source_transition', cat_path)
    cat = importlib.util.module_from_spec(cat_spec)
    cat_spec.loader.exec_module(cat)
    outer_path = Path(repository) / 'tools/current_runtime/fusion_source_transition.py'
    inputs.bind(outer_path, 'e7e70d4ae6e1c8daa3af97ee1db8492c177d4618e25e1ee1dab1e6767f7d95c5')
    outer_spec = importlib.util.spec_from_file_location('qualified_current765_source_transition', outer_path)
    outer = importlib.util.module_from_spec(outer_spec)
    outer_spec.loader.exec_module(outer)
    historical700, outer_proof = outer.restore_source(inputs.bind(Path(build) / 'runtime.hepta', SOURCE), inputs.bind)
    require(FUNCTIONS == 765 and UNITS == 15 and sha(historical700) == cat.SOURCE == outer.PARENT_SOURCE,
            'actual765 profile and exact historical700 source subject')
    restored, cat_proof = cat.restore_source(historical700, inputs.bind)
    require(cat_transition['schema'] == 'heptatorch.current765-cat.source-proof.v1'
            and cat_transition['source_sha256'] == SOURCE
            and cat_transition['parent_source_sha256'] == cat.PARENT_SOURCE == sha(restored)
            and cat_transition['whole_inverse_exact'] is True
            and cat_transition['cat_transition'] == cat_proof
            and cat_transition['outer_source_inverse'] == outer_proof
            and cat_transition['actual_functions'] == 765 and cat_transition['historical_functions'] == 700
            and cat_transition['historical_count_binding_separated'] is True
            and cat_transition['original_source_proofs_modified'] is False
            and cat_transition['runtime_qualification_transferred'] is False,
            'exact complete three-function cat inverse before all original source proofs')
    fusion = cat_transition['logical_fusion']
    require(fusion['schema'] == 'heptatorch.current-logical-fusion.source-proof.v1'
            and fusion['whole_inverse_exact'] is True and fusion['source_sha256'] == cat.PARENT_SOURCE
            and fusion['current_functions'] == 700 and fusion['original_functions'] == 701
            and fusion['unchanged_functions'] == 689
            and fusion['modified_functions'] == ['entry', 'sb_sum_bool', 'si_dispatch_sum', 'si_plan', 'to_command']
            and fusion['added_functions'] == ['logical_axis_mask', 'logical_dispatch', 'logical_plan',
                                             'logical_reduce', 'logical_truth', 'reduction_plan']
            and fusion['removed_functions'] == ['growth_capacity', 'init', 'integer_view_end',
                'integer_view_root_valid', 'mark_concat', 'to_as_strided_direct_copy', 'to_view_root_valid']
            and fusion['pr38_source_reconstructed_sha256'] == fusion['parent_source_sha256']
                == '82360ce87dae6f9a004ef50be4f991b6f83905a83f4b17ca8d6634827b7bd313'
            and fusion['original_FAIL_reports_modified'] is False
            and fusion['runtime_qualification_transferred'] is False,
            'exact whole-current-source inverse preserves every PR38 numerical and cat fix')
    transition = fusion['pr38_to_baseline']
    require(transition['whole_inverse_exact'] is True and transition['unchanged_functions'] == 696
            and transition['modified_functions'] == ['incoming_describe_cat', 'sd_loop2d', 'sd_vector_inner', 'si_dispatch_sum']
            and transition['added_functions'] == ['sf_sum_f32']
            and transition['current_to_avx2']['whole_inverse_exact'] is True
            and transition['current_to_avx2']['unchanged_functions'] == 700
            and transition['current_to_avx2']['modified_functions'] == ['incoming_describe_cat']
            and transition['current_to_avx2']['original_success_return_preserved'] is True
            and transition['current_to_avx2']['shared_cache_and_inplace_guards_unchanged'] is True,
            'complete exact cat-source inverse with the original success and resource guards')
    previous = transition['avx2_to_700']
    require(previous['whole_inverse_exact'] is True and previous['unchanged_functions'] == 697
            and previous['current_to_first701']['modified_constants'] == 6
            and previous['current_to_first701']['whole_inverse_exact'] is True
            and previous['first701_to_700']['whole_inverse_exact'] is True
            and previous['first701_to_700']['unchanged_functions'] == 699
            and previous['reader_identity_transition']['current_to_avx2']['reader_whole_inverse_exact'] is True
            and previous['reader_identity_transition']['avx2_to_first701']['reader_whole_inverse_exact'] is True
            and previous['reader_identity_transition']['physical_input_records_modified'] is False,
            'all three source-inverse layers and complete original reader comparisons')
    require(transition['original_first701_failure_modified'] is False
            and transition['original_avx2_original270_failure_modified'] is False
            and transition['runtime_qualification_transferred'] is False,
            'all prior failures remain unchanged and no earlier PASS transfers')
    require(report['historical_Bool7_changed_indices'] == [2, 3, 5, 6]
            and report['original_Bool7_unchanged_all_pass'] is False
            and report['original700_or_original_controls_modified'] is False
            and report['schema_closure'] is False and report['full_project_status'] == 'HOLD',
            'new contract qualification must not relabel historical results or whole project')
    transports = historical_reader(repository, inputs).native_transports(directory, 33)
    require(transports['frames'] == report['frames'], 'all F32 raw frames and double EOFs')
    return dict(cases=17, native_sessions=33, frames=report['frames'], axis_controls=4,
                dtype_controls=7, entry_controls=10, allocation_controls=2,
                original_dtype_failures_retained=True, independently_observed_Bool_extensions=2)


def qualify_current35(report, build_sha, directory, inputs):
    require(report['schema'] == 'heptatorch.current-runtime.original35.v1'
            and report['status'] == 'PASS' and report['source_sha256'] == SOURCE
            and report['native_sha256'] == IMAGE and report['actual_build_sha256'] == build_sha,
            'complete same-image current35 qualification')
    require(report['native_executed'] is True and report['Torch_executed'] is True
            and report['all_inputs_unchanged'] is True and report['failures'] == [],
            'current35 actual native/live-Torch execution and custody')
    require(report['cases'] == 35 and report['frames'] == 6344 and report['compared_values'] == 2590274
            and report['oracle_provenance_checks'] == 11
            and report['preserved_explicit_state_version_differences'] == 283,
            'all original35 workload and explicitly recorded state differences')
    require(report['positive_controls'] == 4 and report['negative_controls'] == 18,
            'all original copied positive and negative controls')
    require(set(report['executions']) == {'native_and_live_reference', 'strict_report', 'copied_controls'},
            'all current35 execution stages required')
    for name, receipt in report['executions'].items():
        if name == 'native_and_live_reference':
            facts = current35_live_runner(receipt, directory, inputs)
            require(report['live_reference_stderr_admission'] == facts, 'repeat exact live-reference stderr admission')
        else:
            complete_runner(receipt)
    files = {}
    for name, row in report['reports'].items():
        raw = inputs.bind(directory / row['path'], row['sha256'])
        require(len(raw) == row['bytes'], 'current35 original report bytes')
        files[name] = json.loads(raw)
    require(set(files) == {'original', 'observer', 'strict', 'controls'}, 'all current35 original report bindings')
    original, observer, strict, controls = [files[k] for k in ('original', 'observer', 'strict', 'controls')]
    require(original['status'] == observer['status'] == 'PASS' and original['cases'] == len(original['rows']) == 35
            and original['failed_cases'] == 0 and all(row['status'] == 'PASS' for row in original['rows']),
            'all original35 native/Torch comparisons')
    require(original['native_executable_sha256'] == observer['profile']['native_sha256'] == IMAGE
            and observer['profile']['source_sha256'] == SOURCE
            and observer['cold_build_admission']['build_sha256'] == build_sha,
            'original35 source/image/cold-build identity')
    require(observer['all_original_EOF_and_post_close_Version_gates_executed'] is True
            and len(observer['observer_cases']) == 35
            and all(row['private_node_observer']['status'] == 'PASS'
                    and row['private_node_observer']['eof_exit_assertions_executed'] is True
                    and row['private_node_observer']['post_close_parameter_version_assertions_executed'] is True
                    for row in original['rows']), 'all original35 EOF/exit/Version assertions')
    require(strict['status'] == 'PASS_EXACT_ORIGINAL_AND_ADDITIVE_REPORT_BINDING'
            and strict['cases'] == 35 and strict['oracle_provenance_checks'] == 11
            and strict['original_report_sha256'] == report['reports']['original']['sha256']
            and strict['observer_report_sha256'] == report['reports']['observer']['sha256'],
            'frozen full strict report checker actually passed')
    require(controls['status'] == 'PASS_OBSERVER_CONTROLS_ONLY' and controls['failures'] == 0
            and controls['positive_cases'] == len(controls['positive_rows']) == 4
            and controls['negative_cases'] == len(controls['negative_rows']) == 18
            and controls['original_journals_unchanged'] is True and controls['observer_source_unchanged'] is True,
            'all copied controls and unmodified actual journals')
    require(all(row['status'] == 'PASS' for row in controls['positive_rows'])
            and len({row['case'] for row in controls['positive_rows']}) == 4
            and all(row['status'] == 'PASS_EXPECTED_REJECTION' and row['actual_rule'] == row['expected_rule']
                    for row in controls['negative_rows'])
            and len({row['case'] for row in controls['negative_rows']}) == 18,
            'each distinct positive and named negative control must pass')
    require(report['full_project_status'] == 'HOLD' and report['full_pytorch_rewrite_complete'] is False,
            'finite original35 qualification does not establish project closure')
    return dict(cases=35, frames=6344, compared_values=2590274,
                positive_controls=4, negative_controls=18, actual_Torch_reference=True,
                preserved_explicit_state_version_differences=283)


def qualify_current270(report, build_sha, directory, inputs):
    require(report['schema'] == 'heptatorch.current701-original270.v1'
            and report['status'] == 'PASS_CURRENT701_ORIGINAL270' and report['error'] is None,
            'all original270 comparisons must actually pass')
    require(report['source_sha256'] == SOURCE and report['native_sha256'] == IMAGE
            and report['build_sha256'] == build_sha
            and report['cold_build_admission']['build_sha256'] == build_sha,
            'same selected source, image and complete actual cold build')
    require(report['native_executed'] is True and report['Torch_executed'] is True
            and report['all_inputs_unchanged'] is True and report['inputs_unchanged']
            and all(ok is True for ok in report['inputs_unchanged'].values()),
            'actual270 native/live-Torch execution and complete unchanged inputs')
    require(report['expected_cases'] == report['cases'] == 270 and report['expected_frames'] == report['frames'] == 284,
            'all original270 cases and fourteen actual prefix/refusal pairs')
    require(report['historical_runtime_executed'] is False and report['synthetic_reference_values'] is False
            and report['synthetic_native_response_records'] is False
            and report['full_private_allocation_transition_ledger_claimed'] is False
            and report['full_pytorch_rewrite_complete'] is False,
            'original270 scope and actual-data requirements unchanged')
    original_raw = inputs.bind(directory / 'original-report.json', report['original_report']['sha256'])
    require(len(original_raw) == report['original_report']['bytes'], 'complete original270 report bytes')
    original = json.loads(original_raw)
    require(original['status'] == 'PASS' and original['setup_error'] is None and original['failed_cases'] == 0
            and original['expected_cases'] == original['cases'] == len(original['rows']) == 270
            and all(row['status'] == 'PASS' for row in original['rows']), 'all original270 numeric rows')
    require(original['native_executable_sha256'] == IMAGE and original['native_unchanged_during_run'] is True
            and original['seed'] == 1729, 'original270 image/seed binding')
    checks = {'python_matches', 'platform_matches', 'version_backend_matches', 'git_matches_oracle_build_pin',
              'distribution_version_matches', 'direct_url_matches', 'direct_url_archive_sha256_matches',
              'pip_report_url_matches', 'pip_report_archive_sha256_matches', 'pip_report_version_matches',
              'pip_report_requested'}
    provenance = original['oracle_provenance']
    require(provenance == report['oracle_provenance'] and provenance['status'] == 'PASS'
            and set(provenance['checks']) == checks and all(value is True for value in provenance['checks'].values())
            and provenance['lock_sha256'] == 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944',
            'all eleven original live-Torch provenance assertions')
    groups = [('operators', 51), ('spatial_modules', 16), ('optimizers', 8), ('training', 1),
              ('extensions', 77), ('training_sessions', 6), ('concatenation', 27),
              ('concatenation_layouts', 8), ('scatter_index_extensions', 74), ('combined_indexed_training', 2)]
    require([(row['function'], row['cases']) for row in report['groups']] == groups
            and all(row['cases'] == row['expected_cases'] == len(row['case_names']) for row in report['groups']),
            'complete original ten-group census and order')
    transports = report['native_cases']
    require(len(transports) == 270 and len({row['case'] for row in transports}) == 270
            and [row['case'] for row in transports] == [row['case'] for row in original['rows']],
            'each original numeric case has its own new actual native execution')
    require(sum(bool(row['expected_status']) for row in transports) == 14
            and sum(len(row['requests']) for row in transports) == 284, 'all actual native frames and refusal pairs')
    for index, row in enumerate(transports):
        require(row['status'] == 'PASS_ACTUAL_TRANSPORT' and row['error'] is None
                and row['native_started'] is True and row['executed_image_sha256'] == IMAGE
                and row['native_exit'] == 0 and row['stdout_EOF'] is True and row['stderr_EOF'] is True
                and row['stderr_bytes'] == row['unexpected_stdout_bytes'] == 0,
                'each original270 actual child image, exit and complete streams')
        case = directory / 'native' / (str(index).zfill(3) + '-' + sha(row['case'].encode())[:16])
        require(json.loads(inputs.bind(case / 'transport.json')) == row, 'original actual transport receipt')
        expected_status = row['expected_status']
        require(expected_status in (0, 2, 5, 7)
                and len(row['requests']) == (2 if expected_status else 1)
                and row['same_process_success_prefix_and_refusal'] is bool(expected_status)
                and row['synthetic_native_response_records'] is False,
                'actual same-process success prefix and terminal refusal')
        sent, received = [], []
        for number, frame in enumerate(row['requests']):
            request = inputs.bind(case / ('%02d-request.bin' % number), frame['request_sha256'])
            response = inputs.bind(case / ('%02d-response.bin' % number), frame['response_sha256'])
            require(struct.unpack_from('<8sQ', request) == (b'HTOWNR01', len(request) - 16)
                    and struct.unpack_from('<8sQ', response) == (b'HTOWNS01', len(response) - 16),
                    'complete raw current270 request/response envelope')
            magic, version, count, status = struct.unpack_from('<8s3Q', response, 16)
            wanted = expected_status if expected_status and number == 1 else 0
            require(magic == b'HTORSP03' and version == 3 and status == wanted == frame['actual_status']
                    and count == frame['actual_records'] and len(response) - 16 == frame['response_body_bytes']
                    and frame['response_before_stdin_EOF'] is True,
                    'raw original270 frame status, records and live response boundary')
            if wanted:
                require(count == 0 and frame['complete_owner_rollback'] is True, 'atomic real terminal refusal')
            sent.append(request); received.append(response)
        require(inputs.bind(case / 'actual.stdin') == b''.join(sent)
                and inputs.bind(case / 'actual.stdout') == b''.join(received)
                and inputs.bind(case / 'native.stderr') == inputs.bind(case / 'unexpected.stdout') == b'',
                'entire recorded native streams equal all actual frames and empty tails')
        snapshots = row['owner_snapshots']
        labels = [('%02d-%s' % (number, side)) for number in range(len(row['requests'])) for side in ('before', 'after')]
        require([snapshot['label'] for snapshot in snapshots] == labels, 'all original270 full owner boundaries')
        for snapshot in snapshots:
            require(snapshot['file'] == snapshot['label'] + '.json.gz', 'exact owner snapshot filename')
            raw = gzip.decompress(inputs.bind(case / snapshot['file']))
            require(sha(raw) == snapshot['snapshot_sha256'], 'complete original270 owner snapshot bytes')
            decoded = json.loads(raw)
            require(decoded['full_sha256'] == snapshot['full_owner_sha256']
                    and decoded['length'] == snapshot['owner_bytes'], 'actual full owner boundary identity')
    return dict(cases=270, frames=284, native_sessions=270, refusal_pairs=14,
                compared_values=original['compared_values'], oracle_provenance_checks=11,
                actual_Torch_reference=True)
