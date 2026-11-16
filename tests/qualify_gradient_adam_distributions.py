#!/usr/bin/env python3
"""Execute and bind the frozen candidate05 integration qualification.

This is integration orchestration, not a new independent implementation or
semantic review. It executes the frozen producer and independent runners without
rewriting their expectations. --producer-review explicitly reuses an existing
17-step execution receipt after checking every child and its native inputs.
No installed Torch, NumPy, HIA authoring parser, or numerical runtime is imported.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import traceback


SOURCE = '1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae'
IMAGE = '6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe'
IMAGE_NAME = 'tensor-gradient-adam-distributions-session'
LOCK = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
SESSION_LINKER = 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
RUNTIME_LINKER = '0c5176e3aa968631ae592b887c5dc3cf1e8e3f48f73593f076e9d7965ffacfd7'
PROVENANCE = {
    'python_matches', 'platform_matches', 'version_backend_matches',
    'git_matches_oracle_build_pin', 'distribution_version_matches',
    'direct_url_matches', 'direct_url_archive_sha256_matches',
    'pip_report_url_matches', 'pip_report_archive_sha256_matches',
    'pip_report_version_matches', 'pip_report_requested',
}
MANIFESTS = (
    ('tools/gradient_adam_distributions/package-manifest.json',
     '3e8c80053365f7e5de1d24725c511f49f256e8437f5a6d3556f09ff07682dc69', 45),
    ('validation/gradient-adam-distributions/frozen-handoff-05-manifest.json',
     '5a4e12c1ca07726e9e1223ea2fe9a82debd76af881be0eee0c2fb800d966ca01', 132),
    ('validation/gradient-composition-review/frozen-package-05-manifest.json',
     '455a64492cfaf7a75b24724542e3b5fb81a0e9c92b06dccb27211ed64a12a6aa', 78),
    ('validation/adam-distributions-gradient-review/frozen-package-manifest.json',
     'ae10fbd928c28ff626b497b9b18697aa75682004e1d196442ebef0c5dfecb0d2', 51),
    ('validation/adam-distributions-gradient-review/candidate05/frozen-additive-package-manifest.json',
     'a483343a288d68eb0d17e545c0ff489376c3c214d58b3b6164665626bf50f944', 9),
)
MANIFEST_RELOCATIONS = {
    ('validation/adam-distributions-gradient-review/frozen-package-manifest.json', 'README.md'):
        'validation/adam-distributions-gradient-review/history/frozen-package-04/README.md',
    ('validation/adam-distributions-gradient-review/candidate05/frozen-additive-package-manifest.json', 'README.md'):
        'validation/adam-distributions-gradient-review/candidate05/README.md',
}
EXTRA_INPUTS = {
    'compatibility/pytorch-oracle-lock.json': LOCK,
    'tools/cpu_distributions/protocol.py': '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9',
    'tools/cpu_distributions/owner_client.py': 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b',
    'tools/cpu_distributions/entry_probe.hepta': '54f81a65652e2c2ba9eb3f04d608e89b71c92c5ad33ff7061e9a03b89f45f9ea',
    'tools/cpu_distributions/state_handoff.hepta': '4093b8a1a4ce1b19d6943f457649355c950cfba50743ac7afe8a8396437cc7ee',
    'tools/cpu_distributions/evidence/torch-bernoulli-dropout.json': 'c624f5c4762b1c71e7053aba9af55aaa3d8bbaae5aa100b0a6df5f44f58da467',
}
STEPS = (
    ('cold-native-build', 'build.py', None),
    ('original-public-ten', 'replay_original_ten.py', 'torch-gradient-objects.json'),
    ('incoming-25', 'materialization_replay.py', 'torch-gradient-materialization.json'),
    ('alias-edge-12', 'replay_alias_edges.py', 'torch-gradient-alias-edge.json'),
    ('bernoulli-gradient-layout-8', 'replay_bernoulli_gradient_layout.py', 'torch-bernoulli-gradient-layout.json'),
    ('native-preserve-storage-4', 'test_preserve_storage_lifetimes.py', None),
    ('native-private-lifetimes-5', 'test_native_lifetimes.py', None),
    ('typed-gradient-references', 'test_gradient_references.py', None),
    ('signed-zero-13', 'replay_signed_zero.py', 'torch-gradient-signed-zero.json'),
    ('partial-16', 'replay_partial.py', 'torch-backward-partial.json'),
    ('presence-8', 'replay_presence.py', 'torch-gradient-presence.json'),
    ('gradient-clear-10', 'replay_grad_clear.py', 'torch-grad-clear.json'),
    ('distribution-atomic-183', 'replay_distribution_atomic.py', 'torch-bernoulli-dropout.json'),
    ('distribution-explicit-183', 'replay_distribution_explicit.py', 'torch-bernoulli-dropout.json'),
    ('original-optimizer-24-mapping', 'replay_optimizer_objects_v2.py', 'torch-optimizer-objects.json'),
    ('owned-adam-cases21-24', 'replay_adam_objects.py', 'torch-optimizer-objects.json'),
    ('optimizer-gradient-case20', 'replay_optimizer_gradient_alias.py', 'torch-optimizer-objects.json'),
)
OPEN_DISTRIBUTIONS = [4, 9, 14, 19, 24, 29, 34, 39, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174]
ATOMIC_DIFFERENCES = [101, 103, 105, 179]
PRODUCER_EXPECTED = 'validation/gradient-adam-distributions/reports/candidate-05'
PEER_EXPECTED = 'validation/adam-distributions-gradient-review/candidate05'
NATIVE_EXPECTED = 'validation/gradient-composition-review/candidate-05'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, ('duplicate JSON member', key))
        result[key] = value
    return result


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique_object)


def relative_file(root, name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and str(p) == name,
            ('unsafe relative input', name))
    return root / name


class Custody:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None, size=None, mode=None):
        path = Path(path).resolve()
        require(path.is_file(), ('missing input', str(path)))
        info = {'sha256': digest(path), 'bytes': path.stat().st_size,
                'mode': stat.S_IMODE(path.stat().st_mode)}
        require(expected is None or info['sha256'] == expected, ('SHA mismatch', str(path)))
        require(size is None or info['bytes'] == size, ('length mismatch', str(path)))
        require(mode is None or info['mode'] == mode, ('mode mismatch', str(path)))
        require(path not in self.files or self.files[path] == info, ('input changed', str(path)))
        self.files[path] = info
        return info

    def json(self, path):
        self.bind(path)
        return decode(Path(path).read_bytes())

    def finish(self):
        for path, info in list(self.files.items()):
            self.bind(path, info['sha256'], info['bytes'], info['mode'])


def provenance(value):
    require(value['status'] == 'PASS' and value['lock_sha256'] == LOCK, 'reference provenance identity')
    require(set(value['checks']) == PROVENANCE and all(x is True for x in value['checks'].values()),
            'all eleven original provenance checks are required')


def reference(path, custody):
    result = custody.json(path)
    require(result['status'] == 'OBSERVED', ('reference was not observed', str(path)))
    provenance(result['oracle_provenance'])
    return result


def row_identity(value):
    """Check classifications/order without comparing process-specific addresses."""
    keys = {'name', 'case', 'index', 'family', 'variant', 'status', 'reference_status', 'native_executed'}
    result = {key: value[key] for key in keys if key in value}
    for key in ('variants', 'cases', 'records', 'rows'):
        if isinstance(value.get(key), list):
            result[key] = [row_identity(row) for row in value[key] if isinstance(row, dict)]
    return result


def same_summary(current, frozen):
    """The frozen source-bound report fixes scalar counters and classifications.

    Full fresh reports remain in the aggregate. This check does not manufacture
    numerical expectations: the unchanged child runner performs those checks.
    """
    require(set(current) == set(frozen), ('report schema fields changed', set(current) ^ set(frozen)))
    for key, value in frozen.items():
        if not isinstance(value, (dict, list)):
            require(type(current[key]) is type(value) and current[key] == value,
                    ('frozen scalar summary differs', key, current[key], value))
    require(row_identity(current) == row_identity(frozen), 'case identities/order/classifications changed')


def bind_runtime_fields(result):
    source_fields = ('source_sha256', 'runtime_source_sha256', 'native_source_sha256', 'current_source_sha256')
    image_fields = ('native_sha256', 'executable_sha256', 'native_executable_sha256', 'image_sha256', 'current_executable_sha256')
    sources = [result[key] for key in source_fields if key in result]
    images = [result[key] for key in image_fields if key in result]
    require(sources and all(x == SOURCE for x in sources), 'child source identity')
    require(images and all(x == IMAGE for x in images), 'child executable identity')


def manifest_inputs(repo, custody):
    result = []
    for name, expected, count in MANIFESTS:
        path = relative_file(repo, name)
        custody.bind(path, expected)
        value = custody.json(path)
        files = value['files']
        rows = [dict(item, path=key) for key, item in files.items()] if isinstance(files, dict) else files
        require(len(rows) == count and len({row['path'] for row in rows}) == count, ('manifest cardinality', name))
        relocations = {}
        for row in rows:
            installed = MANIFEST_RELOCATIONS.get((name, row['path']), row['path'])
            custody.bind(relative_file(repo, installed), row['sha256'], row['bytes'], row.get('mode'))
            if installed != row['path']:
                relocations[row['path']] = installed
        result.append({'path': name, 'sha256': expected, 'file_count': count,
                       'explicit_document_only_path_relocations': relocations,
                       'all_file_bytes_hashes_and_declared_modes_verified': True})
    for name, expected in EXTRA_INPUTS.items():
        custody.bind(relative_file(repo, name), expected)
    return result


def check_units(build, rows, custody):
    require(len({row['name'] for row in rows}) == len(rows), 'duplicate module name')
    for row in rows:
        require(Path(row['name']).name == row['name'], 'unsafe module name')
        for suffix, field in (('.hepta', 'source_sha256'), ('.o', 'object_sha256')):
            custody.bind(build / (row['name'] + suffix), row[field],
                         row.get('source_bytes' if suffix == '.hepta' else 'object_bytes'))
    return rows


def check_parent(build, kind, profile, peer, custody):
    pin = profile['parents'][kind]
    receipt = custody.json(build / 'build.json')
    require(receipt['status'] == 'PASS' and receipt['schema'] in pin['accepted_receipt_schemas'], 'parent build schema/status')
    require(receipt[pin['receipt_source_field']] == pin['source_sha256'], 'parent source receipt')
    require(receipt[pin['receipt_native_field']] == pin['native_sha256'], 'parent executable receipt')
    custody.bind(build / 'runtime.hepta', pin['source_sha256'])
    names = ('tensor-gradient-object-session', 'tensor-gradient-incoming-session') if kind == 'gradient' else ('tensor-adam-distributions-session',)
    images = [build / name for name in names if (build / name).is_file()]
    require(images, ('missing parent native executable', str(build)))
    for image in images:
        custody.bind(image, pin['native_sha256'])
    check_units(build, peer['units'], custody)
    actual = {row['name']: row for row in receipt['units']}
    require(set(actual) == {row['name'] for row in peer['units']}, 'parent module names')
    for expected in peer['units']:
        for key in ('source_sha256', 'object_sha256', 'functions', 'blocks', 'instructions'):
            require(actual[expected['name']][key] == expected[key], ('parent unit receipt', key))
    return {'path': str(build), 'receipt_sha256': digest(build / 'build.json'),
            'source_sha256': pin['source_sha256'], 'native_sha256': pin['native_sha256'],
            'module_count': len(actual)}


def check_current(build, profile, custody):
    current = custody.json(build / 'build.json')
    require(current['schema'] == 'heptatorch.gradient-adam-distributions-native-build.v1', 'current build schema')
    require(current['status'] == 'PASS' and current['link_exit'] == 0, 'current native build failed')
    for key in ('source_sha256', 'native_sha256', 'native_bytes', 'functions', 'blocks', 'instructions',
                'state_format', 'backing_format', 'descriptor_bytes', 'compiler_sha256', 'linker_sha256', 'parents', 'units'):
        require(current[key] == profile[key], ('current profile differs', key))
    require(current['module_count'] == len(current['units']) == 9, 'current nine native units required')
    custody.bind(build / 'runtime.hepta', SOURCE)
    custody.bind(build / IMAGE_NAME, IMAGE, profile['native_bytes'])
    check_units(build, current['units'], custody)
    return current


def child_record(path, expected_path, custody, runtime=True):
    actual = custody.json(path)
    expected = custody.json(expected_path)
    same_summary(actual, expected)
    if runtime:
        bind_runtime_fields(actual)
    return {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size,
            'frozen_report_path': str(expected_path), 'frozen_report_sha256': digest(expected_path),
            'report': actual}


def reference_binding(child, path, custody):
    ref = reference(path, custody)
    expected = digest(path)
    fields = ('reference_sha256', 'observations_sha256', 'actual_reference_sha256', 'existing_actual_reference_sha256')
    seen = [child[key] for key in fields if key in child]
    require(seen and all(value == expected for value in seen), 'child fixed-reference identity')
    if 'oracle_provenance' in child:
        require(child['oracle_provenance'] == ref['oracle_provenance'], 'child provenance changed')
    if 'actual_reference_provenance_checks' in child:
        require(child['actual_reference_provenance_checks'] == ref['oracle_provenance']['checks'], 'child provenance checks')
    return ref


def distribution_ledger(report, ref, explicit):
    require(report['reference_cases'] == len(report['rows']) == len(ref['rows']) == 183, 'fixed distribution corpus')
    for index, (observed, original) in enumerate(zip(report['rows'], ref['rows'])):
        require((observed['index'], observed['name'], observed['family']) ==
                (index, original['name'], original['family']), 'distribution index/name/family')
        status = ('OPEN_UNREPRESENTED' if index in OPEN_DISTRIBUTIONS else
                  'OPEN_TRANSACTION_DIFFERENCE' if not explicit and index in ATOMIC_DIFFERENCES else 'PASS')
        require(observed['status'] == status, ('distribution classification', index))
    expected = {'OPEN_UNREPRESENTED': 18, 'PASS': 165} if explicit else {'OPEN_TRANSACTION_DIFFERENCE': 4, 'OPEN_UNREPRESENTED': 18, 'PASS': 161}
    require(dict(Counter(row['status'] for row in report['rows'])) == report['counts'] == expected, 'distribution counts')
    require(report['all_rng_complete'] is False and report['full_pytorch_rewrite_complete'] is False, 'distribution scope remains partial')


def optimizer_union(reports, ref):
    original = reports['original-optimizer-24-mapping']['report']['rows']
    adam = reports['owned-adam-cases21-24']['report']['rows']
    case20 = reports['optimizer-gradient-case20']['report']
    require(len(original) == len(ref['rows']) == 24 and len(adam) == 4, 'original optimizer24 cardinality')
    qualified, partial = set(), set()
    for index, (row, expected) in enumerate(zip(original, ref['rows']), 1):
        require(row['case'] == expected['case'], ('optimizer identity', index))
        if row['status'] == 'QUALIFIED_EXPOSED_SCOPE':
            qualified.add(index)
        elif row['status'] == 'PARTIAL_WITH_RETAINED_NONLEAF_OPEN':
            partial.add(index)
        else:
            require(row['status'] == 'OPEN_NOT_EXECUTED' and 20 <= index <= 24, 'optimizer original mapping boundary')
    for row, index in zip(adam, range(21, 25)):
        require(row['index'] == index and row['case'] == ref['rows'][index - 1]['case'] and
                row['status'] == 'QUALIFIED_EXPOSED_SCOPE', 'Adam subset identity')
        qualified.add(index)
    require(case20['reference_case_index'] == 20 and case20['reference_case'] == ref['rows'][19]['case'] and
            case20['status'] == 'QUALIFIED_EXPOSED_SCOPE' and case20['mismatch_fields'] == 0, 'additive case20 identity')
    qualified.add(20)
    require(qualified == set(range(1, 25)) - {14} and partial == {14}, '23 qualified plus one partial original case')
    return {'original_reference_cases': 24, 'qualified_exposed_scope': 23, 'partial_cases': 1,
            'qualified_one_based_indices': sorted(qualified), 'partial_one_based_indices': [14],
            'open_subobservation': 'retain_grad/nonleaf optimizer membership',
            'original_18_plus_1_partial_plus_5_open_report_unchanged': True,
            'case20_and_21_to_24_are_additive_mappings_not_new_reference_cases': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    for name in ('gradient-build', 'adam-distribution-build', 'stack-build', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--producer-review', type=Path,
                        help='Reuse an existing completed 17-step execution; verify all original children and native files.')
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'executed-integration.py').write_bytes(Path(__file__).resolve().read_bytes())
    custody = Custody()
    report = {'schema': 'heptatorch.gradient-adam-distributions-integration-qualification.v1',
              'status': 'RUNNING', 'source_sha256': SOURCE, 'native_sha256': IMAGE,
              'integration_script_sha256': digest(Path(__file__).resolve()),
              'role': 'Integration orchestration of immutable producer and independent review packages; no new independent semantic review.',
              'producer_executed_this_invocation': args.producer_review is None,
              'execution_steps': [], 'producer': None, 'native_review': None, 'peer_review': {},
              'extreme_stride': {}, 'readiness': 'HOLD', 'all_rng_complete': False,
              'full_optimizer_complete': False, 'full_rewrite_complete': False,
              'pytorch_api_denominator': 2590, 'new_Torch_executed_here': False,
              'full_32_live_Torch_executed_here': False, 'full_35_live_Torch_executed_here': False,
              'counting_boundary': '17 producer steps, 3 native ownership gates, 4 independent runners and strict extreme8 overlap; their counts are never added as unique PyTorch APIs.'}

    def save():
        temporary = out / 'report.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(out / 'report.json')

    def execute(name, script, arguments, target, expected_report, timeout=1800):
        custody.bind(script)
        command = [sys.executable, '-I', '-S', '-B', str(script), *map(str, arguments), '--output', str(target)]
        print('EXECUTING ' + name, flush=True)
        completed = subprocess.run(command, cwd=repo, capture_output=True, timeout=timeout)
        stdout, stderr = out / (name + '.stdout'), out / (name + '.stderr')
        stdout.write_bytes(completed.stdout)
        stderr.write_bytes(completed.stderr)
        item = {'name': name, 'command': command, 'exit_code': completed.returncode,
                'stdout_path': stdout.name, 'stdout_sha256': sha(completed.stdout),
                'stderr_path': stderr.name, 'stderr_sha256': sha(completed.stderr),
                'report_path': str(expected_report),
                'report_sha256': digest(expected_report) if expected_report.is_file() else None}
        report['execution_steps'].append(item)
        save()
        require(completed.returncode == 0, ('runner failed', name, completed.returncode, completed.stderr.decode(errors='replace')[-3000:]))
        require(expected_report.is_file(), ('runner did not create report', name))
        custody.bind(stdout, item['stdout_sha256'])
        custody.bind(stderr, item['stderr_sha256'])
        return custody.json(expected_report)

    try:
        custody.bind(Path(__file__).resolve(), report['integration_script_sha256'])
        report['frozen_input_manifests'] = manifest_inputs(repo, custody)
        profile = custody.json(repo / 'tools/gradient_adam_distributions/profile.json')
        require(profile['source_sha256'] == SOURCE and profile['native_sha256'] == IMAGE, 'candidate05 profile')
        peer_pins = custody.json(repo / 'tests/gradient_composition_review/handoff-peer-pins-current.json')
        require(peer_pins['current_source_sha256'] == SOURCE and peer_pins['current_executable_sha256'] == IMAGE, 'native peer profile')
        parent_gradient, parent_adam = args.gradient_build.resolve(), args.adam_distribution_build.resolve()
        report['parent_builds'] = {
            'gradient355': check_parent(parent_gradient, 'gradient', profile, peer_pins['peers']['incoming355'], custody),
            'adam_distributions378': check_parent(parent_adam, 'adam_distribution', profile, peer_pins['peers']['adamdist378'], custody)}
        stack = args.stack_build.resolve()
        for name, expected in (('stack-module-compiler', COMPILER), ('stack-linker-session', SESSION_LINKER), ('stack-linker-runtime', RUNTIME_LINKER)):
            custody.bind(stack / 'native' / name, expected)
        producer = args.producer_review.resolve() if args.producer_review else out / 'producer'
        if args.producer_review is None:
            execute('producer-17', repo / 'tools/gradient_adam_distributions/qualify.py', [
                '--gradient-build', parent_gradient, '--adam-distribution-build', parent_adam,
                '--compiler', stack / 'native/stack-module-compiler', '--linker', stack / 'native/stack-linker-session'],
                producer, producer / 'report.json', timeout=7200)
        original = custody.json(producer / 'report.json')
        frozen_producer = custody.json(repo / PRODUCER_EXPECTED / 'portable-qualification.json')
        same_summary(original, frozen_producer)
        bind_runtime_fields(original)
        require(original['step_count'] == 17 and [step['name'] for step in original['steps']] == [x[0] for x in STEPS], 'producer17 exact sequence')
        native = producer / 'native'
        current_build = check_current(native, profile, custody)
        children = {}
        refs_dir = repo / 'validation/gradient-adam-distributions/reference'
        for step, (name, script, ref_name) in zip(original['steps'], STEPS):
            require(step['exit_code'] == 0, ('producer step failed', name))
            expected_script = repo / ('tools/gradient_adam_distributions' if name == 'cold-native-build' else 'tests/gradient_adam_distributions_review') / script
            require(step['command'][1:4] == ['-I', '-S', '-B'] and Path(step['command'][4]).name == script,
                    ('producer command script', name))
            # Reuse is permitted only while the original executed script is still
            # available and byte-identical to the installed pinned package.
            custody.bind(Path(step['command'][4]), digest(expected_script))
            for stream in ('stdout', 'stderr'):
                custody.bind(producer / (name + '.' + stream), step[stream + '_sha256'])
            child_path = relative_file(producer, step['report_path'])
            custody.bind(child_path, step['report_sha256'])
            require(original['suite_reports'][name] == {'path': step['report_path'], 'sha256': step['report_sha256']}, 'producer child receipt map')
            child = child_record(child_path, repo / PRODUCER_EXPECTED / (name + '.json'), custody)
            require(child['report']['status'] == step['report_status'], 'producer child status')
            if ref_name:
                ref = reference_binding(child['report'], refs_dir / ref_name, custody)
                if name.startswith('distribution-'):
                    distribution_ledger(child['report'], ref, name == 'distribution-explicit-183')
            children[name] = child
        bridge_path = producer / 'distribution-build-field-bridge.json'
        custody.bind(bridge_path, original['distribution_receipt_bridge']['sha256'])
        bridge = custody.json(bridge_path)
        require(bridge == dict(current_build, runtime_source_sha256=SOURCE, executable_sha256=IMAGE), 'distribution bridge only adds the two declared field aliases')
        require(original['distribution_receipt_bridge']['original_cold_build_sha256'] == digest(native / 'build.json'), 'bridge original build binding')
        report['producer'] = {'mode': 'executed_here' if args.producer_review is None else 'verified_existing_execution',
                              'directory': str(producer), 'native_directory': str(native),
                              'original_report_sha256': digest(producer / 'report.json'),
                              'original_report_bytes': (producer / 'report.json').stat().st_size,
                              'original_report': original, 'children': children}
        report['optimizer_reference_union'] = optimizer_union(children, reference(refs_dir / 'torch-optimizer-objects.json', custody))
        report['distribution_reference_ledger'] = {
            'reference_sha256': digest(refs_dir / 'torch-bernoulli-dropout.json'),
            'original_cases': 183, 'atomic': {'qualified': 161, 'transaction_differences': 4, 'unrepresented': 18},
            'explicit_partial_commands': {'qualified': 165, 'unrepresented': 18},
            'late_error_zero_based_indices': ATOMIC_DIFFERENCES,
            'unrepresented_zero_based_indices': OPEN_DISTRIBUTIONS,
            'old80_to85_contract_unchanged': True, 'new86_87_are_explicit_contracts': True}
        report['legacy_regression_boundaries'] = {
            'presence8': {'state_version_differences': 8, 'state_creation_differences': 65},
            'gradient_clear10': {'state_version_differences': 11, 'state_creation_differences': 9,
                                'missing_step_Tensor_observations': 30, 'Adam_different_bits_with_original_tolerance': 6},
            'old_atomic_controls': 'Historical empty86/87 unknown4 refusal conflicts and inverse adaptation remain frozen in the parent378 package; this integration does not relabel or rerun that original negative suite.',
            'full_optimizer_complete': False, 'all_rng_complete': False}
        save()

        native_out = out / 'native-review'
        native_top = execute('native-review-3', repo / 'tests/gradient_composition_review/qualify.py', [
            '--build', native, '--stack-build', stack, '--atomic-tools', repo / 'tools/cpu_distributions',
            '--gradient-build', parent_gradient, '--adam-distributions-build', parent_adam], native_out, native_out / 'report.json')
        require(native_top['status'] == 'PASS' and native_top['suite_count'] == 3 and native_top['input_custody_after'] is True, 'three independent native gates')
        bind_runtime_fields(native_top)
        native_children = {}
        for step, (name, frozen_name) in zip(native_top['suites'], (
            ('recompile', 'recompile-combined-05.json'), ('owned-entry', 'owned-entry-combined-05.json'), ('typed-handoffs', 'handoff-combined-05.json'))):
            require(step['name'] == name and step['exit'] == 0 and step['status'] == 'PASS', 'native gate execution')
            path = relative_file(native_out, step['report_path'])
            custody.bind(path, step['report_sha256'])
            native_children[name] = child_record(path, repo / NATIVE_EXPECTED / frozen_name, custody)
            for stream in ('stdout', 'stderr'):
                custody.bind(native_out / (name + '.' + stream))
        require(len(native_top['suites']) == 3, 'three native gate cardinality')
        recompile = native_children['recompile']['report']
        require(recompile['complete_core_body_count'] == len(recompile['core_body_bindings']) == 428 and
                len(recompile['units']) == 9 and recompile['source_to_units_exact_once'] is True and
                recompile['all_object_and_image_bytes_identical'] is True, 'actual428 bodies nine modules exact coverage')
        require(recompile['core_body_bindings'] == custody.json(repo / NATIVE_EXPECTED / 'recompile-combined-05.json')['core_body_bindings'], 'all body hashes/unit membership unchanged')
        report['native_review'] = {'original_report_sha256': digest(native_out / 'report.json'),
                                   'original_report': native_top, 'children': native_children}
        save()

        peer_tests = repo / 'tests/adam_distributions_gradient_review'
        actual_dir = repo / 'validation/adam-distributions-gradient-review/actual'
        base_args = ['--binary', native / IMAGE_NAME, '--source', native / 'runtime.hepta',
                     '--protocol', peer_tests / 'protocol.py', '--expected-native-sha256', IMAGE,
                     '--expected-source-sha256', SOURCE]
        peer_calls = (
            ('zero-merge-mathematical', 'test_zero_merge_layout.py', 'zero-merge-mathematical-05.json', None),
            ('actual-zero-merge', 'replay_zero_merge_layout.py', 'actual-zero-merge-05.json', actual_dir / 'torch-zero-merge-layout.json'),
            ('actual-scalar-add-nan', 'replay_scalar_add_nan.py', 'actual-scalar-add-nan-05.json', actual_dir / 'torch-scalar-add-nan.json'),
            ('native-cross', 'test_joint_gradients_v2.py', 'native-cross-05.json', repo / 'tools/cpu_distributions/evidence/torch-bernoulli-dropout.json'),
        )
        for name, filename, frozen_name, ref_path in peer_calls:
            arguments = list(base_args)
            if ref_path:
                reference(ref_path, custody)
                arguments += (['--distribution-observations', ref_path] if name == 'native-cross' else
                              ['--observations', ref_path, '--expected-observations-sha256', digest(ref_path)])
            target = out / name
            execute(name, peer_tests / filename, arguments, target, target / 'report.json')
            child = child_record(target / 'report.json', repo / PEER_EXPECTED / frozen_name, custody)
            if ref_path:
                reference_binding(child['report'], ref_path, custody)
            report['peer_review'][name] = child
            save()

        extreme_dir = repo / 'validation/gradient-composition-review/extreme-stride'
        observed = out / 'extreme-stride-native'
        execute('extreme-stride-observe', repo / 'tests/gradient_composition_review/observe_extreme_stride_native.py', [
            '--build', native, '--protocol', repo / 'tools/cpu_distributions/protocol.py',
            '--prepared', extreme_dir / 'prepared.json', '--expected-source', SOURCE, '--expected-image', IMAGE],
            observed, observed / 'report.json')
        observation = child_record(observed / 'report.json', repo / NATIVE_EXPECTED / 'extreme-stride-observations-05.json', custody)
        require(observation['report']['status'] == 'OBSERVED_NOT_TORCH_QUALIFIED' and observation['report']['cases'] == 8 and observation['report']['frames'] == 80, 'native8 observation scope')
        comparison = out / 'extreme-stride-comparison'
        execute('extreme-stride-compare', repo / 'tests/gradient_composition_review/compare_extreme_stride.py', [
            '--reference', extreme_dir / 'torch-extreme-stride.json', '--native-observations', observed / 'report.json',
            '--expected-source', SOURCE, '--expected-image', IMAGE], comparison, comparison / 'report.json')
        # The only dynamic scalar in this comparison is the hash of its fresh
        # observation receipt, which contains fresh process observations.
        strict = custody.json(comparison / 'report.json')
        frozen_strict = custody.json(repo / NATIVE_EXPECTED / 'extreme-stride-comparison-05.json')
        require(strict['native_observations_sha256'] == digest(observed / 'report.json'), 'strict comparison consumes the actual observation')
        comparable = dict(strict, native_observations_sha256=frozen_strict['native_observations_sha256'])
        same_summary(comparable, frozen_strict)
        bind_runtime_fields(strict)
        ref = reference(extreme_dir / 'torch-extreme-stride.json', custody)
        require(strict['reference_sha256'] == digest(extreme_dir / 'torch-extreme-stride.json') and
                strict['oracle_provenance_checks'] == ref['oracle_provenance']['checks'], 'strict8 fixed actual reference')
        require((strict['cases'], strict['passing_cases'], strict['failing_cases'], strict['field_comparisons'], strict['different_fields']) == (8, 8, 0, 344, 0), 'strict8 all43fields')
        report['extreme_stride'] = {
            'observation': observation,
            'comparison': {'path': str(comparison / 'report.json'), 'sha256': digest(comparison / 'report.json'),
                           'bytes': (comparison / 'report.json').stat().st_size,
                           'frozen_report_sha256': digest(repo / NATIVE_EXPECTED / 'extreme-stride-comparison-05.json'),
                           'report': strict}}
        history = repo / 'validation/gradient-composition-review/candidate-04/extreme-stride-comparison-04.json'
        historical = custody.json(history)
        require(historical['status'] == 'FAIL' and historical['failing_cases'] == 4 and historical['passing_cases'] == 4, 'original04 real stride failure retained')
        report['retained_history'] = {'candidate04_extreme_stride_failure': {
            'path': str(history.relative_to(repo)), 'sha256': digest(history), 'report': historical},
            'all_producer_and_peer_historical_payload_bytes_verified_by_frozen_manifests': True,
            'historical_archive_members_not_reexecuted_or_reclassified': True}
        custody.finish()
        report.update(status='QUALIFIED_COMPONENT_WITH_OPEN_SCOPE', input_custody_after=True,
                      input_file_count=len(custody.files),
                      checked_inputs={str(path): value for path, value in sorted(custody.files.items())},
                      producer_steps_verified=17, native_gates_executed=3,
                      independent_runners_executed=4, strict_extreme_stride_cases=8,
                      strict_extreme_stride_fields=344,
                      native_execution_suites_invoked_beyond_producer=8,
                      strict_comparison_invocations_without_additional_native_execution=1)
        save()
        print(json.dumps({'status': report['status'], 'report': str(out / 'report.json'),
                          'sha256': digest(out / 'report.json'), 'producer_steps_verified': 17,
                          'native_gates_executed': 3, 'independent_runners_executed': 4,
                          'strict_extreme_stride_cases': 8, 'readiness': 'HOLD',
                          'full_rewrite_complete': False}, indent=2))
        return 0
    except Exception as error:
        report.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
        save()
        print(report['traceback'], file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
