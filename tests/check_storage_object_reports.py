"""Bind actual native storage/owner executions to their frozen source packages."""
from pathlib import Path
import argparse
import hashlib
import json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def passed(report):
    return report.get('status') == 'PASS' or report.get('pass') is True


p = argparse.ArgumentParser(description=__doc__)
for name in ('session-build', 'session-review', 'native', 'interactive',
             'trace-review', 'interactive-review', 'guard-review', 'model-review', 'output'):
    p.add_argument('--' + name, type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
storage = root / 'tools/storage_objects'
manifest_path = root / 'validation/storage-object-handoff.json'
assert digest(manifest_path) == 'ca5a3354476c24c57c099767f47097ac8285a04303956ecfe2fe4eadbc95bc9b'
manifest = read(manifest_path)
assert manifest['file_count'] == len(manifest['files']) == 456
assert {str(f.relative_to(storage)) for f in storage.rglob('*') if f.is_file()} == set(manifest['files'])
for relative, expected in manifest['files'].items():
    path = storage / relative
    assert path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'], relative
assert digest(storage / 'source-lock.json') == '4c196af5c08d94a15d5ff838e39d8b33350f91bf83800d62bc2eceb6f0ddc8db'
assert digest(root / 'tests/owned_storage_independent.py') == '8df1bd0cb273ede9f4face7775ec4d854903c1768cdd0d8157d09257165d7fa0'
lock = read(storage / 'source-lock.json')
native = read(a.native / 'build.json')
interactive = read(a.interactive / 'build.json')
assert native['status'] == interactive['status'] == 'PASS'
assert native == read(storage / 'candidate-04/build.json')
assert native['modules'] == lock['source_modules']
assert digest(a.native / 'storage-trace') == native['executable_sha256'] == lock['finite_trace_executable_sha256']
assert digest(a.interactive / 'storage-session') == interactive['executable_sha256'] == lock['interactive_executable_sha256']
assert interactive['linker_sha256'] == lock['interactive_linker_sha256']
assert interactive['module_build_receipt_sha256'] == digest(a.native / 'build.json')
for name, row in native['modules'].items():
    assert digest(a.native / (name + '.hepta')) == row['source_sha256']
    assert digest(a.native / (name + '.o')) == row['object_sha256']

reports = {}
identities = {}
for label, directory, frozen in (
        ('trace', a.trace_review, 'qualification-01'),
        ('interactive', a.interactive_review, 'interactive-qualification-01')):
    path = directory / 'report.json'
    report = read(path)
    expected = read(storage / frozen / 'report.json')
    assert report['status'] == 'PASS'
    assert report['cases'] == len(report['rows']) == 12
    assert report['frames'] == 63 and report['scalar_fields_compared'] == 171
    assert report['state_dumps_independently_checked'] == 22
    assert report['binary_sha256'] == expected['binary_sha256']
    assert report['qualification_script_sha256'] == digest(storage / 'qualify.py')
    assert report['wire_codec_sha256'] == digest(storage / 'protocol.py')
    assert report['rows'] == expected['rows']
    assert all(row['pass'] and row['exit_code'] == 0 for row in report['rows'])
    if label == 'interactive':
        assert all(row['all_responses_before_stdin_eof'] for row in report['rows'])
    reports[label] = report
    identities[label] = {'sha256': digest(path), 'cases': 12, 'frames': 63}
assert [r['observed_records'] for r in reports['trace']['rows']] == [r['observed_records'] for r in reports['interactive']['rows']]

guard_path = a.guard_review / 'report.json'
guards = read(guard_path)
old_guards = read(storage / 'extreme-state-qualification-01/report.json')
assert guards['status'] == 'PASS' and guards['cases'] == 5
assert guards['rows'] == [row for row in old_guards['rows'] if not row['preserved_legacy_refusal']]
assert all(row['pass'] for row in guards['rows'])
reports['current-extreme-guards'] = guards
identities['current-extreme-guards'] = {'sha256': digest(guard_path), 'cases': 5}

model_path = a.model_review / 'report.json'
model = read(model_path)
frozen_models = read(root / 'validation/storage-independent-models.json')
expected_model = next(row['report'] for row in frozen_models['reports']
                      if row['report']['executable_sha256'] == lock['interactive_executable_sha256'])
assert model == expected_model
assert model['status'] == 'PASS' and model['native_unchanged']
assert model['interactive_frames'] == 2952 and model['campaigns'] == 3
assert sum(row['exact_scalar_comparisons'] for row in model['rows']) == 11177
assert sum(row['full_state_rollback_comparisons'] for row in model['rows']) == 93
for row in model['rows']:
    directory = a.model_review / str(row['seed'])
    assert row['iterations'] == 300 and row['status'] == 'PASS'
    assert row['all_references_released'] and row['all_observed_before_stdin_eof']
    assert digest(directory / 'requests.bin') == row['request_transcript_sha256']
    assert digest(directory / 'responses.bin') == row['response_transcript_sha256']
reports['independent-model'] = model
identities['independent-model'] = {'sha256': digest(model_path), 'campaigns': 3, 'frames': 2952}

session_manifest = root / 'validation/owned-session-package.sha256'
assert digest(session_manifest) == '7e7bf68f3c9d6bf9252c7544f50f36e3c8e3570da4fe3259efa7f7c421cd0263'
for line in session_manifest.read_text().splitlines():
    expected, relative = line.split('  ', 1)
    assert digest(root / relative) == expected, relative
profile = read(root / 'tools/owned_sessions/profile.json')
rebuilt = read(a.session_build / 'rebuild.json')
qualified = read(a.session_review / 'qualification.json')
assert rebuilt['status'] == qualified['status'] == 'PASS'
assert qualified['cases'] == 131
assert qualified['production_service_sha256'] == lock['interactive_executable_sha256']
assert digest(a.session_review / 'service/service') == lock['interactive_executable_sha256']
assert qualified['production_and_signal_object_text_identical']
for label in ('production', 'qualification', 'signal_qualification'):
    assert digest(a.session_build / (label + '-native/owned-session-linker')) == profile[label]['linker_sha256']
expected_suites = {'entry-admission': 103, 'transport': 20, 'returned-length': 6, 'real-signals': 2}
assert {row['suite']: row['cases'] for row in qualified['rows']} == expected_suites
for row in qualified['rows']:
    path = a.session_review / row['report']
    report = read(path)
    assert row['status'] == 'PASS' and digest(path) == row['report_sha256']
    assert passed(report) and report['cases'] == row['cases']
    reports['owned-session-' + row['suite']] = report
    identities['owned-session-' + row['suite']] = {'sha256': digest(path), 'cases': row['cases']}

result = {'schema': 'heptatorch.hosted-owned-storage-gates.v1', 'status': 'PASS',
          'source_sha256': digest(root / 'src/heptatorch.hepta'),
          'storage_package_manifest_sha256': digest(manifest_path),
          'owned_session_manifest_sha256': digest(session_manifest),
          'storage_source_lock_sha256': digest(storage / 'source-lock.json'),
          'independent_model_script_sha256': digest(root / 'tests/owned_storage_independent.py'),
          'compiler_sha256': lock['compiler_sha256'],
          'native_build': native, 'interactive_build': interactive,
          'owned_session_rebuild': rebuilt, 'owned_session_qualification': qualified,
          'report_identities': identities, 'reports': reports,
          'trace_interactive_observed_records_identical': True,
          'preserved_legacy_guard_is_not_counted_as_current_execution': True,
          'real_tensor_kernel_integration_complete': False,
          'whole_pytorch_rewrite_complete': False,
          'native_schema_inventory_count': 2590,
          'scope': 'Native owned byte storage, independent logical identities and lifetimes, growable pool, frame transactions and owner-aware transport; numerical Tensor storage integration remains separate.'}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
