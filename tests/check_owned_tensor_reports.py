"""Bind actual dynamic Tensor native gates to the frozen candidate and reviewers."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--native', type=Path, required=True)
p.add_argument('--review', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
package = ROOT / 'validation/owned-tensor-package.json'
assert digest(package) == '5385ada6597a8f556f722b12273cf035b003395bd7bf1337c0204e21c05f05b1'
for relative, row in read(package)['files'].items():
    path = ROOT / relative
    assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], relative
for manifest, directory, expected in (
        ('owned-tensor-review-package.json', 'tests/owned_tensor_review', 'ca87a420983fb9ae1bf0d47be34fdb752cc718a6ae58a748f9f16d7552261ddf'),
        ('owned-tensor-oracle-package.json', 'tests/owned_tensor_oracle', 'f45d7399235f0bd8794b7175f8c8b2c863dd88c6da0a8a1d4267ea0b70d58569')):
    path = ROOT / 'validation' / manifest
    assert digest(path) == expected
    for row in read(path)['files']:
        target = ROOT / directory / row['path']
        assert target.stat().st_size == row['bytes'] and digest(target) == row['sha256'], target
profile = read(ROOT / 'tools/owned_tensors/profile.json')
source_map = read(ROOT / 'tools/owned_tensors/source-map.json')
build = read(a.native / 'build.json')
assert build['pass'] and build['numerics_compiled_and_linked_natively']
assert build['canonical_source_sha256'] == digest(ROOT / 'src/heptatorch.hepta') == profile['canonical_source_sha256']
assert build['candidate_source_sha256'] == digest(a.native / 'runtime.hepta') == profile['core_overlay_sha256']
assert build['executable_sha256'] == digest(a.native / 'owned-tensor-session') == profile['executable_sha256']
assert build['canonical_function_count'] == 143 and build['candidate_function_count'] == 150
assert build['canonical_functions_byte_identical'] == source_map['original_byte_identical'] == 126
assert build['function_source_mapping'] == source_map['canonical_functions']
for path, expected in build['input_sources'].items():
    assert digest(path) == expected, path
for row in build['unit_rows']:
    expected = profile['units'][row['name']]
    assert digest(a.native / 'units' / (row['name'] + '.hepta')) == row['source_sha256'] == expected['source_sha256']
    assert digest(a.native / (row['name'] + '.o')) == row['object_sha256'] == expected['object_sha256']

reports, identities = {}, {}
for label in ('native', 'typed', 'independent'):
    path = a.review / label / 'report.json'
    reports[label] = read(path)
    identities[label] = digest(path)
native = reports['native']
assert native['pass'] and native['cases'] == len(native['rows']) == 11
assert native['response_frames'] == 161
assert native['compared_binary32_values'] == 1078028
assert native['rollback_compared_owned_bytes'] == 59310080
assert native['executable_sha256'] == profile['executable_sha256']
assert native['core_overlay_sha256'] == profile['core_overlay_sha256']
assert native['test_sha256'] == digest(ROOT / 'tools/owned_tensors/test_dynamic.py')
assert native['wire_helper_sha256'] == digest(ROOT / 'tools/owned_tensors/protocol.py')
prior = read(ROOT / 'tools/owned_tensors/evidence/portable-native-report.json')
without_timing = lambda rows: [{k: v for k, v in row.items() if k != 'elapsed_seconds'} for row in rows]
assert without_timing(native['rows']) == without_timing(prior['rows'])
typed = reports['typed']
assert typed == read(ROOT / 'tools/owned_tensors/evidence/typed-state-report.json')
assert typed['pass'] and typed['cases'] == 6 and all(row['pass'] for row in typed['rows'])
independent = reports['independent']
assert independent['status'] == 'PASS'
assert independent['candidate_executable_sha256'] == profile['executable_sha256']
assert independent['canonical_executable_sha256'] == digest(ROOT / 'build/heptatorch-monolithic')
assert independent['interactive_frames'] == 16 and independent['exact_scalar_or_f32_fields_compared'] == 612
assert independent['entire_owner_rollback_bytes_compared'] == 41943040
assert independent['responses_before_stdin_eof'] and independent['native_target_execution']
assert independent['transport_observer_sha256'] == digest(ROOT / 'tools/owned_sessions/client.py')
executed = ROOT / 'tests/owned_tensor_review/candidate-03-review/executed-sources'
assert independent['review_script_sha256'] == digest(executed / 'review_dynamic.py')
assert independent['wire_encoder_sha256'] == digest(executed / 'protocol.py')
original = read(ROOT / 'tests/owned_tensor_review/candidate-03-review/report.json')
assert independent['numerical_checks'] == original['numerical_checks']

result = {'schema': 'heptatorch.hosted-owned-tensor-native.v1', 'status': 'PASS',
          'package_manifest_sha256': digest(package), 'profile_sha256': digest(ROOT / 'tools/owned_tensors/profile.json'),
          'canonical_source_sha256': profile['canonical_source_sha256'],
          'native_executable_sha256': profile['executable_sha256'],
          'build': build, 'report_sha256': identities, 'reports': reports,
          'hosted_pytorch_qualified_by_this_report': False,
          'independent_storage_version_objects_integrated': False,
          'whole_pytorch_rewrite_complete': False,
          'scope': 'CPU binary32 numerical Tensor/AD/optimizer execution across genuinely owned allocation and descriptor growth; the strict pinned PyTorch counter gate is separate.'}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
