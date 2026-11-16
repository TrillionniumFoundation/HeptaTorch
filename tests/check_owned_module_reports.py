"""Bind the owned-resource CI campaign to the frozen native profile."""
from pathlib import Path
import argparse
import hashlib
import json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build', type=Path, required=True)
parser.add_argument('--review', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
manifest = Path('validation/owned-module-package.sha256')
assert digest(manifest) == '71431037074b5677c34ef7f58b54eeafae5e72b1d1176beed1e0af6218affd5d'
lines = manifest.read_text().splitlines()
assert len(lines) == 23
for line in lines:
    expected, path = line.split('  ', 1)
    assert digest(path) == expected, path
pins = read('tools/owned_modules/profile.json')
original = read('validation/native-owned-modules.json')
rebuilt = read(args.build / 'rebuild.json')
qualification = read(args.review / 'qualification.json')
assert rebuilt['status'] == qualification['status'] == original['status'] == 'PASS'
assert rebuilt['native_reproduction'] == original['native_reproduction']
assert qualification['compiler_sha256'] == pins['module_compiler_executable_sha256']
assert qualification['runtime_linker_sha256'] == pins['linker_profiles']['abi2-w1m']['executable_sha256']
assert qualification['counted_native_cases'] == 385
assert digest(args.build / 'native/owned-module-compiler') == pins['module_compiler_executable_sha256']
for label, row in pins['linker_profiles'].items():
    assert digest(args.build / ('native/linker-' + label)) == row['executable_sha256']
expected_suites = {'lifecycle': 1, 'source-lifetimes': 34, 'release-startup': 8,
                   'allocator': 13, 'contracts': 315, 'independent-flow': 14,
                   'runtime-equivalence': None}
assert len(qualification['rows']) == len(expected_suites)
assert {row['suite']: row['cases'] for row in qualification['rows']} == expected_suites
reports = {}
for row in qualification['rows']:
    path = args.review / row['report']
    assert row['status'] == 'PASS' and digest(path) == row['report_sha256']
    report = read(path)
    assert report['status'] == 'PASS'
    prior = original['suite_reports'][row['suite']]
    assert report.get('cases') == prior.get('cases')
    if 'rows' in report:
        assert len(report['rows']) == len(prior['rows'])
        assert {case['case'] for case in report['rows']} == {case['case'] for case in prior['rows']}
    reports[row['suite']] = report
runtime = reports['runtime-equivalence']
assert runtime == original['suite_reports']['runtime-equivalence']
assert runtime['linked_executable_sha256'] == digest('build/heptatorch')
assert runtime['ordinary_mode'][0]['source_sha256'] == digest('src/heptatorch.hepta')
assert runtime['ordinary_mode'][0]['ordinary_executable_sha256'] == digest('build/heptatorch-monolithic')
assert runtime['ordinary_mode'][1]['ordinary_executable_sha256'] == digest('build/unhex')
result = {'schema': 'heptatorch.hosted-owned-module-gates.v1', 'status': 'PASS',
          'compiler_sha256': qualification['compiler_sha256'],
          'runtime_linker_sha256': qualification['runtime_linker_sha256'],
          'source_sha256': digest('src/heptatorch.hepta'),
          'package_manifest_sha256': digest(manifest),
          'profile_sha256': digest('tools/owned_modules/profile.json'),
          'rebuild_report_sha256': digest(args.build / 'rebuild.json'),
          'qualification_report_sha256': digest(args.review / 'qualification.json'),
          'counted_native_cases': 385, 'native_reproduction': rebuilt['native_reproduction'],
          'qualification': qualification, 'reports': reports,
          'whole_pytorch_rewrite_complete': False,
          'scope': 'Native owned module ABI, syscall allocation and producer-bound lifetime; general tensor storage and owned interactive execution require separate integration.'}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
