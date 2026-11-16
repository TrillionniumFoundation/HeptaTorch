"""Bind freshly executed ABI3 qualification to frozen source and image identities."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
sha = lambda raw: hashlib.sha256(raw).hexdigest()
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build', type=Path, required=True)
p.add_argument('--application-build', type=Path, required=True)
p.add_argument('--review', type=Path, required=True)
p.add_argument('--root-review', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
manifest = ROOT/'validation/stack-budget-package.sha256'
assert sha(manifest.read_bytes()) == '134eabba13ff5b9edc17cb1207c1173b7bf59719c0683e175902b30676d82f8e'
source_pins = {}
for line in manifest.read_text().splitlines():
    digest, relative = line.split('  ', 1)
    assert sha((ROOT/relative).read_bytes()) == digest, relative
    source_pins[relative] = digest
assert len(source_pins) == 20
strict = ROOT/'tests/owned_tensor_oracle/pytorch_owned_tensor_oracle.py'
assert sha(strict.read_bytes()) == 'ee3cf72154ac9dfa31055557b3facab8fdd3e14aad03ef6000ced55ccd75ed88'
profile = json.loads((ROOT/'tools/stack_budget/profile.json').read_bytes())
reports, report_hashes = {}, {}


def read(name, path):
    raw = path.read_bytes(); reports[name] = json.loads(raw); report_hashes[name] = sha(raw)
    return reports[name]


rebuilt = read('rebuild', a.build/'rebuild.json')
assert rebuilt['status'] == 'PASS' and rebuilt['successive_native_generations_byte_identical']
read('bootstrap', a.build/'native/bootstrap.json')
read('adaptation', a.build/'sources/adaptation.json')
qualified = read('qualification', a.review/'qualification.json')
assert qualified['status'] == 'PASS' and qualified['native_gate_cases'] == 86
assert qualified['tensor_native_response_frames'] == 161 and qualified['tensor_independent_response_frames'] == 16
for row in qualified['rows']:
    report = read(row['suite'], a.review/row['report'])
    assert report['status'] == 'PASS' and report['cases'] == row['cases']
    assert report_hashes[row['suite']] == row['report_sha256']
application = read('application', a.application_build/'build.json')
tensor = read('tensor', a.review/'tensor-execution/report.json')
assert tensor['status'] == 'PASS' and tensor['native_cases'] == 11
independent = read('independent', a.root_review/'report.json')
assert independent['status'] == 'PASS' and len(independent['cases']) == 9
assert all(row['status'] == 'PASS' for row in independent['cases'])
assert independent['script_sha256'] == sha((ROOT/'tests/stack_independent.py').read_bytes())
assert sha((a.build/'native/stack-module-compiler').read_bytes()) == profile['compiler']['executable_sha256']
for name, row in profile['linkers'].items():
    assert sha((a.build/'native'/('stack-linker-'+name)).read_bytes()) == row['executable_sha256']
for folder in (a.application_build, a.review/'tensor-build'):
    assert sha((folder/'owned-tensor-session').read_bytes()) == profile['current_tensor']['executable_sha256']
receipt = {'schema': 'heptatorch.hosted-stack-budget-native.v1', 'status': 'PASS',
           'source_pins': source_pins, 'report_sha256': report_hashes, 'reports': reports,
           'native_gate_cases': 86, 'independent_root_cases': 9,
           'tensor_native_cases': 11, 'tensor_native_response_frames': 161,
           'tensor_independent_response_frames': 16,
           'compiler_sha256': profile['compiler']['executable_sha256'],
           'session_linker_sha256': profile['linkers']['session']['executable_sha256'],
           'tensor_executable_sha256': profile['current_tensor']['executable_sha256'],
           'unchanged_strict_oracle_sha256': sha(strict.read_bytes()),
           'real_pytorch_oracle_executed_by_this_checker': False,
           'whole_pytorch_rewrite_complete': False}
raw = json.dumps(receipt, indent=2)+'\n'; a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(raw); print(raw)
