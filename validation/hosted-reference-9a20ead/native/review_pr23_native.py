#!/usr/bin/env python3
"""Verify original PR23 native reports without rewriting their bytes."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile
from collections import Counter

ROOT = Path('/workspace/scratch/d26699e25fea')
REPO = Path('/dev/shm/heptatorch-d26699e25fea/HeptaTorch-reference-bernoulli-gradient')
ART = Path('/dev/shm/heptatorch-d26699e25fea/pr23-native-9a20ead/reports')
OUT = Path('/dev/shm/heptatorch-d26699e25fea/HeptaTorch-gradient-adam-distributions/validation/hosted-reference-9a20ead/native')
HEAD = '9a20eadbe20a801c8b26727ae369e0ee0832e423'
TREE = '5519a9f84b9768e8dcf9ff041ddd1d5969f3bcbc'
SOURCE = '5d3a3c047d024a3880d257a317a04c2166a2e87bbc6b248bef62649a88546873'
IMAGE = '3fc59ab7ed9c32dd75ff440ef9bfbec31756f50edd77715d665bdac098cc3f22'
ZIP_SHA = '5c7285c3e5326e96036a8d4b1e324e5e45ae4bed27b0e66bd86088d818a26758'
sha = lambda b: hashlib.sha256(b).hexdigest()
checks = []

def check(ok, label):
    assert ok, label
    checks.append(label)

def git(path):
    return subprocess.check_output(['git', '-C', str(REPO), 'show', HEAD + ':' + path])

def provenance(p, label):
    keys = {'python_matches', 'platform_matches', 'version_backend_matches',
            'git_matches_oracle_build_pin', 'distribution_version_matches',
            'direct_url_matches', 'direct_url_archive_sha256_matches',
            'pip_report_url_matches', 'pip_report_archive_sha256_matches',
            'pip_report_version_matches', 'pip_report_requested'}
    check(p['status'] == 'PASS' and p['metadata_error'] is None, label + ': actual provenance PASS')
    check(set(p['checks']) == keys and all(v is True for v in p['checks'].values()), label + ': 11 exact provenance checks')
    check(p['lock_sha256'] == sha(git('compatibility/pytorch-oracle-lock.json')), label + ': committed lock')
    check(p['observed']['version'] == '2.15.0+cpu' and p['observed']['git_commit'] ==
          'd0f2af8b66b20da4baa50644c0cd3f887a2eae2e', label + ': actual wheel build')
    eq = p['release_source_equivalence']
    check(eq['inventory_commit'] == '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457' and
          eq['torch_git_matches_upstream_pin'] is False and eq['changed_paths'] == ['pyproject.toml'],
          label + ': source equivalence is limited, commits are distinct')

manifest_bytes = (ART / 'ci-report-manifest.json').read_bytes()
m = json.loads(manifest_bytes)
zip_bytes = (ART.parent / 'original.zip').read_bytes()
check((len(zip_bytes), sha(zip_bytes)) == (2869276, ZIP_SHA), 'artifact exact original ZIP bytes and digest')
check(m['status'] == 'COMPLETE' and m['job_status_before_export'] == 'success', 'complete successful export')
check(m['checkout_commit'] == 'bb7288e4a764716c20ff92064265dd27f1219a3e' and m['checkout_tree'] == TREE,
      'exact PR merge checkout and tree')
check(m['github']['GITHUB_RUN_ID'] == '43842245579' and m['github']['HEPTATORCH_SOURCE_HEAD_SHA'] == HEAD,
      'exact source head and workflow run')
check(m['tracked_worktree_dirty'] is False and m['original_report_bytes_reserialized'] is False,
      'clean tracked checkout and original-byte export')
workflow = git('.github/workflows/native.yml')
check(sha(workflow) == m['workflow_sha256'] and sha(git('tools/collect_ci_reports.py')) == m['collector_sha256'],
      'actual workflow and collector bind committed source')
envelopes = sorted({v.decode() for v in re.findall(rb'tools/emit_ci_report\.py build/([A-Za-z0-9_.-]+\.json)', workflow)})
check(len(envelopes) == 24 and envelopes == m['required_envelope_reports'] and not m['missing_envelope_reports'],
      '24 required report envelopes exactly match workflow')
reports = {}
with zipfile.ZipFile(ART.parent / 'original.zip') as z:
    names = z.namelist()
    expected = [r['path'] for r in m['reports']] + ['ci-report-manifest.json']
    check(len(m['reports']) == 50 and len(names) == len(set(names)) == 51 and set(names) == set(expected),
          '50 original JSON reports plus one manifest, no missing or extra ZIP members')
    check(z.read('ci-report-manifest.json') == manifest_bytes, 'manifest exact original member bytes')
    for row in m['reports']:
        raw = (ART / row['path']).read_bytes()
        check(len(raw) == row['bytes'] and sha(raw) == row['sha256'] and raw == z.read(row['path']),
              row['path'] + ': archive, file, size and digest agree')
        reports[row['path']] = json.loads(raw)

a = reports['hosted-adam-distributions-native.json']
p = json.loads(git('tools/adam_distributions/profile.json'))
check(a['status'] == 'QUALIFIED_COMPONENT_WITH_OPEN_SCOPE' and a['native_scoped_qualification_passed'] is True,
      'native scoped qualification only')
check(a['runtime_source_sha256'] == p['runtime_source_sha256'] == SOURCE and
      a['executable_sha256'] == p['executable_sha256'] == IMAGE, 'native/profile exact source and executable')
check((a['build']['core_functions'], a['build']['core_blocks'], a['build']['core_instructions']) == (378,4971,18539),
      '378 real core functions, 4971 blocks, 18539 instructions')
check(a['build']['units'] == p['units'] and len(p['units']) == 8, 'all eight compiled source/object unit pins exact')
for unit in p['units']:
    for suffix, field in [('.hepta','source_sha256'),('.o','object_sha256')]:
        ending = '/build/adam-distributions/' + unit['name'] + suffix
        check([v for k,v in a['input_sha256'].items() if k.endswith(ending)] == [unit[field]],
              'actual loaded unit ' + unit['name'] + suffix)
check(a['source_custody_after'] is True and a['build']['input_custody_after'] is True,
      'root and build input custody unchanged')
repo_prefix = '/home/runner/work/HeptaTorch/HeptaTorch/'
tracked = set(subprocess.check_output(['git','-C',str(REPO),'ls-tree','-r','--name-only',HEAD], text=True).splitlines())
committed_inputs = []
for path, digest in a['input_sha256'].items():
    if path.startswith(repo_prefix):
        relative = path[len(repo_prefix):]
        if relative in tracked:
            check(sha(git(relative)) == digest, 'committed input ' + relative)
            committed_inputs.append(relative)
check(a['build']['host_production_numerical_runtime'] is False and a['reproduction']['host_numerical_runtime'] is False,
      'production numerics remain native HIA')
check(a['reproduction']['status'] == 'PASS' and a['reproduction']['all_unit_source_object_and_image_bytes_identical'] is True,
      'cold native source/object/image reproduction actually passed')
check((a['formal_report_count'], a['producer_suite_count'], a['expected_legacy_unknown_opcode_failure_count'],
       a['unexpected_failures']) == (15,13,1,0), '15 formal reports include one preserved expected legacy failure')
check(a['old_atomic_test_reclassified_as_pass'] is False and a['reports']['legacy_atomic_controls']['status'] == 'FAIL',
      'old atomic-only unknown-opcode failure remains an actual failure')
suite = a['reports']['producer']['suites']
check(len(suite) == 13 and all(x['exit'] == x['expected_exit'] for x in suite), 'all 13 producer exit contracts honored')
check([(x['name'],x['exit']) for x in suite if x['exit']] == [('legacy_atomic_controls',1)],
      'only expected nonzero is the original legacy atomic control')
for key, counts in [('atomic_reference',{'PASS':161,'OPEN_TRANSACTION_DIFFERENCE':4,'OPEN_UNREPRESENTED':18}),
                    ('explicit_reference',{'PASS':165,'OPEN_UNREPRESENTED':18})]:
    value = a['reports'][key]
    check(value['reference_cases'] == len(value['rows']) == 183 and dict(Counter(r['status'] for r in value['rows'])) == counts
          and value['counts'] == counts, key + ': unchanged fixed183 denominator and per-row ledger')
    check(len(value['actual_reference_provenance_checks']) == 11 and
          all(v is True for v in value['actual_reference_provenance_checks'].values()), key + ': 11 original provenance checks')
    check(value['runtime_source_sha256'] == SOURCE and value['executable_sha256'] == IMAGE,
          key + ': same exact native image')
check(len(a['optimizer_original_24_ledger']) == 24 and dict(Counter(r['status'] for r in a['optimizer_original_24_ledger'])) ==
      {'QUALIFIED_EXPOSED_SCOPE':22,'PARTIAL_WITH_RETAINED_NONLEAF_OPEN':1,'OPEN_NOT_EXECUTED':1},
      'optimizer original24 remains 22 qualified, one partial, one open')
for key in ['optimizer_reference','adam_reference']:
    provenance(a['reports'][key]['oracle_provenance'], key)
c = a['reports']['independent-joint']
check((c['case_count'],c['frames_before_EOF'],c['exact_assertions'],c['exact_float32_words'],
       c['complete_MT_state_comparisons'],c['rollback_bytes_compared']) == (6,462,334,249,22,4892928),
      'six independent cross contracts retain all actual comparison counts')
check(a['cross_contracts_added_to_existing_case_denominators'] is False, 'cross tests are not additional schema coverage')
for filename, count in [('pytorch-adam-distributions-backward.json',32),('pytorch-adam-distributions-frozen35.json',35)]:
    value = reports[filename]
    check(value['native_executable_sha256'] == IMAGE and value['status'] == 'PASS' and
          value['expected_cases'] == value['cases'] == len(value['rows']) == count and value['failed_cases'] == 0 and
          all(row['status'] == 'PASS' for row in value['rows']), filename + ': actual fixed cases pass on exact new image')
    provenance(value['oracle_provenance'],filename)
    check(all(v is True for v in value['source_files_unchanged_during_run'].values()), filename + ': source custody unchanged')
b = reports['pytorch-adam-distributions-backward.json']
check(b['native_source_sha256'] == SOURCE and (b['frames'],b['exact_binary32_words'],b['serialized_state_bytes'],
      b['absolute_version_comparisons'],b['gradient_presence_comparisons']) == (2132,17455,1137600,238,166),
      'unchanged live32 exact counts and source')
check(b['rows'] == reports['pytorch-owned-adam-backward.json']['rows'] == reports['pytorch-cpu-distributions-backward.json']['rows'],
      'live32 all row bytes/metrics agree with both qualified immediate profiles')
check(b['source_custody']['oracle'] == sha(git('tests/pytorch_tensor_rng_backward_oracle.py')),
      'fixed live32 oracle committed hash unchanged')
f = reports['pytorch-adam-distributions-frozen35.json']
check((f['frames'],f['compared_values'],f['direct_parameter_absolute_version_mismatches'],
       f['explicit_state_absolute_version_differences']) == (6344,2590274,0,283),
      'fixed35 exact counts and all 283 legacy explicit-state differences retained')
check(f['case_plan'] == reports['pytorch-tensor-objects-abi3.json']['case_plan'] and
      f['observed_open_policy_differences'] == reports['pytorch-tensor-objects-abi3.json']['observed_open_policy_differences'],
      'fixed35 original plan and open-policy evidence unchanged')
check(a['full_pytorch_rewrite_complete'] is False and a['all_rng_complete'] is False and
      a['all_optimizer_complete'] is False and a['public_gradient_successor_composed'] is False and
      a['whole_project_decision'] == 'HOLD' and p['pytorch_api_denominator'] == 2590,
      'all broader claims remain open with original denominator')
OUT.mkdir(parents=True, exist_ok=False)
shutil.copyfile(ART.parent / 'original.zip', OUT / 'reports.zip')
shutil.copyfile(ART / 'ci-report-manifest.json', OUT / 'ci-report-manifest.json')
receipt = {'schema':'heptatorch.pr23.root-original-native-review.v1','status':'PASS_SCOPED_ADMISSION_REVIEW',
           'head_sha':HEAD,'tree':TREE,'workflow_run':43842245579,'job':133213530827,'artifact_id':14214767988,
           'artifact_bytes':len(zip_bytes),'artifact_sha256':ZIP_SHA,'report_count':50,'required_envelope_count':24,
           'runtime_source_sha256':SOURCE,'executable_sha256':IMAGE,'checks':checks,'check_count':len(checks),
           'committed_inputs_checked':committed_inputs,'reports':m['reports'],
           'reviewer_sha256':sha(Path(__file__).read_bytes()),
           'known_unqualified_gap':'Inherited command95 scalar add canonicalizes NaN payload; strict later incoming-gradient reference exposes 0x7fc00001 -> 0x7fc00000. This report does not qualify NaN payload behavior; successor05 source1fed and ELF6773 are separately qualified and are not part of this378 hosted receipt.',
           'whole_project_decision':'HOLD','full_schema_parity':0,'pytorch_api_denominator':2590,
           'full_pytorch_rewrite_complete':False}
raw = (json.dumps(receipt,indent=2,ensure_ascii=False) + '\n').encode()
(OUT / 'root-review.json').write_bytes(raw)
shutil.copyfile(__file__,OUT / 'review_pr23_native.py')
print(json.dumps({'status':receipt['status'],'checks':len(checks),'reports':50,'root_review_sha256':sha(raw),
                  'output':str(OUT)},ensure_ascii=False))
