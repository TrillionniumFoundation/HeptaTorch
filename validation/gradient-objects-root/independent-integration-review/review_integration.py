#!/usr/bin/env python3
"""Read-only review of the frozen 355-function integration and its receipts."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('repository', 'cold-review', 'aggregate', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    root, cold, out = a.repository.resolve(), a.cold_review.resolve(), a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    seen = {}
    checks = []

    def digest(b): return hashlib.sha256(b).hexdigest()
    def raw(p):
        p = Path(p).resolve()
        b = p.read_bytes()
        before = seen.setdefault(p, digest(b))
        assert digest(b) == before, p
        return b
    def read(p): return json.loads(raw(p))
    def check(label, condition):
        assert condition, label
        checks.append(label)

    source = '1b18da2cf9791e1b20982f9706434c1359ea194dd983dc6c66eb005f152e77b5'
    image = 'be18ab877f38a4027a527783db80a1418a137c7d6da91e5bc39e2af8c044cb6b'
    checker = root / 'tests/check_gradient_object_reports.py'
    check('root checker exact supplied source', digest(raw(checker)) ==
          'eb51575a34cd92863da7b032f11f201763b328355f848d72f0f67156bd71e4b8')
    manifest_path = root / 'tools/gradient_objects/package-manifest.json'
    manifest = read(manifest_path)
    check('original package manifest pinned', digest(raw(manifest_path)) ==
          '57b5ea629794fd50f35695695295a539291ea061e36b0266159a2de0e45f08ca')
    check('all 124 original inputs retained', len(manifest['files']) == 124)
    for rel, sha in manifest['files'].items():
        check('package ' + rel, digest(raw(root / rel)) == sha)
    agg_raw = raw(a.aggregate)
    check('supplied actual aggregate exact bytes', digest(agg_raw) ==
          'b58f50091f26c764b62713bad6689affef229ac1312c9ed02306c2dc6d7846c5')
    agg = json.loads(agg_raw)
    frozen = read(cold / 'report.json')
    check('aggregate embeds original cold receipt without semantic modification', agg['cold_qualification'] == frozen)
    check('cold receipt has 16 successful steps', frozen['status'] == 'PASS' and frozen['step_count'] == 16)
    check('cold receipt binds target identity', frozen['source_sha256'] == source and frozen['native_sha256'] == image)
    rows = []
    for step in frozen['steps']:
        name = step['name']
        command = step['command']
        check(name + ': actual subprocess success', step['exit_code'] == 0)
        check(name + ': isolated stdlib script invocation', command[1:4] == ['-I', '-S', '-B'])
        script = Path(command[4]).resolve()
        rel = str(script.relative_to(root))
        check(name + ': executed script matches frozen source', digest(raw(script)) == manifest['files'][rel])
        data = raw(cold / step['report_path'])
        report = json.loads(data)
        check(name + ': original report hash', digest(data) == step['report_sha256'] ==
              frozen['suite_reports'][name]['sha256'] == agg['original_report_sha256'][name]['sha256'])
        check(name + ': aggregate preserves full decoded report', agg['reports'][name] == report)
        for stream in ('stdout', 'stderr'):
            check(name + ': ' + stream + ' custody', digest(raw(cold / (name + '.' + stream))) == step[stream + '_sha256'])
        rows.append({'name': name, 'script': rel, 'script_sha256': digest(raw(script)),
                     'report_sha256': digest(data), 'status': report['status'],
                     'native_executed': report.get('native_executed'),
                     'frames': report.get('frames'), 'execution_exit_code': step['exit_code']})
    for directory, filename in [('native', 'tensor-gradient-object-session'),
                                 ('authored-incoming', 'tensor-gradient-incoming-session')]:
        check(directory + ': actual core source hash', digest(raw(cold / directory / 'runtime.hepta')) == source)
        check(directory + ': actual native image hash', digest(raw(cold / directory / filename)) == image)
    build = read(cold / 'native/build.json')
    profile = read(root / 'tools/gradient_objects/profile.json')
    check('355 core and 8 real units', (build['functions'], build['blocks'], build['instructions'],
          build['descriptor_bytes'], len(build['units'])) == (355, 4675, 17450, 576, 8))
    for unit, pin in zip(build['units'], profile['units'], strict=True):
        check('unit name ' + unit['name'], unit['name'] == pin['name'])
        for suffix, field in (('.hepta', 'source_sha256'), ('.o', 'object_sha256')):
            check('unit bytes ' + unit['name'] + suffix,
                  digest(raw(cold / 'native' / (unit['name'] + suffix))) == unit[field] == pin[field])
    provenance_names = {'python_matches', 'platform_matches', 'version_backend_matches',
        'git_matches_oracle_build_pin', 'distribution_version_matches', 'direct_url_matches',
        'direct_url_archive_sha256_matches', 'pip_report_url_matches', 'pip_report_archive_sha256_matches',
        'pip_report_version_matches', 'pip_report_requested'}
    refs = {
        'original-public-ten': 'torch-gradient-objects.json',
        'incoming-25-observations': 'torch-gradient-materialization.json',
        'alias-edge-12-observations': 'torch-gradient-alias-edge.json',
        'signed-zero-13': 'torch-gradient-signed-zero.json',
        'partial-16': 'torch-backward-partial.json',
        'presence-8': 'torch-gradient-presence.json',
        'gradient-clear-10': 'torch-grad-clear.json'}
    for name, filename in refs.items():
        report = agg['reports'][name]
        p = report['oracle_provenance']
        check(name + ': exact eleven actual reference checks', p['status'] == 'PASS' and
              p['metadata_error'] is None and set(p['checks']) == provenance_names and
              all(v is True for v in p['checks'].values()))
        check(name + ': actual reference original bytes', digest(raw(root / 'validation/gradient-objects/reference' / filename)) ==
              report.get('reference_sha256', report.get('observations_sha256')))
        check(name + ': original lock', p['lock_sha256'] == digest(raw(root / 'compatibility/pytorch-oracle-lock.json')))
        check(name + ': exact source/image', report.get('source_sha256', report.get('native_source_sha256')) == source and
              report.get('native_sha256', report.get('native_executable_sha256')) == image)

    workflow_path = root / '.github/workflows/native.yml'
    workflow = raw(workflow_path)
    base_workflow = subprocess.check_output(['git', 'show', 'HEAD:.github/workflows/native.yml'], cwd=root)
    new_names = [b'Reproduce public gradient Tensor objects and incoming ownership',
        b'Compare public gradient owner with the unchanged live joint oracle (32 cases)',
        b'Regress public gradient owner against the unchanged 35 Tensor cases']
    stripped = workflow
    blocks = []
    for name in new_names:
        matches = list(re.finditer(rb'      - name: ' + re.escape(name) + rb'\n.*?(?=      - name: |\Z)', stripped, re.S))
        check('unique new workflow step ' + name.decode(), len(matches) == 1)
        m = matches[0]
        blocks.append(m.group())
        stripped = stripped[:m.start()] + stripped[m.end():]
    check('removing exactly three steps restores old workflow byte for byte', stripped == base_workflow)
    make = raw(root / 'Makefile')
    base_make = subprocess.check_output(['git', 'show', 'HEAD:Makefile'], cwd=root)
    start = make.index(b'GRADIENT_OBJECT_INPUTS :=')
    end = make.index(b'check-compiler:', start)
    check('removing only new Make block restores old Makefile byte for byte', make[:start] + make[end:] == base_make)
    for value in blocks:
        check('new CI has no scratch or tmpfs dependency', b'/workspace/' not in value and b'/dev/shm/' not in value)
    for rel, pin in {
        'tests/pytorch_tensor_rng_backward_oracle.py': '554671168c7a4a408dc1e1df89a19db15c3efcefcf4ae1a300af1b6dcae40160',
        'tests/tensor_object_oracle/pytorch_tensor_objects_oracle.py': '149b41c455444edeb8d25d0ad98045e29683e780f37f7e210e7985242d359597',
    }.items():
        check('unchanged live script ' + rel, digest(raw(root / rel)) == pin)
        check('CI checks live script pin ' + rel, pin.encode() in workflow)
    check('two new live steps pin actual 355 image', blocks[1].count(image.encode()) == 1 and blocks[2].count(image.encode()) == 1)
    envelopes = re.findall(rb'tools/emit_ci_report\.py (build/[A-Za-z0-9_.-]+\.json)(?:\s|$)', workflow)
    old_envelopes = re.findall(rb'tools/emit_ci_report\.py (build/[A-Za-z0-9_.-]+\.json)(?:\s|$)', base_workflow)
    check('24 old envelopes plus exactly three distinct new outputs', len(old_envelopes) == 24 and len(envelopes) == 27 and len(set(envelopes)) == 27)

    # Execute only the read-only report binder, not a new build/native/Torch suite.
    output = out / 'checker-recomputed.json'
    completed = subprocess.run([os.environ.get('PYTHON', 'python3'), '-I', '-S', '-B', str(checker),
        '--build', str(cold / 'native'), '--review', str(cold), '--output', str(output)],
        capture_output=True, timeout=60)
    (out / 'checker.stdout').write_bytes(completed.stdout)
    (out / 'checker.stderr').write_bytes(completed.stderr)
    check('root reader executes successfully on actual cold evidence', completed.returncode == 0)
    check('root reader reproduces original aggregate exact bytes', output.read_bytes() == agg_raw)
    for path, pin in seen.items():
        check('input stayed unchanged ' + str(path), digest(path.read_bytes()) == pin)
    for rel in ('tests/check_gradient_object_reports.py', 'tools/gradient_objects/qualify.py', 'Makefile', '.github/workflows/native.yml'):
        dest = out / 'reviewed-source' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw(root / rel))
    result = {
        'schema': 'heptatorch.independent-gradient-object-integration-review.v1',
        'status': 'PASS_READ_ONLY_REVIEW', 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'source_sha256': source, 'native_sha256': image, 'original_package_inputs': 124,
        'original_cold_steps': 16, 'reference_reports_with_all_eleven_checks': 7,
        'unchanged_previous_envelopes': 24, 'new_envelopes': 3,
        'all_previous_workflow_and_Make_bytes_preserved': True,
        'all_original_script_and_tolerance_bytes_preserved': True,
        'root_checker_actual_exit_code': completed.returncode,
        'aggregate_exact_byte_reproduction_sha256': digest(output.read_bytes()),
        'new_native_execution': False, 'new_Torch_execution': False,
        'new_live32_and_live35_execution_in_this_review': False,
        'review_findings': [], 'steps': rows, 'checks': checks,
        'inputs_sha256': {str(p): pin for p, pin in seen.items()},
        'full_project_complete': False, 'pytorch_schema_denominator': 2590,
    }
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'checks': len(checks),
                      'report_sha256': digest((out / 'report.json').read_bytes())}))


if __name__ == '__main__': main()
