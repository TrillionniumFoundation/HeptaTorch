#!/usr/bin/env python3
"""Read-only check of the final gradient-owner Make/CI command integration."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys

import yaml


BASE = 'e4cf7b6d3645b5a340ec3b47489ecc326e908c13'
LIVE = 'tests/pytorch_tensor_rng_backward_oracle.py'
FROZEN = 'tests/tensor_object_oracle/pytorch_tensor_objects_oracle.py'
LIVE_SHA = '554671168c7a4a408dc1e1df89a19db15c3efcefcf4ae1a300af1b6dcae40160'
FROZEN_SHA = '149b41c455444edeb8d25d0ad98045e29683e780f37f7e210e7985242d359597'
NAMES = (
    'Reproduce public gradient Tensor objects and incoming ownership',
    'Reproduce shared public gradients Adam and CPU distributions',
    'Compare public gradient owner with the unchanged live joint oracle (32 cases)',
    'Regress public gradient owner against the unchanged 35 Tensor cases',
    'Compare shared gradient Adam distribution owner with unchanged live oracle (32 cases)',
    'Regress shared gradient Adam distribution owner against unchanged 35 Tensor cases',
)
PINS = {
    'gradient-objects': {
        'image': 'tensor-gradient-object-session',
        'source': '1b18da2cf9791e1b20982f9706434c1359ea194dd983dc6c66eb005f152e77b5',
        'native': 'be18ab877f38a4027a527783db80a1418a137c7d6da91e5bc39e2af8c044cb6b',
    },
    'gradient-adam-distributions': {
        'image': 'tensor-gradient-adam-distributions-session',
        'source': '1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae',
        'native': '6773485896d961b6c2e62784ebb66320290df47128bd1331aecc0c8b7f6611fe',
    },
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw if isinstance(raw, bytes) else raw.encode())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('repository', 'gradient-build', 'combined-build', 'prior-review', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    for name in ('repository', 'gradient_build', 'combined_build', 'prior_review', 'output'):
        setattr(a, name, getattr(a, name).resolve())
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    stage = a.output / 'cli-worktree'
    custody = {}
    checks = []

    def check(name, value):
        assert value, name
        checks.append(name)

    def read(path):
        path = path.resolve()
        raw = path.read_bytes()
        expected = custody.setdefault(path, {'bytes': len(raw), 'sha256': sha(raw)})
        assert expected == {'bytes': len(raw), 'sha256': sha(raw)}, ('changed input', str(path))
        return raw

    def root(path):
        return read(a.repository / path)

    def snapshot(path):
        raw = root(path)
        save(a.output / 'reviewed-source' / path, raw)
        return raw

    git = lambda *args: subprocess.check_output(['git', *args], cwd=a.repository)
    head = git('rev-parse', 'HEAD').decode().strip()
    workflow = snapshot('.github/workflows/native.yml')
    makefile = snapshot('Makefile')
    old_workflow = git('show', BASE + ':.github/workflows/native.yml')
    old_makefile = git('show', BASE + ':Makefile')
    save(a.output / 'base-native.yml', old_workflow)
    save(a.output / 'base-Makefile', old_makefile)
    old = yaml.safe_load(old_workflow)['jobs']['native']
    new = yaml.safe_load(workflow)['jobs']['native']
    steps = {step['name']: step for step in new['steps']}
    check('all six new steps are present exactly once', all(sum(s.get('name') == n for s in new['steps']) == 1 for n in NAMES))
    check('all original declared steps keep exact parsed bodies and order', [s for s in new['steps'] if s.get('name') not in NAMES] == old['steps'])
    check('63 previous declared steps plus six new steps', (len(old['steps']), len(new['steps'])) == (63, 69))
    check('timeout increases exactly from twenty to forty minutes', old['timeout-minutes'] == 20 and new['timeout-minutes'] == 40)
    sections = re.split(rb'(?=^      - )', workflow, flags=re.M)
    kept = b''.join(section for section in sections if not any(section.startswith(('      - name: ' + name + '\n').encode()) for name in NAMES))
    kept = kept.replace(b'    timeout-minutes: 40\n', b'    timeout-minutes: 20\n', 1)
    check('byte-exact original workflow after six-step removal and timeout reversal', kept == old_workflow)
    reports = lambda raw: sorted(set(re.findall(rb'tools/emit_ci_report\.py (build/[A-Za-z0-9_.-]+\.json)(?:\s|$)', raw)))
    before_envelopes, after_envelopes = reports(old_workflow), reports(workflow)
    check('twenty-four prior envelopes and thirty final envelopes', len(before_envelopes) == 24 and len(after_envelopes) == 30)
    check('all original envelopes retained', set(before_envelopes) < set(after_envelopes))
    check('six separate new report paths', len(set(after_envelopes) - set(before_envelopes)) == 6)
    collector = snapshot('tools/collect_ci_reports.py')
    check('unchanged report collector', collector == git('show', BASE + ':tools/collect_ci_reports.py'))
    check('native job still exports every original JSON report', new['steps'][-1]['with']['path'] == 'build/*.json')
    check('collector still uses native workflow by default', "default=Path('.github/workflows/native.yml')" in collector.decode())
    new_make_start = makefile.index(b'GRADIENT_OBJECT_INPUTS :=')
    new_make_end = makefile.index(b'check-compiler:\n', new_make_start)
    check('Make change is exactly the two new target blocks', makefile[:new_make_start] + makefile[new_make_end:] == old_makefile)
    make_text = makefile.decode()
    expected_rules = (
        'gradient-objects: build/gradient-objects/build.json',
        'build/gradient-objects/build.json: build/owned-optimizers/reproduction.json $(GRADIENT_OBJECT_INPUTS) | build',
        'test-gradient-objects: owned-optimizers',
        'gradient-adam-distributions: build/gradient-adam-distributions/build.json',
        'build/gradient-adam-distributions/build.json: build/gradient-objects/build.json build/adam-distributions/reproduction.json $(GRADIENT_ADAM_DISTRIBUTION_INPUTS) | build',
        'test-gradient-adam-distributions: gradient-objects adam-distributions',
    )
    for rule in expected_rules:
        check('Make rule ' + rule, make_text.count(rule + '\n') == 1)
    for fragment in ('"$$review/observed/native/." build/gradient-objects/',
                     '"$$review/observed/producer/native/." build/gradient-adam-distributions/',
                     '--output build/hosted-gradient-objects-native.json',
                     '"$$review/observed/report.json" build/hosted-gradient-adam-distributions-native.json'):
        check('Make output mapping ' + fragment, fragment in make_text)
    for name in NAMES[:2]:
        check('source root and stack toolchain configured for ' + name,
              'HEPTA_SOURCE_ROOT="$GITHUB_WORKSPACE/.toolchain/hepta"' in steps[name]['run'] and
              'HEPTA_STACK_BUILD="$RUNNER_TEMP/hepta-stack-budget"' in steps[name]['run'])
    input_paths = (
        LIVE, FROZEN, 'tests/tensor_object_oracle/bridge.py', 'tests/tensor_object_oracle/lifecycle.py',
        'tests/owned_tensor_oracle/pytorch_owned_tensor_oracle.py', 'tests/pytorch_session_oracle.py',
        'tests/pytorch_oracle.py', 'compatibility/pytorch-oracle-lock.json', 'tools/emit_ci_report.py',
    )
    for path in input_paths:
        raw = snapshot(path)
        check('original frozen input unchanged ' + path, raw == git('show', BASE + ':' + path))
        save(stage / path, raw)
    check('live32 exact script hash', sha(root(LIVE)) == LIVE_SHA)
    check('frozen35 exact script hash', sha(root(FROZEN)) == FROZEN_SHA)
    for kind, build in (('gradient-objects', a.gradient_build), ('gradient-adam-distributions', a.combined_build)):
        pin = PINS[kind]
        for name, expected in (('runtime.hepta', pin['source']), (pin['image'], pin['native'])):
            path = build / name
            check('cold input pin ' + kind + '/' + name, sha(read(path)) == expected)
            target = stage / 'build' / kind / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(path)
        receipt = json.loads(read(build / 'build.json'))
        check('cold build success ' + kind, receipt['status'] == 'PASS' and receipt['source_sha256'] == pin['source'] and receipt['native_sha256'] == pin['native'])
    original_commands = []
    prepared = []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')

    def argv_from_step(step, script):
        logical = step['run'].replace('\\\n', '').splitlines()
        candidates = [line.strip() for line in logical if line.strip().startswith('python3 ' + script + ' ')]
        assert len(candidates) == 1
        words = shlex.split(candidates[0])
        end = next((i for i, word in enumerate(words) if word in ('>', '||')), len(words))
        return words[:end]

    for index, name in enumerate(NAMES[2:]):
        script = LIVE if index % 2 == 0 else FROZEN
        original = argv_from_step(steps[name], script)
        kind = 'gradient-objects' if index < 2 else 'gradient-adam-distributions'
        pin = PINS[kind]
        check('exact source verification present in ' + name,
              (LIVE_SHA if script == LIVE else FROZEN_SHA) + '  ' + script in steps[name]['run'])
        check('correct native hash in ' + name, original[original.index('--expected-native-sha256') + 1] == pin['native'])
        if script == LIVE:
            check('correct source hash and absence of unsupported flags in ' + name,
                  original[original.index('--expected-source-sha256') + 1] == pin['source'] and
                  '--support' not in original and '--lock' not in original)
            report_path = original[original.index('--output') + 1]
        else:
            report_path = original[3]
        check('output has its own envelope ' + name, report_path.encode() in after_envelopes)
        execution = original + ['--prepare-only']
        result = subprocess.run(execution, cwd=stage, env=env, capture_output=True)
        label = ('355' if index < 2 else '428') + ('-live32' if script == LIVE else '-frozen35')
        save(a.output / 'executions' / (label + '.stdout'), result.stdout)
        save(a.output / 'executions' / (label + '.stderr'), result.stderr)
        check('actual prepare-only exit zero ' + name, result.returncode == 0)
        raw = (stage / report_path).read_bytes()
        observed = json.loads(raw)
        check('no Torch or native cases executed by ' + name,
              observed['status'] == 'NOT_RUN' and observed['cases'] == 0 and observed['expected_cases'] == (32 if script == LIVE else 35))
        check('correct prepared native identity ' + name, observed['native_executable_sha256'] == pin['native'])
        if script == LIVE:
            check('correct prepared source identity ' + name, observed['native_source_sha256'] == pin['source'])
        else:
            check('original frozen AST/custody preparation passed ' + name, observed['source_overlay_checks'] == 'PASS')
        save(a.output / 'prepared' / (label + '.json'), raw)
        original_commands.append({'step': name, 'entire_original_shell_block': steps[name]['run'],
                                  'original_oracle_argv': original, 'executed_argv': execution,
                                  'only_extra_argument': '--prepare-only', 'cwd': str(stage),
                                  'exit_code': result.returncode, 'report_path': report_path,
                                  'report_sha256': sha(raw), 'stdout_sha256': sha(result.stdout),
                                  'stderr_sha256': sha(result.stderr)})
        prepared.append(observed)
    check('both live32 case plans identical', prepared[0]['case_plan'] == prepared[2]['case_plan'])
    check('both frozen35 case plans and tolerances identical', prepared[1]['case_plan'] == prepared[3]['case_plan'])
    prior = read(a.prior_review / 'report.json')
    check('original 517 report preserved', sha(prior) == '4f79d61c09e07f1ae270ac37c68786f755ecc8f9d798f359a7eb1ff198015454')
    save(a.output / 'history/original-517-report.json', prior)
    prior_workflow = read(a.prior_review / 'reviewed-source/.github/workflows/native.yml')
    save(a.output / 'history/original-517-workflow.yml', prior_workflow)
    correction_path = 'validation/gradient-objects-root/ci-cli-correction-01.json'
    correction_raw = snapshot(correction_path)
    correction = json.loads(correction_raw)
    check('correction explicitly records omitted required source hash and unsupported flags',
          '--expected-source-sha256' in correction['reason'] and '--support/--lock' in correction['reason'])
    for row in correction['executed_prepare_only']:
        raw = snapshot(row['path'])
        check('original root correction preparation bytes ' + row['path'], sha(raw) == row['sha256'])
    old_named = {s['name']: s for s in yaml.safe_load(prior_workflow)['jobs']['native']['steps']}
    old_bad = argv_from_step(old_named[NAMES[2]], LIVE)
    check('original erroneous argv remains different and unmodified', '--support' in old_bad and '--lock' in old_bad and '--expected-source-sha256' not in old_bad)
    refused = subprocess.run(old_bad + ['--prepare-only'], cwd=stage, env=env, capture_output=True)
    save(a.output / 'executions/original-355-cli-refused.stdout', refused.stdout)
    save(a.output / 'executions/original-355-cli-refused.stderr', refused.stderr)
    check('original 355 bad CLI actually rejected by unchanged argparse', refused.returncode == 2 and b'--expected-source-sha256' in refused.stderr)
    make_interfaces = []
    for path in ('tools/gradient_objects/build.py', 'tools/gradient_objects/qualify.py',
                 'tests/check_gradient_object_reports.py', 'tools/gradient_adam_distributions/build.py',
                 'tools/gradient_adam_distributions/qualify.py'):
        raw = snapshot(path)
        ast.parse(raw)
        result = subprocess.run([sys.executable, '-B', str(a.repository / path), '--help'], env=env, capture_output=True)
        label = path.replace('/', '_')
        save(a.output / 'executions' / (label + '.help.stdout'), result.stdout)
        save(a.output / 'executions' / (label + '.help.stderr'), result.stderr)
        check('actual Make program help accepted ' + path, result.returncode == 0)
        make_interfaces.append({'path': path, 'sha256': sha(raw), 'help_exit': result.returncode,
                                'required_options': re.findall(r'--[A-Za-z][A-Za-z-]+', result.stdout.decode().split('\n\n')[0])})
    for path, expected in list(custody.items()):
        check('unchanged final input ' + str(path), {'bytes': path.stat().st_size, 'sha256': sha(path.read_bytes())} == expected)
    report = {
        'schema': 'heptatorch.gradient-owner-final-ci-command-review.v1',
        'status': 'PASS_EXECUTED_PREPARATION_AND_READ_ONLY_SOURCE_REVIEW',
        'repository_head_at_read': head, 'comparison_base': BASE,
        'workflow_sha256': sha(workflow), 'Makefile_sha256': sha(makefile),
        'prior_517_report_preserved_sha256': sha(prior),
        'root_cli_correction_receipt_sha256': sha(correction_raw),
        'old_declared_steps': 63, 'new_declared_steps': 69,
        'hosted_prior_68_step_receipt_not_reexecuted_here': True,
        'old_step_bytes_and_order_unchanged': True, 'timeout_minutes': [20, 40],
        'envelope_counts': [24, 30], 'added_envelope_reports': sorted(x.decode() for x in set(after_envelopes) - set(before_envelopes)),
        'oracle_commands': original_commands, 'make_cli_help': make_interfaces,
        'original_bad_355_command_exit': refused.returncode,
        'original_bad_355_command_executed_argv': old_bad + ['--prepare-only'],
        'four_real_argparse_and_prepare_calls_executed': True,
        'same_workflow_relative_paths_in_private_cli_workspace': True,
        'root_files_modified': False, 'root_builds_run': False,
        'Torch_executed': False, 'native_executable_run': False,
        'semantic_test_cases_executed': 0,
        'new_428_aggregate_qualifier_review': 'PENDING independent author final source and executed result; Make expects --gradient-build --adam-distribution-build --stack-build --output and producer/native',
        'all_checked_inputs_unchanged': True,
        'source_custody': {str(path): value for path, value in custody.items()},
        'checks': checks, 'check_count': len(checks), 'full_rewrite_complete': False,
    }
    save(a.output / 'report.json', json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'check_count', 'workflow_sha256', 'Makefile_sha256', 'envelope_counts')}))


if __name__ == '__main__':
    main()
