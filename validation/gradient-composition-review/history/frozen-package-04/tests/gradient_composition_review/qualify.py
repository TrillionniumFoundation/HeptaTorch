#!/usr/bin/env python3
"""Run the three independent native compilation and owned-entry review gates.

This orchestration does not replace numerical, autograd, distribution, optimizer,
or live PyTorch qualification. All inputs and child test sources are pinned by
the sibling profile.json and handoff-peer-pins-current.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'stack-build', 'atomic-tools', 'gradient-build',
                 'adam-distributions-build', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    profile_path = here / 'profile.json'
    profile_raw = profile_path.read_bytes()
    profile = json.loads(profile_raw)
    assert profile['schema'] == 'heptatorch.gradient-composition.native-review-profile.v1'
    custody = {profile_path: profile_raw}
    for name, expected in profile['test_files_sha256'].items():
        assert Path(name).name == name
        path = here / name
        raw = path.read_bytes()
        assert sha(raw) == expected, path
        custody[path] = raw
    build = args.build.resolve()
    image_name = 'tensor-gradient-adam-distributions-session'
    for path, expected in ((build / 'runtime.hepta', profile['current_source_sha256']),
                           (build / image_name, profile['current_executable_sha256'])):
        raw = path.read_bytes()
        assert sha(raw) == expected
        custody[path] = raw
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    compiler = args.stack_build.resolve() / 'native/stack-module-compiler'
    session_linker = args.stack_build.resolve() / 'native/stack-linker-session'
    runtime_linker = args.stack_build.resolve() / 'native/stack-linker-runtime'
    for path, expected in ((compiler, profile['compiler_sha256']),
                           (session_linker, profile['session_linker_sha256']),
                           (runtime_linker, profile['runtime_linker_sha256'])):
        raw = path.read_bytes()
        assert sha(raw) == expected
        custody[path] = raw
    expected_args = ['--expected-source', profile['current_source_sha256'],
                     '--expected-image', profile['current_executable_sha256'],
                     '--image-name', image_name]
    atomic = args.atomic_tools.resolve()
    calls = [
        ('recompile', 'recompile_units.py', [
            '--build', str(build), '--compiler', str(compiler), '--linker', str(session_linker),
            *expected_args]),
        ('owned-entry', 'test_entry_protocol.py', [
            '--build', str(build), '--compiler', str(compiler),
            '--runtime-linker', str(runtime_linker),
            '--probe-source', str(atomic / 'entry_probe.hepta'),
            '--protocol', str(atomic / 'protocol.py'),
            *expected_args, '--include-adam-distributions']),
        ('typed-handoffs', 'typed_handoffs_gad002.py', [
            '--build', str(build), '--rng-package', str(atomic),
            '--stack-build', str(args.stack_build.resolve()),
            '--peer-pins', str(here / 'handoff-peer-pins-current.json'),
            '--peer', 'incoming355=' + str(args.gradient_build.resolve()),
            '--peer', 'adamdist378=' + str(args.adam_distributions_build.resolve())]),
    ]
    report = {
        'schema': 'heptatorch.gradient-composition.independent-native-review.v1',
        'status': 'RUNNING', 'source_sha256': profile['current_source_sha256'],
        'executable_sha256': profile['current_executable_sha256'],
        'profile_sha256': sha(profile_raw), 'suites': [],
        'scope': 'Native unit byte reproduction, owned output admission, typed owner handoff only.',
        'numerical_qualification_claim': False, 'new_live_torch_executed': False,
        'whole_pytorch_rewrite_complete': False,
    }

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    save()
    for name, filename, arguments in calls:
        target = out / name
        command = [sys.executable, '-B', str(here / filename), *arguments,
                   '--output', str(target)]
        completed = subprocess.run(command, capture_output=True, timeout=600)
        (out / (name + '.stdout')).write_bytes(completed.stdout)
        (out / (name + '.stderr')).write_bytes(completed.stderr)
        raw = (target / 'report.json').read_bytes() if (target / 'report.json').is_file() else b''
        child = json.loads(raw) if raw else {}
        row = {'name': name, 'command': command, 'exit': completed.returncode,
               'report_path': name + '/report.json', 'report_sha256': sha(raw),
               'status': child.get('status', 'NO_REPORT')}
        report['suites'].append(row)
        if completed.returncode or completed.stderr or child.get('status') != 'PASS':
            report['status'] = 'FAIL'
            save()
            raise SystemExit(1)
        source = child.get('source_sha256', child.get('current_source_sha256'))
        image = child.get('image_sha256', child.get('current_executable_sha256'))
        assert source == profile['current_source_sha256']
        assert image == profile['current_executable_sha256']
        save()
    for path, raw in custody.items():
        assert path.read_bytes() == raw, path
    report.update(status='PASS', suite_count=len(calls), input_custody_after=True,
                  input_sha256={str(path): sha(raw) for path, raw in custody.items()})
    save()
    print(json.dumps({'status': 'PASS', 'suites': len(calls),
                      'source_sha256': profile['current_source_sha256'],
                      'executable_sha256': profile['current_executable_sha256']}))


if __name__ == '__main__':
    main()
