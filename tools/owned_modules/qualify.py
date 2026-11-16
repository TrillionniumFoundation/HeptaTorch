#!/usr/bin/env python3
"""Reproduce the owned profile's native lifecycle and admission witnesses."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--borrowed-tools', type=Path, required=True)
    parser.add_argument('--baseline-compiler', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = Path(__file__).resolve().parent
    build = args.build.resolve()
    native = build / 'native'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = json.loads((spec / 'profile.json').read_text())
    identity = {}
    for line in (spec / 'source-pins.sha256').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        actual = digest((spec / relative).read_bytes())
        assert actual == expected, relative
        identity[relative] = actual
    assert digest((native / 'owned-module-compiler').read_bytes()) == profile['module_compiler_executable_sha256']
    for label, row in profile['linker_profiles'].items():
        assert digest((native / ('linker-' + label)).read_bytes()) == row['executable_sha256']
    assert digest((args.borrowed_tools / 'module-compiler').read_bytes()) == profile['predecessor_compiler_sha256']
    assert digest((args.borrowed_tools / 'module-linker').read_bytes()) == '1ba40ffe025abfcc2edd8d69f1ad689817598571138c11806af7aa60047fc7b8'
    common = ['--native-build', str(native)]
    borrowed = ['--borrowed-tools', str(args.borrowed_tools.resolve())]
    repo = ['--repo-root', str(args.repo_root.resolve())]
    suites = [
        ('lifecycle', 'lifecycle_probe.py', common, 1),
        ('source-lifetimes', 'review_lifecycle.py', common + borrowed, 34),
        ('release-startup', 'review_release_and_startup.py', common + borrowed, 8),
        ('allocator', 'review_allocator.py', [
            '--source-root', str(args.source_root.resolve()), '--compiler-adaptation',
            str(build / 'sources/compiler/compiler-adaptation.json'), '--allocation-object',
            str(output / 'lifecycle/allocation.o')], 13),
        ('contracts', 'review_contracts.py', common + repo, 315),
        ('independent-flow', 'independent_flow.py', [
            '--compiler', str(native / 'owned-module-compiler'),
            '--linker', str(native / 'linker-abi2-w1m')], 14),
        ('runtime-equivalence', 'review_runtime.py', common + repo + [
            '--baseline-compiler', str(args.baseline_compiler.resolve())], None),
    ]
    rows = []
    for label, script, options, cases in suites:
        command = [sys.executable, '-I', '-S', '-B', str(spec / 'tests' / script),
                   *options, '--output', str(output / label)]
        result = subprocess.run(command, capture_output=True, timeout=180)
        (output / (label + '.stdout')).write_bytes(result.stdout)
        (output / (label + '.stderr')).write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr, (label, result.returncode, result.stderr)
        report_path = output / label / 'report.json'
        report = json.loads(report_path.read_text())
        assert report['status'] == 'PASS', label
        if cases is not None:
            assert report.get('cases', 1) == cases, (label, report.get('cases'))
        rows.append({'suite': label, 'status': 'PASS', 'cases': cases,
                     'report_sha256': digest(report_path.read_bytes()),
                     'report': label + '/report.json'})
        print(label + ': PASS', flush=True)
    for relative, expected in identity.items():
        assert digest((spec / relative).read_bytes()) == expected, relative
    report = {'schema': 'heptatorch.owned-modules-native-qualification.v1', 'status': 'PASS',
              'profile': profile['profile'], 'compiler_sha256': profile['module_compiler_executable_sha256'],
              'runtime_linker_sha256': profile['linker_profiles']['abi2-w1m']['executable_sha256'],
              'counted_native_cases': sum(cases or 0 for _, _, _, cases in suites),
              'rows': rows, 'package_source_sha256': identity,
              'additional_fixture_execution_claim': False,
              'scope': 'Owned module/resource ABI, syscall allocator, bootstrap, and exact current runtime equivalence; not all PyTorch storage.'}
    (output / 'qualification.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
