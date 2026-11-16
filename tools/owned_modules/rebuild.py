#!/usr/bin/env python3
"""Author pinned HIA source and bootstrap the native owned module toolchain.

Python performs checked source authoring and file/byte transport. HIA performs
source admission, code generation, interface checking, linking, and packing.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import subprocess
import sys


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--capacity-adaptation', type=Path, required=True)
    parser.add_argument('--borrowed-tools', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = output / 'sources'
    sources.mkdir()
    identity = {}
    for line in (spec / 'source-pins.sha256').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        actual = digest((spec / relative).read_bytes())
        assert actual == expected, relative
        identity[relative] = actual
    profile = json.loads((spec / 'profile.json').read_text())

    def execute(label, command):
        result = subprocess.run(command, capture_output=True, timeout=180)
        (output / (label + '.stdout')).write_bytes(result.stdout)
        (output / (label + '.stderr')).write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr, (label, result.returncode, result.stderr)

    common = ['--source-root', str(args.source_root.resolve()), '--capacity-adaptation',
              str(args.capacity_adaptation.resolve())]
    execute('author-compiler', [sys.executable, '-I', '-S', '-B', str(spec / 'build_module_compiler.py'),
                                *common, '--output', str(sources / 'compiler')])
    for abi, workspace in ((1, 2097152), (2, 2097152), (2, 1048576)):
        label = f'abi{abi}-w{workspace // 1048576}m'
        execute('author-linker-' + label,
                [sys.executable, '-I', '-S', '-B', str(spec / 'build_module_linker.py'), *common,
                 '--interface-abi', str(abi), '--entry-workspace', str(workspace),
                 '--output', str(sources / ('linker-' + label + '-source'))])
    for relative, expected in profile['generated_source_sha256'].items():
        assert digest((sources / relative).read_bytes()) == expected, relative
    for label, row in profile['linker_profiles'].items():
        authoring = (sources / ('linker-' + label + '-source/linker-authoring.txt')).read_text()
        headers = next(node for node in ast.parse(authoring).body
                       if isinstance(node, ast.FunctionDef) and node.name == 'headers')
        prefix = bytearray(row['startup_prefix_bytes'])
        covered = bytearray(len(prefix))
        for statement in headers.body:
            if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                continue
            call = statement.value
            if not isinstance(call.func, ast.Name) or call.func.id != 'write' or len(call.args) != 4:
                continue
            if not all(isinstance(value, ast.Constant) for value in call.args[1:]):
                continue
            offset, value, size = [argument.value for argument in call.args[1:]]
            assert 0 <= offset <= len(prefix) - size
            prefix[offset:offset + size] = value.to_bytes(size, 'little')
            covered[offset:offset + size] = b'\x01' * size
        assert all(covered) and digest(prefix) == row['startup_template_sha256'], label
    execute('native-bootstrap', [sys.executable, '-I', '-S', '-B', str(spec / 'bootstrap_native.py'),
                                 '--sources', str(sources), '--borrowed-tools', str(args.borrowed_tools.resolve()),
                                 '--output', str(output / 'native')])
    for relative, expected in identity.items():
        assert digest((spec / relative).read_bytes()) == expected, relative
    report = {'schema': 'heptatorch.owned-modules-portable-rebuild.v1', 'status': 'PASS',
              'profile': profile['profile'], 'package_source_sha256': identity,
              'native_reproduction': json.loads((output / 'native/reproduction.json').read_text()),
              'upstream_derived_full_source_vendored': False,
              'target_libc_dependency': False, 'host_numerical_runtime': False}
    (output / 'rebuild.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'profile': profile['profile'],
                      'compiler_sha256': digest((output / 'native/owned-module-compiler').read_bytes()),
                      'runtime_linker_sha256': digest((output / 'native/linker-abi2-w1m').read_bytes())}, indent=2))


if __name__ == '__main__':
    main()
