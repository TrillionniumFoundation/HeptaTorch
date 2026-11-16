#!/usr/bin/env python3
"""Reproduce the combined native image from two frozen predecessor builds."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SHA = lambda data: hashlib.sha256(data).hexdigest()


def verify_sources():
    manifest = HERE / 'source-pins.sha256'
    records = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = (HERE / name).resolve()
        assert path.is_relative_to(HERE) and name not in records
        raw = path.read_bytes()
        assert SHA(raw) == digest, name
        records[name] = digest
    assert {'build.py', 'rebuild.py', 'profile.json', 'upstream-pins.json', 'compact_functions.py'} <= set(records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('adam-build', 'distribution-build', 'upstream', 'stack-build', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    before = verify_sources()
    profile = json.loads((HERE / 'profile.json').read_text())
    command = [sys.executable, str(HERE / 'build.py')]
    for name in ('adam-build', 'distribution-build', 'upstream', 'stack-build', 'output'):
        command.extend(['--' + name, str(getattr(a, name.replace('-', '_')).resolve())])
    result = subprocess.run(command, capture_output=True)
    output = a.output.resolve()
    if output.exists():
        (output / 'builder.stdout').write_bytes(result.stdout)
        (output / 'builder.stderr').write_bytes(result.stderr)
    if result.returncode:
        sys.stdout.buffer.write(result.stdout)
        sys.stderr.buffer.write(result.stderr)
        raise SystemExit(result.returncode)
    built = json.loads((output / 'build.json').read_text())
    assert built['status'] == 'PASS'
    for field in ('profile', 'runtime_source_sha256', 'executable_sha256', 'core_functions',
                  'core_blocks', 'core_instructions', 'compiler_sha256', 'linker_sha256'):
        assert built[field] == profile[field], field
    assert SHA((output / 'runtime.hepta').read_bytes()) == profile['runtime_source_sha256']
    assert SHA((output / 'tensor-adam-distributions-session').read_bytes()) == profile['executable_sha256']
    assert len(built['units']) == len(profile['units']) == 8
    for actual, expected in zip(built['units'], profile['units'], strict=True):
        assert actual == expected
        assert SHA((output / (actual['name'] + '.hepta')).read_bytes()) == actual['source_sha256']
        assert SHA((output / (actual['name'] + '.o')).read_bytes()) == actual['object_sha256']
    assert verify_sources() == before
    receipt = {
        'schema': 'heptatorch.adam-distributions.cold-reproduction.v1', 'status': 'PASS',
        'profile': profile['profile'], 'runtime_source_sha256': profile['runtime_source_sha256'],
        'executable_sha256': profile['executable_sha256'], 'native_units': len(profile['units']),
        'all_unit_source_object_and_image_bytes_identical': True,
        'complete_native_compilation_and_linking_executed': True,
        'host_numerical_runtime': False, 'sources_sha256': before,
        'build_report_sha256': SHA((output / 'build.json').read_bytes()),
        'full_pytorch_rewrite_complete': False,
    }
    (output / 'reproduction.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    print(json.dumps({key: receipt[key] for key in ('status', 'runtime_source_sha256', 'executable_sha256', 'native_units')}))


if __name__ == '__main__':
    main()
