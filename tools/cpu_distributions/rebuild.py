#!/usr/bin/env python3
"""Reproduce the frozen CPU distribution HIA/object/ELF identities."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SHA = lambda data: hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ('baseline', 'common-authoring', 'gradient-authoring', 'upstream', 'stack-build', 'output')
    for name in names:
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    inputs = {}
    for line in (HERE / 'source-pins.sha256').read_text().splitlines():
        pin, name = line.split('  ', 1)
        path = (HERE / name).resolve()
        assert path.is_relative_to(HERE)
        inputs[path] = path.read_bytes()
        assert SHA(inputs[path]) == pin, path
    profile = json.loads(inputs[HERE / 'profile.json'])
    argv = [sys.executable, '-I', '-S', '-B', str(HERE / 'build.py')]
    for name in names:
        argv += ['--' + name, str(getattr(a, name.replace('-', '_')).resolve())]
    subprocess.run(argv, check=True)
    report = json.loads((a.output / 'build.json').read_text())
    for key in ('runtime_source_sha256', 'executable_sha256', 'compiler_sha256', 'linker_sha256',
                'core_functions', 'core_blocks', 'core_instructions'):
        assert report[key] == profile[key], key
    actual = {row['name']: (row['source_sha256'], row['object_sha256']) for row in report['units']}
    expected = {row['name']: (row['source_sha256'], row['object_sha256']) for row in profile['units']}
    assert actual == expected
    assert report['base_function_body_map'] == profile['base_function_body_map']
    for path, data in inputs.items():
        assert path.read_bytes() == data, path
    result = {'schema': 'heptatorch.cpu-distributions-reproduction.v1', 'status': 'PASS',
              'source_pins_sha256': SHA((HERE / 'source-pins.sha256').read_bytes()),
              'runtime_source_sha256': report['runtime_source_sha256'],
              'executable_sha256': report['executable_sha256'],
              'all_source_object_image_bytes_identical': True, 'native_units': len(actual)}
    (a.output / 'reproduction.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
