#!/usr/bin/env python3
"""Reproduce the frozen seven-kind RNG plus nullable-gradient native image."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SHA = lambda raw: hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    names = ('common-build', 'materialization-build', 'nullable-build', 'stack-build', 'output')
    for name in names:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    sources = {}
    for line in (HERE / 'source-pins.sha256').read_text().splitlines():
        pin, name = line.split('  ', 1)
        path = (HERE / name).resolve()
        assert path.is_relative_to(HERE)
        sources[path] = path.read_bytes()
        assert SHA(sources[path]) == pin, path
    profile = json.loads((HERE / 'profile.json').read_text())
    argv = [sys.executable, '-I', '-S', '-B', str(HERE / 'build.py')]
    for name in names:
        argv += ['--' + name, str(getattr(a, name.replace('-', '_')).resolve())]
    subprocess.run(argv, check=True)
    report = json.loads((a.output / 'build.json').read_text())
    for name in ('runtime_source_sha256', 'executable_sha256', 'compiler_sha256', 'linker_sha256'):
        assert report[name] == profile[name], name
    actual = {x['name']: (x['source_sha256'], x['object_sha256']) for x in report['units']}
    expected = {x['name']: (x['source_sha256'], x['object_sha256']) for x in profile['units']}
    assert actual == expected
    for path, raw in sources.items():
        assert path.read_bytes() == raw, path
    result = {'schema': 'heptatorch.tensor-rng-nullable-reproduction.v1', 'status': 'PASS',
              'source_pins_sha256': SHA((HERE / 'source-pins.sha256').read_bytes()),
              'runtime_source_sha256': report['runtime_source_sha256'],
              'executable_sha256': report['executable_sha256'],
              'all_source_object_image_bytes_identical': True, 'native_modules': len(actual)}
    (a.output / 'reproduction.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
