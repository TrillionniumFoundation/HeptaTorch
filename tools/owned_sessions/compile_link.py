#!/usr/bin/env python3
"""Transport checked ABI2 modules to the production dual-entry HIA linker."""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import subprocess


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--linker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('sources', type=Path, nargs='+')
    args = parser.parse_args()
    compiler, linker = args.compiler.resolve(), args.linker.resolve()
    expected = ['0ccdb524ea327ab290b2e70feaac49a4789abb2fee6da8746a095f435c17af73',
                '55a2b0a762c8db574a65d0d0e6a18e3cedc668c01d20080a11c26cf1ccb601b3']
    assert [digest(path.read_bytes()) for path in (compiler, linker)] == expected
    assert 1 <= len(args.sources) <= 64
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows, objects, snapshots = [], [], []
    for index, path in enumerate(args.sources):
        raw = path.read_bytes()
        snapshots.append(raw)
        (output / f'unit-{index}.hepta').write_bytes(raw)
        result = subprocess.run([str(compiler)], input=raw, capture_output=True, env={'PATH': ''}, timeout=30)
        (output / f'unit-{index}.stderr').write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
        objects.append(result.stdout)
        (output / f'unit-{index}.o').write_bytes(result.stdout)
        rows.append({'source_sha256': digest(raw), 'object_sha256': digest(result.stdout)})
    request = b'HIAOWNS1' + struct.pack('<3Q', len(objects), 4, 0) + b'step'
    request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
    (output / 'link-request.bin').write_bytes(request)
    result = subprocess.run([str(linker)], input=request, capture_output=True, env={'PATH': ''}, timeout=30)
    (output / 'link.stderr').write_bytes(result.stderr)
    assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
    for path, raw in zip(args.sources, snapshots):
        assert path.read_bytes() == raw
    assert [digest(path.read_bytes()) for path in (compiler, linker)] == expected
    executable = output / 'application'
    executable.write_bytes(result.stdout)
    executable.chmod(0o700)
    report = {'schema': 'heptatorch.owned-session-native-link.v1', 'status': 'PASS',
              'compiler_sha256': expected[0], 'linker_sha256': expected[1],
              'units': rows, 'request_sha256': digest(request), 'executable_sha256': digest(result.stdout),
              'entry': 'step', 'close': 'close', 'request_magic': 'HIAOWNS1',
              'host_role': 'Checked source snapshots and wire/file transport only.'}
    (output / 'link.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
