#!/usr/bin/env python3
"""Transport source and an explicit ABI2 link frame to native HIA tools."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import struct
import subprocess


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--linker', type=Path, required=True)
    parser.add_argument('--entry', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('sources', type=Path, nargs='+')
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]{0,127}', args.entry)
    assert len(args.sources) <= 64
    compiler, linker = args.compiler.resolve(), args.linker.resolve()
    expected = ['0ccdb524ea327ab290b2e70feaac49a4789abb2fee6da8746a095f435c17af73',
                '23eed229e329fb5b7e2116012f74f210d5de4acd2b4f8475e364890748da86ee']
    assert [digest(path.read_bytes()) for path in (compiler, linker)] == expected
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshots, objects, rows = [], [], []
    for index, path in enumerate(args.sources):
        source = path.read_bytes()
        snapshots.append(source)
        (output / f'unit-{index}.hepta').write_bytes(source)
        result = subprocess.run([str(compiler)], input=source, capture_output=True, env={'PATH': ''}, timeout=30)
        (output / f'unit-{index}.stderr').write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
        (output / f'unit-{index}.o').write_bytes(result.stdout)
        objects.append(result.stdout)
        rows.append({'source_sha256': digest(source), 'object_sha256': digest(result.stdout)})
    entry = args.entry.encode('ascii')
    request = b'HIAOWN02' + struct.pack('<3Q', len(objects), len(entry), 0) + entry
    request += b''.join(struct.pack('<Q', len(obj)) + obj for obj in objects)
    (output / 'link-request.bin').write_bytes(request)
    result = subprocess.run([str(linker)], input=request, capture_output=True, env={'PATH': ''}, timeout=30)
    (output / 'link.stderr').write_bytes(result.stderr)
    assert result.returncode == 0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
    for path, source in zip(args.sources, snapshots):
        assert path.read_bytes() == source
    assert [digest(path.read_bytes()) for path in (compiler, linker)] == expected
    executable = output / 'application'
    executable.write_bytes(result.stdout)
    executable.chmod(0o700)
    report = {'schema': 'heptatorch.owned-module-native-link.v1', 'status': 'PASS',
              'compiler_sha256': expected[0], 'linker_sha256': expected[1], 'entry': args.entry,
              'units': rows, 'request_sha256': digest(request), 'executable_sha256': digest(result.stdout),
              'host_role': 'Source snapshots and file/wire transport only.'}
    (output / 'link.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
