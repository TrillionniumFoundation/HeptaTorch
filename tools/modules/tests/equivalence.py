#!/usr/bin/env python3
"""Compare ordinary native application images emitted by the two compilers."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-compiler', type=Path, required=True)
    parser.add_argument('--module-compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('sources', type=Path, nargs='+')
    args = parser.parse_args()
    compilers = [args.baseline_compiler.resolve(), args.module_compiler.resolve()]
    expected = ['c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6',
                'e250d93bc0b2a5ff878d2188c95780abf7e7ebaf186e4531b136fa2735b2af59']
    assert [digest(path.read_bytes()) for path in compilers] == expected
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for number, source in enumerate(args.sources):
        raw = source.read_bytes()
        assert raw.split(b'\n', 1)[0] == b'hia-asm.v1 hepta.hia.program.v8 selfhost_exec'
        (args.output / (str(number) + '.hepta')).write_bytes(raw)
        images = []
        for label, compiler in zip(('baseline', 'modules'), compilers):
            result = subprocess.run([str(compiler)], input=raw, capture_output=True,
                                    env={'PATH': ''}, timeout=30)
            assert result.returncode == 0 and result.stdout and not result.stderr, (source, label)
            image = args.output / (str(number) + '.' + label)
            image.write_bytes(result.stdout)
            image.chmod(0o700)
            images.append(result.stdout)
        assert images[0] == images[1] and source.read_bytes() == raw, source
        rows.append({'source_name': source.name, 'source_sha256': digest(raw),
                     'image_sha256': digest(images[0]), 'image_bytes': len(images[0]),
                     'byte_identical': True, 'status': 'PASS'})
    assert [digest(path.read_bytes()) for path in compilers] == expected
    report = {'schema': 'heptatorch.modules-ordinary-mode-equivalence.v1', 'status': 'PASS',
              'baseline_compiler_sha256': expected[0], 'module_compiler_sha256': expected[1],
              'cases': len(rows), 'rows': rows}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
