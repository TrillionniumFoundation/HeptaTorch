#!/usr/bin/env python3
"""Execute golden packet fixtures in the actual native linked application."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    executable = args.executable.resolve()
    image = executable.read_bytes()
    manifest = json.loads(args.split_manifest.read_bytes())
    rows = []
    for source in sorted(args.fixtures.glob('*.input.hex')):
        name = source.name.removesuffix('.input.hex')
        output = source.with_name(name + '.expected.hex')
        if not output.is_file():
            output = source.with_name(name + '.output.hex')
        data = bytes.fromhex(source.read_text())
        expected = bytes.fromhex(output.read_text())
        result = subprocess.run([str(executable)], input=data, capture_output=True,
                                env={'PATH': ''}, timeout=30)
        assert result.returncode == 0 and not result.stderr and result.stdout == expected, (
            name, result.returncode, result.stderr, result.stdout.hex(), expected.hex())
        rows.append({'case': name, 'status': 'PASS', 'input_sha256': digest(data),
                     'expected_sha256': digest(expected), 'observed_sha256': digest(result.stdout),
                     'output_bytes': len(result.stdout)})
    assert rows and executable.read_bytes() == image
    report = {'schema': 'heptatorch.linked-runtime-fixtures.v1', 'status': 'PASS',
              'source_sha256': manifest['source_sha256'], 'cases': len(rows),
              'partition_complete': manifest['partition_complete'],
              'source_function_count': manifest['total_function_count'],
              'executable_sha256': digest(image), 'executable_bytes': len(image),
              'runtime_path': 'empty', 'native_execution': True,
              'host_numerical_runtime': False, 'rows': rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
