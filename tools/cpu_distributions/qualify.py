#!/usr/bin/env python3
"""Run scoped native qualification while preserving the fixed OPEN ledger."""
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
    for name in ('build', 'baseline', 'stack-build', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for line in (HERE / 'source-pins.sha256').read_text().splitlines():
        pin, name = line.split('  ', 1)
        path = (HERE / name).resolve()
        assert path.is_relative_to(HERE)
        inputs[path] = path.read_bytes()
        assert SHA(inputs[path]) == pin, path
    profile = json.loads(inputs[HERE / 'profile.json'])
    build = a.build.resolve()
    receipt = json.loads((build / 'build.json').read_text())
    image = build / 'tensor-distributions-session'
    assert SHA(image.read_bytes()) == profile['executable_sha256'] == receipt['executable_sha256']
    assert SHA((build / 'runtime.hepta').read_bytes()) == profile['runtime_source_sha256']
    suites = [
        ('actual-reference', 'replay_reference.py', ['--reference', HERE / 'evidence/torch-bernoulli-dropout.json',
         '--protocol', HERE / 'protocol.py', '--executable', image, '--build-report', build / 'build.json']),
        ('native-controls', 'test_controls.py', ['--protocol', HERE / 'protocol.py', '--executable', image,
         '--build-report', build / 'build.json']),
        ('probability-math', 'test_probability_math.py', ['--build', build, '--stack-build', a.stack_build.resolve()]),
        ('owned-entry', 'test_entry.py', ['--build', build, '--stack-build', a.stack_build.resolve()]),
        ('typed-handoffs', 'test_handoffs.py', ['--build', build, '--rng-package', HERE,
         '--stack-build', a.stack_build.resolve(), '--peer-pins', HERE / 'handoff-peer-pins.json',
         '--peer', 'nullable=' + str(a.baseline.resolve())]),
    ]
    rows = []
    for name, script, args in suites:
        target = out / name
        command = [sys.executable, '-B', str(HERE / script), *map(str, args), '--output', str(target)]
        completed = subprocess.run(command, capture_output=True, timeout=300)
        (out / (name + '.stdout')).write_bytes(completed.stdout)
        (out / (name + '.stderr')).write_bytes(completed.stderr)
        assert completed.returncode == 0, (name, completed.returncode, completed.stderr[-4000:])
        report = json.loads((target / 'report.json').read_bytes())
        if name == 'actual-reference':
            assert report['status'] == 'PARTIAL'
            assert report['counts'] == {'PASS': 161, 'OPEN_TRANSACTION_DIFFERENCE': 4, 'OPEN_UNREPRESENTED': 18}
        else:
            assert report['status'] == 'PASS'
        rows.append({'name': name, 'exit_code': 0, 'status': report['status'],
                     'report_sha256': SHA((target / 'report.json').read_bytes())})
    for path, data in inputs.items():
        assert path.read_bytes() == data, path
    result = {'schema': 'heptatorch.cpu-distributions-native-qualification.v1', 'status': 'PARTIAL',
              'native_scoped_qualification_passed': True, 'executable_sha256': profile['executable_sha256'],
              'runtime_source_sha256': profile['runtime_source_sha256'], 'suites': rows,
              'actual_torch_observations': 183, 'strict_mapped_passes': 161,
              'open_transaction_differences': 4, 'open_unrepresented': 18,
              'all_rng_complete': False, 'all_tensor_dtypes_complete': False,
              'full_pytorch_rewrite_complete': False, 'source_custody_after': True}
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
