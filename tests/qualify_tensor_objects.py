#!/usr/bin/env python3
"""Run frozen object suites on fresh builds; retain full native observations."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'abi2': '4bd22ab115a5428be6ee1ab77fd31a1e20baf14a80756eb38f000574cbad531f',
    'abi3': '2b39ff1a5a258c5fae7be1fbca5dcf0764787b43fa72ad8fdc3ab6fd119e2a91',
}
MANIFESTS = {
    'tensor-object-package.json': '6844366a3b21b87589bb1f4b6ff549a7e2dea2ff3620b9e26d6e570f4a743a27',
    'tensor-object-review-package.json': 'f95a83d699124d4dd45d99784a50d924f8f4556fbc6ff38b60b97749263f057e',
    'tensor-object-oracle-package.json': 'd21807a25e702d48312e11f2c170b7b7da9d076bb6e2e37b3c38f7b89263e147',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def custody():
    verified = {}
    for name, expected in MANIFESTS.items():
        path = ROOT/'validation'/name
        assert digest(path) == expected, name
        manifest = load(path)
        if name == 'tensor-object-package.json':
            files = [(ROOT/p, row) for p, row in manifest['files'].items()]
            assert len(files) == 61
        elif name == 'tensor-object-review-package.json':
            files = [(ROOT/'tests/tensor_object_review'/row['path'], row)
                     for row in manifest['portable_files']]
            assert len(files) == 9
            files.append((ROOT/'validation/tensor-object-independent-history.json', manifest['receipt']))
        else:
            files = [(ROOT/'tests/tensor_object_oracle'/row['path'], row)
                     for row in manifest['files']]
            assert len(files) == 39
        for source, row in files:
            assert source.stat().st_size == row['bytes'] and digest(source) == row['sha256'], source
            verified[str(source.relative_to(ROOT))] = row['sha256']
    assert digest(ROOT/'tools/owned_sessions/client.py') == 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
    return verified


def qualify(mode, directory, output):
    binary = directory/'tensor-object-session'
    assert digest(binary) == PINS[mode]
    build = load(directory/'build.json')
    assert build['status'] == 'PASS' and build['executable_sha256'] == PINS[mode]
    assert build['candidate_source_sha256'] == digest(directory/'runtime.hepta') == '4e2cc7f7612c826ec35280432be527cf98075af7d0f2c35a560df67e35c3904e'
    assert (build['function_count'], build['blocks'], build['instructions']) == (237, 2785, 10863)
    assert build['canonical_functions_byte_identical'] == 99 and len(build['function_map']) == 143
    assert len(build['units']) == 5
    for row in build['units']:
        assert digest(directory/(row['name']+'.hepta')) == row['source_sha256']
        assert digest(directory/(row['name']+'.o')) == row['object_sha256']
        assert row['compiler_exit'] == 0
    for group in ('input_sources', 'upstream_sources'):
        for path, expected in build[group].items():
            assert digest(path) == expected, path
    common = ['--binary', str(binary), '--expected-sha256', PINS[mode],
              '--transport-observer', str(ROOT/'tools/owned_sessions/client.py')]
    suites = [
        ('model', 'tests/tensor_object_review/review_native_objects.py', common),
        ('boundaries', 'tests/tensor_object_review/review_object_boundaries.py',
         common+['--object-state-base', '139264']),
        ('matmul', 'tests/tensor_object_review/replay_matmul_observations.py',
         common+['--observations', str(ROOT/'tests/tensor_object_review/hosted-lifetime-observations.json')]),
        ('aux', 'tools/tensor_objects/test_aux_lifetimes.py', ['--executable', str(binary)]),
        ('references', 'tools/tensor_objects/test_request_references.py', ['--executable', str(binary)]),
        ('scalars', 'tools/tensor_objects/test_scalar_compatibility.py',
         ['--executable', str(binary), '--input-fixture', str(ROOT/'tests/fixtures/numeric-specials.input.hex'),
          '--output-fixture', str(ROOT/'tests/fixtures/numeric-specials.output.hex')]),
    ]
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name, script, args in suites:
        process = subprocess.run([sys.executable, '-B', str(ROOT/script), *args,
                                  '--output', str(output/name)], capture_output=True, timeout=180, cwd=ROOT)
        (output/(name+'.stdout')).write_bytes(process.stdout)
        (output/(name+'.stderr')).write_bytes(process.stderr)
        assert process.returncode == 0 and not process.stderr, (mode, name, process.returncode, process.stderr.decode())
        reports[name] = load(output/name/'report.json')
        assert reports[name]['status'] == 'PASS', (mode, name)
    expectations = {
        'model': {'cases': 13, 'frames': 433, 'exact_scalar_or_binary32_fields': 3252, 'entire_owner_rollback_bytes': 11011792},
        'boundaries': {'cases': 3, 'frames': 68, 'exact_scalar_or_binary32_fields': 93, 'entire_owner_rollback_bytes': 23176112},
        'matmul': {'cases': 6, 'frames': 71, 'exact_scalar_or_binary32_fields': 426, 'entire_owner_rollback_bytes': 892848},
        'aux': {'frames': 140, 'exact_float32_values': 87},
        'references': {'frames': 18, 'exact_numeric_payload_values': 16410, 'rollback_bytes': 2976160},
        'scalars': {'frames': 1, 'exact_scalar_results': 894},
    }
    for name, fields in expectations.items():
        for key, expected in fields.items():
            assert reports[name][key] == expected, (mode, name, key)
        if name in ('model', 'boundaries', 'matmul'):
            assert reports[name]['native_executable_sha256'] == PINS[mode]
            assert all(row['status'] == 'PASS' for row in reports[name]['rows'])
    assert len(reports['aux']['cases']) == 11 and len(reports['references']['cases']) == 5
    smoke = output/'bridge'
    process = subprocess.run([sys.executable, '-B', str(ROOT/'tests/tensor_object_oracle/native_bridge_smoke.py'),
                              str(binary), str(smoke), '--expected-native-sha256', PINS[mode],
                              '--owned-reference', str(ROOT/'tests/owned_tensor_oracle/pytorch_owned_tensor_oracle.py'),
                              '--session-reference', str(ROOT/'tests/pytorch_session_oracle.py'),
                              '--reference', str(ROOT/'tests/pytorch_oracle.py')], capture_output=True, timeout=90, cwd=ROOT)
    (output/'bridge.stdout').write_bytes(process.stdout); (output/'bridge.stderr').write_bytes(process.stderr)
    assert process.returncode == 0 and not process.stderr, (mode, 'bridge', process.stderr.decode())
    bridge = load(smoke/'report.json')
    assert bridge['status'] == 'PASS' and (bridge['cases'], bridge['frames'], bridge['native_only_exact_float_checks']) == (6, 19, 18)
    assert bridge['native_executable_sha256'] == digest(binary) == PINS[mode]
    reports['bridge'] = bridge
    return {'profile': mode, 'status': 'PASS', 'build': build, 'reports': reports,
            'native_suite_frames': 731, 'bridge_frames': 19, 'entire_owner_rollback_bytes': 38056912}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--abi2', type=Path, required=True)
    parser.add_argument('--abi3', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before = custody()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [qualify(mode, getattr(args, mode).resolve(), args.output/mode) for mode in ('abi2', 'abi3')]
    assert (args.abi2/'tensor-object-session').read_bytes()[4096:] == (args.abi3/'tensor-object-session').read_bytes()[4096:]
    assert custody() == before
    report = {'schema': 'heptatorch.hosted-tensor-object-native.v1', 'status': 'PASS',
              'source_files_sha256': before, 'frozen_files_verified': len(before), 'builds': rows,
              'both_elf_payloads_identical': True, 'pinned_pytorch_executed': False,
              'upstream_schema_denominator': 2590, 'whole_pytorch_rewrite_complete': False}
    raw = json.dumps(report, indent=2)+'\n'
    (args.output/'report.json').write_text(raw)
    print(raw, end='')


if __name__ == '__main__':
    main()
