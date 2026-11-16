#!/usr/bin/env python3
"""Execute the combined-profile qualifications and retain the old negative failure.

All target operations run inside native ELFs. Frozen observation replayers,
reference rows and numerical tolerances are unchanged. The only admission
derivative is explicitly recorded as two newly-defined command error classes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SHA = lambda data: hashlib.sha256(data).hexdigest()
DIST_REFERENCE = 'c624f5c4762b1c71e7053aba9af55aaa3d8bbaae5aa100b0a6df5f44f58da467'
OPT_REFERENCE = '7ad0a2e718300a33c91a9212f14ada3c0c7997be7fc77f64bb557c2ecd858160'
SCRIPT_PINS = {
    'atomic_reference': 'f956fe37b5b809b52b8f7dc909acdc6b02125a93d04b1ba15330f23ba1772b40',
    'explicit_reference': '3bff6bf735a9c9e4f8013e5aee1f1a8c41ceba9a3ba4d7f36182ccc382636d49',
    'optimizer_reference': 'ea748d4100b539e0144f032ca301b4886d0f9520c36676c980279e0bbc9bc034',
    'adam_reference': 'b0a3323e8d4d76369e1791b4f68a4889ada640f410ec7a8ee89d162167671001',
}


def verify_sources():
    records = {}
    for line in (HERE / 'source-pins.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = (HERE / name).resolve()
        assert path.is_relative_to(HERE) and name not in records
        assert SHA(path.read_bytes()) == digest, name
        records[name] = digest
    assert {'qualify.py', 'profile.json', 'adapt_tests.py', 'test_extended_admission.py'} <= set(records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'adam-build', 'atomic-build', 'distribution-build', 'stack-build',
                 'atomic-tools', 'partial-tools', 'adam-tools', 'adam-review',
                 'tensor-protocol', 'optimizer-observations', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    for name in vars(a):
        setattr(a, name, getattr(a, name).resolve())
    a.output.mkdir(parents=True, exist_ok=False)
    source_pins = verify_sources()
    source = a.build / 'runtime.hepta'
    image = a.build / 'tensor-adam-distributions-session'
    build_path = a.build / 'build.json'
    profile = json.loads((HERE / 'profile.json').read_text())
    build = json.loads(build_path.read_text())
    assert build['status'] == 'PASS'
    assert SHA(source.read_bytes()) == build['runtime_source_sha256'] == profile['runtime_source_sha256']
    assert SHA(image.read_bytes()) == build['executable_sha256'] == profile['executable_sha256']
    reference = a.atomic_tools / 'evidence/torch-bernoulli-dropout.json'
    assert SHA(reference.read_bytes()) == DIST_REFERENCE
    assert SHA(a.optimizer_observations.read_bytes()) == OPT_REFERENCE
    actual = json.loads(reference.read_text())
    assert actual['cases'] == 183 and actual['probe_error_count'] == 0
    assert len(actual['oracle_provenance']['checks']) == 11 and all(actual['oracle_provenance']['checks'].values())
    tests = a.output / 'adapted-tests'
    adapter = [sys.executable, str(HERE / 'adapt_tests.py'), '--atomic-tools', str(a.atomic_tools),
               '--partial-tools', str(a.partial_tools), '--adam-tools', str(a.adam_tools), '--output', str(tests)]
    result = subprocess.run(adapter, capture_output=True)
    (a.output / 'adapt.stdout').write_bytes(result.stdout)
    (a.output / 'adapt.stderr').write_bytes(result.stderr)
    assert result.returncode == 0 and not result.stderr
    scripts = {
        'atomic_reference': a.atomic_tools / 'replay_reference.py',
        'explicit_reference': a.partial_tools / 'replay_explicit.py',
        'optimizer_reference': a.adam_review / 'replay_optimizer_objects_v2.py',
        'adam_reference': a.adam_review / 'replay_adam_objects.py',
    }
    for name, path in scripts.items():
        assert SHA(path.read_bytes()) == SCRIPT_PINS[name], name
    custody = {path: path.read_bytes() for path in (
        Path(__file__).resolve(), HERE / 'profile.json', HERE / 'adapt_tests.py',
        HERE / 'handoff-peer-pins.json', source, image, build_path, reference,
        a.optimizer_observations, *scripts.values())}
    rows = []
    report = {
        'schema': 'heptatorch.adam-distributions.native-qualification.v1', 'status': 'RUNNING',
        'profile': profile['profile'], 'runtime_source_sha256': profile['runtime_source_sha256'],
        'executable_sha256': profile['executable_sha256'], 'suites': rows,
        'reference_sha256': DIST_REFERENCE, 'optimizer_reference_sha256': OPT_REFERENCE,
        'fixed_reference_rows_and_tolerances_unchanged': True,
        'original_atomic_unknown_opcode_control_passed': False,
        'old_atomic_control_known_failure_preserved': False,
        'whole_project_decision': 'HOLD', 'all_rng_complete': False, 'all_optimizer_complete': False,
        'all_tensor_dtypes_complete': False, 'full_pytorch_rewrite_complete': False,
    }

    def save():
        (a.output / 'report.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')

    def run(name, script, options, expected_exit=0):
        directory = a.output / name
        command = [sys.executable, str(script), *map(str, options), '--output', str(directory)]
        original = script.read_bytes()
        result = subprocess.run(command, capture_output=True)
        (a.output / (name + '.stdout')).write_bytes(result.stdout)
        (a.output / (name + '.stderr')).write_bytes(result.stderr)
        row = {'name': name, 'command': command, 'script_sha256': SHA(original),
               'exit': result.returncode, 'expected_exit': expected_exit,
               'stdout_sha256': SHA(result.stdout), 'stderr_sha256': SHA(result.stderr)}
        rows.append(row)
        if (directory / 'report.json').exists():
            raw = (directory / 'report.json').read_bytes()
            row['report_sha256'] = SHA(raw)
            row['report'] = str((directory / 'report.json').relative_to(a.output))
            detail = json.loads(raw)
            row['reported_status'] = detail['status']
        else:
            detail = None
        save()
        assert result.returncode == expected_exit and script.read_bytes() == original, row
        if expected_exit == 0:
            assert not result.stderr and detail is not None
        return detail, row

    common_dist = ['--reference', reference, '--protocol', a.atomic_tools / 'protocol.py',
                   '--executable', image, '--build-report', build_path]
    open_indices = set(profile['distribution_open_indices'])
    late_indices = set(profile['explicit_late_error_indices'])
    for name in ('atomic_reference', 'explicit_reference'):
        detail, _ = run(name, scripts[name], common_dist)
        assert detail['runtime_source_sha256'] == profile['runtime_source_sha256']
        assert detail['executable_sha256'] == profile['executable_sha256']
        assert len(detail['rows']) == len(actual['rows']) == 183
        expected_counts = profile['atomic_reference_counts' if name == 'atomic_reference' else 'explicit_reference_counts']
        assert dict(Counter(row['status'] for row in detail['rows'])) == detail['counts'] == expected_counts
        for index, (row, observed) in enumerate(zip(detail['rows'], actual['rows'], strict=True)):
            assert row['index'] == index and row['name'] == observed['name'] and row['family'] == observed['family']
            wanted = 'OPEN_UNREPRESENTED' if index in open_indices else (
                'OPEN_TRANSACTION_DIFFERENCE' if name == 'atomic_reference' and index in late_indices else 'PASS')
            assert row['status'] == wanted
        assert detail['status'] == 'PARTIAL'
    common_opt = ['--binary', image, '--source', source,
                  '--controls', a.adam_review / 'native_release_controls.py',
                  '--transport-observer', a.adam_review / 'client.py',
                  '--observations', a.optimizer_observations,
                  '--expected-sha256', profile['executable_sha256'],
                  '--expected-source-sha256', profile['runtime_source_sha256']]
    detail, _ = run('optimizer_reference', scripts['optimizer_reference'], common_opt + ['--suite', 'optimizer'])
    assert detail['native_source_sha256'] == profile['runtime_source_sha256']
    assert detail['native_executable_sha256'] == profile['executable_sha256']
    assert detail['executed_cases'] == 19 and detail['qualified_exposed_scope'] == 18
    assert detail['partial_cases'] == 1 and detail['open_not_executed_cases'] == 5 and detail['unexpected_failures'] == 0
    detail, _ = run('adam_reference', scripts['adam_reference'], common_opt + ['--legacy-replayer', scripts['optimizer_reference']])
    assert detail['native_source_sha256'] == profile['runtime_source_sha256']
    assert detail['native_executable_sha256'] == profile['executable_sha256']
    assert detail['qualified_exposed_scope'] == 4 and detail['failures'] == 0

    atomic_options = ['--executable', image, '--build-report', build_path, '--protocol', a.atomic_tools / 'protocol.py']
    detail, _ = run('legacy_atomic_controls', tests / 'atomic_controls.py', atomic_options, expected_exit=1)
    assert detail['status'] == 'FAIL' and detail['runtime_source_sha256'] == profile['runtime_source_sha256']
    assert detail['executable_sha256'] == profile['executable_sha256']
    assert len(detail['cases']) == 1 and detail['cases'][0]['error'] == 'AssertionError((1, 4))'
    # The last actual request must be opcode86; a coincidental failure elsewhere
    # is not admitted as the known old-contract mismatch.
    transport = a.output / 'legacy_atomic_controls/admission-and-stale-generator'
    packets = (transport / 'requests.frames.bin').read_bytes()
    at, last = 0, None
    while at < len(packets):
        assert packets[at:at + 8] == b'HTOWNR01'
        size = struct.unpack_from('<Q', packets, at + 8)[0]
        last = packets[at + 16:at + 16 + size]
        assert len(last) == size
        at += 16 + size
    assert at == len(packets) and last[:8] == b'HTORCH03'
    assert struct.unpack_from('<Q', last, 8)[0] == 1 and struct.unpack_from('<Q', last, 32)[0] == 86
    report['old_atomic_control_known_failure_preserved'] = True
    detail, _ = run('new_command_admission', HERE / 'test_extended_admission.py',
                    ['--atomic-build', a.atomic_build, '--partial-build', a.distribution_build,
                     '--build', a.build, '--protocol', a.atomic_tools / 'protocol.py'])
    assert detail['status'] == 'PASS' and detail['cases'] == 12
    detail, _ = run('atomic_controls_extended', tests / 'atomic_controls_extended.py', atomic_options)
    assert detail['status'] == 'PASS' and detail['case_count'] == 4
    detail, _ = run('partial_controls', tests / 'partial_controls.py', ['--build', a.build, '--atomic-tools', a.atomic_tools])
    assert detail['status'] == 'PASS' and detail['case_count'] == 14
    native_opt = ['--binary', image, '--source', source, '--protocol', a.tensor_protocol]
    detail, _ = run('adam_controls', tests / 'adam_controls.py', native_opt + [
        '--expected-sha256', profile['executable_sha256'], '--expected-source-sha256', profile['runtime_source_sha256']])
    assert detail['status'] == 'PASS' and len(detail['cases']) == 3
    for name, expected in [('adam_guards', 5), ('adam_step_counter', 11)]:
        detail, _ = run(name, tests / (name + '.py'), native_opt + ['--build-report', build_path])
        assert detail['status'] == 'PASS'
        assert (detail['case_count'] if name == 'adam_guards' else detail['cases']) == expected
    detail, _ = run('owned_entry', tests / 'owned_entry.py',
                    ['--build', a.build, '--atomic-tools', a.atomic_tools, '--stack-build', a.stack_build])
    assert detail['status'] == 'PASS' and detail['raw_processes'] == 34 and len(detail['checks']) == 5
    detail, _ = run('typed_handoffs', tests / 'typed_handoffs.py',
                    ['--build', a.build, '--rng-package', a.atomic_tools, '--stack-build', a.stack_build,
                     '--peer-pins', HERE / 'handoff-peer-pins.json',
                     '--peer', 'adam02=' + str(a.adam_build), '--peer', 'partial01=' + str(a.distribution_build)])
    assert detail['status'] == 'PASS' and detail['cases'] == 4 and len(detail['peers']) == 2
    for path, raw in custody.items():
        assert path.read_bytes() == raw, path
    for row in rows:
        detail = json.loads((a.output / row['report']).read_text())
        # Every normal current-image suite exposes one of these exact schemas.
        if row['name'] == 'new_command_admission':
            continue
        source_fields = ('runtime_source_sha256', 'native_source_sha256', 'source_sha256', 'current_source_sha256')
        image_fields = ('executable_sha256', 'native_executable_sha256', 'native_sha256', 'current_executable_sha256')
        assert any(detail.get(key) == profile['runtime_source_sha256'] for key in source_fields), row['name']
        assert any(detail.get(key) == profile['executable_sha256'] for key in image_fields), row['name']
    report.update(status='QUALIFIED_COMPONENT_WITH_OPEN_SCOPE', suite_count=len(rows),
                  expected_old_negative_failure_count=1, unexpected_failures=0,
                  atomic_reference_counts=profile['atomic_reference_counts'],
                  explicit_reference_counts=profile['explicit_reference_counts'],
                  optimizer_reference_counts=profile['optimizer_reference_counts'],
                  input_custody_after=True, input_sha256={str(path): SHA(raw) for path, raw in custody.items()},
                  tools_source_pins=source_pins)
    assert verify_sources() == source_pins
    assert len(rows) == 13
    save()
    print(json.dumps({key: report[key] for key in ('status', 'suite_count', 'expected_old_negative_failure_count',
                                                  'unexpected_failures', 'runtime_source_sha256', 'executable_sha256')}))


if __name__ == '__main__':
    main()
