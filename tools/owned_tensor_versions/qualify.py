"""Replay exact native suites with two source-backed version expectations.

Actual executed sources and reversible source edits are retained. The real
PyTorch oracle is not adapted; its original bytes and assertions remain fixed.
"""
from pathlib import Path
import argparse
import difflib
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--linker', type=Path, required=True)
    p.add_argument('--canonical-binary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    driver = Path(__file__).read_bytes()
    manifest = json.loads((ROOT / 'validation/owned-tensor-package.json').read_text())
    for relative, identity in manifest['files'].items():
        raw = (ROOT / relative).read_bytes()
        assert len(raw) == identity['bytes'] and sha(raw) == identity['sha256'], relative
    native = a.native.resolve()
    build = json.loads((native / 'build.json').read_text())
    profile = json.loads((HERE / 'profile.json').read_text())
    app = native / 'owned-tensor-session'
    assert build['pass'] and sha(app.read_bytes()) == profile['executable_sha256'] == build['executable_sha256']
    assert all(build[k] == v for k, v in profile.items())
    for row in profile['unit_rows']:
        assert sha((native / 'units' / (row['name'] + '.hepta')).read_bytes()) == row['source_sha256']
        assert sha((native / (row['name'] + '.o')).read_bytes()) == row['object_sha256']
    frozen = ROOT / 'tools/owned_tensors'
    sources = out / 'executed-sources/owned_tensors'; sources.mkdir(parents=True)
    transport = sources.parent / 'owned_sessions'; transport.mkdir()
    shutil.copyfile(ROOT / 'tools/owned_sessions/client.py', transport / 'client.py')
    for name in ('protocol.py', 'owners.hepta', 'test_typed_state.py'):
        shutil.copyfile(frozen / name, sources / name)
    original = (frozen / 'test_dynamic.py').read_text()
    changed = original
    substitutions = (
        ("assert p.meta(metadata)['version'] == step",
         "assert p.meta(metadata)['version'] == step * (2 if kind == 'AdamW' else 1)"),
        ("assert all(p.meta(record)['version'] == 32 for record in records)",
         "assert [p.meta(record)['version'] for record in records] == [32, 64]"),
    )
    for old, new in substitutions:
        assert changed.count(old) == 1
        changed = changed.replace(old, new, 1)
    restored = changed
    for old, new in reversed(substitutions):
        assert restored.count(new) == 1
        restored = restored.replace(new, old, 1)
    assert restored == original
    (sources / 'test_dynamic.py').write_text(changed)
    patch = ''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                                       fromfile='frozen-candidate03/test_dynamic.py',
                                       tofile='executed-candidate04/test_dynamic.py'))
    (out / 'version-expectations.patch').write_text(patch)
    typed_profile = json.loads((frozen / 'profile.json').read_text())
    typed_profile['core_overlay_sha256'] = profile['candidate_source_sha256']
    typed_profile['executable_sha256'] = profile['executable_sha256']
    typed_profile['units'] = {row['name']: {k: v for k, v in row.items() if k != 'name'}
                              for row in profile['unit_rows']}
    (sources / 'profile.json').write_text(json.dumps(typed_profile, indent=2) + '\n')
    peer_base = ROOT / 'tests/owned_tensor_review/candidate-03-review/executed-sources'
    peer = out / 'executed-sources/independent'; peer.mkdir()
    for name in ('protocol.py', 'client.py'):
        shutil.copyfile(peer_base / name, peer / name)
    peer_original = (peer_base / 'review_dynamic.py').read_text()
    before = "TARGET_SHA='" + profile['baseline_executable_sha256'] + "'"
    after = "TARGET_SHA='" + profile['executable_sha256'] + "'"
    assert peer_original.count(before) == 1
    peer_changed = peer_original.replace(before, after, 1)
    assert peer_changed.replace(after, before, 1) == peer_original
    (peer / 'review_dynamic.py').write_text(peer_changed)
    snapshots = {path: path.read_bytes() for path in (out / 'executed-sources').rglob('*') if path.is_file()}

    def run(script, arguments, name):
        result = subprocess.run([sys.executable, '-I', '-S', '-B', str(script), *map(str, arguments)],
                                capture_output=True, timeout=240)
        (out / (name + '.stdout')).write_bytes(result.stdout)
        (out / (name + '.stderr')).write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr, (name, result.returncode, result.stderr[-2000:])

    run(sources / 'test_dynamic.py', ['--service', app, '--build-report', native / 'build.json',
                                    '--output', out / 'native'], 'native')
    run(sources / 'test_typed_state.py', ['--compiler', a.compiler.resolve(), '--linker', a.linker.resolve(),
                                        '--build-directory', native, '--output', out / 'typed'], 'typed')
    run(peer / 'review_dynamic.py', ['--binary', app, '--canonical-binary', a.canonical_binary.resolve(),
                                    '--transport-observer', transport / 'client.py',
                                    '--output', out / 'independent'], 'independent')
    reports = {name: json.loads((out / name / 'report.json').read_text())
               for name in ('native', 'typed', 'independent')}
    first = reports['native']
    assert first['pass'] and first['cases'] == 11 and first['response_frames'] == 161
    assert first['compared_binary32_values'] == 1078028 and first['rollback_compared_owned_bytes'] == 59310080
    assert first['test_sha256'] == sha(changed.encode()) and first['executable_sha256'] == profile['executable_sha256']
    without_timing = lambda rows: [{k: v for k, v in row.items() if k != 'elapsed_seconds'} for row in rows]
    assert without_timing(first['rows']) == without_timing(json.loads((frozen / 'evidence/portable-native-report.json').read_text())['rows'])
    typed = reports['typed']
    assert typed['pass'] and typed['cases'] == 6 and all(r['pass'] for r in typed['rows'])
    review = reports['independent']
    assert review['status'] == 'PASS' and review['interactive_frames'] == 16
    assert review['exact_scalar_or_f32_fields_compared'] == 612
    assert review['entire_owner_rollback_bytes_compared'] == 41943040
    assert review['numerical_checks'] == json.loads((peer_base.parent / 'report.json').read_text())['numerical_checks']

    spec = importlib.util.spec_from_file_location('version_boundary_wire', frozen / 'protocol.py')
    wire = importlib.util.module_from_spec(spec); spec.loader.exec_module(wire)
    boundary = []
    for size in (0, 1):
        for mode, decay, lr in ((0, .0625, .125), (1, 0., .125), (1, -0., .125),
                                (1, .0625, .125), (1, .0625, 0.)):
            label = 'size%d-mode%d-decay%08x-lr%08x' % (size, mode, wire.bits(decay), wire.bits(lr))
            session = wire.Session(app, directory=out / 'counter-boundaries' / label)
            try:
                config = wire.Payload(struct.pack('<6Q', *map(wire.bits, (lr, .5, .75, 1e-8, decay)), mode))
                records = session.run([wire.full([size], 2, 1), wire.full([size], 0), wire.full([size], 0),
                                       (24, 1, 2, 3, 1, config), (13, 1), (13, 2), (13, 3)])
                observed = [wire.meta(record)['version'] for record in records[-3:]]
                expected = [2 if mode == 1 and decay != 0. else 1, 1, 2]
                assert observed == expected, (label, observed, expected)
                session.finish()
                boundary.append({'case': label, 'pass': True, 'versions': observed,
                                 'source_expected_versions': expected, 'frames': session.frames})
            finally:
                if session.poll() is None:
                    os.kill(session.pid, signal.SIGKILL)
                    os.waitpid(session.pid, 0)
    for path, raw in snapshots.items():
        assert path.read_bytes() == raw
    assert sha(app.read_bytes()) == profile['executable_sha256']
    assert Path(__file__).read_bytes() == driver
    report = {'schema': 'heptatorch.owned-tensor-version-qualification.v1', 'status': 'PASS',
              'profile_sha256': sha((HERE / 'profile.json').read_bytes()),
              'qualification_driver_sha256': sha(driver),
              'native_executable_sha256': profile['executable_sha256'], 'build': build,
              'reports': reports, 'counter_boundary_cases': boundary,
              'expectation_adaptation': {'original_native_test_sha256': sha(original.encode()),
                  'executed_native_test_sha256': sha(changed.encode()), 'reversible_exact_text_edits': 2,
                  'numerical_assertions_and_tolerances_unchanged': True,
                  'independent_review_changed_only_by_target_identity': True,
                  'version_expectations_patch_sha256': sha(patch.encode()),
                  'real_pytorch_oracle_changed': False},
              'executed_source_sha256': {str(path.relative_to(out)): sha(raw) for path, raw in snapshots.items()},
              'source_and_native_image_unchanged_during_run': True,
              'full_pytorch_rewrite_complete': False,
              'momentum_state_construction_lifecycle_parity': 'OPEN'}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
