#!/usr/bin/env python3
"""Read-only source/report review. Executes no native or Torch process."""
import ast
import hashlib
import json
from pathlib import Path
import stat

ROOT = Path('/dev/shm/heptatorch-d26699e25fea')
REPO = ROOT / 'HeptaTorch-integer-tensors-integration-07'
RUN = ROOT / 'integer07-root-cold-integration-01'
OUT = Path(__file__).resolve().parent
SOURCE = 'adf6cae67666273aa6fb77989080ca3b1e3632188d2d7956a6d8b71808787e62'
IMAGE = 'b7324639bade3312981d1cbf9c2491b9fcff04fb6c3f808e9c5e1a7c2ca88f08'
PINS = {
    'tests/qualify_integer_tensors.py': 'b158fda1ca21b9cc3f29036d3654827e38151b914d7854f4cfa59b2b930dd52e',
    'tools/integer_tensors/qualify.py': '94de30d2ebd33c9427eb531b1ec28d8513d60dd711bb0624e91d47947e071a7a',
    'tools/integer_tensors/rebuild.py': 'b5a37adfe04a5208194c4c0fbc5047a4d86baf11723c01b12f0cc3f15ab864ca',
}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    for name, pin in PINS.items():
        assert sha(REPO / name) == pin
        ast.parse((REPO / name).read_text())
    report_path = RUN / 'report.json'
    assert sha(report_path) == '245257046806b765aee09d4c37d32d99cde8ed31e612f00772021ef9eb1bb781'
    report = json.loads(report_path.read_bytes())
    assert report['status'] == 'PASS'
    assert report['source_sha256'] == SOURCE and report['native_sha256'] == IMAGE
    assert report['command_count'] == len(report['commands']) == 4
    assert [r['name'] for r in report['commands']] == ['producer10', 'resources6', 'owned-entry5', 'typed-handoff2']
    assert report['producer_executed_this_invocation'] is True
    assert report['producer_completed_this_invocation'] is True
    assert report['whole_pytorch_gate'] == 'HOLD' and report['closed_schema_count'] == 0
    assert report['full_pytorch_rewrite_complete'] is False
    assert report['Torch_executed_here'] is False
    for row in report['commands']:
        assert row['status'] == 'PASS' and row['exit_code'] == 0
        assert row['process_started'] is True and row['process_completed'] is True
        assert row['argv'][1:3] == ['-S', '-B']
        assert sha(RUN / row['report']) == row['report_sha256']
        for stream in ('stdout', 'stderr'):
            file = RUN / (row['name'] + '.' + stream)
            assert file.stat().st_size == row[stream]['bytes']
            assert sha(file) == row[stream]['sha256']
        if row['name'] in ('resources6', 'owned-entry5'):
            assert row['argv'][row['argv'].index('--expected-source-sha256') + 1] == SOURCE
            assert row['argv'][row['argv'].index('--expected-native-sha256') + 1] == IMAGE
    assert len(report['checked_inputs']) == 219
    for name, pin in report['checked_inputs'].items():
        p = Path(name)
        assert p.is_file() and not p.is_symlink()
        assert p.stat().st_size == pin['bytes']
        assert stat.S_IMODE(p.stat().st_mode) == pin['mode']
        assert sha(p) == pin['sha256']
    assert report['all_inputs_unchanged'] is True
    assert report['new_build']['actual_units_verified'] == 11
    assert report['parent_build']['actual_units_verified'] == 11
    assert report['producer_admission']['actual_cold_build_verified'] is True
    assert report['producer_admission']['original_bounded_legacy_statuses_preserved'] is True
    separate = report['separate_required_CI_gates']
    assert separate['live32'] == separate['complete_accounted35_and_copied18'] == 'NOT_EXECUTED_BY_NATIVE_COORDINATOR'
    assert separate['no_inference_from_old_profile_or_inherited_suites'] is True
    evidence = report['complete_current_execution_evidence']
    assert evidence['single_XZ_stream_EOF'] is True
    assert evidence['all_member_bytes_modes_sizes_verified'] is True
    assert sha(RUN / evidence['archive']) == evidence['archive_sha256']
    assert sha(RUN / evidence['manifest']) == evidence['manifest_sha256']
    result = {
        'schema': 'heptatorch.integer-tensor-root-coordinator-source-review.v1',
        'status': 'NO_BLOCKER_IN_REVIEWED_SCOPE',
        'source_sha256': SOURCE, 'native_sha256': IMAGE,
        'reviewed_sources': PINS, 'executed_review_sha256': sha(__file__),
        'actual_root_report_sha256': sha(report_path),
        'checked_existing_input_files': 219,
        'native_executions_by_this_review': 0, 'Torch_executions_by_this_review': 0,
        'scope': [
            'Frozen complete producer, resource, typed and independent40 manifests bind bytes, lengths and modes before execution and again afterward.',
            'Create-only outer and child outputs, actual Popen process receipts, exact cold rebuild source/image and all eleven objects are required.',
            'The producer10 fixed counts and original54 indices are checked; original retained QUALIFIED_EXPOSED_SCOPE and inherited OPEN status boundaries remain explicit.',
            'The three independent native commands consume the new cold directory; resource and entry defaults are overridden with explicit final07 source/image pins.',
            'The canonical stack compiler, session linker, runtime linker and flat compiler have fixed byte identities; the two stack executable paths must be from the same bundle.',
            'XZ verification requires a complete single stream without trailing or concatenated data, every declared member exactly once and exact size, SHA and mode.',
            'Failure paths record process started/completed accurately and preserve argv/stdout/stderr/partial evidence without setting successful suite completion.',
            'Live32, complete accounted35 and copied controls remain separate required gates; bounded suite counts are not summed as unique API coverage.',
        ],
        'limitations': [
            'This is a source and existing-evidence identity review, not another cold or numerical execution.',
            'No hostile external report authenticity guarantee is inferred from locally generated report checks.',
            'No GitHub CI result or full-schema acceptance is inferred from the local four-gate result.',
        ],
    }
    p = OUT / 'report.json'
    p.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'report': str(p), 'sha256': sha(p), 'checked_files': 219}))


if __name__ == '__main__':
    main()
