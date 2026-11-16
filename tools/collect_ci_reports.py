"""Index original CI report bytes for lossless artifact export."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-status', choices=('success', 'failure', 'cancelled'), required=True)
    parser.add_argument('--build', type=Path, default=Path('build'))
    parser.add_argument('--workflow', type=Path, default=Path('.github/workflows/native.yml'))
    args = parser.parse_args()
    workflow = args.workflow.read_bytes()
    required = sorted(set(re.findall(
        rb'tools/emit_ci_report\.py (build/[A-Za-z0-9_.-]+\.json)(?:\s|$)', workflow)))
    assert required, 'workflow has no literal report-envelope paths'
    required = [Path(path.decode('ascii')).name for path in required]
    args.build.mkdir(parents=True, exist_ok=True)
    output = args.build / 'ci-report-manifest.json'
    reports = []
    for path in sorted(args.build.glob('*.json')):
        if path == output:
            continue
        assert path.is_file() and not path.is_symlink(), str(path)
        raw = path.read_bytes()
        reports.append({'path': path.name, 'bytes': len(raw),
                        'sha256': hashlib.sha256(raw).hexdigest()})
    present = {row['path'] for row in reports}
    missing = [name for name in required if name not in present]

    def git(*arguments):
        return subprocess.check_output(['git', *arguments], text=True).strip()

    manifest = {
        'schema': 'heptatorch.ci-report-artifact.v1',
        'status': 'COMPLETE' if reports and not missing else 'PARTIAL',
        'job_status_before_export': args.job_status,
        'checkout_commit': git('rev-parse', 'HEAD'),
        'checkout_tree': git('rev-parse', 'HEAD^{tree}'),
        'tracked_worktree_dirty': bool(git('diff', '--name-only', 'HEAD')),
        'github': {key: os.environ.get(key, '') for key in (
            'GITHUB_REPOSITORY', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT',
            'GITHUB_JOB', 'GITHUB_SHA', 'GITHUB_EVENT_NAME', 'GITHUB_REF',
            'HEPTATORCH_SOURCE_HEAD_SHA')},
        'workflow_sha256': hashlib.sha256(workflow).hexdigest(),
        'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'required_envelope_reports': required,
        'missing_envelope_reports': missing,
        'reports': reports,
        'original_report_bytes_reserialized': False,
    }
    output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'manifest': str(output), 'status': manifest['status'],
                      'reports': len(reports), 'missing': missing,
                      'bytes': sum(row['bytes'] for row in reports)}))
    if args.job_status == 'success':
        assert reports and not missing, 'successful job is missing required original reports'
        assert not manifest['tracked_worktree_dirty'], 'tracked checkout changed during CI'


if __name__ == '__main__':
    main()
