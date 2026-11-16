#!/usr/bin/env python3
"""Stream all independent raw evidence into XZ, then decode and verify each file."""
import argparse
import hashlib
import json
import lzma
from pathlib import Path
import stat
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    a = p.parse_args()
    assert not a.archive.exists() and not a.manifest.exists()
    rows = []
    for path in sorted(a.input.rglob('*')):
        if path.is_file():
            rows.append({'path': path.relative_to(a.input).as_posix(),
                         'bytes': path.stat().st_size, 'sha256': digest(path),
                         'mode': stat.S_IMODE(path.stat().st_mode)})
    with lzma.open(a.archive, 'xb', preset=6) as compressed:
        with tarfile.open(fileobj=compressed, mode='w|') as archive:
            for row in rows:
                archive.add(a.input / row['path'], arcname=row['path'], recursive=False)
    expected = {row['path']: row for row in rows}
    seen = set()
    with tarfile.open(a.archive, 'r|xz') as archive:
        for member in archive:
            assert member.isfile() and member.name in expected and member.name not in seen
            seen.add(member.name)
            row = expected[member.name]
            assert member.size == row['bytes'] and stat.S_IMODE(member.mode) == row['mode']
            h = hashlib.sha256()
            count = 0
            f = archive.extractfile(member)
            while chunk := f.read(1 << 20):
                h.update(chunk)
                count += len(chunk)
            assert count == row['bytes'] and h.hexdigest() == row['sha256']
            assert digest(a.input / member.name) == row['sha256']
    assert seen == set(expected)
    report = {
        'schema': 'heptatorch.adam-distributions.independent-raw-custody.v1',
        'archive': a.archive.name, 'archive_bytes': a.archive.stat().st_size,
        'archive_sha256': digest(a.archive), 'member_count': len(rows),
        'expanded_bytes': sum(r['bytes'] for r in rows),
        'every_member_decoded_and_exact_SHA_and_size_and_mode_verified': True,
        'original_expanded_tmpfs_files_retained': True,
        'contains_first_failures_and_final_passes': True, 'members': rows,
    }
    a.manifest.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'members'}, indent=2))


if __name__ == '__main__':
    main()
