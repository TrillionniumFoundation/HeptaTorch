#!/usr/bin/env python3
"""Stream named evidence roots/files, then verify every original and tar member."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', action='append', default=[], help='archive-prefix=directory')
    p.add_argument('--file', action='append', default=[], help='archive-path=file')
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    a = p.parse_args()
    assert not a.archive.exists() and not a.manifest.exists()
    entries = {}

    def add(name, path):
        posix = PurePosixPath(name)
        assert not posix.is_absolute() and '..' not in posix.parts and name not in entries
        assert path.is_file() and not path.is_symlink()
        entries[name] = path.resolve()

    for item in a.root:
        label, actual = item.split('=', 1)
        root = Path(actual).resolve()
        assert root.is_dir() and not a.archive.resolve().is_relative_to(root)
        for path in sorted(root.rglob('*')):
            if path.is_file():
                add(label + '/' + path.relative_to(root).as_posix(), path)
    for item in a.file:
        name, path = item.split('=', 1)
        add(name, Path(path))
    rows = [{'path': name, 'bytes': path.stat().st_size, 'sha256': digest(path),
             'mode': stat.S_IMODE(path.stat().st_mode)} for name, path in sorted(entries.items())]
    a.archive.parent.mkdir(parents=True, exist_ok=True)
    with a.archive.open('xb') as target:
        with gzip.GzipFile(filename='', fileobj=target, mode='wb', mtime=0, compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as archive:
                for row in rows:
                    info = tarfile.TarInfo(row['path'])
                    info.size, info.mode, info.mtime = row['bytes'], row['mode'], 0
                    with entries[row['path']].open('rb') as raw:
                        archive.addfile(info, raw)
    expected = {row['path']: row for row in rows}
    seen = set()
    with tarfile.open(a.archive, 'r|gz') as archive:
        for member in archive:
            assert member.isfile() and member.name in expected and member.name not in seen
            seen.add(member.name)
            row = expected[member.name]
            assert member.size == row['bytes'] and stat.S_IMODE(member.mode) == row['mode']
            h, count = hashlib.sha256(), 0
            stream = archive.extractfile(member)
            while chunk := stream.read(1 << 20):
                h.update(chunk)
                count += len(chunk)
            assert count == row['bytes'] and h.hexdigest() == row['sha256']
            original = entries[member.name]
            assert original.stat().st_size == row['bytes']
            assert stat.S_IMODE(original.stat().st_mode) == row['mode'] and digest(original) == row['sha256']
    assert seen == set(expected)
    report = {'schema': 'heptatorch.retained-gradient-independent-raw-custody.v1',
              'archive': a.archive.name, 'archive_bytes': a.archive.stat().st_size,
              'archive_sha256': digest(a.archive), 'member_count': len(rows),
              'expanded_bytes': sum(row['bytes'] for row in rows),
              'every_member_decoded_and_size_mode_SHA_verified': True,
              'every_original_file_rechecked_after_archive': True,
              'original_expanded_files_retained': True, 'members': rows}
    a.manifest.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'members'}, indent=2))


if __name__ == '__main__':
    main()
