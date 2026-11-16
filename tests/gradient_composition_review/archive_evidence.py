#!/usr/bin/env python3
"""Stream, then independently verify exact raw qualification evidence.

The archive is a custody record, not an additional native or numerical test.
It never deletes source files. Each source is checked again after the archive
has been decoded, and every archived member retains its original file mode.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import stat
import tarfile


def digest(path):
    state = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        while chunk := stream.read(1 << 20):
            state.update(chunk)
            size += len(chunk)
    return size, state.hexdigest()


def inspect_member(stream):
    state = hashlib.sha256()
    size = 0
    while chunk := stream.read(1 << 20):
        state.update(chunk)
        size += len(chunk)
    return size, state.hexdigest()


def verify(archive, records, check_sources):
    expected = {row['member']: row for row in records}
    assert len(expected) == len(records)
    actual = set()
    with tarfile.open(archive, 'r|gz') as reader:
        for member in reader:
            assert member.isfile() and member.name in expected
            assert member.name not in actual
            actual.add(member.name)
            row = expected[member.name]
            assert member.size == row['bytes']
            assert member.mode == row['mode']
            with reader.extractfile(member) as stream:
                assert inspect_member(stream) == (row['bytes'], row['sha256'])
    assert actual == set(expected)
    if check_sources:
        for row in records:
            path = Path(row['source_at_capture'])
            assert not path.is_symlink() and path.is_file()
            assert stat.S_IMODE(path.stat().st_mode) == row['mode']
            assert digest(path) == (row['bytes'], row['sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', help='archive prefix=source file or directory')
    parser.add_argument('--output', type=Path, help='new output directory')
    parser.add_argument('--verify-manifest', type=Path)
    parser.add_argument('--check-original-sources', action='store_true')
    args = parser.parse_args()
    if args.verify_manifest:
        assert not args.source and not args.output
        record = json.loads(args.verify_manifest.read_text())
        assert record['schema'] == 'heptatorch.independent-raw-archive.v1'
        assert record['status'] == 'PASS'
        archive = args.verify_manifest.parent / record['archive']['file']
        assert digest(archive) == (record['archive']['bytes'], record['archive']['sha256'])
        verify(archive, record['members'], args.check_original_sources)
        print(json.dumps({'status': 'PASS', 'members': len(record['members']),
                          'archive_sha256': record['archive']['sha256'],
                          'original_sources_rechecked': args.check_original_sources}))
        return
    assert args.source and args.output
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = []
    prefixes = set()
    for argument in args.source:
        prefix, original = argument.split('=', 1)
        assert re.fullmatch(r'[a-zA-Z0-9_.-]+', prefix) and prefix not in ('.', '..')
        assert prefix not in prefixes
        prefixes.add(prefix)
        path = Path(original)
        assert not path.is_symlink()
        path = path.resolve()
        assert path.exists() and out != path and path not in out.parents
        if path.is_file():
            sources.append((prefix, path))
        else:
            assert path.is_dir()
            for item in sorted(path.rglob('*')):
                assert not item.is_symlink(), item
                if item.is_file():
                    sources.append((prefix + '/' + item.relative_to(path).as_posix(), item))
                else:
                    assert item.is_dir(), item
    assert sources
    rows = []
    for name, path in sources:
        size, sha = digest(path)
        rows.append({'member': name, 'bytes': size, 'sha256': sha,
                     'mode': stat.S_IMODE(path.stat().st_mode),
                     'source_at_capture': str(path)})
    temporary = out / 'observations.tar.gz.partial'
    with temporary.open('xb') as sink:
        with gzip.GzipFile(filename='', fileobj=sink, mode='wb', mtime=0, compresslevel=9) as zipped:
            with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.PAX_FORMAT) as writer:
                for row in rows:
                    info = tarfile.TarInfo(row['member'])
                    info.size = row['bytes']
                    info.mode = row['mode']
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    with Path(row['source_at_capture']).open('rb') as stream:
                        writer.addfile(info, stream)
    verify(temporary, rows, True)
    archive = out / 'observations.tar.gz'
    temporary.rename(archive)
    size, sha = digest(archive)
    report = {
        'schema': 'heptatorch.independent-raw-archive.v1', 'status': 'PASS',
        'archive': {'file': archive.name, 'bytes': size, 'sha256': sha},
        'members': rows, 'member_count': len(rows),
        'uncompressed_file_bytes': sum(row['bytes'] for row in rows),
        'every_member_decoded_and_checked': True, 'source_bytes_and_modes_rechecked': True,
        'source_files_deleted': False,
        'script_sha256': digest(Path(__file__))[1],
    }
    (out / 'manifest.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'archive', 'member_count',
                                                   'uncompressed_file_bytes')}))


if __name__ == '__main__':
    main()
