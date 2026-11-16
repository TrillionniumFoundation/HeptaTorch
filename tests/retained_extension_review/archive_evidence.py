#!/usr/bin/env python3
"""Stream and verify an isolated evidence directory without removing originals."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import stat
import tarfile


def digest(stream):
    h = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024*1024), b''):
        h.update(block)
        size += len(block)
    return size, h.hexdigest()


def file_meta(path):
    info = path.lstat()
    assert stat.S_ISREG(info.st_mode) and not path.is_symlink(), str(path)
    with path.open('rb') as f:
        size, sha = digest(f)
    return {'bytes': size, 'sha256': sha, 'mode': stat.S_IMODE(info.st_mode)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'archive', 'manifest'):
        p.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args()
    root, archive, manifest = [getattr(a, n).resolve() for n in ('root','archive','manifest')]
    assert root.is_dir() and not archive.exists() and not manifest.exists()
    files = {}
    for path in sorted(root.rglob('*')):
        assert not path.is_symlink(), str(path)
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            files[relative] = file_meta(path)
        else:
            assert path.is_dir(), str(path)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open('xb') as destination:
        with gzip.GzipFile(fileobj=destination, mode='wb', compresslevel=9, filename='', mtime=0) as packed:
            with tarfile.open(fileobj=packed, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for name, value in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = value['bytes']
                    info.mode = value['mode']
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    with (root/name).open('rb') as source:
                        tar.addfile(info, source)
    verified = {}
    with tarfile.open(archive, mode='r|gz') as tar:
        for member in tar:
            assert member.isfile() and member.name in files and member.name not in verified
            with tar.extractfile(member) as f:
                size, sha = digest(f)
            value = {'bytes': size, 'sha256': sha, 'mode': member.mode}
            assert value == files[member.name]
            assert member.size == size
            verified[member.name] = value
    assert verified == files
    # Drain the complete gzip stream independently to validate trailer/CRC too.
    with gzip.open(archive, 'rb') as f:
        tar_bytes, tar_sha = digest(f)
    for name, value in files.items():
        assert file_meta(root/name) == value, ('source changed', name)
    assert {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} == set(files)
    report = {'schema': 'heptatorch.raw-evidence.streamed-archive.v1',
              'status': 'PASS_COMPLETE_ARCHIVE_VERIFIED',
              'archive': {'name': archive.name, **file_meta(archive)},
              'member_count': len(files), 'expanded_file_bytes': sum(v['bytes'] for v in files.values()),
              'tar_stream_bytes': tar_bytes, 'tar_stream_sha256': tar_sha,
              'members': files, 'all_members_verified': True, 'gzip_trailer_verified': True,
              'original_files_unchanged': True, 'original_files_deleted': False}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({key: report[key] for key in ('status','archive','member_count','expanded_file_bytes')}))


if __name__ == '__main__':
    main()
