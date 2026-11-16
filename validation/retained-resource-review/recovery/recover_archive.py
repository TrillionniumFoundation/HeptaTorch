#!/usr/bin/env python3
"""Recover only complete, hashed members of the interrupted evidence archive.

This is a byte recovery operation. It does not execute a native program, repair
the interrupted stream, or reconstruct the unavailable tail of an old run.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile

ARCHIVE = '45deb8ecdba66e41dbea7807c1cf8d4e4db87b62323a0a4dff3a102e518d29b6'
FORENSIC = 'd4b60c7c62f5816016a675c1e3bd2a7e49d0731e9034b77a623d97b240b98d10'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'forensic', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    assert digest(a.archive) == ARCHIVE and digest(a.forensic) == FORENSIC
    prior = json.loads(a.forensic.read_bytes())
    assert prior['tar_status'] == 'PARTIAL' and prior['complete_member_count'] == 114
    rows = prior['members']
    assert len({row['path'] for row in rows}) == len(rows)
    a.output.mkdir(parents=True, exist_ok=False)
    expanded = a.output / 'complete-members'
    expanded.mkdir()
    read_error = None
    count = 0
    try:
        with tarfile.open(a.archive, 'r|gz') as source:
            for member in source:
                if count == len(rows):
                    assert member.name == prior['incomplete_member']['path']
                    assert member.size == prior['incomplete_member']['bytes']
                    # Do not publish the incomplete file as a complete member.
                    fragment = source.extractfile(member)
                    while fragment.read(1 << 20):
                        pass
                    raise AssertionError('previously incomplete member unexpectedly complete')
                row = rows[count]
                assert member.name == row['path'] and member.isfile()
                path = PurePosixPath(member.name)
                assert not path.is_absolute() and '..' not in path.parts
                assert member.size == row['bytes'] and stat.S_IMODE(member.mode) == row['mode']
                raw = source.extractfile(member).read()
                assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
                target = expanded.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as f:
                    f.write(raw)
                target.chmod(row['mode'])
                count += 1
    except tarfile.ReadError as error:
        read_error = str(error)
    assert count == 114 and read_error == 'unexpected end of data'
    archive = a.output / 'complete-members.tar.gz'
    with archive.open('xb') as target:
        with gzip.GzipFile(filename='', fileobj=target, mode='wb', mtime=0, compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as output:
                for row in rows:
                    raw = (expanded / row['path']).read_bytes()
                    assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
                    info = tarfile.TarInfo(row['path'])
                    info.size, info.mode, info.mtime = len(raw), row['mode'], 0
                    output.addfile(info, io.BytesIO(raw))
    observed = []
    with tarfile.open(archive, 'r:gz') as source:
        for member in source:
            row = rows[len(observed)]
            assert member.isfile() and member.name == row['path']
            raw = source.extractfile(member).read()
            assert member.size == len(raw) == row['bytes']
            assert member.mode == row['mode']
            assert hashlib.sha256(raw).hexdigest() == row['sha256']
            assert (expanded / row['path']).read_bytes() == raw
            observed.append(row)
    assert len(observed) == 114
    # Reading to actual gzip EOF separately checks trailer/CRC, not just tar EOF.
    with gzip.open(archive, 'rb') as stream:
        while stream.read(1 << 20):
            pass
    assert digest(a.archive) == ARCHIVE and digest(a.forensic) == FORENSIC
    result = {'schema': 'heptatorch.retained-resource-recovered-members.v1',
              'status': 'COMPLETE_MEMBERS_RECOVERED_WITH_TAIL_LOSS',
              'original_archive_sha256': ARCHIVE, 'forensic_sha256': FORENSIC,
              'original_archive_complete': False, 'old_process_completion': 'UNKNOWN',
              'old_execution_reports_reclassified': False, 'native_executed_by_recovery': False,
              'archive': archive.name, 'archive_bytes': archive.stat().st_size,
              'archive_sha256': digest(archive), 'member_count': len(rows),
              'expanded_bytes': sum(row['bytes'] for row in rows),
              'gzip_complete_crc_verified': True, 'members_size_mode_SHA_verified': True,
              'original_expanded_files_available': False,
              'incomplete_original_member': prior['incomplete_member'],
              'missing_original_tail': 'typed-handoff link request remainder, renamed units, transport, and any subsequent files were not preserved by this interrupted archive',
              'members': rows}
    report = a.output / 'report.json'
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'members'}, indent=2))
    print('report_sha256', digest(report))


if __name__ == '__main__':
    main()
