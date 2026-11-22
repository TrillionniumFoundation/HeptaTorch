"""Publish the exact fixed wheel under its evidence path without cache aliases."""

import argparse
import os
from pathlib import Path
import stat
import tempfile

import pinned_reference_custody as custody


def _accept_existing(target):
    data, identity = custody.stable_bytes(target)
    attached = target.lstat()
    custody.require(len(data) == custody.WHEEL_BYTES and identity['sha256'] == custody.WHEEL_SHA256
                    and attached.st_nlink == 1 and (attached.st_dev, attached.st_ino) ==
                    (identity['device'], identity['inode']),
                    'concurrent target is not the exact independent wheel archive')
    return target


def materialize(repository, wheel=None, cache=None):
    target = Path(repository).resolve() / custody.WHEEL_ARCHIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        before = target.lstat()
    except FileNotFoundError:
        before = None
    if before is not None:
        # Preserve the original refusal for nonregular or incorrect archives;
        # only the exact bytes from an old hardlinked preparation can migrate.
        data, record = custody.stable_bytes(target)
        mode = record['mode']
        custody.require(len(data) == custody.WHEEL_BYTES and
                        record['sha256'] == custody.WHEEL_SHA256,
                        'existing original wheel archive differs')
        attached = target.lstat()
        custody.require((attached.st_dev, attached.st_ino, attached.st_ctime_ns) ==
                        (before.st_dev, before.st_ino, before.st_ctime_ns),
                        'existing original wheel archive changed during validation')
        if attached.st_nlink == 1:
            return target
    else:
        cache = Path.home() / '.cache/pip' if cache is None else Path(cache)
        candidates = ([Path(wheel).resolve(strict=True)] if wheel is not None else
                      sorted(cache.glob('http*/**/*.body')))
        for candidate in candidates:
            if candidate.is_file() and candidate.stat().st_size == custody.WHEEL_BYTES:
                data, record = custody.stable_bytes(candidate)
                if len(data) == custody.WHEEL_BYTES and record['sha256'] == custody.WHEEL_SHA256:
                    mode = record['mode']
                    break
        else:
            raise ValueError('exact original wheel absent; pass --wheel pointing to the official archive')

    # A fresh temporary inode both avoids cache hardlinks and allows atomic
    # publication without writing through a previously linked target.
    fd, name = tempfile.mkstemp(prefix=target.name + '.materializing-', dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            os.fchmod(stream.fileno(), mode)
        del data
        raw, identity = custody.stable_bytes(temporary)
        custody.require(len(raw) == custody.WHEEL_BYTES and identity['sha256'] == custody.WHEEL_SHA256
                        and identity['mode'] == mode and temporary.lstat().st_nlink == 1,
                        'exact unique ordinary wheel archive before publication')
        del raw
        if before is None:
            # Link from our fresh private inode, not the cache, so a concurrent
            # appearance cannot replace an entity we have not checked.
            try:
                os.link(temporary, target)
            except FileExistsError:
                return _accept_existing(target)
            temporary.unlink()
        else:
            attached = target.lstat()
            custody.require((attached.st_dev, attached.st_ino, attached.st_ctime_ns) ==
                            (before.st_dev, before.st_ino, before.st_ctime_ns),
                            'linked original wheel archive changed before migration')
            temporary.replace(target)
        after = target.lstat()
        custody.require(stat.S_ISREG(after.st_mode) and after.st_nlink == 1 and
                        (after.st_dev, after.st_ino, after.st_size, stat.S_IMODE(after.st_mode)) ==
                        (identity['device'], identity['inode'], custody.WHEEL_BYTES, mode),
                        'exact unique ordinary wheel archive after publication')
    finally:
        temporary.unlink(missing_ok=True)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True, type=Path)
    parser.add_argument('--wheel', type=Path)
    args = parser.parse_args()
    materialize(args.repository, args.wheel)
