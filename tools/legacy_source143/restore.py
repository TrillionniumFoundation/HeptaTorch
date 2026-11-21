"""Verify or restore the fixed historical143 HIA source without changing old gates.

The current native runtime never needs this source. Explicit legacy targets and
the historical native, scalar F64 and CPU RNG gates retain their original source
path and exact source/image predicates. This tool only supplies those bytes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import urllib.error
import urllib.request
import uuid


HERE = Path(__file__).absolute().parent
COMMIT = 'f8ad6df1d0d27776b7efb2c62db1356189523f39'
REPOSITORY = 'TrillionniumFoundation/HeptaTorch'
SOURCE_PATH = 'src/heptatorch.hepta'
SOURCE_SHA256 = '264e9c5c4e8ea1834724cc0d8d859952016daffe4af330a8daa7745e7a5e5233'
SOURCE_BLOB = '2e994b4c15f5c6760d65aa555c0bca6c0bd88a56'
SOURCE_BYTES = 481516
SOURCE_FUNCTIONS = 143
SOURCE_MODE = 0o644
RAW_URL = 'https://raw.githubusercontent.com/' + REPOSITORY + '/' + COMMIT + '/' + SOURCE_PATH
EXPECTED_MANIFEST = dict(schema='heptatorch.historical-source143.v1',
    repository=REPOSITORY, commit=COMMIT, path=SOURCE_PATH, git_blob_sha1=SOURCE_BLOB,
    sha256=SOURCE_SHA256, bytes=SOURCE_BYTES, functions=SOURCE_FUNCTIONS,
    git_mode='100644', filesystem_mode=SOURCE_MODE, raw_url=RAW_URL,
    historical_source_only=True, current_runtime_source=False, full_legacy_retirement_claimed=False)


class CustodyError(ValueError):
    pass


def require(value, message):
    if not value:
        raise CustodyError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def absolute(path):
    path = Path(path)
    require('..' not in path.parts, 'parent traversal is not an admitted source path')
    return Path(os.path.abspath(path))


def directory_fd(path, create=False):
    """Walk directories through pinned descriptors, never following a symlink."""
    path = absolute(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(path.anchor, flags)
    try:
        for name in path.parts[1:]:
            try:
                child = os.open(name, flags, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(name, 0o755, dir_fd=fd)
                except FileExistsError:
                    pass
                child = os.open(name, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def read_regular(path, maximum, expected_mode=SOURCE_MODE):
    path = absolute(path)
    parent = directory_fd(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent)
    finally:
        os.close(parent)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), 'only a regular source or custody file is admitted')
        require(before.st_nlink == 1, 'hard-linked source and custody files are not admitted')
        require(stat.S_IMODE(before.st_mode) == expected_mode, 'source or custody file mode differs')
        require(before.st_size <= maximum, 'source or custody file exceeds its complete byte limit')
        chunks = []
        total = 0
        while True:
            chunk = os.read(fd, min(65536, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            require(total <= maximum, 'source or custody file grew beyond its complete byte limit')
        after = os.fstat(fd)
        fields = ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        require(all(getattr(before, key) == getattr(after, key) for key in fields),
                'source or custody file changed during its complete read')
        raw = b''.join(chunks)
        require(len(raw) == before.st_size, 'source or custody read did not reach its complete file size')
        return raw, dict(path=str(path), bytes=len(raw), sha256=sha(raw),
            mode=stat.S_IMODE(before.st_mode), device=before.st_dev, inode=before.st_ino,
            links=before.st_nlink, mtime_ns=before.st_mtime_ns, ctime_ns=before.st_ctime_ns)
    finally:
        os.close(fd)


def source_identity(raw):
    require(len(raw) == SOURCE_BYTES, 'historical143 source byte count differs')
    require(sha(raw) == SOURCE_SHA256, 'historical143 source SHA-256 differs')
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()
    require(blob == SOURCE_BLOB, 'historical143 Git blob identity differs')
    rows = list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]*\n.*?^end\n', raw, re.M | re.S))
    require(len(rows) == SOURCE_FUNCTIONS and len({row[1] for row in rows}) == SOURCE_FUNCTIONS,
            'historical143 unique complete function inventory differs')
    require(raw == b'hia-asm.v1 hepta.hia.program.v8 selfhost_exec\n' + b''.join(row[0] for row in rows),
            'historical143 source contains unaccounted bytes or incomplete functions')
    return dict(bytes=len(raw), sha256=sha(raw), git_blob_sha1=blob, functions=len(rows),
                git_mode='100644', filesystem_mode=SOURCE_MODE)


def read_source(path, optional=False):
    try:
        raw, row = read_regular(path, SOURCE_BYTES)
    except FileNotFoundError:
        if optional:
            return None
        raise
    row.update(source_identity(raw))
    return raw, row


def write_new(path, raw):
    """Publish a complete file atomically; never replace an existing path."""
    path = absolute(path)
    parent = directory_fd(path.parent, create=True)
    temporary = '.legacy143-' + uuid.uuid4().hex + '.tmp'
    fd = None
    made = False
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=parent)
        made = True
        at = 0
        while at < len(raw):
            count = os.write(fd, raw[at:])
            require(count > 0, 'custody write made no progress')
            at += count
        os.fchmod(fd, SOURCE_MODE)
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.link(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        os.unlink(temporary, dir_fd=parent)
        made = False
        os.fsync(parent)
    finally:
        if fd is not None:
            os.close(fd)
        if made:
            os.unlink(temporary, dir_fd=parent)
        os.close(parent)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        raise CustodyError('the fixed historical raw source must not redirect')


def download_source():
    request = urllib.request.Request(RAW_URL, headers={
        'Accept': 'application/octet-stream', 'Accept-Encoding': 'identity',
        'User-Agent': 'HeptaTorch-historical143-source-custody'})
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(request, timeout=30) as response:
        require(response.status == 200 and response.geturl() == RAW_URL,
                'the fixed historical raw source URL did not return status 200')
        require(response.headers.get('Content-Encoding', 'identity') == 'identity',
                'the historical raw source response must contain original unencoded bytes')
        length = response.headers.get('Content-Length')
        require(length is None or length == str(SOURCE_BYTES), 'historical raw Content-Length differs')
        raw = response.read(SOURCE_BYTES + 1)
        require(response.read(1) == b'', 'historical raw source has trailing response bytes')
        identity = source_identity(raw)
    return raw, dict(kind='fixed-commit-HTTPS-download', requested_url=RAW_URL,
        effective_url=RAW_URL, http_status=200, redirects_followed=False,
        content_length_header=length, complete_response_EOF=True, identity=identity)


def snapshot(path, raw):
    try:
        existing, row = read_regular(path, max(len(raw), 1))
    except FileNotFoundError:
        write_new(path, raw)
        existing, row = read_regular(path, max(len(raw), 1))
    require(existing == raw, 'existing custody snapshot differs; refusing replacement')
    return row


def restore(repository, store, source_file=None, offline=False):
    repository, store = absolute(repository), absolute(store)
    require(store != repository and store != Path('/'), 'custody store must be a dedicated directory')
    os.close(directory_fd(repository, create=True))
    os.close(directory_fd(store, create=True))
    target, cache = repository / SOURCE_PATH, store / 'canonical.hepta'
    receipt_path = store / 'receipts' / (uuid.uuid4().hex + '.json')
    record = dict(schema='heptatorch.historical143.source-custody.v1', status='PREPARING',
        repository=str(repository), source_path=str(target), cache_path=str(cache),
        started_utc=datetime.now(timezone.utc).isoformat(), offline=offline,
        historical_source=EXPECTED_MANIFEST, source_installed=False, cache_installed=False,
        current_runtime_dependency_added=False, native_executed=False, Torch_executed=False,
        full_legacy_retirement_claimed=False, errors=[])
    try:
        manifest_raw, manifest_row = read_regular(HERE / 'manifest.json', 16384)
        require(json.loads(manifest_raw) == EXPECTED_MANIFEST, 'fixed historical source manifest differs')
        helper_raw, helper_row = read_regular(Path(__file__), 131072)
        record['manifest'] = manifest_row
        record['restorer'] = helper_row
        record['manifest_snapshot'] = snapshot(store / 'provenance' / (sha(manifest_raw) + '.json'), manifest_raw)
        record['restorer_snapshot'] = snapshot(store / 'provenance' / (sha(helper_raw) + '.py'), helper_raw)
        existing = read_source(target, optional=True)
        cached = read_source(cache, optional=True)
        supplied = read_source(source_file) if source_file is not None else None
        record['source_before'] = existing[1] if existing else None
        record['cache_before'] = cached[1] if cached else None
        record['supplied_source'] = supplied[1] if supplied else None
        if supplied:
            raw, acquisition = supplied[0], dict(kind='verified-local-source', input=supplied[1])
        elif existing:
            raw, acquisition = existing[0], dict(kind='verified-existing-source', input=existing[1])
        elif cached:
            raw, acquisition = cached[0], dict(kind='verified-cache', input=cached[1])
        else:
            require(not offline, 'historical source and cache are absent in offline mode')
            raw, acquisition = download_source()
        record['acquisition'] = acquisition
        record['verified_identity'] = source_identity(raw)
        if not cached:
            write_new(cache, raw)
            record['cache_installed'] = True
        if not existing:
            write_new(target, raw)
            record['source_installed'] = True
        actual, after = read_source(target)
        cache_raw, cache_after = read_source(cache)
        require(actual == cache_raw == raw, 'source and custody cache are not the same complete original bytes')
        require(not existing or after == existing[1], 'existing historical source changed during custody')
        require(not cached or cache_after == cached[1], 'existing historical source cache changed during custody')
        if supplied:
            reread, supplied_after = read_source(source_file)
            require(reread == raw and supplied_after == supplied[1], 'supplied source changed during custody')
        require(read_regular(HERE / 'manifest.json', 16384) == (manifest_raw, manifest_row),
                'historical source manifest changed during custody')
        require(read_regular(Path(__file__), 131072) == (helper_raw, helper_row),
                'historical source restorer changed during custody')
        record.update(status='PASS_HISTORICAL143_SOURCE_CUSTODY', source_after=after,
            cache_after=cache_after, all_original_inputs_unchanged=True,
            original_source_bytes_preserved=True, original_source_mode_preserved=True)
    except Exception as error:
        record.update(status='FAIL_HISTORICAL143_SOURCE_CUSTODY', all_original_inputs_unchanged=False)
        record['errors'].append(dict(type=type(error).__name__, message=str(error)))
    record['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_new(receipt_path, (json.dumps(record, indent=2, sort_keys=True) + '\n').encode('utf-8'))
    receipt_raw, receipt_row = read_regular(receipt_path, 131072)
    require(json.loads(receipt_raw) == record, 'complete written custody receipt differs')
    return record, receipt_row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=HERE.parents[1])
    parser.add_argument('--store', type=Path)
    parser.add_argument('--source-file', type=Path, help='Optional original local source; all fixed identities are still mandatory.')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args(argv)
    repository = absolute(args.repository)
    store = args.store if args.store is not None else repository / 'build/legacy-source143'
    try:
        record, receipt = restore(repository, store, args.source_file, args.offline)
        print(json.dumps(dict(status=record['status'], source_sha256=SOURCE_SHA256,
            source_installed=record['source_installed'], cache_installed=record['cache_installed'],
            receipt=receipt, errors=record['errors']), sort_keys=True))
        return 0 if record['status'] == 'PASS_HISTORICAL143_SOURCE_CUSTODY' else 1
    except Exception as error:
        print(json.dumps(dict(status='FAIL_HISTORICAL143_SOURCE_CUSTODY',
            errors=[dict(type=type(error).__name__, message=str(error))], receipt_written=False)), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
