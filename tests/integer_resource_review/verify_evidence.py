#!/usr/bin/env python3
"""Verify the exact compressed raw qualification archive without executing it."""
import argparse
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath
import tarfile


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(archive, manifest):
    archive, manifest = Path(archive), Path(manifest)
    m = json.loads(manifest.read_text())
    assert m['schema'] == 'heptatorch.i64-resource-raw-evidence.v1'
    assert archive.stat().st_size == m['archive_bytes'] and digest_file(archive) == m['archive_sha256']
    decoder = lzma.LZMADecompressor()
    uncompressed = hashlib.sha256()
    total = 0
    with archive.open('rb') as f:
        while True:
            block = f.read(1 << 20) if decoder.needs_input else b''
            if not block and decoder.needs_input:
                break
            result = decoder.decompress(block, max_length=1 << 20)
            uncompressed.update(result)
            total += len(result)
            if decoder.eof:
                assert not decoder.unused_data and not f.read(1), 'trailing data or concatenated XZ stream'
                break
    assert decoder.eof and uncompressed.hexdigest() == m['tar_sha256'] and total == m['tar_bytes']
    expected = m['files']
    assert expected
    seen = set()
    with lzma.open(archive, 'rb') as stream, tarfile.open(fileobj=stream, mode='r|') as tf:
        for member in tf:
            path = PurePosixPath(member.name)
            assert member.isfile() and not member.issym() and not member.islnk()
            assert not path.is_absolute() and '..' not in path.parts and str(path) == member.name
            assert member.name in expected and member.name not in seen
            e = expected[member.name]
            assert member.size == e['bytes'] and member.mode == e['mode']
            raw = tf.extractfile(member)
            h = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: raw.read(1 << 20), b''):
                h.update(chunk)
                size += len(chunk)
            assert size == e['bytes'] and h.hexdigest() == e['sha256']
            seen.add(member.name)
    assert seen == set(expected) and len(seen) == m['member_count']
    assert sum(e['bytes'] for e in expected.values()) == m['expanded_file_bytes']
    return {'status': 'PASS', 'member_count': len(seen), 'expanded_file_bytes': m['expanded_file_bytes'],
            'archive_sha256': m['archive_sha256'], 'single_XZ_stream_EOF': True,
            'exact_member_set_sizes_modes_and_SHA256': True, 'native_executed': False, 'Torch_executed': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.archive, a.manifest)))


if __name__ == '__main__':
    main()
