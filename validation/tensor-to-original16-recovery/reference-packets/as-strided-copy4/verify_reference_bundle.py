#!/usr/bin/env python3
"""Verify an immutable reference packet; no native or Torch execution."""
import argparse
import hashlib
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--packet', type=Path, required=True)
    ap.add_argument('--extract', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    assert not a.output.exists()
    mraw = (a.packet / 'manifest.json').read_bytes()
    m = json.loads(mraw)
    assert m['schema'] == 'heptatorch.actual-reference-complete-packet.v1'
    assert sha(Path(__file__).read_bytes()) == m['verifier_sha256']
    z = (a.packet / m['archive']['name']).read_bytes()
    assert len(z) == m['archive']['bytes'] and sha(z) == m['archive']['sha256']
    d = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    plain = d.decompress(z)
    assert d.eof and d.unused_data == b''
    assert len(plain) == m['tar']['bytes'] and sha(plain) == m['tar']['sha256']
    expected = m['files']
    assert len(expected) == m['file_count'] and expected
    if a.extract:
        a.extract.mkdir(parents=True, exist_ok=False)
    seen = set()
    expanded = 0
    with tarfile.open(fileobj=io.BytesIO(plain), mode='r:') as t:
        for member in t:
            p = PurePosixPath(member.name)
            assert member.isfile() and not p.is_absolute() and '..' not in p.parts
            assert member.name == p.as_posix() and member.name not in seen
            seen.add(member.name)
            assert member.name in expected
            row = expected[member.name]
            data = t.extractfile(member).read()
            assert len(data) == member.size == row['bytes']
            assert member.mode == row['mode'] and member.mode in (0o600, 0o644, 0o700, 0o755)
            assert sha(data) == row['sha256']
            expanded += len(data)
            if a.extract:
                path = a.extract.joinpath(*p.parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as out:
                    out.write(data)
                os.chmod(path, member.mode)
                assert path.read_bytes() == data and path.stat().st_mode & 0o777 == member.mode
    assert seen == set(expected) and expanded == m['expanded_bytes']
    report = {'schema': 'heptatorch.actual-reference-packet-byte-verification.v1',
              'status': 'PASS', 'manifest_sha256': sha(mraw),
              'archive_sha256': sha(z), 'tar_sha256': sha(plain),
              'file_count': len(seen), 'expanded_bytes': expanded,
              'all_member_bytes_sizes_modes_verified': True,
              'single_xz_stream_verified_through_EOF': True,
              'original_reference_status': 'OBSERVED',
              'native_executed': False, 'torch_executed_this_verifier': False,
              'extracted': a.extract is not None}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
