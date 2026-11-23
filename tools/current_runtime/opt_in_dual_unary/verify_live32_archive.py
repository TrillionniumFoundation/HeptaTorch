#!/usr/bin/env python3
"""Verify every path, byte, regular-file mode and directory mode in an XZ custody container."""
from pathlib import Path, PurePosixPath
import hashlib,json,stat,sys,tarfile
root,dest=map(Path,sys.argv[1:])
seen=set();sha=hashlib.sha256()
bytes_files=files=dirs=0
with tarfile.open(dest,'r|xz') as tf:
    for t in tf:
        p=PurePosixPath(t.name)
        assert not p.is_absolute() and '..' not in p.parts and p not in seen
        seen.add(p)
        src=root/p
        s=src.lstat()
        assert stat.S_IMODE(s.st_mode)==t.mode
        if t.isdir():
            assert stat.S_ISDIR(s.st_mode)
            dirs+=1
            continue
        assert t.isfile() and stat.S_ISREG(s.st_mode) and s.st_size==t.size
        with src.open('rb') as f:
            original=hashlib.file_digest(f,'sha256').hexdigest()
        assert original==hashlib.file_digest(tf.extractfile(t),'sha256').hexdigest()
        sha.update(json.dumps((p.as_posix(),original,t.size,t.mode),separators=(',',':')).encode()+b'\n')
        bytes_files+=t.size
        files+=1
expected={PurePosixPath('.')} | {PurePosixPath(str(p.relative_to(root))) for p in root.rglob('*')}
assert seen==expected
with dest.open('rb') as f:
 digest=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(status='PASS_ARCHIVE_BYTE_AND_MODE_ROUNDTRIP',files=files,directories=dirs,
   raw_payload_bytes=bytes_files,compressed_bytes=dest.stat().st_size,archive_sha256=digest,
   manifest_sha256_in_tar_order=sha.hexdigest()),sort_keys=True))
