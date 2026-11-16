"""Split immutable archive transport without changing any original evidence byte."""
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess

REPO = Path('/dev/shm/heptatorch-d26699e25fea/HeptaTorch-mixed-integer-autograd-integration-10')
REVIEW = Path('/workspace/scratch/d26699e25fea/mixed-integer-autograd-integration-review')
ROOT = REPO / 'validation/mixed-integer-autograd-root'
ARCHIVE_SHA = '0ee1ad222782b03e3f065153c08ba4e14cfec8e02b41fe1cefd5351ead26e5bf'
ARCHIVE_GIT = '0a63e420a48235ad8139afb2f10219e4c7ecf2a2'
ARCHIVE_BYTES = 15368228
sha = lambda b: hashlib.sha256(b).hexdigest()


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


archive = ROOT / 'raw-evidence.tar.xz'
assert archive.stat().st_size == ARCHIVE_BYTES and digest(archive) == ARCHIVE_SHA
assert stat.S_IMODE(archive.stat().st_mode) == 0o644
transport = ROOT / 'transport'
transport.mkdir(exist_ok=True)
history = transport / 'history'
history.mkdir(exist_ok=True)
old_reader_path = history / 'verify_evidence-before-parts.py'
old_reader = old_reader_path.read_bytes() if old_reader_path.exists() else (ROOT / 'verify_evidence.py').read_bytes()
if not old_reader_path.exists():
    old_reader_path.write_bytes(old_reader)
shutil.copy2(REVIEW / 'blob-size-rejection-original.json', history / 'blob-size-rejection-original.json')
parts, offset = [], 0
with archive.open('rb') as stream:
    for i, n in enumerate((8388608, 6979620), 1):
        raw = stream.read(n)
        assert len(raw) == n
        name = f'raw-evidence.tar.xz.part{i:02d}'
        path = transport / name
        if path.exists():
            assert path.read_bytes() == raw
        else:
            with path.open('xb') as f:
                f.write(raw)
        path.chmod(0o644)
        assert digest(path) == sha(raw)
        parts.append(dict(path='transport/' + name, offset=offset, bytes=n,
                          sha256=sha(raw), mode=0o644))
        offset += n
    assert stream.read(1) == b'' and offset == ARCHIVE_BYTES
index = dict(schema='heptatorch.immutable-archive-transport-parts.v1',
    original_archive=dict(path='raw-evidence.tar.xz', bytes=ARCHIVE_BYTES,
                          sha256=ARCHIVE_SHA, git_blob_sha1=ARCHIVE_GIT, mode=0o644),
    parts=parts, concatenation_order='Listed order; exact opaque byte concatenation.',
    reason='GitHub connector rejected the base64 request because its body exceeds 16 MiB.',
    original_archive_changed=False, original_manifest_changed=False,
    original_reports_changed=False, old_native_or_Torch_results_reclassified=False,
    failed_attempts=[dict(attempt=1, result='wrapper reported remote blob mismatch undefined',
                         exact_original_tool_response_available=False),
                     dict(attempt=2, result='HTTP 400: MCP request body is invalid or exceeds the 16 MiB limit',
                          exact_original_tool_response='transport/history/blob-size-rejection-original.json')],
    first_import_mapping='handoff-path-map.json', original_handoff_manifest='handoff-manifest-original.json')
index_raw = (json.dumps(index, indent=2, sort_keys=True) + '\n').encode()
(transport / 'index.json').write_bytes(index_raw)

helper = '''
class ConcatenatedArchive:
    """Sequential binary reader; never extracts or rewrites the original XZ."""
    def __init__(self, root, parts):
        self.paths = [root / p['path'] for p in parts]
        self.index = 0
        self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.stream is not None:
            self.stream.close()

    def read(self, n=-1):
        assert n >= 0, 'Archive readers must use bounded reads'
        blocks, remaining = [], n
        while remaining and self.index < len(self.paths):
            if self.stream is None:
                self.stream = self.paths[self.index].open('rb')
            data = self.stream.read(remaining)
            blocks.append(data)
            remaining -= len(data)
            if not data:
                self.stream.close()
                self.stream = None
                self.index += 1
        return b''.join(blocks)


def open_archive(root, parts):
    if parts is None:
        return (root / 'raw-evidence.tar.xz').open('rb')
    return ConcatenatedArchive(root, parts)


def verify_archive_input(root, parts_only):
    whole = root / 'raw-evidence.tar.xz'
    if whole.exists() and not parts_only:
        assert whole.is_file() and not whole.is_symlink()
        assert whole.stat().st_size == 15368228
        assert stat.S_IMODE(whole.stat().st_mode) == 0o644
        assert digest(whole) == ARCHIVE
        return None, 'original-single-file'
    index_path = root / 'transport/index.json'
    assert index_path.is_file() and not index_path.is_symlink()
    assert digest(index_path) == TRANSPORT_INDEX_SHA
    index = json.loads(index_path.read_bytes())
    assert index['original_archive'] == {
        'path': 'raw-evidence.tar.xz', 'bytes': 15368228, 'sha256': ARCHIVE,
        'git_blob_sha1': '0a63e420a48235ad8139afb2f10219e4c7ecf2a2', 'mode': 0o644}
    parts = index['parts']
    assert len(parts) == 2
    offset = 0
    for p in parts:
        name = PurePosixPath(p['path'])
        assert not name.is_absolute() and '..' not in name.parts and str(name) == p['path']
        assert name.parent == PurePosixPath('transport')
        q = root / name
        assert q.is_file() and not q.is_symlink()
        assert p['offset'] == offset and p['bytes'] > 0
        assert q.stat().st_size == p['bytes']
        assert stat.S_IMODE(q.stat().st_mode) == p['mode'] == 0o644
        assert digest(q) == p['sha256']
        offset += p['bytes']
    assert offset == 15368228
    h, git_hash, total = hashlib.sha256(), hashlib.sha1(b'blob 15368228\\0'), 0
    with open_archive(root, parts) as stream:
        while block := stream.read(1 << 20):
            h.update(block)
            git_hash.update(block)
            total += len(block)
    assert total == 15368228 and h.hexdigest() == ARCHIVE
    assert git_hash.hexdigest() == index['original_archive']['git_blob_sha1']
    return parts, 'two-exact-transport-parts'

'''
reader = old_reader.decode()
anchor = '\n\ndef main():\n'
assert reader.count(anchor) == 1
reader = reader.replace(anchor, '\n\nTRANSPORT_INDEX_SHA = ' + repr(sha(index_raw)) + '\n' + helper + '\ndef main():\n')
edits = [
    ("    parser.add_argument('--output', type=Path)\n", "    parser.add_argument('--output', type=Path)\n    parser.add_argument('--parts-only', action='store_true', help='Verify transport parts even if the original archive is present')\n"),
    ("    for name, expected in [('report.json', REPORT), ('evidence-manifest.json', MANIFEST),\n                           ('raw-evidence.tar.xz', ARCHIVE)]:\n", "    for name, expected in [('report.json', REPORT), ('evidence-manifest.json', MANIFEST)]:\n"),
    ("    report = json.loads((root / 'report.json').read_bytes())\n", "    parts, archive_input = verify_archive_input(root, args.parts_only)\n    report = json.loads((root / 'report.json').read_bytes())\n"),
    ("    with tarfile.open(root / 'raw-evidence.tar.xz', mode='r|xz') as archive:\n", "    with open_archive(root, parts) as raw_stream, tarfile.open(fileobj=raw_stream, mode='r|xz') as archive:\n"),
    ("    with (root / 'raw-evidence.tar.xz').open('rb') as stream:\n", "    with open_archive(root, parts) as stream:\n"),
    ("              'members': len(seen), 'file_bytes': 259777728, 'single_XZ_EOF': True,\n", "              'archive_input': archive_input, 'parts_only_requested': args.parts_only,\n              'transport_index_sha256': TRANSPORT_INDEX_SHA if parts is not None else None,\n              'members': len(seen), 'file_bytes': 259777728, 'single_XZ_EOF': True,\n")]
for before, after in edits:
    assert reader.count(before) == 1
    reader = reader.replace(before, after)
compile(reader, str(ROOT / 'verify_evidence.py'), 'exec')
inverse = reader.replace('\n\nTRANSPORT_INDEX_SHA = ' + repr(sha(index_raw)) + '\n' + helper + '\ndef main():\n', anchor)
for before, after in reversed(edits):
    assert inverse.count(after) == 1
    inverse = inverse.replace(after, before)
assert inverse.encode() == old_reader
assert (ROOT / 'verify_evidence.py').stat().st_nlink == 1
assert (ROOT / 'verify_evidence.py').read_bytes() in (old_reader, reader.encode())
(ROOT / 'verify_evidence.py').write_text(reader)

# Remove only the newly added oversized blob from the index. Keep every original
# whole archive path on disk and in the immutable checkpoint unchanged.
relative = 'validation/mixed-integer-autograd-root/raw-evidence.tar.xz'
assert subprocess.run(['git','cat-file','-e','HEAD:'+relative], cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0
subprocess.run(['git','rm','--cached','--',relative], cwd=REPO, check=True)
exclude = REPO / '.git/info/exclude'
existing = exclude.read_bytes() if exclude.exists() else b''
line = ('/' + relative + '\n').encode()
assert line not in existing
exclude.write_bytes(existing + (b'\n' if existing and not existing.endswith(b'\n') else b'') + line)
assert digest(archive) == ARCHIVE_SHA
receipt = dict(schema='heptatorch.root10.git-transport-adaptation.v1', status='PREPARED_REQUIRES_PARTS_ONLY_BYTE_REVIEW',
    original_archive_sha256=ARCHIVE_SHA, original_git_blob_sha1=ARCHIVE_GIT,
    original_reader_sha256=sha(old_reader), new_reader_sha256=sha(reader.encode()),
    transport_index_sha256=sha(index_raw), parts=parts, documented_reader_replacements=len(edits),
    inverse_entire_original_reader_byte_identical=True, original_validation_predicates_preserved=True,
    original_archive_still_on_disk=True, only_new_index_entry_removed=relative,
    original_reports_manifests_unchanged=True, native_or_Torch_executed=False)
(transport / 'adaptation.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
(ROOT / 'TRANSPORT.md').write_text('''# Exact transport of the original cold7 archive

The original 15,368,228-byte XZ exceeds the connector's 16 MiB request limit
after base64 encoding. Its two opaque transport parts concatenate, in the order
in `transport/index.json`, to exactly the original archive. The archive SHA-256
is `0ee1ad222782b03e3f065153c08ba4e14cfec8e02b41fe1cefd5351ead26e5bf`;
its original Git blob SHA-1 is `0a63e420a48235ad8139afb2f10219e4c7ecf2a2`.
No report, tar member, compression byte, or original manifest was changed.

The original whole archive remains in the original execution checkpoint. The
Git checkout distributes the two parts. `handoff-manifest-original.json` and
`handoff-path-map.json` retain their original import history; the transport
index records the later distribution representation separately.

Verify the entire archive, every one of its 2,225 member sizes, modes and hashes,
and the complete single XZ stream directly from the parts, without extraction:

```sh
python3 -B validation/mixed-integer-autograd-root/verify_evidence.py --parts-only --output NEW_REPORT.json
```

The verifier also accepts the original whole archive when present. The
`--parts-only` flag deliberately prevents that fallback. Its original verifier
is retained under `transport/history/`; every original evidence predicate is
preserved. The actual byte review is separate from the seven native commands
already recorded in the immutable report. There were 474 inputs captured before
execution and 648 total bound files at completion. Neither count denotes APIs.

To reconstruct the original archive, concatenate `raw-evidence.tar.xz.part01`
and `raw-evidence.tar.xz.part02` in `transport/`, verify the whole SHA-256 above,
and set its mode to 0644. Its size must be exactly 15,368,228 bytes. Original
native/Torch results and the whole-project HOLD decision remain unchanged.
''')
print(json.dumps(dict(status=receipt['status'], index_sha256=sha(index_raw),
    reader_sha256=receipt['new_reader_sha256'], original_unchanged=True, parts=parts)))
