"""Verify original cold7 bytes without running a native or Torch process."""
import argparse
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath
import stat
import tarfile

REPORT = '77ae37fddbce0ad15628c0bf265c9e41649f9614714deaec99b91c15b7c3fbb5'
MANIFEST = '71d5cac1363e07c34159992c0b70e0355675cc3b129c387dfe36179f217b0568'
ARCHIVE = '0ee1ad222782b03e3f065153c08ba4e14cfec8e02b41fe1cefd5351ead26e5bf'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    for name, expected in [('report.json', REPORT), ('evidence-manifest.json', MANIFEST),
                           ('raw-evidence.tar.xz', ARCHIVE)]:
        assert (root / name).is_file() and not (root / name).is_symlink()
        assert digest(root / name) == expected, name
    report = json.loads((root / 'report.json').read_bytes())
    manifest = json.loads((root / 'evidence-manifest.json').read_bytes())
    assert report['status'] == 'PASS' and report['command_count'] == 7
    assert report['producer_executed_this_invocation'] and report['producer_completed_this_invocation']
    assert report['producer_reused'] is False and report['Torch_executed_here'] is False
    assert report['inputs_captured_before_execution'] == 474 and len(report['checked_inputs']) == 648
    assert report['whole_pytorch_gate'] == 'HOLD' and report['closed_schema_count'] == 0
    assert [r['name'] for r in report['commands']] == [
        'producer5', 'source-custody', 'owner4-copied17', 'resources6', 'owned-entry5', 'Saved-entry6', 'typed-handoff2']
    for row in report['commands']:
        assert row['process_started'] and row['process_completed'] and row['exit_code'] == 0
        assert row['status'] == ('SOURCE_AND_UNIT_CUSTODY_PASS' if row['name'] == 'source-custody' else 'PASS')
    files = manifest['files']
    assert len(files) == manifest['members'] == 2225
    assert manifest['uncompressed_bytes'] == sum(row['bytes'] for row in files.values()) == 259777728
    assert manifest['archive_sha256'] == ARCHIVE and manifest['archive_bytes'] == 15368228
    seen = set()
    with tarfile.open(root / 'raw-evidence.tar.xz', mode='r|xz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            assert not name.is_absolute() and '..' not in name.parts and str(name) == member.name
            assert member.isfile() and member.name in files and member.name not in seen
            expected = files[member.name]
            assert member.size == expected['bytes'] and stat.S_IMODE(member.mode) == expected['mode']
            with archive.extractfile(member) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected['sha256']
            seen.add(member.name)
    assert seen == set(files)
    decoder = lzma.LZMADecompressor()
    tar_hash, length = hashlib.sha256(), 0
    with (root / 'raw-evidence.tar.xz').open('rb') as stream:
        while not decoder.eof:
            block = stream.read(65536) if decoder.needs_input else b''
            assert block or not decoder.needs_input
            raw = decoder.decompress(block, max_length=1 << 20)
            length += len(raw)
            tar_hash.update(raw)
        assert not decoder.unused_data and not stream.read(1)
    assert length == manifest['decompressed_tar_bytes'] == 261509120
    assert tar_hash.hexdigest() == manifest['decompressed_tar_sha256'] == '833a37a75ddaae51040a2560e70104c5ff2f6273dd9e55ef938408c50604ec23'
    result = {'schema': 'heptatorch.mixed10.root-byte-custody.v1', 'status': 'PASS_BYTE_CUSTODY_ONLY',
              'report_sha256': REPORT, 'manifest_sha256': MANIFEST, 'archive_sha256': ARCHIVE,
              'members': len(seen), 'file_bytes': 259777728, 'single_XZ_EOF': True,
              'all_sizes_modes_hashes_verified': True, 'recorded_outer_commands': 7,
              'inputs_captured_before_execution': 474, 'total_bound_files_at_completion': 648,
              'new_native_execution': False, 'new_Torch_execution': False,
              'nested_archives': 'Original nested archive bodies are checked as exact outer members; their independent full verification remains recorded by the actual coordinator.',
              'whole_project_decision': 'HOLD', 'closed_full_schemas': 0}
    if args.output is not None:
        assert not args.output.exists()
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
