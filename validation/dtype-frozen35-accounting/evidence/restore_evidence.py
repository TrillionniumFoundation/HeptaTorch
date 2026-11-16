#!/usr/bin/env python3
"""Reassemble and verify frozen evidence bytes; never execute a test or image."""
import argparse
import hashlib
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import traceback

INDEX_SHA256 = '0a772282288e7a1985ba76f7d05b46c56a892b5799a9b4f794bced035e014b65'
BLOCK = 1024 * 1024


def need(condition, message):
    if not condition:
        raise ValueError(message)


def safe_relative(name):
    need(isinstance(name, str) and bool(name) and '\\' not in name, 'relative path')
    p = PurePosixPath(name)
    need(not p.is_absolute() and '..' not in p.parts and '.' not in p.parts,
         'path escape')
    need(p.as_posix() == name and name != '.', 'noncanonical path')
    return p


def input_path(root, name):
    relative = safe_relative(name)
    path = root
    for part in relative.parts:
        path = path / part
        need(not path.is_symlink(), 'input symlink: ' + name)
    need(path.is_file(), 'missing input: ' + name)
    return path


def metadata(path):
    st = path.lstat()
    need(stat.S_ISREG(st.st_mode), 'not regular: ' + str(path))
    digest = hashlib.sha256()
    count = 0
    with path.open('rb') as source:
        for raw in iter(lambda: source.read(BLOCK), b''):
            count += len(raw)
            digest.update(raw)
    need(count == st.st_size, 'input changed length: ' + str(path))
    return {'bytes': count, 'sha256': digest.hexdigest(),
            'mode': stat.S_IMODE(st.st_mode)}


def same_bytes(actual, expected, label):
    need(actual['bytes'] == expected['bytes'] and
         actual['sha256'] == expected['sha256'], label)


def verify_xz(path):
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    expanded = 0
    with path.open('rb') as source:
        while True:
            raw = source.read(BLOCK)
            if not raw:
                break
            need(not decoder.eof, 'trailing XZ bytes')
            expanded += len(decoder.decompress(raw, max_length=BLOCK))
            while not decoder.needs_input and not decoder.eof:
                expanded += len(decoder.decompress(b'', max_length=BLOCK))
            if decoder.eof:
                need(not decoder.unused_data and source.read(1) == b'',
                     'concatenated or trailing XZ bytes')
                break
    need(decoder.eof, 'incomplete XZ stream')
    return expanded


def verify_members(path, manifest, expected, extract_to):
    files = manifest['files']
    need(isinstance(files, dict) and len(files) == expected['member_count'] > 0,
         'member inventory count')
    need(manifest['member_count'] == expected['member_count'] and
         manifest['expanded_bytes'] == expected['expanded_bytes'], 'manifest totals')
    for name, meta in files.items():
        safe_relative(name)
        need(type(meta['bytes']) is int and meta['bytes'] >= 0 and
             type(meta['mode']) is int and 0 <= meta['mode'] <= 0o777,
             'invalid member metadata')
    seen = set()
    total = 0
    if extract_to is not None:
        extract_to.mkdir(exist_ok=False)
    with tarfile.open(path, 'r:xz') as archive:
        for member in archive:
            name = member.name
            relative = safe_relative(name)
            need(member.isfile() and name in files and name not in seen,
                 'unexpected, linked, or duplicate member: ' + name)
            meta = files[name]
            need(member.size == meta['bytes'] and member.mode == meta['mode'],
                 'member size or mode: ' + name)
            destination = None
            if extract_to is not None:
                destination = extract_to.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            output = destination.open('xb') if destination is not None else None
            try:
                with archive.extractfile(member) as source:
                    for raw in iter(lambda: source.read(BLOCK), b''):
                        count += len(raw)
                        need(count <= meta['bytes'], 'member overflow: ' + name)
                        digest.update(raw)
                        if output is not None:
                            output.write(raw)
            finally:
                if output is not None:
                    output.close()
            need(count == meta['bytes'] and digest.hexdigest() == meta['sha256'],
                 'member bytes: ' + name)
            if destination is not None:
                destination.chmod(meta['mode'])
                need(metadata(destination) == {k: meta[k] for k in ('bytes', 'sha256', 'mode')},
                     'extracted bytes or mode: ' + name)
            seen.add(name)
            total += count
    need(seen == set(files) and total == expected['expanded_bytes'],
         'incomplete archive inventory')
    return {'member_count': len(seen), 'expanded_bytes': total,
            'every_member_size_SHA_mode_verified': True,
            'extracted_and_rechecked': extract_to is not None}


def restore_one(package, output, spec, extract, inputs):
    name = spec['name']
    safe_relative(name)
    need('/' not in name, 'archive name')
    folder = output / name
    folder.mkdir(exist_ok=False)
    manifest_path = input_path(package, spec['manifest']['file'])
    manifest_meta = metadata(manifest_path)
    same_bytes(manifest_meta, spec['manifest'], 'manifest bytes')
    inputs[spec['manifest']['file']] = manifest_meta
    manifest = json.loads(manifest_path.read_bytes())
    same_bytes(manifest['archive'], spec['archive'], 'manifest archive identity')
    restored = folder / safe_relative(spec['restored_filename'])
    need(restored.parent == folder, 'restored archive filename')
    count = 0
    digest = hashlib.sha256()
    used_parts = set()
    need(isinstance(spec['parts'], list) and bool(spec['parts']), 'empty parts')
    with restored.open('xb') as target:
        for part in spec['parts']:
            need(part['offset'] == count and part['bytes'] > 0 and
                 part['file'] not in used_parts, 'part order, length, or duplication')
            p = input_path(package, part['file'])
            before = metadata(p)
            same_bytes(before, part, 'part bytes: ' + part['file'])
            with p.open('rb') as source:
                for raw in iter(lambda: source.read(BLOCK), b''):
                    target.write(raw)
                    digest.update(raw)
                    count += len(raw)
            need(metadata(p) == before, 'part changed during read')
            inputs[part['file']] = before
            used_parts.add(part['file'])
    actual = {'bytes': count, 'sha256': digest.hexdigest()}
    same_bytes(actual, spec['archive'], 'restored complete archive')
    need(type(spec['archive']['mode']) is int and
         0 <= spec['archive']['mode'] <= 0o777, 'archive mode')
    restored.chmod(spec['archive']['mode'])
    tar_bytes = verify_xz(restored)
    members = verify_members(restored, manifest, spec,
                             folder / 'files' if extract else None)
    need(metadata(manifest_path) == manifest_meta, 'manifest changed during read')
    return {'name': name, 'restored_archive': str(restored),
            'archive': metadata(restored), 'member_manifest': spec['manifest'],
            'XZ_EOF_verified': True, 'single_XZ_stream': True,
            'decompressed_tar_bytes': tar_bytes, **members}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extract', action='store_true',
                        help='also extract and recheck every regular member in new directories')
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    output = args.output.resolve()
    need(not output.exists(), 'output must be a new directory')
    need(output != package and package not in output.parents,
         'output cannot modify the frozen package')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'heptatorch.dtype523-evidence-restoration.v1',
              'status': 'RUNNING', 'native_executed': False, 'Torch_executed': False,
              'archives': [], 'original_child_bytes_rewritten': False,
              'whole_pytorch_gate': 'HOLD', 'whole_pytorch_rewrite_complete': False}
    exit_code = 0
    try:
        index_path = input_path(package, 'evidence-index.json')
        index_before = metadata(index_path)
        need(index_before['sha256'] == INDEX_SHA256, 'pinned evidence index')
        index = json.loads(index_path.read_bytes())
        need(index['schema'] == 'heptatorch.dtype523-frozen35-recoverable-evidence.v1',
             'index schema')
        need([x['name'] for x in index['archives']] == ['remote-raw', 'local-supplement'],
             'exact archive groups')
        inputs = {'evidence-index.json': index_before}
        for spec in index['archives']:
            report['archives'].append(restore_one(package, output, spec, args.extract, inputs))
        need(metadata(index_path) == index_before, 'index changed during read')
        for name, meta in inputs.items():
            need(metadata(input_path(package, name)) == meta, 'final input custody: ' + name)
        report.update(status='PASS_EVIDENCE_RESTORATION', input_custody=inputs,
                      input_files_unchanged=True, source_package_manifest_sha256=index['source_package_manifest_sha256'],
                      native_source_sha256=index['native_source_sha256'],
                      native_executable_sha256=index['native_executable_sha256'])
    except Exception as exc:
        report.update(status='FAIL_EVIDENCE_RESTORATION', error=repr(exc),
                      traceback=traceback.format_exc(), partial_output_preserved=True)
        exit_code = 1
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'report': str(output / 'report.json'),
                      'archive_groups_completed': len(report['archives'])}))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
