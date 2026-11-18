"""Restore fixed reference/source fixtures without executing their programs."""
import ast
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile

BASELINE_ARCHIVE_SHA = '7656fa9e918ed4ef73d29009da3cf6fb0840fe0b4d0775f3a37b7871ef25bde8'
BASELINE_MANIFEST_SHA = 'e4ac41437792606e36524d85e1f2049c106527ffcbf8055cebf068c4f83547ca'
F32_REFERENCE_ARCHIVE_SHA = '1494c31492261a84b3311d568109992c48c5062b52802b2909da5f82f9f5604d'
F32_REFERENCE_MANIFEST_SHA = '5f08c46df30f4f747968270e332687861add4449d8771cebb3d26018a2b13903'
F32_READER_MANIFEST_SHA = 'a97c262047f17c50a9ad77815754644e1df19350ec2511f274997b4711dda974'
LEGACY_PROOF_SHA = '93b3ce1d1fe1620537bde437197797cd585eb5b17282555f965a2d8d77ba61ae'
CURRENT35_ARCHIVE_SHA = '188655c40e46bcdc9ec2f12385c293941f3a7556a8ab30761619e5752e67362f'
CURRENT35_MANIFEST_SHA = '0241d2f320765f1ad53aa28721a4e30304aedc21d7cca8a44668ee5567510f57'
CURRENT270_ARCHIVE_SHA = '47afac432997a33aa5b874c2c088fa1b238262903f7d2c230abc2be34a2ad40a'
CURRENT270_MANIFEST_SHA = '45e251c6cd275797c5bf121a7e190c5e89232e2f6592ae95bed37369a313ecef'
SOURCE700 = 'b83a8346b162e12c5ed77e0287d626619a18f84bf7a1ab02a2514d30ae4295b4'


def require(condition, rule):
    if not condition:
        raise ValueError(rule)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, expected=None, inputs=None):
    path = Path(path).absolute()
    require(not path.is_symlink(), 'fixture input is a symlink: ' + str(path))
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), 'fixture input is not a regular file')
    raw = inputs.bind(path, expected) if inputs is not None else path.read_bytes()
    after = path.stat()
    fields = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(fields(before) == fields(after), 'fixture input changed during read')
    require(expected is None or sha(raw) == expected, 'fixture SHA mismatch: ' + str(path))
    return raw


def safe_name(name):
    relative = PurePosixPath(name)
    require(relative.parts and not relative.is_absolute() and '..' not in relative.parts
            and '\\' not in name and str(relative) == name, 'unsafe fixture member name')
    return relative


def restore(repository, destination, relative, archive_sha, manifest_sha, schema,
            inputs=None, embedded_manifest=False, archive_prefix='', directory_modes=None):
    repository, destination = Path(repository), Path(destination)
    archive_path = repository / relative / 'complete.tar.xz'
    raw = read(archive_path, archive_sha, inputs)
    manifest_raw = read(archive_path.with_name('manifest.json'), manifest_sha, inputs)
    manifest = json.loads(manifest_raw)
    require(manifest['schema'] == schema, 'exact fixture manifest schema')
    pins = dict(manifest['files'])
    count = manifest.get('file_count', manifest.get('files_count', manifest.get('payload_files')))
    require(count == len(pins) and sum(p['bytes'] for p in pins.values()) == manifest['payload_bytes'],
            'complete fixed fixture denominator')
    if 'archive' in manifest:
        declared = manifest['archive']
        require(declared['path'] == 'complete.tar.xz' and declared['sha256'] == sha(raw)
                and declared['bytes'] == len(raw), 'exact fixture archive identity')
    if embedded_manifest:
        require('package-manifest.json' not in pins, 'distinct embedded manifest')
        pins['package-manifest.json'] = dict(bytes=len(manifest_raw), sha256=sha(manifest_raw), mode=0o644)
    destination.mkdir(parents=True, exist_ok=False)
    seen, seen_directories = set(), set()
    directory_modes = directory_modes or {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:xz') as archive:
        for member in archive:
            safe_name(member.name)
            if member.isdir():
                require(member.name in directory_modes and member.name not in seen_directories
                        and member.mode == directory_modes[member.name], 'only declared original directories')
                seen_directories.add(member.name)
                continue
            require(not archive_prefix or member.name.startswith(archive_prefix + '/'),
                    'exact fixture archive prefix')
            name = member.name[len(archive_prefix) + 1:] if archive_prefix else member.name
            relative_name = safe_name(name)
            require(member.isfile() and name not in seen and name in pins,
                    'only unique, explicitly declared regular fixture files')
            seen.add(name)
            payload = archive.extractfile(member).read()
            pin = pins[name]
            require(len(payload) == member.size == pin['bytes'] and sha(payload) == pin['sha256']
                    and member.mode == pin['mode'], 'original fixture bytes/mode changed: ' + member.name)
            path = destination.joinpath(*relative_name.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(payload)
            path.chmod(member.mode)
            require(stat.S_IMODE(path.stat().st_mode) == pin['mode'], 'restored fixture mode changed')
            read(path, pin['sha256'], inputs)
    require(seen == set(pins) and seen_directories == set(directory_modes),
            'all original fixture files and directories must be restored')
    for name, mode in directory_modes.items():
        relative_name = name[len(archive_prefix) + 1:] if name != archive_prefix else ''
        directory = destination / relative_name
        require(directory.is_dir() and not directory.is_symlink(), 'restored original directory')
        directory.chmod(mode)
        require(stat.S_IMODE(directory.stat().st_mode) == mode, 'original directory mode restored')
    return destination


def restore_baseline(repository, destination, inputs=None):
    return restore(repository, destination, 'validation/current-runtime-baseline',
                   BASELINE_ARCHIVE_SHA, BASELINE_MANIFEST_SHA,
                   'heptatorch.current-runtime.frozen700-sources.v1', inputs)


def restore_f32_reference(repository, destination, inputs=None):
    return restore(repository, destination, 'validation/sum-f32-reference17',
                   F32_REFERENCE_ARCHIVE_SHA, F32_REFERENCE_MANIFEST_SHA,
                   'heptatorch.current-runtime.fixed-f32-reference17.v1', inputs)


def restore_current35(repository, destination, inputs=None):
    return restore(repository, destination, 'validation/current-runtime35-source',
                   CURRENT35_ARCHIVE_SHA, CURRENT35_MANIFEST_SHA,
                   'heptatorch.current701-cat-empty-original35.prepared-package.v1',
                   inputs, embedded_manifest=True)


def restore_current270(repository, destination, inputs=None):
    return restore(repository, destination, 'validation/current-runtime270-source',
                   CURRENT270_ARCHIVE_SHA, CURRENT270_MANIFEST_SHA,
                   'heptatorch.current270.source-package.v1', inputs,
                   embedded_manifest=True, archive_prefix='current270',
                   directory_modes={'current270': 0o755, 'current270/frozen': 0o755})


def prepare_f32_reader(repository, destination, baseline, inputs=None):
    repository, destination = Path(repository), Path(destination)
    folder = repository / 'tests/sum_f32_review'
    manifest_raw = read(folder / 'source-manifest.json', F32_READER_MANIFEST_SHA, inputs)
    manifest = json.loads(manifest_raw)
    require(manifest['schema'] == 'heptatorch.current-runtime.fixed-f32-reader17.v1'
            and manifest['file_count'] == len(manifest['files']) == 12,
            'complete fixed F32 source package')
    destination.mkdir(parents=True, exist_ok=False)
    total = 0
    for name, pin in manifest['files'].items():
        safe_name(name)
        path = folder / name
        raw = read(path, pin['sha256'], inputs)
        require(len(raw) == pin['bytes'] and stat.S_IMODE(path.stat().st_mode) == pin['mode'],
                'fixed reader source size/mode')
        target = destination / name
        with target.open('xb') as stream:
            stream.write(raw)
        target.chmod(pin['mode'])
        read(target, pin['sha256'], inputs)
        total += len(raw)
    require(total == manifest['payload_bytes'], 'complete fixed reader source payload')
    baseline_path = Path(baseline) / 'tools/current_runtime/source/runtime.hepta'
    raw = read(baseline_path, SOURCE700, inputs)
    require(stat.S_IMODE(baseline_path.stat().st_mode) == 0o644, 'fixed baseline source mode')
    (destination / 'baseline').mkdir()
    target = destination / 'baseline/runtime.hepta'
    with target.open('xb') as stream:
        stream.write(raw)
    target.chmod(0o644)
    read(target, SOURCE700, inputs)
    return destination


def verify_legacy_sources(repository, baseline, inputs=None):
    repository, baseline = Path(repository), Path(baseline)
    proof = json.loads(read(repository / 'tests/current_runtime_legacy/identity-proof.json',
                            LEGACY_PROOF_SHA, inputs))
    require(proof['schema'] == 'heptatorch.current-runtime.legacy701-identity.v1'
            and proof['comparison_predicate_or_expected_value_edits'] == []
            and set(proof['files']) == {'tests/current_runtime_legacy/run.py',
                                       'tests/current_runtime_legacy/sum32.py'},
            'complete exact legacy identity proof')
    for relative, row in proof['files'].items():
        original = read(baseline / relative, row['original_sha256'], inputs)
        current = read(repository / relative, row['derived_sha256'], inputs)
        require(len(original) == row['original_bytes'] and len(current) == row['derived_bytes'],
                'complete legacy source sizes')
        inverse = current.decode()
        for edit in reversed(row['edits']):
            require(edit['count'] == 1 and inverse.count(edit['after']) == 1,
                    'one exact legacy identity edit')
            inverse = inverse.replace(edit['after'], edit['before'])
        require(inverse.encode() == original and row['whole_source_inverse_exact'] is True
                and row['comparison_predicate_or_expected_value_edits'] == [],
                'complete original legacy source inverse')
        def bodies(raw):
            return {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse(raw).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        old, new = bodies(original), bodies(current)
        require(old.keys() == new.keys() and sorted(k for k in old if old[k] != new[k])
                == row['identity_wrapper_AST_changes'], 'only declared identity admission bodies changed')
    return proof
