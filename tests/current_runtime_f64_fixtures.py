"""Restore reviewed F64 48 sources and exact original Torch observations.

All102 original producer inputs retain their bytes and modes. The original
terminal and four streams are available; its actual argv and Python executable
entity were not retained. Current native observer identities do not fill that gap.
This module performs byte transport and source/provenance reading only.
"""
import hashlib
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import stat
import tarfile

SOURCE_PACKAGE = 'validation/current-f64-48-source'
REFERENCE_PACKAGE = 'validation/current-f64-48-reference'
REFERENCE = 'b1dafeed270b4dd226c349dece9d7f6213c2dabd34681685d97ec13c068e9b11'
READER = '1deb2fa84d9ea9653ca56769a5527f0102571ba2a4690f8deb85f5d4bdc038b3'
SOURCE = '51820849f7becec549be15babace00f00687601ae7f53bcee0e636ca02f721f9'
IMAGE = 'aa3cc8f0c8738a9505d31c9e8ba5a523b7e08b591b61fc7bfb8592295491618b'
TERMINAL = 'f3ed8ff73e2ac433494fcfe57bf7db27c0c1748e1634689a777a6ae7b31fc02f'
SOURCE_MANIFEST = '29c36cfe0b0b2861419457e69918d7e1382ecce96a7d46a1a7ac1c841fbc2edc'
REFERENCE_MANIFEST = 'e220f593f80844b6ec5959e302561fbf0fb8e6be5826deb5211910351682bf60'
SOURCE_ARCHIVE = '10029f4ee37980f435f562795c5a7e7a51e10762248f1780a460c76c9741135d'
REFERENCE_ARCHIVE = '22971ae1c4b94f633ae99e3decda88704391c644faa1b1bd7d49528f7f04dc3b'
PACKAGES = {'source': ('validation/current-f64-48-source', '29c36cfe0b0b2861419457e69918d7e1382ecce96a7d46a1a7ac1c841fbc2edc', '10029f4ee37980f435f562795c5a7e7a51e10762248f1780a460c76c9741135d', 488232, 110, 7735468, 'heptatorch.current736.f64-48.portable-source-package.v1'), 'reference': ('validation/current-f64-48-reference', 'e220f593f80844b6ec5959e302561fbf0fb8e6be5826deb5211910351682bf60', '22971ae1c4b94f633ae99e3decda88704391c644faa1b1bd7d49528f7f04dc3b', 9900816, 190, 27453209, 'heptatorch.f64-48.fixed-original-reference-package.v1')}


def require(condition, message):
    if not condition:
        raise ValueError(message)



def sha(raw):
    return hashlib.sha256(raw).hexdigest()



def safe_name(name):
    path = PurePosixPath(name)
    require(path.parts and not path.is_absolute() and '..' not in path.parts
            and '\\' not in name and str(path) == name, 'normalized package member path')
    return path



def package(repository, kind, inputs):
    relative, manifest_sha, archive_sha, archive_bytes, count, payload_bytes, schema = PACKAGES[kind]
    root = Path(repository) / relative
    manifest = json.loads(inputs.bind(root / 'manifest.json', manifest_sha))
    require(manifest['schema'] == schema and manifest['file_count'] == len(manifest['files']) == count
            and manifest['payload_bytes'] == payload_bytes
            and sum(row['bytes'] for row in manifest['files'].values()) == payload_bytes
            and manifest['archive'] == dict(path='complete.tar.xz',bytes=archive_bytes,sha256=archive_sha),
            'complete fixed F64 package identity and denominator')
    for name in manifest['files']:
        safe_name(name)
    archive = inputs.bind(root / 'complete.tar.xz', archive_sha)
    require(len(archive) == archive_bytes, 'complete original F64 archive bytes')
    return manifest, archive



def bind_directory(repository, directory, kind, inputs):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), 'ordinary restored F64 directory')
    manifest, _ = package(repository, kind, inputs)
    paths = list(directory.rglob('*'))
    require(all(not p.is_symlink() and (p.is_dir() or p.is_file()) for p in paths),
            'only ordinary restored F64 files and directories')
    actual = {p.relative_to(directory).as_posix() for p in paths if p.is_file()}
    require(actual == set(manifest['files']), 'exact restored F64 file census')
    for name, row in manifest['files'].items():
        path = directory.joinpath(*safe_name(name).parts)
        raw = inputs.bind(path, row['sha256'])
        require(len(raw) == row['bytes'] and stat.S_IMODE(path.stat().st_mode) == row['mode'],
                'restored F64 file extent or mode changed: ' + name)
    return manifest



def restore(repository, directory, kind, inputs):
    directory = Path(directory)
    require(not directory.exists() and not directory.is_symlink(), 'fresh F64 package destination')
    manifest, archive = package(repository, kind, inputs)
    decoder = lzma.LZMADecompressor()
    raw_tar = decoder.decompress(archive,
        max_length=manifest['payload_bytes'] + (manifest['file_count'] + 2) * 10240)
    require(decoder.eof and not decoder.unused_data, 'complete single XZ stream with no trailing data')
    directory.mkdir(parents=True,exist_ok=False)
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(raw_tar),mode='r:') as tar:
        for member in tar:
            name = safe_name(member.name)
            require(member.isfile() and member.name not in seen and member.name in manifest['files'],
                    'ordinary, unique, expected F64 archive member')
            row = manifest['files'][member.name]
            raw = tar.extractfile(member).read()
            require(member.size == len(raw) == row['bytes'] and member.mode == row['mode']
                    and sha(raw) == row['sha256'], 'complete original F64 member bytes and mode')
            path = directory.joinpath(*name.parts)
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as output:
                output.write(raw)
            path.chmod(member.mode)
            seen.add(member.name)
    require(seen == set(manifest['files']), 'no omitted F64 package members')
    return bind_directory(repository,directory,kind,inputs)



def restore_source(repository, directory, inputs):
    manifest = restore(repository,directory,'source',inputs)
    require(manifest['source_manifest_sha256'] == READER and manifest['source_sha256'] == SOURCE
            and manifest['native_sha256'] == IMAGE, 'fixed reviewed F64 source identity')
    inputs.bind(Path(directory)/'source-manifest.json',READER)
    return Path(directory)



def bind_reference(repository, directory, source_directory, inputs):
    source = bind_directory(repository, source_directory, 'source', inputs)
    reference = bind_directory(repository, directory, 'reference', inputs)
    directory = Path(directory)
    report = json.loads(inputs.bind(directory / 'report.json', REFERENCE))
    require(reference['original_report_sha256'] == REFERENCE
            and reference['source_package_manifest_sha256'] == SOURCE_MANIFEST
            and reference['original_producer_input_count'] == len(reference['input_path_map'])
            == len(report['inputs']) == 102, 'all original102 F64 reference producer inputs')
    require(set(reference['input_path_map']) == set(report['inputs']), 'exact historical F64 input path mapping')
    for original, identity in report['inputs'].items():
        target = reference['input_path_map'][original]
        require(set(target) == {'package', 'member'} and target['package'] in PACKAGES,
                'explicit source or reference package mapping')
        metadata = source if target['package'] == 'source' else reference
        require(target['member'] in metadata['files'] and metadata['files'][target['member']] ==
                {key: identity[key] for key in ('bytes', 'sha256', 'mode')},
                'the original producer input bytes and mode differ: ' + original)
    require(sum(row['package'] == 'source' for row in reference['input_path_map'].values()) == 98
            and reference['original_frozen_source_inputs_in_source_package'] == 98
            and reference['original_support_inputs_in_reference_package'] == 4
            and reference['original_support_modes_preserved'] is True,
            'exact98 frozen source and4 original support inputs')
    require(report['schema'] == 'heptatorch.f64.tensor-reference33.v1'
            and report['status'] == 'OBSERVED' and report['torch_executed'] is True
            and report['native_executed'] is False and report['inputs_unchanged'] is True
            and report['failures'] == [] and report['expected_groups'] == len(report['groups']) == 33
            and report['expected_boundary_cases'] == len(report['boundary_cases']) == 12
            and report['producer_manifest_sha256'] == '23d09ddb6414eca8610e80ff17519129dc6175c02789f8c693a8e1ef36bd590e'
            and report['oracle_provenance']['status'] == 'PASS'
            and len(report['oracle_provenance']['checks']) == 11
            and all(value is True for value in report['oracle_provenance']['checks'].values()),
            'complete actual original F64 observations and all11 fixed-wheel checks')
    require(reference['original_reference_output_members'] == 181
            and reference['terminal_public_member'] == 'terminal/public-terminal.json'
            and reference['terminal_public_sha256'] == TERMINAL
            and reference['terminal_stream_members'] == 4,
            'the original181 outputs and complete public terminal/stream entities')
    terminal = json.loads(inputs.bind(directory / 'terminal/public-terminal.json', TERMINAL))
    require(terminal['schema'] == 'heptatorch.original-f64719.selected-terminal-readback.v1'
            and terminal['public_fields_are_lossless_projection'] is True
            and terminal['executed_here'] is False
            and terminal['actual_argv_present_in_selected_terminal_artifact'] is False
            and terminal['actual_interpreter_entity_present_in_selected_terminal_artifact'] is False,
            'the actual public terminal projection and its historical evidence limits')
    require(terminal['executions']['reference']['report'] == reference['files']['report.json'],
            'original terminal report identity equals the complete bound reference')
    streams = {}
    for role in ('reference', 'native'):
        execution = terminal['executions'][role]
        status = execution['status']
        require(status['state'] == 'succeeded' and status['exit_code'] == 0
                and status['timed_out'] is False and status['cancelled'] is False
                and status['output_complete'] is True and status['output_durable'] is True
                and status['terminal_publication_confirmed'] is True
                and status['terminal_publication_protocol'] == 1,
                'the preserved original terminal completed with durable streams')
        for name in ('stdout', 'stderr'):
            row = execution['streams'][name]
            require(row['path'] == role + '.' + name and row['EOF'] is True
                    and row['offset'] == 0 and row['next_offset'] == row['bytes']
                    and status['log_errors'][name] is None and status['log_synced'][name] is True
                    and status['log_truncated'][name] is False
                    and status['log_bytes'][name] == dict(seen=row['bytes'], stored=row['bytes']),
                    'complete original terminal stream offsets, durability and extent')
            path = directory / 'terminal' / row['path']
            raw = inputs.bind(path, row['sha256'])
            require(len(raw) == row['bytes'] and stat.S_IMODE(path.stat().st_mode) == row['mode'],
                    'the physical original terminal stream bytes and mode')
            streams[role + '.' + name] = raw
    lines = [json.loads(line) for line in streams['reference.stdout'].splitlines()]
    require(len(lines) == 3
            and lines[0] == dict(name='source', exit_code=0, status='PASS_SOURCE_AND_WIRE_ONLY_NOT_RUN', stdout_bytes=127, stderr_bytes=0)
            and lines[1] == dict(name='reference', exit_code=0, status='PREPARED_NOT_RUN', stdout_bytes=136, stderr_bytes=0)
            and lines[2] == dict(status='OBSERVED', expected_groups=33, expected_boundary_cases=12,
                                 torch_executed=True, native_executed=False),
            'the preserved source/preparation/actual-reference stdout records')
    require(reference['historical_reference_execution_receipt_available'] is True
            and reference['historical_reference_stdout_stderr_available'] is True
            and reference['terminal_also_retains_old_native_metadata_without_transferring_qualification'] is True,
            'the available historical terminal evidence is retained without native qualification transfer')
    for key in ('historical_reference_actual_argv_available',
                'historical_reference_interpreter_executable_entity_available',
                'historical_actual_argv_or_interpreter_identity_inferred',
                'old_native_runtime_PASS_transferred', 'original_report_paths_rewritten'):
        require(reference[key] is False, 'historical F64 evidence limits remain explicit: ' + key)
    return dict(root=directory, report=directory / 'report.json', source=Path(source_directory),
                manifest=reference, original_producer_inputs=102,
                historical_actual_argv_available=False,
                historical_interpreter_executable_entity_available=False)


def restore_reference(repository, directory, source_directory, inputs):
    restore(repository,directory,'reference',inputs)
    return bind_reference(repository,directory,source_directory,inputs)



def reader_cli(repository, build, build_sha, reference):
    return ['--candidate',str(repository),'--build-directory',str(build),
            '--reference',str(reference['report']),'--expected-source-sha256',SOURCE,
            '--expected-native-sha256',IMAGE,'--expected-build-sha256',build_sha,
            '--expected-reference-sha256',REFERENCE]



