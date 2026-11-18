"""Restore the exact public F64 reference and reread the unchanged 192 observer.

This module performs only byte transport, source admission and evidence reading.
The coordinator launches the source-derived qualify736.py separately. Two native scope
controls and six outside-scope Torch observations are not compatibility cases.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
from types import SimpleNamespace


PACKAGE = 'validation/public-f64-logical192-reference'
MANIFEST_SHA = '7bbed5ef853b5ab4f011c69139d101a9f84b6d424e38de8e81023dfddb2051f3'
ARCHIVE_SHA = '08d4487ff3e3134a8b02683cc11e91b7729d8fa97fe64ed74fd3e5aa51fbd905'
ARCHIVE_BYTES = 2185904
REFERENCE_SHA = 'fa7e876bad6c7ade4d096bf8819f1d5cbb0482dc8f81388f458c35ab88d44610'
EXECUTION_SHA = '7f236625bae0da2a06109a53682a769d031a5f716b77f24b3434fde44fd4e25c'
READER_FILES = {
    'cases.py': 'e689699b519e57eec44edac5d4c319d3ea718a3777056d1514d1f0e43a08bef3',
    'common.py': '29d25567c08e309c25d96d56f4476e3842ceda164c9647eb5c9397a04fad181a',
    'native.py': 'cc48760e9dad13dfdd834dca5086bc5fb96803dc29914d43b20f29cfb239320e',
    'qualify.py': '98b150d9295451c3be9ff4209d9f42350430e46ec0c3cd42ab3e84eab91b84f2',
    'reference.py': '2531920b4b13c991b33a3608ea2c168b184e4e7d6891b870ae874e95b1c9e296',
    'run.py': 'a155cea27556e7c8647835f11641b4732557dd882d3ec024820a06cd96aeab5f',
}

WRAPPER_FILES = {
    'derive736.py': '4c9435a8b38a8be01719069f2274b69c577f90ab9e20161e0e0dccf2eb1c0a3d',
    'run736.py': 'f272f83d2b77635a61223d1b8bbc717182ec5d7e7917e7075ccf2323afbae09d',
    'qualify736.py': '1756617c260cdc77a97c99cb640c7c3ec9bbe7f249c8e4aeb9f9fc0d043ac7b0',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def package_manifest(repository, inputs):
    package = repository / PACKAGE
    manifest = json.loads(inputs.bind(package / 'manifest.json', MANIFEST_SHA))
    require(manifest['schema'] == 'heptatorch.public-f64-logical192.reference-package.v1'
            and manifest['status'] == 'PACKAGED_ORIGINAL_REFERENCE_BYTES_ONLY'
            and manifest['file_count'] == len(manifest['files']) == len(manifest['roles']) == 16
            and set(manifest['roles'].values()) == set(manifest['files'])
            and manifest['archive'] == dict(path='complete.tar.xz', bytes=ARCHIVE_BYTES, sha256=ARCHIVE_SHA),
            'the complete fixed sixteen-file public F64 reference package is required')
    raw = inputs.bind(package / 'complete.tar.xz', ARCHIVE_SHA)
    require(len(raw) == ARCHIVE_BYTES, 'public F64 reference archive extent changed')
    return manifest, raw


def bind_reference(repository, directory, inputs):
    """Bind all original physical inputs without rewriting historical JSON paths."""
    manifest, _ = package_manifest(repository, inputs)
    payloads = {}
    for name, identity in manifest['files'].items():
        path = directory / name
        raw = inputs.bind(path, identity['sha256'])
        require(len(raw) == identity['bytes']
                and stat.S_IMODE(path.stat().st_mode) == identity['mode'],
                'restored public F64 reference member bytes or mode changed: ' + name)
        payloads[name] = raw
    roles = manifest['roles']
    reference = json.loads(payloads[roles['report']])
    execution = json.loads(payloads[roles['execution']])
    require(sha(payloads[roles['report']]) == manifest['original_report_sha256'] == REFERENCE_SHA
            and sha(payloads[roles['execution']]) == manifest['original_execution_sha256'] == EXECUTION_SHA
            and reference['cases'] == 192 and reference['allocation_observations'] == 2
            and reference['boundary_observations'] == 6
            and reference['total_actual_observations'] == execution['total_actual_observations'] == 200,
            'the original 192 + 2 + 6 actual observations changed')
    staged = execution['staged_inputs'] + execution['support_inputs']
    mapping = manifest['input_path_map']
    require(len(mapping) == len(staged) == 6 and set(mapping) == set(reference['inputs']),
            'all six original physical reference inputs are required')
    for old_path, identity in reference['inputs'].items():
        name = mapping[old_path]
        raw = payloads[name]
        rows = [row for row in staged if row.get('staged_path', row['path']) == old_path]
        require(len(rows) == 1 and rows[0]['bytes'] == identity['bytes'] == len(raw)
                and rows[0]['sha256'] == identity['sha256'] == sha(raw)
                and rows[0]['mode'] == manifest['files'][name]['mode'],
                'original physical reference source/provenance input differs: ' + old_path)
    require(manifest['python_original_resolved_path'] == execution['python']['resolved']
            and manifest['files'][roles['python']] == {
                key: execution['python'][key] for key in ('bytes', 'sha256', 'mode')},
            'the original actual Python executable entity is required')
    staging = json.loads(payloads[roles['staging']])
    require(staging['command'] == execution['command'] and staging['cwd'] == execution['cwd']
            and staging['python'] == execution['python']
            and staging['files'] == execution['staged_inputs']
            and staging['support_inputs'] == execution['support_inputs'],
            'original staging, actual argv and interpreter custody differ')
    for role, field in (('producer', 'source'), ('cases', 'case_source'), ('staging', 'staging_report')):
        identity = execution[field]
        require(identity['path'] == roles[role] and identity['bytes'] == len(payloads[roles[role]])
                and identity['sha256'] == sha(payloads[roles[role]]),
                'original local execution reference differs: ' + field)
    transfer = execution['lossless_report_transport']
    for role, identity in (
            ('transport_archive', transfer['archive']), ('transport_seal', transfer['seal_report']),
            ('transport_stdout', transfer['streams']['stdout']),
            ('transport_stderr', transfer['streams']['stderr'])):
        raw = payloads[roles[role]]
        require(identity['path'] == roles[role] and identity['bytes'] == len(raw)
                and identity['sha256'] == sha(raw), 'complete original public transfer chain differs')
    decoder = lzma.LZMADecompressor()
    restored = decoder.decompress(payloads[roles['transport_archive']], max_length=len(payloads[roles['report']]) + 1)
    require(decoder.eof and not decoder.unused_data and restored == payloads[roles['report']],
            'original reference lossless transport is incomplete')
    seal = json.loads(payloads[roles['transport_seal']])
    require(seal['lossless_roundtrip'] is True and seal['original_unchanged'] is True
            and seal['source']['sha256'] == REFERENCE_SHA
            and seal['archive']['sha256'] == transfer['archive']['sha256'],
            'original report lossless seal changed')
    return dict(root=directory, report=directory / roles['report'],
                execution=directory / roles['execution'], manifest=manifest)


def restore_reference(repository, directory, inputs):
    manifest, archive_raw = package_manifest(repository, inputs)
    require(not directory.exists() and not directory.is_symlink(), 'reference output must be new')
    directory.mkdir(parents=True, exist_ok=False)
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode='r:xz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            require(not name.is_absolute() and name.parts and '..' not in name.parts
                    and '\\' not in member.name and str(name) == member.name
                    and member.isfile() and member.name not in seen
                    and member.name in manifest['files'], 'unsafe or unexpected reference archive member')
            identity = manifest['files'][member.name]
            require(member.size == identity['bytes'] and member.mode == identity['mode'],
                    'reference archive member extent or mode differs')
            raw = archive.extractfile(member).read()
            require(len(raw) == member.size and sha(raw) == identity['sha256'],
                    'reference archive member is incomplete')
            path = directory.joinpath(*name.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(raw)
            path.chmod(member.mode)
            seen.add(member.name)
    require(seen == set(manifest['files']) and sum(row['bytes'] for row in manifest['files'].values())
            == manifest['payload_bytes'], 'reference archive omitted an original member')
    return bind_reference(repository, directory, inputs)


@contextmanager
def frozen_reader(repository, inputs):
    """Load the original modules and verify both complete 736 entry derivations."""
    here = repository / 'tests/public_f64_logical'
    original_source = {name: inputs.bind(here / name, expected)
                       for name, expected in READER_FILES.items()}
    wrapper_source = {name: inputs.bind(here / name, expected)
                      for name, expected in WRAPPER_FILES.items()}
    names = ('cases', 'common', 'native', 'derive736', 'run736', '_current_public_f64_qualification')
    absent = object()
    previous = {name: sys.modules.get(name, absent) for name in names}
    old_path, old_bytecode = sys.path[:], sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(here))
        modules = {}
        for name in names:
            filename = 'qualify736.py' if name == names[-1] else name + '.py'
            spec = importlib.util.spec_from_file_location(name, here / filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[name] = module
        generated, _ = modules['derive736'].derive(original_source, wrapper_source['derive736.py'])
        require(set(generated) == {'run736.py', 'qualify736.py'}
                and all(raw == wrapper_source[name] for name, raw in generated.items()),
                'complete public F64 736 source derivation differs')
        require(modules['run736'].ROOT == repository, 'public F64 reader physical checkout differs')
        yield modules[names[-1]]
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
        for name, module in previous.items():
            if module is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def reader_arguments(repository, build, build_sha, output, reference):
    return SimpleNamespace(repository=repository, build_directory=build,
        expected_build_sha256=build_sha, reference=reference['report'],
        expected_reference_sha256=REFERENCE_SHA, reference_execution=reference['execution'],
        expected_execution_sha256=EXECUTION_SHA, output=output,
        binary=build / 'heptatorch', prepare_only=False)


def reader_cli(repository, build, build_sha, reference):
    return ['--repository', str(repository), '--build-directory', str(build),
            '--expected-build-sha256', build_sha, '--reference', str(reference['report']),
            '--expected-reference-sha256', REFERENCE_SHA,
            '--reference-execution', str(reference['execution']),
            '--expected-execution-sha256', EXECUTION_SHA]


def read_process(directory, inputs, argv_tail, timeout):
    receipt = json.loads(inputs.bind(directory / 'runner.json'))
    require(receipt['started'] is True and receipt['exit_code'] == 0
            and receipt['stdout_EOF'] is True and receipt['stderr_EOF'] is True
            and receipt['stderr_bytes'] == 0 and not receipt.get('timed_out')
            and not receipt.get('forced_cleanup') and not receipt.get('execution_error')
            and not receipt.get('cleanup_errors') and not receipt.get('running_descendants_after_cleanup')
            and receipt['timeout_seconds'] == timeout and receipt['argv'][1:] == argv_tail
            and Path(receipt['argv'][0]).is_absolute(),
            'complete actual public F64 observer process and exact command are required')
    inputs.bind(Path(receipt['argv'][0]).resolve())
    streams = {}
    for name in ('stdout', 'stderr'):
        raw = inputs.bind(directory / ('runner.' + name), receipt[name + '_sha256'])
        require(len(raw) == receipt[name + '_bytes'], 'public F64 observer physical stream extent differs')
        streams[name] = raw
    return receipt, streams


def qualify_public192(report, repository, build, build_sha, directory, reference_directory,
                      inputs, execution_directory):
    """Reapply the frozen admission and every original complete-evidence predicate."""
    reference = bind_reference(repository, reference_directory, inputs)
    require(report == json.loads(inputs.bind(directory / 'report.json')),
            'public F64 qualification must be its actual physical report')
    require(report['schema'] == 'heptatorch.public-f64-dense-logical.qualification.v1'
            and report['status'] == 'PASS_PUBLIC_F64_LOGICAL192_ONLY'
            and report['native_executed'] is True and report['native_attempted'] is True
            and report['torch_executed'] is False and report['compiler_executed'] is False
            and report['original147_executed'] is False and report['full_pytorch_rewrite_complete'] is False
            and report['full_schema_closure'] is False and report['prepare_only'] is False
            and report['all_inputs_unchanged'] is True and report['input_recheck_errors'] == [],
            'the finite public F64 qualification scope and actual execution are required')
    cli = reader_cli(repository, build, build_sha, reference)
    here = repository / 'tests/public_f64_logical'
    outer, outer_streams = read_process(execution_directory, inputs,
        ['-I', '-S', '-B', str(here / 'qualify736.py'), *cli, '--output', str(directory)], 1200)
    inner, inner_streams = read_process(directory, inputs,
        ['-E', '-S', '-B', str(here / 'run736.py'), *cli, '--output', str(directory / 'native')], 900)
    require(report['runner'] == inner and Path(outer['argv'][0]).resolve() == Path(inner['argv'][0]).resolve(),
            'outer and inner public F64 runner custody differs')
    for path, identity in report['inputs'].items():
        raw = inputs.bind(Path(path), identity['sha256'])
        require(len(raw) == identity['bytes'] and inputs.records[str(Path(path).absolute())] == identity,
                'original public F64 qualification physical input changed: ' + path)
    for path in [here / name for name in READER_FILES] + [Path(inner['argv'][0]).resolve()]:
        require(str(path) in report['inputs'] and report['inputs'][str(path)] == inputs.records[str(path)],
                'public F64 qualifier omitted an original source or actual interpreter input')
    for name in WRAPPER_FILES:
        path = here / name
        require(str(path) in report['inputs'] and report['inputs'][str(path)] == inputs.records[str(path)],
                'public F64 qualifier omitted an actual source-derived entry or derivation input')
    args = reader_arguments(repository, build, build_sha, directory, reference)
    with frozen_reader(repository, inputs) as original:
        built, actual_reference, actual_execution, cold_inputs = original.run.admit(args, inputs)
        require(report['source_sha256'] == built['source_sha256']
                and report['native_sha256'] == built['native_sha256']
                and report['actual_build_sha256'] == build_sha
                and report['reference_sha256'] == REFERENCE_SHA
                and report['reference_execution_sha256'] == EXECUTION_SHA
                and report['cold_admission_inputs'] == cold_inputs and len(cold_inputs) == 69,
                'public F64 requires the same complete actual cold build and independent reference')
        checked = original.read_back(args, inputs, built, actual_reference, cold_inputs, directory / 'native')
        require(all(report[key] == value for key, value in checked.items()),
                'public F64 outer report differs from complete original raw-frame and owner readback')
    native_report = json.loads(inputs.bind(directory / 'native/report.json', checked['native_report_sha256']))
    require(len(inner_streams['stdout'].splitlines()) == len(outer_streams['stdout'].splitlines()) == 1,
            'both original public F64 process summaries are required')
    require(json.loads(inner_streams['stdout']) == {key: native_report.get(key) for key in
                ('status', 'cases', 'expected_cases', 'checks', 'frames', 'error')}
            and json.loads(outer_streams['stdout']) == {key: report.get(key) for key in
                ('status', 'cases', 'packet_controls', 'allocation_controls', 'native_children', 'frames', 'error')},
            'physical public F64 stdout summaries differ from their original reports')
    return dict(cases=192, frames=checked['frames'], assertions=checked['total_checks'],
        native_sessions=196, packet_controls=18, allocation_controls=2, native_boundary_controls=2,
        boundary_controls_counted_as_torch_compatibility=False, outside_native_scope_torch_observations=6,
        original147_replaced=False, complete_original_reader_reapplied=True,
        complete_raw_frames_reparsed=True, full_owner_archives_rechecked=True,
        reference_manifest_sha256=MANIFEST_SHA, reference_execution_sha256=EXECUTION_SHA)
