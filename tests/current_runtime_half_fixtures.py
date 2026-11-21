"""Transport the frozen Half reader and original Torch reference bytes.

These packages preserve the original 140 producer input identities and modes.
The historical reference did not retain an independent terminal/argv/interpreter
receipt. This module does not infer those missing facts from a later execution.
It invokes no Torch expression, compiler, or native application.
"""
import hashlib
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
import types


SOURCE_PACKAGE = 'validation/current-half47-source'
SOURCE_MANIFEST = '6a0b3541412976672e7b4abeddc714aa198139f9ec705fcaa769748e7e72f9a0'
SOURCE_ARCHIVE = '875d8dc72d0e8c7d15b3aa0f9c90176cbe4c90a532895d5a85d73395e4a87c0b'
REFERENCE_PACKAGE = 'validation/current-half47-reference'
REFERENCE_MANIFEST = '544ce7446e11c09c115509b82e5b536cb454806cebe790bd444826f8df4b54c0'
REFERENCE_ARCHIVE = '241b11a19d29e7615b9d9b08da3bf75ed333af9a853cbf2482494138c4061102'
REFERENCE = 'ae6079c8f0960db53d26e056cc67fbe0a8eee7de04aa0644f4475afc368a75fe'
READER = '1f67d27b8a5af133955d8a8cdfdb3094415efe31d1d9ee7ab09437f683ecef06'
SOURCE = '51820849f7becec549be15babace00f00687601ae7f53bcee0e636ca02f721f9'
IMAGE = 'aa3cc8f0c8738a9505d31c9e8ba5a523b7e08b591b61fc7bfb8592295491618b'
CURRENT_SOURCE = 'e0c8e0a3c3ee16d11cad8af6be1b52bd56295581e74bdcc65131f71a57540025'
CURRENT_IMAGE = '3ee1fae0e9d20d10dfd2645a3575d23357bcaca0dae03e5f0ff6bf1d4b4197a5'
PACKAGES = {
    'source': (SOURCE_PACKAGE, SOURCE_MANIFEST, SOURCE_ARCHIVE, 471936, 146, 11400278,
               'heptatorch.current736-half47.source-package.v1'),
    'reference': (REFERENCE_PACKAGE, REFERENCE_MANIFEST, REFERENCE_ARCHIVE, 907880, 134, 7472736,
                  'heptatorch.half47.fixed-original-reference-package.v1'),
}


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
            'complete fixed Half package identity and denominator')
    for name in manifest['files']:
        safe_name(name)
    archive = inputs.bind(root / 'complete.tar.xz', archive_sha)
    require(len(archive) == archive_bytes, 'complete original Half archive bytes')
    return manifest, archive


def bind_directory(repository, directory, kind, inputs):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), 'ordinary restored Half directory')
    manifest, _ = package(repository, kind, inputs)
    paths = list(directory.rglob('*'))
    require(all(not p.is_symlink() and (p.is_dir() or p.is_file()) for p in paths),
            'only ordinary restored Half files and directories')
    actual = {p.relative_to(directory).as_posix() for p in paths if p.is_file()}
    require(actual == set(manifest['files']), 'exact restored Half file census')
    for name, row in manifest['files'].items():
        path = directory.joinpath(*safe_name(name).parts)
        raw = inputs.bind(path, row['sha256'])
        require(len(raw) == row['bytes'] and stat.S_IMODE(path.stat().st_mode) == row['mode'],
                'restored Half file extent or mode changed: ' + name)
    return manifest


def restore(repository, directory, kind, inputs):
    directory = Path(directory)
    require(not directory.exists() and not directory.is_symlink(), 'fresh Half package destination')
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
                    'ordinary, unique, expected Half archive member')
            row = manifest['files'][member.name]
            raw = tar.extractfile(member).read()
            require(member.size == len(raw) == row['bytes'] and member.mode == row['mode']
                    and sha(raw) == row['sha256'], 'complete original Half member bytes and mode')
            path = directory.joinpath(*name.parts)
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as output:
                output.write(raw)
            path.chmod(member.mode)
            seen.add(member.name)
    require(seen == set(manifest['files']), 'no omitted Half package members')
    return bind_directory(repository,directory,kind,inputs)


def restore_source(repository, directory, inputs):
    manifest = restore(repository,directory,'source',inputs)
    require(manifest['source_manifest_sha256'] == READER and manifest['source_sha256'] == SOURCE
            and manifest['native_sha256'] == IMAGE, 'fixed reviewed Half source identity')
    inputs.bind(Path(directory)/'source-manifest.json',READER)
    return Path(directory)


def bind_reference(repository, directory, source_directory, inputs):
    source = bind_directory(repository,source_directory,'source',inputs)
    reference = bind_directory(repository,directory,'reference',inputs)
    report = json.loads(inputs.bind(Path(directory)/'report.json',REFERENCE))
    require(reference['original_report_sha256'] == REFERENCE
            and reference['source_package_manifest_sha256'] == SOURCE_MANIFEST
            and reference['original_producer_input_count'] == len(reference['input_path_map'])
            == len(report['inputs']) == 140, 'all original 140 reference producer inputs')
    require(set(reference['input_path_map']) == set(report['inputs']), 'exact historical input path mapping')
    for original, identity in report['inputs'].items():
        target = reference['input_path_map'][original]
        require(set(target) == {'package','member'} and target['package'] in PACKAGES,
                'explicit source or reference package mapping')
        metadata = source if target['package'] == 'source' else reference
        require(target['member'] in metadata['files'] and metadata['files'][target['member']] ==
                {key:identity[key] for key in ('bytes','sha256','mode')},
                'the original producer input bytes and mode differ: ' + original)
    require(report['status'] == 'OBSERVED' and report['Torch_executed'] is True
            and report['native_executed'] is False and report['inputs_unchanged'] is True
            and report['failures'] == [] and report['oracle_provenance']['status'] == 'PASS'
            and len(report['oracle_provenance']['checks']) == 11
            and all(value is True for value in report['oracle_provenance']['checks'].values()),
            'the original reference observation and fixed-wheel checks are retained')
    for key in ('historical_reference_execution_receipt_available','historical_reference_stdout_stderr_available',
                'historical_actual_argv_or_interpreter_identity_inferred','old_native_runtime_PASS_transferred'):
        require(reference[key] is False, 'historical evidence limits must remain explicit')
    return dict(root=Path(directory),report=Path(directory)/'report.json',source=Path(source_directory),
                manifest=reference,original_producer_inputs=140)


def restore_reference(repository, directory, source_directory, inputs):
    restore(repository,directory,'reference',inputs)
    return bind_reference(repository,directory,source_directory,inputs)


def reader_cli(repository, build, build_sha, reference, half_mul, half_neg, half_addsub):
    return ['--reader-directory',str(reference['source']),
            '--candidate',str(repository),'--build-directory',str(build),
            '--reference',str(reference['report']),'--expected-source-sha256',CURRENT_SOURCE,
            '--expected-native-sha256',CURRENT_IMAGE,'--expected-build-sha256',build_sha,
            '--expected-reference-sha256',REFERENCE,
            '--half-mul-reference-directory',str(half_mul['root']),
            '--half-mul-reference-execution',str(half_mul['execution']),
            '--half-neg-reference-directory',str(half_neg['root']),
            '--half-neg-reference-execution',str(half_neg['execution']),
            '--half-addsub-reference-directory',str(half_addsub['root']),
            '--half-addsub-reference-execution',str(half_addsub['execution'])]


def refusal_cli(repository, build, build_sha, source):
    return ['--bridge-directory',str(source),'--repository',str(repository),
            '--build-directory',str(build),'--expected-build-sha256',build_sha]


def current_bridge(repository, reader_directory, half_mul=None, half_neg=None, half_addsub=None):
    """Wrap the frozen bridge with the exact three-function source inverse."""
    repository = Path(repository).resolve(strict=True)
    helper_path = repository / 'tests/current_runtime_f64_fixtures.py'
    helper = types.ModuleType('half_current_source_adapter')
    helper.__file__ = str(helper_path)
    exec(compile(helper_path.read_bytes(), str(helper_path), 'exec'), helper.__dict__)
    require(helper.CURRENT_SOURCE == CURRENT_SOURCE and helper.CURRENT_IMAGE == CURRENT_IMAGE,
            'same exact current image in both original dtype readers')
    directory = Path(reader_directory).resolve(strict=True)
    bridge_path = directory / 'bridge.py'
    raw = bridge_path.read_bytes()
    require(sha(raw) == 'f3b19a8dcc1a004323e033f414e801a52f12b0f0d556e6cbc005169d26fdef9e',
            'entire unchanged original Half47 bridge')
    original = helper.load_module(raw, bridge_path, 'half_current_original47_bridge')
    require(original.SOURCE == SOURCE and original.IMAGE == IMAGE and original.REFERENCE == REFERENCE,
            'the immutable original bridge keeps its historical identities')
    adapted = types.ModuleType('half_current47_bridge')
    adapted.__dict__.update(original.__dict__)
    adapted.SOURCE, adapted.IMAGE = CURRENT_SOURCE, CURRENT_IMAGE
    adapted.Adapter = helper.current_adapter_type(original.Adapter, original, repository,
                                                   (helper_path, Path(__file__)), 'half')
    if half_mul is not None:
        successor_path = repository / 'tests/current_runtime_half_mul_successor.py'
        successor = helper.load_module(successor_path.read_bytes(), successor_path,
                                       'current_half47_mul22_successor')
        if half_neg is None:
            require(half_addsub is None, 'add/sub admission requires both unchanged predecessor helpers')
            adapted.Adapter = successor.adapter_type(adapted.Adapter, repository,
                                                      directory, half_mul)
        else:
            neg_path = repository / 'tests/current_runtime_half_neg_successor.py'
            neg = helper.load_module(neg_path.read_bytes(), neg_path,
                                     'current_half47_neg2_successor')
            if half_addsub is None:
                adapted.Adapter = neg.adapter_type(adapted.Adapter, successor, repository,
                                                    directory, half_mul, half_neg)
            else:
                addsub_path = repository / 'tests/current_runtime_half_addsub_successor.py'
                addsub = helper.load_module(addsub_path.read_bytes(), addsub_path,
                                           'current_half47_addsub40_successor')
                adapted.Adapter = addsub.adapter_type(adapted.Adapter, successor, neg, repository,
                                                       directory, half_mul, half_neg, half_addsub)
    else:
        require(half_neg is None and half_addsub is None,
                'new arithmetic admission requires the unchanged22 multiplication adapter')
    return adapted


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reader-directory', type=Path, required=True)
    parser.add_argument('--expected-reference-sha256', required=True)
    parser.add_argument('--half-mul-reference-directory', type=Path, required=True)
    parser.add_argument('--half-mul-reference-execution', type=Path, required=True)
    parser.add_argument('--half-neg-reference-directory', type=Path, required=True)
    parser.add_argument('--half-neg-reference-execution', type=Path, required=True)
    parser.add_argument('--half-addsub-reference-directory', type=Path, required=True)
    parser.add_argument('--half-addsub-reference-execution', type=Path, required=True)
    args, _ = parser.parse_known_args()
    require(args.expected_reference_sha256 == REFERENCE, 'exact original actual Half reference')
    bridge = current_bridge(args.candidate, args.reader_directory,
        dict(root=args.half_mul_reference_directory, execution=args.half_mul_reference_execution),
        dict(root=args.half_neg_reference_directory, execution=args.half_neg_reference_execution),
        dict(root=args.half_addsub_reference_directory, execution=args.half_addsub_reference_execution))
    original_argv = list(sys.argv)
    for flag in ('--reader-directory', '--half-mul-reference-directory', '--half-mul-reference-execution',
                 '--half-neg-reference-directory', '--half-neg-reference-execution',
                 '--half-addsub-reference-directory', '--half-addsub-reference-execution'):
        at = sys.argv.index(flag)
        sys.argv = sys.argv[:at] + sys.argv[at + 2:]
    try:
        with bridge.original_modules() as (originals, old_run, startup):
            return bridge.Adapter(originals, old_run, startup).current_main()()
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    raise SystemExit(main())
