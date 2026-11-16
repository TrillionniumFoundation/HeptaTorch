#!/usr/bin/env python3
"""Cold-build the frozen I64 profile and execute four independent native gates.

Production arithmetic, allocation, lifetime and AD run in the HIA executables.
This coordinator checks original evidence; it does not implement Tensor math.
Live32, the complete accounted35 and its copied controls remain separate gates.
There is no existing-image substitute for the cold producer invocation.
"""
import argparse
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tarfile
import traceback

SOURCE = 'adf6cae67666273aa6fb77989080ca3b1e3632188d2d7956a6d8b71808787e62'
IMAGE = 'b7324639bade3312981d1cbf9c2491b9fcff04fb6c3f808e9c5e1a7c2ca88f08'
PARENT_SOURCE = '6ffe509afce28ff15705b4c7b269eaef8f8f89f9d8b2486d4d9fa45b6be66ef3'
PARENT_IMAGE = '830f0b4ef8d6ad8c6a105be56d383628b45c84e3e43905fee6812c7c52d35a51'
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
RUNTIME_LINKER = '0c5176e3aa968631ae592b887c5dc3cf1e8e3f48f73593f076e9d7965ffacfd7'
FLAT = 'c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6'
MANIFESTS = (
    ('validation/integer-tensors/package-manifest.json', '375714e956b99ec5b3e78e1b0369da83d7f690ef18defddfb60477ffb5d9229c', 70),
    ('tools/integer_tensors/package-manifest.json', 'ece14e766825fe0843dfe624fdab8a3b4ec407d27f7a0b5e155e2ee15ee77b71', 54),
    ('validation/int64-inplace-review/package-manifest.json', '60db4e5b20cb3f2ad83383fb4ed6c47a10741a5de9032a0b7db815197cd01e33', 14),
    ('validation/integer-resource-review/package-manifest.json', 'd78bb9575187ea8140174219ee231973b20efdf15db1d91cbf987d4496a4c3b0', 15),
    ('validation/integer-resource-review/typed-handoff-07/package-manifest.json', '9e3ceca498e2dd8bed52c176a5a2aa30f67efdc93a75caaa0e8f0f08b10bd991', 8),
)
PRODUCER_STEPS = (
    'cold-native-build', 'original-storage12', 'original-arithmetic30',
    'original-dropout12', 'alias-mutation8', 'integer-RNE1237',
    'CPU-promotion20', 'view-base-lifetime6', 'independent-inplace40',
    'legacy523-behavior12',
)


def require(value, message):
    if not value:
        raise AssertionError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, ('duplicate JSON key', key))
        result[key] = value
    return result


def decode(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def relative(root, name):
    path = PurePosixPath(name)
    require(name and not path.is_absolute() and '..' not in path.parts
            and str(path) == name, ('unsafe relative path', name))
    return root / name


class Custody:
    def __init__(self):
        self.files = {}

    def bind(self, path, expected=None, size=None, mode=None):
        original = Path(path)
        require(not original.is_symlink(), ('input symlink', str(original)))
        path = original.resolve()
        require(path.is_file(), ('missing input', str(path)))
        value = {'bytes': path.stat().st_size,
                 'mode': stat.S_IMODE(path.stat().st_mode), 'sha256': digest(path)}
        require(expected is None or value['sha256'] == expected, ('input SHA', str(path)))
        require(size is None or value['bytes'] == size, ('input size', str(path)))
        require(mode is None or value['mode'] == mode, ('input mode', str(path)))
        require(path not in self.files or self.files[path] == value, ('input changed', str(path)))
        self.files[path] = value
        return value

    def read(self, path, expected=None):
        self.bind(path, expected)
        return decode(path)

    def finish(self):
        for path, value in list(self.files.items()):
            self.bind(path, value['sha256'], value['bytes'], value['mode'])


def manifest(repo, name, pin, count, custody):
    value = custody.read(relative(repo, name), pin)
    rows = value['files']
    rows = [dict(row, path=key) for key, row in rows.items()] if isinstance(rows, dict) else rows
    require(len(rows) == count == len({row['path'] for row in rows}), ('manifest members', name))
    for row in rows:
        custody.bind(relative(repo, row['path']), row['sha256'], row['bytes'], row['mode'])
    return {'path': name, 'sha256': pin, 'members': count,
            'all_bytes_modes_lengths_verified': True}


def runtime_identity(value):
    require(value['source_sha256'] == SOURCE and value['native_sha256'] == IMAGE,
            'exact I64 source and native executable identity')


def build_identity(directory, profile, custody, parent=False):
    value = custody.read(directory / 'build.json')
    require(value['status'] == 'PASS' and value['link_exit'] == 0, 'native build success')
    for key in ('source_sha256', 'native_sha256', 'functions', 'blocks', 'instructions',
                'descriptor_bytes', 'state_format', 'backing_format', 'units'):
        require(value[key] == profile[key], ('exact native profile', key))
    require(value['compiler_sha256'] == COMPILER and value['linker_sha256'] == LINKER,
            'native build tool identities')
    require(len(value['units']) == 11 == len({row['name'] for row in value['units']}),
            'eleven actual independent units')
    custody.bind(directory / 'runtime.hepta', PARENT_SOURCE if parent else SOURCE)
    custody.bind(directory / ('tensor-dtype-dropout-session' if parent else 'tensor-integer-session'),
                 PARENT_IMAGE if parent else IMAGE)
    for row in value['units']:
        require(Path(row['name']).name == row['name'] and row['compiler_exit'] == 0,
                'actual module compilation')
        custody.bind(directory / (row['name'] + '.hepta'), row['source_sha256'])
        custody.bind(directory / (row['name'] + '.o'), row['object_sha256'])
    return {'build_report_sha256': digest(directory / 'build.json'),
            'source_sha256': value['source_sha256'], 'native_sha256': value['native_sha256'],
            'actual_units_verified': 11}


def verify_archive(path, files):
    """Check every original member and the single complete XZ stream."""
    decoder = lzma.LZMADecompressor()
    tar_hash = hashlib.sha256()
    tar_bytes = 0
    with path.open('rb') as source:
        while True:
            block = source.read(1 << 20) if decoder.needs_input else b''
            if not block and decoder.needs_input:
                break
            raw = decoder.decompress(block, max_length=1 << 20)
            tar_hash.update(raw)
            tar_bytes += len(raw)
            if decoder.eof:
                require(not decoder.unused_data and not source.read(1), 'trailing or concatenated XZ data')
                break
    require(decoder.eof, 'complete XZ EOF')
    seen = set()
    with lzma.open(path, 'rb') as source, tarfile.open(fileobj=source, mode='r|') as tar:
        for member in tar:
            relative(Path('.'), member.name)
            require(member.isfile() and not member.issym() and not member.islnk(), 'regular evidence member')
            require(member.name in files and member.name not in seen, ('unique declared member', member.name))
            pin = files[member.name]
            require(member.size == pin['bytes'] and member.mode == pin['mode'], ('member metadata', member.name))
            with tar.extractfile(member) as stream:
                require(hashlib.file_digest(stream, 'sha256').hexdigest() == pin['sha256'], ('member SHA', member.name))
            seen.add(member.name)
    require(seen == set(files), 'complete archive member set')
    return {'members': len(seen), 'decompressed_tar_bytes': tar_bytes,
            'decompressed_tar_sha256': tar_hash.hexdigest(), 'single_XZ_stream_EOF': True,
            'all_member_bytes_modes_sizes_verified': True}


def producer_identity(root, custody):
    value = custody.read(root / 'report.json')
    runtime_identity(value)
    require(value['status'] == 'PASS' and value['step_count'] == len(value['steps']) == 10, 'producer ten gates')
    require(tuple(row['name'] for row in value['steps']) == PRODUCER_STEPS, 'exact producer gate order')
    require(value['package_manifest_sha256'] == MANIFESTS[1][1], 'producer source manifest')
    require(value['independent_inplace40_package_sha256'] == MANIFESTS[2][1], 'independent actual40 manifest')
    for key in ('all_package_members_unchanged', 'native_cold_rebuild_executed',
                'all_named_gate_statuses_and_counts_matched'):
        require(value[key] is True, ('producer admission', key))
    require(value['original54_reference_indices_exactly_once'] == list(range(54)), 'all original54 exactly once')
    archives = []
    for row in value['steps']:
        require(row['status'] == 'PASS' and row['exit_code'] == 0 and row['process_started'] is True,
                ('actual producer process', row['name']))
        for suffix in ('stdout', 'stderr'):
            custody.bind(root / (row['name'] + '.' + suffix), row[suffix + '_sha256'])
        child = custody.read(relative(root, row['report']), row['report_sha256'])
        runtime_identity(child)
        require(child['status'] == 'PASS', 'original child report PASS')
        if row['name'] == 'cold-native-build':
            require(child['native_build_process_started'] is True and child['native_build_process_completed'] is True
                    and child['existing_integer_image_reused'] is False and child['units_reproduced_exactly'] == 11,
                    'actual create-only cold build')
        else:
            archive_pin = row['archive']
            archive = relative(root / 'evidence', archive_pin['archive'])
            custody.bind(archive, archive_pin['archive_sha256'], archive_pin['archive_bytes'])
            receipt = custody.read(root / 'evidence' / (row['name'] + '.manifest.json'))
            require(receipt['archive_sha256'] == archive_pin['archive_sha256'], 'producer archive identity')
            verified = verify_archive(archive, receipt['members'])
            require(verified['members'] == archive_pin['member_count'] == receipt['member_count'], 'producer raw count')
            archives.append(dict(verified, gate=row['name'], archive_sha256=archive_pin['archive_sha256']))
        if row['name'] == 'legacy523-behavior12':
            require(child['step_count'] == len(child['steps']) == 12 and child['existing_native_units_verified'] == 11,
                    'all twelve original behavior gates')
            labels = [(x['name'], x['status']) for x in child['steps']]
            require(labels[-2:] == [('actual-retained12', 'QUALIFIED_EXPOSED_SCOPE'), ('inherited16', 'PASS')],
                    'original bounded status labels retained')
            require(child['all_inputs_unchanged'] is True and child['all_expected_statuses_and_counts_matched'] is True,
                    'legacy suite source and count preservation')
    return {'report_sha256': digest(root / 'report.json'), 'steps': 10,
            'actual_cold_build_verified': True, 'raw_archives': archives,
            'original_bounded_legacy_statuses_preserved': True}


def archive_evidence(out):
    excluded = {'report.json', 'evidence-manifest.json', 'raw-evidence.tar.xz'}
    files = {}
    for path in sorted(out.rglob('*')):
        require(not path.is_symlink(), ('evidence symlink', str(path)))
        if path.is_file() and path.relative_to(out).as_posix() not in excluded:
            files[path.relative_to(out).as_posix()] = {'bytes': path.stat().st_size,
                'mode': stat.S_IMODE(path.stat().st_mode), 'sha256': digest(path)}
    path = out / 'raw-evidence.tar.xz'
    with lzma.open(path, 'wb', preset=3) as stream, tarfile.open(fileobj=stream, mode='w|') as tar:
        for name, pin in files.items():
            original = relative(out, name)
            require(digest(original) == pin['sha256'], 'archive input unchanged')
            member = tarfile.TarInfo(name)
            member.size, member.mode, member.mtime = pin['bytes'], pin['mode'], 0
            with original.open('rb') as source:
                tar.addfile(member, source)
    verified = verify_archive(path, files)
    value = dict(verified, schema='heptatorch.integer-tensor-fresh-raw-evidence.v1', status='VERIFIED',
                 source_sha256=SOURCE, native_sha256=IMAGE, archive_sha256=digest(path),
                 archive_bytes=path.stat().st_size, uncompressed_bytes=sum(v['bytes'] for v in files.values()),
                 files=files, scope='Byte custody is separate from suite admission.')
    dump(out / 'evidence-manifest.json', value)
    return {key: item for key, item in value.items() if key != 'files'} | {
        'archive': path.name, 'manifest': 'evidence-manifest.json',
        'manifest_sha256': digest(out / 'evidence-manifest.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'parent-build', 'upstream', 'compiler', 'linker', 'flat-compiler', 'stack-build', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    repo, parent, stack, out = args.repo.resolve(), args.parent_build.resolve(), args.stack_build.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    custody = Custody()
    report = {'schema': 'heptatorch.integer-tensor-native-integration.v1', 'status': 'RUNNING',
              'source_sha256': SOURCE, 'native_sha256': IMAGE, 'producer_mode': 'FRESH_COLD_REQUIRED',
              'producer_executed_this_invocation': False, 'producer_completed_this_invocation': False,
              'Torch_executed_here': False, 'commands': [], 'whole_pytorch_gate': 'HOLD',
              'full_pytorch_rewrite_complete': False, 'pytorch_api_denominator': 2590, 'closed_schema_count': 0,
              'counting_boundary': 'Named suites overlap. No sum is a unique upstream test or schema count.'}

    def save():
        dump(out / 'report.json', report)

    def execute(name, script, arguments, expected):
        receipt = out / name / 'report.json'
        require(not receipt.exists(), 'fresh child output required')
        custody.bind(script)
        argv = [sys.executable, '-S', '-B', str(script), *map(str, arguments), '--output', str(out / name)]
        row = {'name': name, 'argv': argv, 'status': 'STARTING', 'process_started': False, 'process_completed': False}
        report['commands'].append(row)
        dump(out / (name + '.argv.json'), argv)
        save()
        try:
            with (out / (name + '.stdout')).open('wb') as so, (out / (name + '.stderr')).open('wb') as se:
                process = subprocess.Popen(argv, cwd=repo, stdout=so, stderr=se)
                row.update(status='RUNNING', process_started=True, pid=process.pid)
                if name == 'producer10':
                    report['producer_executed_this_invocation'] = True
                save()
                try:
                    process.wait(timeout=2400)
                except BaseException:
                    process.kill()
                    process.wait()
                    raise
                row.update(process_completed=True, exit_code=process.returncode)
                if name == 'producer10':
                    report['producer_completed_this_invocation'] = True
        finally:
            for suffix in ('stdout', 'stderr'):
                path = out / (name + '.' + suffix)
                if path.is_file():
                    row[suffix] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
            save()
        value = decode(receipt) if receipt.is_file() else {}
        row.update(status=value.get('status', 'NO_REPORT'))
        if receipt.is_file():
            row.update(report=receipt.relative_to(out).as_posix(), report_sha256=digest(receipt))
        save()
        require(process.returncode == 0 and value.get('status') == 'PASS', ('actual gate failure', name))
        custody.bind(receipt, row['report_sha256'])
        runtime_identity(value)
        for key, wanted in expected.items():
            require(type(value[key]) is type(wanted) and value[key] == wanted, ('fixed gate obligation', name, key))
        return value

    try:
        custody.bind(Path(__file__))
        report['manifests'] = [manifest(repo, *row, custody) for row in MANIFESTS]
        profile = custody.read(repo / 'tools/integer_tensors/profile.json')
        parent_profile = custody.read(repo / 'tools/native_dropout/profile.json')
        runtime_identity(profile)
        require(parent_profile['source_sha256'] == PARENT_SOURCE and parent_profile['native_sha256'] == PARENT_IMAGE,
                'exact previous typed-owner profile')
        custody.bind(args.compiler, COMPILER)
        custody.bind(args.linker, LINKER)
        custody.bind(args.flat_compiler, FLAT)
        require(args.compiler.resolve() == (stack / 'native/stack-module-compiler').resolve()
                and args.linker.resolve() == (stack / 'native/stack-linker-session').resolve(), 'explicit same stack build')
        custody.bind(stack / 'native/stack-linker-runtime', RUNTIME_LINKER)
        report['parent_build'] = build_identity(parent, parent_profile, custody, parent=True)
        execute('producer10', repo / 'tools/integer_tensors/qualify.py', [
            '--parent', parent, '--repository', repo, '--upstream', args.upstream.resolve(),
            '--compiler', args.compiler.resolve(), '--linker', args.linker.resolve(),
            '--flat-compiler', args.flat_compiler.resolve()], {'step_count': 10})
        report['producer_admission'] = producer_identity(out / 'producer10', custody)
        native = out / 'producer10/native'
        report['new_build'] = build_identity(native, profile, custody)
        tests = repo / 'tests/integer_resource_review'
        protocol = repo / 'tests/integer_tensor_review/wire/protocol.py'
        reference = repo / 'validation/integer-tensors/reference/torch-int64-tensor-reference.json'
        common = ['--build', native, '--protocol', protocol, '--reference', reference,
                  '--expected-source-sha256', SOURCE, '--expected-native-sha256', IMAGE]
        execute('resources6', tests / 'test_integer_resources.py', common, {
            'case_count': 6, 'passed': 6, 'failed': 0, 'frames_before_EOF': 527, 'checks': 157,
            'complete_rollback_bytes': 44811376, 'module_count': 11, 'native_executed': True,
            'complete_six_case_plan_selected': True, 'source_custody_after': True})
        execute('owned-entry5', tests / 'test_integer_owned_entry.py', common + [
            '--stack-build', stack, '--probe-source', tests / 'entry_probe.hepta'], {
            'case_count': 5, 'passed': 5, 'linked_modules': 12, 'native_probe_processes': 46,
            'interactive_frames': 33, 'complete_owner_bytes_compared': 32155104, 'source_custody_after': True})
        execute('typed-handoff2', tests / 'test_integer_typed_handoff.py', [
            '--build', native, '--previous-build', parent, '--stack-build', stack,
            '--protocol', protocol, '--controller', tests / 'state_handoff.hepta'], {
            'case_count': 2, 'passed': 2, 'failed': 0, 'linked_modules': 23, 'frame_count': 13,
            'rollback_bytes': 1398048, 'EOF_exit_code': 0, 'source_custody_after': True,
            'previous_source_sha256': PARENT_SOURCE, 'previous_native_sha256': PARENT_IMAGE})
        report['separate_required_CI_gates'] = {
            'live32': 'NOT_EXECUTED_BY_NATIVE_COORDINATOR',
            'complete_accounted35_and_copied18': 'NOT_EXECUTED_BY_NATIVE_COORDINATOR',
            'no_inference_from_old_profile_or_inherited_suites': True}
        report['counts'] = {'producer_top_level_steps': 10, 'legacy_behavior_gates': 12,
            'inherited_named_suites_with_original_OPENs': 16, 'actual_reference_foundation_cases': 54,
            'alias_reference_cases': 8, 'RNE_integers': 1237, 'promotion_reference_cases': 20,
            'view_base_reference_cases': 6, 'independent_inplace_reference_cases': 40,
            'native_resource_controls': 6, 'native_entry_controls': 5, 'native_typed_directions': 2}
        custody.finish()
        report['complete_current_execution_evidence'] = archive_evidence(out)
        custody.finish()
        report.update(status='PASS', command_count=len(report['commands']), all_inputs_unchanged=True,
                      checked_inputs={str(path): value for path, value in custody.files.items()})
        save()
        print(json.dumps({'status': 'PASS', 'report_sha256': digest(out / 'report.json'),
                          'commands_executed': len(report['commands']), 'source_sha256': SOURCE, 'native_sha256': IMAGE}))
        return 0
    except BaseException as error:
        report.update(status='FAIL', error=repr(error), traceback=traceback.format_exc(),
                      successful_suite_completion_claimed=False)
        save()
        try:
            report['partial_failed_execution_evidence'] = archive_evidence(out)
        except BaseException as archive_error:
            report['archive_error'] = repr(archive_error)
        report['checked_inputs'] = {str(path): value for path, value in custody.files.items()}
        save()
        print(json.dumps({'status': 'FAIL', 'error': repr(error), 'report': str(out / 'report.json')}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
