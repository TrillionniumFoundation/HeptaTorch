#!/usr/bin/env python3
"""Cold-build mixed integer autograd and run its seven outer native checks.

All production Tensor arithmetic and ownership run in the native HIA image.
The unchanged producer executes five gates; six independent commands then
check source custody, owner roles, resources, entry capacities and typed
handoff. Live32 and complete accounted35 remain separate required CI gates.
This entry has no existing-mixed-image or producer-report reuse option.
"""
import argparse
import hashlib
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tarfile
import traceback

SOURCE = '5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49'
IMAGE = '69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6'
PARENT_SOURCE = 'adf6cae67666273aa6fb77989080ca3b1e3632188d2d7956a6d8b71808787e62'
PARENT_IMAGE = 'b7324639bade3312981d1cbf9c2491b9fcff04fb6c3f808e9c5e1a7c2ca88f08'
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
RUNTIME_LINKER = '0c5176e3aa968631ae592b887c5dc3cf1e8e3f48f73593f076e9d7965ffacfd7'
FLAT = 'c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6'
PROFILE = '79ec4de7f4ffe1640e92eac91f9f52be9b0b10d1085c50c2175a36cde1712c15'
MANIFESTS = (
    ('validation/mixed-integer-autograd-evidence/handoff-manifest.json', '9fbb13cc0637057b5de90d8554cde7df1397710879db2a9302d8bd30fb1451c8', 118),
    ('tools/mixed_integer_autograd/package-manifest.json', '8e8f775ea42fbec4898af318bc0f51be6dcb1510a1eff13b96e2e8f74cf6daca', 85),
    ('validation/saved-public-lifetime/package-manifest.json', '144b0cfc7e11787581614403cbc2b3e643d4650846582440a10c676f9e072661', 17),
    ('validation/mixed-integer-owner-review/package-manifest.json', '4c2ac36f0e96390f47ef29a2d7361de4747736037c006d6a9fc98d5b8d289040', 55),
    ('tools/integer_tensors/package-manifest.json', 'ece14e766825fe0843dfe624fdab8a3b4ec407d27f7a0b5e155e2ee15ee77b71', 54),
    ('tools/native_dropout/package-manifest.json', '3e087c9ddeae0e34f29120ee9e05632e3048aaf5d17b449330faaaed15633708', 99),
    ('validation/int64-inplace-review/package-manifest.json', '60db4e5b20cb3f2ad83383fb4ed6c47a10741a5de9032a0b7db815197cd01e33', 14),
    ('tools/gradient_adam_distributions/package-manifest.json', '3e8c80053365f7e5de1d24725c511f49f256e8437f5a6d3556f09ff07682dc69', 45),
    ('tools/retained_gradients/package-manifest.json', 'd91e5fc032d6a38c262a09b73813668b5b087905d8d4ec27643c04546072dcad', 13),
)
PRODUCER_STEPS = ('cold-native-build', 'fixed-mixed-actual36', 'inherited-I64-eight',
                  'legacy523-twelve', 'fixed-saved-public14')
OUTER_STEPS = ('producer5', 'source-custody', 'owner4-copied17', 'resources6',
               'owned-entry5', 'Saved-entry6', 'typed-handoff2')
CHECKS = ('python_matches', 'platform_matches', 'version_backend_matches',
          'git_matches_oracle_build_pin', 'distribution_version_matches',
          'direct_url_matches', 'direct_url_archive_sha256_matches',
          'pip_report_url_matches', 'pip_report_archive_sha256_matches',
          'pip_report_version_matches', 'pip_report_requested')
INTEGER_GATES = (
    ('storage12', {'cases': 12, 'passed': 12, 'failed': 0, 'frames': 235}),
    ('arithmetic30', {'cases': 30, 'passed': 30, 'failed': 0, 'frames': 1122}),
    ('dropout12', {'cases': 12, 'passed': 12, 'failed': 0, 'frames': 333}),
    ('alias8', {'cases': 8, 'passed': 8, 'native_frames': 516, 'exact_checks': 2792}),
    ('RNE1237', {'cases': 5, 'integers': 1237, 'native_frames': 60, 'exact_checks': 1282}),
    ('promotion20', {'cases': 20, 'passed': 20, 'original_mixed_operations': 40,
                     'native_frames': 2280, 'exact_checks': 10580}),
    ('view-base6', {'cases': 6, 'passed': 6, 'native_frames': 450,
                   'exact_checks': 2046, 'native_role_checks': 168}),
    ('inplace40', {'cases': 40, 'passed': 40}),
)
LEGACY_GATES = (
    ('typed-slice', 'PASS', {'frames': 48}),
    ('typed-lifetimes', 'PASS', {'frames': 343}),
    ('bool-logical-not', 'PASS', {'frames': 66}),
    ('dtype-admission101', 'PASS', {'frames': 126}),
    ('owned-allocation', 'PASS', {'frames': 120}),
    ('history-mul-controls8', 'PASS', {'cases': 8, 'passed': 8, 'frames': 216}),
    ('actual-native-dropout28', 'PASS', {'cases': 28, 'frames': 1816}),
    ('original-native-dropout8', 'PASS', {'cases': 8, 'passed': 8, 'frames': 140}),
    ('actual-existing-optimizer6', 'PASS', {'cases': 6, 'passed': 6, 'frames': 2114}),
    ('actual-sgd-effective-layout6', 'PASS', {'cases': 6, 'passed': 6, 'frames': 891}),
    ('actual-retained12', 'QUALIFIED_EXPOSED_SCOPE',
     {'variants_executed': 12, 'variants_qualified': 12, 'unexpected_failures': 0}),
    ('inherited16', 'PASS', {'step_count': 16, 'all_sources_unchanged': True}),
)
INHERITED_NAMES = ('original-public-ten', 'incoming-25', 'alias-edge-12',
    'bernoulli-gradient-layout-8', 'native-preserve-storage-4', 'native-private-lifetimes-5',
    'typed-gradient-references', 'signed-zero-13', 'partial-16', 'presence-8',
    'gradient-clear-10', 'distribution-atomic-183', 'distribution-explicit-183',
    'original-optimizer-24-mapping', 'owned-adam-cases21-24', 'optimizer-gradient-case20')


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


def loads(raw):
    return json.loads(raw, object_pairs_hook=unique)


def decode(path):
    return loads(Path(path).read_bytes())


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def relative(root, name):
    path = PurePosixPath(name)
    require(name and not path.is_absolute() and '..' not in path.parts
            and str(path) == name and name != '.', ('unsafe relative path', name))
    return root / name


def fields(value, expected, label):
    for key, wanted in expected.items():
        require(key in value and type(value[key]) is type(wanted) and value[key] == wanted,
                ('exact field', label, key, value.get(key), wanted))


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
        custody.bind(relative(repo, row['path']), row['sha256'], row['bytes'], row.get('mode'))
    return {'path': name, 'sha256': pin, 'members': count,
            'declared_modes': sum('mode' in row for row in rows),
            'all_declared_bytes_modes_lengths_verified': True,
            'all_actual_modes_captured_before_execution': True}


def runtime_identity(value):
    fields(value, {'source_sha256': SOURCE, 'native_sha256': IMAGE}, 'mixed runtime')


def build_identity(directory, profile, custody, parent=False):
    value = custody.read(directory / 'build.json')
    fields(value, {'status': 'PASS', 'link_exit': 0}, 'native build')
    for key in ('source_sha256', 'native_sha256', 'functions', 'blocks', 'instructions',
                'descriptor_bytes', 'state_format', 'backing_format', 'units'):
        require(value[key] == profile[key], ('exact native profile', key))
    fields(value, {'compiler_sha256': COMPILER, 'linker_sha256': LINKER}, 'native tools')
    require(len(value['units']) == 11 == len({row['name'] for row in value['units']}),
            'eleven actual independent units')
    custody.bind(directory / 'runtime.hepta', PARENT_SOURCE if parent else SOURCE)
    custody.bind(directory / ('tensor-integer-session' if parent else 'tensor-mixed-autograd-session'),
                 PARENT_IMAGE if parent else IMAGE)
    for row in value['units']:
        require(Path(row['name']).name == row['name'] and row['compiler_exit'] == 0,
                'actual module compilation')
        custody.bind(directory / (row['name'] + '.hepta'), row['source_sha256'])
        custody.bind(directory / (row['name'] + '.o'), row['object_sha256'])
    return {'build_report_sha256': digest(directory / 'build.json'),
            'source_sha256': value['source_sha256'], 'native_sha256': value['native_sha256'],
            'actual_units_verified': 11}


def open_raw(path_or_bytes):
    return io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).open('rb')


def archive_members(path_or_bytes, wanted):
    found = {}
    with open_raw(path_or_bytes) as raw, lzma.LZMAFile(raw, 'rb') as source, \
            tarfile.open(fileobj=source, mode='r|') as tar:
        for member in tar:
            if member.name in wanted:
                require(member.name not in found and member.isfile(), 'unique readable archive member')
                with tar.extractfile(member) as stream:
                    found[member.name] = stream.read()
    require(set(found) == set(wanted), ('archive reads complete', set(wanted) - set(found)))
    return found


def verify_archive(path_or_bytes, files, recurse=True):
    """Stream-check exact members and each nested tar's complete single XZ EOF."""
    require(isinstance(files, dict) and files, 'nonempty declared archive inventory')
    decoder = lzma.LZMADecompressor()
    tar_hash, tar_bytes = hashlib.sha256(), 0
    with open_raw(path_or_bytes) as source:
        while not decoder.eof:
            block = source.read(1 << 20) if decoder.needs_input else b''
            require(block or not decoder.needs_input, 'complete XZ EOF')
            raw = decoder.decompress(block, max_length=1 << 20)
            tar_hash.update(raw)
            tar_bytes += len(raw)
        require(not decoder.unused_data and not source.read(1), 'trailing or concatenated XZ data')
    seen = set()
    with open_raw(path_or_bytes) as raw, lzma.LZMAFile(raw, 'rb') as source, \
            tarfile.open(fileobj=source, mode='r|') as tar:
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
    nested = []
    if recurse:
        for name in sorted(n for n in files if n.endswith('.tar.xz')):
            manifest_name = name[:-7] + '.manifest.json'
            require(manifest_name in files, ('nested archive has exact manifest', name))
            pair = archive_members(path_or_bytes, {name, manifest_name})
            receipt = loads(pair[manifest_name])
            fields(receipt, {'archive': PurePosixPath(name).name,
                            'archive_sha256': files[name]['sha256'],
                            'archive_bytes': files[name]['bytes']}, 'nested archive binding')
            require(receipt['member_count'] == len(receipt['members']) and
                    receipt['original_bytes'] == sum(x['bytes'] for x in receipt['members'].values()),
                    'nested declared count and length')
            checked = verify_archive(pair[name], receipt['members'])
            nested.append(dict(checked, path=name, manifest_sha256=files[manifest_name]['sha256'],
                               archive_sha256=files[name]['sha256']))
    return {'members': len(seen), 'decompressed_tar_bytes': tar_bytes,
            'decompressed_tar_sha256': tar_hash.hexdigest(), 'single_XZ_stream_EOF': True,
            'all_member_bytes_modes_sizes_verified': True, 'nested_archives': nested}


def reference_identity(repo, profile, custody):
    result = []
    for role, spec in (('mixed36', profile['fixed_reference']),
                       ('Saved14', profile['independent_saved_lifetime14'])):
        path = spec['path'] if role == 'mixed36' else spec['reference_path']
        pin = spec['sha256'] if role == 'mixed36' else spec['reference_sha256']
        value = custody.read(relative(repo, path), pin)
        custody.bind(relative(repo, spec['probe_path']), spec['probe_sha256'])
        fields(value, {'status': 'OBSERVED', 'cases': spec['cases'], 'probe_error_count': 0,
                      'source_files_unchanged': True, 'probe_sha256': spec['probe_sha256']}, role)
        provenance = value['oracle_provenance']
        fields(provenance, {'status': 'PASS'}, role + ' actual provenance')
        require(set(provenance['checks']) == set(CHECKS) and
                all(provenance['checks'][key] is True for key in CHECKS), 'all eleven exact reference checks')
        result.append({'role': role, 'reference': path, 'sha256': pin,
                       'probe_sha256': spec['probe_sha256'], 'cases': spec['cases'],
                       'all_eleven_original_provenance_checks_true': True,
                       'live_Torch_executed_this_invocation': False})
    return result


def bind_report_inputs(value, custody, out):
    for key in ('inputs_sha256', 'own_inputs_sha256', 'inputs', 'checked_inputs'):
        entries = value.get(key, {})
        if not isinstance(entries, dict):
            continue
        for name, pin in entries.items():
            path = Path(name)
            if not path.is_absolute():
                continue  # Named roles are checked by their original frozen caller.
            resolved = path.resolve()
            require(resolved in custody.files or out in resolved.parents,
                    ('external input was not captured before execution', key, name))
            if isinstance(pin, str):
                custody.bind(path, pin)
            else:
                require(isinstance(pin, dict) and 'sha256' in pin, ('input digest schema', key, name))
                custody.bind(path, pin['sha256'], pin.get('bytes'), pin.get('mode'))


def child_report(root, row, custody, out, expected_status, counts, path_key='report'):
    require(row['exit_code'] == 0, ('original child process exit', row['name']))
    status = row.get('status', row.get('report_status'))
    require(status == expected_status, ('original child bounded status', row['name'], status))
    for suffix in ('stdout', 'stderr'):
        path = root / (row['name'] + '.' + suffix)
        custody.bind(path, row.get(suffix + '_sha256'))
    child = custody.read(relative(root, row[path_key]), row['report_sha256'])
    fields(child, dict(counts, status=expected_status), row['name'])
    bind_report_inputs(child, custody, out)
    return child


def producer_identity(root, repo, profile, custody, out):
    value = custody.read(root / 'report.json')
    runtime_identity(value)
    fields(value, {'schema': 'heptatorch.mixed-integer-autograd-portable-qualification.v1',
        'status': 'PASS', 'step_count': 5, 'profile_sha256': PROFILE,
        'package_manifest_sha256': MANIFESTS[1][1], 'native_cold_rebuild_executed': True,
        'pinned_Torch_executed_here': False, 'all_package_members_unchanged': True,
        'all_external_package_members_unchanged': True,
        'all_named_gate_statuses_and_counts_matched': True}, 'fresh producer five')
    require(tuple(row['name'] for row in value['steps']) == PRODUCER_STEPS, 'exact producer gate order')
    require(value['fixed_reference'] == profile['fixed_reference'] and
            value['independent_saved_lifetime14'] == profile['independent_saved_lifetime14'] and
            value['external_manifest_pins'] == profile['external_manifests'], 'original reference and source plans')
    archives, children = [], {}
    for row in value['steps']:
        fields(row, {'process_started': True, 'process_completed': True, 'exit_code': 0}, row['name'])
        child = child_report(root, row, custody, out, 'PASS', {})
        runtime_identity(child)
        children[row['name']] = child
        if row['name'] == 'cold-native-build':
            fields(child, {'native_build_process_started': True, 'native_build_process_completed': True,
                'existing_mixed_image_reused': False, 'units_reproduced_exactly': 11,
                'all_package_members_unchanged': True, 'exit_code': 0,
                'profile_sha256': PROFILE, 'package_manifest_sha256': MANIFESTS[1][1]}, 'actual cold native build')
            custody.bind(root / 'native/build.json', child['actual_build_report_sha256'])
            for suffix in ('stdout', 'stderr'):
                custody.bind(root / ('native/builder.' + suffix), child[suffix + '_sha256'])
        else:
            pin = row['archive']
            archive = relative(root / 'evidence', pin['archive'])
            custody.bind(archive, pin['archive_sha256'], pin['archive_bytes'])
            receipt = custody.read(root / 'evidence' / (row['name'] + '.manifest.json'))
            fields(receipt, {key: pin[key] for key in pin}, 'producer archive receipt')
            require(receipt['member_count'] == len(receipt['members']) and
                    receipt['original_bytes'] == sum(x['bytes'] for x in receipt['members'].values()),
                    'producer archive complete declared inventory')
            checked = verify_archive(archive, receipt['members'])
            archives.append(dict(checked, gate=row['name'], archive_sha256=pin['archive_sha256']))
            fields(child, {'all_inputs_unchanged': True}, row['name'])
    for name, spec, counts in (
        ('fixed-mixed-actual36', profile['fixed_reference'],
         {'cases': 36, 'passed': 36, 'failed': 0, 'frames': 4728, 'checks': 21477,
          'exact_words': 4670, 'rollback_bytes': 58018992, 'rng_bytes': 940416,
          'reference_indices': list(range(36))}),
        ('fixed-saved-public14', profile['independent_saved_lifetime14'],
         {'cases': 14, 'passed': 14, 'failed': 0, 'frames': 2058, 'checks': 8842,
          'exact_words': 2166, 'rollback_bytes': 27261936, 'rng_bytes': 480320,
          'reference_indices': list(range(14))})):
        child = children[name]
        fields(child, counts | {'reference_sha256': spec.get('sha256', spec.get('reference_sha256')),
                               'probe_sha256': spec['probe_sha256'],
                               'native_executed': True, 'Torch_executed_here': False}, name)
        require(set(child['provenance_checks']) == set(CHECKS) and
                all(child['provenance_checks'][key] is True for key in CHECKS), name + ' original eleven checks')
    integer = children['inherited-I64-eight']
    require(tuple(row['name'] for row in integer['steps']) == tuple(x[0] for x in INTEGER_GATES), 'original eight I64 gates')
    for row, (name, counts) in zip(integer['steps'], INTEGER_GATES):
        current = dict(row, report=name + '/report.json')
        child_report(root / 'inherited-I64-eight', current, custody, out, 'PASS', counts)
        if name != 'inplace40':
            require(row['expected_counts'] == counts, ('unchanged I64 expected counts', name))
    legacy = children['legacy523-twelve']
    fields(legacy, {'step_count': 12, 'existing_native_units_verified': 11,
        'all_expected_statuses_and_counts_matched': True, 'cold_build_executed_by_this_driver': False}, 'legacy twelve')
    require(tuple(row['name'] for row in legacy['steps']) == tuple(x[0] for x in LEGACY_GATES), 'legacy twelve order')
    for row, (name, status, counts) in zip(legacy['steps'], LEGACY_GATES):
        child = child_report(root / 'legacy523-twelve', row, custody, out, status, counts)
        if name == 'actual-retained12':
            require(child['inputs']['observations'] == 'ca8b5d3a7c679dcc5ad9a3d7b2bbd38f664c90e6474cf3e3c28496d5b716d1b3',
                    'original retained12 actual reference')
            require(set(child['reference_provenance_checks']) == set(CHECKS) and
                    all(child['reference_provenance_checks'][key] is True for key in CHECKS), 'retained12 eleven checks')
        if name == 'inherited16':
            require(tuple(x['name'] for x in child['steps']) == INHERITED_NAMES, 'all sixteen original suite identities')
            fields(child, {'native_build_reproduced_exactly': False, 'existing_native_build_receipt_verified': True}, 'inherited16 scope')
            for inner in child['steps']:
                original = child_report(root / 'legacy523-twelve/inherited16', inner, custody, out,
                                        inner['report_status'], {}, path_key='report_path')
                require(original['status'] in ('PASS', 'QUALIFIED_EXPOSED_SCOPE',
                    'QUALIFIED_WITH_EXPLICIT_OPEN_BOUNDARIES', 'QUALIFIED_WITH_OPEN_STATE_LIFECYCLE',
                    'QUALIFIED_WITH_UNREPRESENTED_VARIANTS', 'QUALIFIED_WITH_OPEN_BOUNDARIES'),
                    ('explicit inherited bounded status', inner['name'], original['status']))
    return {'report_sha256': digest(root / 'report.json'), 'steps': 5,
            'actual_cold_build_verified': True, 'raw_archives': archives,
            'original_eight_I64_and_twelve_legacy_gates_verified': True,
            'original_sixteen_inherited_reports_and_bounded_statuses_preserved': True}


def archive_evidence(out):
    excluded = {'report.json', 'evidence-manifest.json', 'raw-evidence.tar.xz'}
    files = {}
    for path in sorted(out.rglob('*')):
        require(not path.is_symlink(), ('evidence symlink', str(path)))
        if path.is_file() and path.relative_to(out).as_posix() not in excluded:
            files[path.relative_to(out).as_posix()] = {'bytes': path.stat().st_size,
                'mode': stat.S_IMODE(path.stat().st_mode), 'sha256': digest(path)}
    archive = out / 'raw-evidence.tar.xz'
    with lzma.open(archive, 'wb', preset=3) as stream, tarfile.open(fileobj=stream, mode='w|') as tar:
        for name, pin in files.items():
            original = relative(out, name)
            require(digest(original) == pin['sha256'] and original.stat().st_size == pin['bytes'] and
                    stat.S_IMODE(original.stat().st_mode) == pin['mode'], 'archive input unchanged')
            member = tarfile.TarInfo(name)
            member.size, member.mode, member.mtime = pin['bytes'], pin['mode'], 0
            with original.open('rb') as source:
                tar.addfile(member, source)
    verified = verify_archive(archive, files, recurse=False)
    value = dict(verified, schema='heptatorch.mixed-integer-autograd-fresh-raw-evidence.v1', status='VERIFIED',
                 source_sha256=SOURCE, native_sha256=IMAGE, archive_sha256=digest(archive),
                 archive_bytes=archive.stat().st_size, uncompressed_bytes=sum(v['bytes'] for v in files.values()),
                 files=files, self_reference_exclusions=sorted(excluded),
                 scope='Every current output file except the three root self-references; nested producer archives independently verified.')
    dump(out / 'evidence-manifest.json', value)
    return {key: item for key, item in value.items() if key != 'files'} | {
        'archive': archive.name, 'manifest': 'evidence-manifest.json',
        'manifest_sha256': digest(out / 'evidence-manifest.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'parent-build', 'upstream', 'compiler', 'linker', 'flat-compiler', 'stack-build', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    repo, parent, stack, out = args.repo.resolve(), args.parent_build.resolve(), args.stack_build.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    custody = Custody()
    report = {'schema': 'heptatorch.mixed-integer-autograd-native-integration.v1', 'status': 'RUNNING',
              'source_sha256': SOURCE, 'native_sha256': IMAGE, 'producer_mode': 'FRESH_COLD_REQUIRED',
              'producer_executed_this_invocation': False, 'producer_completed_this_invocation': False,
              'producer_reused': False, 'Torch_executed_here': False, 'commands': [],
              'whole_pytorch_gate': 'HOLD', 'full_pytorch_rewrite_complete': False,
              'pytorch_api_denominator': 2590, 'closed_schema_count': 0,
              'counting_boundary': 'Named suites overlap. No sum is a unique upstream test or schema count.'}

    def save():
        dump(out / 'report.json', report)

    def execute(name, script, arguments, expected, expected_status='PASS'):
        receipt = out / name / 'report.json'
        require(not (out / name).exists(), 'fresh child output required')
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
                if name == 'producer5':
                    report['producer_executed_this_invocation'] = True
                save()
                try:
                    process.wait(timeout=3000)
                except BaseException:
                    process.kill()
                    process.wait()
                    row.update(exit_code=process.returncode, process_reaped_after_interrupt=True)
                    raise
                row.update(process_completed=True, exit_code=process.returncode)
                if name == 'producer5':
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
        require(process.returncode == 0 and value.get('status') == expected_status, ('actual gate failure', name))
        custody.bind(receipt, row['report_sha256'])
        runtime_identity(value)
        fields(value, expected, name)
        bind_report_inputs(value, custody, out)
        return value

    save()
    try:
        custody.bind(Path(__file__))
        report['manifests'] = [manifest(repo, *row, custody) for row in MANIFESTS]
        profile = custody.read(repo / 'tools/mixed_integer_autograd/profile.json', PROFILE)
        parent_profile = custody.read(repo / 'tools/integer_tensors/profile.json')
        runtime_identity(profile)
        fields(parent_profile, {'source_sha256': PARENT_SOURCE, 'native_sha256': PARENT_IMAGE}, 'parent07')
        custody.bind(args.compiler, COMPILER)
        custody.bind(args.linker, LINKER)
        custody.bind(args.flat_compiler, FLAT)
        require(args.compiler.resolve() == (stack / 'native/stack-module-compiler').resolve() and
                args.linker.resolve() == (stack / 'native/stack-linker-session').resolve(), 'explicit same stack build')
        custody.bind(stack / 'native/stack-linker-runtime', RUNTIME_LINKER)
        report['parent_build'] = build_identity(parent, parent_profile, custody, parent=True)
        custody.bind(parent / 'overlay-authoring.py', '27a1dd831a43dbb1167f078052adf9a836c1be5ce81f077ee17fd85de55cd0bd')
        author = custody.read(repo / 'tools/owned_tensors/profile.json', '69642a730f9da8567a01cc5afd9d15963a49ff1ab83f7761039bb301d0d2d242')
        custody.bind(repo / 'tools/owned_tensors/authoring.py', '2845fc2e3f03ccbadd5ea6c59734d0be6db6477bf8434d288259fab5ee570ba6')
        for name, pin in author['upstream_authoring_sources'].items():
            custody.bind(relative(args.upstream.resolve(), name), pin)
        for name in ('tests/retained_nonleaf_review/protocol.py', 'tests/retained_resource_review/protocol.py'):
            custody.bind(repo / name, '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9')
            custody.bind((repo / name).with_name('owner_client.py'))
        report['fixed_actual_references'] = reference_identity(repo, profile, custody)
        report['inputs_captured_before_execution'] = len(custody.files)
        execute('producer5', repo / 'tools/mixed_integer_autograd/qualify.py', [
            '--parent', parent, '--repository', repo, '--upstream', args.upstream.resolve(),
            '--compiler', args.compiler.resolve(), '--linker', args.linker.resolve(),
            '--flat-compiler', args.flat_compiler.resolve()], {'step_count': 5})
        native = out / 'producer5/native'
        report['new_build'] = build_identity(native, profile, custody)
        report['producer_admission'] = producer_identity(out / 'producer5', repo, profile, custody, out)
        tests = repo / 'tests/mixed_integer_owner_review'
        protocol = repo / 'tests/retained_resource_review/protocol.py'
        reference = repo / 'validation/integer-tensors/reference/torch-int64-tensor-reference.json'
        mixed_reference = relative(repo, profile['fixed_reference']['path'])
        source = execute('source-custody', tests / 'audit_source10.py', [
            '--build', native, '--previous-build', parent, '--protocol', protocol], {
            'functions': 556, 'blocks': 7420, 'instructions': 26889,
            'all_556_core_function_bytes_present_in_real_units': True,
            'unchanged_parent_function_count': 531, 'observer_whole_inverse_exact': True,
            'new_native_execution': False, 'new_Torch_execution': False,
            'previous_source_sha256': PARENT_SOURCE, 'previous_native_sha256': PARENT_IMAGE},
            expected_status='SOURCE_AND_UNIT_CUSTODY_PASS')
        require(len(source['units']) == 11 and len(source['changed_parent_functions']) == 13 and
                len(source['added_functions']) == 12 and len(source['function_bodies']) == 556,
                'complete source and unit audit')
        owner = execute('owner4-copied17', tests / 'test_saved_owner10.py', [
            '--build', native, '--protocol', protocol], {'frames_before_EOF': 58,
            'snapshot_checks': 144, 'isolated_transition_checks': 53, 'ordinary_rollback_bytes': 3495120,
            'checks': 105, 'new_Torch_reference_cases': 0})
        require(len(owner['cases']) == 4 and all(x['status'] == 'PASS' and x['EOF_exit_code'] == 0 and
                x['all_responses_before_stdin_EOF'] is True for x in owner['cases']), 'four actual owner cases and EOF')
        for key in ('frames_before_EOF', 'snapshot_checks', 'isolated_transition_checks', 'ordinary_rollback_bytes', 'checks'):
            require(sum(row[key] for row in owner['cases']) == owner[key], ('owner case sum', key))
        negatives = owner['negative_controls']
        fields(negatives, {'status': 'PASS', 'negative_count': 17, 'new_native_executions': 0,
                          'target_memory_written': False}, 'copied owner refusals')
        frozen_owner = custody.read(repo / 'validation/mixed-integer-owner-review/reports/owner4-copied17.json')
        require(negatives == frozen_owner['negative_controls'], 'all original seventeen exact negative rules')
        require([x['name'] for x in owner['cases']] == [x['name'] for x in frozen_owner['cases']], 'owner recipe order')
        common = ['--build', native, '--protocol', protocol, '--reference', reference,
                  '--expected-source-sha256', SOURCE, '--expected-native-sha256', IMAGE]
        resource = execute('resources6', tests / 'test_integer_resources.py', common + [
            '--mixed-reference', mixed_reference], {'case_count': 6, 'passed': 6, 'failed': 0,
            'frames_before_EOF': 536, 'checks': 162, 'complete_rollback_bytes': 44811376,
            'module_count': 11, 'native_executed': True, 'complete_six_case_plan_selected': True,
            'source_custody_after': True})
        require(len(resource['cases']) == 6 and resource['selected_cases'] == resource['declared_complete_plan'], 'entire six resource plan')
        entry = execute('owned-entry5', tests / 'test_integer_owned_entry.py', common + [
            '--stack-build', stack, '--probe-source', tests / 'entry_probe.hepta'], {
            'case_count': 5, 'passed': 5, 'linked_modules': 12, 'native_probe_processes': 46,
            'interactive_frames': 33, 'complete_owner_bytes_compared': 32155104, 'source_custody_after': True,
            'compiler_sha256': COMPILER, 'runtime_linker_sha256': RUNTIME_LINKER})
        require(len(entry['cases']) == 5, 'five entry recipes')
        saved = execute('Saved-entry6', tests / 'test_saved_entry10.py', [
            '--build', native, '--probe-build', out / 'owned-entry5', '--protocol', protocol], {
            'case_count': 6, 'passed': 6, 'linked_modules': 12, 'native_probe_processes': 14,
            'interactive_frames': 17, 'complete_owner_bytes_compared': 12582432,
            'native_probe_recompiled_this_invocation': False, 'EOF_exit_code': 0, 'source_custody_after': True})
        require(len(saved['cases']) == 6 and saved['real_previous_compile_and_link_receipt_sha256'] ==
                digest(out / 'owned-entry5/report.json') and saved['probe_source_sha256'] == entry['probe_source_sha256'] and
                saved['probe_image_sha256'] == entry['probe_image_sha256'], '107 controls use this invocation real probe')
        handoff = execute('typed-handoff2', tests / 'test_mixed_typed_handoff.py', [
            '--build', native, '--previous-build', parent, '--stack-build', stack,
            '--protocol', protocol, '--controller', tests / 'state_handoff.hepta'], {
            'case_count': 2, 'passed': 2, 'failed': 0, 'linked_modules': 23, 'frame_count': 14,
            'rollback_bytes': 1398048, 'EOF_exit_code': 0, 'source_custody_after': True,
            'previous_source_sha256': PARENT_SOURCE, 'previous_native_sha256': PARENT_IMAGE,
            'compiler_sha256': COMPILER, 'linker_sha256': LINKER})
        require(len(handoff['cases']) == 2 and len(handoff['units']) == 22 and len(handoff['frames']) == 14,
                'both real owned handoff directions and modules')
        require(tuple(row['name'] for row in report['commands']) == OUTER_STEPS, 'all seven actual outer commands')
        report['separate_required_CI_gates'] = {
            'live32': 'NOT_EXECUTED_BY_NATIVE_COORDINATOR',
            'complete_accounted35_and_copied18': 'NOT_EXECUTED_BY_NATIVE_COORDINATOR',
            'no_inference_from_old_profile_or_inherited_suites': True}
        report['counts'] = {'producer_top_level_steps': 5, 'actual_mixed_reference_cases': 36,
            'actual_Saved_public_reference_cases': 14, 'inherited_I64_named_gates': 8,
            'legacy_behavior_gates': 12, 'inherited_named_suites_with_original_OPENs': 16,
            'source_core_functions': 556, 'actual_compiled_units': 11,
            'native_owner_cases': 4, 'copied_new_role_refusals': 17,
            'native_resource_controls': 6, 'native_entry_controls': 5,
            'native_Saved_entry_controls': 6, 'native_typed_directions': 2}
        custody.finish()
        dump(out / 'checked-inputs-before-archive.json', {str(path): value for path, value in custody.files.items()})
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
