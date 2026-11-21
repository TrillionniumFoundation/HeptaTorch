#!/usr/bin/env python3
"""Seal current CI bytes and modes; never grant a numerical qualification."""
import argparse
import hashlib
import io
import importlib.util
import json
import lzma
import re
import stat
import sys
import tarfile
import traceback
from pathlib import Path, PurePosixPath


GROUPS = ('bool27', 'original76', 'integral24', 'sum_f32_17', 'current35',
          'base270', 'sum32', 'live32', 'logical147', 'resource7', 'asstrided16',
          'cat4', 'metadata10', 'generator131', 'public_f64_logical192',
          'f64_48', 'half47', 'half_logical264', 'half_transactions12', 'numeric_predicates', 'bf16_copy', 'half_mul', 'half_neg', 'half_addsub', 'wide_neg')
SOURCE = 'e0c8e0a3c3ee16d11cad8af6be1b52bd56295581e74bdcc65131f71a57540025'
IMAGE = '3ee1fae0e9d20d10dfd2645a3575d23357bcaca0dae03e5f0ff6bf1d4b4197a5'
SOURCE_MANIFEST = '32d826cfaaaaca910c7e2b209d977dd705a314f64267cfd0808f01f08d3865d1'
NUMERIC_CASES = '272c9b4fbde339481c32f2ed071dd8c60c9c3b17732e71e2178d973914c79609'
NUMERIC_PROOF = dict(cases=485, native_sessions=496, frames=11244, assertions=42645,
    owner_snapshots=3615, mapping_snapshots=10, additional_packet_controls=56,
    additional_successful_FE_transactions=5, additional_failed_FE_transactions=5,
    additional_allocation_recoveries=5, physical_replay=True,
    native_executed_by_qualifier=False, Torch_executed_by_qualifier=False,
    host_expected_values_computed=False, old_dispatcher_preserved=True)
BF16_CASES = '71c9d50659b51f5dc28f8838fc5aad0ae0f9b5d53e95c26e016c80c98877e869'
BF16_PROOF = dict(cases=280, native_sessions=280, frames=6088, assertions=23096,
    owner_snapshots=1960, bitwise_elements=1312796, physical_replay=True,
    native_executed_by_qualifier=False, Torch_executed_by_qualifier=False,
    host_expected_values_computed=False, full_project_status='HOLD')
HALF_MUL_CASES = 'd93e12ad6d96b3c0ec77ceb63986711f0105fc2ff9c9b696572ae0a7bd7576a5'
HALF_MUL_PROOF = dict(cases=312, native_sessions=312, frames=9664, assertions=37168,
    owner_snapshots=2868, bitwise_elements=1313606, physical_replay=True,
    native_executed_by_qualifier=False, Torch_executed_by_qualifier=False,
    host_expected_values_computed=False, full_project_status='HOLD',
    cpu_capability='AVX2', requested_cpu_capability='avx2', torch_threads=1,
    original_refusal_successors=[
        'half-binary-left-5-0-3', 'half-binary-right-5-0-3',
        'half-binary-left-5-1-3', 'half-binary-right-5-1-3',
        'half-binary-left-5-2-3', 'half-binary-right-5-2-3',
        'half-binary-left-5-4-3', 'half-binary-right-5-4-3',
        'half-binary-left-5-5-3', 'half-binary-left-5-6-3', 'half-binary-right-5-6-3',
        'half-binary-left-6-0-3', 'half-binary-right-6-0-3',
        'half-binary-left-6-1-3', 'half-binary-right-6-1-3',
        'half-binary-left-6-2-3', 'half-binary-right-6-2-3',
        'half-binary-left-6-4-3', 'half-binary-right-6-4-3',
        'half-binary-left-6-5-3', 'half-binary-right-6-5-3', 'half-binary-left-6-6-3'],
    replaced_arithmetic_refusals=['promotion-5-5-3:operation-1', 'promotion-5-5-3:operation-2',
                                 'promotion-6-6-3:operation-1', 'promotion-6-6-3:operation-2'])
HALF_NEG_CASES = 'cb177901644162f6f127fd4aa6e4dfe0665f7c708d91d5afdd593b52fcd2897c'
HALF_NEG_PROOF = dict(cases=371, native_sessions=371, frames=11377, assertions=37744,
    owner_snapshots=2717, bitwise_elements=272444, mapping_reads=4,
    actual_allocation_refusals=2, identical_request_recoveries=2, protocol_controls=46,
    physical_replay=True, native_executed_by_qualifier=False, Torch_executed_by_qualifier=False,
    host_expected_values_computed=False, full_project_status='HOLD',
    cpu_capability='AVX2', requested_cpu_capability='avx2', torch_threads=1,
    original_refusal_successors=['unsupported-half5-neg', 'unsupported-half6-neg'])
WIDE_NEG_CASES = '22063936e50a4e5da53be543fecc446a0fef529712da140a9afdb863b69da73a'
WIDE_NEG_PROOF = dict(cases=405, native_sessions=405, frames=10484, assertions=37314,
    owner_snapshots=2959, bitwise_elements=287275, mapping_reads=4,
    actual_allocation_refusals=2, identical_request_recoveries=2, protocol_controls=46,
    physical_replay=True, native_executed_by_qualifier=False, Torch_executed_by_qualifier=False,
    host_expected_values_computed=False, full_project_status='HOLD',
    cpu_capability='AVX2', requested_cpu_capability='avx2', torch_threads=1,
    original_refusal_successors=['unsupported-F64-neg'])
HALF_ADDSUB_CASES = '07fee181f5b4ea4eff68ccb98c15fbfaf1b81e4b39f0573b21489681443aeea8'
HALF_ADDSUB_PROOF = dict(cases=1006, native_sessions=1006, frames=31468, assertions=119842,
    owner_snapshots=9166, bitwise_elements=2899312, mapping_reads=12,
    actual_allocation_refusals=4, identical_request_recoveries=4, protocol_controls=70,
    physical_replay=True, native_executed_by_qualifier=False, Torch_executed_by_qualifier=False,
    host_expected_values_computed=False, full_project_status='HOLD',
    cpu_capability='AVX2', requested_cpu_capability='avx2', torch_threads=1,
    original1004_case_sha256='553e1a15e63bc66a60faf1276e827410165c9ed8ff0b05d6fb013421930cc6d8',
    original1004_cases_unchanged=True,
    original_fe_successors=['same-frame-half-FE-binary-5', 'same-frame-half-FE-binary-6'],
    original_refusal_successors=[
        'half-binary-left-5-0-1', 'half-binary-right-5-0-1',
        'half-binary-left-5-0-2', 'half-binary-right-5-0-2',
        'half-binary-left-5-1-1', 'half-binary-right-5-1-1',
        'half-binary-left-5-2-1', 'half-binary-right-5-2-1',
        'half-binary-left-5-2-2', 'half-binary-right-5-2-2',
        'half-binary-left-5-4-1', 'half-binary-right-5-4-1',
        'half-binary-left-5-4-2', 'half-binary-right-5-4-2',
        'half-binary-left-5-5-1', 'half-binary-left-5-5-2',
        'half-binary-left-5-6-1', 'half-binary-right-5-6-1',
        'half-binary-left-5-6-2', 'half-binary-right-5-6-2',
        'half-binary-left-6-0-1', 'half-binary-right-6-0-1',
        'half-binary-left-6-0-2', 'half-binary-right-6-0-2',
        'half-binary-left-6-1-1', 'half-binary-right-6-1-1',
        'half-binary-left-6-2-1', 'half-binary-right-6-2-1',
        'half-binary-left-6-2-2', 'half-binary-right-6-2-2',
        'half-binary-left-6-4-1', 'half-binary-right-6-4-1',
        'half-binary-left-6-4-2', 'half-binary-right-6-4-2',
        'half-binary-left-6-5-1', 'half-binary-right-6-5-1',
        'half-binary-left-6-5-2', 'half-binary-right-6-5-2',
        'half-binary-left-6-6-1', 'half-binary-left-6-6-2'],
    original_bool_subtraction_refusals=['half-binary-left-5-1-2', 'half-binary-right-5-1-2',
                                       'half-binary-left-6-1-2', 'half-binary-right-6-1-2'],
    multiplication_refusal_successors=['promotion-5-5-3:operation-1', 'promotion-5-5-3:operation-2',
                                     'promotion-6-6-3:operation-1', 'promotion-6-6-3:operation-2'])
TOOLCHAINS = ('hepta-compiler', 'hepta-profile-sources', 'hepta-expanded',
              'hepta-capacity-review', 'hepta-modules-sources', 'hepta-modules-native',
              'hepta-owned-modules', 'hepta-owned-session-tools', 'hepta-stack-budget',
              'hepta-current-entropy')
REPOSITORY_PATHS = (
    '.github/workflows/current-runtime.yml', 'Makefile', 'tools/seal_current_ci_evidence.py',
    'tools/current_runtime', 'tools/differentiable_view_recovered_r3/profile.json',
    'tools/stack_budget/inspect_stack.py',
    'validation/view-reshape-r4-native76/archive-location.json',
    'build/pytorch-install.json', 'tests', 'tests/numeric_predicates', 'tests/bf16_copy', 'tests/half_mul', 'tests/half_neg', 'tests/half_addsub', 'tests/wide_neg', 'compatibility',
    'continuation/20261117/evidence',
    'continuation/20261117/source-packages/sum-bool-source-01/hia',
    'validation/sum-bool27-reference', 'validation/current-runtime-references',
    'validation/current-runtime76-reference', 'validation/current-runtime-baseline',
    'validation/sum-f32-reference17', 'validation/current-runtime270-source',
    'validation/current-runtime32-source', 'validation/tensor-to-original16-recovery',
    'validation/logical-reductions-reference', 'validation/bool-dtype-extension-reference',
    'validation/cat-metadata10-reference', 'validation/current-generator-original26',
    'validation/current-generator-candidate', 'validation/current-generator131-source',
    'validation/public-f64-logical192-reference',
    'validation/current-f64-48-source', 'validation/current-f64-48-reference',
    'validation/current-half47-source', 'validation/current-half47-reference',
    'validation/current-runtime767/qualification-plan.json',
    'validation/half-logical264-reference',
)
REFERENCE_INTERPRETER_PACKAGES = (
    'validation/cat-metadata10-reference', 'validation/public-f64-logical192-reference',
)
NUMERIC_REFERENCE_OUTPUTS = (
    'review/numeric_predicates/reference/report.json',
    'review/numeric_predicates/reference-execution/process.json',
    'review/numeric_predicates/reference-execution/runner.json',
    'review/numeric_predicates/reference-execution/runner.stdout',
    'review/numeric_predicates/reference-execution/runner.stderr',
    'review/numeric_predicates-execution/intent.json',
    'review/numeric_predicates-execution/runner.json',
    'review/numeric_predicates-execution/runner.stdout',
    'review/numeric_predicates-execution/runner.stderr',
)
REQUIRED_OUTPUTS = ('build/build.json', 'build/heptatorch', 'build/runtime.hepta', 'build/manifest.json',
                    'qualification.json', 'review/qualification.json') + tuple(
                        'review/' + name + '/report.json' for name in GROUPS) + (
                        'review/f64_48/physical-qualification.json',
                        'review/half47/physical-qualification.json',
                        'review/half_logical264/physical-qualification.json',
        'review/half_transactions12/physical-qualification.json',
        'review/numeric_predicates/physical-qualification.json') + NUMERIC_REFERENCE_OUTPUTS
REQUIRED_OUTPUTS += (
    'review/bf16_copy/physical-qualification.json', 'review/bf16_copy/reference/report.json',
    'review/bf16_copy/reference-execution/process.json', 'review/bf16_copy/reference-execution/runner.json',
    'review/bf16_copy/reference-execution/runner.stdout', 'review/bf16_copy/reference-execution/runner.stderr',
    'review/bf16_copy-execution/intent.json', 'review/bf16_copy-execution/runner.json',
    'review/bf16_copy-execution/runner.stdout', 'review/bf16_copy-execution/runner.stderr',
)
REQUIRED_OUTPUTS += (
    'review/half_mul/physical-qualification.json', 'review/references/half-mul312/report.json',
    'review/references/half-mul312-execution/process.json', 'review/references/half-mul312-execution/runner.json',
    'review/references/half-mul312-execution/runner.stdout', 'review/references/half-mul312-execution/runner.stderr',
    'review/half_mul-execution/intent.json', 'review/half_mul-execution/runner.json',
    'review/half_mul-execution/runner.stdout', 'review/half_mul-execution/runner.stderr',
)
REQUIRED_OUTPUTS += (
    'review/half_neg/physical-qualification.json', 'review/references/half-neg371/report.json',
    'review/references/half-neg371-execution/process.json', 'review/references/half-neg371-execution/runner.json',
    'review/references/half-neg371-execution/runner.stdout', 'review/references/half-neg371-execution/runner.stderr',
    'review/half_neg-execution/intent.json', 'review/half_neg-execution/runner.json',
    'review/half_neg-execution/runner.stdout', 'review/half_neg-execution/runner.stderr',
)
REQUIRED_OUTPUTS += (
    'review/wide_neg/physical-qualification.json', 'review/references/wide-neg405/report.json',
    'review/references/wide-neg405-execution/process.json', 'review/references/wide-neg405-execution/runner.json',
    'review/references/wide-neg405-execution/runner.stdout', 'review/references/wide-neg405-execution/runner.stderr',
    'review/wide_neg-execution/intent.json', 'review/wide_neg-execution/runner.json',
    'review/wide_neg-execution/runner.stdout', 'review/wide_neg-execution/runner.stderr',
)
REQUIRED_OUTPUTS += (
    'review/half_addsub/physical-qualification.json', 'review/references/half-addsub1006/report.json',
    'review/references/half-addsub1006-execution/process.json', 'review/references/half-addsub1006-execution/runner.json',
    'review/references/half-addsub1006-execution/runner.stdout', 'review/references/half-addsub1006-execution/runner.stderr',
    'review/half_addsub-execution/intent.json', 'review/half_addsub-execution/runner.json',
    'review/half_addsub-execution/runner.stdout', 'review/half_addsub-execution/runner.stderr',
)
INPUT_REPORTS = ('build/build.json', 'review/qualification.json',) + tuple(
    'review/' + name + '/report.json' for name in GROUPS) + (
        'review/base270/current-adapter.json', 'review/base270/cold-build-admission.json',
        'review/public_f64_logical192/native/report.json',
        'review/f64_48/physical-qualification.json',
        'review/half47/physical-qualification.json',
        'review/half_logical264/physical-qualification.json',
        'review/half_transactions12/physical-qualification.json',
        'review/numeric_predicates/physical-qualification.json',
        'review/numeric_predicates/reference/report.json',
        'review/numeric_predicates/reference-execution/process.json',
        'review/numeric_predicates/reference-execution/runner.json',
        'review/numeric_predicates-execution/intent.json',
        'review/numeric_predicates-execution/runner.json')
INPUT_REPORTS += ('review/bf16_copy/physical-qualification.json',
                 'review/bf16_copy/reference/report.json',
                 'review/bf16_copy/reference-execution/process.json',
                 'review/bf16_copy/reference-execution/runner.json',
                 'review/bf16_copy-execution/intent.json',
                 'review/bf16_copy-execution/runner.json')
INPUT_REPORTS += ('review/half_mul/physical-qualification.json',
                 'review/references/half-mul312/report.json',
                 'review/references/half-mul312-execution/process.json',
                 'review/references/half-mul312-execution/runner.json',
                 'review/half_mul-execution/intent.json',
                 'review/half_mul-execution/runner.json')
INPUT_REPORTS += ('review/half_neg/physical-qualification.json',
                 'review/references/half-neg371/report.json',
                 'review/references/half-neg371-execution/process.json',
                 'review/references/half-neg371-execution/runner.json',
                 'review/half_neg-execution/intent.json',
                 'review/half_neg-execution/runner.json')
INPUT_REPORTS += ('review/wide_neg/physical-qualification.json',
                 'review/references/wide-neg405/report.json',
                 'review/references/wide-neg405-execution/process.json',
                 'review/references/wide-neg405-execution/runner.json',
                 'review/wide_neg-execution/intent.json',
                 'review/wide_neg-execution/runner.json')
INPUT_REPORTS += ('review/half_addsub/physical-qualification.json',
                 'review/references/half-addsub1006/report.json',
                 'review/references/half-addsub1006-execution/process.json',
                 'review/references/half-addsub1006-execution/runner.json',
                 'review/half_addsub-execution/intent.json',
                 'review/half_addsub-execution/runner.json')
INPUT_TABLES = ('checked_inputs', 'inputs', 'files', 'source_custody')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, obj):
    with Path(path).open('xb') as stream:
        stream.write((json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def info(path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError(('regular file required', str(path)))
    raw = path.read_bytes()
    after = path.stat()
    attrs = ('st_dev', 'st_ino', 'st_size', 'st_mode', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, key) != getattr(after, key) for key in attrs):
        raise ValueError(('file changed', str(path)))
    return raw, {'bytes': len(raw), 'sha256': digest(raw), 'mode': stat.S_IMODE(before.st_mode)}


def matches(actual, expected):
    return all(actual[key] == value for key, value in expected.items())


def verify_archive(archive, members):
    """Repeat the original R4 single-XZ, member, mode and zero-tail checks."""
    _, archive_meta = info(archive)
    decoder = lzma.LZMADecompressor()
    with archive.open('rb') as stream:
        while not decoder.eof:
            piece = stream.read(65536) if decoder.needs_input else b''
            if not piece and decoder.needs_input:
                raise ValueError('truncated XZ')
            decoder.decompress(piece, max_length=1048576)
        if decoder.unused_data or stream.read(1):
            raise ValueError('trailing or concatenated XZ')
    with tarfile.open(archive, 'r:xz') as bundle:
        rows = bundle.getmembers()
        if (len(rows) != len(members) or len({row.name for row in rows}) != len(members)
                or {row.name for row in rows} != set(members)):
            raise ValueError('archive member denominator')
        for row in rows:
            if not row.isfile():
                raise ValueError(('nonregular archive member', row.name))
            raw = bundle.extractfile(row).read()
            expected = members[row.name]
            if (len(raw) != row.size or len(raw) != expected['bytes']
                    or digest(raw) != expected['sha256'] or row.mode != expected['mode']):
                raise ValueError(('archived bytes or mode', row.name))
        end = max(row.offset_data + ((row.size + 511) // 512) * 512 for row in rows)
        bundle.fileobj.seek(end)
        tail = bundle.fileobj.read()
        if len(tail) < 1024 or any(tail):
            raise ValueError('TAR zero end padding')
    if info(archive)[1] != archive_meta:
        raise ValueError('archive changed during independent readback')
    return archive_meta


def seal_archive(archive, selected, declared, parsed_lists):
    """Read original bytes, archive them once, then independently read both copies."""
    members, errors = {}, []
    source_paths = {str(path) for path in selected.values()}
    for path in sorted((set(declared) | set(parsed_lists)) - source_paths):
        errors.append(('declared-input-not-selected', path))
    for path, expected in parsed_lists.items():
        try:
            if info(Path(path))[1] != expected:
                raise ValueError(('parsed input list changed before archive', path))
        except BaseException as error:
            errors.append(('parsed-input-list', path, repr(error)))
    with tarfile.open(archive, 'x:xz', format=tarfile.PAX_FORMAT, preset=1) as bundle:
        for name, path in sorted(selected.items()):
            try:
                raw, meta = info(path)
            except BaseException as error:
                errors.append(('archive-read', str(path), repr(error)))
                continue
            if str(path) in parsed_lists and meta != parsed_lists[str(path)]:
                errors.append(('input-list-at-archive', str(path)))
            for expected in declared.get(str(path), []):
                if not matches(meta, expected):
                    errors.append(('input-at-archive', str(path), meta, expected))
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode, entry.mtime = len(raw), meta['mode'], 0
            bundle.addfile(entry, io.BytesIO(raw))
            members[name] = {**meta, 'source_path': str(path)}
    if not members:
        raise ValueError('no readable actual evidence was archived')
    archive_meta = verify_archive(archive, members)
    for name, archived in members.items():
        try:
            _, current = info(selected[name])
            if current != {key: archived[key] for key in ('bytes', 'sha256', 'mode')}:
                raise ValueError(('evidence changed', name))
            for expected in declared.get(str(selected[name]), []):
                if not matches(current, expected):
                    raise ValueError(('declared input changed', name))
        except BaseException as error:
            errors.append(('post-archive-custody', str(selected[name]), repr(error)))
    return members, archive_meta, errors


def reference_interpreter_role(repo, relative):
    """Read a historical Python entity inside its original package; never execute it."""
    root = repo / relative
    manifest_path, archive = root / 'manifest.json', root / 'complete.tar.xz'
    manifest_raw, manifest_meta = info(manifest_path)
    manifest = json.loads(manifest_raw)
    schemas = {
        'validation/cat-metadata10-reference': 'heptatorch.cat-metadata10.reference-package.v1',
        'validation/public-f64-logical192-reference': 'heptatorch.public-f64-logical192.reference-package.v1',
    }
    if relative not in schemas or manifest.get('schema') != schemas[relative]:
        raise ValueError(('unsupported fixed historical reference schema', relative))
    member = manifest['roles']['python']
    if member != 'interpreter/python3.12':
        raise ValueError(('unsupported fixed historical Python member', relative))
    safe = PurePosixPath(member)
    if (not isinstance(member, str) or safe.is_absolute() or '..' in safe.parts
            or safe.as_posix() != member or '\\' in member):
        raise ValueError('unsafe historical Python member')
    expected = manifest['files'][member]
    archive_declaration = manifest['archive']
    if (not isinstance(archive_declaration, dict)
            or set(archive_declaration) != {'path', 'bytes', 'sha256'}
            or archive_declaration['path'] != 'complete.tar.xz'):
        raise ValueError(('unsupported fixed reference archive metadata', relative))
    if not isinstance(expected, dict) or set(expected) != {'bytes', 'sha256', 'mode'}:
        raise ValueError(('unsupported fixed historical Python metadata', relative))
    for declaration in (archive_declaration, expected):
        if (type(declaration['bytes']) is not int or declaration['bytes'] < 0
                or not isinstance(declaration['sha256'], str)
                or re.fullmatch('[0-9a-f]{64}', declaration['sha256']) is None):
            raise ValueError(('invalid historical byte extent or digest', relative))
    if type(expected['mode']) is not int or not 0 <= expected['mode'] <= 0o7777:
        raise ValueError(('historical mode must be integer POSIX permission bits', relative))
    archive_expected = {key: archive_declaration[key] for key in ('bytes', 'sha256')}
    path_map = manifest['input_path_map']
    if (not isinstance(path_map, dict) or any(not isinstance(path, str) or not isinstance(mapped, str)
                                              for path, mapped in path_map.items())):
        raise ValueError(('historical input path map required', relative))
    mapped_python_paths = sorted(path for path, mapped in path_map.items() if mapped == member)
    if relative == 'validation/cat-metadata10-reference':
        original_paths = mapped_python_paths
        provenance_field = 'input_path_map'
    else:
        if mapped_python_paths:
            raise ValueError(('public192 Python provenance is separate from its six report inputs', relative))
        original_paths = [manifest['python_original_resolved_path']]
        provenance_field = 'python_original_resolved_path'
    if (len(original_paths) != 1 or not isinstance(original_paths[0], str)
            or not PurePosixPath(original_paths[0]).is_absolute()):
        raise ValueError(('one original absolute historical Python path required', relative))
    _, archive_meta = info(archive)
    if not matches(archive_meta, archive_expected):
        raise ValueError(('historical reference archive identity', relative))
    with tarfile.open(archive, 'r:xz') as bundle:
        rows = bundle.getmembers()
        selected = [row for row in rows if row.name == member]
        if len({row.name for row in rows}) != len(rows) or len(selected) != 1:
            raise ValueError(('historical Python member denominator', relative))
        row = selected[0]
        if not row.isfile():
            raise ValueError(('historical Python must be a regular archived entity', relative))
        raw = bundle.extractfile(row).read()
        actual = {'bytes': len(raw), 'sha256': digest(raw), 'mode': row.mode}
        if len(raw) != row.size or actual != expected:
            raise ValueError(('historical Python archived bytes or mode', relative, actual, expected))
    if info(archive)[1] != archive_meta or info(manifest_path)[1] != manifest_meta:
        raise ValueError(('historical Python package changed while reading', relative))
    role = {'role': 'HISTORICAL_REFERENCE_INTERPRETER_NOT_THIS_CI_EXECUTION',
            'archive_member': 'repository/' + relative + '/complete.tar.xz',
            'inner_member': member, 'original_entity': actual,
            'original_paths_are_provenance_only': True,
            'original_absolute_paths': original_paths,
            'original_python_provenance_field': provenance_field,
            'archive': archive_meta, 'manifest': manifest_meta,
            'entity_executed_by_custody': False}
    return role, manifest_path, manifest_meta, archive, archive_expected


def generator_reference_compilers(current):
    """Bind the actual native11 g++ process to its original checked input.

    This is a narrow file-custody role.  It does not execute the compiler or
    infer numerical success from a subprocess exit or a completed archive.
    """
    qualification_path = current / 'review/qualification.json'
    generator_path = current / 'review/generator131/report.json'
    qualification_raw, qualification_meta = info(qualification_path)
    generator_raw, generator_meta = info(generator_path)
    qualification = json.loads(qualification_raw)
    generator = json.loads(generator_raw)
    native11 = generator.get('groups', {}).get('native11', {})
    rows = native11.get('independent_reference_processes', [])
    if not isinstance(rows, list):
        raise ValueError('Generator native11 reference process list required')
    roles, bindings = [], {str(qualification_path): qualification_meta,
                           str(generator_path): generator_meta}
    process_root = current / 'review/generator131/native11/independent-reference-processes'
    reference_source = current / 'review/generator131-source/reference.cpp'
    reference_image = current / 'review/generator131/native11/original/std-mt-reference'
    for index, row in enumerate(rows):
        argv = row['argv']
        if (not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv)
                or not Path(argv[0]).is_absolute()):
            raise ValueError('absolute original reference process argv required')
        resolved = Path(argv[0]).resolve(strict=True)
        if resolved.is_relative_to(current):
            continue
        if (argv[0] not in ('/bin/g++', '/usr/bin/g++')
                or resolved.parent != Path('/usr/bin')
                or re.fullmatch(r'x86_64-linux-gnu-g\+\+-[0-9]+', resolved.name) is None
                or argv[1:] != ['-std=c++17', '-O2', str(reference_source), '-o', str(reference_image)]
                or index != 0 or row['index'] != 0 or row['production_runtime'] is not False):
            raise ValueError(('unsupported external Generator reference executable', argv))
        expected = row['executable']
        if (not isinstance(expected, dict) or set(expected) != {'bytes', 'sha256', 'mode'}
                or type(expected['bytes']) is not int or expected['bytes'] < 0
                or type(expected['mode']) is not int or not 0 <= expected['mode'] <= 0o7777
                or not isinstance(expected['sha256'], str)
                or re.fullmatch('[0-9a-f]{64}', expected['sha256']) is None):
            raise ValueError('complete original reference compiler byte/mode identity required')
        _, actual = info(resolved)
        declared = qualification['checked_inputs'].get(str(resolved))
        if actual != expected or declared != expected:
            raise ValueError(('reference process, actual compiler and checked input differ', str(resolved)))
        directory = Path(row['directory'])
        if (directory != process_root / 'process-0000' or directory.resolve(strict=True) != directory):
            raise ValueError('exact original reference compiler process directory required')
        process_path = directory / 'process.json'
        process_raw, process_meta = info(process_path)
        if json.loads(process_raw) != row:
            raise ValueError('reference compiler process record differs from Generator report')
        bindings[str(process_path)] = process_meta
        for stream in ('stdout', 'stderr'):
            path = directory / (stream + '.bin')
            raw, meta = info(path)
            if len(raw) != row[stream + '_bytes'] or digest(raw) != row[stream + '_sha256']:
                raise ValueError(('original reference compiler stream identity differs', stream))
            bindings[str(path)] = meta
        bindings[str(resolved)] = actual
        roles.append(dict(role='THIS_CI_GENERATOR_NATIVE11_REFERENCE_COMPILER',
            requested_path=argv[0], original_path=str(resolved),
            member='external/generator-native11-reference-compiler/' + resolved.name, **actual,
            qualification_member='current-runtime/review/qualification.json',
            generator_report_member='current-runtime/review/generator131/report.json',
            process_member='current-runtime/' + process_path.relative_to(current).as_posix(),
            original_process={key: row[key] for key in
                              ('index', 'argv', 'exit_code', 'error', 'stdout_EOF', 'stderr_EOF', 'production_runtime')},
            actual_executable_matches_process_and_checked_input=True,
            entity_executed_by_custody=False, runtime_qualification_granted=False))
    if len(roles) > 1:
        raise ValueError('at most one original native11 reference compiler process is supported')
    return roles, bindings


def live_reference_custody(repo, current, group):
    """Bind an original live-reference proof and its actual reference bytes.

    This reads the original reports, argv, process streams and declared inputs.
    It performs no Tensor calculation, native execution or case admission.
    The reference digest comes from this run's physical report, never a prior run.
    """
    if group not in ('numeric_predicates', 'bf16_copy', 'half_mul', 'half_neg', 'half_addsub', 'wide_neg'):
        raise ValueError('unknown mandatory live-reference group')
    numeric = group == 'numeric_predicates'
    multiplication = group == 'half_mul'
    negation = group in ('half_neg', 'wide_neg')
    addition = group == 'half_addsub'
    shared_reference = {'half_mul': 'half-mul312', 'half_neg': 'half-neg371',
                        'half_addsub': 'half-addsub1006', 'wide_neg': 'wide-neg405'}.get(group)
    namespace = {'numeric_predicates': 'numeric-predicates', 'bf16_copy': 'bf16-copy',
                 'half_mul': 'half-mul', 'half_neg': 'half-neg', 'half_addsub': 'half-addsub', 'wide_neg': 'wide-neg'}[group]
    reference_status = 'OBSERVED_' + group.upper()
    physical_status = 'PASS_PHYSICAL_' + group.upper()
    template = {'numeric_predicates': NUMERIC_PROOF, 'bf16_copy': BF16_PROOF,
                'half_mul': HALF_MUL_PROOF, 'half_neg': HALF_NEG_PROOF, 'half_addsub': HALF_ADDSUB_PROOF, 'wide_neg': WIDE_NEG_PROOF}[group]
    cases = template['cases']
    case_sha = {'numeric_predicates': NUMERIC_CASES, 'bf16_copy': BF16_CASES,
                'half_mul': HALF_MUL_CASES, 'half_neg': HALF_NEG_CASES, 'half_addsub': HALF_ADDSUB_CASES, 'wide_neg': WIDE_NEG_CASES}[group]
    bindings = {}

    def need(condition, message):
        if not condition:
            raise ValueError(message)

    def bind(path, expected=None):
        path = Path(path)
        need(path.is_absolute() and str(path.resolve()) == str(path), 'canonical predicate custody path')
        raw, metadata = info(path)
        need(expected is None or metadata['sha256'] == expected, 'predicate custody SHA: ' + str(path))
        need(str(path) not in bindings or bindings[str(path)] == metadata, 'predicate custody input changed')
        bindings[str(path)] = metadata
        return raw

    def document(path):
        result = json.loads(bind(path))
        need(isinstance(result, dict), 'predicate custody JSON object required')
        return result

    def check_fields(actual, expected, label):
        need(all(type(actual.get(key)) is type(value) and actual[key] == value
                 for key, value in expected.items()), label)

    qualification_path = current / 'review/qualification.json'
    qualification = document(qualification_path)
    check_fields(qualification, dict(schema='heptatorch.current-runtime.qualification.v1',
        required_suites=list(GROUPS), required_cases_total=4250, functions=767, units=15,
        source_sha256=SOURCE, native_sha256=IMAGE), 'exact current767 twenty-five-suite inventory and image')
    manifest_raw = bind(repo / 'tools/current_runtime/source/manifest.json', SOURCE_MANIFEST)
    need(bind(current / 'build/manifest.json', SOURCE_MANIFEST) == manifest_raw,
         'the actual cold build used the final complete source manifest')
    manifest = json.loads(manifest_raw)
    need(manifest['runtime']['sha256'] == SOURCE and manifest['runtime']['functions'] == 767
         and manifest['expected_native_sha256'] == IMAGE and len(manifest['units']) == 15,
         'final complete source and image pins')
    bind(current / 'build/runtime.hepta', SOURCE)
    bind(current / 'build/heptatorch', IMAGE)
    build_raw = bind(current / 'build/build.json')
    build_sha = digest(build_raw)
    check_fields(json.loads(build_raw), dict(status='PASS_NATIVE_BUILD_ONLY', source_sha256=SOURCE,
        native_sha256=IMAGE, manifest_sha256=SOURCE_MANIFEST, functions=767, units_planned=15),
        'actual final cold build identity')
    need(qualification['actual_build_sha256'] == build_sha, 'qualifier binds this actual cold build')
    rows = [row for row in qualification['suites'] if row['name'] == group]
    need(len(rows) == 1, 'one mandatory numeric predicate suite')
    suite = rows[0]
    check_fields(suite, dict(status='PASS', qualified=True, **template),
                 'complete original predicate comparison, process and control census')
    directory = current / 'review' / group
    native_path = directory / 'report.json'
    native_raw = bind(native_path)
    native = json.loads(native_raw)
    need(suite['original_report'] == dict(path=group + '/report.json',
         bytes=len(native_raw), sha256=digest(native_raw)), 'unchanged original native report binding')
    native_extras = dict(compiler_executed=False, original_reference_modified=False,
                        mapping_reads=10) if numeric else dict(full_project_status='HOLD')
    if negation:
        native_extras['mapping_reads'] = 4
    if addition:
        native_extras['mapping_reads'] = 12
    schema_version = 'v2' if multiplication or addition else 'v1'
    check_fields(native, dict(schema='heptatorch.' + namespace + '.native-observation.' + schema_version, status='PASS',
        group=group, cases=cases, expected_cases=cases, case_sha256=case_sha,
        source_sha256=SOURCE, native_sha256=IMAGE, actual_build_sha256=build_sha,
        native_executed=True, Torch_executed=True, host_expected_values_computed=False,
        all_inputs_unchanged=True, frames=template['frames'], checks=template['assertions'],
        owner_reads=template['owner_snapshots'], failures=[], **native_extras),
        'original complete native live-reference report')
    names = ['case-%03d' % index for index in range(cases)]
    if numeric:
        names += ['packet-controls']
        names += ['transaction-%d' % opcode for opcode in range(120, 125)]
        names += ['allocation-%d' % opcode for opcode in range(120, 125)]
    need([row['name'] for row in native['results']] == names
         and all(row['status'] == 'PASS' for row in native['results']),
         'all original native observation records')
    for row in native['results']:
        need(document(directory / row['name'] / 'observation.json') == row,
             'each original per-session observer report is retained')
    reference = current / 'review/references' / shared_reference if shared_reference else directory / 'reference'
    execution = current / 'review/references' / (shared_reference + '-execution') if shared_reference else directory / 'reference-execution'
    if numeric or shared_reference:
        need(native['reference_directory'] == str(reference)
             and native['reference_process_directory'] == str(execution),
             'exact this-run independent reference locations')
    reference_raw = bind(reference / 'report.json')
    reference_sha = digest(reference_raw)
    observed = json.loads(reference_raw)
    reference_extras = dict(compiler_executed=False, all_observed_blobs_unchanged=True) if numeric else {}
    reference_version = 'v2' if addition else 'v1'
    check_fields(observed, dict(schema='heptatorch.' + namespace + '.reference.' + reference_version,
        status=reference_status, cases=cases, expected_cases=cases, case_sha256=case_sha,
        Torch_executed=True, native_executed=False, host_expected_values_computed=False,
        inputs_unchanged=True, rng_unchanged=True, failures=[], **reference_extras),
        'complete actual independent Torch observation')
    if not numeric:
        need(observed['cpu_capability'] in ('AVX2', 'AVX512')
             and native['cpu_capability'] == observed['cpu_capability'], 'actual fixed-wheel CPU copy capability')
    if shared_reference:
        check_fields(observed, dict(cpu_capability='AVX2', requested_cpu_capability='avx2',
            cpu_environment='avx2', torch_threads=1), 'explicit actual AVX2 single-thread arithmetic oracle')
        check_fields(native, dict(cpu_capability='AVX2', requested_cpu_capability='avx2', torch_threads=1),
            'native arithmetic reader retains the actual reference execution contract')
    need(native['reference_sha256'] == suite['reference_sha256'] == reference_sha,
         'this-run actual reference report digest')
    proof_raw = bind(directory / 'physical-qualification.json')
    proof = json.loads(proof_raw)
    expected_proof = dict(template, reference_sha256=reference_sha)
    if addition:
        fe_path = repo / 'tools/current_runtime/half_addsub_fe_observer_transition.py'
        bind(fe_path)
        fe_spec = importlib.util.spec_from_file_location('sealed_half_addsub_FE_observer_inverse', fe_path)
        fe_observer = importlib.util.module_from_spec(fe_spec)
        fe_spec.loader.exec_module(fe_observer)
        _, fe_proof = fe_observer.restore_observers(bind)
        need(qualification['half_addsub_observer_transition'] == fe_proof,
             'all original1004 observer bytes and the two exact FE successor requests are retained')
    successor_bindings = {}
    successor_evidence = None
    if multiplication:
        successor_evidence, successor_bindings = live_reference_custody(repo, current, 'half_addsub')
        successor_sha = successor_evidence['reference_sha256']
        need(native['addsub_reference_directory'] == str(current / 'review/references/half-addsub1006')
             and native['addsub_reference_process_directory'] == str(current / 'review/references/half-addsub1006-execution')
             and native['addsub_reference_sha256'] == suite['addsub_reference_sha256'] == successor_sha,
             'all four changed multiplication controls bind this same original1006 Torch reference')
        expected_proof['addsub_reference_sha256'] = successor_sha
        observer_path = repo / 'tools/current_runtime/half_addsub_observer_transition.py'
        bind(observer_path)
        spec = importlib.util.spec_from_file_location('sealed_half_addsub_observer_inverse', observer_path)
        observer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(observer)
        _, observer_proof = observer.restore_mul_runner(bind(repo / observer.PATH), bind)
        need(qualification['half_mul_observer_transition'] == observer_proof,
             'the complete original312 observer inverse and unchanged371 source proof are retained')
    need(set(proof) == set(expected_proof), 'complete fixed predicate physical-proof fields')
    check_fields(proof, expected_proof, 'exact original physical predicate proof and current reference')
    need(suite['physical_qualification'] == dict(path=group + '/physical-qualification.json',
         bytes=len(proof_raw), sha256=digest(proof_raw), status=physical_status),
         'exact physical-proof byte binding')
    process = document(execution / 'process.json')
    need(document(execution / 'runner.json') == process, 'unchanged actual reference supervisor receipt')
    check_fields(process, dict(started=True, exit_code=0, stdout_EOF=True, stderr_EOF=True,
        forced_cleanup=False, stderr_bytes=0, cleanup_errors=[], running_descendants_after_cleanup=[],
        terminated_process_groups=[]), 'actual reference normal exit and both physical EOFs')
    need(not process.get('timed_out') and not process.get('error') and not process.get('execution_error'),
         'reference did not time out or fail execution')
    argv = process['argv']
    argv_tail = ['-B', str(repo / 'tests' / group / 'reference.py')]
    if numeric:
        argv_tail += ['--repository', str(repo)]
    argv_tail += ['--output', str(reference)]
    need(isinstance(argv, list) and len(argv) == len(argv_tail) + 1 and Path(argv[0]).is_absolute()
         and argv[1:] == argv_tail
         and observed['actual_process']['argv'] == argv
         and observed['actual_process']['pid'] == process['pid'], 'original actual Torch argv and PID')
    for channel in ('stdout', 'stderr'):
        raw = bind(execution / ('runner.' + channel), process[channel + '_sha256'])
        need(len(raw) == process[channel + '_bytes'], 'complete actual reference ' + channel)
    need(bind(execution / 'runner.stderr') == b''
         and json.loads(bind(execution / 'runner.stdout')) == dict(status=reference_status,
             cases=cases, expected_cases=cases, error=None), 'original reference stdout and stderr')
    python = Path(argv[0]).resolve(strict=True)
    bind(python)
    need(observed['actual_process']['python'] == dict(path=str(python), **bindings[str(python)]),
         'actual reference interpreter original bytes and POSIX mode')
    provenance = observed['oracle_provenance']
    lock_path = repo / 'compatibility/pytorch-oracle-lock.json'
    lock_raw = bind(lock_path, 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944')
    bind(repo / 'tests/pytorch_oracle.py', 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae')
    need(provenance['status'] == 'PASS' and len(provenance['checks']) == 11
         and all(value is True for value in provenance['checks'].values())
         and provenance['expected'] == json.loads(lock_raw)['oracle']
         and provenance['lock_sha256'] == digest(lock_raw), 'all original fixed-wheel provenance checks')
    reference_inputs = {str(path) for path in (repo / 'tests/numeric_predicates/reference.py',
        repo / 'tests/numeric_predicates/cases.py', repo / 'tests/numeric_predicates/__init__.py',
        repo / 'tests/pytorch_oracle.py', lock_path, repo / 'compatibility/upstream-lock.json',
        repo / 'build/pytorch-install.json', python)}
    if not numeric:
        reference_inputs |= {str(repo / 'tests' / group / name)
                             for name in ('reference.py', 'cases.py', '__init__.py')}
    if multiplication:
        reference_inputs.add(str(repo / 'tests/half_mul/legacy_controls.py'))
    need(set(observed['inputs']) == reference_inputs, 'all original independent reference input entities')
    for spelling, metadata in observed['inputs'].items():
        bind(Path(spelling), metadata['sha256'])
        need(metadata == dict(path=spelling, **bindings[spelling]), 'reference input bytes and mode unchanged')
    need(len(observed['rows']) == cases, 'all original independent reference rows')
    for index, row in enumerate(observed['rows']):
        if multiplication or addition:
            original_flags = (row.get('inputs_unchanged') is True and row.get('rng_unchanged') is True
                and row.get('accepted') is (not row['spec'].get('expected_error', False))
                and len(row.get('inputs', [])) == 2)
            if row.get('accepted') is True:
                original_flags = original_flags and row.get('fresh_tensor') is True and row.get('fresh_storage') is True
            else:
                if addition and row['spec'].get('expected_error') == 'bool_subtraction':
                    original_flags = (original_flags and row['error']['type'] == 'NotImplementedError'
                        and 'bool tensor' in row['error']['message'])
                else:
                    original_flags = (original_flags and row['error']['type'] == 'RuntimeError'
                        and 'size' in row['error']['message'] and 'match' in row['error']['message'])
            values = list(row['inputs']) + ([row['result']] if row['accepted'] else [])
        else:
            original_flags = row.get('accepted') is True if numeric else all(row.get(key) is True for key in
                ('input_unchanged', 'rng_unchanged', 'fresh_tensor', 'fresh_storage'))
            if negation:
                original_flags = original_flags and row.get('accepted') is True
            values = (row['input'], row['result'])
        need(row['index'] == index and original_flags
             and document(reference / ('case-%03d.json' % index)) == row,
             'complete unchanged independent reference row')
        for value in values:
            for field in ('logical', 'storage'):
                blob = value[field]
                need(set(blob) == {'path', 'bytes', 'sha256'}
                     and re.fullmatch('[0-9a-f]{64}', blob['sha256']) is not None
                     and blob['path'] == 'blobs/' + blob['sha256'] + '.bin', 'safe original reference blob')
                need(len(bind(reference / blob['path'], blob['sha256'])) == blob['bytes'],
                     'complete original reference blob bytes')
    outer = current / 'review' / (group + '-execution')
    need(document(outer / 'runner.json') == suite['runner'], 'original predicate runner receipt retained')
    check_fields(suite['runner'], dict(started=True, exit_code=0, stdout_EOF=True, stderr_EOF=True,
        forced_cleanup=False, stderr_bytes=0, cleanup_errors=[], running_descendants_after_cleanup=[]),
        'original native observer runner normal exit and complete EOFs')
    for channel in ('stdout', 'stderr'):
        raw = bind(outer / ('runner.' + channel), suite['runner'][channel + '_sha256'])
        need(len(raw) == suite['runner'][channel + '_bytes'], 'original observer runner stream extent')
    intent_raw = bind(outer / 'intent.json')
    need(suite['execution_intent'] == dict(path=group + '-execution/intent.json',
         bytes=len(intent_raw), sha256=digest(intent_raw))
         and json.loads(intent_raw)['argv'] == suite['runner']['argv'], 'original observer intent binding')
    # The coordinator's own output cannot be its own checked input. The
    # supervisor writes runner.json separately; the reference reader consumes
    # its byte-identical process.json. These original supervisor outputs are
    # bound here to the already recorded receipts and stream digests above.
    custody_outputs = {str(qualification_path), str(execution / 'runner.json'),
                       str(outer / 'runner.json'), str(outer / 'runner.stdout'),
                       str(outer / 'runner.stderr')}
    for spelling, metadata in bindings.items():
        if spelling in custody_outputs:
            continue
        need(qualification['checked_inputs'].get(spelling) == metadata,
             'current qualifier checked every declared predicate custody input: ' + spelling)
    if group == 'wide_neg':
        f64_report = document(current / 'review/f64_48/report.json')
        f64_proof = document(current / 'review/f64_48/physical-qualification.json')
        successor = f64_report['source_admission']['wide_neg_successor']
        check_fields(successor, dict(original_request_count=87, retained_refusals=86,
            accepted_successors=1, labels=['unsupported-F64-neg'],
            reference_directory=str(reference), reference_execution=str(execution),
            reference_sha256=reference_sha, actual_Torch_reference_verified=True,
            host_expected_values_computed=False), 'complete original87 F64 requests share actual405 reference')
        need(f64_proof['wide_neg_successor'] == successor and f64_proof['cases'] == 48
             and f64_proof['retained_refusals'] == 86 and f64_proof['actual_numeric_successors'] == 1
             and f64_proof['original_OOM_controls'] == 2,
             'all48 F64 physical cases and one unchanged-recipe numeric successor')
        for path in (current / 'review/f64_48/report.json', current / 'review/f64_48/physical-qualification.json'):
            need(qualification['checked_inputs'].get(str(path)) == bindings[str(path)],
                 'both original F64 reports are physically bound by the current qualifier')
    role = dict(role='THIS_CI_' + group.upper() + '_TORCH_REFERENCE_INTERPRETER',
        requested_path=argv[0], original_path=str(python), **bindings[str(python)],
        report_member='current-runtime/' + (reference / 'report.json').relative_to(current).as_posix(),
        process_member='current-runtime/' + (execution / 'process.json').relative_to(current).as_posix(),
        original_process={key: process[key] for key in ('argv', 'pid', 'exit_code', 'stdout_EOF', 'stderr_EOF')},
        entity_executed_by_custody=False, runtime_qualification_granted=False)
    result = dict(schema='heptatorch.' + namespace + '.original-evidence-custody.v1',
        original_proof=proof, reference_sha256=reference_sha,
        native_report_sha256=digest(native_raw), physical_proof_sha256=digest(proof_raw),
        source_manifest_sha256=SOURCE_MANIFEST, reference_interpreter=role,
        complete_reference_process_streams_and_inputs_bound=True,
        additional_original_supervisor_outputs_bound_by_custody=sorted(custody_outputs),
        numerical_cases_readmitted=False, native_executed=False, Torch_executed=False)
    if successor_evidence is not None:
        for spelling, metadata in successor_bindings.items():
            need(spelling not in bindings or bindings[spelling] == metadata,
                 'the shared successor evidence retains one original byte identity')
            bindings[spelling] = metadata
        result.update(addsub_reference_sha256=successor_evidence['reference_sha256'],
            addsub_physical_proof_sha256=successor_evidence['physical_proof_sha256'],
            complete_shared_addsub_reference_custody_bound=True,
            multiplication_observer_inverse=observer_proof)
    return result, bindings


def numeric_predicate_custody(repo, current):
    return live_reference_custody(repo, current, 'numeric_predicates')


def bf16_copy_custody(repo, current):
    return live_reference_custody(repo, current, 'bf16_copy')


def half_mul_custody(repo, current):
    return live_reference_custody(repo, current, 'half_mul')


def half_neg_custody(repo, current):
    return live_reference_custody(repo, current, 'half_neg')


def half_addsub_custody(repo, current):
    return live_reference_custody(repo, current, 'half_addsub')


def wide_neg_custody(repo, current):
    return live_reference_custody(repo, current, 'wide_neg')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--current-output', type=Path, required=True)
    parser.add_argument('--toolchain-root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args(argv)
    repo, current, temporary = (value.resolve() for value in
                                (args.repository, args.current_output, args.toolchain_root))
    prefix = args.output_prefix.resolve()
    if re.fullmatch('[0-9a-f]{40}', args.source_commit) is None:
        raise ValueError('source commit must be a full Git SHA')
    if prefix.is_relative_to(current) or prefix.is_relative_to(repo):
        raise ValueError('archive output must be outside the selected current output and repository')
    destinations = {key: Path(str(prefix) + suffix) for key, suffix in (
        ('archive', '.tar.xz'), ('manifest', '.manifest.json'),
        ('report', '.custody.json'), ('sha256', '.sha256'))}
    if any(path.exists() or path.is_symlink() for path in destinations.values()):
        raise ValueError('fresh evidence destinations required')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema='heptatorch.current.ci-byte-mode-custody.v2', status='RUNNING',
        source_commit=args.source_commit, repository=str(repo), current_output=str(current),
        native_executed=False, Torch_executed=False, compiler_executed=False,
        case_admissions_executed_here=0, full_pytorch_rewrite_complete=False,
        whole_pytorch_gate='HOLD',
        scope='File-byte and original POSIX-mode custody only; original numerical reports retain their statuses.',
        required_output_paths=list(REQUIRED_OUTPUTS), required_repository_paths=list(REPOSITORY_PATHS),
        required_toolchain_directories=list(TOOLCHAINS),
        required_suites=list(GROUPS), required_cases_total=4250,
        reference_interpreters=[], execution_interpreter=None,
        reference_process_executables=[], numeric_predicates_evidence=None,
        bf16_copy_evidence=None, half_mul_evidence=None, half_neg_evidence=None, half_addsub_evidence=None, wide_neg_evidence=None, missing=[], errors=[])
    selected, declared, parsed_lists, input_declarations = {}, {}, {}, []
    report_declared_paths = set()
    roots = [(current, 'current-runtime'), (repo, 'repository')] + [
        (temporary / name, 'toolchain/' + name) for name in TOOLCHAINS]
    roots.sort(key=lambda row: len(row[0].parts), reverse=True)
    interpreter = Path(sys.executable).resolve()
    external_executables = {interpreter: 'external/ci-execution-python/' + interpreter.name}

    def problem(stage, path, error):
        report['errors'].append(dict(stage=stage, path=str(path), error=repr(error)))

    def add(path):
        if path.is_symlink():
            raise ValueError(('symlink forbidden', str(path)))
        resolved = path.resolve()
        if any(resolved == target for target in destinations.values()):
            raise ValueError('cannot archive own output')
        if resolved in external_executables:
            name = external_executables[resolved]
        else:
            name = None
            for root, alias in roots:
                if resolved.is_relative_to(root):
                    name = alias + '/' + resolved.relative_to(root).as_posix()
                    break
            if name is None:
                raise ValueError(('undeclared external input', str(resolved)))
        if name in selected and selected[name] != resolved:
            raise ValueError(('member collision', name))
        selected[name] = resolved

    def select(path, label):
        if not path.exists() or path.is_symlink():
            report['missing'].append(label)
            return
        try:
            members = sorted(path.rglob('*')) if path.is_dir() else [path]
        except BaseException as error:
            problem('directory-traversal', path, error)
            return
        for member in members:
            try:
                if member.is_symlink():
                    raise ValueError(('symlink in evidence', str(member)))
                if member.is_file():
                    add(member)
                elif not member.is_dir():
                    raise ValueError(('nonregular evidence', str(member)))
            except BaseException as error:
                problem('directory-member', member, error)

    try:
        # This ordinary original entity is always selected, including early failures.
        # It is never substituted for either historical reference package entity.
        try:
            add(interpreter)
            _, interpreter_meta = info(interpreter)
            declared.setdefault(str(interpreter), []).append(interpreter_meta)
            input_declarations.append(dict(input_report='sealer-process', table='execution_interpreter',
                                           source_path=str(interpreter), expected=interpreter_meta))
            report['execution_interpreter'] = dict(
                requested_path=sys.executable, original_path=str(interpreter),
                member='external/ci-execution-python/' + interpreter.name, **interpreter_meta,
                role='THIS_CI_EXECUTION_INTERPRETER_NOT_HISTORICAL_REFERENCE_INTERPRETER',
                archived_entity_executed=False, metadata10_worker_identity_checked=False)
        except BaseException as error:
            problem('execution-interpreter', interpreter, error)
        try:
            reference_roles, reference_bindings = generator_reference_compilers(current)
            for role in reference_roles:
                path = Path(role['original_path'])
                if path in external_executables:
                    raise ValueError('reference compiler cannot replace the actual CI interpreter role')
                external_executables[path] = role['member']
                add(path)
                report['reference_process_executables'].append(role)
            for spelling, metadata in reference_bindings.items():
                declared.setdefault(spelling, []).append(metadata)
                input_declarations.append(dict(input_report='generator-reference-process-evidence',
                    table='reference_executable_custody', source_path=spelling, expected=metadata))
        except BaseException as error:
            problem('generator-reference-executable', current / 'review/generator131/report.json', error)
        for group in ('numeric_predicates', 'bf16_copy', 'half_mul', 'half_neg', 'half_addsub', 'wide_neg'):
            try:
                evidence, bindings = live_reference_custody(repo, current, group)
                role = evidence['reference_interpreter']
                reference_python = Path(role['original_path'])
                if reference_python not in external_executables:
                    external_executables[reference_python] = (
                        'external/' + group.replace('_', '-') + '-reference-python/' + reference_python.name)
                role['member'] = external_executables[reference_python]
                report['reference_process_executables'].append(role)
                report[group + '_evidence'] = evidence
                for spelling, metadata in bindings.items():
                    declared.setdefault(spelling, []).append(metadata)
                    input_declarations.append(dict(input_report=group + '-original-evidence-custody',
                        table='original_report_and_reference_byte_bindings', source_path=spelling, expected=metadata))
                    add(Path(spelling))
            except BaseException as error:
                problem(group + '-original-evidence', current / 'review' / group / 'report.json', error)
        select(current, 'current-runtime/')
        for relative in REPOSITORY_PATHS:
            select(repo / relative, 'repository/' + relative)
        for name in TOOLCHAINS:
            select(temporary / name, 'toolchain/' + name)
        for relative in REQUIRED_OUTPUTS:
            if not (current / relative).is_file():
                report['missing'].append('current-runtime/' + relative)
        for relative in REFERENCE_INTERPRETER_PACKAGES:
            try:
                role, path, meta, archive, archive_expected = reference_interpreter_role(repo, relative)
                parsed_lists[str(path)] = meta
                declared.setdefault(str(archive), []).append(archive_expected)
                input_declarations.append(dict(input_report=relative + '/manifest.json', table='archive',
                                               source_path=str(archive), expected=archive_expected))
                report['reference_interpreters'].append(role)
            except BaseException as error:
                problem('historical-reference-interpreter', repo / relative, error)
        input_reports = [(relative, current / relative) for relative in INPUT_REPORTS]
        input_reports.append(('toolchain/hepta-current-entropy/build.json',
                              temporary / 'hepta-current-entropy/build.json'))
        for relative, path in input_reports:
            if not path.is_file():
                report['missing'].append(relative if relative.startswith('toolchain/') else 'current-runtime/' + relative)
                continue
            try:
                raw, meta = info(path)
                parsed_lists[str(path)] = meta
                document = json.loads(raw)
                if not isinstance(document, dict):
                    raise ValueError('input report object required')
            except BaseException as error:
                problem('input-report', path, error)
                continue
            if relative == 'review/qualification.json':
                report['original_qualification_status'] = document.get('status')
                report['original_required_suites'] = document.get('required_suites')
                report['original_required_cases'] = document.get('required_cases_total')
                if (document.get('schema') != 'heptatorch.current-runtime.qualification.v1'
                        or document.get('required_suites') != list(GROUPS)
                        or document.get('required_cases_total') != 4250):
                    problem('qualification-inventory', path,
                            ValueError('current twenty-five-group qualification input inventory differs'))
            if relative == 'review/metadata10/report.json':
                worker = document.get('worker_interpreter')
                if worker is not None:
                    try:
                        actual = report['execution_interpreter']
                        if actual is None or any(worker[key] != actual[key] for key in
                                                  ('original_path', 'bytes', 'sha256', 'mode')):
                            raise ValueError('metadata10 actual worker interpreter differs from this CI interpreter')
                        actual['metadata10_worker_identity_checked'] = True
                    except BaseException as error:
                        problem('execution-interpreter-binding', path, error)
            for key in INPUT_TABLES:
                table = document.get(key)
                if table is None:
                    continue
                if not isinstance(table, dict):
                    problem('input-table', path, ValueError(('input metadata table required', relative, key)))
                    continue
                for spelling, metadata in table.items():
                    try:
                        source = Path(spelling)
                        if not source.is_absolute() or str(source.resolve()) != spelling:
                            raise ValueError(('noncanonical bound input', spelling))
                        if (not isinstance(metadata, dict) or type(metadata.get('bytes')) is not int
                                or metadata['bytes'] < 0
                                or re.fullmatch('[0-9a-f]{64}', str(metadata.get('sha256'))) is None):
                            raise ValueError(('complete declared SHA and extent required', relative, key, spelling))
                        expected = {name: metadata[name] for name in ('bytes', 'sha256')}
                        if 'mode' in metadata:
                            if type(metadata['mode']) is not int or not 0 <= metadata['mode'] <= 4095:
                                raise ValueError(('invalid original declared mode', spelling))
                            expected['mode'] = metadata['mode']
                        declared.setdefault(spelling, []).append(expected)
                        if relative in INPUT_REPORTS:
                            report_declared_paths.add(spelling)
                        input_declarations.append(dict(input_report=relative, table=key,
                                                       source_path=spelling, expected=expected))
                        add(source)
                        _, actual = info(source)
                        if not matches(actual, expected):
                            raise ValueError(('input custody mismatch', spelling, actual, expected))
                    except BaseException as error:
                        problem('declared-input', spelling, error)
        if not selected:
            raise ValueError('no actual evidence available')
        if not report_declared_paths:
            problem('declared-inputs', current, ValueError('nonempty declared inputs required'))
        members, archive_meta, errors = seal_archive(destinations['archive'], selected, declared, parsed_lists)
        report['errors'].extend(errors)
        manifest = dict(schema='heptatorch.current.ci-complete-members.v2', source_commit=args.source_commit,
            repository=str(repo), current_output=str(current), member_count=len(members),
            member_bytes=sum(row['bytes'] for row in members.values()), members=members,
            archive=archive_meta, input_list_bindings=parsed_lists, declared_input_bindings=input_declarations,
            execution_interpreter=report['execution_interpreter'],
            reference_interpreters=report['reference_interpreters'],
            reference_process_executables=report['reference_process_executables'],
            numeric_predicates_evidence=report['numeric_predicates_evidence'],
            bf16_copy_evidence=report['bf16_copy_evidence'], half_mul_evidence=report['half_mul_evidence'],
            half_neg_evidence=report['half_neg_evidence'], half_addsub_evidence=report['half_addsub_evidence'],
            wide_neg_evidence=report['wide_neg_evidence'],
            external_path_mapping=[dict(original_root=str(root), prefix=alias) for root, alias in roots]
                + [dict(original_file=str(path), member=name)
                   for path, name in sorted(external_executables.items(), key=lambda row: str(row[0]))])
        dump(destinations['manifest'], manifest)
        destinations['sha256'].write_text(archive_meta['sha256'] + '  ' + destinations['archive'].name + '\n')
        report.update(member_count=len(members), member_bytes=manifest['member_bytes'],
            archive=archive_meta, manifest=info(destinations['manifest'])[1],
            complete_single_XZ_EOF=True, complete_TAR_zero_padding=True,
            all_original_member_modes_preserved=True, all_selected_files_unchanged=not report['errors'],
            declared_input_count=len(declared), input_declaration_count=len(input_declarations),
            declared_runtime_report_input_count=len(report_declared_paths),
            unarchived_selected_members=sorted(set(selected) - set(members)))
        report['missing'] = sorted(set(report['missing']))
        report['status'] = 'FAIL' if report['missing'] or report['errors'] else 'PASS'
    except BaseException as error:
        report.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
    dump(destinations['report'], report)
    print(json.dumps({key: report.get(key) for key in ('status', 'member_count', 'member_bytes', 'missing', 'error')}))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
