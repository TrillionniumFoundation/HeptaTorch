"""Restore reviewed F64 48 sources and exact original Torch observations.

All102 original producer inputs retain their bytes and modes. The original
terminal and four streams are available; its actual argv and Python executable
entity were not retained. Current native observer identities do not fill that gap.
This module performs byte transport and source/provenance reading only.
"""
import hashlib
import importlib.util
import ast
import contextlib
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
import types

SOURCE_PACKAGE = 'validation/current-f64-48-source'
REFERENCE_PACKAGE = 'validation/current-f64-48-reference'
REFERENCE = 'b1dafeed270b4dd226c349dece9d7f6213c2dabd34681685d97ec13c068e9b11'
READER = '1deb2fa84d9ea9653ca56769a5527f0102571ba2a4690f8deb85f5d4bdc038b3'
SOURCE = '51820849f7becec549be15babace00f00687601ae7f53bcee0e636ca02f721f9'
IMAGE = 'aa3cc8f0c8738a9505d31c9e8ba5a523b7e08b591b61fc7bfb8592295491618b'
CURRENT_SOURCE = '0061acfaa8e886885053e8f539cf08187be1c1e10265c19e52df07435a71594b'
CURRENT_IMAGE = '955ffc996b8870ef67752f45647f5c1ca9a093f41ff402487091f03223372293'
CURRENT_PARTITION = '2e442fea4961a5111004a1a721fb46e817f5568f50c7068047fc767721df8e21'
HALF_SOURCE = 'bf3641b1a9061c6f92b8cbeef75adfac89e82ee16d32a9f650cb938d9e918386'
PREDICATE_INVERSE = '71244cd75fb41ce07876d7f5699b66ae43c73adc5dd5d09fb2da0dec9c8c7439'
HALF_LOGICAL_INVERSE = 'e0a2eac589ddfe8b09794fb55c569a73d7d7802888130646f59d33adca266f66'
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



def reader_cli(repository, build, build_sha, reference, wide_neg, public_f64_arithmetic):
    return ['--reader-directory',str(reference['source']),
            '--candidate',str(repository),'--build-directory',str(build),
            '--reference',str(reference['report']),'--expected-source-sha256',CURRENT_SOURCE,
            '--expected-native-sha256',CURRENT_IMAGE,'--expected-build-sha256',build_sha,
            '--expected-reference-sha256',REFERENCE,
            '--wide-neg-reference-directory',str(wide_neg['root']),
            '--wide-neg-reference-execution',str(wide_neg['execution']),
            '--public-f64-arithmetic-reference-directory',str(public_f64_arithmetic['root']),
            '--public-f64-arithmetic-reference-execution',str(public_f64_arithmetic['execution'])]


def load_module(raw, path, name):
    result = types.ModuleType(name)
    result.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), result.__dict__)
    return result


def current_adapter_type(original_type, bridge, repository, extension_files, kind):
    """Add a current-image source proof around the complete frozen reader.

    Frozen packages, reference namespaces, comparison functions and error and
    resource recipes are unchanged. Only source/build/owner admission and the
    identity namespace of the already derived main are extended. The complete
    new source must invert to the exact736 predecessor before its original
    source inverse is allowed to run.
    """
    repository = Path(repository).resolve(strict=True)
    frozen_source = bridge.SOURCE
    require(frozen_source == SOURCE and bridge.IMAGE == IMAGE,
            'the frozen bridge retains its original736 source and image')
    producer = bridge.PRODUCER if kind == 'f64' else bridge.PRODUCER_MANIFEST
    candidate = bridge.FROZEN_CANDIDATE if kind == 'f64' else bridge.PRODUCER_CANDIDATE

    class CurrentAdapter(original_type):
        def bind_package(self, inputs):
            digest = super().bind_package(inputs)
            for path in extension_files:
                inputs.bind(Path(path).resolve(strict=True))
            return digest

        def candidate_admission(self, directory, inputs):
            actual = Path(directory).resolve(strict=True)
            require(actual == repository, 'explicit actual repository for current reader')
            manifest_raw = inputs.bind(repository / 'tools/current_runtime/source/manifest.json', CURRENT_PARTITION)
            planned = bridge.parse(manifest_raw)
            require(planned['runtime']['sha256'] == CURRENT_SOURCE
                    and planned['runtime']['functions'] == 797
                    and planned['expected_native_sha256'] == CURRENT_IMAGE
                    and len(planned['units']) == 15, 'exact current Half-logical source and image')
            source = inputs.bind(repository / 'tools/current_runtime/source/runtime.hepta', CURRENT_SOURCE)
            index_path = repository / 'tools/current_runtime/index_signbit_source_transition.py'
            inputs.bind(index_path)
            index_spec = importlib.util.spec_from_file_location('index_signbit_current_source', index_path)
            indexing = importlib.util.module_from_spec(index_spec); index_spec.loader.exec_module(indexing)
            source, index_inverse = indexing.restore_source(source, inputs.bind)
            require(index_inverse['functions'] == 797 and index_inverse['parent_functions'] == 775
                    and index_inverse['whole_source_inverse_exact'] is True
                    and index_inverse['whole_round_trip_exact'] is True,
                    'complete797 source restores every original775 byte before earlier source proofs')
            relu_path = repository / 'tools/current_runtime/typed_relu_source_transition.py'
            relu = load_module(inputs.bind(relu_path), relu_path, kind + '_typed_relu_source_inverse')
            source, relu_proof = relu.restore_source(source, inputs.bind)
            require(relu.SOURCE == indexing.PARENT_SOURCE and sha(source) == relu.PARENT_SOURCE
                    and relu_proof['functions'] == 775 and relu_proof['parent_functions'] == 773
                    and relu_proof['changed_functions'] == ['unary']
                    and relu_proof['whole_source_inverse_exact'] is True
                    and relu_proof['whole_round_trip_exact'] is True,
                    'complete typed ReLU775 inverse before the frozen same-F64 source')
            same_path = repository / 'tools/current_runtime/same_f64_arithmetic_source_transition.py'
            same = load_module(inputs.bind(same_path), same_path, kind + '_same_f64_arithmetic_source_inverse')
            source, same_proof = same.restore_source(source, inputs.bind)
            require(same.SOURCE == relu.PARENT_SOURCE and sha(source) == same.PARENT_SOURCE
                    and same_proof['functions'] == same_proof['parent_functions'] == 773
                    and same_proof['changed_functions'] == ['half_mul_priority','public_f64_binary']
                    and same_proof['whole_source_inverse_exact'] is True
                    and same_proof['whole_round_trip_exact'] is True,
                    'exact same-F64 route and singleton priority inverse before mixed773')
            arithmetic_path = repository / 'tools/current_runtime/public_f64_arithmetic_source_transition.py'
            arithmetic = load_module(inputs.bind(arithmetic_path), arithmetic_path, kind + '_public_f64_arithmetic_inverse')
            source, arithmetic_proof = arithmetic.restore_source(source, inputs.bind)
            require(arithmetic.SOURCE == same.PARENT_SOURCE and sha(source) == arithmetic.PARENT_SOURCE,
                    'exact mixedF64773 extension inverted before Half771')
            div_path = repository / 'tools/current_runtime/half_div_source_transition.py'
            division = load_module(inputs.bind(div_path), div_path, kind + '_half_div_source_inverse')
            source, div_proof = division.restore_source(source, inputs.bind)
            require(division.SOURCE == arithmetic.PARENT_SOURCE and sha(source) == division.PARENT_SOURCE,
                    'exact Half771 division inverted before wide767')
            wide_path = repository / 'tools/current_runtime/wide_neg_source_transition.py'
            wide = load_module(inputs.bind(wide_path), wide_path, kind + '_wide_neg_source_inverse')
            source, wide_proof = wide.restore_source(source, inputs.bind)
            require(wide.SOURCE == division.PARENT_SOURCE and sha(source) == wide.PARENT_SOURCE,
                    'exact wide767 negation inverted before Half765')
            addsub_path = repository / 'tools/current_runtime/half_addsub_source_transition.py'
            addsub = load_module(inputs.bind(addsub_path), addsub_path, kind + '_half_addsub_source_inverse')
            source, addsub_proof = addsub.restore_source(source, inputs.bind)
            require(addsub.SOURCE == wide.PARENT_SOURCE and sha(source) == addsub.PARENT_SOURCE,
                    'exact Half765 add/sub extension inverted before Half761')
            neg_path = repository / 'tools/current_runtime/half_neg_source_transition.py'
            neg = load_module(inputs.bind(neg_path), neg_path, kind + '_half_neg_source_inverse')
            source, neg_proof = neg.restore_source(source, inputs.bind)
            require(neg.SOURCE == addsub.PARENT_SOURCE and sha(source) == neg.PARENT_SOURCE,
                    'exact Half761 negation extension inverted before Half758')
            mul_path = repository / 'tools/current_runtime/half_mul_source_transition.py'
            mul = load_module(inputs.bind(mul_path), mul_path, kind + '_half_mul_source_inverse')
            source, mul_proof = mul.restore_source(source, inputs.bind)
            require(mul.SOURCE == neg.PARENT_SOURCE and sha(source) == mul.PARENT_SOURCE,
                    'exact Half758 multiplication extension inverted before BF16742')
            bf16_path = repository / 'tools/current_runtime/bf16_source_transition.py'
            bf16 = load_module(inputs.bind(bf16_path), bf16_path, kind + '_bf16_source_inverse')
            source, bf16_proof = bf16.restore_source(source, inputs.bind)
            require(bf16.SOURCE == mul.PARENT_SOURCE and sha(source) == bf16.PARENT_SOURCE,
                    'exact BF16 copy correction inverted before the original740 source witness')
            predicate_path = repository / 'tools/current_runtime/predicate_source_transition.py'
            predicate = load_module(inputs.bind(predicate_path, PREDICATE_INVERSE), predicate_path,
                                    kind + '_current_predicate_inverse')
            source, predicate_proof = predicate.restore_source(source, inputs.bind)
            require(predicate.SOURCE == bf16.PARENT_SOURCE and predicate.PARENT_SOURCE == HALF_SOURCE
                    and sha(source) == HALF_SOURCE,
                    'new predicates retain the exact Half parent and its original opcode routing')
            inverse_path = repository / 'tools/current_runtime/half_logical_source_transition.py'
            inverse = load_module(inputs.bind(inverse_path, HALF_LOGICAL_INVERSE), inverse_path,
                                  kind + '_current_half_logical_inverse')
            previous, extension_proof = inverse.restore_source(source, inputs.bind)
            require(inverse.SOURCE == HALF_SOURCE and inverse.PARENT_SOURCE == frozen_source
                    and sha(previous) == frozen_source
                    and extension_proof['changed_functions'] == ['si_dispatch_sum', 'logical_plan', 'logical_truth']
                    and extension_proof['unchanged_parent_functions'] == 733
                    and extension_proof['whole_source_inverse_exact'] is True,
                    'the exact three-function extension reconstructs every original736 byte')
            helper_path = bridge.HERE / 'source_transition.py'
            helper = load_module(inputs.bind(helper_path, bridge.SOURCE_HELPER), helper_path,
                                 kind + '_unchanged_frozen_source_inverse')
            restored, original_proof = helper.restore_source(previous, inputs.bind)
            require(restored == inputs.bind(candidate / 'candidate/runtime.hepta', bridge.PRODUCER_SOURCE),
                    'complete original producer source reconstructed through both inverses')
            original_admission = self.old['common'].candidate_admission(candidate, inputs)
            qpath = repository / 'tools/current_runtime/qualification.py'
            profile = load_module(inputs.bind(qpath, bridge.QUALIFICATION), qpath,
                                  kind + '_current_unchanged_shared_admission')
            require(profile.ROOT == repository and profile.PROFILE_BYTES == manifest_raw
                    and profile.SOURCE == CURRENT_SOURCE and profile.IMAGE == CURRENT_IMAGE
                    and profile.FUNCTIONS == 797 and profile.UNITS == 15,
                    'unchanged strict shared source/build admission for the actual current profile')
            inputs.bind(repository / 'tools/current_runtime/build.py', bridge.BUILDER)
            self.repository, self.profile = repository, profile
            result = dict(schema='heptatorch.half-logical.' + kind + '.source-admission.v1',
                source_sha256=CURRENT_SOURCE, native_sha256=CURRENT_IMAGE, functions=797, units=15,
                predicate_source_inverse=predicate_proof,
                bf16_source_inverse=bf16_proof,
                half_mul_source_inverse=mul_proof,
                half_neg_source_inverse=neg_proof,
                half_addsub_source_inverse=addsub_proof,
                wide_neg_source_inverse=wide_proof, half_div_source_inverse=div_proof,
                public_f64_arithmetic_source_inverse=arithmetic_proof,
                typed_relu_source_inverse=relu_proof,
                index_signbit_source_inverse=index_inverse,
                same_f64_arithmetic_source_inverse=same_proof,
                half_logical_to_original736_inverse=extension_proof,
                original736_source_sha256=frozen_source,
                current_source_manifest_sha256=CURRENT_PARTITION,
                strict_shared_admission_sha256=bridge.QUALIFICATION,
                owner_identity_binding=dict(name='F64_SOURCE_SHA256', original=frozen_source,
                    current=CURRENT_SOURCE, original_owner_source_sha256=bridge.OWNER,
                    original_function_bodies_changed=['bind_profile'],
                    original_owner_predicates_changed=False,
                    original_function_count=736, current_function_count=797,
                    only_exact_profile_count_guard_changed=True,
                    exact_profile_count_guards_changed=2),
                original_reference_producer_source_sha256=bridge.PRODUCER_SOURCE,
                original_reference_producer_manifest_sha256=producer,
                reader_identity_inverse=self.identity, frozen_reader_package_unchanged=True,
                numerical_expected_values_changed=False, prior_runtime_PASS_transferred=False,
                native_executed=False, Torch_executed=False, compiler_executed=False)
            if kind == 'f64':
                result.update(complete736_to719_inverse=original_proof,
                    original719_source_admission=original_admission,
                    original87_expectations_changed=True,
                    original87_requests_preserved=True, wide_neg_numeric_successors=1, original_strict55_failures_preserved=True,
                    original_boundary_contract_is_original719_scope=True,
                    original_sum_refusal_is_default_dtype_U64MAX=True,
                    current_explicit_Bool_sum_and_any_all_outside_this48_suite=True)
                return dict(schema='heptatorch.half-logical.f64-48.explicit-controls-source-admission.v1',
                    source_sha256=CURRENT_SOURCE, native_sha256=CURRENT_IMAGE, functions=797, units=15,
                    original_identity_source_admission=result,
                    original87_source_bytes_unchanged=True,
                    current87_expected_status_changes=self.control_proof,
                    two_factory_status_expectations_changed=True,
                    other85_expected_statuses_unchanged=False,
                    other84_expected_statuses_unchanged=True,
                    numerical_cases_comparison_and_allocation_loops_unchanged=True,
                    original_reference_namespace_preserved=True, original_strict55_FAIL_preserved=True,
                    historical_or_unmodified_original48_PASS_transferred=False,
                    native_executed=False, Torch_executed=False, compiler_executed=False)
            result.update(complete736_to737_inverse=original_proof,
                original737_source_admission=original_admission,
                original_numerical_and_boundary_predicates_changed=False,
                original239_frozen_refusal_recipes_preserved=True,
                original239_expectations_require_explicit_half_mul_successor=True,
                old12_standalone_refusals_transferred=False)
            return result

        def cold_admission(self, directory, image_sha, report_sha, *args):
            require(len(args) == (1 if kind == 'f64' else 2), 'original cold-admission argument shape')
            inputs = args[-1]
            require(self.repository == repository and self.profile is not None,
                    'current source admission precedes current build admission')
            if kind == 'half':
                require(Path(args[0]).resolve(strict=True) == repository, 'same actual Half candidate')
            require(image_sha == CURRENT_IMAGE, 'actual new image for strict cold admission')
            directory = Path(directory).resolve(strict=True)
            physical = {}
            def bound(path, expected=None):
                raw = inputs.bind(path, expected)
                key = str(Path(path).resolve(strict=True))
                physical[key] = inputs.rows[key]
                return raw
            built = self.profile.admit_build(directory, bound, report_sha)
            require(len(physical) == 73 and built['functions'] == 797
                    and len(built['units']) == 15 and len(built['processes']) == 16,
                    'all73 current physical inputs,15 fresh compilers and1 linker')
            require(inputs.bind(directory / 'manifest.json', CURRENT_PARTITION) == self.profile.PROFILE_BYTES,
                    'current source and actual cold-build manifests are identical')
            if kind == 'half':
                for path, row in built['inputs'].items():
                    inputs.bind(path, row['sha256'], row['bytes'])
            return dict(schema='heptatorch.half-logical.' + kind + '.cold-admission.v1',
                status='PASS_BUILD_ADMISSION_ONLY', source_sha256=CURRENT_SOURCE, native_sha256=CURRENT_IMAGE,
                build_sha256=report_sha, functions=797, units=15, physical_inputs=73, files=physical,
                compiler_processes=15, linker_processes=1,
                shared_admission_sha256=bridge.QUALIFICATION, builder_sha256=bridge.BUILDER,
                manifest_sha256=CURRENT_PARTITION, original_shared_predicates_unchanged=True,
                actual_original_build_inputs_rechecked=kind == 'half', link_request_magic='HIACGS01',
                four_argument_linker=True, historical_fixed_build_receipt_used=False,
                native_executed_by_admission=False, Torch_executed_by_admission=False,
                runtime_qualification_inferred=False)

        def make_owner(self, candidate_directory, image_sha, inputs):
            require(Path(candidate_directory).resolve(strict=True) == repository
                    and self.repository == repository and image_sha == CURRENT_IMAGE,
                    'full owner bound to actual new image and admitted repository')
            path = bridge.HERE / 'owner736.py'
            original_owner = inputs.bind(path, bridge.OWNER)
            # The frozen observer binds both its original owner and the additive
            # F64 wrapper. Each has its own exact function-count guard.
            guards = (
                b'    need(type(functions) is int and functions == 736, "sum02-actual-function-count-required")\n',
                b'    need(source_sha256 == F64_SOURCE_SHA256 and functions == 736,\n'
                b'         "f64-source-candidate-profile-required")\n',
            )
            require(original_owner.count(b'functions == 736') == len(guards)
                    and b'functions == 797' not in original_owner,
                    'exactly the two original owner profile-count guards')
            current_owner = original_owner
            replacements = []
            for before_guard in guards:
                after_guard = before_guard.replace(b'functions == 736', b'functions == 797')
                require(current_owner.count(before_guard) == 1 and after_guard not in current_owner,
                        'one complete original owner profile-count guard at each layer')
                current_owner = current_owner.replace(before_guard, after_guard, 1)
                replacements.append((before_guard, after_guard))
            restored_owner = current_owner
            for before_guard, after_guard in reversed(replacements):
                require(restored_owner.count(after_guard) == 1,
                        'one exact current owner profile-count guard at each layer')
                restored_owner = restored_owner.replace(after_guard, before_guard, 1)
            require(restored_owner == original_owner,
                    'complete original owner source inverse; all owner predicates stay unchanged')
            owner = load_module(current_owner, path, kind + '_complete_owner_current_profile')
            require(owner.F64_SOURCE_SHA256 == frozen_source and owner.SOURCE is None and owner.IMAGE is None,
                    'the complete original observer starts with its frozen identity and unbound image')
            # Source admission proves the original opcode paths. Only the source
            # identity and the exact function-count guard describe the new image;
            # every structural/numerical owner predicate remains original.
            owner.F64_SOURCE_SHA256 = CURRENT_SOURCE
            owner.bind_profile(CURRENT_SOURCE, CURRENT_IMAGE, 797)
            return owner

        def current_main(self):
            filename = 'derived719_run.py' if kind == 'f64' else 'derived737_run.py'
            path = bridge.HERE / filename
            raw = bridge.regular(path, bridge.DERIVED_RUN)
            namespace = dict(self.run.__dict__)
            namespace.update(HERE=bridge.HERE, SOURCE=CURRENT_SOURCE, IMAGE=CURRENT_IMAGE,
                bind_package=self.bind_package, candidate_admission=self.candidate_admission,
                cold_admission=self.cold_admission, make_owner=self.make_owner)
            if kind == 'f64':
                namespace.update(FROZEN_CANDIDATE=candidate, FROZEN_PRODUCER_MANIFEST=producer,
                                 Reference=self.reference)
                code = bridge.definition(raw, 'main')
            else:
                namespace.update(PRODUCER_SOURCE=bridge.PRODUCER_SOURCE, PRODUCER_MANIFEST=producer,
                                 groups=self.groups, comparison_functions=self.comparisons)
                tree = ast.parse(raw)
                functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main']
                require(len(functions) == 1, 'the exact original derived main')
                code = ast.Module(body=functions, type_ignores=[])
            exec(compile(code, str(path) + '::current-identity-main', 'exec'), namespace)
            require(self.run.Reference.__init__.__globals__['SOURCE'] == bridge.PRODUCER_SOURCE,
                    'the original reference namespace is never rebound')
            require(namespace['Driver'] is self.run.Driver and namespace['run_child'] is self.run.run_child,
                    'all original driver and child recipes are unchanged')
            return namespace['main']

    return CurrentAdapter


@contextlib.contextmanager
def current_reader(repository, reader_directory, wide_neg=None, public_f64_arithmetic=None):
    directory = Path(reader_directory).resolve(strict=True)
    entry_path = directory / 'current48.py'
    entry_raw = entry_path.read_bytes()
    require(sha(entry_raw) == '32196771085c4dbf24751ce9fa3b052a35655bd9c7cddb392d019174ffa50a44',
            'complete immutable original F64 entry')
    entry = load_module(entry_raw, entry_path, 'f64_current_original48_entry')
    bridge_path = directory / 'bridge.py'
    bridge_raw = bridge_path.read_bytes()
    require(sha(bridge_raw) == entry.BRIDGE, 'complete immutable original F64 bridge')
    bridge = load_module(bridge_raw, bridge_path, 'f64_current_original48_bridge')
    adapter = current_adapter_type(entry.adapter_type(bridge), bridge, repository,
                                   (Path(__file__),), 'f64')
    require(wide_neg is not None, 'the original F64 negation requires the actual shared wide_neg reference')
    successor_path = Path(repository).resolve() / 'tests/current_runtime_f64_neg_successor.py'
    successor = load_module(successor_path.read_bytes(), successor_path, 'current_f64_neg1_successor')
    adapter = successor.adapter_type(adapter, repository, directory, wide_neg)
    require(public_f64_arithmetic is not None, 'the original mixed F64 requests require the actual shared arithmetic reference')
    arithmetic_path = Path(repository).resolve() / 'tests/current_runtime_f64_arithmetic_successor.py'
    arithmetic = load_module(arithmetic_path.read_bytes(), arithmetic_path, 'current_f64_arithmetic23_successor')
    adapter = arithmetic.adapter_type(adapter, repository, directory, public_f64_arithmetic)
    with bridge.original_modules() as (originals, old_run, startup):
        yield adapter(originals, old_run, startup)


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reader-directory', type=Path, required=True)
    parser.add_argument('--wide-neg-reference-directory', type=Path, required=True)
    parser.add_argument('--wide-neg-reference-execution', type=Path, required=True)
    parser.add_argument('--public-f64-arithmetic-reference-directory', type=Path, required=True)
    parser.add_argument('--public-f64-arithmetic-reference-execution', type=Path, required=True)
    args, _ = parser.parse_known_args()
    original_argv = list(sys.argv)
    for flag in ('--reader-directory', '--wide-neg-reference-directory', '--wide-neg-reference-execution',
                 '--public-f64-arithmetic-reference-directory', '--public-f64-arithmetic-reference-execution'):
        at = sys.argv.index(flag)
        sys.argv = sys.argv[:at] + sys.argv[at + 2:]
    try:
        reference = dict(root=args.wide_neg_reference_directory, execution=args.wide_neg_reference_execution)
        arithmetic = dict(root=args.public_f64_arithmetic_reference_directory, execution=args.public_f64_arithmetic_reference_execution)
        with current_reader(args.candidate, args.reader_directory, reference, arithmetic) as adapter:
            return adapter.current_main()()
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    raise SystemExit(main())



