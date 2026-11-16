"""Qualify real owned Adam state without reclassifying frozen SGD reports."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import subprocess
import sys
import tarfile

SOURCE = '253438648a437a75ef2dbc77138eca74cb5fe6a6e2ea322c7c38e5a535e5d2a6'
IMAGE = 'fa92356954dbb9e0c454f2a58c2f78aa33950103d377c57e7b659ec9904183af'
BASE = '7c68a96b08aa9a21a52e2fd12be912c5f47dbde889e462f9a0dbb38cb3c09edf'
BASE_IMAGE = 'b6634158e079512e797d8237cca12d65052db18df23b12b8414adeb8d0b4b0e1'
REFERENCE = '7ad0a2e718300a33c91a9212f14ada3c0c7997be7fc77f64bb557c2ecd858160'
TEXT_MANIFESTS = {
    'owned-adam-package.sha256': ('d48fb1b29e3a41d78f007f6561bcfbd93385a250545bab8a2a7fc3927d904072', 54),
    'adam-object-review-package.sha256': ('c316382321ec57a591fb3bea335231ee701aa91e305b48bb38fef1e31cc5ab7f', 17),
    'tensor-rng-composition-package.sha256': ('3050c8205dce52ed707569c47afa3166b8c70df73a92e6ea448e087fc15f6c70', 38),
    'tensor-rng-nullable-package.sha256': ('1735945e726a5effaf616092b8b1213db8921ceea2908441a2b4f78b3062e15e', 20),
    'tensor-rng-package.sha256': ('d971907c9053c05b177a6ac51303a7602da384f2564cfdea7e0ab508fd51a6ca', 38),
}
JSON_MANIFESTS = {
    'owned-optimizer-package.json': ('506e1567e1428c70c7d534396067155606b20eca7539a86dcde183ab79ee6d82', 52),
    'optimizer-object-replay-package.json': ('74a133ea1735534f3d92a256474cc8172e84b3943e7566e1878375eea34eeadc', 283),
    'accumulator-lifetime-replay-package.json': ('2169549500e4ac297473f372540830d70835da56e37a87e640bf122b38c16494', 25),
    'backward-partial-package.json': ('33b7677b91a3615149c86322b5c4e624749e682e7331061367cfe35fba49e340', 39),
    'gradient-none-package.json': ('2c64b99ba392898b28581beb03a60d0e7b72790cc06c51843077be0f78fd66f4', 25),
}


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'baseline', 'stack-build', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    for key, value in vars(a).items():
        setattr(a, key, value.resolve())
    repo = Path(__file__).resolve().parents[1]
    package = repo / 'tools/owned_adam'
    review = repo / 'tests/adam_object_review'
    prior_review = repo / 'tests/optimizer_object_review'
    evidence = repo / 'validation/adam-object-review'
    partial = repo / 'tools/backward_partial'
    nullable = repo / 'tools/gradient_none'
    common = repo / 'tools/tensor_rng_composition'
    joint = repo / 'tools/tensor_rng_nullable'
    rng = repo / 'tools/tensor_rng'
    protocol = repo / 'tools/tensor_objects/protocol.py'
    observer = repo / 'tools/owned_sessions/client.py'
    compiler = a.stack_build / 'native/stack-module-compiler'
    linker = a.stack_build / 'native/stack-linker-session'
    inputs = {Path(__file__)}
    for path, expected in (
        (protocol, '5b99b0e508c2093fe830e509fffbb46b709f65e85b669ab6d79a32d425aa95ac'),
        (observer, 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'),
        (compiler, 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'),
        (linker, 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'),
    ):
        assert sha(path) == expected
        inputs.add(path)
    for name, (expected, count) in TEXT_MANIFESTS.items():
        manifest = repo / 'validation' / name
        assert sha(manifest) == expected
        rows = manifest.read_text().splitlines()
        assert len(rows) == count
        inputs.add(manifest)
        for row in rows:
            digest, relative = row.split('  ', 1)
            path = (repo / relative).resolve()
            assert path.is_relative_to(repo) and sha(path) == digest, relative
            inputs.add(path)
    for name, (expected, count) in JSON_MANIFESTS.items():
        manifest = repo / 'validation' / name
        assert sha(manifest) == expected
        rows = json.loads(manifest.read_bytes())['files']
        if isinstance(rows, list):
            rows = {row['path']: row for row in rows}
        assert len(rows) == count
        inputs.add(manifest)
        for relative, row in rows.items():
            path = (repo / relative).resolve()
            assert path.is_relative_to(repo) and sha(path) == row['sha256'] and path.stat().st_size == row['bytes']
            inputs.add(path)
    profile = json.loads((package / 'profile.json').read_bytes())
    for directory, filename, source, image, units in (
        (a.build, 'tensor-adam-session', SOURCE, IMAGE, 8),
        (a.baseline, 'tensor-optimizer-session', BASE, BASE_IMAGE, 7),
    ):
        assert sha(directory / 'runtime.hepta') == source and sha(directory / filename) == image
        build = json.loads((directory / 'build.json').read_bytes())
        assert build['status'] == 'PASS' and len(build['units']) == units
        inputs.update((directory / 'runtime.hepta', directory / filename, directory / 'build.json'))
        for unit in build['units']:
            for extension, key in (('.hepta', 'source_sha256'), ('.o', 'object_sha256')):
                path = directory / (unit['name'] + extension)
                assert sha(path) == unit[key]
                inputs.add(path)
        if directory == a.build:
            for key in ('source_sha256', 'native_sha256', 'compiler_sha256', 'linker_sha256',
                        'functions', 'blocks', 'instructions', 'baseline_functions_unchanged'):
                assert build[key] == profile[key], key
            assert [{k: u[k] for k in ('name', 'source_sha256', 'object_sha256')} for u in build['units']] == [
                {k: u[k] for k in ('name', 'source_sha256', 'object_sha256')} for u in profile['units']]
    reproduction_path = a.build / 'reproduction.json'
    reproduction = json.loads(reproduction_path.read_bytes())
    assert reproduction['status'] == 'PASS'
    assert reproduction['source_sha256'] == SOURCE and reproduction['native_sha256'] == IMAGE
    assert reproduction['native_modules'] == 8 and reproduction['all_source_object_image_bytes_identical'] is True
    assert reproduction['all_package_input_bytes_unchanged'] is True
    assert reproduction['host_authoring_aggregate_document_admission_claimed'] is False
    assert reproduction['all_complete_native_units_and_actual_linked_call_graph_checked'] is True
    # The frozen rebuild recipe checks all eight units and exact source/image
    # identities; this qualifier also re-reads every artifact independently.
    inputs.update((reproduction_path, a.build / 'adam-authoring.py'))
    before = {str(path): sha(path) for path in sorted(inputs)}
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    reports, hashes = {}, {}

    def execute(name, script, *arguments, report='report.json', output_is_file=False):
        target = out / (name + '.json' if output_is_file else name)
        command = [sys.executable, '-I', '-S', '-B', str(script), *map(str, arguments), '--output', str(target)]
        result = subprocess.run(command, cwd=repo, capture_output=True, timeout=300)
        (out / (name + '.stdout')).write_bytes(result.stdout)
        (out / (name + '.stderr')).write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr, (name, result.returncode, result.stderr[-3000:])
        path = target if output_is_file else target / report
        reports[name], hashes[name] = json.loads(path.read_bytes()), sha(path)

    manifest = json.loads((evidence / 'evidence/archive-manifest.json').read_bytes())
    archive = evidence / 'evidence/native-observations.tar.gz'
    assert sha(archive) == manifest['archive_sha256'] == 'f04c95e52674b7b02d1b580038c7d2ca97b11c12d7ab96caccd9694651f08f67'
    members = {row['path']: row for row in manifest['members']}
    assert len(members) == len(manifest['members']) == 229
    seen = set()
    previous = out / 'source-history/adam01'
    previous.mkdir(parents=True)
    with tarfile.open(archive, 'r:gz') as tf:
        for member in tf:
            assert member.isfile() and member.name in members and member.name not in seen
            row = members[member.name]
            source = tf.extractfile(member)
            digest, length = hashlib.sha256(), 0
            selected = member.name == 'source-baselines/adam01/runtime.hepta'
            sink = (previous / 'runtime.hepta').open('xb') if selected else None
            try:
                for block in iter(lambda: source.read(1 << 20), b''):
                    digest.update(block); length += len(block)
                    if sink is not None:
                        sink.write(block)
            finally:
                if sink is not None:
                    sink.close()
            assert digest.hexdigest() == row['sha256'] and length == row['bytes'] and member.mode == row['mode']
            seen.add(member.name)
    assert seen == set(members)
    assert sha(previous / 'runtime.hepta') == 'bd38b67c9f39859ddb81f304d9043af625bdcf3305d60e98ba988a6208e42885'
    execute('source-audit', review / 'audit_native_source.py', '--baseline', a.baseline,
            '--previous', previous, '--build', a.build, '--upstream-sources', package / 'upstream', output_is_file=True)
    audit = reports['source-audit']
    assert audit['status'] == 'PASS' and audit['current_source_sha256'] == SOURCE and audit['baseline_source_sha256'] == BASE
    assert audit['baseline_functions_unchanged'] == 308 and audit['floating_operations_unchanged']
    assert audit['four_access_changes_and_name_inverse_byte_identical'] and audit['source_custody_after']

    arguments = ['--binary', a.build / 'tensor-adam-session', '--source', a.build / 'runtime.hepta',
                 '--expected-sha256', IMAGE, '--expected-source-sha256', SOURCE]
    reference = evidence / 'torch-optimizer-objects.json'
    replay_args = arguments + ['--controls', review / 'native_release_controls.py', '--transport-observer', review / 'client.py']
    execute('optimizer-original', review / 'replay_optimizer_objects_v2.py', *replay_args,
            '--suite', 'optimizer', '--observations', reference)
    execute('adam-21-24', review / 'replay_adam_objects.py', *replay_args,
            '--legacy-replayer', review / 'replay_optimizer_objects_v2.py', '--observations', reference)
    old, added = reports['optimizer-original'], reports['adam-21-24']
    assert old['observations_sha256'] == added['actual_reference_sha256'] == REFERENCE
    assert (old['reference_cases'], old['qualified_exposed_scope'], old['partial_cases'], old['open_not_executed_cases'], old['unexpected_failures']) == (24, 18, 1, 5, 0)
    assert (added['original_actual_case_count'], added['qualified_exposed_scope'], added['failures']) == (24, 4, 0)
    assert (old['frames'], old['exact_comparisons'], old['rollback_bytes']) == (2332, 9535, 4587120)
    assert (added['frames'], added['exact_comparisons'], added['rollback_bytes']) == (908, 3469, 1223232)
    assert added['first20_test_source_modified'] is False and added['target_numerical_host_runtime'] is False
    assert added['executed_original_indices'] == [21, 22, 23, 24]
    actual = json.loads(reference.read_bytes())
    assert sha(reference) == REFERENCE and actual['cases'] == 24
    assert len(old['rows']) == len(actual['rows']) == 24
    assert [row['index'] for row in added['rows']] == [21, 22, 23, 24]
    old_rows = dict(enumerate(old['rows'], 1))
    added_rows = {row['index']: row for row in added['rows']}
    ledger = []
    for index, observed in enumerate(actual['rows'], 1):
        original = old_rows[index]
        assert original['case'] == observed['case']
        chosen = added_rows.get(index, original)
        assert chosen['case'] == original['case']
        if index in added_rows:
            assert original['status'] == 'OPEN_NOT_EXECUTED' and chosen['status'] == 'QUALIFIED_EXPOSED_SCOPE'
        ledger.append({'index': index, 'case': chosen['case'], 'status': chosen['status'],
                       'evidence': 'adam-21-24' if index in added_rows else 'optimizer-original'})
    assert Counter(row['status'] for row in ledger) == {'QUALIFIED_EXPOSED_SCOPE': 22, 'PARTIAL_WITH_RETAINED_NONLEAF_OPEN': 1, 'OPEN_NOT_EXECUTED': 1}
    assert ledger[13]['status'] == 'PARTIAL_WITH_RETAINED_NONLEAF_OPEN' and ledger[19]['status'] == 'OPEN_NOT_EXECUTED'
    execute('requires', prior_review / 'replay_optimizer_objects_v2.py', *replay_args,
            '--suite', 'requires', '--observations', repo / 'validation/optimizer-object-review/reference/torch-requires-grad.json')
    private_args = arguments + ['--replay', prior_review / 'replay_optimizer_objects_v2.py',
                               '--controls', prior_review / 'native_release_controls.py', '--observer', prior_review / 'client.py']
    execute('private-roots', prior_review / 'check_private_optimizer_roots.py', *private_args)
    original_accumulator = prior_review / 'replay_accumulator_lifetime.py'
    original = original_accumulator.read_bytes()
    edits = [(b"b'HTOPT002'", b"b'HTADAM01'"), (b"b'HTOOPT02'", b"b'HTOADAM1'")]
    adapted = original
    for old_text, new_text in edits:
        assert adapted.count(old_text) == 1 and new_text not in adapted
        adapted = adapted.replace(old_text, new_text)
    reverse = adapted
    for old_text, new_text in reversed(edits):
        assert reverse.count(new_text) == 1
        reverse = reverse.replace(new_text, old_text)
    assert reverse == original
    adapted_path = out / 'accumulator-observer.py'
    adapted_path.write_bytes(adapted)
    execute('accumulator-lifetime', adapted_path, *private_args,
            '--observations', repo / 'validation/optimizer-object-review/accumulator-lifetime-08/reference/torch-accumulator-lifetime.json')
    assert (reports['requires']['qualified_exposed_scope'], reports['requires']['unexpected_failures']) == (6, 0)
    assert reports['private-roots']['status'] == 'PASS'
    assert len(reports['private-roots']['frames']) == 72 and reports['private-roots']['whole_owner_rollback_bytes'] == 29066368
    assert (reports['accumulator-lifetime']['qualified'], reports['accumulator-lifetime']['unexpected_failures']) == (3, 0)

    execute('native', package / 'test_native.py', *arguments, '--protocol', protocol)
    common_args = ['--binary', a.build / 'tensor-adam-session', '--source', a.build / 'runtime.hepta',
                   '--build-report', a.build / 'build.json', '--protocol', protocol]
    execute('guards', package / 'test_guards.py', *common_args)
    execute('step-counter', package / 'test_step_counter.py', *common_args)
    execute('legacy-sgd-controls', repo / 'tools/owned_optimizers/test_native.py', *arguments, '--protocol', protocol)
    expected = {'native': {'frames': 148, 'exact_values': 55, 'rollback_bytes': 917424},
                'guards': {'case_count': 5, 'frames': 142, 'rejections': 57, 'rollback_bytes': 22373904},
                'step-counter': {'cases': 11, 'frames': 72, 'rollback_bytes': 1834848, 'None_skip_precedes_step_domain_validation': True},
                'legacy-sgd-controls': {'frames': 100, 'exact_values': 29, 'rollback_bytes': 1834848}}
    for name, wanted in expected.items():
        assert reports[name]['status'] == 'PASS'
        assert all(reports[name][key] == value for key, value in wanted.items()), name
    assert len(reports['step-counter']['rows']) == 11 and all(row['status'] == 'PASS' for row in reports['step-counter']['rows'])
    execute('observer-adaptation', package / 'adapt_regressions.py', '--grad-clear', nullable / 'replay_grad_clear.py',
            '--common-joint', common / 'test_joint.py', '--nullable-joint', joint / 'test_combination.py', report='adaptation.json')
    adaptation = reports['observer-adaptation']
    assert adaptation['status'] == 'PASS' and adaptation['native_executable_sha256'] == IMAGE
    assert all(adaptation[key] is False for key in ('changed_input_packets', 'changed_expected_numeric_values', 'changed_numerical_tolerances', 'changed_error_or_commit_expectations'))
    old_replay = arguments + ['--controls', partial / 'native_release_controls.py', '--transport-observer', observer]
    for name, script, observation in (
        ('signedzero', partial / 'replay_gradient_signed_zero.py', partial / 'evidence/actual-torch-signed-zero-observations.json'),
        ('partial', partial / 'replay_partial_observations.py', partial / 'evidence/actual-torch-16-observations.json'),
        ('presence', partial / 'replay_gradient_presence.py', partial / 'evidence/actual-torch-gradient-presence-observations.json'),
        ('grad-clear', out / 'observer-adaptation/replay_grad_clear.py', nullable / 'evidence/actual-torch-grad-clear-observations.json'),
    ):
        execute(name, script, *old_replay, '--observations', observation)
    assert (reports['signedzero']['passed'], reports['signedzero']['different_word_observations']) == (13, 0)
    assert reports['partial']['passed'] == 16
    presence, clear = reports['presence'], reports['grad-clear']
    assert (presence['cases'], presence['unexpected_failures'], presence['adam_compared_values_with_different_bits'], presence['state_version_difference_observations'], presence['state_creation_difference_observations']) == (8, 0, 0, 8, 65)
    assert (clear['cases'], clear['qualified'], clear['unexpected_failures'], clear['Adam_different_bits'], clear['state_version_difference_observations'], clear['state_creation_difference_observations'], clear['native_step_Tensor_absence_observations']) == (10, 10, 0, 6, 11, 9, 30)
    assert clear['numeric_policy']['Adam_updated_values'] == {'rtol': 4e-5, 'atol': 5e-6}
    mt = out / 'independent-mt-reference'
    result = subprocess.run(['g++', '-std=c++17', '-O2', str(rng / 'reference.cpp'), '-o', str(mt)], capture_output=True, timeout=60)
    (out / 'reference-build.stdout').write_bytes(result.stdout)
    (out / 'reference-build.stderr').write_bytes(result.stderr)
    assert result.returncode == 0 and not result.stderr
    rng_args = ['--executable', a.build / 'tensor-adam-session', '--rng-package', rng, '--cpp-reference', mt]
    execute('common-rng', out / 'observer-adaptation/test_joint.py', *rng_args)
    execute('nullable-rng', out / 'observer-adaptation/test_combination.py', *rng_args)
    assert (reports['common-rng']['case_count'], reports['common-rng']['frames'], reports['common-rng']['full_generator_states'], reports['common-rng']['joint_owner_rollbacks']) == (3, 155, 14, 5)
    assert len(reports['nullable-rng']['cases']) == 4
    assert (reports['nullable-rng']['frames'], reports['nullable-rng']['exact_words'], reports['nullable-rng']['full_generator_states'], reports['nullable-rng']['rollback_bytes']) == (56, 393, 34, 2517280)
    execute('handoff-build', package / 'build_state_handoff.py', '--predecessor', 'optimizer008',
            '--old', a.baseline, '--new', a.build, '--compiler', compiler, '--linker', linker)
    execute('handoff', package / 'test_state_handoff.py', '--executable', out / 'handoff-build/state-handoff-session',
            '--build-receipt', out / 'handoff-build/report.json', '--protocol', protocol)
    assert reports['handoff-build']['status'] == reports['handoff']['status'] == 'PASS'
    assert reports['handoff-build']['native_sha256'] == reports['handoff']['native_sha256'] == profile['handoff_sha256']
    assert reports['handoff']['old_source_sha256'] == BASE and reports['handoff']['new_source_sha256'] == SOURCE
    assert (reports['handoff']['cases'], reports['handoff']['frames'], reports['handoff']['preserved_owner_bytes']) == (2, 49, 611616)
    assert len(reports['handoff']['rows']) == 2 and all(row['status'] == 'PASS' for row in reports['handoff']['rows'])
    for name, report in reports.items():
        for key in ('native_executable_sha256', 'native_sha256', 'executable_sha256'):
            if key in report and name not in ('handoff-build', 'handoff'):
                assert report[key] == IMAGE, (name, key)
        for key in ('native_source_sha256', 'source_sha256'):
            if key in report:
                assert report[key] == SOURCE, (name, key)
        if 'oracle_provenance' in report:
            provenance = report['oracle_provenance']
            assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11
            assert all(value is True for value in provenance['checks'].values())
    after = {str(path): sha(path) for path in sorted(inputs)}
    assert before == after
    result = {
        'schema': 'heptatorch.integrated-owned-adam-qualification.v1', 'status': 'QUALIFIED_EXPOSED_ADAM_WITH_OPEN_SCOPE',
        'native_source_sha256': SOURCE, 'native_executable_sha256': IMAGE,
        'baseline_source_sha256': BASE, 'baseline_executable_sha256': BASE_IMAGE,
        'source_custody_before': before, 'source_custody_after': after,
        'cold_reproduction': reproduction, 'reports': reports, 'original_report_sha256': hashes,
        'actual_reference_sha256': REFERENCE, 'reference_cases': 24, 'native_executed_cases': 23,
        'qualified_exposed_scope': 22, 'partial_cases': 1, 'open_not_executed_cases': 1, 'case_ledger': ledger,
        'original_first20_replay_and_assertions_unchanged': True, 'original_old24_report_reclassified': False,
        'accumulator_read_only_adapter': {'original_sha256': hashlib.sha256(original).hexdigest(),
            'adapted_sha256': hashlib.sha256(adapted).hexdigest(), 'inverse_byte_identical': True,
            'only_edits': [{'before': old.decode(), 'after': new.decode()} for old, new in edits]},
        'independent_history_archive_members_verified': len(seen),
        'host_production_numerical_runtime': False, 'pytorch_executed_here': False,
        'all_actual_native_units_and_linked_call_graph_checked': True,
        'public_grad_object_scope': 'OPEN', 'retained_nonleaf_parameter_scope': 'PARTIAL',
        'arbitrary_real_external_step_exponents': 'OPEN', 'general_optimizer_error_partial_mutation': 'OPEN',
        'legacy_explicit_state_differences_preserved': True,
        'full_optimizer_complete': False, 'full_pytorch_rewrite_complete': False, 'pytorch_api_denominator': 2590,
        'count_policy': 'The 24-case ledger joins disjoint executed indices only; regression executions and controls do not add upstream schema coverage.',
    }
    assert len(reports) == 19
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'reports': len(reports), 'qualified': 22, 'partial': 1, 'open': 1,
                      'report_sha256': sha(out / 'report.json')}))


if __name__ == '__main__':
    main()
