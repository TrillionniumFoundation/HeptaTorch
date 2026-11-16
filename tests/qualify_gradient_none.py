"""Qualify frozen standalone and joint HIA nullable-gradient owner contracts."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

NULL_SOURCE = 'cebd8b474c2604c635bdc39a8a4e75721ff04a0544c5cbff9b706d5e5c85b126'
NULL_IMAGE = 'e84156898c42a69ca92ecc58d7e086098d356cbb9ddf6fe3e8c69b09287cfcad'
JOINT_SOURCE = '490fc144160f6e4440aed001363f86ccff56c335f9781e38b058fc6106b1be09'
JOINT_IMAGE = '5baa9cd5e5625f0e2324941486982b714dcdedfe897f5c16686a3ac8882ce242'
JSON_MANIFESTS = {
    'gradient-none-package.json': ('2c64b99ba392898b28581beb03a60d0e7b72790cc06c51843077be0f78fd66f4', 25),
    'backward-partial-package.json': ('33b7677b91a3615149c86322b5c4e624749e682e7331061367cfe35fba49e340', 39),
}
TEXT_MANIFESTS = {
    'tensor-rng-nullable-package.sha256': ('1735945e726a5effaf616092b8b1213db8921ceea2908441a2b4f78b3062e15e', 20),
    'tensor-rng-composition-package.sha256': ('3050c8205dce52ed707569c47afa3166b8c70df73a92e6ea448e087fc15f6c70', 38),
    'tensor-rng-package.sha256': ('d971907c9053c05b177a6ac51303a7602da384f2564cfdea7e0ab508fd51a6ca', 38),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('nullable-build', 'build', 'previous-common', 'materialization', 'stack-build', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        setattr(args, key, value.resolve())
    repo = Path(__file__).resolve().parents[1]
    nullable, joint, partial, rng, common = (repo / 'tools' / name for name in (
        'gradient_none', 'tensor_rng_nullable', 'backward_partial', 'tensor_rng', 'tensor_rng_composition'))
    protocol = repo / 'tools/tensor_objects/protocol.py'
    observer = repo / 'tools/owned_sessions/client.py'
    compiler = args.stack_build / 'native/stack-module-compiler'
    linker = args.stack_build / 'native/stack-linker-session'
    assert sha(protocol) == '5b99b0e508c2093fe830e509fffbb46b709f65e85b669ab6d79a32d425aa95ac'
    assert sha(observer) == 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
    assert sha(compiler) == 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
    assert sha(linker) == 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
    inputs = {Path(__file__), protocol, observer, compiler, linker}
    for name, (pin, count) in JSON_MANIFESTS.items():
        path = repo / 'validation' / name
        assert sha(path) == pin
        files = json.loads(path.read_bytes())['files']
        assert len(files) == count
        inputs.add(path)
        for relative, expected in files.items():
            target = (repo / relative).resolve()
            assert target.is_relative_to(repo) and sha(target) == expected['sha256'], relative
            assert target.stat().st_size == expected['bytes'], relative
            inputs.add(target)
    for name, (pin, count) in TEXT_MANIFESTS.items():
        path = repo / 'validation' / name
        assert sha(path) == pin
        lines = path.read_text().splitlines()
        assert len(lines) == count
        inputs.add(path)
        for line in lines:
            digest, relative = line.split('  ', 1)
            target = (repo / relative).resolve()
            assert target.is_relative_to(repo) and sha(target) == digest, relative
            inputs.add(target)
    profiles = {
        'standalone': (args.nullable_build, 'tensor-nullable-session', NULL_SOURCE, NULL_IMAGE),
        'joint': (args.build, 'tensor-rng-session', JOINT_SOURCE, JOINT_IMAGE),
        'previous-common': (args.previous_common, 'tensor-rng-session',
                            '7447c169d3b2cad1e55cb7b1b8970d67f4103c5866f6db0a60c0b4d05d02cc0b',
                            'cdcaebb2785639585de7fd2a4e1f98d41a56e885f1657c3485ce5f187f89d726'),
        'materialization': (args.materialization, 'tensor-partial-session',
                            '9feab47afea6723284e15a2ad6c9071dc8c5bdc27b572d2eef4ab4628ed1aea9',
                            '4a62b796d38317e630381b7e2e57e8e7824d78c1f721e3d69bd59f59e04ca894'),
    }
    for directory, image, core_pin, image_pin in profiles.values():
        assert sha(directory / 'runtime.hepta') == core_pin and sha(directory / image) == image_pin
        build = json.loads((directory / 'build.json').read_bytes())
        inputs.update((directory / 'runtime.hepta', directory / image, directory / 'build.json'))
        for unit in build['units']:
            source = directory / (unit['name'] + '.hepta')
            obj = directory / (unit['name'] + '.o')
            assert sha(source) == unit['source_sha256'] and sha(obj) == unit['object_sha256']
            inputs.update((source, obj))
    composition = json.loads((args.build / 'build.json').read_bytes())
    assert composition['common_functions_unchanged'] == 264
    assert composition['entry_byte_identical'] and composition['rng_module_byte_identical'] and composition['owner_module_byte_identical']
    assert composition['host_production_numerical_runtime'] is False
    reproduction = json.loads((args.build / 'reproduction.json').read_bytes())
    assert reproduction['status'] == 'PASS' and reproduction['native_modules'] == 7
    assert reproduction['all_source_object_image_bytes_identical'] is True
    inputs.add(args.build / 'reproduction.json')
    before = {str(path): sha(path) for path in sorted(inputs)}
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    reports = {}
    report_hashes = {}

    def execute(name, script, *arguments, report='report.json'):
        command = [sys.executable, '-I', '-S', '-B', str(script), *map(str, arguments), '--output', str(out / name)]
        result = subprocess.run(command, cwd=repo, capture_output=True, timeout=300)
        (out / (name + '.stdout')).write_bytes(result.stdout)
        (out / (name + '.stderr')).write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr, (name, result.returncode, result.stderr[-3000:])
        path = out / name / report
        reports[name] = json.loads(path.read_bytes())
        report_hashes[name] = sha(path)

    execute('observer-adaptation', joint / 'adapt_tests.py', '--nullable-package', nullable,
            '--common-package', common, report='adaptation.json')
    assert reports['observer-adaptation'] == json.loads((joint / 'evidence/private-observer-adaptation.json').read_bytes())
    for label in ('standalone', 'joint'):
        directory, binary, core_pin, image_pin = profiles[label]
        test_source = nullable if label == 'standalone' else out / 'observer-adaptation'
        arguments = ['--binary', directory / binary, '--source', directory / 'runtime.hepta',
                     '--expected-sha256', image_pin, '--expected-source-sha256', core_pin]
        execute(label + '-nullable', test_source / 'test_nullable.py', *arguments, '--protocol', protocol)
        r = reports[label + '-nullable']
        assert r['status'] == 'PASS' and len(r['rows']) == 6 and all(row['status'] == 'PASS' for row in r['rows'])
        assert (r['frames'], r['exact_gradient_words'], r['committed_node_checks']) == (179, 131101, 261)
        assert r['rollback_bytes'] == (10638464 if label == 'standalone' else 10671232)
        replay = arguments + ['--controls', partial / 'native_release_controls.py', '--transport-observer', observer]
        execute(label + '-grad-clear', test_source / 'replay_grad_clear.py', *replay,
                '--observations', nullable / 'evidence/actual-torch-grad-clear-observations.json')
        r = reports[label + '-grad-clear']
        assert r['status'] == 'QUALIFIED_GRAD_CLEAR_WITH_OPEN_OBJECT_SCOPE'
        assert r['cases'] == r['qualified'] == 10 and r['unexpected_failures'] == 0
        assert (r['frames'], r['exact_comparisons'], r['Adam_tolerance_comparisons'], r['Adam_different_bits']) == (780, 2669, 288, 6)
        assert (r['state_version_difference_observations'], r['state_creation_difference_observations'], r['native_step_Tensor_absence_observations']) == (11, 9, 30)
        assert r['numeric_policy']['Adam_updated_values'] == {'rtol': 4e-5, 'atol': 5e-6}
        for name, script, observation in (
            ('signedzero', 'replay_gradient_signed_zero.py', 'actual-torch-signed-zero-observations.json'),
            ('partial', 'replay_partial_observations.py', 'actual-torch-16-observations.json'),
            ('presence', 'replay_gradient_presence.py', 'actual-torch-gradient-presence-observations.json'),
        ):
            execute(label + '-' + name, partial / script, *replay, '--observations', partial / 'evidence' / observation)
        zero, partial_report, presence = (reports[label + '-' + name] for name in ('signedzero', 'partial', 'presence'))
        assert zero['status'] == 'PASS' and zero['cases'] == zero['passed'] == 13 and zero['failed'] == 0
        assert (zero['frames'], zero['exact_binary32_word_comparisons'], zero['different_word_observations']) == (166, 70, 0)
        assert partial_report['status'] == 'PASS' and partial_report['cases'] == partial_report['passed'] == 16
        assert (partial_report['frames'], partial_report['exact_scalar_or_binary32_comparisons']) == (606, 1587)
        assert presence['status'] == 'QUALIFIED_WITH_OPEN_STATE_LIFECYCLE'
        assert presence['cases'] == presence['parameter_values_versions_gradient_presence_and_skip_passed'] == 8
        assert presence['unexpected_failures'] == 0
        assert (presence['frames'], presence['exact_scalar_or_binary32_comparisons'], presence['adam_tolerance_comparisons'], presence['adam_compared_values_with_different_bits']) == (880, 2088, 180, 0)
        assert (presence['state_version_difference_observations'], presence['state_creation_difference_observations']) == (8, 65)
        for item in (r, zero, partial_report, presence):
            assert item['native_executable_sha256'] == image_pin and item['native_source_sha256'] == core_pin
            provenance = item['oracle_provenance']
            assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11
            assert all(value is True for value in provenance['checks'].values())

    execute('standalone-handoff-build', nullable / 'build_state_handoff.py',
            '--old', args.materialization, '--new', args.nullable_build, '--compiler', compiler, '--linker', linker)
    execute('standalone-handoff', nullable / 'test_state_handoff.py',
            '--executable', out / 'standalone-handoff-build/state-handoff-session',
            '--build-receipt', out / 'standalone-handoff-build/report.json', '--protocol', protocol)
    handoff = reports['standalone-handoff']
    assert handoff['status'] == 'PASS' and (handoff['cases'], handoff['frames'], handoff['entire_owner_bytes_unchanged']) == (2, 31, 595232)
    historical = json.loads((nullable / 'evidence/legal-handoff-build.json').read_bytes())
    assert reports['standalone-handoff-build']['native_sha256'] == historical['native_sha256']
    reference = out / 'independent-mt-reference'
    result = subprocess.run(['g++', '-std=c++17', '-O2', str(rng / 'reference.cpp'), '-o', str(reference)], capture_output=True, timeout=60)
    (out / 'reference-build.stdout').write_bytes(result.stdout)
    (out / 'reference-build.stderr').write_bytes(result.stderr)
    assert result.returncode == 0 and not result.stderr
    rng_arguments = ['--executable', args.build / 'tensor-rng-session', '--rng-package', rng, '--cpp-reference', reference]
    execute('previous-joint', out / 'observer-adaptation/test_joint.py', *rng_arguments)
    r = reports['previous-joint']
    assert r['status'] == 'PASS' and (r['case_count'], r['frames']) == (3, 155)
    assert (r['full_generator_states'], r['joint_owner_rollbacks'], r['stage_zero_observations']) == (14, 5, 42)
    execute('joint-combination', joint / 'test_combination.py', *rng_arguments)
    r = reports['joint-combination']
    assert r['status'] == 'PASS' and len(r['cases']) == 4 and all(row['status'] == 'PASS' for row in r['cases'])
    assert (r['frames'], r['exact_words'], r['full_generator_states'], r['rollback_bytes'], r['committed_node_invariant_checks']) == (56, 393, 34, 2500896, 62)
    execute('joint-handoff', joint / 'test_handoffs.py', '--build', args.build, '--rng-package', rng,
            '--stack-build', args.stack_build, '--peer-pins', joint / 'handoff-peer-pins.json',
            '--peer', 'common05=' + str(args.previous_common), '--peer', 'nullable70=' + str(args.nullable_build))
    handoff = reports['joint-handoff']
    assert handoff['status'] == 'PASS' and (handoff['cases'], handoff['frames']) == (4, 17)
    assert handoff['all_source_custody_unchanged'] and all(row['inverse_byte_identical'] for row in handoff['source_bindings'])
    historical = json.loads((joint / 'evidence/two-predecessor-handoff-01.json').read_bytes())
    assert {row['name']: row['qualification_executable_sha256'] for row in handoff['peers']} == {
        row['name']: row['qualification_executable_sha256'] for row in historical['peers']}
    assert all(row['new_state_has_genuinely_cleared_nullable_leaf'] and row['both_closed_owner_mappings_absent']
               and row['owner_pointer_length_and_bytes_preserved'] for row in handoff['peers'])
    after = {str(path): sha(path) for path in sorted(inputs)}
    assert before == after
    report = {
        'schema': 'heptatorch.integrated-nullable-gradient-qualification.v1',
        'status': 'QUALIFIED_GRAD_CLEAR_WITH_OPEN_OBJECT_SCOPE', 'nullable_native_contracts': 'PASS',
        'standalone_source_sha256': NULL_SOURCE, 'standalone_executable_sha256': NULL_IMAGE,
        'joint_source_sha256': JOINT_SOURCE, 'joint_executable_sha256': JOINT_IMAGE,
        'source_custody_before': before, 'source_custody_after': after, 'reports': reports,
        'original_report_sha256': report_hashes, 'observer_only_adaptation': True,
        'separate_open_observation_sets': {
            'ten_clear_cases': {'momentum_Version': 11, 'lazy_state_creation': 9, 'missing_step_Tensor': 30},
            'eight_presence_cases': {'momentum_Version': 8, 'lazy_state_creation': 65},
            'aggregation_policy': 'Repeated runs and different observation sets are never added into API coverage.'},
        'raw_Adam_bit_differences_retained_per_clear_replay': 6,
        'pytorch_executed_in_this_process': False, 'reference_is_actual_pinned_hosted_observation': True,
        'public_grad_Tensor_identity_Version_and_alias_lifetime': 'OPEN',
        'optimizer_object_state_lifecycle': 'OPEN',
        'pytorch_api_denominator': 2590, 'whole_pytorch_rewrite_complete': False}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'nullable_native_contracts': 'PASS',
                      'report_sha256': sha(out / 'report.json')}))


if __name__ == '__main__':
    main()
