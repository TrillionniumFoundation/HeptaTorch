"""Reproduce source-bound owned SGD, captured edges, and historical failures."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import tarfile

SOURCE = '7c68a96b08aa9a21a52e2fd12be912c5f47dbde889e462f9a0dbb38cb3c09edf'
IMAGE = 'b6634158e079512e797d8237cca12d65052db18df23b12b8414adeb8d0b4b0e1'
OLD_SOURCE = '1b143984d6fce05ba9a92f5f528c7a1b59b4e8aaa9ceae2c6e2299c43a651297'
OLD_IMAGE = 'b86215580b319a3ad3f030b2bf3e41febd6dffb3fe15212601d60e019fea9991'
MANIFESTS = {
    'owned-optimizer-package.json': ('506e1567e1428c70c7d534396067155606b20eca7539a86dcde183ab79ee6d82', 52),
    'optimizer-object-replay-package.json': ('74a133ea1735534f3d92a256474cc8172e84b3943e7566e1878375eea34eeadc', 283),
    'accumulator-lifetime-replay-package.json': ('2169549500e4ac297473f372540830d70835da56e37a87e640bf122b38c16494', 25),
    'backward-partial-package.json': ('33b7677b91a3615149c86322b5c4e624749e682e7331061367cfe35fba49e340', 39),
    'gradient-none-package.json': ('2c64b99ba392898b28581beb03a60d0e7b72790cc06c51843077be0f78fd66f4', 25),
}
TEXT_MANIFESTS = {
    'tensor-rng-nullable-package.sha256': ('1735945e726a5effaf616092b8b1213db8921ceea2908441a2b4f78b3062e15e', 20),
    'tensor-rng-composition-package.sha256': ('3050c8205dce52ed707569c47afa3166b8c70df73a92e6ea448e087fc15f6c70', 38),
    'tensor-rng-package.sha256': ('d971907c9053c05b177a6ac51303a7602da384f2564cfdea7e0ab508fd51a6ca', 38),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'nullable-build', 'stack-build', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    for key, value in vars(a).items():
        setattr(a, key, value.resolve())
    repo = Path(__file__).resolve().parents[1]
    package = repo / 'tools/owned_optimizers'
    review = repo / 'tests/optimizer_object_review'
    evidence = repo / 'validation/optimizer-object-review'
    partial = repo / 'tools/backward_partial'
    nullable = repo / 'tools/gradient_none'
    common = repo / 'tools/tensor_rng_composition'
    joint = repo / 'tools/tensor_rng_nullable'
    rng = repo / 'tools/tensor_rng'
    protocol = repo / 'tools/tensor_objects/protocol.py'
    observer = repo / 'tools/owned_sessions/client.py'
    compiler = a.stack_build / 'native/stack-module-compiler'
    linker = a.stack_build / 'native/stack-linker-session'
    pins = {
        protocol: '5b99b0e508c2093fe830e509fffbb46b709f65e85b669ab6d79a32d425aa95ac',
        observer: 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b',
        compiler: 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f',
        linker: 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16',
    }
    assert all(sha(path) == pin for path, pin in pins.items())
    inputs = set(pins) | {Path(__file__)}
    for name, (pin, count) in MANIFESTS.items():
        path = repo / 'validation' / name
        assert sha(path) == pin
        files = json.loads(path.read_bytes())['files']
        if isinstance(files, list):
            files = {row['path']: row for row in files}
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
    for directory, binary, source_pin, image_pin in (
        (a.build, 'tensor-optimizer-session', SOURCE, IMAGE),
        (a.nullable_build, 'tensor-rng-session',
         '490fc144160f6e4440aed001363f86ccff56c335f9781e38b058fc6106b1be09',
         '5baa9cd5e5625f0e2324941486982b714dcdedfe897f5c16686a3ac8882ce242'),
    ):
        assert sha(directory / 'runtime.hepta') == source_pin and sha(directory / binary) == image_pin
        receipt = json.loads((directory / 'build.json').read_bytes())
        assert receipt['status'] == 'PASS' and len(receipt['units']) == 7
        inputs.update((directory / 'runtime.hepta', directory / binary, directory / 'build.json'))
        for unit in receipt['units']:
            source, obj = (directory / (unit['name'] + suffix) for suffix in ('.hepta', '.o'))
            assert sha(source) == unit['source_sha256'] and sha(obj) == unit['object_sha256']
            inputs.update((source, obj))
    reproduction = json.loads((a.build / 'reproduction.json').read_bytes())
    assert reproduction['status'] == 'PASS' and reproduction['native_modules'] == 7
    assert reproduction['all_source_object_image_bytes_identical'] is True
    assert reproduction['host_authoring_aggregate_document_admission_claimed'] is False
    assert reproduction['all_complete_native_units_and_actual_linked_call_graph_checked'] is True
    inputs.add(a.build / 'reproduction.json')
    boundaries = json.loads((package / 'authoring-boundaries.json').read_bytes())
    assert boundaries['aggregate_document_admission_claimed'] is False
    assert boundaries['native_module_admission_required'] is True
    assert boundaries['baseline_old_and_new_boundary_function_bytes_identical'] is True
    before = {str(path): sha(path) for path in sorted(inputs)}
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    reports, report_hashes = {}, {}

    def execute(name, script, *arguments, report='report.json', expected_exit=0):
        command = [sys.executable, '-I', '-S', '-B', str(script), *map(str, arguments), '--output', str(out / name)]
        result = subprocess.run(command, cwd=repo, capture_output=True, timeout=300)
        (out / (name + '.stdout')).write_bytes(result.stdout)
        (out / (name + '.stderr')).write_bytes(result.stderr)
        assert result.returncode == expected_exit and not result.stderr, (name, result.returncode, result.stderr[-3000:])
        path = out / name / report
        reports[name] = json.loads(path.read_bytes())
        report_hashes[name] = sha(path)

    history = json.loads((package / 'evidence/historical-builds-and-failures.json').read_bytes())
    archive = package / 'evidence' / history['archive']
    assert sha(archive) == history['archive_sha256']
    expected = {row['path']: row for row in history['files']}
    with tarfile.open(archive, 'r:xz') as tf:
        members = tf.getmembers()
        assert len({member.name for member in members}) == len(members)
        assert {m.name for m in members} == set(expected) | {'ARCHIVE-MANIFEST.json'}
        for member in members:
            assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            raw = tf.extractfile(member).read()
            if member.name == 'ARCHIVE-MANIFEST.json':
                continue
            row = expected[member.name]
            assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
            if member.name.startswith('candidate-06/'):
                destination = out / 'historical' / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
    old = out / 'historical/candidate-06'
    assert sha(old / 'runtime.hepta') == OLD_SOURCE and sha(old / 'tensor-optimizer-session') == OLD_IMAGE
    (old / 'tensor-optimizer-session').chmod(0o755)
    arguments = ['--binary', a.build / 'tensor-optimizer-session', '--source', a.build / 'runtime.hepta',
                 '--expected-sha256', IMAGE, '--expected-source-sha256', SOURCE]
    replay = arguments + ['--controls', review / 'native_release_controls.py', '--transport-observer', review / 'client.py']
    for suite, reference_name, counts in (
        ('optimizer', 'torch-optimizer-objects.json', (24, 18, 1, 5, 2332, 9535)),
        ('requires', 'torch-requires-grad.json', (6, 6, 0, 0, 600, 2489)),
    ):
        observations = evidence / 'reference' / reference_name
        execute(suite, review / 'replay_optimizer_objects_v2.py', *replay, '--suite', suite, '--observations', observations)
        r = reports[suite]
        assert r['status'] == 'QUALIFIED_WITH_EXPLICIT_OPEN_SCOPE' and r['unexpected_failures'] == 0
        assert tuple(r[key] for key in ('reference_cases', 'qualified_exposed_scope', 'partial_cases', 'open_not_executed_cases', 'frames', 'exact_comparisons')) == counts
        execute('historical-' + suite, review / 'replay_optimizer_objects_v2.py',
                '--binary', old / 'tensor-optimizer-session', '--source', old / 'runtime.hepta',
                '--expected-sha256', OLD_IMAGE, '--expected-source-sha256', OLD_SOURCE,
                '--controls', review / 'native_release_controls.py', '--transport-observer', review / 'client.py',
                '--suite', suite, '--observations', observations, expected_exit=1)
        r = reports['historical-' + suite]
        previous = evidence / 'trajectories' / ('hosted-optimizer-native-06' if suite == 'optimizer' else 'hosted-requires-native-06-v2') / 'report.json'
        original = json.loads(previous.read_bytes())
        assert r['status'] == 'FAIL' and r['unexpected_failures'] == 1
        assert [(row['case'], row['status']) for row in r['rows']] == [(row['case'], row['status']) for row in original['rows']]
        assert tuple(r[key] for key in ('frames', 'exact_comparisons')) == ((2269, 9225) if suite == 'optimizer' else (594, 2466))
    private_args = arguments + ['--replay', review / 'replay_optimizer_objects_v2.py',
                               '--controls', review / 'native_release_controls.py', '--observer', review / 'client.py']
    execute('private-roots', review / 'check_private_optimizer_roots.py', *private_args)
    r = reports['private-roots']
    assert r['status'] == 'PASS' and len(r['frames']) == 72 and r['whole_owner_rollback_bytes'] == 29066368
    assert r['parameter_and_state_metadata_preserved_across_reset_and_growth'] and r['final_public_aliases_outlive_optimizer'] and r['final_weak_expiry_after_last_public_drop']
    execute('accumulator-lifetime', review / 'replay_accumulator_lifetime.py', *private_args,
            '--observations', evidence / 'accumulator-lifetime-08/reference/torch-accumulator-lifetime.json')
    r = reports['accumulator-lifetime']
    assert r['status'] == 'QUALIFIED_STORAGE_AND_PRIVATE_GRAD_LIFETIME'
    assert (r['cases'], r['qualified'], r['unexpected_failures'], r['frames'], r['exact_observation_comparisons']) == (3, 3, 0, 106, 102)
    for name in ('optimizer', 'requires', 'historical-optimizer', 'historical-requires', 'accumulator-lifetime'):
        provenance = reports[name]['oracle_provenance']
        assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11 and all(v is True for v in provenance['checks'].values())
    execute('native', package / 'test_native.py', *arguments, '--protocol', protocol)
    r = reports['native']
    assert r['status'] == 'PASS' and len(r['rows']) == 4
    assert (r['frames'], r['exact_values'], r['rollback_bytes']) == (100, 29, 1834848)
    execute('guards', package / 'test_guards.py', '--binary', a.build / 'tensor-optimizer-session',
            '--source', a.build / 'runtime.hepta', '--profile', package / 'profile.json', '--protocol', protocol)
    r = reports['guards']
    assert r['status'] == 'PASS' and (r['case_count'], r['frames'], r['rejections'], r['rollback_bytes']) == (4, 77, 37, 13280976)
    execute('observer-adaptation', package / 'adapt_regressions.py', '--grad-clear', nullable / 'replay_grad_clear.py',
            '--common-joint', common / 'test_joint.py', '--nullable-joint', joint / 'test_combination.py', report='adaptation.json')
    assert reports['observer-adaptation'] == json.loads((package / 'evidence/regression-adaptation.json').read_bytes())
    old_replay = arguments + ['--controls', partial / 'native_release_controls.py', '--transport-observer', observer]
    for name, script, observation in (
        ('signedzero', partial / 'replay_gradient_signed_zero.py', partial / 'evidence/actual-torch-signed-zero-observations.json'),
        ('partial', partial / 'replay_partial_observations.py', partial / 'evidence/actual-torch-16-observations.json'),
        ('presence', partial / 'replay_gradient_presence.py', partial / 'evidence/actual-torch-gradient-presence-observations.json'),
        ('grad-clear', out / 'observer-adaptation/replay_grad_clear.py', nullable / 'evidence/actual-torch-grad-clear-observations.json'),
    ):
        execute(name, script, *old_replay, '--observations', observation)
        r = reports[name]
        assert r['native_executable_sha256'] == IMAGE and r['native_source_sha256'] == SOURCE
        assert len(r['oracle_provenance']['checks']) == 11 and all(v is True for v in r['oracle_provenance']['checks'].values())
    zero, partial_report, presence, clear = (reports[key] for key in ('signedzero', 'partial', 'presence', 'grad-clear'))
    assert zero['status'] == 'PASS' and (zero['cases'], zero['passed'], zero['frames'], zero['exact_binary32_word_comparisons'], zero['different_word_observations']) == (13, 13, 166, 70, 0)
    assert partial_report['status'] == 'PASS' and (partial_report['cases'], partial_report['passed'], partial_report['frames'], partial_report['exact_scalar_or_binary32_comparisons']) == (16, 16, 606, 1587)
    assert presence['status'] == 'QUALIFIED_WITH_OPEN_STATE_LIFECYCLE' and presence['unexpected_failures'] == 0
    assert tuple(presence[k] for k in ('cases', 'frames', 'exact_scalar_or_binary32_comparisons', 'adam_tolerance_comparisons', 'adam_compared_values_with_different_bits', 'state_version_difference_observations', 'state_creation_difference_observations')) == (8, 880, 2088, 180, 0, 8, 65)
    assert clear['status'] == 'QUALIFIED_GRAD_CLEAR_WITH_OPEN_OBJECT_SCOPE' and clear['unexpected_failures'] == 0
    assert tuple(clear[k] for k in ('cases', 'qualified', 'frames', 'exact_comparisons', 'Adam_tolerance_comparisons', 'Adam_different_bits', 'state_version_difference_observations', 'state_creation_difference_observations', 'native_step_Tensor_absence_observations')) == (10, 10, 780, 2669, 288, 6, 11, 9, 30)
    assert clear['numeric_policy']['Adam_updated_values'] == {'rtol': 4e-5, 'atol': 5e-6}
    reference = out / 'independent-mt-reference'
    result = subprocess.run(['g++', '-std=c++17', '-O2', str(rng / 'reference.cpp'), '-o', str(reference)], capture_output=True, timeout=60)
    (out / 'reference-build.stdout').write_bytes(result.stdout)
    (out / 'reference-build.stderr').write_bytes(result.stderr)
    assert result.returncode == 0 and not result.stderr
    rng_args = ['--executable', a.build / 'tensor-optimizer-session', '--rng-package', rng, '--cpp-reference', reference]
    execute('common-rng', out / 'observer-adaptation/test_joint.py', *rng_args)
    r = reports['common-rng']
    assert r['status'] == 'PASS' and (r['case_count'], r['frames'], r['full_generator_states'], r['joint_owner_rollbacks'], r['stage_zero_observations']) == (3, 155, 14, 5, 42)
    execute('nullable-rng', out / 'observer-adaptation/test_combination.py', *rng_args)
    r = reports['nullable-rng']
    assert r['status'] == 'PASS' and len(r['cases']) == 4
    assert (r['frames'], r['exact_words'], r['full_generator_states'], r['rollback_bytes'], r['committed_node_invariant_checks']) == (56, 393, 34, 2517280, 62)
    for predecessor, directory, frames, rollback in (('nullable', a.nullable_build, 41, 607520), ('optimizer001', old, 50, 611616)):
        execute(predecessor + '-handoff-build', package / 'build_state_handoff.py', '--predecessor', predecessor,
                '--old', directory, '--new', a.build, '--compiler', compiler, '--linker', linker)
        execute(predecessor + '-handoff', package / 'test_state_handoff.py',
                '--executable', out / (predecessor + '-handoff-build/state-handoff-session'),
                '--build-receipt', out / (predecessor + '-handoff-build/report.json'), '--protocol', protocol)
        r = reports[predecessor + '-handoff']
        assert r['status'] == 'PASS' and (r['cases'], r['frames'], r['entire_owner_bytes_unchanged']) == (2, frames, rollback)
    after = {str(path): sha(path) for path in sorted(inputs)}
    assert before == after
    report = {
        'schema': 'heptatorch.integrated-owned-sgd-qualification.v1',
        'status': 'QUALIFIED_EXPOSED_SGD_AND_LEAF_FLAGS_WITH_EXPLICIT_OPEN_SCOPE',
        'native_source_sha256': SOURCE, 'native_executable_sha256': IMAGE,
        'source_custody_before': before, 'source_custody_after': after,
        'reports': reports, 'original_report_sha256': report_hashes,
        'historical_native_failures_reproduced': ['nongrad transpose leaf admission', 'forward edge absent before requires_grad enable'],
        'historical_failures_expected_exit': 1,
        'authoring_aggregate_admission_claimed': False,
        'all_actual_native_modules_and_linked_call_graph_checked': True,
        'host_production_numerical_runtime': False, 'pytorch_executed_in_this_process': False,
        'reference_is_actual_pinned_hosted_observation': True,
        'public_grad_Tensor_identity_Version_and_alias_lifetime': 'OPEN',
        'owned_Adam_AdamW_step_and_moment_Tensors': 'OPEN',
        'late_Optimizer_error_partial_mutation': 'OPEN',
        'retained_nonleaf_parameter_recipe': 'PARTIAL',
        'old_explicit_state_differences_remain_separate': True,
        'count_policy': 'Do not sum repeated replays or separate suites into unique upstream API coverage.',
        'pytorch_api_denominator': 2590, 'whole_pytorch_rewrite_complete': False,
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'report_sha256': sha(out / 'report.json'), 'reports': len(reports)}))


if __name__ == '__main__':
    main()
