"""Qualify one native Adam/distribution owner with separately counted evidence."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import stat
import subprocess
import sys
import tarfile

SOURCE = '5d3a3c047d024a3880d257a317a04c2166a2e87bbc6b248bef62649a88546873'
IMAGE = '3fc59ab7ed9c32dd75ff440ef9bfbec31756f50edd77715d665bdac098cc3f22'
ADAM = '253438648a437a75ef2dbc77138eca74cb5fe6a6e2ea322c7c38e5a535e5d2a6'
ADAM_IMAGE = 'fa92356954dbb9e0c454f2a58c2f78aa33950103d377c57e7b659ec9904183af'
ATOMIC = 'a72a98232c7b7f10efa67ce29b21e417c58201383bef9aacc266fe247ff356ce'
ATOMIC_IMAGE = '515790a191b3db1ecbfcb3f2afab644ef980b51ad85c5d6daf1b8c5a9a5a6e28'
PARTIAL = '55ed0c6e16317add03974c01ffd85e38684fc535a3e5cb6423ed043f4217f05b'
PARTIAL_IMAGE = 'bb4c25e35a170f08276d464db6b630cf03765ff816e6a9ca9aa1651360a1ccf4'
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER = 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
TEXT_MANIFESTS = {
    'adam-distributions-package.sha256': ('b186fcd7d065f12e7f94b25afa993bbf8f783beb7626aed2df4daa4720f6307f', 41),
    'adam-distributions-review-package.sha256': ('a05b73cfd2be4bcf1ebcafd65362ab7284084dbf1af55c4dfd1010fbfdad1d4f', 17),
    'owned-adam-package.sha256': ('d48fb1b29e3a41d78f007f6561bcfbd93385a250545bab8a2a7fc3927d904072', 54),
    'adam-object-review-package.sha256': ('c316382321ec57a591fb3bea335231ee701aa91e305b48bb38fef1e31cc5ab7f', 17),
    'cpu-distributions-package.sha256': ('0a22dd70b079851d6ae109fb9f28fe77a4e002f211fcbc0fc79a0c0c35afc867', 37),
    'cpu-distributions-partial-package.sha256': ('49992445c887ba87eeda65acf38c8b22ce2b01bca59bb893f3b4a7acd3644737', 28),
}
OPEN_DISTRIBUTIONS = [4, 9, 14, 19, 24, 29, 34, 39, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174]
LATE_ERRORS = [101, 103, 105, 179]


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'adam-build', 'atomic-build', 'distribution-build', 'stack-build', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    for name, value in vars(a).items():
        setattr(a, name, value.resolve())
    repo = Path(__file__).resolve().parents[1]
    package = repo / 'tools/adam_distributions'
    review = repo / 'tests/adam_distributions_review'
    inputs = {Path(__file__).resolve()}
    transport = repo / 'validation/adam-distributions/evidence/archive-parts.json'
    restorer = repo / 'tools/restore_frozen_archive.py'
    restored = subprocess.run([sys.executable, '-I', '-S', '-B', str(restorer),
                               '--manifest', str(transport)], capture_output=True)
    assert restored.returncode == 0 and not restored.stderr, restored.stderr
    restoration = json.loads(restored.stdout)
    assert restoration['status'] == 'PASS'
    assert restoration['sha256'] == '0ec89d5c56f5bd3a39db09bf3a9ed56a2c13e759648327f8388488ec0f25a907'
    inputs.update((transport, restorer))
    for row in json.loads(transport.read_bytes())['parts']:
        path = (repo / row['path']).resolve()
        assert path.is_relative_to(repo) and sha(path) == row['sha256']
        inputs.add(path)
    for name, (digest, count) in TEXT_MANIFESTS.items():
        path = repo / 'validation' / name
        assert sha(path) == digest, name
        rows = path.read_text().splitlines()
        assert len(rows) == count
        inputs.add(path)
        names = set()
        for row in rows:
            expected, relative = row.split('  ', 1)
            target = (repo / relative).resolve()
            assert relative not in names and target.is_relative_to(repo) and sha(target) == expected, relative
            names.add(relative)
            inputs.add(target)
    for name, digest, count in (
        ('adam-distributions-package-manifest.json', '2ba1c7c63cf6d5f56289a1262e3592341c8267291239111708e960545d7d52af', 41),
        ('adam-distributions-review/package-manifest.json', '1baa3e37b4d077a57c8ff67c2072adc2ba14f685eb50fefc027edce528968f9c', 17),
    ):
        path = repo / 'validation' / name
        assert sha(path) == digest
        rows = json.loads(path.read_bytes())['files']
        assert len(rows) == count and len({row['path'] for row in rows}) == count
        for row in rows:
            target = (repo / row['path']).resolve()
            assert target in inputs and sha(target) == row['sha256']
            assert target.stat().st_size == row['bytes'] and stat.S_IMODE(target.stat().st_mode) == row['mode']
        inputs.add(path)
    for relative, expected in (
        ('tools/tensor_objects/protocol.py', '5b99b0e508c2093fe830e509fffbb46b709f65e85b669ab6d79a32d425aa95ac'),
        ('tools/owned_sessions/client.py', 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'),
    ):
        path = repo / relative
        assert sha(path) == expected
        inputs.add(path)
    for filename, expected in (('stack-module-compiler', COMPILER), ('stack-linker-session', LINKER)):
        path = a.stack_build / 'native' / filename
        assert sha(path) == expected
        inputs.add(path)
    builds = {}
    for label, directory, filename, source, image, count in (
        ('common', a.build, 'tensor-adam-distributions-session', SOURCE, IMAGE, 8),
        ('adam02', a.adam_build, 'tensor-adam-session', ADAM, ADAM_IMAGE, 8),
        ('atomic03', a.atomic_build, 'tensor-distributions-session', ATOMIC, ATOMIC_IMAGE, 7),
        ('partial01', a.distribution_build, 'tensor-distributions-session', PARTIAL, PARTIAL_IMAGE, 7),
    ):
        assert sha(directory / 'runtime.hepta') == source and sha(directory / filename) == image
        path = directory / 'build.json'
        build = json.loads(path.read_bytes())
        assert build['status'] == 'PASS' and len(build['units']) == count
        assert build.get('runtime_source_sha256', build.get('source_sha256')) == source
        assert build.get('executable_sha256', build.get('native_sha256')) == image
        inputs.update((path, directory / 'runtime.hepta', directory / filename))
        for unit in build['units']:
            for extension, key in (('.hepta', 'source_sha256'), ('.o', 'object_sha256')):
                artifact = directory / (unit['name'] + extension)
                assert sha(artifact) == unit[key]
                inputs.add(artifact)
        builds[label] = build
    profile = json.loads((package / 'profile.json').read_bytes())
    build = builds['common']
    assert build['units'] == profile['units']
    assert (build['core_functions'], build['core_blocks'], build['core_instructions']) == (378, 4971, 18539)
    assert build['compiler_sha256'] == COMPILER and build['linker_sha256'] == LINKER
    assert build['host_production_numerical_runtime'] is False and build['aggregate_core_host_check'] is False
    for key in ('all_distribution_functions_byte_identical', 'eight_kind_layout_and_optimizer_gc_unchanged',
                'graph_capture_and_audit_unchanged', 'owner_module_byte_identical', 'rng_module_byte_identical',
                'old_adam_entry_reproduced_exactly', 'input_custody_after'):
        assert build[key] is True, key
    reproduction_path = a.build / 'reproduction.json'
    reproduction = json.loads(reproduction_path.read_bytes())
    assert reproduction['status'] == 'PASS' and reproduction['native_units'] == 8
    assert reproduction['runtime_source_sha256'] == SOURCE and reproduction['executable_sha256'] == IMAGE
    assert reproduction['all_unit_source_object_and_image_bytes_identical'] is True
    assert reproduction['complete_native_compilation_and_linking_executed'] is True
    assert reproduction['build_report_sha256'] == sha(a.build / 'build.json')
    inputs.add(reproduction_path)
    before = {str(path): sha(path) for path in sorted(inputs)}
    a.output.mkdir(parents=True, exist_ok=False)
    archives = {}
    for label, manifest_path, filename, expected, count, total in (
        ('producer', repo / 'validation/adam-distributions/evidence/archive-manifest.json',
         'native-observations.tar.gz', '0ec89d5c56f5bd3a39db09bf3a9ed56a2c13e759648327f8388488ec0f25a907', 3190, 139822732),
        ('independent', repo / 'validation/adam-distributions-review/all-raw-observations.json',
         'all-raw-observations.tar.xz', '7b04428fa1098fa8e57c6a6006a0a01d3f4eb67faee7b98f7088c1e3217e5b74', 118, 18928809),
    ):
        manifest = json.loads(manifest_path.read_bytes())
        archive = manifest_path.with_name(filename)
        assert sha(archive) == manifest['archive_sha256'] == expected
        assert archive.stat().st_size == manifest['archive_bytes']
        members = {row['path']: row for row in manifest['members']}
        assert len(members) == len(manifest['members']) == count
        seen, size = set(), 0
        with tarfile.open(archive, 'r|*') as tf:
            for member in tf:
                assert member.isfile() and member.name in members and member.name not in seen
                row = members[member.name]
                digest, length = hashlib.sha256(), 0
                with tf.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(1 << 20), b''):
                        digest.update(block)
                        length += len(block)
                assert digest.hexdigest() == row['sha256'] and length == row['bytes'] and member.mode == row['mode']
                seen.add(member.name)
                size += length
        assert seen == set(members) and size == total
        archives[label] = {'sha256': expected, 'verified_members': len(seen), 'verified_expanded_bytes': size}

    reports, hashes = {}, {}

    def execute(name, script, *arguments):
        target = a.output / name
        command = [sys.executable, '-B', str(script), *map(str, arguments), '--output', str(target)]
        result = subprocess.run(command, capture_output=True)
        (a.output / (name + '.stdout')).write_bytes(result.stdout)
        (a.output / (name + '.stderr')).write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr, (name, result.returncode, result.stderr[-3000:])
        path = target / 'report.json'
        reports[name], hashes[name] = json.loads(path.read_bytes()), sha(path)

    execute('producer', package / 'qualify.py', '--build', a.build, '--adam-build', a.adam_build,
            '--atomic-build', a.atomic_build, '--distribution-build', a.distribution_build,
            '--stack-build', a.stack_build, '--atomic-tools', repo / 'tools/cpu_distributions',
            '--partial-tools', repo / 'tools/cpu_distributions_partial', '--adam-tools', repo / 'tools/owned_adam',
            '--adam-review', repo / 'tests/adam_object_review', '--tensor-protocol', repo / 'tools/tensor_objects/protocol.py',
            '--optimizer-observations', repo / 'validation/adam-object-review/torch-optimizer-objects.json')
    producer = reports['producer']
    assert producer['status'] == 'QUALIFIED_COMPONENT_WITH_OPEN_SCOPE' and producer['suite_count'] == 13
    assert producer['unexpected_failures'] == 0 and producer['expected_old_negative_failure_count'] == 1
    assert producer['runtime_source_sha256'] == SOURCE and producer['executable_sha256'] == IMAGE
    assert producer['old_atomic_control_known_failure_preserved'] is True
    assert producer['original_atomic_unknown_opcode_control_passed'] is False
    assert producer['input_custody_after'] is True
    for row in producer['suites']:
        path = a.output / 'producer' / row['report']
        assert path.resolve().is_relative_to(a.output) and sha(path) == row['report_sha256']
        script = Path(row['command'][1])
        assert sha(script) == row['script_sha256']
        assert row['exit'] == row['expected_exit'] == (1 if row['name'] == 'legacy_atomic_controls' else 0)
        detail = json.loads(path.read_bytes())
        if row['name'] != 'new_command_admission':
            assert SOURCE in [detail.get(k) for k in ('runtime_source_sha256', 'native_source_sha256', 'source_sha256', 'current_source_sha256')]
            assert IMAGE in [detail.get(k) for k in ('executable_sha256', 'native_executable_sha256', 'native_sha256', 'current_executable_sha256')]
        reports[row['name']], hashes[row['name']] = detail, sha(path)
    assert len(producer['suites']) == len({r['name'] for r in producer['suites']}) == 13
    reference = json.loads((repo / 'tools/cpu_distributions/evidence/torch-bernoulli-dropout.json').read_bytes())
    for name, partial_errors in (('atomic_reference', LATE_ERRORS), ('explicit_reference', [])):
        report = reports[name]
        assert len(report['rows']) == len(reference['rows']) == 183
        for index, (row, original) in enumerate(zip(report['rows'], reference['rows'], strict=True)):
            assert (row['index'], row['name'], row['family']) == (index, original['name'], original['family'])
            expected = ('OPEN_UNREPRESENTED' if index in OPEN_DISTRIBUTIONS else
                        'OPEN_TRANSACTION_DIFFERENCE' if index in partial_errors else 'PASS')
            assert row['status'] == expected
        assert dict(Counter(row['status'] for row in report['rows'])) == report['counts']
        assert report['actual_reference_provenance_checks'] == reference['oracle_provenance']['checks']
        assert len(reference['oracle_provenance']['checks']) == 11 and all(reference['oracle_provenance']['checks'].values())
    old, added = reports['optimizer_reference'], reports['adam_reference']
    actual = json.loads((repo / 'validation/adam-object-review/torch-optimizer-objects.json').read_bytes())
    assert len(old['rows']) == len(actual['rows']) == 24
    assert [row['index'] for row in added['rows']] == [21, 22, 23, 24]
    additions = {row['index']: row for row in added['rows']}
    ledger = []
    for index, (row, original) in enumerate(zip(old['rows'], actual['rows'], strict=True), 1):
        chosen = additions.get(index, row)
        assert chosen['case'] == row['case'] == original['case']
        if index in additions:
            assert row['status'] == 'OPEN_NOT_EXECUTED' and chosen['status'] == 'QUALIFIED_EXPOSED_SCOPE'
        ledger.append({'index': index, 'case': chosen['case'], 'status': chosen['status'],
                       'report': 'adam_reference' if index in additions else 'optimizer_reference'})
    assert Counter(row['status'] for row in ledger) == {'QUALIFIED_EXPOSED_SCOPE': 22, 'PARTIAL_WITH_RETAINED_NONLEAF_OPEN': 1, 'OPEN_NOT_EXECUTED': 1}
    assert ledger[13]['status'] == 'PARTIAL_WITH_RETAINED_NONLEAF_OPEN' and ledger[19]['status'] == 'OPEN_NOT_EXECUTED'
    admission = reports['new_command_admission']
    assert admission['status'] == 'PASS' and len(admission['rows']) == 12
    observed = set()
    profiles = {'atomic03': (ATOMIC, ATOMIC_IMAGE), 'partial01': (PARTIAL, PARTIAL_IMAGE), 'combined02': (SOURCE, IMAGE)}
    for row in admission['rows']:
        key = row['profile'], row['opcode']
        assert key not in observed and key[0] in profiles and key[1] in (86, 87, 88, 89)
        observed.add(key)
        assert (row['source_sha256'], row['executable_sha256']) == profiles[key[0]]
        assert row['actual_error'] == row['expected_error'] == (1 if key[0] != 'atomic03' and key[1] in (86, 87) else 4)
        assert row['owner_identity_and_bytes_unchanged'] is True
    handoffs = reports['typed_handoffs']
    units = {(role, row['name']): row['source_sha256'] for role in ('common', 'adam02', 'partial01') for row in builds[role]['units']}
    assert len(handoffs['source_bindings']) == len(units) == 23
    assert {(b['role'], b['name']) for b in handoffs['source_bindings']} == set(units)
    for binding in handoffs['source_bindings']:
        assert binding['original_sha256'] == units[binding['role'], binding['name']]
        assert binding['inverse_byte_identical'] is True
        prefix = 'new_' if binding['role'] == 'common' else 'old_'
        path = a.output / 'producer/typed_handoffs' / binding['role'] / (prefix + binding['name'] + '.hepta')
        assert sha(path) == binding['renamed_sha256']
    peers = {'adam02': (ADAM, '72104531b1bfe157f9c7af88e0c22559bbbd6b8e3feef3388bc4046c4755050f'),
             'partial01': (PARTIAL, '41354a48758c448c9af872519e1654c58e534c441006853a904689ee85c03c70')}
    assert len(handoffs['peers']) == 2 and {p['name'] for p in handoffs['peers']} == set(peers)
    for peer in handoffs['peers']:
        source, image = peers[peer['name']]
        assert peer['status'] == 'PASS' and peer['peer_source_sha256'] == source
        assert peer['qualification_executable_sha256'] == sha(a.output / 'producer/typed_handoffs' / peer['name'] / 'handoff-session') == image
        assert peer['both_directions_refuse_and_recover'] and peer['owner_pointer_length_and_bytes_preserved'] and peer['both_closed_owner_mappings_absent']
    execute('independent-source', review / 'source_audit.py', '--adam-build', a.adam_build,
            '--distribution-build', a.distribution_build, '--build', a.build)
    audit = reports['independent-source']
    assert audit['status'] == 'PASS_SOURCE_CUSTODY_AND_SCOPED_REVIEW'
    assert audit['pins'] == {'adam': ADAM, 'distribution': PARTIAL, 'source': SOURCE, 'binary': IMAGE}
    assert (audit['unchanged_old_bodies'], audit['distribution_functions_copied_exact_bytes']) == (345, 25)
    for key in ('entry_inverse_restores_Adam_exact_AST_text', 'old_dispatch_retained_as_real_alias_exact_bytes',
                'gradient_direct_only_extends_upper_opcode_40_to_41', 'input_custody_after'):
        assert audit[key] is True
    execute('independent-joint', review / 'test_joint_contracts.py', '--build', a.build,
            '--protocol', repo / 'tools/cpu_distributions/protocol.py',
            '--distribution-observations', repo / 'tools/cpu_distributions/evidence/torch-bernoulli-dropout.json',
            '--adam-observations', repo / 'validation/adam-object-review/torch-optimizer-objects.json')
    joint = reports['independent-joint']
    assert joint['status'] == 'PASS' and joint['runtime_source_sha256'] == SOURCE and joint['executable_sha256'] == IMAGE
    assert (joint['case_count'], joint['frames_before_EOF'], joint['exact_assertions'], joint['exact_float32_words'],
            joint['complete_MT_state_comparisons'], joint['rollback_bytes_compared']) == (6, 462, 334, 249, 22, 4892928)
    assert joint['native_executed'] and joint['input_custody_after']
    assert joint['Torch_executed_here'] is False and joint['whole_PyTorch_complete'] is False
    assert {str(path): sha(path) for path in sorted(inputs)} == before
    result = {
        'schema': 'heptatorch.adam-distributions.root-qualification.v1',
        'status': 'QUALIFIED_COMPONENT_WITH_OPEN_SCOPE', 'native_scoped_qualification_passed': True,
        'runtime_source_sha256': SOURCE, 'executable_sha256': IMAGE,
        'build': build, 'reproduction': reproduction, 'reports': reports, 'report_sha256': hashes,
        'formal_report_count': 15, 'producer_suite_count': 13,
        'expected_legacy_unknown_opcode_failure_count': 1, 'unexpected_failures': 0,
        'old_atomic_test_reclassified_as_pass': False,
        'explicit_distribution_counts': reports['explicit_reference']['counts'],
        'atomic_distribution_counts': reports['atomic_reference']['counts'],
        'optimizer_original_24_ledger': ledger,
        'optimizer_counts': dict(Counter(row['status'] for row in ledger)),
        'new_independent_cross_contracts': 6,
        'cross_contracts_added_to_existing_case_denominators': False,
        'raw_history_archives': archives, 'source_custody_after': True, 'input_sha256': before,
        'frozen_archive_restoration': restoration,
        'all_rng_complete': False, 'all_optimizer_complete': False,
        'public_gradient_successor_composed': False,
        'full_pytorch_rewrite_complete': False, 'whole_project_decision': 'HOLD',
    }
    assert len(reports) == 16  # Producer summary plus 13 suites and two independent reports.
    (a.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'runtime_source_sha256', 'executable_sha256',
                      'formal_report_count', 'expected_legacy_unknown_opcode_failure_count', 'unexpected_failures')}))


if __name__ == '__main__':
    main()
