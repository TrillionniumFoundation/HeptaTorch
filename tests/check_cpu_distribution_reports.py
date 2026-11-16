"""Bind complete native distribution reports to frozen source and owner profiles."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import tarfile

PROFILES = {
    'atomic': ('cpu_distributions', 'cpu-distributions-package.sha256',
               '0a22dd70b079851d6ae109fb9f28fe77a4e002f211fcbc0fc79a0c0c35afc867', 37,
               'a72a98232c7b7f10efa67ce29b21e417c58201383bef9aacc266fe247ff356ce',
               '515790a191b3db1ecbfcb3f2afab644ef980b51ad85c5d6daf1b8c5a9a5a6e28'),
    'partial': ('cpu_distributions_partial', 'cpu-distributions-partial-package.sha256',
                '49992445c887ba87eeda65acf38c8b22ce2b01bca59bb893f3b4a7acd3644737', 28,
                '55ed0c6e16317add03974c01ffd85e38684fc535a3e5cb6423ed043f4217f05b',
                'bb4c25e35a170f08276d464db6b630cf03765ff816e6a9ca9aa1651360a1ccf4'),
}
REFERENCE = 'c624f5c4762b1c71e7053aba9af55aaa3d8bbaae5aa100b0a6df5f44f58da467'
COMPILER = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
SESSION_LINKER = 'aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
RUNTIME_LINKER = '0c5176e3aa968631ae592b887c5dc3cf1e8e3f48f73593f076e9d7965ffacfd7'
OPEN_INDICES = [4, 9, 14, 19, 24, 29, 34, 39, 109, 110, 167, 168, 169, 170, 171, 172, 173, 174]
LATE_ERROR_INDICES = [101, 103, 105, 179]
REPORT_SCHEMAS = {
    'atomic/actual-reference': 'heptatorch.cpu-distributions.actual-torch-replay.v1',
    'atomic/native-controls': 'heptatorch.cpu-distributions-native-controls.v1',
    'atomic/probability-math': 'heptatorch.distribution-probability-integer-qualification.v1',
    'atomic/owned-entry': 'heptatorch.cpu-distributions-owned-entry-guards.v1',
    'atomic/typed-handoffs': 'heptatorch.cpu-distributions-profile-handoffs.v1',
    'partial/atomic-reference': 'heptatorch.cpu-distributions.actual-torch-replay.v1',
    'partial/explicit-reference': 'heptatorch.cpu-distributions.explicit-partial-actual-replay.v1',
    'partial/four-late-errors': 'heptatorch.cpu-distributions-late-error-actual-replay.v1',
    'partial/native-controls': 'heptatorch.cpu-distributions-partial-native-controls.v1',
    'partial/owned-entry': 'heptatorch.cpu-distributions-partial-owned-entry.v1',
    'partial/typed-handoffs': 'heptatorch.cpu-distributions-partial-profile-handoffs.v1',
}


def check_report_identity(key, report, profile, peer_pins, archive_members):
    """Bind each concrete report schema, including qualification-only identities."""
    assert report['schema'] == REPORT_SCHEMAS[key], key
    source, image = profile['runtime_source_sha256'], profile['executable_sha256']
    if key.endswith('/typed-handoffs'):
        assert report['current_source_sha256'] == source and report['current_executable_sha256'] == image, key
        assert report['compiler_sha256'] == COMPILER and report['linker_sha256'] == SESSION_LINKER
        assert report['raw_pointer_import'] is False and report['all_source_custody_unchanged'] is True
        assert peer_pins['current_source_sha256'] == source and peer_pins['current_executable_sha256'] == image
        assert [p['name'] for p in report['peers']] == list(peer_pins['peers'])
        units = {('common', u['name']): u['source_sha256'] for u in profile['units']}
        for peer in report['peers']:
            expected = peer_pins['peers'][peer['name']]
            assert peer['peer_source_sha256'] == expected['source_sha256']
            assert peer['state_magic'] == expected['state_magic'] and peer['status'] == 'PASS'
            assert peer['frames'] == 9 and peer['modules'] == 15
            for field in ('both_directions_refuse_and_recover', 'owner_pointer_length_and_bytes_preserved',
                          'both_closed_owner_mappings_absent', 'new_state_has_genuinely_cleared_nullable_leaf'):
                assert peer[field] is True, (key, field)
            expected_image = archive_members['handoffs-01/' + peer['name'] + '/handoff-session']['sha256']
            assert peer['qualification_executable_sha256'] == expected_image
            units.update({(peer['name'], u['name']): u['source_sha256'] for u in expected['units']})
        bindings = report['source_bindings']
        assert len(bindings) == len(units)
        assert {(b['role'], b['name']) for b in bindings} == set(units)
        for binding in bindings:
            assert binding['original_sha256'] == units[binding['role'], binding['name']]
            assert binding['inverse_byte_identical'] is True
            prefix = 'new_' if binding['role'] == 'common' else 'old_'
            archived = 'handoffs-01/' + binding['role'] + '/' + prefix + binding['name'] + '.hepta'
            assert binding['renamed_sha256'] == archive_members[archived]['sha256']
        return
    if key == 'atomic/probability-math':
        assert report['runtime_source_sha256'] == source and report['production_executable_sha256'] == image
        assert report['probe_source_sha256'] == 'cea4cfa904cac7aa1950a7e83c336b6778c4a76827a250a6c90d323de3c31c8c'
        assert report['probe_executable_sha256'] == '3746a0cce5f7149ec34ac1a0c5c9a96ea1bc0a7a8e3ec804754f2d06eb595c25'
        assert report['production_objects_byte_identical'] is True and report['qualification_only_extra_entry'] is True
    else:
        assert report['executable_sha256'] == image, key
        # This original schema records the production ELF and the probe source,
        # but does not contain a production runtime-source field.
        if key != 'atomic/owned-entry':
            assert report['runtime_source_sha256'] == source, key
    if key.endswith('/owned-entry'):
        assert report['probe_source_sha256'] == '54f81a65652e2c2ba9eb3f04d608e89b71c92c5ad33ff7061e9a03b89f45f9ea'
        expected_probe = ('1308c99993d0f3e7be0e0adcc525058ace091f372c31a330b58f37b505b22756'
                          if key.startswith('atomic/') else
                          '980ba796baab3ea2851cb0048115260d1bf4784a5a3b9c945ff34572c7616e3d')
        assert report['probe_sha256'] == expected_probe
        if key.startswith('atomic/'):
            assert report['compiler_sha256'] == COMPILER and report['runtime_linker_sha256'] == RUNTIME_LINKER
            assert report['native_production_uses_host_reference'] is False
        else:
            assert report['qualification_only_owned_byte_faults'] is True and report['foreign_owner_pointer_imported'] is False
    if key != 'atomic/owned-entry':
        assert report['input_custody_after'] is True, key


def check_reference_rows(report, reference, explicit=False):
    rows = report['rows']
    assert len(reference['rows']) == len(rows) == 183
    assert [r['index'] for r in rows] == list(range(183))
    for observed, original in zip(rows, reference['rows'], strict=True):
        assert (observed['name'], observed['family']) == (original['name'], original['family'])
    assert dict(Counter(r['status'] for r in rows)) == report['counts']
    assert [r['index'] for r in rows if r['status'] == 'OPEN_UNREPRESENTED'] == OPEN_INDICES
    assert [r['index'] for r in rows if r['status'] == 'OPEN_TRANSACTION_DIFFERENCE'] == ([] if explicit else LATE_ERROR_INDICES)
    assert report['actual_reference_provenance_checks'] == reference['oracle_provenance']['checks']


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('atomic-build', 'build', 'atomic-review', 'review', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    for key, value in vars(a).items():
        setattr(a, key, value.resolve())
    repo = Path(__file__).resolve().parents[1]
    files = {Path(__file__)}
    builds, reproductions, summaries, reports, report_hashes, archives = {}, {}, {}, {}, {}, {}
    for label, (directory, manifest_name, manifest_pin, count, source_pin, image_pin) in PROFILES.items():
        package = repo / 'tools' / directory
        manifest = repo / 'validation' / manifest_name
        assert digest(manifest) == manifest_pin
        rows = manifest.read_text().splitlines()
        assert len(rows) == count
        files.add(manifest)
        for row in rows:
            expected, relative = row.split('  ', 1)
            path = (repo / relative).resolve()
            assert path.is_relative_to(repo) and digest(path) == expected, relative
            files.add(path)
        profile = json.loads((package / 'profile.json').read_bytes())
        peer_pins = json.loads((package / 'handoff-peer-pins.json').read_bytes())
        directory = a.atomic_build if label == 'atomic' else a.build
        for name, expected in (('runtime.hepta', source_pin), ('tensor-distributions-session', image_pin)):
            path = directory / name
            assert digest(path) == expected
            files.add(path)
        build = json.loads((directory / 'build.json').read_bytes())
        reproduction = json.loads((directory / 'reproduction.json').read_bytes())
        files.update((directory / 'build.json', directory / 'reproduction.json'))
        assert build['status'] == reproduction['status'] == 'PASS'
        assert build['runtime_source_sha256'] == reproduction['runtime_source_sha256'] == source_pin
        assert build['executable_sha256'] == reproduction['executable_sha256'] == image_pin
        assert reproduction['all_source_object_image_bytes_identical'] is True and reproduction['native_units'] == 7
        assert build['host_production_numerical_runtime'] is False and build['aggregate_core_host_check'] is False
        assert build['rng_module_byte_identical'] and build['owner_module_byte_identical'] and build['input_custody_after']
        assert build['compiler_sha256'] == COMPILER
        assert build['linker_sha256'] == SESSION_LINKER
        assert len(build['units']) == len(profile['units']) == 7
        for unit, original in zip(build['units'], profile['units'], strict=True):
            for key in ('name', 'source_sha256', 'object_sha256'):
                assert unit[key] == original[key]
            for extension, key in (('.hepta', 'source_sha256'), ('.o', 'object_sha256')):
                path = directory / (unit['name'] + extension)
                assert digest(path) == unit[key]
                files.add(path)
        if label == 'partial':
            assert build['all_predecessor_dist_functions_byte_identical'] and build['old_atomic_commands_80_85_unchanged']
            assert reproduction['old_atomic_function_bodies_identical']
        builds[label], reproductions[label] = build, reproduction
        history = json.loads((package / 'evidence/native-observations.json').read_bytes())
        archive = package / 'evidence' / history['archive']
        assert digest(archive) == history['archive_sha256'] and archive.stat().st_size == history['archive_bytes']
        members = history.get('members', history.get('files'))
        expected = {row['path']: row for row in members}
        assert len(expected) == len(members)
        seen, decoded_bytes = set(), 0
        with tarfile.open(archive, 'r|gz') as tf:
            for member in tf:
                assert member.isfile() and member.name in expected and member.name not in seen
                row = expected[member.name]
                stream = tf.extractfile(member)
                value, size = hashlib.sha256(), 0
                for block in iter(lambda: stream.read(1 << 20), b''):
                    value.update(block); size += len(block)
                assert value.hexdigest() == row['sha256'] and size == row['bytes'] and member.mode == row['mode']
                seen.add(member.name)
                decoded_bytes += size
        assert seen == set(expected)
        assert decoded_bytes == history['original_bytes']
        archives[label] = {'archive_sha256': history['archive_sha256'], 'archive_bytes': archive.stat().st_size,
                           'verified_members': len(seen), 'original_bytes': history['original_bytes']}
        review = a.atomic_review if label == 'atomic' else a.review
        summary_path = review / 'report.json'
        summary = json.loads(summary_path.read_bytes())
        files.add(summary_path)
        assert summary['status'] == 'PARTIAL' and summary['native_scoped_qualification_passed'] is True
        assert summary['runtime_source_sha256'] == source_pin and summary['executable_sha256'] == image_pin
        assert summary['all_rng_complete'] is False and summary['all_tensor_dtypes_complete'] is False
        assert summary['full_pytorch_rewrite_complete'] is False
        assert summary['source_custody_after'] is True
        expected_names = (['actual-reference', 'native-controls', 'probability-math', 'owned-entry', 'typed-handoffs']
                          if label == 'atomic' else ['atomic-reference', 'explicit-reference', 'four-late-errors',
                                                   'native-controls', 'owned-entry', 'typed-handoffs'])
        assert [row['name'] for row in summary['suites']] == expected_names
        for row in summary['suites']:
            path = review / row['name'] / 'report.json'
            assert row['exit_code'] == 0 and digest(path) == row['report_sha256']
            report = json.loads(path.read_bytes())
            assert report['status'] == row['status']
            key = label + '/' + row['name']
            check_report_identity(key, report, profile, peer_pins, expected)
            if 'inputs' in report:
                inputs = report['inputs']
                assert inputs[str(directory / 'build.json')] == digest(directory / 'build.json')
                if row['name'] == 'probability-math':
                    assert inputs[str(directory / 'runtime.hepta')] == source_pin
                else:
                    assert inputs[str(directory / 'tensor-distributions-session')] == image_pin
            reports[key], report_hashes[key] = report, digest(path)
            files.add(path)
        summaries[label] = summary
    before = {str(path): digest(path) for path in sorted(files)}
    reference_path = repo / 'tools/cpu_distributions/evidence/torch-bernoulli-dropout.json'
    assert digest(reference_path) == REFERENCE
    reference = json.loads(reference_path.read_bytes())
    assert reference['status'] == 'OBSERVED' and reference['cases'] == 183
    assert len(reference['oracle_provenance']['checks']) == 11 and all(v is True for v in reference['oracle_provenance']['checks'].values())
    atomic = reports['atomic/actual-reference']
    old = reports['partial/atomic-reference']
    explicit = reports['partial/explicit-reference']
    for row in (atomic, old, explicit):
        assert row['reference_sha256'] == REFERENCE and row['reference_cases'] == 183 and row['input_custody_after'] is True
        assert len(row['actual_reference_provenance_checks']) == 11 and all(v is True for v in row['actual_reference_provenance_checks'].values())
        assert row['status'] == 'PARTIAL' and row['all_rng_complete'] is False and row['full_pytorch_rewrite_complete'] is False
        check_reference_rows(row, reference, explicit=row is explicit)
    assert atomic['counts'] == old['counts'] == {'PASS': 161, 'OPEN_TRANSACTION_DIFFERENCE': 4, 'OPEN_UNREPRESENTED': 18}
    assert explicit['counts'] == {'PASS': 165, 'OPEN_UNREPRESENTED': 18}
    assert [(x['index'], x['name'], x['status']) for x in atomic['rows']] == [(x['index'], x['name'], x['status']) for x in old['rows']]
    changed = []
    for previous, current in zip(old['rows'], explicit['rows'], strict=True):
        assert (previous['index'], previous['name']) == (current['index'], current['name'])
        if previous['status'] != current['status']:
            assert previous['status'] == 'OPEN_TRANSACTION_DIFFERENCE' and current['status'] == 'PASS'
            changed.append(current['index'])
    assert changed == LATE_ERROR_INDICES
    late_errors = reports['partial/four-late-errors']
    assert late_errors['reference_sha256'] == REFERENCE
    assert [r['reference_index'] for r in late_errors['cases']] == LATE_ERROR_INDICES
    for row in late_errors['cases']:
        assert row['name'] == reference['rows'][row['reference_index']]['name'] and row['status'] == 'PASS'
        dropout = row['reference_index'] == 179
        assert (row['atomic_opcode'], row['explicit_partial_opcode'], row['error'],
                row['completed_writes'], row['consumed_raw32'], row['Version_after_error']) == (
                    (84, 87, 5, 0, 8, 0) if dropout else (81, 86, 7, 2, 2, 0))
        for field in ('prefix_commit_suffix_stop', 'original_reference_words_and_complete_state_exact',
                      'old_atomic_contract_unchanged', 'temporary_mask_live_rows_reclaimed'):
            assert row[field] is True, field
    for row in (atomic, old):
        assert (row['frames_before_EOF'], row['exact_float32_words'], row['complete_state_comparisons'], row['rollback_bytes_compared']) == (1894, 49069, 182, 10861632)
    assert (explicit['frames_before_EOF'], explicit['exact_float32_words'], explicit['complete_state_comparisons'], explicit['rollback_bytes_compared']) == (1908, 49085, 182, 9051360)
    expected_counts = {
        'atomic/native-controls': {'case_count': 4, 'frames_before_EOF': 98, 'exact_value_comparisons': 1000023, 'rollback_bytes': 44193488},
        'atomic/probability-math': {'cases': 106488, 'native_u64_results': 425952, 'production_objects_byte_identical': True},
        'atomic/owned-entry': {'cases': 86, 'probe_processes': 84, 'live_frames': 6},
        'atomic/typed-handoffs': {'cases': 2, 'frames': 9, 'all_source_custody_unchanged': True},
        'partial/four-late-errors': {'case_count': 4, 'frames_before_EOF': 91, 'exact_words': 32, 'complete_state_comparisons': 16, 'rollback_bytes': 2413696},
        'partial/native-controls': {'case_count': 14, 'frames_before_EOF': 159, 'exact_float32_words': 108153, 'complete_state_comparisons': 19, 'rollback_bytes_compared': 6034240},
        'partial/owned-entry': {'raw_processes': 34, 'live_frames': 20, 'complete_owner_bytes_compared': 10559920},
        'partial/typed-handoffs': {'cases': 2, 'frames': 9, 'all_source_custody_unchanged': True},
    }
    for name, expected in expected_counts.items():
        assert reports[name]['status'] == 'PASS'
        assert all(reports[name][key] == value for key, value in expected.items()), name
    after = {str(path): digest(path) for path in sorted(files)}
    assert before == after
    result = {
        'schema': 'heptatorch.integrated-cpu-distributions-qualification.v2', 'status': 'PARTIAL',
        'native_scoped_qualification_passed': True,
        'atomic_source_sha256': PROFILES['atomic'][4], 'atomic_executable_sha256': PROFILES['atomic'][5],
        'source_sha256': PROFILES['partial'][4], 'executable_sha256': PROFILES['partial'][5],
        'reference_sha256': REFERENCE, 'oracle_provenance': reference['oracle_provenance'],
        'source_custody_before': before, 'source_custody_after': after,
        'cold_reproduction': reproductions, 'verified_failure_and_execution_archives': archives,
        'summaries': summaries, 'reports': reports, 'original_report_sha256': report_hashes,
        'original_actual_torch_observations': 183, 'explicit_strict_mapped_passes': 165,
        'open_unrepresented': 18, 'old_atomic_transaction_differences_retained': 4,
        'closed_explicit_error_indices_zero_based': changed,
        'open_unrepresented_indices_zero_based': OPEN_INDICES,
        'all_11_report_source_image_and_peer_identities_checked': True,
        'typed_renamed_sources_and_images_match_verified_archive': True,
        'all_reference_indices_names_families_and_status_counts_checked': True,
        'host_numeric_production_runtime': False, 'full_aggregate_host_admission_claimed': False,
        'real_module_and_linked_call_graph_admission': True,
        'all_rng_complete': False, 'all_tensor_dtypes_complete': False,
        'full_pytorch_rewrite_complete': False, 'pytorch_api_denominator': 2590,
        'count_policy': 'Repeated profiles, finite native controls and case subsets are not added into upstream schema coverage.',
        'open_scope': ['8 float64 probability Tensor observations', '2 int64 probability Tensor observations',
                       '8 distinct native_dropout bool-mask observations', 'normal-grad-mode tracked nonleaf inplace',
                       'kind-8 Optimizer and public-gradient-object composition', 'all other unsupported schemas'],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PARTIAL', 'native_scoped_qualification_passed': True,
                      'reports': len(reports), 'actual_cases': 183, 'strict_mapped': 165,
                      'unrepresented': 18, 'report_sha256': digest(a.output)}))


if __name__ == '__main__':
    main()
