"""Bind session CI results to frozen sources, native images and oracle provenance."""
from pathlib import Path
import argparse
import hashlib
import json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check_sources():
    manifest = read('validation/session-source-manifest.json')
    rows = manifest['rows']
    assert manifest['files'] == len(rows) == len({row['path'] for row in rows}) == 17
    assert manifest['original_frozen_package_files'] == 15
    for row in rows:
        assert Path(row['path']).stat().st_size == row['bytes']
        assert digest(row['path']) == row['sha256'], row['path']
    pins = read('tools/session/profile.json')
    assert digest('src/heptatorch.hepta') == pins['canonical_source_sha256']
    assert digest('tools/session/entry.hepta') == pins['session_entry_sha256']
    assert digest('tools/modules/split_runtime.py') == pins['splitter_sha256']
    assert digest('tests/pytorch_session_oracle.py') == 'b219fb63e7698f09951454cce9073157689398d38b1ea3d21e22b412a0dce606'
    return pins


def native(review):
    pins = check_sources()
    historical = read('validation/persistent-session-native.json')
    reports = []
    for mode, executable in [('modular', 'build/heptatorch-session'),
                             ('monolithic', 'build/heptatorch-session-monolithic')]:
        expected = pins[mode + '_session_sha256']
        assert digest(executable) == expected
        build = read(executable + '-build.json')
        assert build['pass'] is True and build['mode'] == mode
        assert build['source_sha256'] == pins['canonical_source_sha256']
        assert build['overlay_sha256'] == pins['overlay_source_sha256']
        assert build['application_sha256'] == expected
        assert build['unchanged_function_count'] == 142 and build['init_unchanged'] is True
        assert build['changed_functions'] == ['entry']
        assert build['producer_source_entry_splitter_hashes_verified_before_and_after'] is True
        assert build['unit_snapshots_unchanged'] is True
        path = Path('build/session-independent-' + mode + '.json')
        observed = read(path)
        original = historical['reports']['root_independent_' + mode]
        assert observed == original['report'], mode
        assert digest(path) == original['original_file_sha256']
        assert observed['status'] == 'PASS' and observed['cases'] == len(observed['rows']) == 477
        assert observed['fixture_interoperation_cases'] == 461
        assert observed['executable_sha256'] == expected
        assert observed['executable_unchanged'] is True
        assert observed['every_successful_exchange_observed_before_stdin_eof'] is True
        assert all(row['status'] == 'PASS' for row in observed['rows'])
        reports.append({'build': mode, 'status': 'PASS', 'native_sha256': expected,
                        'cases': 477, 'fixture_interoperation_cases': 461,
                        'full_report_sha256': digest(path),
                        'exactly_matches_source_bound_independent_receipt': True,
                        'runtime_build_receipt_sha256': digest(executable + '-build.json')})

    suites = []
    for name, count, key in [('packer', 22, 'wide_packer_boundaries'),
                             ('growth', 3, 'actual_modular_growth'),
                             ('flat', 18, 'peer_native_flat_linker')]:
        path = review / name / 'report.json'
        report = read(path)
        original = historical['reports'][key]['report']
        assert report.get('pass') is True or report.get('status') == 'PASS'
        assert report.get('cases', report.get('checks')) == len(report['rows']) == count
        assert {row['case'] for row in report['rows']} == {row['case'] for row in original['rows']}
        assert all(row.get('pass') is True or row.get('status') == 'PASS' for row in report['rows'])
        if name == 'packer':
            assert report['executable_sha256'] == pins['wide_packer_sha256']
            assert report['image_sha256'] == pins['wide_packer_image_sha256']
            assert report['maximum_image_bytes_actually_packed'] == 8 << 20
            assert report['maximum_input_bytes_actually_admitted'] == (8 << 20) + 16
            assert report['maximum_output_bytes_actually_produced'] == (8 << 20) + 774
        elif name == 'growth':
            assert report['module_compiler_sha256'] == pins['module_compiler_sha256']
            assert report['flat_linker_sha256'] == pins['flat_linker_sha256']
            assert report['wide_packer_sha256'] == pins['wide_packer_sha256']
            assert report['synthetic_capacity_probe_not_pytorch_operator_coverage'] is True
        else:
            assert report['linker_executable_sha256'] == pins['flat_linker_sha256']
            assert report['linker_source_sha256'] == pins['flat_linker_source_sha256']
            assert report['unchanged_hia_functions'] == 25
            assert report['raw_image_executions'] == 6 and report['malformed_request_refusals'] == 8
        suites.append({'suite': name, 'status': 'PASS', 'cases': count,
                       'report_sha256': digest(path), 'report': report})
    return {'schema': 'heptatorch.hosted-persistent-session-native-gates.v1',
            'status': 'PASS', 'source_sha256': pins['canonical_source_sha256'],
            'source_manifest_sha256': digest('validation/session-source-manifest.json'),
            'builds': reports, 'tool_suites': suites,
            'whole_pytorch_rewrite_complete': False}


def oracle():
    pins = check_sources()
    expected_checks = {'python_matches', 'platform_matches', 'version_backend_matches',
                       'git_matches_oracle_build_pin', 'distribution_version_matches',
                       'direct_url_matches', 'direct_url_archive_sha256_matches',
                       'pip_report_url_matches', 'pip_report_archive_sha256_matches',
                       'pip_report_version_matches', 'pip_report_requested'}
    reports = []
    names = None
    for mode, executable in [('modular', 'build/heptatorch-session'),
                             ('monolithic', 'build/heptatorch-session-monolithic')]:
        path = Path('build/pytorch-session-' + mode + '.json')
        report = read(path)
        assert report['status'] == 'PASS' and report['failed_cases'] == 0
        assert report['expected_cases'] == report['cases'] == len(report['rows']) == 21
        assert report['native_unchanged_during_run'] is True
        assert report['native_executable_sha256'] == digest(executable) == pins[mode + '_session_sha256']
        assert report['oracle_script_sha256'] == digest('tests/pytorch_session_oracle.py')
        assert report['reference_support_sha256'] == digest('tests/pytorch_oracle.py')
        provenance = report['oracle_provenance']
        assert provenance['status'] == 'PASS' and provenance['metadata_error'] is None
        assert set(provenance['checks']) == expected_checks
        assert all(value is True for value in provenance['checks'].values())
        assert provenance['lock_sha256'] == digest('compatibility/pytorch-oracle-lock.json')
        case_names = {row['case'] for row in report['rows']}
        assert len(case_names) == 21 and all(row['status'] == 'PASS' for row in report['rows'])
        assert all(row['frames'] > 0 for row in report['rows'])
        assert report['frames'] == sum(row['frames'] for row in report['rows'])
        assert report['compared_values'] == sum(row['compared_values'] for row in report['rows'])
        if names is not None:
            assert names == case_names
        names = case_names
        reports.append({'build': mode, 'status': 'PASS', 'cases': 21,
                        'frames': report['frames'], 'compared_values': report['compared_values'],
                        'native_executable_sha256': report['native_executable_sha256'],
                        'oracle_provenance_checks': len(expected_checks),
                        'report_path': str(path), 'report_sha256': digest(path)})
    return {'schema': 'heptatorch.hosted-persistent-session-oracle-summary.v1',
            'status': 'PASS', 'source_sha256': pins['canonical_source_sha256'],
            'oracle_script_sha256': digest('tests/pytorch_session_oracle.py'),
            'reference_support_sha256': digest('tests/pytorch_oracle.py'),
            'builds': reports, 'whole_pytorch_rewrite_complete': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['sources', 'native', 'oracle'])
    parser.add_argument('--review', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'sources':
        pins = check_sources()
        result = {'schema': 'heptatorch.session-source-custody.v1', 'status': 'PASS',
                  'source_sha256': pins['canonical_source_sha256'], 'frozen_files': 17}
    elif args.mode == 'native':
        assert args.review is not None
        result = native(args.review)
    else:
        result = oracle()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
