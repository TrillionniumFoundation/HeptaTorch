"""Bind the frozen public-gradient qualification to source and native bytes.

This reader preserves every original report hash and embeds the complete
decoded reports for the hosted artifact. It does not execute or emulate a
Tensor operation, change a fixed expectation, or count suites as API coverage.
"""
from pathlib import Path
import argparse
import hashlib
import json


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'review', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = args.build.resolve()
    review = args.review.resolve()
    reads = {}

    def raw(path):
        path = Path(path).resolve()
        data = path.read_bytes()
        reads[path] = sha(data)
        return data

    def read(path):
        return json.loads(raw(path))

    manifest_path = root / 'tools/gradient_objects/package-manifest.json'
    manifest = read(manifest_path)
    assert reads[manifest_path] == '57b5ea629794fd50f35695695295a539291ea061e36b0266159a2de0e45f08ca'
    assert len(manifest['files']) == 124
    for name, digest in manifest['files'].items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert sha(raw(root / relative)) == digest, name
    profile = read(root / 'tools/gradient_objects/profile.json')
    source = '1b18da2cf9791e1b20982f9706434c1359ea194dd983dc6c66eb005f152e77b5'
    image = 'be18ab877f38a4027a527783db80a1418a137c7d6da91e5bc39e2af8c044cb6b'
    assert profile['source_sha256'] == manifest['source_sha256'] == source
    assert profile['native_sha256'] == manifest['native_sha256'] == image
    assert sha(raw(build / 'runtime.hepta')) == source
    assert sha(raw(build / 'tensor-gradient-object-session')) == image
    built = read(build / 'build.json')
    assert built['status'] == 'PASS' and built['source_sha256'] == source and built['native_sha256'] == image
    assert (built['functions'], built['blocks'], built['instructions'], built['descriptor_bytes']) == (355, 4675, 17450, 576)
    assert len(built['units']) == len(profile['units']) == 8
    for unit, pinned in zip(built['units'], profile['units'], strict=True):
        assert unit['name'] == pinned['name']
        for suffix, field in (('.hepta', 'source_sha256'), ('.o', 'object_sha256')):
            assert sha(raw(build / (unit['name'] + suffix))) == unit[field] == pinned[field]
    frozen = read(review / 'report.json')
    assert frozen['status'] == 'PASS' and frozen['step_count'] == 16
    assert frozen['source_sha256'] == source and frozen['native_sha256'] == image
    assert frozen['cold_overlay_and_full_authoring_image_identical'] is True
    assert frozen['all_package_sources_unchanged'] is True
    assert frozen['package_manifest_sha256'] == reads[manifest_path]
    expected_steps = [
        'cold-HIA-overlay-build', 'author-public-foundation', 'reproduce-full-authoring-lineage',
        'original-public-ten', 'incoming-25-observations', 'alias-edge-12-observations',
        'native-ownership-lifetimes', 'typed-result-references', 'signed-zero-13', 'partial-16',
        'presence-8', 'gradient-clear-10', 'build-handoff-optimizer', 'execute-handoff-optimizer',
        'build-handoff-public-foundation', 'execute-handoff-public-foundation']
    assert [step['name'] for step in frozen['steps']] == expected_steps
    reports = {}
    original_hashes = {}
    for step in frozen['steps']:
        name = step['name']
        assert step['exit_code'] == 0
        path = Path(step['report_path'])
        assert not path.is_absolute() and '..' not in path.parts
        data = raw(review / path)
        assert sha(data) == step['report_sha256'] == frozen['suite_reports'][name]['sha256']
        assert sha(raw(review / (name + '.stdout'))) == step['stdout_sha256']
        assert sha(raw(review / (name + '.stderr'))) == step['stderr_sha256']
        reports[name] = json.loads(data)
        assert reports[name]['status'] == step['report_status']
        original_hashes[name] = {'sha256': sha(data), 'bytes': len(data)}
    for folder, binary in [('native', 'tensor-gradient-object-session'),
                           ('authored-incoming', 'tensor-gradient-incoming-session')]:
        assert sha(raw(review / folder / 'runtime.hepta')) == source
        assert sha(raw(review / folder / binary)) == image
    reference_files = {
        'original-public-ten': 'torch-gradient-objects.json',
        'incoming-25-observations': 'torch-gradient-materialization.json',
        'alias-edge-12-observations': 'torch-gradient-alias-edge.json',
        'signed-zero-13': 'torch-gradient-signed-zero.json',
        'partial-16': 'torch-backward-partial.json',
        'presence-8': 'torch-gradient-presence.json',
        'gradient-clear-10': 'torch-grad-clear.json'}
    provenance_names = {'python_matches', 'platform_matches', 'version_backend_matches',
                        'git_matches_oracle_build_pin', 'distribution_version_matches',
                        'direct_url_matches', 'direct_url_archive_sha256_matches',
                        'pip_report_url_matches', 'pip_report_archive_sha256_matches',
                        'pip_report_version_matches', 'pip_report_requested'}
    for name, reference_name in reference_files.items():
        value = reports[name]
        assert value.get('source_sha256', value.get('native_source_sha256')) == source
        assert value.get('native_sha256', value.get('native_executable_sha256')) == image
        reference = raw(root / 'validation/gradient-objects/reference' / reference_name)
        assert value.get('reference_sha256', value.get('observations_sha256')) == sha(reference)
        provenance = value['oracle_provenance']
        assert provenance['status'] == 'PASS' and provenance['metadata_error'] is None
        assert set(provenance['checks']) == provenance_names
        assert all(item is True for item in provenance['checks'].values())
        assert provenance['lock_sha256'] == sha(raw(root / 'compatibility/pytorch-oracle-lock.json'))
        assert value['full_rewrite_complete'] is False and value['pytorch_api_denominator'] == 2590
    ten = reports['original-public-ten']
    assert (ten['qualified_groups'], ten['partial_groups'], ten['failed_groups']) == (9, 1, 0)
    incoming = reports['incoming-25-observations']
    assert (incoming['reference_variants'], incoming['executed_variants'], incoming['qualified_variants'],
            incoming['unrepresented_variants'], incoming['failed_variants']) == (25, 22, 22, 3, 0)
    alias = reports['alias-edge-12-observations']
    assert (alias['reference_variants'], alias['qualified_variants'], alias['failed_variants']) == (12, 12, 0)
    lifetime = reports['native-ownership-lifetimes']
    assert lifetime['cases'] == lifetime['passed'] == 5 and lifetime['native_executed'] is True
    assert reports['typed-result-references']['refusal_cases'] == 12
    assert (reports['signed-zero-13']['cases'], reports['signed-zero-13']['passed'], reports['signed-zero-13']['failed']) == (13, 13, 0)
    assert reports['partial-16']['cases'] == reports['partial-16']['passed'] == 16
    presence = reports['presence-8']
    assert presence['parameter_values_versions_gradient_presence_and_skip_passed'] == 8 and presence['unexpected_failures'] == 0
    assert (presence['state_version_difference_observations'], presence['state_creation_difference_observations']) == (8, 65)
    clear = reports['gradient-clear-10']
    assert clear['qualified'] == 10 and clear['unexpected_failures'] == 0
    assert (clear['state_version_difference_observations'], clear['state_creation_difference_observations'],
            clear['native_step_Tensor_absence_observations'], clear['Adam_different_bits']) == (11, 9, 30, 6)
    for name in ['optimizer', 'public-foundation']:
        gate = reports['execute-handoff-' + name]
        assert gate['status'] == 'PASS' and gate['native_executed'] is True and gate['new_source_sha256'] == source
        assert (gate['cases'], gate['frames']) == (2, 50)
        assert reports['build-handoff-' + name]['new_core_sha256'] == source
    raw(Path(__file__))
    assert all(sha(path.read_bytes()) == digest for path, digest in reads.items())
    result = {
        'schema': 'heptatorch.gradient-objects.root-qualification.v1',
        'status': 'QUALIFIED_COMPONENT_WITH_OPEN_SCOPE', 'source_sha256': source, 'native_sha256': image,
        'package_manifest_sha256': reads[manifest_path], 'build': built, 'cold_qualification': frozen,
        'reports': reports, 'original_report_sha256': original_hashes, 'formal_report_count': 16,
        'all_input_custody_unchanged': True, 'inputs_sha256': {str(path): digest for path, digest in reads.items()},
        'original_10_counts': {'qualified': 9, 'partial': 1, 'failed': 0},
        'incoming_25_counts': {'qualified': 22, 'unrepresented': 3, 'failed': 0},
        'alias_edge_12_counts': {'qualified': 12, 'failed': 0},
        'different_suite_counts_are_unique_upstream_API_coverage': False,
        'Adam_distribution_composition_qualified': False, 'hosted_live35_executed_here': False,
        'whole_project_decision': 'HOLD', 'full_pytorch_rewrite_complete': False, 'pytorch_api_denominator': 2590}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['status', 'source_sha256', 'native_sha256', 'formal_report_count', 'whole_project_decision']}))


if __name__ == '__main__':
    main()
