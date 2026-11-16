#!/usr/bin/env python3
"""Reproduce frozen release contracts and retain the known partial-error difference."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

CORE = 'e9e325374153f64744fbddfd2ccee00e570585cc42e4ad239361ce035ad4d2c2'
IMAGE = 'f146b71b5614b7d1022ef41875f5fbfd857e80028d33b2d634ecd255163f320b'
HANDOFF = '95f41e5cd03e682dd10b1b37e2ae5c8b944e34d51a8ff99308ae475a6766540f'
MANIFEST = 'f84af45658190243dcdbffdee124fe008327c9caa8293736fba536d0d79e5fc3'
OBSERVATIONS = '83bc3a03a51c5e2eb9499fb24ad73b9ab73da7452eb8ff4dc5b57e18a3958c7d'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--stack-build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for key in ('build', 'baseline', 'stack_build', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    repo = Path(__file__).resolve().parents[1]
    package = repo / 'tools/backward_release'
    manifest_path = repo / 'validation/backward-release-package.json'
    assert sha(manifest_path) == MANIFEST
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest['file_count'] == len(manifest['files']) == 30
    inputs = [Path(__file__), manifest_path]
    for relative, pin in manifest['files'].items():
        path = repo / relative
        assert sha(path) == pin['sha256'] and path.stat().st_size == pin['bytes'], relative
        inputs.append(path)
    compiler = args.stack_build / 'native/stack-module-compiler'
    linker = args.stack_build / 'native/stack-linker-session'
    observer = repo / 'tools/owned_sessions/client.py'
    protocol = repo / 'tools/tensor_objects/protocol.py'
    image = args.build / 'tensor-release-session'
    core = args.build / 'runtime.hepta'
    observations = package / 'evidence/actual-torch-36-observations.json'
    assert sha(image) == IMAGE and sha(core) == CORE and sha(observations) == OBSERVATIONS
    assert sha(args.baseline / 'tensor-object-session') == '2b39ff1a5a258c5fae7be1fbca5dcf0764787b43fa72ad8fdc3ab6fd119e2a91'
    for directory, glob in [(args.build, 'release_*.hepta'), (args.baseline, 'objects_*.hepta')]:
        inputs.extend(sorted(directory.glob(glob)))
        inputs.extend([directory / 'runtime.hepta', directory / 'build.json'])
    inputs.extend([compiler, linker, observer, protocol, image, args.baseline / 'tensor-object-session'])
    before = {str(path): sha(path) for path in inputs}
    args.output.mkdir(parents=True, exist_ok=False)

    def execute(name, script, *arguments):
        # The frozen replay scripts import their frozen sibling controls. -B
        # preserves that explicit script-directory import without modifying them.
        command = [sys.executable, '-B', str(package / script), *map(str, arguments)]
        result = subprocess.run(command, cwd=repo, capture_output=True, timeout=240)
        (args.output / (name + '.stdout')).write_bytes(result.stdout)
        (args.output / (name + '.stderr')).write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr, (name, result.returncode, result.stderr[-2000:])

    common = ['--binary', image, '--expected-sha256', IMAGE, '--source', core,
              '--expected-source-sha256', CORE, '--transport-observer', observer]
    execute('native', 'test_release.py', '--executable', image, '--protocol', protocol,
            '--output', args.output / 'native')
    execute('independent', 'native_release_controls.py', *common, '--output', args.output / 'independent')
    for name, script in [('operators', 'replay_backward_release.py'), ('lifetimes', 'replay_backward_lifetimes.py')]:
        execute(name, script, *common, '--observations', observations,
                '--expected-observations-sha256', OBSERVATIONS, '--output', args.output / name)
    execute('handoff-build', 'build_state_handoff.py', '--old', args.baseline, '--new', args.build,
            '--compiler', compiler, '--linker', linker, '--output', args.output / 'handoff-build')
    execute('handoff', 'test_state_handoff.py', '--executable', args.output / 'handoff-build/state-handoff-session',
            '--build-receipt', args.output / 'handoff-build/report.json', '--protocol', protocol,
            '--output', args.output / 'handoff')
    paths = {name: args.output / name / 'report.json' for name in
             ('native', 'independent', 'operators', 'lifetimes', 'handoff-build', 'handoff')}
    reports = {name: json.loads(path.read_bytes()) for name, path in paths.items()}
    native, independent, operators, lifetime, handoff_build, handoff = (reports[x] for x in paths)
    assert len(native['cases']) == 10 and all(x['status'] == 'PASS' for x in native['cases'])
    assert native['status'] == 'PASS' and native['frames'] == 144 and native['rollback_bytes'] == 13317008
    assert independent['status'] == 'PASS' and independent['cases'] == independent['passed'] == 14
    assert independent['frames'] == 207 and independent['exact_scalar_or_binary32_comparisons'] == 107
    assert operators['status'] == 'PASS' and operators['cases'] == operators['passed'] == 20
    assert operators['frames'] == 353 and operators['exact_scalar_or_binary32_comparisons'] == 1363
    assert lifetime['status'] == 'QUALIFIED_WITH_OPEN_DIFFERENCE'
    assert lifetime['cases'] == lifetime['native_contract_verified_cases'] == 16
    assert lifetime['pytorch_behavior_matched_cases'] == 15 and lifetime['declared_difference_cases'] == 1
    assert lifetime['unexpected_failures'] == 0 and lifetime['frames'] == 183
    assert lifetime['exact_scalar_or_binary32_comparisons'] == 142 and lifetime['approximate_binary32_comparisons'] == 15
    for report in (operators, lifetime):
        provenance = report['oracle_provenance']
        assert provenance['status'] == 'PASS' and len(provenance['checks']) == 11
        assert all(x is True for x in provenance['checks'].values())
        assert report['observations_sha256'] == OBSERVATIONS
        assert report['native_executable_sha256'] == IMAGE and report['native_source_sha256'] == CORE
    assert handoff_build['status'] == handoff['status'] == 'PASS'
    assert handoff_build['native_sha256'] == handoff['native_sha256'] == HANDOFF
    assert sha(args.output / 'handoff-build/state-handoff-session') == HANDOFF
    assert handoff_build['module_count'] == 11 and handoff['cases'] == 2 and handoff['frames'] == 28
    assert handoff['entire_owner_bytes_unchanged'] == 595232
    after = {str(path): sha(path) for path in inputs}
    assert before == after
    build_report = json.loads((args.build / 'build.json').read_bytes())
    assert build_report['native_sha256'] == IMAGE and build_report['source_sha256'] == CORE
    report = {'schema': 'heptatorch.integrated-backward-release-qualification.v1',
              'status': 'QUALIFIED_WITH_OPEN_DIFFERENCE', 'native_contracts': 'PASS',
              'native_executable_sha256': IMAGE, 'native_source_sha256': CORE,
              'manifest_sha256': MANIFEST, 'source_custody_before': before,
              'source_custody_after': after, 'build': build_report, 'reports': reports,
              'report_sha256': {name: sha(path) for name, path in paths.items()},
              'matched_pytorch_observations': 35, 'declared_partial_failure_difference': 1,
              'pytorch_executed_in_this_process': False, 'reference_is_actual_pinned_hosted_observation': True,
              'pytorch_api_denominator': 2590, 'whole_pytorch_rewrite_complete': False}
    output = args.output / 'report.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'native_contracts': 'PASS',
                      'matched_pytorch_observations': 35, 'declared_difference': 1,
                      'report_sha256': sha(output)}, indent=2))


if __name__ == '__main__':
    main()
