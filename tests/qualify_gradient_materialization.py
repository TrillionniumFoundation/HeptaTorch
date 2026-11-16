"""Bind the frozen partial04 failure to strict native gradient05 qualification."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

CORE = '9feab47afea6723284e15a2ad6c9071dc8c5bdc27b572d2eef4ab4628ed1aea9'
IMAGE = '4a62b796d38317e630381b7e2e57e8e7824d78c1f721e3d69bd59f59e04ca894'
OLD_CORE = '302677af52d3e04520e6f132aaf17b98cbe5d2cf404aec040bf5144e324d2d4d'
OLD_IMAGE = '8f1fae7cc6a9181d788ef80c82a43e24c3ba526e77d01f8a761ea5d9913be4ae'
MANIFESTS = {
    'backward-partial-package.json': ('33b7677b91a3615149c86322b5c4e624749e682e7331061367cfe35fba49e340', 39),
    'gradient-materialization-package.json': ('d37865593b3e2c64d0f7ba6e9b134fe0258f7a8098164d5bcd524e0a1116963b', 24),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'partial-build', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    build, old_build, out = args.build.resolve(), args.partial_build.resolve(), args.output.resolve()
    package = repo/'tools/backward_partial'
    material = repo/'tools/gradient_materialization'
    observer, protocol = repo/'tools/owned_sessions/client.py', repo/'tools/tensor_objects/protocol.py'
    binary, source = build/'tensor-partial-session', build/'runtime.hepta'
    old_binary, old_source = old_build/'tensor-partial-session', old_build/'runtime.hepta'
    assert (sha(binary), sha(source), sha(old_binary), sha(old_source)) == (IMAGE, CORE, OLD_IMAGE, OLD_CORE)
    inputs = [Path(__file__), binary, source, old_binary, old_source, observer, protocol]
    for name, (digest, count) in MANIFESTS.items():
        path = repo/'validation'/name
        assert sha(path) == digest
        inputs.append(path)
        manifest = json.loads(path.read_bytes())
        assert manifest['file_count'] == len(manifest['files']) == count
        for relative, pin in manifest['files'].items():
            item = repo/relative
            assert sha(item) == pin['sha256'] and item.stat().st_size == pin['bytes'], relative
            inputs.append(item)
    before = {str(path): sha(path) for path in inputs}
    out.mkdir(parents=True, exist_ok=False)
    reports = {}

    def execute(name, script, *arguments, expected_exit=0):
        command = [sys.executable, '-B', str(script), *map(str, arguments), '--output', str(out/name)]
        result = subprocess.run(command, cwd=repo, capture_output=True, timeout=240)
        (out/(name+'.stdout')).write_bytes(result.stdout)
        (out/(name+'.stderr')).write_bytes(result.stderr)
        assert result.returncode == expected_exit and not result.stderr, (name, result.returncode, result.stderr[-2000:])
        reports[name] = json.loads((out/name/'report.json').read_bytes())

    common = ['--binary', binary, '--source', source, '--expected-sha256', IMAGE,
              '--expected-source-sha256', CORE, '--transport-observer', observer]
    replay = common+['--controls', package/'native_release_controls.py']
    old_replay = ['--binary', old_binary, '--source', old_source, '--expected-sha256', OLD_IMAGE,
                  '--expected-source-sha256', OLD_CORE, '--transport-observer', observer,
                  '--controls', package/'native_release_controls.py']
    execute('original-signedzero-04', package/'replay_gradient_signed_zero.py', *old_replay,
            '--observations', package/'evidence/actual-torch-signed-zero-observations.json', expected_exit=1)
    execute('partial-contracts', package/'test_partial.py', '--executable', binary, '--protocol', protocol,
            '--observations', repo/'tools/backward_release/evidence/actual-torch-36-observations.json')
    execute('presence-contracts', package/'test_presence_bounds.py', '--executable', binary, '--protocol', protocol)
    execute('legacy57', package/'native_release_controls.py', *common)
    execute('partial-controls', package/'native_partial_controls.py', *replay,
            '--observations', repo/'tools/backward_release/evidence/actual-torch-36-observations.json')
    for name, script, observations in [
        ('signedzero', 'replay_gradient_signed_zero.py', 'actual-torch-signed-zero-observations.json'),
        ('partial', 'replay_partial_observations.py', 'actual-torch-16-observations.json'),
        ('presence', 'replay_gradient_presence.py', 'actual-torch-gradient-presence-observations.json'),
    ]:
        execute(name, package/script, *replay, '--observations', package/'evidence'/observations)
    execute('materialization', material/'test_materialization.py', '--binary', binary, '--source', source,
            '--expected-sha256', IMAGE, '--expected-source-sha256', CORE, '--protocol', protocol)
    traces = [item for item in sorted((out/'materialization').iterdir()) if (item/'transport.json').is_file()]
    assert len(traces) == 4
    execute('owner-fields', material/'check_materialization_owner_marks.py', *common,
            *[part for trace in traces for part in ('--trace', trace)])

    old = reports['original-signedzero-04']
    assert old['status'] == 'FAIL' and (old['cases'], old['passed'], old['failed'], old['different_word_observations']) == (13, 4, 9, 15)
    assert old['native_executable_sha256'] == OLD_IMAGE and old['native_source_sha256'] == OLD_CORE
    for name, cases, frames in [('partial-contracts',11,136), ('presence-contracts',8,216), ('legacy57',14,207), ('partial-controls',4,55)]:
        r = reports[name]
        assert r['status'] == 'PASS' and r['cases'] == cases and r['frames'] == frames, name
    zero, partial, presence = reports['signedzero'], reports['partial'], reports['presence']
    assert zero['status'] == 'PASS' and zero['cases'] == zero['passed'] == 13 and zero['failed'] == 0
    assert (zero['frames'], zero['exact_metadata_comparisons'], zero['exact_binary32_word_comparisons'], zero['different_word_observations']) == (166,131,70,0)
    assert partial['status'] == 'PASS' and partial['cases'] == partial['passed'] == 16
    assert (partial['frames'], partial['exact_scalar_or_binary32_comparisons']) == (606,1587)
    assert presence['status'] == 'QUALIFIED_WITH_OPEN_STATE_LIFECYCLE'
    assert presence['cases'] == presence['parameter_values_versions_gradient_presence_and_skip_passed'] == 8
    assert presence['unexpected_failures'] == 0
    assert (presence['frames'], presence['exact_scalar_or_binary32_comparisons'], presence['adam_tolerance_comparisons'], presence['adam_compared_values_with_different_bits']) == (880,2088,180,0)
    assert (presence['state_version_difference_observations'], presence['state_creation_difference_observations']) == (8,65)
    for r in (zero, partial, presence):
        assert r['native_executable_sha256'] == IMAGE and r['native_source_sha256'] == CORE
        p = r['oracle_provenance']
        assert p['status'] == 'PASS' and len(p['checks']) == 11 and all(x is True for x in p['checks'].values())
    first = reports['materialization']
    assert first['status'] == 'PASS' and len(first['rows']) == 4
    assert (first['frames'],first['exact_gradient_words'],first['zero_transient_fields']) == (104,204,897)
    owner = reports['owner-fields']
    assert owner['status'] == 'PASS' and owner['native_control_cases'] == 4
    assert owner['frames'] == 104 and owner['zero_ephemeral_word_observations'] > 0
    after = {str(path): sha(path) for path in inputs}; assert before == after
    report = {'schema': 'heptatorch.integrated-gradient-materialization-qualification.v1',
              'status': 'QUALIFIED_WITH_OPEN_STATE_LIFECYCLE', 'gradient_materialization_contracts': 'PASS',
              'native_executable_sha256': IMAGE, 'native_source_sha256': CORE,
              'source_custody_before': before, 'source_custody_after': after,
              'original_signed_zero_failure_reproduced': True,
              'signed_zero_cases': 13, 'partial_execution_cases': 16, 'presence_cases': 8,
              'open_state_version_observations': 8, 'open_lazy_state_creation_observations': 65,
              'reports': reports, 'report_sha256': {name: sha(out/name/'report.json') for name in reports},
              'pytorch_executed_in_this_process': False, 'reference_is_actual_pinned_hosted_observation': True,
              'pytorch_api_denominator': 2590, 'whole_pytorch_rewrite_complete': False}
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'], 'materialization_contracts':'PASS',
                      'original_failure_retained':True, 'report_sha256':sha(out/'report.json')}))


if __name__ == '__main__':
    main()
