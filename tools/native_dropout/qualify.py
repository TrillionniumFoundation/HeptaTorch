"""Cold-build the typed profile and execute its separately named native suites.

No expected result comes from a candidate execution. Fixed actual observations
and frozen test scripts are hashed before and after the run. Overlapping suites
are not added into a unique PyTorch test or API count.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'repository', 'upstream', 'compiler', 'linker', 'flat-compiler', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('inherited-package', 'retained-package', 'retained-review-package'):
        parser.add_argument('--'+name, type=Path)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    repo = a.repository.resolve()
    inherited = (a.inherited_package or repo).resolve()
    retained = (a.retained_package or repo).resolve()
    retained_review = (a.retained_review_package or repo).resolve()
    tests = PACKAGE/'tests/native_dropout_review'
    refs = PACKAGE/'validation/native-dropout/reference'
    profile_raw = (HERE/'profile.json').read_bytes()
    profile = json.loads(profile_raw)
    manifest_path = HERE/'package-manifest.json'
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)

    def custody():
        assert manifest_path.read_bytes() == manifest_raw
        assert (HERE/'profile.json').read_bytes() == profile_raw
        for name, pin in manifest['files'].items():
            path = PACKAGE/name
            raw = path.read_bytes()
            assert len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], name
            assert path.stat().st_mode & 0o777 == pin['mode'], ('mode', name)

    custody()
    assert sha(a.compiler.read_bytes()) == profile['compiler_sha256']
    assert sha(a.linker.read_bytes()) == profile['linker_sha256']
    assert sha(a.flat_compiler.read_bytes()) == profile['flat_compiler_sha256']
    for name, pin in profile['fixed_references'].items():
        path = refs/name
        raw = path.read_bytes()
        assert sha(raw) == pin['sha256']
        record = json.loads(raw)
        assert record['status'] == 'OBSERVED' and record['cases'] == pin['cases']
        assert record['probe_error_count'] == 0 and len(record['oracle_provenance']['checks']) == 11
        assert all(value is True for value in record['oracle_provenance']['checks'].values())
    report = {'schema':'heptatorch.native-dropout-portable-qualification.v1', 'status':'RUNNING',
              'source_sha256':profile['source_sha256'], 'native_sha256':profile['native_sha256'],
              'profile_sha256':sha(profile_raw), 'package_manifest_sha256':sha(manifest_raw),
              'steps':[], 'counting_boundary':'Named suites overlap; totals are not a unique upstream API or test count.',
              'Torch_executed_here':False, 'full_pytorch_rewrite_complete':False,
              'pytorch_api_denominator':2590}

    def save():
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

    def run(name, script, arguments, receipt, expected_status, counts=None):
        command = [sys.executable, '-B', str(script), *map(str, arguments)]
        stdout, stderr = out/(name+'.stdout'), out/(name+'.stderr')
        with stdout.open('wb') as so, stderr.open('wb') as se:
            process = subprocess.run(command, stdout=so, stderr=se, timeout=900)
        row = {'name':name, 'command':command, 'exit_code':process.returncode,
               'stdout_sha256':sha(stdout.read_bytes()), 'stderr_sha256':sha(stderr.read_bytes())}
        report['steps'].append(row)
        save()
        assert process.returncode == 0, (name, process.returncode)
        raw = receipt.read_bytes()
        result = json.loads(raw)
        row.update(report=str(receipt.relative_to(out)), report_sha256=sha(raw),
                   status=result['status'], expected_status=expected_status)
        save()
        assert result['status'] == expected_status, (name, result['status'], expected_status)
        for key, expected in (counts or {}).items():
            assert result[key] == expected, (name, key, result[key], expected)
        return result

    try:
        native = out/'native'
        run('cold-native-build', HERE/'build.py', [
            '--parent', a.parent.resolve(), '--repository', repo, '--upstream', a.upstream.resolve(),
            '--compiler', a.compiler.resolve(), '--linker', a.linker.resolve(), '--output', native],
            native/'profile-rebuild.json', 'PASS', {'units_reproduced_exactly':11})
        wire = tests/'owned-tensor-object-extension/protocol.py'
        common = ['--binary', native/'tensor-dtype-dropout-session', '--source', native/'runtime.hepta',
                  '--protocol', wire]
        for name, filename, extra, counts in (
            ('typed-slice', 'test_slice.py', [], {'frames':48}),
            ('typed-lifetimes', 'test_typed_lifetimes.py', [], {'frames':343}),
            ('bool-logical-not', 'test_logical_not.py', ['--flat-compiler', a.flat_compiler.resolve()], {'frames':66}),
            ('dtype-admission101', 'test_dtype_admission_v3.py', [], {'frames':126}),
            ('owned-allocation', 'test_dtype_allocation_v2.py', [], {'frames':120}),
            ('history-mul-controls8', 'test_history_mul_controls.py',
             ['--reference', refs/'torch-empty-dropout-identity.json'], {'cases':8, 'passed':8, 'frames':216}),
            ('actual-native-dropout28', 'replay_contract_identity.py',
             ['--reference', refs/'torch-native-dropout.json'], {'cases':28, 'frames':1816}),
            ('original-native-dropout8', 'replay_original_eight.py',
             ['--reference', inherited/'validation/gradient-adam-distributions/reference/torch-bernoulli-dropout.json'],
             {'cases':8, 'passed':8, 'frames':140}),
            ('actual-existing-optimizer6', 'replay_existing_optimizer.py',
             ['--observer', tests/'identity_observer.py', '--reference', refs/'torch-empty-dropout-existing-optimizer.json'],
             {'cases':6, 'passed':6, 'frames':2114}),
            ('actual-sgd-effective-layout6', 'replay_sgd_effective_layout.py',
             ['--observer', tests/'identity_observer.py', '--reference', refs/'torch-sgd-effective-layout.json'],
             {'cases':6, 'passed':6, 'frames':891})):
            run(name, tests/filename, common+extra+['--output', out/name], out/name/'report.json', 'PASS', counts)
        retained_result = run('actual-retained12', tests/'inherited-adapters/replay_retained_nonleaf.py', [
            '--observations', retained_review/'validation/retained-nonleaf-review/reference/torch-retained-nonleaf.json',
            '--probe', retained_review/'validation/retained-nonleaf-review/reference/pytorch_retained_nonleaf_probe.py',
            '--protocol', retained_review/'tests/retained_nonleaf_review/protocol.py',
            '--binary', native/'tensor-dtype-dropout-session', '--source', native/'runtime.hepta',
            '--expected-sha256', profile['native_sha256'], '--expected-source-sha256', profile['source_sha256'],
            '--state-tag', profile['state_format'], '--backing-tag', profile['backing_format'],
            '--output', out/'actual-retained12'], out/'actual-retained12/report.json',
            'QUALIFIED_EXPOSED_SCOPE', {'variants_executed':12, 'variants_qualified':12, 'unexpected_failures':0})
        assert retained_result['inputs']['observations'] == 'ca8b5d3a7c679dcc5ad9a3d7b2bbd38f664c90e6474cf3e3c28496d5b716d1b3'
        assert len(retained_result['reference_provenance_checks']) == 11
        assert all(value is True for value in retained_result['reference_provenance_checks'].values())
        assert retained_result['all_inputs_unchanged'] is True
        run('inherited16', tests/'run_inherited.py', [
            '--native-build', native, '--parent-package', inherited, '--retained-package', retained,
            '--adapters', tests/'inherited-adapters', '--output', out/'inherited16'],
            out/'inherited16/report.json', 'PASS', {'step_count':16, 'all_sources_unchanged':True})
        custody()
        assert sha((native/'runtime.hepta').read_bytes()) == profile['source_sha256']
        assert sha((native/'tensor-dtype-dropout-session').read_bytes()) == profile['native_sha256']
        report.update(status='PASS', step_count=len(report['steps']), all_package_members_unchanged=True,
                      all_expected_statuses_and_counts_matched=True)
        save()
        print(json.dumps({key:value for key,value in report.items() if key != 'steps'}))
        return 0
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
