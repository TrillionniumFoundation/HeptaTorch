"""Rebuild retained HIA and replay separately counted inherited command suites.

This coordinator reads immutable parent tests/references. Three explicit host
adapters change only private tags and the now-implemented93/94 admission tests;
all inverses are checked. Actual retain12 remains an independent qualification.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys
HERE=Path(__file__).resolve().parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('native-build','parent-package','retained-package','adapters','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    package=a.parent_package.resolve()
    tests=package/'tests/gradient_adam_distributions_review'
    references=package/'validation/gradient-adam-distributions/reference'
    local_package=a.retained_package.resolve()
    adapters=local_package/'tests/retained_gradient_regression'
    parent_manifest_file=package/'tools/gradient_adam_distributions/package-manifest.json'
    parent_manifest_bytes=parent_manifest_file.read_bytes()
    assert sha(parent_manifest_bytes)=='3e8c80053365f7e5de1d24725c511f49f256e8437f5a6d3556f09ff07682dc69'
    parent_manifest=json.loads(parent_manifest_bytes)
    profile=json.loads((a.native_build/'build.json').read_bytes())
    assert profile['status']=='PASS' and profile['descriptor_bytes']==640
    native_path=a.native_build/'tensor-integer-session'
    assert sha(native_path.read_bytes())==profile['native_sha256']
    assert sha((a.native_build/'runtime.hepta').read_bytes())==profile['source_sha256']
    own_manifest_file=local_package/'tools/retained_gradients/package-manifest.json'
    own_manifest_bytes=own_manifest_file.read_bytes()
    own_manifest=json.loads(own_manifest_bytes)
    own_inputs={local_package/name:(local_package/name).read_bytes() for name in own_manifest['files']}
    def custody():
        assert parent_manifest_file.read_bytes()==parent_manifest_bytes
        for name,pin in parent_manifest['files'].items():
            raw=(package/name).read_bytes()
            assert len(raw)==pin['bytes'] and sha(raw)==pin['sha256'],('parent package changed',name)
        assert own_manifest_file.read_bytes()==own_manifest_bytes
        for name,pin in own_manifest['files'].items():
            path=local_package/name;raw=path.read_bytes()
            assert len(raw)==pin['bytes'] and sha(raw)==pin['sha256'] and raw==own_inputs[path],path
        for row in json.loads(own_inputs[adapters/'custody.json'])['rows']:
            old=(tests/row['file']).read_bytes();new=(adapters/row['file']).read_bytes()
            assert sha(old)==row['original_sha256'] and sha(new)==row['adapted_sha256']
            inverse=new
            for edit in reversed(row['edits']):
                before,after=edit['before'].encode(),edit['after'].encode()
                assert inverse.count(after)==1;inverse=inverse.replace(after,before,1)
            assert inverse==old
    custody()
    report={'schema':'heptatorch.dtype-inherited-command-qualification.v1','status':'RUNNING',
        'source_sha256':profile['source_sha256'],'native_sha256':profile['native_sha256'],
        'parent_manifest_sha256':sha(parent_manifest_bytes),'package_manifest_sha256':sha(own_manifest_bytes),'own_inputs_sha256':{str(p):sha(r) for p,r in own_inputs.items()},
        'steps':[],'suite_reports':{},'native_executed':True,'pinned_Torch_executed_here':False,
        'scope':'Existing separately compiled dtype native receipt plus inherited commands, with exact inverse tag/descriptor-stride observer adaptations. No cold build or new live Torch is claimed here.',
        'counting_boundary':'These suites overlap; no sum is a unique upstream test or API count.',
        'full_rewrite_complete':False,'pytorch_api_denominator':2590}
    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(name,script,arguments,expected_report):
        replacement=a.adapters/script.name
        if replacement.exists():
            script=replacement
        command=[sys.executable,'-I','-S','-B',str(script),*map(str,arguments)]
        completed=subprocess.run(command,capture_output=True,timeout=600)
        (out/(name+'.stdout')).write_bytes(completed.stdout);(out/(name+'.stderr')).write_bytes(completed.stderr)
        step={'name':name,'command':command,'exit_code':completed.returncode,
            'stdout_sha256':sha(completed.stdout),'stderr_sha256':sha(completed.stderr)}
        report['steps'].append(step);save()
        assert completed.returncode==0,(name,completed.returncode,completed.stderr.decode(errors='replace')[-3000:])
        raw=expected_report.read_bytes();result=json.loads(raw)
        step.update(report_sha256=sha(raw),report_path=expected_report.relative_to(out).as_posix(),report_status=result['status'])
        report['suite_reports'][name]={'path':step['report_path'],'sha256':step['report_sha256']};save()
        return result
    try:
        native=a.native_build.resolve()
        cold=json.loads((native/'build.json').read_bytes())
        assert cold['source_sha256']==profile['source_sha256'] and cold['native_sha256']==profile['native_sha256']
        binary, source = native / 'tensor-integer-session', native / 'runtime.hepta'
        common = ['--binary', binary, '--source', source, '--controls', tests / 'native_release_controls.py',
                  '--expected-sha256', profile['native_sha256'], '--expected-source-sha256', profile['source_sha256']]
        layout = ['--state-format', profile['state_format'], '--backing-format', profile['backing_format'],
                  '--descriptor-bytes', '640']
        observer = ['--observer', tests / 'client.py']
        object_common = common + observer + layout + ['--base-replay', tests / 'base_replay.py',
                                                      '--layout-adapter', tests / 'materialization_replay.py']
        result = run('original-public-ten', tests / 'replay_original_ten.py', object_common + [
            '--observations', references / 'torch-gradient-objects.json', '--output', out / 'original-public-ten'],
            out / 'original-public-ten/report.json')
        assert (result['qualified_groups'], result['partial_groups'], result['failed_groups']) == (9, 1, 0)
        result = run('incoming-25', tests / 'materialization_replay.py', common + observer + layout + [
            '--base-replay', tests / 'base_replay.py', '--observations', references / 'torch-gradient-materialization.json',
            '--output', out / 'incoming-25'], out / 'incoming-25/report.json')
        assert (result['qualified_variants'], result['unrepresented_variants'], result['failed_variants']) == (22, 3, 0)
        result = run('alias-edge-12', tests / 'replay_alias_edges.py', object_common + [
            '--observations', references / 'torch-gradient-alias-edge.json', '--output', out / 'alias-edge-12'],
            out / 'alias-edge-12/report.json')
        assert result['qualified_variants'] == 12 and result['failed_variants'] == 0
        result = run('bernoulli-gradient-layout-8', tests / 'replay_bernoulli_gradient_layout.py', object_common + [
            '--observations', references / 'torch-bernoulli-gradient-layout.json',
            '--observations-sha256', '07bc446c10d37f942b1c3a6917b869874100c4d9ebc71622245f3505198f1eda',
            '--output', out / 'bernoulli-gradient-layout-8'], out / 'bernoulli-gradient-layout-8/report.json')
        assert result['qualified'] == 8 and result['failed'] == result['mismatch_fields'] == 0
        result = run('native-preserve-storage-4', tests / 'test_preserve_storage_lifetimes.py', [
            '--binary', binary, '--source', source, '--controls', tests / 'native_release_controls.py',
            '--observer', tests / 'client.py', '--expected-native-sha256', profile['native_sha256'],
            '--expected-source-sha256', profile['source_sha256'],
            '--state-format', profile['state_format'], '--backing-format', profile['backing_format'],
            '--output', out / 'native-preserve-storage-4'], out / 'native-preserve-storage-4/report.json')
        assert result['cases'] == result['passed'] == 4
        result = run('native-private-lifetimes-5', adapters / 'test_native_lifetimes.py', common + observer + [
            '--output', out / 'native-private-lifetimes-5'], out / 'native-private-lifetimes-5/report.json')
        assert result['cases'] == result['passed'] == 5
        result = run('typed-gradient-references', adapters / 'test_gradient_references.py', common + observer + [
            '--output', out / 'typed-gradient-references'], out / 'typed-gradient-references/report.json')
        assert result['status'] == 'PASS' and result['refusal_cases'] == 12
        inherited = common + ['--transport-observer', tests / 'client.py']
        for label, script, filename in (
            ('signed-zero-13', 'replay_signed_zero.py', 'torch-gradient-signed-zero.json'),
            ('partial-16', 'replay_partial.py', 'torch-backward-partial.json'),
            ('presence-8', 'replay_presence.py', 'torch-gradient-presence.json'),
            ('gradient-clear-10', 'replay_grad_clear.py', 'torch-grad-clear.json')):
            result = run(label, (adapters / script) if script == 'replay_grad_clear.py' else (tests / script), inherited + [
                '--observations', references / filename, '--output', out / label], out / label / 'report.json')
            if label == 'signed-zero-13':
                assert (result['cases'], result['passed'], result['failed']) == (13, 13, 0)
            elif label == 'partial-16':
                assert result['cases'] == result['passed'] == 16
            elif label == 'presence-8':
                assert result['parameter_values_versions_gradient_presence_and_skip_passed'] == 8
                assert result['unexpected_failures'] == 0
                assert (result['state_version_difference_observations'], result['state_creation_difference_observations']) == (8, 65)
                assert result['adam_compared_values_with_different_bits'] == 0
            else:
                assert result['qualified'] == 10 and result['unexpected_failures'] == 0
                assert (result['state_version_difference_observations'], result['state_creation_difference_observations'],
                        result['native_step_Tensor_absence_observations']) == (11, 9, 30)
                assert result['Adam_different_bits'] == 6
        bridge = dict(cold, runtime_source_sha256=cold['source_sha256'], executable_sha256=cold['native_sha256'])
        bridge_file = out / 'distribution-build-field-bridge.json'
        bridge_file.write_text(json.dumps(bridge, indent=2) + '\n')
        report['distribution_receipt_bridge'] = {'sha256': sha(bridge_file.read_bytes()),
            'only_added_fields': {'runtime_source_sha256': 'source_sha256', 'executable_sha256': 'native_sha256'},
            'original_cold_build_sha256': sha((native / 'build.json').read_bytes())}
        for label, script, counts in (
            ('distribution-atomic-183', 'replay_distribution_atomic.py', {'OPEN_TRANSACTION_DIFFERENCE': 4, 'OPEN_UNREPRESENTED': 18, 'PASS': 161}),
            ('distribution-explicit-183', 'replay_distribution_explicit.py', {'OPEN_UNREPRESENTED': 18, 'PASS': 165})):
            result = run(label, tests / script, ['--reference', references / 'torch-bernoulli-dropout.json',
                '--protocol', tests / 'distribution_protocol.py', '--executable', binary,
                '--build-report', bridge_file, '--output', out / label], out / label / 'report.json')
            assert result['counts'] == counts
        result = run('original-optimizer-24-mapping', tests / 'replay_optimizer_objects_v2.py', inherited + [
            '--observations', references / 'torch-optimizer-objects.json', '--suite', 'optimizer',
            '--output', out / 'original-optimizer-24-mapping'], out / 'original-optimizer-24-mapping/report.json')
        assert (result['qualified_exposed_scope'], result['partial_cases'], result['open_not_executed_cases'],
                result['unexpected_failures']) == (18, 1, 5, 0)
        result = run('owned-adam-cases21-24', tests / 'replay_adam_objects.py', inherited + [
            '--observations', references / 'torch-optimizer-objects.json',
            '--legacy-replayer', tests / 'replay_optimizer_objects_v2.py',
            '--output', out / 'owned-adam-cases21-24'], out / 'owned-adam-cases21-24/report.json')
        assert result['qualified_exposed_scope'] == 4 and result['failures'] == 0
        result = run('optimizer-gradient-case20', tests / 'replay_optimizer_gradient_alias.py', object_common + [
            '--observations', references / 'torch-optimizer-objects.json', '--output', out / 'optimizer-gradient-case20'],
            out / 'optimizer-gradient-case20/report.json')
        assert result['status'] == 'QUALIFIED_EXPOSED_SCOPE' and result['mismatch_fields'] == 0
        custody()
        report.update(status='PASS',step_count=len(report['steps']),all_sources_unchanged=True,
            native_build_reproduced_exactly=False, existing_native_build_receipt_verified=True,
            inherited_optimizer_mapping={'reference_cases':24,'qualified_cases':23,'partial_cases':1,
                'remaining_in_this_frozen_mapping':'retain_grad was not invoked by the inherited runner; the new additive actual12/row14 qualification is separate.'})
        save();print(json.dumps({k:v for k,v in report.items() if k not in ('steps','suite_reports','own_inputs_sha256')},indent=2));return 0
    except Exception as error:
        report.update(status='FAIL',error=repr(error));save();raise
if __name__=='__main__':raise SystemExit(main())
