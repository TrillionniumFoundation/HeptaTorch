"""Qualify the exact common HIA Tensor/RNG/backward image and retained gaps."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

CORE = '7447c169d3b2cad1e55cb7b1b8970d67f4103c5866f6db0a60c0b4d05d02cc0b'
IMAGE = 'cdcaebb2785639585de7fd2a4e1f98d41a56e885f1657c3485ce5f187f89d726'
MANIFESTS = {
    'tensor-rng-package.sha256': ('d971907c9053c05b177a6ac51303a7602da384f2564cfdea7e0ab508fd51a6ca', 38),
    'tensor-rng-composition-package.sha256': ('3050c8205dce52ed707569c47afa3166b8c70df73a92e6ea448e087fc15f6c70', 38),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'tensor09', 'release', 'materialization', 'rng-build', 'stack-build', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    build = args.build.resolve(); native = build/'native'; out = args.output.resolve()
    package, rng, partial = (repo/'tools'/name for name in ('tensor_rng_composition','tensor_rng','backward_partial'))
    image, core = native/'tensor-rng-session', native/'runtime.hepta'
    assert sha(image) == IMAGE and sha(core) == CORE
    inputs = [Path(__file__), image, core, native/'build.json', build/'composition.json', build/'reproduction.json']
    for name, (pin, count) in MANIFESTS.items():
        manifest = repo/'validation'/name; assert sha(manifest) == pin; inputs.append(manifest)
        lines = manifest.read_text().splitlines(); assert len(lines) == count
        for line in lines:
            digest, relative = line.split('  ',1); path = (repo/relative).resolve()
            assert path.is_relative_to(repo) and sha(path) == digest, relative
            inputs.append(path)
    partial_manifest = repo/'validation/backward-partial-package.json'
    assert sha(partial_manifest) == '33b7677b91a3615149c86322b5c4e624749e682e7331061367cfe35fba49e340'
    inputs.append(partial_manifest)
    for relative, pin in json.loads(partial_manifest.read_bytes())['files'].items():
        path = repo/relative; assert sha(path) == pin['sha256']; inputs.append(path)
    composition = json.loads((build/'composition.json').read_bytes())
    assert composition['entry_route_inverse_ast_identical'] is True
    assert composition['runtime_source_sha256'] == CORE and composition['executable_sha256'] == IMAGE
    assert composition['unchanged_parent_functions'] == 237
    peers = {'tensor09':args.tensor09.resolve(), 'release04':args.release.resolve(),
             'materialization05':args.materialization.resolve(), 'rng03':args.rng_build.resolve()}
    peer_pins = json.loads((package/'handoff-peer-pins.json').read_bytes())
    for label, directory in peers.items():
        assert sha(directory/'runtime.hepta') == peer_pins['peers'][label]['source_sha256']
        inputs.append(directory/'runtime.hepta')
        for unit in peer_pins['peers'][label]['units']:
            path = directory/(unit['name']+'.hepta'); assert sha(path) == unit['source_sha256']; inputs.append(path)
    observer = repo/'tools/owned_sessions/client.py'; inputs.append(observer)
    for name in ('stack-module-compiler','stack-linker-session'):
        inputs.append(args.stack_build.resolve()/'native'/name)
    before = {str(path):sha(path) for path in inputs}
    out.mkdir(parents=True,exist_ok=False)
    reports = {}

    def execute(name, script, *arguments, report='report.json'):
        command = [sys.executable,'-B',str(script),*map(str,arguments),'--output',str(out/name)]
        result = subprocess.run(command,cwd=repo,capture_output=True,timeout=300)
        (out/(name+'.stdout')).write_bytes(result.stdout); (out/(name+'.stderr')).write_bytes(result.stderr)
        assert result.returncode == 0 and not result.stderr, (name,result.returncode,result.stderr[-3000:])
        reports[name] = json.loads((out/name/report).read_bytes())

    execute('rng-adaptation',package/'replay_rng.py','--rng-package',rng,'--build',native,report='adaptation.json')
    rng_report_path = out/'rng-adaptation/native/report.json'
    reports['rng-contracts'] = json.loads(rng_report_path.read_bytes())
    cpp_reference = out/'rng-adaptation/native/std-mt-reference'
    assert cpp_reference.is_file()
    common = ['--executable',image,'--rng-package',rng,'--cpp-reference',cpp_reference]
    execute('joint',package/'test_joint.py',*common)
    execute('range',package/'test_range.py',*common,'--torch-observation',package/'evidence/pytorch-uniform-range-probe.json')
    execute('entry',rng/'test_entry.py','--build',native,'--stack-build',args.stack_build.resolve(),
            '--cpp-reference',cpp_reference)
    for opcode in (57,58):
        execute('training'+str(opcode),rng/'test_composition.py','--executable',image,
                '--cpp-reference',cpp_reference,'--backward-code',opcode,'--presence')
    execute('handoff',package/'test_profile_handoffs.py','--build',native,'--rng-package',rng,
            '--stack-build',args.stack_build.resolve(),'--peer-pins',package/'handoff-peer-pins.json',
            *[value for label,path in peers.items() for value in ('--peer',label+'='+str(path))])
    replay = ['--binary',image,'--source',core,'--expected-sha256',IMAGE,'--expected-source-sha256',CORE,
              '--controls',partial/'native_release_controls.py','--transport-observer',observer]
    for name,script,observations in [
        ('signedzero','replay_gradient_signed_zero.py','actual-torch-signed-zero-observations.json'),
        ('partial','replay_partial_observations.py','actual-torch-16-observations.json'),
        ('presence','replay_gradient_presence.py','actual-torch-gradient-presence-observations.json'),
    ]:
        execute(name,partial/script,*replay,'--observations',partial/'evidence'/observations)

    adapted = reports['rng-adaptation']
    frozen_adaptation = json.loads((package/'evidence/rng-replay-adaptation-05.json').read_bytes())
    assert adapted == frozen_adaptation
    original = json.loads((package/'evidence/original-rn03-contract-failure.json').read_bytes())
    assert original['status'] == 'FAIL' and original['executable_sha256'] == IMAGE
    r = reports['rng-contracts']
    assert r['status'] == 'PASS' and len(r['cases']) == 8 and all(x['status']=='PASS' for x in r['cases'])
    assert (r['frames'],r['exact_values'],r['full_mt_states'],r['whole_owner_rollbacks']) == (729,33193,160,42)
    joint = reports['joint']
    assert joint['status'] == 'PASS' and joint['case_count'] == 3 and joint['frames'] == 155
    assert (joint['full_generator_states'],joint['joint_owner_rollbacks'],joint['stage_zero_observations']) == (14,5,42)
    ranges = reports['range']
    assert ranges['status'] == 'PASS' and ranges['cases'] == 88 and ranges['frames'] == 352 and ranges['whole_owner_rollbacks'] == 28
    assert ranges['actual_fixed_torch_observation_compared'] is True
    assert ranges['actual_observation_sha256'] == '467e2de683758881175bbfb82446a9d174fac91feb686baea7f9cbe40de1d70a'
    assert all(row['actual_fixed_torch_exact_match'] is True for row in ranges['rows'])
    entry = reports['entry']
    assert entry['status'] == 'PASS' and (entry['cases'],entry['probe_processes'],entry['live_frames']) == (88,84,20)
    for opcode in (57,58):
        r = reports['training'+str(opcode)]
        assert r['status'] == 'PASS' and r['backward_code'] == opcode and r['gradient_presence_checked'] is True
        assert (r['frames'],r['updates'],r['exact_values'],r['full_generator_states']) == (14,12,192,12)
    handoff = reports['handoff']
    assert handoff['status'] == 'PASS' and (handoff['cases'],handoff['frames']) == (8,33)
    assert handoff['current_source_sha256'] == CORE and handoff['current_executable_sha256'] == IMAGE
    assert handoff['all_source_custody_unchanged'] and all(x['inverse_byte_identical'] for x in handoff['source_bindings'])
    historical_handoff = json.loads((package/'evidence/four-predecessor-handoff-05.json').read_bytes())
    actual_images = {row['name']:row['qualification_executable_sha256'] for row in handoff['peers']}
    assert actual_images == {row['name']:row['qualification_executable_sha256'] for row in historical_handoff['peers']}
    zero, partial_report, presence = (reports[name] for name in ('signedzero','partial','presence'))
    assert zero['status'] == 'PASS' and zero['cases'] == zero['passed'] == 13 and zero['failed'] == 0
    assert (zero['frames'],zero['exact_binary32_word_comparisons'],zero['different_word_observations']) == (166,70,0)
    assert partial_report['status'] == 'PASS' and partial_report['cases'] == partial_report['passed'] == 16
    assert (partial_report['frames'],partial_report['exact_scalar_or_binary32_comparisons']) == (606,1587)
    assert presence['status'] == 'QUALIFIED_WITH_OPEN_STATE_LIFECYCLE'
    assert presence['cases'] == presence['parameter_values_versions_gradient_presence_and_skip_passed'] == 8
    assert presence['unexpected_failures'] == 0
    assert (presence['frames'],presence['exact_scalar_or_binary32_comparisons'],presence['adam_tolerance_comparisons'],presence['adam_compared_values_with_different_bits']) == (880,2088,180,0)
    assert (presence['state_version_difference_observations'],presence['state_creation_difference_observations']) == (8,65)
    for r in (zero,partial_report,presence):
        assert r['native_executable_sha256'] == IMAGE and r['native_source_sha256'] == CORE
        p = r['oracle_provenance']; assert p['status']=='PASS' and len(p['checks'])==11 and all(x is True for x in p['checks'].values())
    after = {str(path):sha(path) for path in inputs}; assert before == after
    report = {'schema':'heptatorch.integrated-tensor-rng-backward-qualification.v1',
              'status':'QUALIFIED_WITH_OPEN_STATE_LIFECYCLE','joint_native_contracts':'PASS',
              'native_executable_sha256':IMAGE,'native_source_sha256':CORE,
              'source_custody_before':before,'source_custody_after':after,'reports':reports,
              'original_rn03_strict_range_failure_preserved':True,'actual88_qualifies_new_binary64_admission':True,
              'open_optimizer_state_version_observations':8,'open_optimizer_state_creation_observations':65,
              'pytorch_executed_in_this_process':False,'reference_is_actual_pinned_hosted_observation':True,
              'pytorch_api_denominator':2590,'whole_pytorch_rewrite_complete':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'joint_native_contracts':'PASS','report_sha256':sha(out/'report.json')}))


if __name__ == '__main__':
    main()
