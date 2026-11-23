"""Retain the original spatial forward suite on the exact joint opt-in executable."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import types

BASE = '083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5'
DUAL = '5af793857b1606f4760a2cae1d5ce9c5e8f821e07fb499ac0349567f2a503b72'
BASE_SOURCE = 'aab2e736ff0ef4a282ff73fb1a69a95d2ceeea5260bbdada0af5b30cc16dcd3c'
DUAL_SOURCE = 'f4c43f037a3fb046425f7634215ec382aa5f8d6b0e7abe126ef64236adb5753a'
CHANGES = {'wide_neg_tensor', 'entry', 'predicate_command', 'predicate_bits'}
SUITES = {
    'spatial_forward': dict(cases=344, sessions=347, controls=28, recoveries=3, autograd=186, mappings=24),
}
READER = 'tools/current_runtime/qualification.py'
ADMIT = 'tools/current_runtime/opt_in_admission/admit.py'
OLD = b'built=profile('
NEW = b'built=joint_profile('
PINS = {
    'compatibility/pytorch-official-wheel-members.json': '0e5377c5bd2767e35bfbcb81346dbc522ad9864e0d2d68b6a3705e0befd042f0',
    'compatibility/pytorch-oracle-lock.json': 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944',
    'compatibility/pytorch-reference-requirements.txt': 'cd49cdce73d64b096d08b346f171d13c29fd1c40b40e3845383b05c9ac4a0b89',
    'compatibility/upstream-lock.json': 'd204c5a8268a0d0f0a1dd162b5e0ab184beee83310bcad643b084deb691f3f49',
    'tests/current_runtime_process.py': '36e3432deac48085ba0586c85547cee8432d1533c5915238db2439b43dbca523',
    'tests/differentiable_view_native_review/owner_client.py': 'dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b',
    'tests/differentiable_view_native_review/protocol.py': '7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9',
    'tests/half_logical/__init__.py': '4c5fbdf0c7f8da54986ba8c0aef475abc92284941bcf215c0545a3528dc8b8fb',
    'tests/half_logical/native.py': '9c2cd19a5bb04eaa037f5905a52bce2dcc31042101f72d07e71205745160a14f',
    'tests/half_logical/reference.py': '5bb6a3804717968dba37f3174809a7c1eee5b225f0a3085b44fcec7656155528',
    'tests/half_logical/run.py': 'f358bc08f4d35ad48c16ebd024f2341fb4bde8a8f58cab2746635912a0fc7199',
    'tests/indexed_f32/__init__.py': '4f0decb788ca2416d9ade7ef53ea8e7ad1d197e7dc948aa440eb41213de52c22',
    'tests/indexed_f32/cases.py': '33df926d14fc236168aca8df417089661cdb1ee387478a18882351e3076634a3',
    'tests/indexed_f32/native.py': 'fb9f14b99866ff58ec89a1424b612b7745c44e0fc61871ea689cbf5c1e5acc50',
    'tests/indexed_f32/reference.py': '8f670fe2358d20430007b60c78a4263bec51039524a06ef28dd187dc0f60df7a',
    'tests/indexed_f32/run.py': 'cbb61de9f5fc53fb2e6afc868ca56afc0c64f6477f00419de04afa9cde9c44ea',
    'tests/numeric_predicates/__init__.py': '7d9b42b2f3b86a533773fdb030f2cb539632f68ffdb22874a5071d0a4cc1b64f',
    'tests/numeric_predicates/cases.py': 'e69bebed091db88ef81eda2ea2052af6b2d88c74840909ee27e2cbdf6a9ec694',
    'tests/numeric_predicates/evidence.py': 'a3d39ee66d5511d49bb484bc9b95a131504085679546d25266b281e26b879c84',
    'tests/numeric_predicates/native.py': 'c0e9d1ba6f0dac5499f091333611a5bea0fb6f4a25e74faa39f35486fd404d2d',
    'tests/numeric_predicates/reference.py': '3ec85934110f2d8f273bc1e209a9c9ce13331cc4ff915dc9d7058d81913d0abd',
    'tests/numeric_predicates/run.py': 'e9fe4a7361cef25aca14c24ecd17c5413f058e5ccb3c35190d66b0be3b479991',
    'tests/pinned_reference_custody.py': 'fa91a644b8743493ed6ed8f29da966494983cf5eed7300faa715ee4e08c7212e',
    'tests/pytorch_oracle.py': 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae',
    'tests/spatial_forward/__init__.py': 'e000a036520089d7ba7ca01717c8565df14004519ad81abe9bfc8c850aaf41e8',
    'tests/spatial_forward/cases.py': 'f6dc5c14434ad05415058f51004d917df4055b317f1a032f6dfb18f2040a2316',
    'tests/spatial_forward/native.py': '71f8cbfe758b256d76fb947ef0358fe0b57222e0e6d66be3ba3970348e015b8d',
    'tests/spatial_forward/reference.py': '39bb705aea70b711de671744d1b8acb4e75a5f4e279d25521f9abd36e0ca27bd',
    'tests/spatial_forward/run.py': 'ab1fe3a834e7e241f0c859fa1dcc126b4f8444e37b14619cd5076008edf98591',
    'tools/current_runtime/build.py': '9c93f8ff3fcfda8358e51a4c12efb3b0e0962abed9b5a750ca32d5a5568ac14d',
    'tools/current_runtime/forward_source_delta.json': '5d3e68d50ebdf26b4132aaf1536cdb9a4201982b1f37dfd83ff92f3ecf9a7f74',
    'tools/current_runtime/forward_source_transition.py': 'c8fca08e2c2756873c331cef92003f94296e36cbe2e52014bfd8bc6942d7a1b6',
    'tools/current_runtime/index_signbit_source_delta.json': '2f6d013eb9ca2a77dbe399bfbc14d9f417230fd89747bba73c0e034bcae7a8f1',
    'tools/current_runtime/index_signbit_source_transition.py': 'c1761c7958d0a536d9c61f3b181a1d59d633fc3c21583b295cd37d35cb88bfee',
    'tools/current_runtime/indexed_forward_source_transition.py': 'bfac3860d7b9c7d6cff9c0f2f5d70b51addfeb2b531160c6682a4f94f0556200',
    'tools/current_runtime/opt_in_admission/admit.py': '51b32b1835fe8a294f1f42b98ca3b87b10a5096dec08d34840ac7acab61f672d',
    'tools/current_runtime/public_f64_view_source_delta.json': '36f3da06db594870b93354feae99a65de8762cf0a90d078190b98ae2883606cf',
    'tools/current_runtime/public_f64_view_source_transition.py': 'b0f73d2732f3045ef00f106ab02b379f83afdb7a45feb6e054bcdf2655554b5d',
    'tools/current_runtime/qualification.py': 'f4be6d41c29659c3132f1f64014ad9d8ebcf612db3a9cb556ee5ddd178a540b6',
    'tools/current_runtime/source/manifest.json': 'a5cf09f191db4887b4452001e0b72e75fffb2147f2476544cc3be49d8ffa2c85',
    'tools/current_runtime/source/runtime.hepta': 'aab2e736ff0ef4a282ff73fb1a69a95d2ceeea5260bbdada0af5b30cc16dcd3c',
    'tools/current_runtime/source/units/owners.hepta': '867a27b3a86634ea6a873909eb8bc16d5d8027c98c7cb9340376fba51312ec86',
    'tools/current_runtime/source/units/rng03.hepta': '7c32981c916bc277e16221cce1c3dd2381ce0018642e21b20836d861cf585865',
    'tools/current_runtime/source/units/runtime800_00.hepta': 'fc16ea8e128c192f7c1d353d636bf73ba67c574039ecdbf2733d59a3ef24b13d',
    'tools/current_runtime/source/units/runtime800_01.hepta': '26a2c3be048f07b1310500d52cdb5983e1ddf7626ba52bfae091e66f3aa33c2c',
    'tools/current_runtime/source/units/runtime800_02.hepta': 'e857cbe7bdf4dfed74a8a626a9f9e5e7ca1ec4b7642bd6b31f5a7fe494d20e0d',
    'tools/current_runtime/source/units/runtime800_03.hepta': 'd917391a32cd4cdd633c59733d0f10eb87bc410189588b7aaa4ecc124d5503e6',
    'tools/current_runtime/source/units/runtime800_04.hepta': '8aedec9bd90a5cd3f49abf515f02d61aeb6f4404c91de0322a72126e72759caa',
    'tools/current_runtime/source/units/runtime800_05.hepta': '2fbcbac61359d8ea4d222771e799f7e7ae8d6cc7f72ce65141bf05356a8cc5b0',
    'tools/current_runtime/source/units/runtime800_06.hepta': 'e94651b09a0ae9abc7c7b871bb2c965d0b8c1fe4b7cc38a4227a50f1b83c6d48',
    'tools/current_runtime/source/units/runtime800_07.hepta': '114712800106ef1f7d015f3c1e462383ab1fbe7a9a562d6b36c39cc347255d97',
    'tools/current_runtime/source/units/runtime800_08.hepta': '366db82fb76605b56f19332df25ba099652fd9606680168a554721d4a35b28c5',
    'tools/current_runtime/source/units/runtime800_09.hepta': '446910b0e2bb22477ddd4fb8548945fae5f48cd545c8eebe114729b1423000f6',
    'tools/current_runtime/source/units/runtime800_10.hepta': '945318f0195661e0015e48d8bf11d9835f0a71836c85506d2c1ee2585801b48f',
    'tools/current_runtime/source/units/runtime800_11.hepta': 'bc94220020d7b7c63f9e79abaf46a79cf735c4d01e09e5c9f892f45958226408',
    'tools/current_runtime/source/units/runtime800_12.hepta': '56969135c361828fdc3433dffc91066a2e9d5b1d2e94060b6dd608311cdda759',
    'tools/stack_budget/inspect_stack.py': '71b72260014417db72ffcebd80c86543e257c80d76cdd6c878b41feaa117cf92',
}


def need(predicate, message):
    if not predicate:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    first = path.lstat()
    need(stat.S_ISREG(first.st_mode) and path == path.resolve(), 'ordinary input required: '+str(path))
    raw = path.read_bytes()
    last = path.lstat()
    def key(s):
        return s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_ctime_ns, s.st_mtime_ns
    need(key(first) == key(last) and len(raw) == last.st_size, 'input changed: '+str(path))
    return raw


def module(name, path, code, **extra):
    obj = types.ModuleType(name)
    obj.__file__ = str(path)
    obj.__dict__.update(extra)
    exec(compile(code, str(path), 'exec', dont_inherit=True, optimize=0), obj.__dict__)
    return obj


def exact_inverse(before, after):
    def functions(raw):
        matches = list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n', raw, re.M | re.S))
        bodies = {m[1].decode('ascii'): m[0] for m in matches}
        header = raw.split(b'\n', 1)[0] + b'\n'
        need(len(matches) == len(bodies) == 800 and header + b''.join(bodies.values()) == raw,
             'noncanonical function list')
        return header, bodies
    header, old = functions(before)
    new_header, new = functions(after)
    need(header == new_header and tuple(old) == tuple(new), 'definition order or envelope changed')
    changes = {n for n in old if old[n] != new[n]}
    need(changes == CHANGES, 'unexpected function body changes')
    need(header+b''.join(old[n] if n in changes else new[n] for n in new) == before
         and header+b''.join(new[n] if n in changes else old[n] for n in old) == after,
         'complete joint source inverse did not round trip')
    return sorted(changes)


def run(a, result):
    repo, base, dual = a.repository, a.baseline_build, a.dual_build
    contents = {p: read(repo/p) for p in PINS}
    need(all(sha(raw) == PINS[p] for p, raw in contents.items()), 'exact observer/source inputs changed')
    manifest = repo/'tools/current_runtime/source/manifest.json'
    admit = module('joint_fixed_admission', repo/ADMIT, contents[ADMIT]).admit
    canonical = admit('baseline', manifest, base, {})
    selected = admit('dual', manifest, dual, {'dual_scoped': a.dual_scoped})
    need(canonical['native_image_sha256'] == BASE and selected['native_image_sha256'] == DUAL,
         'baseline or dual image mismatched')
    before, after = map(read, (repo/'tools/current_runtime/source/runtime.hepta', dual/'runtime.hepta'))
    need(sha(before) == BASE_SOURCE and sha(after) == DUAL_SOURCE, 'exact source pair mismatched')
    result.update(candidate_image_sha256=DUAL, baseline_image_sha256=BASE,
                  source_changes=exact_inverse(before,after), complete_source_inverse=True,
                  canonical_build_report_sha256=canonical['build_report_sha256'],
                  candidate_build_report_sha256=selected['build_report_sha256'], admission_checks=[])
    runner_path = f'tests/{a.suite}/run.py'
    expected = SUITES[a.suite]
    raw = contents[runner_path]
    need(raw.count(OLD)==2 and NEW not in raw, 'unexpected original profile call sites')
    derived = raw.replace(OLD, NEW)
    need(derived.replace(NEW, OLD)==raw,'observer edit did not invert')
    legacy = module('joint_unmodified_reader', repo/READER, contents[READER])

    def joint_profile(repository, build, build_sha, inputs):
        need(repository == repo and build == dual and build_sha == selected['build_report_sha256'],
             'unexpected selected build inputs')
        try:
            legacy.admit_build(build, inputs.bind, build_sha)
        except ValueError as error:
            need(str(error) == 'build used another source manifest', 'unexpected original refusal: '+str(error))
        else:
            raise ValueError('unmodified reader unexpectedly accepted the joint image')
        # Exact canonical baseline satisfies the complete original historical proof.
        # That proof applies to source reconstruction, not any joint native results.
        old = observer.profile(repo, base, canonical['build_report_sha256'], inputs)
        need(old['source_sha256']==BASE_SOURCE and old['native_sha256']==BASE,'wrong original proof target')
        need(admit('dual', manifest, dual, {'dual_scoped':a.dual_scoped})==selected,
             'joint native build changed after original proof')
        inputs.bind(dual/'heptatorch',DUAL)
        inputs.bind(dual/'runtime.hepta',DUAL_SOURCE)
        result['admission_checks'].append('original canonical admission denied joint; full baseline source proof passed')
        return json.loads(inputs.bind(dual/'build.json', build_sha))

    cwd, argv = Path.cwd(), sys.argv[:]
    old_cache = sys.pycache_prefix
    old_env = os.environ.get('PYTHONPYCACHEPREFIX')
    cache = str(a.output/'clean-bytecode-cache')
    # Neither host nor child may read cached original sources.
    sys.pycache_prefix = cache
    os.environ['PYTHONPYCACHEPREFIX'] = cache
    try:
        os.chdir(repo)
        observer = module('joint_spatial_observer_run', repo/runner_path, derived, joint_profile=joint_profile)
        out = a.output/'observer'
        sys.argv = [str(repo/runner_path),'--build-directory',str(dual),'--expected-build-sha256',
                    selected['build_report_sha256'],'--torch-python',str(a.torch_python),'--output',str(out)]
        code = observer.main()
    finally:
        os.chdir(cwd)
        sys.argv = argv
        sys.pycache_prefix = old_cache
        if old_env is None:
            os.environ.pop('PYTHONPYCACHEPREFIX',None)
        else:
            os.environ['PYTHONPYCACHEPREFIX'] = old_env

    report_raw = read(out/'report.json')
    report = json.loads(report_raw)
    result.update(observer_exit_code=code, observer_status=report['status'],
                  observer_error=report.get('error'), case_failures=report['failures'],
                  native_executed=report['native_executed'], torch_executed=report['Torch_executed'],
                  actual_observer_report_sha256=sha(report_raw), cases=report['cases'],
                  frames=report['frames'], checks=report['checks'])
    need(all(read(repo/p)==contents[p] for p in contents), 'source custody changed during execution')
    need(admit('dual',manifest,dual,{'dual_scoped':a.dual_scoped})==selected,
         'native custody changed during execution')
    if code != 0 or report['status'] != 'PASS':
        if (out/'physical-qualification.json').exists():
            result['physical_failure'] = json.loads(read(out/'physical-qualification.json'))
        return
    proof_raw = read(out/'physical-qualification.json')
    proof = json.loads(proof_raw)
    need(report['native_executed'] is True and report['Torch_executed'] is True
         and report['native_sha256']==DUAL and report['source_sha256']==DUAL_SOURCE
         and report['actual_build_sha256']==selected['build_report_sha256']
         and report['all_inputs_unchanged'] is True and not report['failures']
         and report['cases']==proof['cases']==expected['cases'] and len(result['admission_checks'])==2
         and proof['complete_ci_admission'] is True and proof['production_oracle_qualified'] is True
         and proof['full_installation_provenance_verified'] is True
         and proof['complete_official_wheel_archive_verified'] is True
         and proof['complete_installed_raw_snapshot_verified'] is True
         and proof['actual_mapped_torch_cpu_code_verified'] is True
         and proof['mapping_snapshots']==expected['mappings']
         and proof['actual_autograd_cases']==expected['autograd']
         and proof['actual_gradient_tensors']==258
         and proof['physical_replay'] is True and proof['frames']==report['frames']
         and proof['assertions']==report['checks'] and proof['native_sessions']==len(report['results'])==expected['sessions']
         and proof['additional_protocol_controls']==expected['controls']
         and proof['additional_allocation_recoveries']==expected['recoveries'],
         'the original independent physical replay did not qualify all mandatory predicates')
    result.update(status='PASS_RETAINED_SPATIAL_FORWARD_ON_DUAL',
                  physical_report_sha256=sha(proof_raw), independent_physical_readback=proof)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repository','baseline-build','dual-build','dual-scoped','torch-python','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--suite',choices=SUITES,required=True)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize, 'run with python3 -I -B')
    for key,value in vars(a).items():
        if key=='suite':
            continue
        need(key=='torch_python' or not value.is_symlink(), 'symlinked input: '+str(value))
        setattr(a,key,value.absolute() if key=='torch_python' else value.resolve())
    a.output.mkdir()
    result={'schema':'heptatorch.opt-in.spatial-observer-replay.v1',
            'retained_suite':a.suite,
            'status':'FAIL_RETAINED_SPATIAL_FORWARD_ON_DUAL',
            'canonical_qualification_transferred':False,'production_activated':False,
            'full_candidate_suite_qualification':False,'source_pins':PINS}
    try:
        run(a,result)
    except (OSError,ValueError,AssertionError,KeyError,TypeError) as e:
        result['failure']=dict(type=type(e).__name__,message=str(e))
    with (a.output/'receipt.json').open('x') as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n')
    print(result['status'], flush=True)
    sys.exit(0 if result['status'].startswith('PASS_') else 1)
