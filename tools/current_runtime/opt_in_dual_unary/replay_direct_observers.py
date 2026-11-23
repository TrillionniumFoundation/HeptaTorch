#!/usr/bin/env python3
"""Replay two unchanged retained mandatory observers on the exact joint image."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import types

BASE = '083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5'
DUAL = '5af793857b1606f4760a2cae1d5ce9c5e8f821e07fb499ac0349567f2a503b72'
SUITES = {'typed_relu': 776, 'bf16_copy': 280}
ADMIT = 'tools/current_runtime/opt_in_admission/admit.py'
CANONICAL = 'tools/current_runtime/qualification.py'
OLD = b'qualification.admit_build(build,inputs.bind,'
NEW = b'admit_joint(qualification,build,inputs.bind,'
SOURCE_SHA256 = {
    "compatibility/pytorch-oracle-lock.json": "ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944",
    "compatibility/upstream-lock.json": "d204c5a8268a0d0f0a1dd162b5e0ab184beee83310bcad643b084deb691f3f49",
    "tests/bf16_copy/__init__.py": "7cb9113cc9fa7ce6569c7fc1ac7f31a52ea77703aa351cc41a55c266ecfee899",
    "tests/bf16_copy/cases.py": "bb6c161e31ca33af78a81a13143175a26a8abc229d5140f2e007b9872cc33095",
    "tests/bf16_copy/reference.py": "849adab42c0fa066f8fa2a9cafe4d83aca7b9c101e91abf65070fa3e4cf5309a",
    "tests/bf16_copy/run.py": "677f3fc718f9745ca3c6c1e6c17e695233801e84a7ed9f6395709169144d4cb9",
    "tests/current_runtime_process.py": "36e3432deac48085ba0586c85547cee8432d1533c5915238db2439b43dbca523",
    "tests/differentiable_view_native_review/native_observer_direct01.py": "d82c8335aa949466923ee059161b553606ebc2514a500adf420f238d4f24559d",
    "tests/differentiable_view_native_review/native_observer_direct02.py": "1e9c197867cab126ec6022e4316505288ab79e63cccc95fc9ecec02f8c884814",
    "tests/differentiable_view_native_review/owner_client.py": "dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b",
    "tests/differentiable_view_native_review/prepare_reference.py": "84360e70b09280edd3f2c0c326a8ff4df39e20e35973b4a73a98a776881f6b5b",
    "tests/differentiable_view_native_review/protocol.py": "7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9",
    "tests/differentiable_view_native_review/replay_direct_view14.py": "52c3b3242a73fb412102c83c5ba47cf8d95a4eb3c01e56f41bfe593066506d05",
    "tests/half_logical/__init__.py": "4c5fbdf0c7f8da54986ba8c0aef475abc92284941bcf215c0545a3528dc8b8fb",
    "tests/half_logical/native.py": "9c2cd19a5bb04eaa037f5905a52bce2dcc31042101f72d07e71205745160a14f",
    "tests/half_logical/reference.py": "5bb6a3804717968dba37f3174809a7c1eee5b225f0a3085b44fcec7656155528",
    "tests/half_logical/run.py": "f358bc08f4d35ad48c16ebd024f2341fb4bde8a8f58cab2746635912a0fc7199",
    "tests/half_neg/__init__.py": "f02584b81106f7440ca54bf8f59beb9a3e05feb8c1123964325d5e946b839163",
    "tests/half_neg/cases.py": "643da7cd50f0b9364444bac11a091b99b1fd6670b91979dca40e3227ee07cd69",
    "tests/half_neg/reference.py": "b6f7ee4cd1ec107800abb650ce0aabbd63fd7edbbbb642a2290c5ae4d178a2bf",
    "tests/half_neg/run.py": "f36fe06d82b539d5f4d3324a91ce2d0d9ab6a9d2033892e274d8c1377c53bec2",
    "tests/integer_tensor_review/wire/owner_client.py": "dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b",
    "tests/integer_tensor_review/wire/protocol.py": "7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9",
    "tests/numeric_predicates/__init__.py": "7d9b42b2f3b86a533773fdb030f2cb539632f68ffdb22874a5071d0a4cc1b64f",
    "tests/numeric_predicates/cases.py": "e69bebed091db88ef81eda2ea2052af6b2d88c74840909ee27e2cbdf6a9ec694",
    "tests/numeric_predicates/evidence.py": "a3d39ee66d5511d49bb484bc9b95a131504085679546d25266b281e26b879c84",
    "tests/numeric_predicates/native.py": "c0e9d1ba6f0dac5499f091333611a5bea0fb6f4a25e74faa39f35486fd404d2d",
    "tests/numeric_predicates/reference.py": "3ec85934110f2d8f273bc1e209a9c9ce13331cc4ff915dc9d7058d81913d0abd",
    "tests/numeric_predicates/run.py": "e9fe4a7361cef25aca14c24ecd17c5413f058e5ccb3c35190d66b0be3b479991",
    "tests/pytorch_oracle.py": "ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae",
    "tests/typed_relu/__init__.py": "838cea508ffc398ae7319d22d3666cea7a0a06050b24fc67aa025d1d43d2a481",
    "tests/typed_relu/cases.py": "385968517dbd6f5ecd5606384fb8b8920544e0195fc4bf349dc0df76524538ac",
    "tests/typed_relu/reference.py": "f0ecb1e92449743714c24d7ef50ede74477075c3298fbde3426e397617ef4b26",
    "tests/typed_relu/run.py": "4bcaad8ebe594c7e09985159f0f13c5ab3f18fbc43cdb0b03c9a5dde08789edf",
    "tests/wide_neg/__init__.py": "01ea777820ede90061e80091e3f765f6eac9844d2c43d60dab95dcb3c48a2ba6",
    "tests/wide_neg/cases.py": "5796a0561637004f3cd5a5ac16cc0612e0ed51bd0c381c0440c2ced666ec0b84",
    "tests/wide_neg/reference.py": "b40d811d0d3a030f45a8d086b6cf7df1e549cb1be778a6f7ea430346ec342ba3",
    "tests/wide_neg/run.py": "2ea0b003bdc371446da8efee479fb1b501808410fe9e48de8232babcbd4ad35f",
    "tools/current_runtime/opt_in_admission/admit.py": "51b32b1835fe8a294f1f42b98ca3b87b10a5096dec08d34840ac7acab61f672d",
    "tools/current_runtime/qualification.py": "f4be6d41c29659c3132f1f64014ad9d8ebcf612db3a9cb556ee5ddd178a540b6"
}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and path.resolve() == path, 'not ordinary input: ' + str(path))
    data = path.read_bytes()
    after = path.lstat()
    fields = lambda s: (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(fields(before) == fields(after) and len(data) == after.st_size, 'input changed: ' + str(path))
    return data


def load(name, path, data, **extra):
    result = types.ModuleType(name)
    result.__file__ = str(path)
    result.__dict__.update(extra)
    exec(compile(data, str(path), 'exec', dont_inherit=True, optimize=0), result.__dict__)
    return result


def run(args, result):
    repo, build, baseline, scoped = args.repository, args.dual_build, args.baseline_build, args.dual_scoped
    runner_path = f'tests/{args.suite}/run.py'
    sources = {p: read(repo / p) for p in SOURCE_SHA256}
    need(all(sha(data) == SOURCE_SHA256[p] for p,data in sources.items()), 'original observer or admission changed')
    for folder in {(repo / p).parent for p in sources if p.endswith('.py')} | {repo/'tests'}:
        need(not (folder/'__pycache__').exists(), 'use clean source directories: ' + str(folder))
    canonical_manifest = repo/'tools/current_runtime/source/manifest.json'
    admit = load('fixed_dual_admission', repo/ADMIT, sources[ADMIT]).admit
    legacy = load('fixed_legacy_admission', repo/CANONICAL, sources[CANONICAL])
    old_build = admit('baseline', canonical_manifest, baseline, {})
    legacy_record = legacy.admit_build(baseline, lambda path, pin=None: bind(path, pin), old_build['build_report_sha256'])
    need(legacy_record['native_sha256'] == old_build['native_image_sha256'] == BASE, 'old reader did not admit exact baseline')
    selected = admit('dual', canonical_manifest, build, {'dual_scoped':scoped})
    need(selected['native_image_sha256'] == DUAL, 'unexpected candidate image')
    result.update(candidate_image_sha256=DUAL, baseline_image_sha256=BASE,
                  dual_admission=selected, legacy_denials=[])
    raw = sources[runner_path]
    need(raw.count(OLD)==2 and NEW not in raw, 'unexpected original call sites')
    patch = raw.replace(OLD,NEW)
    need(patch.replace(NEW,OLD)==raw, 'observer change did not invert')

    def admit_joint(qualification, chosen, track, build_sha):
        need(chosen == build and build_sha == selected['build_report_sha256'], 'different execution build')
        try:
            qualification.admit_build(chosen, track, build_sha)
        except ValueError as e:
            need(str(e)=='build used another source manifest', 'unexpected legacy admission failure: '+str(e))
        else:
            raise ValueError('legacy unexpectedly admitted candidate')
        need(admit('dual', canonical_manifest, build, {'dual_scoped':scoped}) == selected, 'candidate custody changed')
        result['legacy_denials'].append('build used another source manifest')
        track(build/'heptatorch', DUAL)
        track(build/'runtime.hepta', selected['source_sha256'])
        return json.loads(track(build/'build.json', build_sha))

    cwd,argv,historical_cache = Path.cwd(),sys.argv[:],os.environ.get('PYTHONPYCACHEPREFIX')
    # Child uses the unchanged original supervisor, but no stale pyc may satisfy source identity.
    os.environ['PYTHONPYCACHEPREFIX'] = str(args.output/'clean-bytecode-cache')
    try:
        os.chdir(repo)
        runner=load('dual_retained_'+args.suite,repo/runner_path,patch,admit_joint=admit_joint)
        sys.argv=[str(repo/runner_path),'--build-directory',str(build),'--expected-build-sha256',
                  selected['build_report_sha256'],'--torch-python',str(args.torch_python),'--output',str(args.output/'observer')]
        code=runner.main()
    finally:
        os.chdir(cwd)
        sys.argv=argv
        if historical_cache is None:
            os.environ.pop('PYTHONPYCACHEPREFIX',None)
        else:
            os.environ['PYTHONPYCACHEPREFIX']=historical_cache
    original=read(args.output/'observer/report.json')
    observer=json.loads(original)
    result.update(original_observer_status=observer['status'], original_observer_exit=code,
                  original_observer_report_sha256=sha(original), first_failures=observer['failures'][:10],
                  original_observer_error=observer.get('error'), numeric_oracle_fresh=observer['Torch_executed'])
    need(code==0 and observer['status']=='PASS' and observer['native_executed'] is True
         and observer['Torch_executed'] is True and observer['native_sha256']==DUAL
         and observer['all_inputs_unchanged'] is True, 'unchanged original execution did not pass')
    proof_raw=read(args.output/'observer/physical-qualification.json')
    proof=json.loads(proof_raw)
    need(proof['physical_replay'] is True and proof['cases']==SUITES[args.suite]
         and proof['frames']==observer['frames'] and proof['assertions']==observer['checks']
         and len(result['legacy_denials'])==2, 'unchanged original physical readback did not pass')
    if args.suite=='typed_relu':
        need(proof['protocol_controls']==90 and
             proof['actual_allocation_refusals']==proof['identical_request_recoveries']==4,
             'original allocation or protocol controls not preserved')
    need(admit('dual',canonical_manifest,build,{'dual_scoped':scoped})==selected
         and all(read(repo/p)==data for p,data in sources.items()), 'source/image custody changed after replay')
    result.update(status='PASS_RETAINED_SUITE_ON_DUAL', physical_report_sha256=sha(proof_raw),
                  authorized_admission_only_observer_sha256=sha(patch), observer_physical_readback=proof)


def bind(path,pin):
    raw=read(path)
    need(pin is None or sha(raw)==pin,'legacy input did not bind: '+str(path))
    return raw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repository','dual-build','baseline-build','dual-scoped','torch-python','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--suite',choices=SUITES,required=True)
    args=p.parse_args()
    need(sys.flags.isolated and not sys.flags.optimize and sys.dont_write_bytecode,'use python3 -I -B')
    for key, value in vars(args).items():
        if key=='suite':
            continue
        if key!='torch_python':
            need(not value.is_symlink(),'unexpected symlink: '+str(value))
        setattr(args,key,value.absolute() if key=='torch_python' else value.resolve())
    args.output.mkdir()
    result=dict(schema='heptatorch.opt-in.direct-observer-replay.v1',status='FAIL_RETAINED_SUITE_ON_DUAL',retained_suite=args.suite,
                full_candidate_suite_qualification=False,canonical_qualification_transferred=False,
                production_activated=False, original_sources_sha256=SOURCE_SHA256)
    try:
        run(args,result)
    except (OSError,ValueError,AssertionError,KeyError,TypeError) as e:
        result['error']=dict(message=str(e),type=type(e).__name__)
    with (args.output/'receipt.json').open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(result['status'],flush=True)
    return 0 if result['status']=='PASS_RETAINED_SUITE_ON_DUAL' else 1


if __name__=='__main__':
    raise SystemExit(main())
