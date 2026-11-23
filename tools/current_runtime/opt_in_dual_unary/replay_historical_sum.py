"""Replay historical bounded sum suites on the exact joint opt-in executable."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types

DUAL='5af793857b1606f4760a2cae1d5ce9c5e8f821e07fb499ac0349567f2a503b72'
BASE='083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5'
DUAL_SOURCE='f4c43f037a3fb046425f7634215ec382aa5f8d6b0e7abe126ef64236adb5753a'
BASE_SOURCE='aab2e736ff0ef4a282ff73fb1a69a95d2ceeea5260bbdada0af5b30cc16dcd3c'
QUAL='tools/current_runtime/qualification.py'
FUSION='tests/sum_f32_review/fusion_support.py'
ADMIT='tools/current_runtime/opt_in_admission/admit.py'
PIN_SHA='72af93aac71f58a635809b7ef5aecee26601ac78f6f674f78bc507283b882a34'
CANONICAL_MANIFEST='tools/current_runtime/source/manifest.json'
NUMBERS={'sum_f32_17':(17,33,8194),'bool27':(27,36,6099),'integral24':(24,29,14504)}
QUAL_SUFFIX = b'''

# Scratch-only exact joint admission; original qualifier retained as a clone.
_historical_config_path = ROOT / 'historical-bridge-config.json'
_historical = json.loads(_historical_config_path.read_bytes())
_legacy = types.SimpleNamespace(SOURCE=SOURCE, IMAGE=IMAGE, FUNCTIONS=FUNCTIONS, UNITS=UNITS,
    admit_build=types.FunctionType(admit_build.__code__, dict(globals()), 'legacy_admit_build', admit_build.__defaults__))
SOURCE, IMAGE = _historical['candidate_source'], _historical['candidate_image']
_joint_manifest_path = Path(_historical['candidate_source_directory']) / 'manifest.json'
_joint_manifest_raw = _joint_manifest_path.read_bytes()
_joint_profile = json.loads(_joint_manifest_raw)
_joint = types.FunctionType(admit_build.__code__,
    {**globals(), 'PROFILE_PATH':_joint_manifest_path, 'PROFILE_BYTES':_joint_manifest_raw, 'PROFILE':_joint_profile},
    'joint_admit_build', admit_build.__defaults__)

def admit_build(directory, bind, expected=None):
    directory = Path(directory).resolve()
    require(directory == Path(_historical['candidate_build_directory']) and expected == _historical['candidate_build_hash'],
            'exact physically admitted joint build required')
    try:
        _legacy.admit_build(directory,bind,expected)
    except ValueError as error:
        require(str(error) == 'build used another source manifest', 'unexpected canonical refusal')
    else:
        raise ValueError('canonical reader unexpectedly admitted candidate')
    old = _legacy.admit_build(Path(_historical['baseline_build_directory']),bind,_historical['baseline_build_hash'])
    require(old['native_sha256']==_legacy.IMAGE and old['source_sha256']==_legacy.SOURCE,
            'wrong exact canonical build')
    actual = _joint(directory,bind,expected)
    require(actual['native_sha256']==IMAGE and actual['source_sha256']==SOURCE,
            'wrong exact joint build')
    bind(_historical_config_path)
    return actual
'''
FUSION_SUFFIX = b'''

# Scratch-only identity inverse. The historical proof sees its exact old source.
_historical_source_inverse = source_inverse

def source_inverse(source, inputs, support):
    profile = current_profile()
    require(sha(source)==profile.SOURCE, 'exact admitted joint monolithic source'
            )
    canonical = inputs.bind(ROOT/'tools/current_runtime/source/runtime.hepta',profile._legacy.SOURCE)
    helper = types.FunctionType(_historical_source_inverse.__code__,
        {**globals(), 'current_profile':lambda:profile._legacy}, 'historical_original_source_inverse')
    proof = helper(canonical,inputs,support)
    require(proof['source_sha256']==profile._legacy.SOURCE and proof['whole_inverse_exact'] is True,
            'unchanged complete historical inverse on independently admitted canonical source')
    return dict(joint_source_sha256=sha(source), canonical_source_sha256=sha(canonical),
                exact_joint_source_inverse_preverified=True, historical_canonical_proof=proof,
                runtime_qualification_transferred=False)
'''

def need(condition, message):
    if not condition:
        raise ValueError(message)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path):
    before=path.lstat()
    need(stat.S_ISREG(before.st_mode) and path.resolve()==path,'ordinary exact file: '+str(path))
    raw=path.read_bytes()
    after=path.lstat()
    def key(x): return x.st_dev,x.st_ino,x.st_mode,x.st_size,x.st_mtime_ns,x.st_ctime_ns
    need(key(before)==key(after) and len(raw)==after.st_size,'input changed: '+str(path))
    return raw

def dump(path, value):
    with path.open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n')

def extent(raw):
    matches=list(re.finditer(rb'^fn ([A-Za-z_][A-Za-z_0-9]*)\(.*?^end\n',raw,re.M|re.S))
    bodies={m[1].decode():m[0] for m in matches}
    header=raw.split(b'\n',1)[0]+b'\n'
    need(len(bodies)==len(matches)==800 and raw==header+b''.join(bodies.values()),'complete source extent')
    return header,bodies

def main(a,result):
    pin_raw=read(a.pins)
    need(sha(pin_raw)==PIN_SHA,'wrong exact source pin inventory')
    pins=json.loads(pin_raw)
    checkout=a.repository
    for p,h in pins.items():
        need(sha(read(checkout/p))==h,'exact checkout pin changed: '+p)
    old,dual=read(checkout/QUAL),read(checkout/ADMIT)
    mod=types.ModuleType('fixed_historical_dual_admit')
    mod.__file__=str(checkout/ADMIT)
    exec(compile(dual,mod.__file__,'exec'),mod.__dict__)
    baseline=mod.admit('baseline',checkout/CANONICAL_MANIFEST,a.baseline_build,{})
    joint=mod.admit('dual',checkout/CANONICAL_MANIFEST,a.dual_build,{'dual_scoped':a.dual_scoped})
    need(baseline['native_image_sha256']==BASE and joint['native_image_sha256']==DUAL and
         sha(read(a.dual_source/'manifest.json'))==joint['manifest_sha256'], 'exact executable and source manifest')
    current,after=[read(p/'runtime.hepta') for p in ((checkout/CANONICAL_MANIFEST).parent,a.dual_source)]
    need(sha(current)==BASE_SOURCE and sha(after)==DUAL_SOURCE and
         read(a.dual_build/'runtime.hepta')==after,'candidate monolith bytes')
    header,before=extent(current); second,changed=extent(after)
    diff={n for n in before if before[n]!=changed.get(n)}
    need(header==second and tuple(before)==tuple(changed) and
         diff=={'wide_neg_tensor','entry','predicate_bits','predicate_command'} and
         current==header+b''.join(before[n] if n in diff else changed[n] for n in changed) and
         after==header+b''.join(changed[n] if n in diff else before[n] for n in before),'whole exact inverse')
    result.update(source_change_names=sorted(diff),whole_exact_inverse=True,
                  canonical_image_sha256=BASE,joint_image_sha256=DUAL,
                  joint_build_report_sha256=joint['build_report_sha256'])
    repo=a.output/'scratch-source';repo.mkdir()
    bridge={QUAL:QUAL_SUFFIX,FUSION:FUSION_SUFFIX}
    for p,h in pins.items():
        dest=repo/p;dest.parent.mkdir(parents=True,exist_ok=True)
        if p in bridge:
            raw=read(checkout/p);suffix=bridge[p]
            derived=raw+suffix
            need(derived.endswith(suffix) and derived[:-len(suffix)]==raw,'append-only inverse')
            dest.write_bytes(derived)
        else:
            # Same-workspace immutable source link, without copying frozen archives.
            os.link(checkout/p,dest)
    cfg=dict(candidate_source=DUAL_SOURCE,candidate_image=DUAL,
        candidate_source_directory=str(a.dual_source),candidate_build_directory=str(a.dual_build),
        candidate_build_hash=joint['build_report_sha256'],baseline_build_directory=str(a.baseline_build),
        baseline_build_hash=baseline['build_report_sha256'])
    dump(repo/'historical-bridge-config.json',cfg)
    if a.suite=='sum_f32_17':
        argv=[str(repo/'tests/sum_f32_review/fusion.py'),'--repository',str(repo),
              '--binary',str(a.dual_build/'heptatorch'),'--source',str(a.dual_build/'runtime.hepta'),
              '--build-report',str(a.dual_build/'build.json'),'--expected-source-sha256',DUAL_SOURCE,
              '--expected-native-sha256',DUAL,'--reference',str(a.sum_reference),
              '--expected-reference-sha256','c86b0a9384c788eaf7765480b022416075a0e3aa38455f48dcbd7f40476cecb0',
              '--reader-directory',str(a.owner_directory)]
    else:
        argv=[str(repo/'tests/current_runtime_historical/run.py'),'--suite',a.suite,'--repository',str(repo),
              '--build-directory',str(a.dual_build),'--bool-reference',str(a.bool_reference),
              '--references-directory',str(a.references_directory),'--owner-directory',str(a.owner_directory)]
    report_dir=a.output/'observer'
    argv+=['--expected-build-sha256',joint['build_report_sha256'],'--output',str(report_dir)]
    proc=subprocess.run([sys.executable,'-I','-S','-B',*argv],cwd=repo,capture_output=True)
    (a.output/'stdout').write_bytes(proc.stdout);(a.output/'stderr').write_bytes(proc.stderr)
    result['observer_exit_code']=proc.returncode
    raw=read(report_dir/'report.json');report=json.loads(raw)
    result.update(original_gate_status=report['status'],original_gate_error=report.get('error'),original_outer_digest=sha(raw))
    if proc.returncode or report.get('native_executed') is not True:
        return
    original_raw=read(report_dir/'original/report.json');original=json.loads(original_raw)
    cases,sessions,frames=NUMBERS[a.suite]
    result.update(retained_historical_failure_sha256=sha(original_raw))
    source=report['source_sha256']
    image=report['native_sha256']
    need(original['status']=='FAIL' and original['native_executed'] is True and source==DUAL_SOURCE and image==DUAL
         and report['actual_build_sha256']==joint['build_report_sha256'],'only actual joint result allowed')
    transport=report['transport']
    need(transport['frames']==frames and len(original['cases'])==cases,'original case/transport census')
    if a.suite=='integral24':need(report['original_events_completed']==260,'all original integral events')
    need(all(sha(read(checkout/p))==h for p,h in pins.items()), 'all original checkout bytes remain fixed')
    need(mod.admit('dual',checkout/CANONICAL_MANIFEST,a.dual_build,{'dual_scoped':a.dual_scoped})==joint,
         'actual joint build changed during observation')
    result.update(status='PASS_RETAINED_HISTORICAL_SUM_ON_DUAL',cases=cases,sessions=sessions,frames=frames,
                  existing_historical_failures_preserved=True,
                  paired_bridge_sha256={k:sha(read(repo/k)) for k in bridge})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=NUMBERS,required=True)
    for name in ('repository','pins','baseline-build','dual-build','dual-source','dual-scoped','sum-reference','bool-reference',
                 'references-directory','owner-directory','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize,'run with python3 -I -B')
    for k,v in vars(a).items():
        if isinstance(v,Path):
            need(not v.is_symlink(),'no symbolic path aliases')
            setattr(a,k,v.resolve())
    a.output.mkdir()
    result=dict(status='FAIL_RETAINED_HISTORICAL_SUM_ON_DUAL',suite=a.suite,
                production_activated=False,canonical_qualification_transferred=False)
    try:
        main(a,result)
    except (OSError,ValueError,AssertionError,KeyError,TypeError) as e:
        result['failure']=dict(type=type(e).__name__,message=str(e))
    dump(a.output/'receipt.json',result)
    print(result['status'])
    sys.exit(not result['status'].startswith('PASS_'))
