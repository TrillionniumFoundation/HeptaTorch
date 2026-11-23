#!/usr/bin/env python3
"""Replay the original metadata10 worker and physical readback with joint admission only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

SOURCE = 'f4c43f037a3fb046425f7634215ec382aa5f8d6b0e7abe126ef64236adb5753a'
ELF = '5af793857b1606f4760a2cae1d5ce9c5e8f821e07fb499ac0349567f2a503b72'
CANONICAL_SOURCE = 'aab2e736ff0ef4a282ff73fb1a69a95d2ceeea5260bbdada0af5b30cc16dcd3c'
CANONICAL_ELF = '083a92ad78977396f6d2cacff7b7f559cec4871dfcfb67624db48b6f8d64c3d5'
META='tests/current_runtime_metadata10/'
DIRECT='tests/current_runtime270.py'
BINDING='tests/current_runtime_original35/binding.py'
QUALIFICATION='tools/current_runtime/qualification.py'
OFFLINE='tools/current_runtime/opt_in_admission/admit.py'
PROCESS='tests/current_runtime_process.py'
CATALOG=Path(__file__).with_name('metadata10-pins.json')
CATALOG_SHA='27b78c34174e407b8a271d5111ce21abb0b5b53115315e3f09ac18078a9a1d94'

def need(ok, message):
    if not ok:
        raise ValueError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def exact(raw, old, new):
    need(raw.count(old)==1 and new not in raw, 'fixed identity anchor changed')
    out=raw.replace(old,new,1)
    need(out.replace(new,old,1)==raw, 'identity transformation does not invert')
    return out

def load(path, raw, name, **kwargs):
    m=types.ModuleType(name)
    m.__file__=str(path)
    m.__dict__.update(kwargs)
    exec(compile(raw,str(path),'exec',dont_inherit=True,optimize=0),m.__dict__)
    return m

class Scoped:
    def __init__(self, repo, build, source, evidence):
        self.repo, self.build, self.source, self.evidence = repo, build, source, evidence
        data=CATALOG.read_bytes()
        need(sha(data)==CATALOG_SHA,'wrong scope source catalog')
        self.pins=json.loads(data)
        self.verify_pins()
        self.offline=load(repo/OFFLINE,self.raw(OFFLINE),'metadata_fixed_dual')
        self.admission=self.read_admission()
        need(self.admission['native_image_sha256']==ELF and self.admission['source_sha256']==SOURCE
             and sha((source/'manifest.json').read_bytes())==self.admission['manifest_sha256'],
             'physical build and staged sources must match the exact joint profile')
        p=QUALIFICATION
        raw=exact(self.raw(p),b'PROFILE_PATH = ROOT / "tools/current_runtime/source/manifest.json"',
                  b'PROFILE_PATH = scoped_manifest')
        self.profile=load(repo/p,raw,'metadata_dual_complete_admission',scoped_manifest=source/'manifest.json')
        p=BINDING
        raw=exact(self.raw(p),b'built = module.admit_build(directory, inputs.bind, expected)',
                  b'built = scoped_admit(module, directory, inputs.bind, expected)')
        self.binding=load(repo/p,raw,'metadata_dual_current270_binding',scoped_admit=self.admit)
        self.calls=[]

    def verify_pins(self):
        for relative,pin in self.pins.items():
            p=self.repo/relative
            need(p.resolve()==p and sha(p.read_bytes())==pin,'fixed input changed: '+relative)
            need(not (p.parent/'__pycache__').exists(),'fresh checkout without bytecode caches required')

    def raw(self,p):
        return (self.repo/p).read_bytes()

    def read_admission(self):
        return self.offline.admit('dual',self.repo/'tools/current_runtime/source/manifest.json',self.build,
                                  {'dual_scoped': self.evidence})

    def admit(self, legacy, build, bind, expected):
        need(build==self.build and expected==self.admission['build_report_sha256'],'different candidate')
        try:
            legacy.admit_build(build,bind,expected)
        except ValueError as e:
            need(str(e)=='build used another source manifest','unexpected legacy denial: '+str(e))
        else:
            raise ValueError('canonical admission must still deny candidate')
        self.calls.append('build used another source manifest')
        result=self.profile.admit_build(build,bind,expected)
        need(result['native_sha256']==ELF and result['source_sha256']==SOURCE,'wrong full-profile identity')
        return result

    def adapter(self,m):
        m.bind_build=self.binding.bind_build
        return m

    def common(self):
        p=META+'common.py'
        raw=exact(self.raw(p),b'built=profile.admit_build(build,inputs.bind,expected)',
                  b'built=scoped_admit(profile,build,inputs.bind,expected)')
        raw=exact(raw,b"built['source_sha256']==pins['source_sha256'] and built['native_sha256']==pins['native_sha256']",
                  b"scoped_image_matches(built,pins)")
        def matches(built,pins):
            return ((pins['source_sha256'],pins['native_sha256'])==(CANONICAL_SOURCE,CANONICAL_ELF)
                    and (built['source_sha256'],built['native_sha256'])==(SOURCE,ELF))
        m=load(self.repo/p,raw,'metadata_scoped_common',scoped_admit=self.admit,scoped_image_matches=matches)
        previous=m.load
        def hook(path,name,inputs,expected=None):
            orig=previous(path,name,inputs,expected)
            return self.adapter(orig) if path==self.repo/DIRECT else orig
        m.load=hook
        return m

    def driver(self, driver):
        previous=driver.load
        def hook(path,name):
            orig=previous(path,name)
            return self.adapter(orig) if path==self.repo/DIRECT else orig
        driver.load=hook

def run_worker(repo,build,source,evidence):
    need(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode and not sys.flags.optimize,
         'actual worker requires -I -S -B')
    ctx=Scoped(*(Path(p) for p in (repo,build,source,evidence)))
    p=META+'worker.py'
    raw=exact(ctx.raw(p),
              b'common=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(common)',
              b'common=scoped_common_module')
    raw=exact(raw,b"exec(compile(source,str(driver_path),'exec'),driver.__dict__)",
              b"exec(compile(source,str(driver_path),'exec'),driver.__dict__);scope_driver_admission(driver)")
    worker=load(ctx.repo/p,raw,'metadata_exact_worker',scoped_common_module=ctx.common(),
                scope_driver_admission=ctx.driver)
    rc=worker.main()
    if rc==0:
        ctx.verify_pins()
        need(ctx.read_admission()==ctx.admission and len(ctx.calls)>=2,'worker did not complete both full admissions')
    return rc

def run(a):
    repo,build,out=a.repository,a.dual_build,a.output
    ctx=Scoped(repo,build,a.dual_source,a.dual_scoped)
    python=Path(sys.executable).resolve()
    launcher=out/'scoped-worker.py'
    # The physical runner records this actual launcher path and genuine argv;
    # it is not relabelled as the original worker. The worker source is pinned.
    config=tuple(map(str,(repo, build, a.dual_source, a.dual_scoped)))
    here=Path(__file__).resolve()
    launcher.write_text('import hashlib, runpy\nfrom pathlib import Path\n'
      + f'p=Path({str(here)!r})\n'
      + f'if hashlib.sha256(p.read_bytes()).hexdigest()!={sha(here.read_bytes())!r}: raise ValueError("wrapper changed")\n'
      + f'raise SystemExit(runpy.run_path(str(p))["run_worker"](*{config!r}))\n')
    launcher_sha=sha(launcher.read_bytes())
    process=load(repo/PROCESS,ctx.raw(PROCESS),'metadata_bound_physical_process')
    execution=out/'execution'
    execution.mkdir()
    details=process.execute([str(python),'-I','-S','-B',str(launcher),
         '--repository',str(repo),'--build-directory',str(build),'--expected-build-sha256',
         ctx.admission['build_report_sha256'],'--output',str(out/'observer')],execution,300,cwd=repo)
    (execution/'runner.json').write_text(json.dumps(details,indent=2,sort_keys=True)+'\n')
    need(sha(launcher.read_bytes())==launcher_sha,'actual worker launcher changed')
    observer=(out/'observer/report.json').read_bytes()
    report=json.loads(observer)
    result=dict(status='FAIL_METADATA10_REPLAY',observer_status=report['status'],observer_exit=details['exit_code'],
                report_sha256=sha(observer),worker_sha256=launcher_sha,
                full_candidate_suite_qualification=False,production_activated=False)
    if details['exit_code']!=0:
        result['error']=report.get('error')
        return result
    p=META+'qualify.py'
    raw=exact(ctx.raw(p),
              b'common=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(common)',
              b'common=scoped_common_module')
    raw=exact(raw,b"argv[4]==str(HERE/'worker.py')",b"argv[4]==str(scoped_worker_path)")
    reviewer=load(repo/p,raw,'metadata_bound_physical_readback',
                  scoped_common_module=ctx.common(),scoped_worker_path=launcher)
    inputs=ctx.binding.Inputs()
    inputs.bind(launcher, result['worker_sha256'])
    readback=reviewer.qualify_metadata10(report,repo,build,ctx.admission['build_report_sha256'],
                                         out/'observer',inputs,execution)
    need(ctx.read_admission()==ctx.admission and not inputs.recheck(), 'original readback inputs changed')
    ctx.verify_pins()
    need(len(ctx.calls)==2 and readback['cases']==10 and readback['frames']==791
         and readback['original_assertions']==814 and readback['native_sessions']==10,
         'complete retained metadata10 result and readback admission required')
    result.update(status='PASS_RETAINED_METADATA10',readback=readback,actual_build_sha256=ctx.admission['build_report_sha256'],
                  native_sha256=ELF,source_sha256=SOURCE, worker_stream_seal_sha256=sha((execution/'runner.json').read_bytes()))
    return result

def main():
    p=argparse.ArgumentParser()
    for key in ('repository','dual-build','dual-source','dual-scoped','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize,'use python3 -I -B')
    for k,v in vars(a).items():
        need(not v.is_symlink(),f'unexpected symlink: {v}')
        setattr(a,k,v.resolve())
    a.output.mkdir()
    try:
        r=run(a)
    except Exception as e:
        r=dict(status='BLOCKED_OR_FAILED',error=f'{type(e).__name__}: {e}')
    (a.output/'receipt.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(r['status'])
    return 0 if r['status']=='PASS_RETAINED_METADATA10' else 1

if __name__=='__main__':
    raise SystemExit(main())
