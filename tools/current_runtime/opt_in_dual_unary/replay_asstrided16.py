#!/usr/bin/env python3
"""Run the four immutable historical AsStrided groups on the exact opt-in dual image."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import struct
import sys

RESOURCE_SHA='831f5f5ddfc78a87cfbd981d31273c25037e63a914ca5f907cfdfab32eaccf13'
PINS=dict(common='edc725271b693a18e556710aae42795ddf28222950250e89093de328c55911fa',
          worker='c81f5eb69cdd98847f10e28280588dec03a369e3fd94ca3a41ad2d2bf4f20bc0',
          qualify='0e090798a82bee05d809f7c15b4b24c96a0196bb21aeedc9ba2ce56d137ad13e')
RESOURCE=Path(__file__).with_name('replay_resource7.py')
if hashlib.sha256(RESOURCE.read_bytes()).hexdigest()!=RESOURCE_SHA:
    raise ValueError("shared exact dual inverse changed")
R=runpy.run_path(str(RESOURCE))
need,sha,exact,load,Joint=[R[x] for x in ('need','sha','exact','load','Joint')]
BASE_SOURCE,BASE_ELF,SOURCE,ELF=[R[x] for x in ('BASE_SOURCE','BASE_ELF','SOURCE','ELF')]

def scoped(repo,build,source,receipt):
    ctx=Joint(repo,build,source,receipt)
    gate=repo/'tests/current_runtime_asstrided'
    def read(name):
        p=gate/(name+'.py');data=p.read_bytes()
        need(p.resolve()==p and sha(data)==PINS[name],'pinned original AsStrided source differs: '+name)
        return p,data

    def build_admit(path, bind, expected):
        ctx.prove(bind(path/'runtime.hepta',SOURCE),bind)
        return ctx.profile.admit_build(path,bind,expected)

    p,raw=read('common')
    raw=exact(raw,b'built = current.admit_build(build, inputs.bind, expected_build_sha)',
                  b'built = scoped_build_admit(build, inputs.bind, expected_build_sha)')
    raw=exact(raw,b"raw = inputs.bind(build/'runtime.hepta', SOURCE)",
                  b"raw = inputs.bind(repository/'tools/current_runtime/source/runtime.hepta', SOURCE)")
    raw=exact(raw,b'return current, built, reference, reference_path, proof',
                  b'return scoped_profile, built, reference, reference_path, bind_joint_proof(proof, build, inputs)')
    def attach(proof,path,inputs):
        route=ctx.prove(inputs.bind(path/'runtime.hepta',SOURCE),inputs.bind)
        return dict(proof,source_sha256=SOURCE,native_sha256=ELF,
                    exact_joint_to_canonical800=dict(canonical_source=BASE_SOURCE,canonical_image_is_not_executed=BASE_ELF,
                        dispatcher=route['obligations']['to_command']['predicate_entry_route'],
                        **route['joint_original_resource_route']))

    common=load(p,raw,'scoped_asstrided16_common',scoped_build_admit=build_admit,
                  scoped_profile=ctx.profile,bind_joint_proof=attach)
    def load_driver(name, wrapper=None):
        p,raw=read(name)
        # Redirect just the local common import; keep distinct canonical
        # ancestor constants inside its historical lineage proof.
        start=raw.index(b'_spec=importlib.util.spec_from_file_location')
        end=raw.index(b'\n',raw.index(b'_spec.loader.exec_module(common)',start))+1
        raw=exact(raw,raw[start:end],b'common=scoped_common\n')
        names=(b'Inputs,SOURCE,IMAGE,SUITES,COUNTS=common.Inputs,common.SOURCE,common.IMAGE,common.SUITES,common.COUNTS' if name=='worker' else
               b'SOURCE,IMAGE,SUITES,COUNTS,SOURCES_SHA=common.SOURCE,common.IMAGE,common.SUITES,common.COUNTS,common.SOURCES_SHA')
        raw=exact(raw,names,names.replace(b'common.SOURCE,common.IMAGE',b'physical_source,physical_elf'))
        if wrapper is not None:
            raw=exact(raw,b"str(HERE/'worker.py')",b"str(scoped_launcher)")
            raw=exact(raw,b"str(Path(report['source_directory'])/'worker.py')",b"scoped_launcher.as_posix()")
            raw=exact(raw,b"        require(32<=response_size<=1<<20",
                b"        require(not ({3,115,116,117,118,119,127} & {struct.unpack_from('<Q',request,32+64*i)[0] for i in range(count)}), 'original request leaves exact unchanged source domain')\n        require(32<=response_size<=1<<20")
        return load(p,raw,'scoped_asstrided16_'+name,scoped_common=common,
                    scoped_launcher=wrapper,physical_source=SOURCE,physical_elf=ELF)
    return ctx,common,load_driver

def worker(*paths):
    need(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode and not sys.flags.optimize,
         'worker isolation differs')
    ctx,common,driver=scoped(*(Path(p) for p in paths))
    m=driver('worker')
    return m.main()

def run(repo,build,source,receipt,out):
    self=Path(__file__).resolve()
    wrapper=out/'scoped-worker.py'
    wrapper.write_text('import hashlib,runpy\nfrom pathlib import Path\n'
        +f'p=Path({str(self)!r})\nif hashlib.sha256(p.read_bytes()).hexdigest()!={sha(self.read_bytes())!r}: raise ValueError("scope changed")\n'
        +f'raise SystemExit(runpy.run_path(str(p))["worker"](*{tuple(map(str,(repo,build,source,receipt)))!r}))\n')
    wrapper_sha=sha(wrapper.read_bytes())
    ctx,common,driver=scoped(repo,build,source,receipt)
    m=driver('qualify',wrapper)
    build_sha=ctx.id['build_report_sha256']
    observer=out/'observer'
    sys.argv=[m.__file__,'--repository',str(repo),'--build-directory',str(build),
        '--expected-build-sha256',build_sha,'--output',str(observer)]
    exit_code=m.main()
    report_path=observer/'report.json'
    report=json.loads(report_path.read_bytes())
    if exit_code:
        return dict(status='FAIL_ASSTRIDED16',error=report.get('error') or report['failures'])
    # Reread immutable originals, original source inverses, physical runner
    # outputs, both EOFs, and all raw packet/owner evidence from a fresh binder.
    actual=driver('qualify',wrapper).qualify_asstrided16(
        report,build_sha,observer,common.Inputs(),repository=repo)
    need(actual==report['readback'] and sha(wrapper.read_bytes())==wrapper_sha
         and ctx.admission()==ctx.id,'independent readback/candidate provenance differs')
    return dict(status='PASS_RETAINED_ASSTRIDED16',readback=actual,
        build_report_sha256=build_sha,original_physical_report_sha256=sha(report_path.read_bytes()),
        physical_wrapper_sha256=wrapper_sha,source_sha256=SOURCE,native_image_sha256=ELF,
        native_executed=True,compiler_executed=False,Torch_executed=False,production_activated=False,
        old_pass_transferred=False,full_candidate_suite_qualification=False)

def main():
    p=argparse.ArgumentParser()
    for key in ('repository','dual-build','dual-source','dual-scoped','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize,'use python3 -I -B')
    for k,v in vars(a).items():
        need(not v.is_symlink(),'unexpected symlink')
        setattr(a,k,v.resolve())
    a.output.mkdir()
    try:
        row=run(a.repository,a.dual_build,a.dual_source,a.dual_scoped,a.output)
    except Exception as e:
        row=dict(status='BLOCKED_OR_FAILED',error=f'{type(e).__name__}: {e}')
    (a.output/'receipt.json').write_text(json.dumps(row,indent=2,sort_keys=True)+'\n')
    return int(row['status']!='PASS_RETAINED_ASSTRIDED16')

if __name__=='__main__':
    raise SystemExit(main())
