#!/usr/bin/env python3
"""Bind exact joint image to original live32 worker and complete raw independent verification."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import struct
import sys
RESOURCE=Path(__file__).with_name('replay_resource7.py')
if hashlib.sha256(RESOURCE.read_bytes()).hexdigest()!='831f5f5ddfc78a87cfbd981d31273c25037e63a914ca5f907cfdfab32eaccf13':
    raise ValueError('fixed complete joint source proof differs')
R=runpy.run_path(str(RESOURCE))
Joint,need,sha,load,bodies=[R[x] for x in ('Joint','need','sha','load','bodies')]
SOURCE,IMAGE=R['SOURCE'],R['ELF']
PINS={
 'current_runtime_live32.py':'4439a7b32254d01609b2d0b1639c1416349df8d0acbac5bbe2d21bb36fba3cfe',
 'current_runtime_original35/binding.py':'29d25567c08e309c25d96d56f4476e3842ceda164c9647eb5c9397a04fad181a',
 'current_runtime_process.py':'36e3432deac48085ba0586c85547cee8432d1533c5915238db2439b43dbca523',
 'current_runtime_admission.py':'fbf7fb0eef2d0436cce8a0ba01113b8ae5bede307c2c059dbf6ef722c6514b61',
 'current_runtime_torch.py':'570b5c300bc9967e9f7d92e934008b85bdb8b3e6e58818d67ec6a549e82598ca',
}

def setup(repo,build,source,receipt):
    ctx=Joint(repo,build,source,receipt)
    for rel,pin in PINS.items():
        p=repo/'tests'/rel
        need(p.resolve()==p and sha(p.read_bytes())==pin,'original pinned live32 source differs: '+rel)
    p=repo/'tests/current_runtime_live32.py'
    sys.argv=[str(p),'--repository',str(repo)]
    m=load(p,p.read_bytes(),'unchanged_historical_live32_gate')
    def scoped_binding(directory,expected,inputs):
        need(directory==build and expected==ctx.id['build_report_sha256'],'exact physical build path or receipt differs')
        proof=ctx.prove(inputs.bind(build/'runtime.hepta',SOURCE),inputs.bind)
        need(proof['joint_original_resource_route']['source_inverse_exact'] is True,'original domain proof did not pass')
        native=ctx.profile.admit_build(build,inputs.bind,expected)
        return native,bodies(inputs.bind(build/'runtime.hepta',SOURCE))[1]
    orig=m.load_source
    def scoped_load(path,raw,name,extra=None):
        obj=orig(path,raw,name,extra)
        if name=='actual_current_live32_readback':
            # Rebind only exactly the two globals the readback imported.
            need((obj.SOURCE,obj.IMAGE)==(R['BASE_SOURCE'],R['BASE_ELF']),'legacy independent profile changed')
            obj.SOURCE,obj.IMAGE=SOURCE,IMAGE
        return obj
    m.SOURCE,m.IMAGE=SOURCE,IMAGE
    m.bind_build=scoped_binding
    m.load_source=scoped_load
    return ctx,m

def census(output):
    ops=set();rows=0
    for path in (output/'original/native').glob('*/frame-*/request.bin'):
        raw=path.read_bytes()
        need(raw[:8]==b'HTOWNR01' and raw[16:24]==b'HTORCH03','original raw transport magic')
        n,extent,res=struct.unpack_from('<3Q',raw,24)
        need(48+64*n<=len(raw)==16+extent and res==0,'full original raw command inventory')
        ops.update(struct.unpack_from('<Q',raw,48+64*i)[0] for i in range(n));rows+=1
    need(rows==2132 and not ops&{3,127},'actual request leaves exact joint unchanged domain')
    return sorted(ops)

def main():
    p=argparse.ArgumentParser()
    for n in ('repository','dual-build','dual-source','dual-scoped','source-package','torch-python','output'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    repo,build,source,receipt,package,out=(p.resolve() for p in (a.repository,a.dual_build,a.dual_source,a.dual_scoped,a.source_package,a.output))
    ctx,m=setup(repo,build,source,receipt)
    build_sha=ctx.id['build_report_sha256']
    arguments=['--repository',str(repo),'--source-package',str(package),'--build-directory',str(build),
        '--expected-build-sha256',build_sha,'--torch-python',str(a.torch_python.parent.resolve() / a.torch_python.name),'--output',str(out)]
    code=m.main(arguments+(['--prepare-only'] if a.prepare_only else []))
    if not code and not a.prepare_only:
        raw=(out/'report.json').read_bytes()
        report=json.loads(raw)
        inputs=m.Inputs()
        actual=m.qualify_live32(report,build_sha,out,inputs)
        need(not inputs.recheck() and actual==report['readback'] and ctx.admission()==ctx.id,
             'fresh complete exact original raw readback differs')
        row=dict(status='PASS_RETAINED_LIVE32_ONE_PHYSICAL_COLD_BUILD',readback=actual,actual_build_sha256=build_sha,
                 source_sha256=SOURCE,native_image_sha256=IMAGE,original_report_sha256=sha(raw),
                 actual_opcodes=census(out),old_pass_transferred=False,full_candidate_suite_qualification=False,
                 production_activated=False,compiler_executed=False)
        (out/'joint-scope-result.json').write_text(json.dumps(row,indent=2,sort_keys=True)+'\n')
        print(json.dumps(row,sort_keys=True))
    return code

if __name__=='__main__':
    raise SystemExit(main())
