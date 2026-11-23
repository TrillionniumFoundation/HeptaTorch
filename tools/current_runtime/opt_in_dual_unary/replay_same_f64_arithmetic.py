#!/usr/bin/env python3
"""Retain exact same-F64 arithmetic gate on joint opt-in current800."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import struct
import sys

RESOURCE=Path(__file__).with_name('replay_resource7.py')
if hashlib.sha256(RESOURCE.read_bytes()).hexdigest()!='831f5f5ddfc78a87cfbd981d31273c25037e63a914ca5f907cfdfab32eaccf13':
    raise ValueError('exact complete joint proof changed')
R=runpy.run_path(str(RESOURCE))
Joint,need,sha,load=[R[n] for n in ('Joint','need','sha','load')]
SOURCE,IMAGE=R['SOURCE'],R['ELF']
SUITE='tests/same_f64_arithmetic/run.py'
PINS={'tests/same_f64_arithmetic/run.py':'b85c19db360a112650cde116b1de3790702a0a940b82465e5511045091063d5f',
 'tests/same_f64_arithmetic/cases.py':'3d93f6c0c8e0bda799fb5b4debb1a246f559edf7faf3e243d911286e04f9200a',
 'tests/same_f64_arithmetic/reference.py':'e9bff3a1a8b93771b375b84ae655728600b9ec54e155185f2faa1bf3be935bc5',
 'tests/same_f64_arithmetic/__init__.py':'a6dcddc48ca2370b1caf50c6065927e241493089d1494f131b479d3e3f7fcff1',
 'tests/numeric_predicates/run.py':'e9fe4a7361cef25aca14c24ecd17c5413f058e5ccb3c35190d66b0be3b479991',
 'tests/numeric_predicates/native.py':'c0e9d1ba6f0dac5499f091333611a5bea0fb6f4a25e74faa39f35486fd404d2d',
 'tests/numeric_predicates/reference.py':'3ec85934110f2d8f273bc1e209a9c9ce13331cc4ff915dc9d7058d81913d0abd',
 'tests/numeric_predicates/cases.py':'e69bebed091db88ef81eda2ea2052af6b2d88c74840909ee27e2cbdf6a9ec694',
 'tests/half_logical/native.py':'9c2cd19a5bb04eaa037f5905a52bce2dcc31042101f72d07e71205745160a14f',
 'tests/current_runtime_process.py':'36e3432deac48085ba0586c85547cee8432d1533c5915238db2439b43dbca523',
}

def census(out):
    ops=set(); frames=0
    for file in sorted(out.glob('case-*/raw/requests.frames.bin')):
        raw=file.read_bytes();offset=0
        while offset<len(raw):
            magic,n=struct.unpack_from('<8sQ',raw,offset);offset+=16
            need(magic==b'HTOWNR01' and raw[offset:offset+8]==b'HTORCH03' and 32<=n<=len(raw)-offset,
                 'exact original wire framing')
            c,extent,res=struct.unpack_from('<3Q',raw,offset+8)
            need(32+64*c<=n==extent and res==0,'exact complete original packet census')
            ops.update(struct.unpack_from('<Q',raw,offset+32+64*j)[0] for j in range(c))
            frames+=1;offset+=n
    need(not ops & {3,127},'physical arithmetic requests must stay in the original domain')
    return frames,sorted(ops)

def setup(repo,build,source,receipt):
    ctx=Joint(repo,build,source,receipt)
    for n,h in PINS.items():
        p=repo/n;need(p.resolve()==p and sha(p.read_bytes())==h,'original arithmetic source differs: '+n)
    m=load(repo/SUITE,(repo/SUITE).read_bytes(),'unchanged_same_f64_arithmetic')
    orig_m,orig_source=m.module,m.source_inverse
    def scoped_module(p,n,bind):
        if p==repo/'tools/current_runtime/qualification.py':
            bind(p,ctx.pins[R['PROFILE']])
            return ctx.profile
        return orig_m(p,n,bind)
    def scoped_inverse(actual,inputs):
        need(actual==ctx.profile.admit_build(build,inputs.bind,ctx.id['build_report_sha256']),
             'actual full candidate admission differs')
        route=ctx.prove(inputs.bind(build/'runtime.hepta',SOURCE),inputs.bind)
        # The predecessor proof only accepts the canonical source SHA. Pass its
        # exact restored source identity, not a canonical executable or receipt.
        ancestor=orig_source(dict(source_sha256=R['BASE_SOURCE']),inputs)
        return dict(canonical_exact_source_chain=ancestor,
            actual_candidate_source_sha256=SOURCE,actual_candidate_native_sha256=IMAGE,
            joint_to_canonical_exact_inverse=route['joint_original_resource_route'])
    m.module=scoped_module;m.source_inverse=scoped_inverse
    return ctx,m

def main():
    p=argparse.ArgumentParser()
    for n in ('repository','dual-build','dual-source','dual-scoped','torch-python','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    repo,build,source,receipt,out=(q.resolve() for q in (a.repository,a.dual_build,a.dual_source,a.dual_scoped,a.output))
    ctx,m=setup(repo,build,source,receipt)
    build_sha=ctx.id['build_report_sha256']
    sys.argv=[str(repo/SUITE),'--build-directory',str(build),'--expected-build-sha256',build_sha,
              '--torch-python',str(a.torch_python.parent.resolve()/a.torch_python.name),'--output',str(out)]
    code=m.main()
    if code==0:
        actual_raw=(out/'report.json').read_bytes();actual=json.loads(actual_raw)
        inputs=m.Inputs()
        physical=m.qualify(actual,repo,build,build_sha,out,inputs)
        frames,ops=census(out)
        need(physical==json.loads((out/'physical-qualification.json').read_bytes())
             and frames==physical['frames'] and ctx.admission()==ctx.id,
             'complete second physical readback/source/packet denominator changed')
        result=dict(status='PASS_RETAINED_SAME_F64_ARITHMETIC_ONE_PHYSICAL_COLD_BUILD',
             source_sha256=SOURCE,native_image_sha256=IMAGE,actual_build_sha256=build_sha,
             original_report_sha256=sha(actual_raw),replay=physical,retained_opcodes=ops,
             old_pass_transferred=False,production_activated=False,full_candidate_suite_qualification=False)
        (out/'joint-result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
        print(json.dumps(result,sort_keys=True))
    return code

if __name__=='__main__':
    raise SystemExit(main())
