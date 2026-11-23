#!/usr/bin/env python3
"""Run the fixed original F32 Sum32 reader on the exact joint candidate."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import runpy
import struct
import sys

RESOURCE=Path(__file__).with_name('replay_resource7.py')
if hashlib.sha256(RESOURCE.read_bytes()).hexdigest()!='831f5f5ddfc78a87cfbd981d31273c25037e63a914ca5f907cfdfab32eaccf13':
    raise ValueError('exact joint inverse changed')
R=runpy.run_path(str(RESOURCE))
need,sha,load,Joint=[R[x] for x in ('need','sha','load','Joint')]
SOURCE,IMAGE=R['SOURCE'],R['ELF']
PINS={'tests/current_runtime_legacy/sum32.py':'7cec93c961401e5013ea4e2b1b2cf227ba18a91bf5ae06d4fd75e15520ecabca',
      'tests/current_runtime_legacy/run.py':'e4f3c6118ce66bcec7489533a876652075674bca86ecb05cb4183b0a1780cb42',
      'tests/current_runtime_fixtures.py':'a8c00f1958bf9166d1e36c012bec95b29333a6cd434a19b2296c5ce4434c175b',
      'tests/qualify_current_runtime.py':'ad0beeee9884c313db46a089011043e634cc255d06718f6349b3377ecfc98618'}

def selected(repo, relative):
    p=repo/relative; raw=p.read_bytes()
    need(p.resolve()==p and sha(raw)==PINS[relative],'wrong fixed original input: '+relative)
    return p,raw

def worker(paths):
    repo,build,source,receipt,shared,frozen,intent,out=map(Path,paths)
    ctx=Joint(repo,build,source,receipt)
    arg=['--build-directory',str(build),'--expected-build-sha256',ctx.id['build_report_sha256'],
        '--original-source',str(shared/'original32-source'),'--parent-source',str(shared/'parent-native32-source'),
        '--reference-source',str(shared/'reference32-source'),'--reference',str(shared/'reference32/actual.json'),
        '--output',str(out)]
    p,raw=selected(repo,'tests/current_runtime_legacy/sum32.py')
    original=load(p,raw,'unchanged_sum32_worker')
    binder=original.Inputs()
    route=ctx.prove(binder.bind(build/'runtime.hepta',SOURCE),binder.bind)
    path,raw=selected(repo,'tests/current_runtime_fixtures.py')
    # The unchanged fixture proof is isolated from unrelated fixture imports.
    proof=load(path,raw,'unchanged_legacy_source_proof').verify_legacy_sources(repo,frozen,binder)
    hpath,hraw=selected(repo,'tests/current_runtime_legacy/run.py')
    legacy=load(hpath,hraw,'exact_legacy_sum32_physical_binding')
    legacy.current_profile=ctx.profile
    legacy.SOURCE_SHA,legacy.NATIVE_SHA=SOURCE,IMAGE
    old=original.load
    def scoped_load(path,name):
        if path==hpath:
            need(name=='current_runtime_build_admission','unexpected legacy admission use')
            return legacy
        return old(path,name)
    original.load=scoped_load
    original.INTEGRAL_INTENT=intent
    sys.argv=[str(p),*arg]
    code=original.main()
    if code==0:
        need(not binder.recheck() and ctx.admission()==ctx.id,'exact proof inputs changed')
        (out/'joint-source-proof.json').write_text(json.dumps(dict(route=route['joint_original_resource_route'],
            legacy_proof=proof, actual_scope_only=True, original_qualifier_comparisons_modified=False),
            indent=2,sort_keys=True)+'\n')
    return code

def inspect_packet_file(raw, census):
    # Parent Native wire archive: complete little-endian length frames.
    need(raw[:8]==b'HTOWNR01','unexpected historical Sum32 request frame stream identity')
    at=8; count=0
    while at<len(raw):
        need(at+8<=len(raw),'truncated outer frame')
        n=struct.unpack_from('<Q',raw,at)[0]; at+=8
        need(at+n<=len(raw) and n>=32 and raw[at:at+8]==b'HTORCH03','invalid Tensor packet in original stream')
        c,extent,reserved=struct.unpack_from('<3Q',raw,at+8)
        need(32+64*c<=n==extent and reserved==0,'invalid complete native request table')
        ops={struct.unpack_from('<Q',raw,at+32+64*i)[0] for i in range(c)}
        need(not (ops & {3,115,116,117,118,119,127}),'request leaves exact unchanged source domain')
        census.update(ops); count+=1; at+=n
    return count

def independent(repo,out,build_sha):
    p,raw=selected(repo,'tests/qualify_current_runtime.py')
    fn,=[node for node in ast.parse(raw).body if isinstance(node,ast.FunctionDef) and node.name=='qualify_sum32']
    original=ast.get_source_segment(raw.decode(),fn).encode()+b'\n'
    reader=load(p,original,'unchanged_independent_sum32_readback',IMAGE=IMAGE,require=need,
        SUM32_REFERENCE='c14e4b18f657429ad5dac5498c3abcae303b6a7e7550db7d15a3c9179bdcd728')
    report=json.loads((out/'report.json').read_bytes())
    counts=reader.qualify_sum32(report,build_sha)
    need(report['source_sha256']==SOURCE and report['native_sha256']==IMAGE,'actual candidate report identity')
    census=set(); sessions=[]
    for i, row in enumerate(report['transports']):
        d=out/('case-%02d'%i)/'raw'
        raw=(d/'requests.frames.bin').read_bytes()
        case=json.loads((d.parent/'report.json').read_bytes())
        ledger=json.loads((d/'transport.json').read_bytes())
        need(case==report['cases'][i] and all(ledger[k]==v for k,v in row.items()),'original case/transport readback differs')
        count=inspect_packet_file(raw,census)
        need(count==ledger['frames'],'physical request frame count differs')
        sessions.append(dict(index=i,frames=count,request_sha256=sha(raw),response_sha256=sha((d/'responses.frames.bin').read_bytes())))
    return dict(status='PASS_RETAINED_SUM32',readback=counts,physical_sessions=sessions,opcodes=sorted(census),
        native_image_sha256=IMAGE,source_sha256=SOURCE,actual_build_sha256=build_sha,
        original_report_sha256=sha((out/'report.json').read_bytes()), production_activated=False,
        old_pass_transferred=False, full_candidate_suite_qualification=False)

def main():
    p=argparse.ArgumentParser()
    for n in ('repository','dual-build','dual-source','dual-scoped','shared','frozen','integral-intent','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    need(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode and not sys.flags.optimize,'use python -I -S -B')
    paths=tuple(str(p.resolve()) for p in vars(a).values())
    out=Path(paths[-1]); out.mkdir()
    repo,build,source,receipt,*_=map(Path,paths)
    try:
        ctx=Joint(repo,build,source,receipt)
        process=load(repo/'tests/current_runtime_process.py',(repo/'tests/current_runtime_process.py').read_bytes(),'sum32_isolated_worker_process')
        self=Path(__file__).resolve()
        launch=out/'worker.py'
        launcher_paths=paths[:-1]+(str(out/'actual'),)
        launch.write_text('import hashlib,runpy\nfrom pathlib import Path\n'
          +f'p=Path({str(self)!r})\nif hashlib.sha256(p.read_bytes()).hexdigest()!={sha(self.read_bytes())!r}: raise ValueError("scope changed")\n'
          +f'raise SystemExit(runpy.run_path(str(p))["worker"]({launcher_paths!r}))\n')
        launch_sha=sha(launch.read_bytes())
        proc=out/'worker-execution';proc.mkdir()
        argv=[sys.executable,'-I','-S','-B',str(launch)]
        run=process.execute(argv,proc,900,cwd=repo,mirror_stderr=True)
        need(run['argv']==argv and run['started'] is True and run['exit_code']==0 and run['stdout_EOF'] is True
             and run['stderr_EOF'] is True and run['stderr_bytes']==0 and sha(launch.read_bytes())==launch_sha,
             'physical worker did not complete')
        result=independent(repo,out/'actual',ctx.id['build_report_sha256'])
        result['scoped_worker_sha256']=launch_sha
    except Exception as e:
        result=dict(status='BLOCKED_OR_FAILED',error=f'{type(e).__name__}: {e}')
    (out/'receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return int(result['status']!='PASS_RETAINED_SUM32')
if __name__=='__main__':
    raise SystemExit(main())
