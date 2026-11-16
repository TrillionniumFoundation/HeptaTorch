#!/usr/bin/env python3
"""Execute all guarded-stack gates and the unchanged current Tensor program."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def main():
    spec=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--tensor-baseline',type=Path,required=True)
    p.add_argument('--canonical-binary',type=Path,required=True)
    p.add_argument('--service-sources',type=Path,nargs=5,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();B=a.build.resolve();R=a.repo.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=False)
    sha=lambda raw:hashlib.sha256(raw).hexdigest()
    identity={}
    for line in (spec/'source-pins.sha256').read_text().splitlines():
        expected,relative=line.split('  ',1)
        assert sha((spec/relative).read_bytes())==expected,relative
        identity[relative]=expected
    profile=json.loads((spec/'profile.json').read_bytes())
    assert sha((B/'native/stack-module-compiler').read_bytes())==profile['compiler']['executable_sha256']
    for name,row in profile['linkers'].items():
        assert sha((B/'native'/('stack-linker-'+name)).read_bytes())==row['executable_sha256']
    common=['--sources',B/'sources','--native',B/'native']

    def run(label,script,arguments):
        result=subprocess.run([sys.executable,'-I','-S','-B',str(spec/script),*map(str,arguments)],capture_output=True,timeout=300)
        (O/(label+'.stdout')).write_bytes(result.stdout);(O/(label+'.stderr')).write_bytes(result.stderr)
        assert result.returncode==0 and not result.stderr,(label,result.returncode,result.stderr[-2000:])
        print(label+': PASS',flush=True)

    run('boundaries','test_stack.py',common+['--service-sources',*[p.resolve() for p in a.service_sources],
        '--owner-client',R/'tools/owned_sessions/client.py','--output',O/'boundaries'])
    run('production-capacity','test_large.py',common+['--output',O/'production-capacity'])
    run('dirty-workspace','test_dirty.py',common+['--review',O/'boundaries','--output',O/'dirty-workspace'])
    run('machine-call-binding','test_binding.py',common+['--review',O/'boundaries',
        '--owned-flow',R/'tools/owned_modules/tests/independent_flow.py','--output',O/'machine-call-binding'])
    baseline=a.tensor_baseline.resolve()
    run('tensor-build','compile_application.py',common+['--baseline-build',baseline,'--output',O/'tensor-build',
        *[baseline/'units'/(name+'.hepta') for name in ('ht_kernel','ht_runtime','owned_tensor_session')]])
    assert sha((O/'tensor-build/owned-tensor-session').read_bytes())==profile['current_tensor']['executable_sha256']
    run('tensor-execution','qualify_application.py',['--repo',R,'--application-build',O/'tensor-build',
        '--canonical-binary',a.canonical_binary.resolve(),'--output',O/'tensor-execution'])
    rows=[]
    for name,count in [('boundaries',43),('production-capacity',5),('dirty-workspace',6),('machine-call-binding',32)]:
        report=O/name/'report.json'
        data=json.loads(report.read_bytes())
        assert data['status']=='PASS' and data['cases']==count,(name,data.get('cases'))
        rows.append({'suite':name,'cases':count,'report_sha256':sha(report.read_bytes()),'report':name+'/report.json'})
    tensor=json.loads((O/'tensor-execution/report.json').read_bytes())
    assert tensor['status']=='PASS' and tensor['native_cases']==11 and tensor['independent_response_frames']==16
    for relative,expected in identity.items():assert sha((spec/relative).read_bytes())==expected,relative
    assert sha((B/'native/stack-module-compiler').read_bytes())==profile['compiler']['executable_sha256']
    for name,row in profile['linkers'].items():assert sha((B/'native'/('stack-linker-'+name)).read_bytes())==row['executable_sha256']
    report={'schema':'heptatorch.stack-budget-portable-qualification.v1','status':'PASS','native_gate_cases':86,'rows':rows,
        'package_source_sha256':identity,'compiler_sha256':profile['compiler']['executable_sha256'],
        'session_linker_sha256':profile['linkers']['session']['executable_sha256'],
        'tensor_executable_sha256':profile['current_tensor']['executable_sha256'],
        'tensor_build_report_sha256':sha((O/'tensor-build/build.json').read_bytes()),
        'tensor_execution_report_sha256':sha((O/'tensor-execution/report.json').read_bytes()),
        'tensor_native_cases':11,'tensor_native_response_frames':161,'tensor_independent_response_frames':16,
        'all_tool_and_source_bytes_rechecked_after':True,'real_pytorch_oracle_executed_by_this_driver':False,
        'scope':'Fixed producer ABI3 native stack admission and guarded execution, including the complete current 151-function tensor core; not a full PyTorch rewrite or foreign-code verifier.'}
    (O/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='package_source_sha256'},indent=2))


if __name__=='__main__':main()
