#!/usr/bin/env python3
"""Reproduce the separate ABI3 compiler and guarded-stack native linkers."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def main():
    spec=Path(__file__).resolve().parent
    repo=spec.parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--owned-build',type=Path,required=True)
    p.add_argument('--session-build',type=Path,required=True)
    p.add_argument('--borrowed-tools',type=Path,required=True)
    p.add_argument('--owned-profile',type=Path,default=repo/'tools/owned_modules/profile.json')
    p.add_argument('--session-profile',type=Path,default=repo/'tools/owned_sessions/profile.json')
    p.add_argument('--session-template',type=Path,default=repo/'tools/owned_sessions/wrapper.py')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sha=lambda raw:hashlib.sha256(raw).hexdigest()
    identity={}
    for line in (spec/'source-pins.sha256').read_text().splitlines():
        expected,relative=line.split('  ',1)
        assert sha((spec/relative).read_bytes())==expected,relative
        identity[relative]=expected
    profile=json.loads((spec/'profile.json').read_bytes())
    sources=out/'sources';native=out/'native'

    def run(label,script,arguments):
        result=subprocess.run([sys.executable,'-I','-S','-B',str(spec/script),*map(str,arguments)],capture_output=True,timeout=240)
        (out/(label+'.stdout')).write_bytes(result.stdout);(out/(label+'.stderr')).write_bytes(result.stderr)
        assert result.returncode==0 and not result.stderr,(label,result.returncode,result.stderr[-2000:])

    run('adapt','adapt.py',['--source-root',a.source_root.resolve(),'--owned-build',a.owned_build.resolve(),
        '--owned-profile',a.owned_profile.resolve(),'--session-build',a.session_build.resolve(),
        '--session-profile',a.session_profile.resolve(),'--session-template',a.session_template.resolve(),'--output',sources])
    adaptation=json.loads((sources/'adaptation.json').read_bytes())
    assert adaptation['compiler']['source_sha256']==profile['compiler']['source_sha256']
    for name,row in profile['linkers'].items():
        for key in ('source_sha256','authoring_sha256','request_magic','stack_bytes','stack_budget','workspace_bytes','startup_prefix_sha256','startup_authoring_sha256'):
            assert adaptation['linkers'][name][key]==row[key],(name,key)
    run('bootstrap','bootstrap.py',['--sources',sources,'--owned-tools',a.owned_build.resolve()/'native',
        '--borrowed-tools',a.borrowed_tools.resolve(),'--output',native])
    bootstrap=json.loads((native/'bootstrap.json').read_bytes())
    assert bootstrap['status']=='PASS' and bootstrap['compiler_sha256']==profile['compiler']['executable_sha256']
    assert bootstrap['abi3_successive_compilers_byte_identical'] and bootstrap['abi3_successive_linkers_byte_identical']
    assert sha((native/'stack-module-compiler').read_bytes())==profile['compiler']['executable_sha256']
    for name,row in profile['linkers'].items():
        assert bootstrap['linker_sha256'][name]==row['executable_sha256']
        assert sha((native/('stack-linker-'+name)).read_bytes())==row['executable_sha256']
    for relative,expected in identity.items():assert sha((spec/relative).read_bytes())==expected,relative
    report={'schema':'heptatorch.stack-budget-portable-rebuild.v1','status':'PASS',
        'package_source_sha256':identity,'profile_sha256':sha((spec/'profile.json').read_bytes()),
        'compiler_sha256':bootstrap['compiler_sha256'],'linker_sha256':bootstrap['linker_sha256'],
        'adaptation_sha256':sha((sources/'adaptation.json').read_bytes()),'bootstrap_sha256':sha((native/'bootstrap.json').read_bytes()),
        'native_compilation_and_linking':True,'successive_native_generations_byte_identical':True,
        'old_profiles_unchanged':True,'host_numerical_or_owner_runtime':False}
    (out/'rebuild.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='package_source_sha256'},indent=2))


if __name__=='__main__':main()
