#!/usr/bin/env python3
"""Reproduce the frozen common native runtime from its exact component inputs."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
SHA=lambda b:hashlib.sha256(b).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('tensor09','partial-build','entry-authoring','rng-package','rng-source','upstream','stack-build','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();inputs={}
    for line in (HERE/'source-pins.sha256').read_text().splitlines():
        pin,name=line.split('  ',1);path=(HERE/name).resolve();assert path.is_relative_to(HERE)
        raw=path.read_bytes();assert SHA(raw)==pin;inputs[path]=raw
    profile=json.loads((HERE/'profile.json').read_text())
    argv=[sys.executable,'-I','-S','-B',str(HERE/'compose.py'),'--base-pin',str(HERE/'base-05.json')]
    for name in ('tensor09','partial-build','entry-authoring','rng-package','rng-source','upstream','stack-build','output'):
        argv += ['--'+name,str(getattr(a,name.replace('-','_')).resolve())]
    subprocess.run(argv,check=True)
    report=json.loads((a.output/'composition.json').read_text());build=json.loads((a.output/'native/build.json').read_text())
    for key in ('runtime_source_sha256','executable_sha256','compiler_sha256','linker_sha256'):
        assert report[key]==profile[key],key
    actual={x['name']:(x['source_sha256'],x['object_sha256']) for x in build['units']}
    expected={x['name']:(x['source_sha256'],x['object_sha256']) for x in profile['units']}
    assert actual==expected and report['entry_route_inverse_ast_identical']
    for path,raw in inputs.items():assert path.read_bytes()==raw,path
    result={'schema':'heptatorch.tensor-rng-common-reproduction.v1','status':'PASS','profile':profile['profile'],
        'source_pins_sha256':SHA((HERE/'source-pins.sha256').read_bytes()),'executable_sha256':report['executable_sha256'],
        'runtime_source_sha256':report['runtime_source_sha256'],'native_modules':len(actual),
        'all_source_and_object_bytes_identical':True,'preserved_parent_functions':report['unchanged_parent_functions']}
    (a.output/'reproduction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
