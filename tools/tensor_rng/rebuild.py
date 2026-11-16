#!/usr/bin/env python3
"""Check this frozen adaptation, rebuild all native modules, compare exact pins."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
SHA=lambda b:hashlib.sha256(b).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--rng-source',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True);p.add_argument('--stack-build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    pins={}
    for line in (HERE/'source-pins.sha256').read_text().splitlines():
        pin,name=line.split('  ',1);path=(HERE/name).resolve();assert path.is_relative_to(HERE)
        assert len(pin)==64 and SHA(path.read_bytes())==pin,name;pins[path]=pin
    profile=json.loads((HERE/'profile.json').read_text())
    subprocess.run([sys.executable,'-I','-S','-B',str(HERE/'build.py'),
        '--baseline',str(a.baseline.resolve()),'--rng-source',str(a.rng_source.resolve()),
        '--upstream',str(a.upstream.resolve()),'--stack-build',str(a.stack_build.resolve()),
        '--output',str(a.output.resolve())],check=True)
    report=json.loads((a.output/'build.json').read_text())
    for key in ('runtime_source_sha256','executable_sha256','compiler_sha256','linker_sha256'):
        assert report[key]==profile[key],key
    actual={x['name']:(x['source_sha256'],x['object_sha256']) for x in report['units']}
    expected={x['name']:(x['source_sha256'],x['object_sha256']) for x in profile['units']}
    assert actual==expected
    for path,pin in pins.items():assert SHA(path.read_bytes())==pin,path
    result={'schema':'heptatorch.tensor-rng-native-reproduction.v1','status':'PASS',
        'profile':profile['profile'],'profile_sha256':SHA((HERE/'profile.json').read_bytes()),
        'source_pins_sha256':SHA((HERE/'source-pins.sha256').read_bytes()),
        'executable_sha256':report['executable_sha256'],'runtime_source_sha256':report['runtime_source_sha256'],
        'modules':len(actual),'all_source_and_object_bytes_identical':True}
    (a.output/'reproduction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
