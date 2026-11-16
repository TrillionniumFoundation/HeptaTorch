#!/usr/bin/env python3
"""Package exact native objects for the independently checked owned entry gate."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

LINKER_SHA='55a2b0a762c8db574a65d0d0e6a18e3cedc668c01d20080a11c26cf1ccb601b3'
COMPILER_SHA='0ccdb524ea327ab290b2e70feaac49a4789abb2fee6da8746a095f435c17af73'
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--modules',type=Path,required=True)
    p.add_argument('--linker',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    linker=a.linker.resolve()
    assert sha(linker.read_bytes())==LINKER_SHA
    build_raw=(a.modules/'build.json').read_bytes();build=json.loads(build_raw)
    assert build['status']=='PASS' and build['compiler_sha256']==COMPILER_SHA
    objects,identities=[],{}
    for name in ['bytes','records','audit','ops','owners']:
        source=(a.modules/(name+'.hepta')).read_bytes()
        obj=(a.modules/(name+'.o')).read_bytes()
        row=build['modules'][name]
        assert sha(source)==row['source_sha256'] and sha(obj)==row['object_sha256']
        objects.append(obj)
        identities[name]={'source_sha256':sha(source),'object_sha256':sha(obj)}
    request=b'HIAOWNS1'+struct.pack('<3Q',len(objects),4,0)+b'step'
    request+=b''.join(struct.pack('<Q',len(o))+o for o in objects)
    (a.output/'link-request.bin').write_bytes(request)
    result=subprocess.run([str(linker)],input=request,capture_output=True,env={'PATH':''},timeout=60)
    (a.output/'link.stdout').write_bytes(result.stdout)
    (a.output/'link.stderr').write_bytes(result.stderr)
    report={'schema':'heptatorch.storage-object-owned-session-build.v1','status':'FAIL',
            'compiler_sha256':COMPILER_SHA,'linker_sha256':LINKER_SHA,'module_identities':identities,
            'module_build_receipt_sha256':sha(build_raw),'link_request_sha256':sha(request),
            'exit_code':result.returncode,'execution_qualification_claimed':False,
            'host_performs_native_linking':False}
    (a.output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
    native=a.output/'storage-session';native.write_bytes(result.stdout);native.chmod(0o700)
    report.update(status='PASS',executable_sha256=sha(result.stdout),executable_bytes=len(result.stdout))
    (a.output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
