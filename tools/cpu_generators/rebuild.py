#!/usr/bin/env python3
"""Verify source custody and reproduce the native Generator executable."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,required=True);p.add_argument('--stack-build',type=Path,required=True)
    p.add_argument('--rng-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sha=lambda raw:hashlib.sha256(raw).hexdigest();identity={}
    for line in (root/'source-pins.sha256').read_text().splitlines():
        expected,relative=line.split('  ',1);assert sha((root/relative).read_bytes())==expected,relative;identity[relative]=expected
    result=subprocess.run([sys.executable,'-I','-S','-B',str(root/'build.py'),'--upstream',str(a.upstream.resolve()),
        '--stack-build',str(a.stack_build.resolve()),'--rng-source',str(a.rng_source.resolve()),'--output',str(a.output.resolve())],
        capture_output=True,timeout=180)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    (out/'rebuild.stdout').write_bytes(result.stdout);(out/'rebuild.stderr').write_bytes(result.stderr)
    assert result.returncode==0 and not result.stderr,(result.returncode,result.stderr[-3000:])
    for relative,expected in identity.items():assert sha((root/relative).read_bytes())==expected,relative
    report=json.loads((out/'build.json').read_bytes());assert report['status']=='PASS'
    receipt={'schema':'heptatorch.cpu-generator-portable-rebuild.v1','status':'PASS','package_source_sha256':identity,
             'build_receipt_sha256':sha((out/'build.json').read_bytes()),'executable_sha256':report['executable_sha256'],
             'generator_linker_sha256':report['generator_linker_sha256'],'native_compile_and_link':True,
             'host_production_rng_or_owner_runtime':False}
    (out/'rebuild.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
