#!/usr/bin/env python3
"""Replay frozen RN tests with an explicit, source-bound admission correction.

The historical RN03 contract report is preserved separately. Two cases that
are valid under the actual CPU double predicate become above-midpoint negative
controls; no value, rounding, state, lifetime or numerical comparison changes.
The newly accepted cases have their own exact C++/integer/Torch-observation gate.
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

SHA=lambda b:hashlib.sha256(b).hexdigest()
MANIFEST='5f0833b00e8f3068da6b60308f9941b7319c77793d10f90df5d779f520e9437b'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('rng-package','build','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();package=a.rng_package.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=(package/'source-pins.sha256').read_bytes();assert SHA(manifest)==MANIFEST
    pins=dict(line.split('  ',1)[::-1] for line in manifest.decode().splitlines())
    for name,pin in pins.items():assert SHA((package/name).read_bytes())==pin,name
    source=(package/'qualify.py').read_text()
    before='(0xff7fffff,0x3f800000),(0xbf800000,0x7f7fffff)'
    after='(0xff7fffff,0x64800001),(0xe4800001,0x7f7fffff)'
    assert source.count(before)==1;changed=source.replace(before,after)
    helper=out/'source';helper.mkdir()
    for name in ('protocol.py','owner_client.py','reference.py','reference.cpp','reference_fp.py'):
        (helper/name).write_bytes((package/name).read_bytes())
    (helper/'qualify.py').write_text(changed)
    report={'schema':'heptatorch.tensor-rng-contract-replay-adaptation.v1','status':'RUNNING',
        'original_test_sha256':pins['qualify.py'],'adapted_test_sha256':SHA(changed.encode()),
        'frozen_manifest_sha256':MANIFEST,'exact_edit':{'before':before,'after':after,'occurrences':1},
        'changed_expected_error_status':False,'changed_numerical_tolerance':False,
        'rationale':'Replace two newly accepted boundary inputs by the next binary32 value above 2^74; separate range gate checks every newly accepted family.'}
    def save():(out/'adaptation.json').write_text(json.dumps(report,indent=2)+'\n')
    save();r=subprocess.run([sys.executable,'-B',str(helper/'qualify.py'),'--build',str(a.build.resolve()),
        '--torch-observation',str(package/'evidence/pytorch-uniform-probe.json'),'--output',str(out/'native')],capture_output=True)
    (out/'stdout').write_bytes(r.stdout);(out/'stderr').write_bytes(r.stderr)
    report.update(exit_code=r.returncode,status='PASS' if r.returncode==0 else 'FAIL');save()
    assert r.returncode==0,r.stderr.decode(errors='replace')
    for name,pin in pins.items():assert SHA((package/name).read_bytes())==pin,name
    print(json.dumps(report))

if __name__=='__main__':main()
