#!/usr/bin/env python3
"""Narrow independent zero/nonzero gradient merge layout counterexample.

Expected d[(p*c)+Bernoulli(p)]/dp with an all-one seed is exactly c.
Changing branch creation order cannot transpose this logical derivative.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import struct

SHA=lambda data:hashlib.sha256(data).hexdigest()
WIRE='7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
CLIENT='dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['binary','source','protocol','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-native-sha256',required=True)
    p.add_argument('--expected-source-sha256',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=a.source.read_bytes();binary=a.binary.read_bytes()
    assert SHA(source)==a.expected_source_sha256 and SHA(binary)==a.expected_native_sha256
    assert SHA(a.protocol.read_bytes())==WIRE and SHA(a.protocol.with_name('owner_client.py').read_bytes())==CLIENT
    script=Path(__file__).read_bytes();(a.output/'executed-test.py').write_bytes(script)
    spec=importlib.util.spec_from_file_location('zero_merge_wire',a.protocol)
    w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
    report={'schema':'heptatorch.independent-zero-gradient-merge-layout.v1',
            'status':'RUNNING','native_source_sha256':a.expected_source_sha256,
            'native_executable_sha256':a.expected_native_sha256,'test_sha256':SHA(script),
            'expected_derivation':'For every logical element, all-one seed * coefficient + exact Bernoulli zero = coefficient.',
            'expected_bits':[w.bits(i) for i in range(1,7)],'Torch_executed_here':False,
            'production_host_numerics':False,'cases':[]}
    for zero_first in [True,False]:
        name='zero-contribution-first' if zero_first else 'nonzero-contribution-first'
        s=w.Session(a.binary,directory=a.output/name/'transport')
        one=lambda *cmd:s.run([cmd])[0]
        base=one(*w.full((2,3),.5))['value']
        param=one(6,base,0,1)['value'];one(90,param,1)
        coeff=one(*w.tensor((3,2),[1,2,3,4,5,6]))['value']
        generator=one(60,17)['value']
        # Ready nonleaf nodes run in descending creation sequence. Create the
        # zero branch last to deliver the fresh strided zero buffer first.
        if zero_first:
            mul=one(2,3,param,coeff)['value'];bern=one(82,param,generator,0)['value']
        else:
            bern=one(82,param,generator,0)['value'];mul=one(2,3,param,coeff)['value']
        joined=one(2,1,mul,bern)['value']
        seedbase=one(*w.full((2,3),1))['value'];seed=one(6,seedbase,0,1)['value']
        one(57,joined,seed,0)
        grad=one(91,param)['value'];rec=one(39,grad,0,6)
        actual=list(struct.unpack('<6I',rec['payload']))
        row={'name':name,'status':'PASS' if actual==report['expected_bits'] else 'FAIL',
             'parameter_metadata':w.meta(one(13,param)),'seed_metadata':w.meta(one(13,seed)),
             'gradient_metadata':w.meta(one(13,grad)),'actual_bits':actual,
             'expected_bits':report['expected_bits'],'audit':one(53)['value'],'frames':s.frames}
        report['cases'].append(row);s.finish()
    assert a.source.read_bytes()==source and a.binary.read_bytes()==binary
    report.update(status='PASS' if all(x['status']=='PASS' for x in report['cases']) else 'FAIL',
                  frames=sum(x['frames'] for x in report['cases']),source_custody_after=True)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if report['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
