"""Execute unchanged fixed I64 numerical recipes on the new mixed AD image."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SHA=lambda raw:hashlib.sha256(raw).hexdigest()
HERE=Path(__file__).resolve().parent
SOURCE='5f8ab1f7ca1593dc21b8b6a3537e210cfbc42b37ddefcb322239c5a316d88c49'
IMAGE='69045063c00395d07fc6c0101ed1b8aba18376cb1daea54e0a01c2c5caeb49b6'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('build','repo','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    build,repo=a.build.resolve(),a.repo.resolve()
    native,source=build/'tensor-mixed-autograd-session',build/'runtime.hepta'
    assert SHA(native.read_bytes())==IMAGE and SHA(source.read_bytes())==SOURCE
    tests=HERE/'integer-adapters';refs=repo/'validation/integer-tensors/reference'
    protocol=repo/'tests/integer_tensor_review/wire/protocol.py'
    custody=json.loads((tests/'custody.json').read_bytes())
    inputs={p.resolve():p.read_bytes() for p in [Path(__file__),tests/'custody.json',native,source,build/'build.json',protocol,protocol.with_name('owner_client.py')]}
    for row in custody['rows']:
        old=repo/row['original_path'];new=tests/row['path'];before=old.read_bytes();after=new.read_bytes()
        assert SHA(before)==row['original_sha256'] and SHA(after)==row['new_sha256'];inverse=after
        for e in reversed(row['edits']):assert inverse.count(e['after'].encode())==e['count'];inverse=inverse.replace(e['after'].encode(),e['before'].encode())
        assert inverse==before;inputs[old.resolve()]=before;inputs[new.resolve()]=after
    steps=[]
    for name,script,reference,counts in (
        ('storage12','replay_storage12_v2.py','torch-int64-tensor-reference.json',{'cases':12,'passed':12,'failed':0,'frames':235}),
        ('arithmetic30','replay_arithmetic30.py','torch-int64-tensor-reference.json',{'cases':30,'passed':30,'failed':0,'frames':1122}),
        ('dropout12','replay_dropout12.py','torch-int64-tensor-reference.json',{'cases':12,'passed':12,'failed':0,'frames':333}),
        ('alias8','replay_alias8_v2.py','torch-int64-alias-reference.json',{'cases':8,'passed':8,'native_frames':516,'exact_checks':2792}),
        ('RNE1237','replay_rne_v2.py','torch-int64-f32-rne.json',{'cases':5,'integers':1237,'native_frames':60,'exact_checks':1282}),
        ('promotion20','replay_promotion20.py','torch-int64-promotion-reference.json',{'cases':20,'passed':20,'original_mixed_operations':40,'native_frames':2280,'exact_checks':10580}),
        ('view-base6','replay_view_base6.py','torch-int64-view-base-lifetime.json',{'cases':6,'passed':6,'native_frames':450,'exact_checks':2046,'native_role_checks':168})):
        path=refs/reference;inputs[path.resolve()]=path.read_bytes()
        cmd=[sys.executable,'-B',str(tests/script),'--binary',str(native),'--source',str(source),'--protocol',str(protocol),'--reference',str(path),'--output',str(out/name)]
        with (out/(name+'.stdout')).open('wb') as so,(out/(name+'.stderr')).open('wb') as se:
            r=subprocess.run(cmd,stdout=so,stderr=se,timeout=900)
        report=out/name/'report.json';j=json.loads(report.read_bytes()) if report.exists() else None
        passed=r.returncode==0 and j is not None and j['status']=='PASS' and all(j[k]==v for k,v in counts.items())
        steps.append({'name':name,'command':cmd,'exit_code':r.returncode,'status':'PASS' if passed else 'FAIL','report_sha256':SHA(report.read_bytes()) if report.exists() else None,'expected_counts':counts})
        (out/'progress.json').write_text(json.dumps(steps,indent=2)+'\n')
    script=repo/'tests/int64_inplace_review/replay.py';reference=repo/'validation/int64-inplace-review/fixed-reference40.json'
    inputs[script.resolve()]=script.read_bytes();inputs[reference.resolve()]=reference.read_bytes()
    cmd=[sys.executable,'-B',str(script),'--binary',str(native),'--source',str(source),'--protocol',str(protocol),'--reference',str(reference),'--observer',str(tests/'native_observer.py'),'--expected-source-sha256',SOURCE,'--expected-native-sha256',IMAGE,'--output',str(out/'inplace40')]
    with (out/'inplace40.stdout').open('wb') as so,(out/'inplace40.stderr').open('wb') as se:r=subprocess.run(cmd,stdout=so,stderr=se,timeout=900)
    report=out/'inplace40/report.json';j=json.loads(report.read_bytes()) if report.exists() else None
    steps.append({'name':'inplace40','command':cmd,'exit_code':r.returncode,'status':'PASS' if r.returncode==0 and j and j['status']=='PASS' and j['cases']==j['passed']==40 else 'FAIL','report_sha256':SHA(report.read_bytes()) if report.exists() else None})
    assert all(p.read_bytes()==raw for p,raw in inputs.items())
    result={'schema':'heptatorch.mixed08-inherited-int64-eight-gates.v1','status':'PASS' if all(x['status']=='PASS' for x in steps) else 'FAIL','source_sha256':SOURCE,'native_sha256':IMAGE,'steps':steps,'all_inputs_unchanged':True,'inputs_sha256':{str(p):SHA(raw) for p,raw in inputs.items()},'native_executed':True,'Torch_executed_here':False,'full_pytorch_rewrite_complete':False,'pytorch_api_denominator':2590}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('steps','inputs_sha256')}));return int(result['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
