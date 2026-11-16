#!/usr/bin/env python3
"""Run both unchanged atomic and explicit-partial interfaces against fixed facts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
SHA=lambda raw:hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build','baseline','atomic-tools','stack-build','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    inputs={}
    def verify(directory):
        manifest=directory/'source-pins.sha256';inputs[manifest]=manifest.read_bytes()
        for line in inputs[manifest].decode().splitlines():
            digest,name=line.split('  ',1);path=(directory/name).resolve()
            assert path.is_relative_to(directory);raw=path.read_bytes()
            assert SHA(raw)==digest,path;inputs[path]=raw
    verify(HERE);profile=json.loads(inputs[HERE/'profile.json'])
    atomic=a.atomic_tools.resolve()
    assert SHA((atomic/'source-pins.sha256').read_bytes())==profile['predecessor_source_pins_sha256']
    verify(atomic)
    build=a.build.resolve();baseline=a.baseline.resolve();stack=a.stack_build.resolve()
    image=build/'tensor-distributions-session'
    for path,digest in [(image,profile['executable_sha256']),(build/'runtime.hepta',profile['runtime_source_sha256']),
                        (baseline/'runtime.hepta',profile['baseline_source_sha256'])]:
        inputs[path]=path.read_bytes();assert SHA(inputs[path])==digest
    common=['--reference',atomic/'evidence/torch-bernoulli-dropout.json','--protocol',atomic/'protocol.py',
            '--executable',image,'--build-report',build/'build.json']
    suites=[('atomic-reference',atomic/'replay_reference.py',common),
            ('explicit-reference',HERE/'replay_explicit.py',common),
            ('four-late-errors',HERE/'replay_late_errors.py',common),
            ('native-controls',HERE/'test_controls.py',['--build',build,'--atomic-tools',atomic]),
            ('owned-entry',HERE/'test_entry.py',['--build',build,'--atomic-tools',atomic,'--stack-build',stack]),
            ('typed-handoffs',HERE/'test_handoffs.py',['--build',build,'--rng-package',atomic,
               '--stack-build',stack,'--peer-pins',HERE/'handoff-peer-pins.json','--peer','atomic03='+str(baseline)])]
    rows=[]
    for name,script,args in suites:
        destination=out/name
        result=subprocess.run([sys.executable,'-B',str(script),*map(str,args),'--output',str(destination)],
                              capture_output=True,timeout=300)
        (out/(name+'.stdout')).write_bytes(result.stdout);(out/(name+'.stderr')).write_bytes(result.stderr)
        row={'name':name,'exit_code':result.returncode};rows.append(row)
        assert result.returncode==0,(name,result.returncode,result.stderr[-3000:])
        raw=(destination/'report.json').read_bytes();report=json.loads(raw)
        row.update(status=report['status'],report_sha256=SHA(raw))
        if name=='atomic-reference':
            assert report['status']=='PARTIAL' and report['counts']=={'PASS':161,'OPEN_TRANSACTION_DIFFERENCE':4,'OPEN_UNREPRESENTED':18}
        elif name=='explicit-reference':
            assert report['status']=='PARTIAL' and report['counts']=={'PASS':165,'OPEN_UNREPRESENTED':18}
        else:assert report['status']=='PASS'
    for path,raw in inputs.items():assert path.read_bytes()==raw,path
    result={'schema':'heptatorch.cpu-distributions-partial-qualification.v1','status':'PARTIAL',
            'native_scoped_qualification_passed':True,'suites':rows,
            'runtime_source_sha256':profile['runtime_source_sha256'],'executable_sha256':profile['executable_sha256'],
            'original_actual_torch_observations':183,'explicit_strict_mapped_passes':165,
            'original_atomic_strict_mapped_passes':161,'original_atomic_transaction_differences':4,
            'explicit_late_error_differences_remaining':0,'open_unrepresented':18,
            'all_rng_complete':False,'all_tensor_dtypes_complete':False,'full_pytorch_rewrite_complete':False,
            'source_custody_after':True}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
