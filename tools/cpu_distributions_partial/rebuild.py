#!/usr/bin/env python3
"""Reproduce explicit-partial HIA units from the frozen atomic predecessor."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
SHA=lambda raw:hashlib.sha256(raw).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','upstream','stack-build','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();inputs={}
    for line in (HERE/'source-pins.sha256').read_text().splitlines():
        digest,name=line.split('  ',1);path=(HERE/name).resolve();assert path.is_relative_to(HERE)
        raw=path.read_bytes();assert SHA(raw)==digest,path;inputs[path]=raw
    profile=json.loads(inputs[HERE/'profile.json'])
    baseline=a.baseline.resolve();authoring=baseline/'overlay-authoring.py'
    assert SHA(authoring.read_bytes())==profile['baseline_authoring_sha256']
    command=[sys.executable,'-I','-S','-B',str(HERE/'build.py'),'--baseline',str(baseline),
             '--common-authoring',str(authoring),'--gradient-authoring',str(authoring),
             '--upstream',str(a.upstream.resolve()),'--stack-build',str(a.stack_build.resolve()),
             '--output',str(a.output.resolve())]
    subprocess.run(command,check=True)
    report=json.loads((a.output/'build.json').read_text())
    for key in ('runtime_source_sha256','executable_sha256','compiler_sha256','linker_sha256',
                'core_functions','core_blocks','core_instructions','base_function_body_map'):
        assert report[key]==profile[key],key
    fields=('name','source_sha256','object_sha256')
    assert [{k:r[k] for k in fields} for r in report['units']]==[{k:r[k] for k in fields} for r in profile['units']]
    assert report['all_predecessor_dist_functions_byte_identical']
    assert report['old_atomic_commands_80_85_unchanged'] and not report['aggregate_core_host_check']
    for path,raw in inputs.items():assert path.read_bytes()==raw,path
    receipt={'schema':'heptatorch.cpu-distributions-partial-reproduction.v1','status':'PASS',
             'source_pins_sha256':SHA((HERE/'source-pins.sha256').read_bytes()),
             'runtime_source_sha256':report['runtime_source_sha256'],
             'executable_sha256':report['executable_sha256'],'native_units':len(report['units']),
             'all_source_object_image_bytes_identical':True,'old_atomic_function_bodies_identical':True,
             'source_custody_after':True}
    (a.output/'reproduction.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
