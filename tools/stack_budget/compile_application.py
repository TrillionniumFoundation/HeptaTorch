#!/usr/bin/env python3
"""Compile unchanged HIA application units under the distinct guarded ABI3 profile."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from inspect_stack import inspect,longest

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--sources',type=Path,required=True)
p.add_argument('--native',type=Path,required=True)
p.add_argument('--baseline-build',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('units',type=Path,nargs='+')
a=p.parse_args();S=a.sources.resolve();N=a.native.resolve();B=a.baseline_build.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=False)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
pins=json.loads((N/'bootstrap.json').read_bytes());adapt=json.loads((S/'adaptation.json').read_bytes())
C=N/'stack-module-compiler';L=N/'stack-linker-session'
assert sha(C.read_bytes())==pins['compiler_sha256'] and sha(L.read_bytes())==pins['linker_sha256']['session']
snapshots={path:path.read_bytes() for path in [C,L,B/'build.json',B/'runtime.hepta',B/'owned-tensor-session',*a.units]}
baseline=json.loads(snapshots[B/'build.json'])
assert sha(snapshots[B/'owned-tensor-session'])==baseline['executable_sha256']
assert sha(snapshots[B/'runtime.hepta'])==baseline['candidate_source_sha256']
baseline_units={row['name']:row for row in baseline['unit_rows']}
(O/'units').mkdir();objects=[];views=[];rows=[];definitions={}

def text(raw):
    table=struct.unpack_from('<Q',raw,40)[0]
    offset,size=struct.unpack_from('<2Q',raw,table+64+24)
    return raw[offset:offset+size]

for source in a.units:
    name=source.stem;raw=snapshots[source]
    assert sha(raw)==baseline_units[name]['source_sha256'],name
    baseline_source=B/'units'/source.name
    snapshots[baseline_source]=baseline_source.read_bytes()
    assert raw==snapshots[baseline_source],name
    (O/'units'/source.name).write_bytes(raw)
    result=subprocess.run([str(C)],input=raw,capture_output=True,env={'PATH':''},timeout=60)
    (O/(name+'.compiler.stdout')).write_bytes(result.stdout);(O/(name+'.compiler.stderr')).write_bytes(result.stderr)
    assert result.returncode==0 and not result.stderr,(name,result.returncode)
    (O/(name+'.o')).write_bytes(result.stdout);objects.append(result.stdout)
    view=inspect(raw,result.stdout);views.append(view)
    old_object=(B/(name+'.o')).read_bytes();snapshots[B/(name+'.o')]=old_object
    assert sha(old_object)==baseline_units[name]['object_sha256'],name
    assert text(old_object)==text(result.stdout),name
    local=[]
    for match in re.finditer(rb'^fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]*entry\([^\n]*\n.*?^end\n',raw,re.M|re.S):
        function=match.group(1).decode();body=match.group(0)
        assert function not in definitions,function
        definitions[function]={'unit':name,'sha256':sha(body)};local.append(function)
    rows.append({'name':name,'source_sha256':sha(raw),'object_sha256':sha(result.stdout),
                 'baseline_object_sha256':sha(old_object),'machine_text_sha256':sha(text(result.stdout)),
                 'machine_text_byte_identical_to_baseline':True,'definitions':local,'inspection':view})
core={m.group(1).decode():m.group(0) for m in re.finditer(rb'^fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]*entry\([^\n]*\n.*?^end\n',snapshots[B/'runtime.hepta'],re.M|re.S)}
for name,body in core.items():assert definitions[name]['sha256']==sha(body),name
assert len(core)==151,len(core)
request=b'HIASTKS1'+struct.pack('<3Q',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(raw))+raw for raw in objects)
(O/'link.request.bin').write_bytes(request)
result=subprocess.run([str(L)],input=request,capture_output=True,env={'PATH':''},timeout=60)
(O/'link.stdout').write_bytes(result.stdout);(O/'link.stderr').write_bytes(result.stderr)
assert result.returncode==0 and not result.stderr,result.returncode
app=O/'owned-tensor-session';app.write_bytes(result.stdout);app.chmod(0o700)
assert result.stdout[4096:]==snapshots[B/'owned-tensor-session'][4096:]
budget=longest(views)
startup=json.loads((S/'linker-session/startup.json').read_bytes())
assert struct.unpack_from('<Q',result.stdout,startup['stack']['budget_hole'])[0]==budget['required_stack_bytes']
for path,raw in snapshots.items():assert path.read_bytes()==raw,path
report={'schema':'heptatorch.stack-budget-current-tensor-build.v1','pass':True,
        'compiler_sha256':pins['compiler_sha256'],'linker_sha256':pins['linker_sha256']['session'],
        'executable_sha256':sha(result.stdout),'baseline_executable_sha256':baseline['executable_sha256'],
        'baseline_build_sha256':sha(snapshots[B/'build.json']),
        'runtime_source_sha256':sha(snapshots[B/'runtime.hepta']),
        'candidate_source_sha256':sha(snapshots[B/'runtime.hepta']),
        'canonical_source_sha256':baseline['canonical_source_sha256'],
        'core_functions_unchanged':len(core),'all_functions_source_mapping':definitions,
        'all_unit_sources_and_function_bodies_unchanged':True,'unit_rows':rows,
        'linked_application_text_after_startup_byte_identical':True,'budget':budget,
        'startup_prefix_sha256':adapt['linkers']['session']['startup_prefix_sha256'],
        'builder_sha256':sha(Path(__file__).read_bytes()),'producer_and_source_custody_rechecked':True,
        'native_execution_performed':False,'full_PyTorch_closure_claimed':False}
(O/'build.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('all_functions_source_mapping','unit_rows','budget')},indent=2))
print(json.dumps({k:v for k,v in budget.items() if k!='per_function_path_bytes'},indent=2))
