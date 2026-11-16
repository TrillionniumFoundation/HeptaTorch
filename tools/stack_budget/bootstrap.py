#!/usr/bin/env python3
"""Bootstrap ABI3 source units and reproduce guarded native tools."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import struct
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sources',type=Path,required=True)
parser.add_argument('--owned-tools',type=Path,required=True)
parser.add_argument('--borrowed-tools',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
S=args.sources.resolve(); OWN=args.owned_tools.resolve(); OLD=args.borrowed_tools.resolve()
O=args.output.resolve(); O.mkdir(parents=True,exist_ok=False)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
expected={'owned-module-compiler':'0ccdb524ea327ab290b2e70feaac49a4789abb2fee6da8746a095f435c17af73',
          'linker-abi2-w2m':'da68e6fa9c360931981f97f46a29668bb0864ead17fc266ce750f72b741e725c'}
tools={OWN/name:pin for name,pin in expected.items()}
tools[OLD/'module-compiler']='e250d93bc0b2a5ff878d2188c95780abf7e7ebaf186e4531b136fa2735b2af59'
tools[OLD/'linker-packer']='98f0181dd509ffeb09679c6602751c86c2b972e22c5f1b9c3fd26a1e677f972b'
for path,pin in tools.items():assert sha(path.read_bytes())==pin,path
adapt=json.loads((S/'adaptation.json').read_text())
source_snapshots={path:path.read_bytes() for path in S.rglob('*') if path.is_file()}


def native(tool,request,destination):
    process=subprocess.run([str(tool)],input=request,capture_output=True,env={'PATH':''},timeout=60)
    destination.with_suffix(destination.suffix+'.stderr').write_bytes(process.stderr)
    destination.with_suffix(destination.suffix+'.exit').write_text(str(process.returncode)+'\n')
    assert process.returncode==0 and not process.stderr and process.stdout,(destination,process.returncode,process.stderr)
    destination.write_bytes(process.stdout)
    return process.stdout


def link(tool,magic,entry,objects,directory,name):
    request=magic+struct.pack('<3Q',len(objects),len(entry),0)+entry
    request+=b''.join(struct.pack('<Q',len(obj))+obj for obj in objects)
    (directory/(name+'.request')).write_bytes(request)
    target=directory/name
    raw=native(tool,request,target);target.chmod(0o700)
    return target,{'request_sha256':sha(request),'object_sha256':[sha(obj) for obj in objects],
                   'executable_sha256':sha(raw),'producer_linker_sha256':sha(tool.read_bytes())}


bootstrap_linkers={}
initial=O/'legacy-flat-linkers';initial.mkdir()
for label,row in adapt['linkers'].items():
    raw=(S/('linker-'+label)/'source.hepta').read_bytes();assert sha(raw)==row['source_sha256']
    image=native(OWN/'owned-module-compiler',raw,initial/(label+'.image'))
    baseline=native(OLD/'module-compiler',raw,initial/(label+'.baseline-image'))
    assert image==baseline
    target=initial/label
    native(OLD/'linker-packer',b'HIAFLT01'+struct.pack('<Q',len(image))+image,target)
    target.chmod(0o700);bootstrap_linkers[label]=target

raw=(S/'compiler/source.hepta').read_bytes();assert sha(raw)==adapt['compiler']['source_sha256']
names=json.loads((S/'compiler/functions.json').read_text())
matches=list(re.finditer(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]*\n.*?^end\n)',raw,re.M|re.S))
functions={match.group(2).decode():match.group(1) for match in matches}
assert len(functions)==len(matches)==len(names)
assert raw[:matches[0].start()]==b'hia-asm.v1 hepta.hia.program.v8 selfhost\n'
assert all(not raw[left.end():right.start()].strip() for left,right in zip(matches,matches[1:]))
assert not raw[matches[-1].end():].strip()
callees={name:set(match.decode() for match in re.findall(rb'\{"function":"([A-Za-z_][A-Za-z_0-9]*)"\}',body)) for name,body in functions.items()}
kernel={names[name] for name in ('operation_dispatch','owned_instruction','fix_all','exports_finish',
                               'native_allocator_helpers','chain_drop_helper','cli_finish','function_start','wrapper')}
while True:
    extended=kernel|set().union(*(callees[name] for name in kernel))
    if extended==kernel:break
    kernel=extended
frontend=set(functions)-kernel
imports=set().union(*(callees[name] for name in frontend))&kernel
ordered=lambda choice:[name for name in functions if name in choice]
units=[]
for label,local,external,exports in [('frontend',frontend,imports,{names['compile']}),('backend',kernel,set(),imports)]:
    unit=('hia-asm.v1 hepta.hia.program.v8 stack_compiler_'+label+'\nexports('+','.join(ordered(exports))+')\n').encode()
    for name in ordered(external):
        header,count=re.subn(rb'entry\([^)]*\)$',b'external()',functions[name].split(b'\n')[0])
        assert count==1 and b'return_loans' not in header
        unit+=header+b'\n'
    unit+=b''.join(functions[name] for name in ordered(local))
    (O/(label+'.hepta')).write_bytes(unit)
    counts={'definitions':len(local),'external_declarations':len(external),'functions':len(local)+len(external),
            'blocks':len(re.findall(rb'^block ',unit,re.M)),'instructions':unit.count(b' = '),'bytes':len(unit)}
    assert counts['functions']<=256 and counts['blocks']<=2560 and counts['instructions']<=8192 and len(unit)<=1048576
    units.append({'name':label,'source_sha256':sha(unit),**counts,
                  'unchanged_function_body_sha256':{name:sha(functions[name]) for name in ordered(local)}})
assert set(frontend)|kernel==set(functions) and not frontend&kernel
partition={'source_sha256':sha(raw),'function_count':len(functions),'units':units,
           'complete_original_function_bodies':True,'per_unit_limits_unchanged':True}
(O/'partition.json').write_text(json.dumps(partition,indent=2)+'\n')


def compiler_generation(previous,linker,magic,label):
    directory=O/label;directory.mkdir()
    objects=[native(previous,(O/(unit['name']+'.hepta')).read_bytes(),directory/(unit['name']+'.o')) for unit in units]
    executable,row=link(linker,magic,names['compile'].encode(),objects,directory,'compiler')
    row.update(label=label,compiler_producer_sha256=sha(previous.read_bytes()))
    print(json.dumps(row),flush=True)
    return executable,row


compiler,stage0=compiler_generation(OWN/'owned-module-compiler',OWN/'linker-abi2-w2m',b'HIAOWNC2','stage0-abi2')
generations=[stage0]
for number in (1,2):
    compiler,row=compiler_generation(compiler,bootstrap_linkers['compiler'],b'HIASTKC1','stage'+str(number)+'-abi3')
    generations.append(row)
assert generations[1]['executable_sha256']==generations[2]['executable_sha256']

linker_generations=[]
final_linkers={}
prior_tool_linker=bootstrap_linkers['linker']
for generation in (1,2):
    directory=O/('linkers-generation-'+str(generation));directory.mkdir()
    rows=[]
    current={}
    for label in adapt['linkers']:
        source=(S/('linker-'+label)/'source.hepta').read_bytes()
        first,rest=source.split(b'\n',1)
        module=first+b'\nexports(f0)\n'+rest
        (directory/(label+'.hepta')).write_bytes(module)
        obj=native(compiler,module,directory/(label+'.o'))
        executable,row=link(prior_tool_linker,b'HIASTKL1',b'f0',[obj],directory,label)
        row.update(label=label,module_source_sha256=sha(module),compiler_producer_sha256=sha(compiler.read_bytes()))
        rows.append(row);current[label]=executable
    if generation==2:
        for label,path in current.items():
            assert path.read_bytes()==final_linkers[label].read_bytes(),label
    final_linkers=current;prior_tool_linker=current['linker'];linker_generations.append(rows)

compiler,row=compiler_generation(compiler,final_linkers['compiler'],b'HIASTKC1','stage3-abi3-final-linker')
generations.append(row)
assert len({row['executable_sha256'] for row in generations[1:]})==1
shutil.copy2(compiler,O/'stack-module-compiler')
for label,path in final_linkers.items():shutil.copy2(path,O/('stack-linker-'+label))
for path,pin in tools.items():assert sha(path.read_bytes())==pin,path
for path,snapshot in source_snapshots.items():assert path.read_bytes()==snapshot,path
report={'schema':'heptatorch.stack-budget-native-bootstrap.v1','status':'PASS','interface_abi':3,
        'adaptation_sha256':sha((S/'adaptation.json').read_bytes()),'partition':partition,
        'compiler_generations':generations,'linker_generations':linker_generations,
        'abi3_successive_compilers_byte_identical':True,'abi3_successive_linkers_byte_identical':True,
        'compiler_sha256':sha(compiler.read_bytes()),'linker_sha256':{label:sha(path.read_bytes()) for label,path in final_linkers.items()},
        'all_final_tools_have_native_admitted_guarded_private_stacks':True,
        'legacy_bootstrap_tool_identities':{path.name:pin for path,pin in tools.items()},
        'producer_source_bytes_rechecked_after':True,'host_owner_or_numerical_runtime':False}
(O/'bootstrap.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('partition','compiler_generations','linker_generations')},indent=2))
