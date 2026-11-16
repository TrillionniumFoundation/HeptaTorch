"""Compose checked retained-gradient HIA bodies and compile the native owner.

The host verifies immutable HIA bytes and transports them to the frozen native
compiler/linker. It performs no Tensor, gradient, optimizer or owner execution.
"""
from pathlib import Path
import argparse, hashlib, json, re, struct, subprocess

HERE=Path(__file__).resolve().parent
FUNCTION=re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)',re.M|re.S)
CALL=re.compile(rb'\{"function":"(\w+)"\}')
RNG_EXPORTS={'rng_seed','rng_restore','rng_next','rng_next64','rng_unit32','rng_unit64'}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def functions(raw):
    pairs=FUNCTION.findall(raw);result={name.decode():body for body,name in pairs}
    assert len(pairs)==len(result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-build','compiler','linker','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    parent=a.parent_build.resolve();compiler=a.compiler.resolve();linker=a.linker.resolve()
    paths=[Path(__file__).resolve(),HERE/'profile.json',HERE/'functions.json',HERE/'overlay.hepta']
    paths.extend(parent/name for name in ('runtime.hepta','rng03.hepta','owners.hepta','build.json','tensor-gradient-adam-distributions-session'))
    inputs={p:p.read_bytes() for p in paths}
    profile=json.loads(inputs[HERE/'profile.json']);mapping=json.loads(inputs[HERE/'functions.json'])
    assert sha(compiler.read_bytes())==profile['compiler_sha256']
    assert sha(linker.read_bytes())==profile['linker_sha256']
    pin=profile['parent'];receipt=json.loads(inputs[parent/'build.json'])
    assert receipt['status']=='PASS' and receipt['schema'] in pin['accepted_receipt_schemas']
    assert receipt['source_sha256']==pin['source_sha256'] and receipt['native_sha256']==pin['native_sha256']
    assert sha(inputs[parent/'runtime.hepta'])==pin['source_sha256']
    assert sha(inputs[parent/'tensor-gradient-adam-distributions-session'])==pin['native_sha256']
    assert sha(inputs[parent/'rng03.hepta'])==profile['rng_source_sha256']
    assert sha(inputs[parent/'owners.hepta'])==profile['owners_source_sha256']
    assert sha(inputs[HERE/'functions.json'])==profile['functions_map_sha256']
    assert sha(inputs[HERE/'overlay.hepta'])==profile['overlay_sha256']
    sources={'parent':functions(inputs[parent/'runtime.hepta']),'overlay':functions(inputs[HERE/'overlay.hepta'])}
    assert len(sources['parent'])==pin['functions'] and len(sources['overlay'])==profile['overlay_functions']
    combined={}
    for item in mapping:
        name,origin=item['name'],item['origin']
        assert name not in combined and origin in sources
        body=sources[origin][name];assert sha(body)==item['sha256']
        old=sources['parent'].get(name)
        assert item['parent_sha256']==(sha(old) if old is not None else None)
        combined[name]=body
    assert set(sources['overlay'])=={item['name'] for item in mapping if item['origin']=='overlay'}
    raw=profile['runtime_header'].encode()+b''.join(combined.values())
    assert sha(raw)==profile['source_sha256'] and len(combined)==profile['functions']
    assert len(re.findall(rb'^block ',raw,re.M))==profile['blocks'] and raw.count(b' = ')==profile['instructions']
    unchanged=sum(body==sources['parent'].get(name) for name,body in combined.items())
    assert unchanged==profile['unchanged_parent_functions']
    assert all(body==sources['parent'][name] for name,body in combined.items() if name.startswith('dist_'))
    for name in ('gradient_accumulate','incoming_adopt_gradient','owned_optimizer_gradient_value',
                 'optimizer_new_momentum','owned_adam_new_moment','owned_adam_element','scalar_add_nograd'):
        assert combined[name]==sources['parent'][name],name
    (out/'runtime.hepta').write_bytes(raw)
    source_custody={'schema':'heptatorch.retained-gradient-source-custody.v1','source_sha256':sha(raw),
        'parent':pin,'functions_map_sha256':profile['functions_map_sha256'],'overlay_sha256':profile['overlay_sha256'],
        'actual_function_order_and_hashes_match':True,'unchanged_parent_functions':unchanged,
        'leaf_incoming_ownership_and_optimizer_numerical_bodies_unchanged':True,
        'all_25_distribution_bodies_unchanged':True,'retention_adds_no_persistent_strong_root':True}
    (out/'source-custody.json').write_text(json.dumps(source_custody,indent=2)+'\n')
    rng=functions(inputs[parent/'rng03.hepta']);assert len(rng)==17
    ordered=[];done=set();active=set()
    def visit(name):
        if name in RNG_EXPORTS:return
        assert name in combined and name not in active,('missing or recursive function',name)
        if name in done:return
        active.add(name)
        for target in sorted({v.decode() for v in CALL.findall(combined[name])}):visit(target)
        active.remove(name);done.add(name);ordered.append(name)
    for name in combined:visit(name)
    groups=[];group=[];ins=blocks=0
    for name in ordered:
        body=combined[name];ni=body.count(b' = ');nb=len(re.findall(rb'^block ',body,re.M))
        if group and (ins+ni>3600 or blocks+nb>1000 or len(group)>=65):
            groups.append(group);group=[];ins=blocks=0
        group.append(name);ins+=ni;blocks+=nb
    if group:groups.append(group)
    units=[('rng03',inputs[parent/'rng03.hepta'])]
    for i,names in enumerate(groups):
        calls={v.decode() for name in names for v in CALL.findall(combined[name])};imports=sorted(calls-set(names));declarations=[]
        for name in imports:
            body=rng[name] if name in RNG_EXPORTS else combined[name]
            declarations.append(re.sub(rb' entry\(\w+\)$',b' external()',body.split(b'\n')[0])+b'\n')
        assert len(names)+len(imports)<=256
        unit=f'retained_gradient_{i:02d}'
        body=f'hia-asm.v1 hepta.hia.program.v8 {unit}\nexports({",".join(names)})\n'.encode()+b''.join(declarations)+b''.join(combined[n] for n in names)
        units.append((unit,body))
    units.append(('owners',inputs[parent/'owners.hepta']))
    assert len(units)==len(profile['units'])
    report={'schema':'heptatorch.retained-gradient-native-build.v1','status':'BUILDING',
        'source_sha256':sha(raw),'functions':profile['functions'],'blocks':profile['blocks'],'instructions':profile['instructions'],
        'state_format':profile['state_format'],'backing_format':profile['backing_format'],'descriptor_bytes':576,
        'compiler_sha256':profile['compiler_sha256'],'linker_sha256':profile['linker_sha256'],'parent':pin,
        'inputs_sha256':{str(p):sha(data) for p,data in inputs.items()},'units':[],
        'full_rewrite_complete':False,'pytorch_api_denominator':2590}
    def save():(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    save();objects=[]
    for (name,body),expected in zip(units,profile['units'],strict=True):
        assert name==expected['name'] and sha(body)==expected['source_sha256']
        (out/(name+'.hepta')).write_bytes(body)
        result=subprocess.run([str(compiler)],input=body,capture_output=True,env={'PATH':''},timeout=120)
        (out/(name+'.o')).write_bytes(result.stdout);(out/(name+'.stderr')).write_bytes(result.stderr)
        row={'name':name,'source_sha256':sha(body),'object_sha256':sha(result.stdout),'functions':len(functions(body)),
            'blocks':len(re.findall(rb'^block ',body,re.M)),'instructions':body.count(b' = '),'compiler_exit':result.returncode}
        report['units'].append(row);save()
        assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF'),name
        assert all(row[k]==expected[k] for k in ('object_sha256','functions','blocks','instructions'))
        objects.append(result.stdout)
    request=b'HIASTKS1'+struct.pack('<3Q',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(o))+o for o in objects)
    (out/'link.request.bin').write_bytes(request)
    result=subprocess.run([str(linker)],input=request,capture_output=True,env={'PATH':''},timeout=120)
    (out/'link.stderr').write_bytes(result.stderr);report['link_exit']=result.returncode;save()
    assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
    assert sha(result.stdout)==profile['native_sha256'] and len(result.stdout)==profile['native_bytes']
    binary=out/'tensor-retained-session';binary.write_bytes(result.stdout);binary.chmod(0o500)
    assert all(p.read_bytes()==data for p,data in inputs.items())
    assert sha(compiler.read_bytes())==profile['compiler_sha256'] and sha(linker.read_bytes())==profile['linker_sha256']
    report.update(status='PASS',native_sha256=sha(result.stdout),native_bytes=len(result.stdout),module_count=len(units));save()
    print(json.dumps({k:report[k] for k in ('status','source_sha256','native_sha256','functions','blocks','instructions','module_count')}))
if __name__=='__main__':main()
