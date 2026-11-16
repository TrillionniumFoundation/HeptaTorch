"""Compile the source-bound public-gradient HIA overlay with native ABI3 tools.

Python performs deterministic source composition and compiler/linker transport.
The complete actual HIA bodies are checked and executed natively. No numerical,
Tensor, ownership, gradient or optimizer operation executes in this host code.
"""
from pathlib import Path
import argparse, hashlib, json, re, struct, subprocess

HERE=Path(__file__).resolve().parent
FUNCTION=re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)',re.M|re.S)
CALL=re.compile(rb'\{"function":"(\w+)"\}')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def functions(raw):
    pairs=FUNCTION.findall(raw);result={name.decode():body for body,name in pairs}
    assert len(pairs)==len(result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','compiler','linker','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    base=a.baseline.resolve();compiler=a.compiler.resolve();linker=a.linker.resolve()
    paths=[Path(__file__).resolve(),HERE/'profile.json',HERE/'functions.json',HERE/'overlay.hepta',
           base/'runtime.hepta',base/'rng03.hepta',base/'owners.hepta',base/'build.json']
    inputs={path:path.read_bytes() for path in paths};profile=json.loads(inputs[HERE/'profile.json'])
    mapping=json.loads(inputs[HERE/'functions.json']);baseline_receipt=json.loads(inputs[base/'build.json'])
    assert sha(compiler.read_bytes())==profile['compiler_sha256'] and sha(linker.read_bytes())==profile['linker_sha256']
    assert baseline_receipt['status']=='PASS' and baseline_receipt['source_sha256']==profile['baseline_source_sha256']
    assert baseline_receipt['native_sha256']==profile['baseline_native_sha256']
    assert sha(inputs[base/'runtime.hepta'])==profile['baseline_source_sha256']
    assert sha(inputs[base/'rng03.hepta'])==profile['rng_source_sha256']
    assert sha(inputs[base/'owners.hepta'])==profile['owners_source_sha256']
    assert sha(inputs[HERE/'overlay.hepta'])==profile['overlay_sha256']
    assert sha(inputs[HERE/'functions.json'])==profile['functions_map_sha256']
    old=functions(inputs[base/'runtime.hepta']);overlay=functions(inputs[HERE/'overlay.hepta']);rng=functions(inputs[base/'rng03.hepta'])
    assert len(old)==profile['baseline_functions'] and len(overlay)==profile['overlay_functions'] and len(rng)==17
    combined=dict(old);combined.update(overlay)
    assert list(combined)==[row['name'] for row in mapping]
    for row in mapping:
        name=row['name'];assert sha(combined[name])==row['sha256']
        assert row['origin']==('overlay' if name in overlay else 'baseline')
        if name in old:assert row['baseline_sha256']==sha(old[name])
    raw=profile['runtime_header'].encode()+b''.join(combined.values())
    assert sha(raw)==profile['source_sha256']
    assert len(combined)==profile['functions'] and raw.count(b' = ')==profile['instructions']
    assert len(re.findall(rb'^block ',raw,re.M))==profile['blocks']
    (out/'runtime.hepta').write_bytes(raw)
    (out/'source-custody.json').write_text(json.dumps({'baseline_source_sha256':profile['baseline_source_sha256'],
        'overlay_sha256':profile['overlay_sha256'],'source_sha256':sha(raw),
        'baseline_functions_unchanged':sum(name not in overlay for name in old),
        'function_order_and_hashes_match':True},indent=2)+'\n')
    external={'rng_seed','rng_restore','rng_next','rng_next64','rng_unit32','rng_unit64'}
    ordered=[];done=set();active=set()
    def visit(name):
        if name in external:return
        assert name in combined and name not in active,('missing/recursive function',name)
        if name in done:return
        active.add(name)
        for target in sorted({n.decode() for n in CALL.findall(combined[name])}):visit(target)
        active.remove(name);done.add(name);ordered.append(name)
    for name in combined:visit(name)
    groups=[];group=[];ins=blocks=0
    for name in ordered:
        body=combined[name];ni=body.count(b' = ');nb=len(re.findall(rb'^block ',body,re.M))
        if group and (ins+ni>3600 or blocks+nb>1000 or len(group)>=65):
            groups.append(group);group=[];ins=blocks=0
        group.append(name);ins+=ni;blocks+=nb
    if group:groups.append(group)
    units=[('rng03',inputs[base/'rng03.hepta'])]
    for i,names in enumerate(groups):
        calls={n.decode() for name in names for n in CALL.findall(combined[name])}
        imports=sorted(calls-set(names));declarations=[]
        for name in imports:
            body=rng[name] if name in external else combined[name]
            declarations.append(re.sub(rb' entry\(\w+\)$',b' external()',body.split(b'\n')[0])+b'\n')
        assert len(names)+len(imports)<=256
        unit=f'gradient_incoming_{i:02d}'
        body=f'hia-asm.v1 hepta.hia.program.v8 {unit}\nexports({",".join(names)})\n'.encode()+b''.join(declarations)+b''.join(combined[n] for n in names)
        units.append((unit,body))
    units.append(('owners',inputs[base/'owners.hepta']))
    assert len(units)==len(profile['units'])
    report={'schema':'heptatorch.gradient-objects-native-build.v1','status':'BUILDING',
        'baseline_source_sha256':profile['baseline_source_sha256'],'source_sha256':sha(raw),
        'functions':profile['functions'],'blocks':profile['blocks'],'instructions':profile['instructions'],
        'state_format':profile['state_format'],'backing_format':profile['backing_format'],
        'descriptor_bytes':profile['descriptor_bytes'],'compiler_sha256':profile['compiler_sha256'],'linker_sha256':profile['linker_sha256'],
        'inputs_sha256':{str(path):sha(data) for path,data in inputs.items()},'units':[],
        'full_rewrite_complete':False,'pytorch_api_denominator':2590}
    def save():(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    save();objects=[]
    for (name,body),pin in zip(units,profile['units'],strict=True):
        assert name==pin['name'] and sha(body)==pin['source_sha256']
        (out/(name+'.hepta')).write_bytes(body)
        result=subprocess.run([str(compiler)],input=body,capture_output=True,env={'PATH':''},timeout=120)
        (out/(name+'.o')).write_bytes(result.stdout);(out/(name+'.stderr')).write_bytes(result.stderr)
        row={'name':name,'source_sha256':sha(body),'object_sha256':sha(result.stdout),'source_bytes':len(body),
             'object_bytes':len(result.stdout),'functions':len(functions(body)),'blocks':len(re.findall(rb'^block ',body,re.M)),
             'instructions':body.count(b' = '),'compiler_exit':result.returncode};report['units'].append(row);save()
        assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
        assert row['object_sha256']==pin['object_sha256'] and row['functions']==pin['functions']
        assert row['blocks']==pin['blocks'] and row['instructions']==pin['instructions']
        objects.append(result.stdout)
    request=b'HIASTKS1'+struct.pack('<3Q',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(obj))+obj for obj in objects)
    (out/'link.request.bin').write_bytes(request)
    result=subprocess.run([str(linker)],input=request,capture_output=True,env={'PATH':''},timeout=120)
    (out/'link.stderr').write_bytes(result.stderr);report['link_exit']=result.returncode;save()
    assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
    assert sha(result.stdout)==profile['native_sha256'] and len(result.stdout)==profile['native_bytes']
    binary=out/'tensor-gradient-object-session';binary.write_bytes(result.stdout);binary.chmod(0o500)
    assert all(path.read_bytes()==data for path,data in inputs.items())
    assert sha(compiler.read_bytes())==profile['compiler_sha256'] and sha(linker.read_bytes())==profile['linker_sha256']
    report.update(status='PASS',native_sha256=sha(result.stdout),native_bytes=len(result.stdout));save()
    print(json.dumps({k:report[k] for k in ('status','source_sha256','native_sha256','functions','blocks','instructions')}))
if __name__=='__main__':main()
