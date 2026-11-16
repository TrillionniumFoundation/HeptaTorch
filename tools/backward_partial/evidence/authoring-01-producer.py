"""Native HIA ready-queue backward overlay, with runtime errors in command results."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,re,struct,subprocess,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=ROOT/'backward-release-extension/portable-04'
V3=ROOT/'owned-tensor-object-extension/final-portable-abi3'
REPO=ROOT/'HeptaTorch-tensor-objects'
UPSTREAM=Path('/workspace/scratch/dfc64d0acfc2/hia-toolchain-cold')
COMPILER=ROOT/'stack-budget-extension/portable-final-build/native/stack-module-compiler'
LINKER=ROOT/'stack-budget-extension/portable-final-build/native/stack-linker-session'
CORE_SHA='e9e325374153f64744fbddfd2ccee00e570585cc42e4ad239361ce035ad4d2c2'
COMPILER_SHA='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER_SHA='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
FUNCTION=re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)',re.M|re.S)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
functions=lambda raw:{n.decode():body for body,n in FUNCTION.findall(raw)}
py_functions=lambda source:{n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    inputs={p:p.read_bytes() for p in [Path(__file__),HERE/'partial.py',BASE/'runtime.hepta',BASE/'overlay-authoring.py',BASE/'release_owners.hepta',V3/'overlay-authoring.py',REPO/'tools/owned_tensors/authoring.py',REPO/'docs/functions.json',REPO/'tools/owned_tensors/profile.json']}
    assert sha(inputs[BASE/'runtime.hepta'])==CORE_SHA
    assert sha(inputs[V3/'overlay-authoring.py'])=='4f3fa887e6372f19777ecde955d39716ae633a1bb9be7408c065d6e94e0fffbd'
    assert sha(inputs[REPO/'tools/owned_tensors/authoring.py'])=='2845fc2e3f03ccbadd5ea6c59734d0be6db6477bf8434d288259fab5ee570ba6'
    profile=json.loads(inputs[REPO/'tools/owned_tensors/profile.json'])
    for rel,digest in profile['upstream_authoring_sources'].items():
        path=UPSTREAM/rel;data=path.read_bytes();assert sha(data)==digest,rel;inputs[path]=data
    assert sha(COMPILER.read_bytes())==COMPILER_SHA and sha(LINKER.read_bytes())==LINKER_SHA
    (out/'producer.py').write_bytes(inputs[Path(__file__)])
    (out/'partial-authoring.py').write_bytes(inputs[HERE/'partial.py'])
    sys.path.insert(0,str(UPSTREAM))
    spec=importlib.util.spec_from_file_location('partial_emitter',REPO/'tools/owned_tensors/authoring.py')
    emitter=importlib.util.module_from_spec(spec);spec.loader.exec_module(emitter)
    old=functions(inputs[BASE/'runtime.hepta'])
    nodes=py_functions(inputs[V3/'overlay-authoring.py'].decode())
    nodes.update(py_functions(inputs[BASE/'overlay-authoring.py'].decode()))
    signatures={};declarations=[]
    for name,body in old.items():
        line=body.split(b'\n',1)[0]
        declarations.append(re.sub(rb' entry\(\w+\)$',b' external()',line)+b'\n')
        m=re.search(r'^fn \w+\((.*?)\) -> \((.*?)\)',line.decode())
        signatures[name]=([p.split(':')[1] for p in m[1].split(',') if p],m[2])
    # An authoring module imports the exact pinned callee signatures. The
    # production link below verifies all real definitions and effects again.
    # This avoids feeding the monolithic >16k-line source to a text parser whose
    # native unit profile is bounded; no parser or compiler limit is changed.
    context_source=b'hia-asm.v1 hepta.hia.program.v8 partial_authoring_context\n'+b''.join(declarations)
    context=out/'authoring-context';(context/'src').mkdir(parents=True);(context/'docs').mkdir()
    (context/'src/heptatorch.hepta').write_bytes(context_source)
    (context/'docs/functions.json').write_bytes(inputs[REPO/'docs/functions.json'])
    emitter.ROOT=context
    for name in ['entry','object_command']:
        emitted,_=emitter.emit(ast.unparse(nodes[name]),signatures)
        assert functions(emitted.encode())[name]==old[name],('baseline mismatch',name)
    entry=ast.unparse(nodes['entry'])
    assert entry.count('    at = 32\n')==1
    entry=entry.replace('    at = 32\n','    at = 32\n    halted = 0\n',1)
    assert entry.count('    while i < count:')==1
    entry=entry.replace('    while i < count:','    while i < count and halted == 0:',1)
    assert entry.count('        at = next\n')==1
    entry=entry.replace('        at = next\n','        if getr(s,32+i*64) == 58 and getw(o,at+16) != 0:\n            halted = 1\n        at = next\n',1)
    command=ast.unparse(nodes['object_command'])
    assert command.count('code > 57')==1
    command=command.replace('code > 57','code > 58',1)
    needle='    if code == 57:\n        result = backward_command(w, a, b, c)\n'
    assert command.count(needle)==1
    command=command.replace(needle,needle+'    if code == 58:\n        result = partial_command(w,o,at+32,a,b,c)\n        payload = 32\n',1)
    authoring=inputs[HERE/'partial.py'].decode()+'\n'+entry+'\n'+command+'\n'
    (out/'overlay-authoring.py').write_text(authoring)
    try:
        fragment,mapping=emitter.emit(authoring,signatures)
    except Exception as error:
        (out/'refusal.json').write_text(json.dumps({'stage':'source authoring','error':repr(error),'native_attempted':False},indent=2)+'\n')
        raise
    replacement=functions(fragment.encode());combined=dict(old);combined.update(replacement)
    raw=b'hia-asm.v1 hepta.hia.program.v8 tensor_backward_partial\n'+b''.join(combined.values())
    (out/'runtime.hepta').write_bytes(raw);(out/'overlay.hepta').write_text(fragment)
    (out/'functions.json').write_text(json.dumps(mapping,indent=2)+'\n')
    changed=[name for name in old if old[name]!=combined[name]]
    assert set(changed)=={'entry','object_command'}
    ordered=[];active=set();done=set()
    def visit(name):
        assert name not in active,('recursion',name)
        if name in done:return
        active.add(name)
        for target in sorted({n.decode() for n in re.findall(rb'\{"function":"(\w+)"\}',combined[name])}):
            assert target in combined,(name,target);visit(target)
        active.remove(name);done.add(name);ordered.append(name)
    for name in combined:visit(name)
    groups=[];group=[];ins=0;blocks=0
    for name in ordered:
        body=combined[name];count=body.count(b' = ');nblocks=len(re.findall(rb'^block ',body,re.M))
        if group and (ins+count>3600 or blocks+nblocks>1000 or len(group)>=65):
            groups.append(group);group=[];ins=0;blocks=0
        group.append(name);ins+=count;blocks+=nblocks
    if group:groups.append(group)
    units=[]
    for i,names in enumerate(groups):
        local=set(names)
        calls={n.decode() for name in names for n in re.findall(rb'\{"function":"(\w+)"\}',combined[name])}
        imports=sorted(calls-local);declarations=[]
        for name in imports:
            line=combined[name].split(b'\n',1)[0]
            declarations.append(re.sub(rb' entry\(\w+\)$',b' external()',line)+b'\n')
        assert len(names)+len(imports)<=256
        unit=f'partial_{i:02d}'
        source=f'hia-asm.v1 hepta.hia.program.v8 {unit}\nexports({",".join(names)})\n'.encode()+b''.join(declarations)+b''.join(combined[n] for n in names)
        units.append((unit,source))
    units.append(('partial_owners',inputs[BASE/'release_owners.hepta']))
    objects=[];rows=[]
    report={'schema':'heptatorch.backward-partial-native-build.v1','status':'BUILDING',
            'baseline_source_sha256':CORE_SHA,'source_sha256':sha(raw),'functions':len(combined),
            'state_format':'HTSTATE4','wire_request':'HTORCH03','wire_response':'HTORSP03',
            'blocks':len(re.findall(rb'^block ',raw,re.M)),'instructions':raw.count(b' = '),
            'baseline_functions_unchanged':len(old)-len(changed),'changed_baseline_functions':changed,
            'compiler_sha256':COMPILER_SHA,'linker_sha256':LINKER_SHA,
            'input_sha256':{str(p):sha(b) for p,b in inputs.items()},'units':rows}
    def save(): (out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    for name,source in units:
        (out/(name+'.hepta')).write_bytes(source)
        result=subprocess.run([str(COMPILER)],input=source,capture_output=True,env={'PATH':''},timeout=120)
        (out/(name+'.o')).write_bytes(result.stdout);(out/(name+'.stderr')).write_bytes(result.stderr)
        row={'name':name,'source_sha256':sha(source),'source_bytes':len(source),
             'compiler_exit':result.returncode,'object_sha256':sha(result.stdout),
             'functions':len(functions(source)),'blocks':len(re.findall(rb'^block ',source,re.M)),
             'instructions':source.count(b' = ')}
        rows.append(row);save()
        assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF'),row
        objects.append(result.stdout)
    request=b'HIASTKS1'+struct.pack('<3Q',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(o))+o for o in objects)
    (out/'link.request.bin').write_bytes(request)
    result=subprocess.run([str(LINKER)],input=request,capture_output=True,env={'PATH':''},timeout=120)
    (out/'link.stdout').write_bytes(result.stdout);(out/'link.stderr').write_bytes(result.stderr)
    report['link_exit']=result.returncode;save()
    assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
    app=out/'tensor-partial-session';app.write_bytes(result.stdout);app.chmod(0o500)
    for p,data in inputs.items():assert p.read_bytes()==data,p
    for (name,source),obj in zip(units,objects):
        assert (out/(name+'.hepta')).read_bytes()==source
        assert (out/(name+'.o')).read_bytes()==obj
    assert (out/'runtime.hepta').read_bytes()==raw and app.read_bytes()==result.stdout
    assert sha(COMPILER.read_bytes())==COMPILER_SHA and sha(LINKER.read_bytes())==LINKER_SHA
    report.update(status='PASS',native_sha256=sha(result.stdout),native_bytes=len(result.stdout));save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','units')},indent=2))

if __name__=='__main__':main()
