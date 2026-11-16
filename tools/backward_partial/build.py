"""Native HIA ready-queue backward overlay, with runtime errors in command results."""
from pathlib import Path
import argparse,ast,copy,hashlib,importlib.util,json,re,struct,subprocess,sys

HERE=Path(__file__).resolve().parent
CORE_SHA='e9e325374153f64744fbddfd2ccee00e570585cc42e4ad239361ce035ad4d2c2'
COMPILER_SHA='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER_SHA='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
FUNCTION=re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)',re.M|re.S)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
functions=lambda raw:{n.decode():body for body,n in FUNCTION.findall(raw)}
py_functions=lambda source:{n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('repository','source-root','baseline','tensor-objects','compiler','linker','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    BASE=args.baseline.resolve();V3=args.tensor_objects.resolve();REPO=args.repository.resolve()
    UPSTREAM=args.source_root.resolve();COMPILER=args.compiler.resolve();LINKER=args.linker.resolve()
    inputs={p:p.read_bytes() for p in [Path(__file__),HERE/'partial.py',HERE/'profile.json',BASE/'runtime.hepta',BASE/'overlay-authoring.py',BASE/'release_owners.hepta',V3/'overlay-authoring.py',REPO/'tools/owned_tensors/authoring.py',REPO/'docs/functions.json',REPO/'tools/owned_tensors/profile.json']}
    frozen=json.loads(inputs[HERE/'profile.json'])
    assert sha(inputs[BASE/'runtime.hepta'])==CORE_SHA
    assert sha(inputs[BASE/'overlay-authoring.py'])=='c0c60f05692d25028dd1bfbfc90e1fbd30040a974d710ffb0cf6e9230b6ed1af'
    assert sha(inputs[BASE/'release_owners.hepta'])=='993651c88e69be93dea49ef473b8409b9f609e5b226c6fcec10de230ef4af99e'
    assert sha(inputs[REPO/'tools/owned_tensors/profile.json'])=='69642a730f9da8567a01cc5afd9d15963a49ff1ab83f7761039bb301d0d2d242'
    assert sha(inputs[REPO/'docs/functions.json'])=='6ecffeb9a13a0a8fe6b78f75728fc46fef1bed06f1c49196f2d7d855ee7effca'
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
    # Parse each exact real function in a bounded source document, then run the
    # unchanged complete Core verifier/CFG compactor over the combined document.
    # The upstream host text parser does not implement adapter-only external()
    # declarations. No stub body or relaxed parser/native limit is substituted.
    context_header=b'hia-asm.v1 hepta.hia.program.v8 partial_authoring_context\n'
    context_source=context_header+b''.join(old.values())
    context=out/'authoring-context';(context/'src').mkdir(parents=True);(context/'docs').mkdir()
    (context/'src/heptatorch.hepta').write_bytes(context_source)
    (context/'docs/functions.json').write_bytes(inputs[REPO/'docs/functions.json'])
    emitter.ROOT=context
    parsed=[emitter._parse_assembly_document((context_header+body).decode()) for body in old.values()]
    context_doc=copy.deepcopy(parsed[0]);context_doc['functions']=[d['functions'][0] for d in parsed]
    def emit(source, extra_signatures):
        parsed_nodes=ast.parse(source).body
        names={name:name for name in extra_signatures}
        sigs=dict(extra_signatures)
        for node in parsed_nodes:
            names[node.name]=node.name
            sigs[node.name]=([a.annotation.id for a in node.args.args],node.returns.id)
        total=emitter.total_scalar_functions(parsed_nodes);new_functions=[]
        for node in parsed_nodes:
            b=emitter.NativeBuilder(node,names,sigs,total);b.statements(node.body)
            new_functions.append({'name':node.name,'params':[b.param(v) for v in b.params],
                'returns':[b.returns],'effects':['read','write','trap'],'entry':'b0','blocks':b.blocks})
        new_names={node.name for node in parsed_nodes};doc=copy.deepcopy(context_doc)
        doc['functions']=[f for f in doc['functions'] if f['name'] not in new_names]+new_functions
        doc=emitter.compact_authoring_cfg(doc)
        doc['functions']=[f for f in doc['functions'] if f['name'] in new_names]
        text=emitter._render_assembly_document(doc)
        mapping={n.name:{'name':n.name,'params':[{'name':a.arg,'type':a.annotation.id} for a in n.args.args],
                         'returns':n.returns.id} for n in parsed_nodes}
        return text.split('\n',1)[1],mapping
    for name in ['entry','object_command','backward_execute','sgd']:
        emitted,_=emit(ast.unparse(nodes[name]),signatures)
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
    command=command.replace('code > 57','code > 59',1)
    needle='    if code == 57:\n        result = backward_command(w, a, b, c)\n'
    assert command.count(needle)==1
    command=command.replace(needle,needle+'    if code == 58:\n        result = partial_command(w,o,at+32,a,b,c)\n        payload = 32\n    if code == 59:\n        result = gradient_present(w,a)\n',1)
    legacy=ast.unparse(nodes['backward_execute'])
    needle='        if getw(w, pi + 168) == epoch and getw(w, pi + 40) == 0:\n'
    assert legacy.count(needle)==1
    legacy=legacy.replace(needle,needle+'            putw(w,pi+360,1)\n',1)
    sgd=ast.unparse(nodes['sgd'])
    needle='    if version_can_bump(w, t, 1) == 0:\n'
    assert sgd.count(needle)==1
    sgd=sgd.replace(needle,'    if getw(w,p+360) == 0:\n        return 1\n'+needle,1)
    authoring=inputs[HERE/'partial.py'].decode()+'\n'+entry+'\n'+command+'\n'+legacy+'\n'+sgd+'\n'
    (out/'overlay-authoring.py').write_text(authoring)
    try:
        fragment,mapping=emit(authoring,signatures)
    except Exception as error:
        (out/'refusal.json').write_text(json.dumps({'stage':'source authoring','error':repr(error),'native_attempted':False},indent=2)+'\n')
        raise
    replacement=functions(fragment.encode());combined=dict(old);combined.update(replacement)
    for name in ('init_state','state_header'):
        old_tag=b'{"value":3766509302929642568}'
        assert combined[name].count(old_tag)==1,name
        combined[name]=combined[name].replace(old_tag,b'{"value":3838566896967570504}')
    # Preserve every existing hyperparameter/state check, then skip numerical
    # updates and version bumps when the parameter has no accumulated gradient.
    def guard_gradient(body,label,workspace,parameter):
        header=re.search(rb'^block '+label.encode()+rb'\(([^)]*)\)\n',body,re.M)
        assert header is not None
        params=[p.split(b':') for p in header[1].split(b',')]
        renames={name:b'guard_'+name for name,_ in params}
        arguments=b','.join(renames[name] for name,_ in params)
        guard_header=b'block '+label.encode()+b'('+b','.join(renames[n]+b':'+typ for n,typ in params)+b')\n'
        guard_workspace=renames[workspace.encode()].decode()
        guard_parameter=renames[parameter.encode()].decode()
        guard=(guard_header+
            f'  gradient_pointer:u64 = call({guard_workspace},{guard_parameter}) {{"function":"dptr"}}\n'.encode()+
            b'  gradient_offset:u64 = const() {"value":360}\n'+
            b'  gradient_address:u64 = int.add.wrap(gradient_pointer,gradient_offset)\n'+
            f'  gradient_flag:u64 = call({guard_workspace},gradient_address) {{"function":"getw"}}\n'.encode()+
            b'  gradient_zero:u64 = const() {"value":0}\n'+
            b'  gradient_absent:bool = compare.eq(gradient_flag,gradient_zero)\n'+
            b'  branch gradient_absent no_gradient() present_'+label.encode()+b'('+arguments+b')\n'+
            b'block present_'+label.encode()+b'('+header[1]+b')\n')
        body=body.replace(header[0],guard,1)
        assert body.endswith(b'end\n')
        return body[:-4]+b'block no_gradient()\n  gradient_success:u64 = const() {"value":1}\n  ret(gradient_success)\nend\n'
    combined['adam']=guard_gradient(combined['adam'],'b24','v311','v312')
    combined['sgd_momentum']=guard_gradient(combined['sgd_momentum'],'b21','v237','v238')
    raw=b'hia-asm.v1 hepta.hia.program.v8 tensor_backward_partial\n'+b''.join(combined.values())
    assert sha(raw)==frozen['source_sha256']
    (out/'runtime.hepta').write_bytes(raw);(out/'overlay.hepta').write_text(fragment)
    (out/'functions.json').write_text(json.dumps(mapping,indent=2)+'\n')
    changed=[name for name in old if old[name]!=combined[name]]
    assert set(changed)=={'entry','object_command','backward_execute','sgd','adam','sgd_momentum','init_state','state_header'}
    (out/'complete-overlay.hepta').write_bytes(b'hia-asm.v1 hepta.hia.program.v8 backward_partial_overlay\n'+b''.join(
        body for name,body in combined.items() if name not in old or old[name]!=body))
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
            'state_format':'HTSTATE5','wire_request':'HTORCH03','wire_response':'HTORSP03',
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
    assert sha(result.stdout)==frozen['native_sha256']
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
