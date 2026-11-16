"""Build native nullable gradients over the frozen materialization05 core."""
from pathlib import Path
import argparse,ast,copy,hashlib,importlib.util,json,re,struct,subprocess,sys

HERE=Path(__file__).resolve().parent
CORE_SHA='9feab47afea6723284e15a2ad6c9071dc8c5bdc27b572d2eef4ab4628ed1aea9'
AUTHOR_SHA='8385161dd8719d53385759e0119441739cbe82ce5fea2b896d5cdd3ce9b31c78'
COMPILER_SHA='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER_SHA='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
FUNCTION=re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)',re.M|re.S)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
functions=lambda raw:{n.decode():body for body,n in FUNCTION.findall(raw)}

def replace_once(source,old,new):
    assert source.count(old)==1,old
    return source.replace(old,new,1)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('repository','source-root','baseline','partial-baseline','tensor-objects','compiler','linker','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    base=args.baseline.resolve();repo=args.repository.resolve();upstream=args.source_root.resolve()
    compiler=args.compiler.resolve();linker=args.linker.resolve()
    partial=args.partial_baseline.resolve();objects=args.tensor_objects.resolve()
    paths=[Path(__file__),HERE/'nullable.py',HERE/'profile.json',base/'runtime.hepta',base/'overlay-authoring.py',base/'materialization_owners.hepta',repo/'tools/owned_tensors/authoring.py',repo/'docs/functions.json',repo/'tools/owned_tensors/profile.json',partial/'overlay-authoring.py',objects/'overlay-authoring.py']
    inputs={p:p.read_bytes() for p in paths}
    frozen=json.loads(inputs[HERE/'profile.json'])
    assert sha(inputs[HERE/'nullable.py'])==frozen['authoring_sha256']
    pins={base/'runtime.hepta':CORE_SHA,base/'overlay-authoring.py':AUTHOR_SHA,
          base/'materialization_owners.hepta':'993651c88e69be93dea49ef473b8409b9f609e5b226c6fcec10de230ef4af99e',
          repo/'tools/owned_tensors/authoring.py':'2845fc2e3f03ccbadd5ea6c59734d0be6db6477bf8434d288259fab5ee570ba6',
          repo/'docs/functions.json':'6ecffeb9a13a0a8fe6b78f75728fc46fef1bed06f1c49196f2d7d855ee7effca',
          repo/'tools/owned_tensors/profile.json':'69642a730f9da8567a01cc5afd9d15963a49ff1ab83f7761039bb301d0d2d242'}
    pins[partial/'overlay-authoring.py']='21f758fa6a0fd040d71b89c9aaa96279c9df2a3ff4e879dfd360e15a143581c1'
    pins[objects/'overlay-authoring.py']='4f3fa887e6372f19777ecde955d39716ae633a1bb9be7408c065d6e94e0fffbd'
    for path,digest in pins.items(): assert sha(inputs[path])==digest,path
    profile=json.loads(inputs[repo/'tools/owned_tensors/profile.json'])
    for rel,digest in profile['upstream_authoring_sources'].items():
        path=upstream/rel;data=path.read_bytes();assert sha(data)==digest,path;inputs[path]=data
    assert sha(compiler.read_bytes())==COMPILER_SHA and sha(linker.read_bytes())==LINKER_SHA
    (out/'producer.py').write_bytes(inputs[Path(__file__)])
    (out/'nullable-authoring.py').write_bytes(inputs[HERE/'nullable.py'])
    sys.path.insert(0,str(upstream))
    spec=importlib.util.spec_from_file_location('nullable_emitter',repo/'tools/owned_tensors/authoring.py')
    emitter=importlib.util.module_from_spec(spec);spec.loader.exec_module(emitter)
    old=functions(inputs[base/'runtime.hepta'])
    old_nodes={n.name:n for n in ast.parse(inputs[base/'overlay-authoring.py'].decode()).body if isinstance(n,ast.FunctionDef)}
    signatures={}
    for name,body in old.items():
        header=body.split(b'\n',1)[0].decode()
        m=re.search(r'^fn \w+\((.*?)\) -> \((.*?)\)',header)
        signatures[name]=([p.split(':')[1] for p in m[1].split(',') if p],m[2])
    for layer in (objects,partial):
        old_nodes.update({n.name:n for n in ast.parse(inputs[layer/'overlay-authoring.py'].decode()).body if isinstance(n,ast.FunctionDef)})
    context_header=b'hia-asm.v1 hepta.hia.program.v8 nullable_gradient_context\n'
    context=out/'authoring-context';(context/'src').mkdir(parents=True);(context/'docs').mkdir()
    (context/'src/heptatorch.hepta').write_bytes(context_header+b''.join(old.values()))
    (context/'docs/functions.json').write_bytes(inputs[repo/'docs/functions.json'])
    emitter.ROOT=context
    parsed=[emitter._parse_assembly_document((context_header+body).decode()) for body in old.values()]
    context_doc=copy.deepcopy(parsed[0]);context_doc['functions']=[d['functions'][0] for d in parsed]
    def emit(source):
        nodes=ast.parse(source).body;names={name:name for name in signatures};sigs=dict(signatures)
        for node in nodes:
            names[node.name]=node.name;sigs[node.name]=([a.annotation.id for a in node.args.args],node.returns.id)
        total=emitter.total_scalar_functions(nodes);new=[]
        for node in nodes:
            builder=emitter.NativeBuilder(node,names,sigs,total);builder.statements(node.body)
            new.append({'name':node.name,'params':[builder.param(v) for v in builder.params],
                        'returns':[builder.returns],'effects':['read','write','trap'],'entry':'b0','blocks':builder.blocks})
        selected={node.name for node in nodes};doc=copy.deepcopy(context_doc)
        doc['functions']=[f for f in doc['functions'] if f['name'] not in selected]+new
        doc=emitter.compact_authoring_cfg(doc)
        doc['functions']=[f for f in doc['functions'] if f['name'] in selected]
        text=emitter._render_assembly_document(doc).split('\n',1)[1]
        mapping={n.name:{'name':n.name,'params':[{'name':a.arg,'type':a.annotation.id} for a in n.args.args],
                         'returns':n.returns.id} for n in nodes}
        return text,mapping
    for name in ('descriptor','object_command'):
        emitted,_=emit(ast.unparse(old_nodes[name]))
        assert functions(emitted.encode())[name]==old[name],name
    descriptor=ast.unparse(old_nodes['descriptor'])
    begin=descriptor.index('        grad = alloc(w, n * 4)\n')
    end=descriptor.index('        work = alloc(w, n * 4)\n',begin)
    descriptor=descriptor[:begin]+descriptor[end:]
    command=ast.unparse(old_nodes['object_command'])
    command=replace_once(command,'    if code < 41 or code > 59:\n',
        '    if code < 41 or (code > 59 and code != 70):\n')
    needle='    if code == 59:\n        result = gradient_present(w, a)\n'
    command=replace_once(command,needle,needle+"    if code == 70:\n        if b != 0 or c != 0 or getr(s,cp+32) != 0 or getr(s,cp+40) != 0 or getr(s,cp+48) != 0 or getr(s,cp+56) != 0:\n            terror(w,7)\n        else:\n            t = public_node(w,a)\n            if t != 0:\n                result = zero_grad_none(w,t)\n")
    authoring=inputs[HERE/'nullable.py'].decode()+'\n'+descriptor+'\n'+command+'\n'
    (out/'overlay-authoring.py').write_text(authoring)
    try: fragment,mapping=emit(authoring)
    except Exception as error:
        (out/'refusal.json').write_text(json.dumps({'stage':'source authoring','error':repr(error),'native_attempted':False},indent=2)+'\n');raise
    replacement=functions(fragment.encode());combined=dict(old);combined.update(replacement)
    for name in ('init_state','state_header'):
        before=b'{"value":3838566896967570504}'
        assert combined[name].count(before)==1,name
        combined[name]=combined[name].replace(before,b'{"value":3910624491005498440}')
    changed=[name for name in old if old[name]!=combined[name]]
    assert set(changed)=={'descriptor','object_command','zero_grad','tgrad','ensure_backward_work','release_backward_scratch','init_state','state_header'},changed
    raw=b'hia-asm.v1 hepta.hia.program.v8 tensor_nullable_gradients\n'+b''.join(combined.values())
    assert sha(raw)==frozen['source_sha256']
    (out/'runtime.hepta').write_bytes(raw);(out/'overlay.hepta').write_text(fragment)
    (out/'functions.json').write_text(json.dumps(mapping,indent=2)+'\n')
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
        local=set(names);calls={n.decode() for name in names for n in re.findall(rb'\{"function":"(\w+)"\}',combined[name])}
        imports=sorted(calls-local);declarations=[]
        for name in imports:
            line=combined[name].split(b'\n',1)[0]
            declarations.append(re.sub(rb' entry\(\w+\)$',b' external()',line)+b'\n')
        assert len(names)+len(imports)<=256
        unit=f'nullable_{i:02d}'
        source=f'hia-asm.v1 hepta.hia.program.v8 {unit}\nexports({",".join(names)})\n'.encode()+b''.join(declarations)+b''.join(combined[n] for n in names)
        units.append((unit,source))
    units.append(('nullable_owners',inputs[base/'materialization_owners.hepta']))
    objects=[];rows=[]
    report={'schema':'heptatorch.nullable-gradient-native-build.v1','status':'BUILDING','baseline_source_sha256':CORE_SHA,
            'source_sha256':sha(raw),'functions':len(combined),'state_format':'HTSTATE6','wire_request':'HTORCH03','wire_response':'HTORSP03',
            'blocks':len(re.findall(rb'^block ',raw,re.M)),'instructions':raw.count(b' = '),'baseline_functions_unchanged':len(old)-len(changed),
            'changed_baseline_functions':changed,'added_functions':[n for n in combined if n not in old],
            'compiler_sha256':COMPILER_SHA,'linker_sha256':LINKER_SHA,'input_sha256':{str(p):sha(b) for p,b in inputs.items()},'units':rows}
    def save(): (out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    for name,source in units:
        (out/(name+'.hepta')).write_bytes(source)
        result=subprocess.run([str(compiler)],input=source,capture_output=True,env={'PATH':''},timeout=120)
        (out/(name+'.o')).write_bytes(result.stdout);(out/(name+'.stderr')).write_bytes(result.stderr)
        row={'name':name,'source_sha256':sha(source),'source_bytes':len(source),'compiler_exit':result.returncode,
             'object_sha256':sha(result.stdout),'functions':len(functions(source)),'blocks':len(re.findall(rb'^block ',source,re.M)),
             'instructions':source.count(b' = ')};rows.append(row);save()
        if result.returncode!=0 or result.stderr or not result.stdout.startswith(b'\x7fELF'):
            report['status']='NATIVE_COMPILER_REFUSAL';save();raise AssertionError(row)
        objects.append(result.stdout)
    request=b'HIASTKS1'+struct.pack('<3Q',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(o))+o for o in objects)
    (out/'link.request.bin').write_bytes(request)
    result=subprocess.run([str(linker)],input=request,capture_output=True,env={'PATH':''},timeout=120)
    (out/'link.stdout').write_bytes(result.stdout);(out/'link.stderr').write_bytes(result.stderr)
    report['link_exit']=result.returncode;save()
    assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
    assert sha(result.stdout)==frozen['native_sha256']
    app=out/'tensor-nullable-session';app.write_bytes(result.stdout);app.chmod(0o500)
    for p,data in inputs.items():assert p.read_bytes()==data,p
    for (name,source),obj in zip(units,objects):
        assert (out/(name+'.hepta')).read_bytes()==source and (out/(name+'.o')).read_bytes()==obj
    assert (out/'runtime.hepta').read_bytes()==raw and app.read_bytes()==result.stdout
    assert sha(compiler.read_bytes())==COMPILER_SHA and sha(linker.read_bytes())==LINKER_SHA
    report.update(status='PASS',native_sha256=sha(result.stdout),native_bytes=len(result.stdout));save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','units')},indent=2))

if __name__=='__main__':main()
