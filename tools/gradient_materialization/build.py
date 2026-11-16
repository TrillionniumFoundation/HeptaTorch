"""Build native gradient materialization over the frozen partial04 core."""
from pathlib import Path
import argparse,ast,copy,hashlib,importlib.util,json,re,struct,subprocess,sys

HERE=Path(__file__).resolve().parent
CORE_SHA='302677af52d3e04520e6f132aaf17b98cbe5d2cf404aec040bf5144e324d2d4d'
AUTHOR_SHA='21f758fa6a0fd040d71b89c9aaa96279c9df2a3ff4e879dfd360e15a143581c1'
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
    for name in ('repository','source-root','baseline','compiler','linker','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    base=args.baseline.resolve();repo=args.repository.resolve();upstream=args.source_root.resolve()
    compiler=args.compiler.resolve();linker=args.linker.resolve()
    paths=[Path(__file__),HERE/'materialization.py',HERE/'profile.json',base/'runtime.hepta',base/'overlay-authoring.py',base/'partial_owners.hepta',repo/'tools/owned_tensors/authoring.py',repo/'docs/functions.json',repo/'tools/owned_tensors/profile.json']
    inputs={p:p.read_bytes() for p in paths}
    frozen=json.loads(inputs[HERE/'profile.json'])
    assert sha(inputs[HERE/'materialization.py'])==frozen['authoring_sha256']
    pins={base/'runtime.hepta':CORE_SHA,base/'overlay-authoring.py':AUTHOR_SHA,
          base/'partial_owners.hepta':'993651c88e69be93dea49ef473b8409b9f609e5b226c6fcec10de230ef4af99e',
          repo/'tools/owned_tensors/authoring.py':'2845fc2e3f03ccbadd5ea6c59734d0be6db6477bf8434d288259fab5ee570ba6',
          repo/'docs/functions.json':'6ecffeb9a13a0a8fe6b78f75728fc46fef1bed06f1c49196f2d7d855ee7effca',
          repo/'tools/owned_tensors/profile.json':'69642a730f9da8567a01cc5afd9d15963a49ff1ab83f7761039bb301d0d2d242'}
    for path,digest in pins.items(): assert sha(inputs[path])==digest,path
    profile=json.loads(inputs[repo/'tools/owned_tensors/profile.json'])
    for rel,digest in profile['upstream_authoring_sources'].items():
        path=upstream/rel;data=path.read_bytes();assert sha(data)==digest,path;inputs[path]=data
    assert sha(compiler.read_bytes())==COMPILER_SHA and sha(linker.read_bytes())==LINKER_SHA
    (out/'producer.py').write_bytes(inputs[Path(__file__)])
    (out/'materialization-authoring.py').write_bytes(inputs[HERE/'materialization.py'])
    sys.path.insert(0,str(upstream))
    spec=importlib.util.spec_from_file_location('materialization_emitter',repo/'tools/owned_tensors/authoring.py')
    emitter=importlib.util.module_from_spec(spec);spec.loader.exec_module(emitter)
    old=functions(inputs[base/'runtime.hepta'])
    old_nodes={n.name:n for n in ast.parse(inputs[base/'overlay-authoring.py'].decode()).body if isinstance(n,ast.FunctionDef)}
    signatures={}
    for name,body in old.items():
        header=body.split(b'\n',1)[0].decode()
        m=re.search(r'^fn \w+\((.*?)\) -> \((.*?)\)',header)
        signatures[name]=([p.split(':')[1] for p in m[1].split(',') if p],m[2])
    context_header=b'hia-asm.v1 hepta.hia.program.v8 gradient_materialization_context\n'
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
    for name in ('partial_apply_node','partial_execute','backward_execute'):
        emitted,_=emit(ast.unparse(old_nodes[name]))
        assert functions(emitted.encode())[name]==old[name],name
    apply=ast.unparse(old_nodes['partial_apply_node'])
    apply=replace_once(apply,'        j = 0\n','        present = getw(w,p+360)\n        j = 0\n')
    apply=replace_once(apply,
        '            put4(w, dst + j * 4, addf(get4(w, dst + j * 4), get4(w, src + j * 4)))\n',
        '            value = get4(w,src+j*4)\n            if present != 0:\n                value = addf(get4(w,dst+j*4),value)\n            put4(w,dst+j*4,value)\n')
    partial=ast.unparse(old_nodes['partial_execute'])
    partial=replace_once(partial,'    queue = alloc(w, queue_bytes)\n',
        '    stage_bytes = gradient_workspace_size(w,t,epoch)\n    if getw(w,0) != 0:\n        return 0\n    queue = alloc(w,queue_bytes+stage_bytes)\n')
    partial=replace_once(partial,'            memory_fill(w, getw(w, p + 160), getw(w, p + 8) * 4, 0)\n',
        '            memory_fill(w, getw(w, p + 160), getw(w, p + 8) * 4, 0)\n            putw(w,p+336,0)\n')
    partial=replace_once(partial,'    if partial_push(w, queue, t) == 0:\n',
        '    putw(w,p+336,1)\n    if partial_push(w, queue, t) == 0:\n')
    partial=replace_once(partial,'            partial_apply_node(w, current)\n',
        '            gradient_apply(w,current,queue+queue_bytes)\n')
    partial=replace_once(partial,'    if release_backward_scratch(w, t, epoch) == 0:\n',
        '    if gradient_clear_marks(w,t,epoch) == 0:\n        return 0\n    if release_backward_scratch(w, t, epoch) == 0:\n')
    legacy=ast.unparse(old_nodes['backward_execute'])
    legacy=replace_once(legacy,'    if ensure_backward_work(w, limit, epoch) == 0:\n        return 0\n',
        '    if ensure_backward_work(w, limit, epoch) == 0:\n        return 0\n    stage_bytes = gradient_workspace_size(w,limit,epoch)\n    if getw(w,0) != 0:\n        return 0\n    scratch = alloc(w,stage_bytes+1)\n    if scratch == 0:\n        return 0\n    scratch_owner = getw(w,80)\n')
    legacy=replace_once(legacy,'            memory_fill(w, getw(w, pi + 160), getw(w, pi + 8) * 4, 0)\n',
        '            memory_fill(w, getw(w, pi + 160), getw(w, pi + 8) * 4, 0)\n            putw(w,pi+336,0)\n')
    legacy=replace_once(legacy,'    i = limit\n','    putw(w,p+336,1)\n    i = limit\n')
    begin=legacy.index('            if op >= 1 and op <= 4:\n')
    end=legacy.index('            if getw(w, 0) != 0:\n',begin)
    legacy=legacy[:begin]+'            if op != 0:\n                gradient_apply(w,current,scratch)\n'+legacy[end:]
    begin=legacy.index('            putw(w, pi + 360, 1)\n')
    end=legacy.index('        i = i + 1\n',begin)
    legacy=legacy[:begin]+'            partial_apply_node(w,current)\n'+legacy[end:]
    legacy=replace_once(legacy,'    if retain == 0:\n        return release_backward_scratch(w, limit, epoch)\n',
        '    if gradient_clear_marks(w,limit,epoch) == 0:\n        return 0\n    if hold_release(w,scratch_owner) == 0:\n        return 0\n    if retain == 0:\n        return release_backward_scratch(w, limit, epoch)\n')
    authoring=inputs[HERE/'materialization.py'].decode()+'\n'+apply+'\n'+partial+'\n'+legacy+'\n'
    (out/'overlay-authoring.py').write_text(authoring)
    try: fragment,mapping=emit(authoring)
    except Exception as error:
        (out/'refusal.json').write_text(json.dumps({'stage':'source authoring','error':repr(error),'native_attempted':False},indent=2)+'\n');raise
    replacement=functions(fragment.encode());combined=dict(old);combined.update(replacement)
    changed=[name for name in old if old[name]!=combined[name]]
    assert set(changed)=={'gadd','partial_apply_node','partial_execute','backward_execute'},changed
    raw=b'hia-asm.v1 hepta.hia.program.v8 tensor_gradient_materialization\n'+b''.join(combined.values())
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
        unit=f'materialization_{i:02d}'
        source=f'hia-asm.v1 hepta.hia.program.v8 {unit}\nexports({",".join(names)})\n'.encode()+b''.join(declarations)+b''.join(combined[n] for n in names)
        units.append((unit,source))
    units.append(('materialization_owners',inputs[base/'partial_owners.hepta']))
    objects=[];rows=[]
    report={'schema':'heptatorch.gradient-materialization-native-build.v1','status':'BUILDING','baseline_source_sha256':CORE_SHA,
            'source_sha256':sha(raw),'functions':len(combined),'state_format':'HTSTATE5','wire_request':'HTORCH03','wire_response':'HTORSP03',
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
    app=out/'tensor-partial-session';app.write_bytes(result.stdout);app.chmod(0o500)
    for p,data in inputs.items():assert p.read_bytes()==data,p
    for (name,source),obj in zip(units,objects):
        assert (out/(name+'.hepta')).read_bytes()==source and (out/(name+'.o')).read_bytes()==obj
    assert (out/'runtime.hepta').read_bytes()==raw and app.read_bytes()==result.stdout
    assert sha(compiler.read_bytes())==COMPILER_SHA and sha(linker.read_bytes())==LINKER_SHA
    report.update(status='PASS',native_sha256=sha(result.stdout),native_bytes=len(result.stdout));save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','units')},indent=2))

if __name__=='__main__':main()
