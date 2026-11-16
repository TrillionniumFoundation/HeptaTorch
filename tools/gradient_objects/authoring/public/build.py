"""Author and compile genuine gradient Tensor bindings over frozen Optimizer08."""
from pathlib import Path
import argparse, ast, copy, hashlib, importlib.util, json, re, struct, subprocess, sys

HERE=Path(__file__).resolve().parent
BASE_SHA='7c68a96b08aa9a21a52e2fd12be912c5f47dbde889e462f9a0dbb38cb3c09edf'
COMPILER_SHA='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER_SHA='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
FUNCTION=re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)',re.M|re.S)
CALL=re.compile(rb'\{"function":"(\w+)"\}')
sha=lambda x:hashlib.sha256(x).hexdigest()
functions=lambda raw:{n.decode():b for b,n in FUNCTION.findall(raw)}

def once(source,old,new):
    assert source.count(old)==1,(old,source.count(old))
    return source.replace(old,new,1)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('baseline','repository','upstream','compiler','linker','output'):
        ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    base=a.baseline.resolve();repo=a.repository.resolve();upstream=a.upstream.resolve()
    compiler=a.compiler.resolve();linker=a.linker.resolve()
    frozen=json.loads((HERE/'baseline-authoring/manifest.json').read_text())
    paths=[Path(__file__),HERE/'gradients.py',HERE/'lifetime.py',HERE/'baseline-authoring/manifest.json',
           base/'runtime.hepta',base/'rng03.hepta',base/'owners.hepta',
           repo/'tools/owned_tensors/authoring.py',repo/'tools/owned_tensors/profile.json']
    paths.extend(HERE/'baseline-authoring'/name for name in frozen['files'])
    inputs={p:p.read_bytes() for p in paths}
    assert sha(inputs[base/'runtime.hepta'])==BASE_SHA
    assert sha(inputs[base/'rng03.hepta'])=='a3fa507339b40435285dc1fa65e3cd519b6a0921a4c93bccf6ea834bb24763a3'
    assert sha(inputs[base/'owners.hepta'])=='9c02b75c44eee8459c476049f4d9ebe10935d9d9ed8b27f5842c699f8e6a0884'
    assert sha(inputs[repo/'tools/owned_tensors/authoring.py'])=='2845fc2e3f03ccbadd5ea6c59734d0be6db6477bf8434d288259fab5ee570ba6'
    assert sha(inputs[repo/'tools/owned_tensors/profile.json'])=='69642a730f9da8567a01cc5afd9d15963a49ff1ab83f7761039bb301d0d2d242'
    for name,pin in frozen['files'].items():assert sha(inputs[HERE/'baseline-authoring'/name])==pin,name
    profile=json.loads(inputs[repo/'tools/owned_tensors/profile.json'])
    for relative,pin in profile['upstream_authoring_sources'].items():
        p=upstream/relative;data=p.read_bytes();assert sha(data)==pin,p;inputs[p]=data
    assert sha(compiler.read_bytes())==COMPILER_SHA and sha(linker.read_bytes())==LINKER_SHA
    (out/'producer.py').write_bytes(inputs[Path(__file__)])
    for name in ('gradients','lifetime'):(out/(name+'-authoring.py')).write_bytes(inputs[HERE/(name+'.py')])
    sys.path.insert(0,str(upstream))
    spec=importlib.util.spec_from_file_location('gradient_object_emitter',repo/'tools/owned_tensors/authoring.py')
    emitter=importlib.util.module_from_spec(spec);spec.loader.exec_module(emitter)
    spec=importlib.util.spec_from_file_location('bounded_compactor',HERE/'baseline-authoring/compact_functions.py')
    compactor=importlib.util.module_from_spec(spec);spec.loader.exec_module(compactor)
    class Builder(emitter.NativeBuilder):
        def expr(self,node):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='move_within':
                w,dst,src,n=[self.expr(arg) for arg in node.args]
                self.op('buffer.move_within',[w,self.cast(dst,'usize'),self.cast(src,'usize'),self.cast(n,'usize')])
                return self.c(0)
            return super().expr(node)
    old=functions(inputs[base/'runtime.hepta']);rng=functions(inputs[base/'rng03.hepta'])
    assert len(old)==315 and len(rng)==17
    nodes={}
    for name in frozen['layers']:
        nodes.update({n.name:n for n in ast.parse(inputs[HERE/'baseline-authoring'/name]).body if isinstance(n,ast.FunctionDef)})
    header=b'hia-asm.v1 hepta.hia.program.v8 gradient_object_context\n'
    docs=[emitter._parse_assembly_document((header+body).decode()) for body in {**old,**rng}.values()]
    context=copy.deepcopy(docs[0]);context['functions']=[d['functions'][0] for d in docs]
    signatures={}
    for name,body in {**old,**rng}.items():
        m=re.search(r'^fn \w+\((.*?)\) -> \((.*?)\)',body.split(b'\n')[0].decode())
        signatures[name]=([p.split(':')[1] for p in m[1].split(',') if p],m[2])
    def emit(source):
        authored=ast.parse(source).body;names={n:n for n in signatures};sigs=dict(signatures)
        for node in authored:
            names[node.name]=node.name;sigs[node.name]=([p.annotation.id for p in node.args.args],node.returns.id)
        total=emitter.total_scalar_functions(authored);new=[]
        for node in authored:
            b=Builder(node,names,sigs,total);b.statements(node.body)
            new.append({'name':node.name,'params':[b.param(p) for p in b.params],'returns':[b.returns],
                        'effects':['read','write','trap'],'entry':'b0','blocks':b.blocks})
        selected={n.name for n in authored};doc=copy.deepcopy(context)
        doc['functions']=[f for f in doc['functions'] if f['name'] not in selected]+new
        doc['functions'],boundaries=compactor.compact(doc['functions'],selected)
        return emitter._render_assembly_document(doc).split('\n',1)[1],boundaries
    selected=('dptr','descriptor_page','descriptor','state_header','init_state','so_state_init','so_state_header',
              'entry','record','partial_apply_node','graph_node_release','graph_reset','graph_audit',
              'optimizer_audit','optimizer_sgd_element','optimizer_sgd_occurrence','optimizer_new_momentum')
    original={name:ast.unparse(nodes[name]) for name in selected}
    regenerated,bounds=emit('\n\n'.join(original.values()))
    regenerated_functions=functions(regenerated.encode())
    for name in selected:assert regenerated_functions[name]==old[name],('baseline authoring differs',name)
    edits={}
    edits['dptr']=once(original['dptr'],'* 384','* 448')
    edits['descriptor_page']=once(original['descriptor_page'],'98304','114688')
    edits['descriptor']=once(original['descriptor'],'memory_fill(w, p, 384, 0)','memory_fill(w, p, 448, 0)')
    for name in ('init_state','state_header'):
        edits[name]=once(original[name],str(int.from_bytes(b'HTOPT002','little')),str(int.from_bytes(b'HTGRAD01','little')))
    edits['state_header']=once(edits['state_header'],'98304','114688')
    edits['state_header']=once(edits['state_header'],'    next = getw(w, 8)\n',
        '    if getw(w,3968) != 0 or getw(w,3976) != 0 or getw(w,3984) != 0:\n        return 7\n    next = getw(w, 8)\n')
    for name in ('so_state_init','so_state_header'):
        edits[name]=once(original[name],str(int.from_bytes(b'HTOOPT02','little')),str(int.from_bytes(b'HTOGRAD1','little')))
    edits['entry']=once(original['entry'],'        if code >= 71 and code <= 79 or code == 90:\n',
        '        if code >= 91 and code <= 94:\n            next = gradient_object_command(s,o,w,32+i*64,at)\n        elif code >= 71 and code <= 79 or code == 90:\n')
    edits['entry']=once(edits['entry'],'        error = getw(w, 0)\n',
        '        if getw(w,0) == 0 and getw(w,3968) != 0:\n            graph_collect_since(w,getw(w,8))\n        error = getw(w, 0)\n')
    edits['record']=once(original['record'],'    p = dptr(w, t)\n',
        '    p = dptr(w, t)\n    if op == 10 or op == 16:\n        putw(w,p+408,1)\n')
    apply=original['partial_apply_node'];start=apply.index('    if op == 0:\n');end=apply.index('    if op >= 1 and op <= 4:\n',start)
    edits['partial_apply_node']=apply[:start]+'    if op == 0:\n        return gradient_accumulate(w,t)\n'+apply[end:]
    edits['graph_node_release']=once(original['graph_node_release'],
        'if getw(w, p + 312) != 0 or getw(w, p + 88) != 0:',
        'if getw(w, p + 312) != 0 or getw(w, p + 88) != 0 or getw(w,p+400) != 0:')
    edits['graph_reset']=once(original['graph_reset'],
        '            if getw(w, p + 312) == 0 or getw(w, p + 40) != 0:\n                return terror(w, 9)\n','')
    edits['graph_reset']=once(edits['graph_reset'],'memory_fill(w, p, 384, 0)','memory_fill(w, p, 448, 0)')
    edits['graph_audit']=once(original['graph_audit'],
        'getw(w, p + 376) + getw(w, p + 312)',
        'getw(w, p + 376) + getw(w, p + 312) + getw(w,p+400)')
    edits['optimizer_audit']=once(original['optimizer_audit'],'    slots = so_uget(w, 16)\n',
        '    if gradient_audit(w) != 0:\n        return 9\n    slots = so_uget(w, 16)\n')
    edits['optimizer_sgd_element']=once(original['optimizer_sgd_element'],
        'get4(w, getw(w, p + 24) + index * 4)','tgrad(w,parameter,index)')
    edits['optimizer_sgd_occurrence']=once(original['optimizer_sgd_occurrence'],
        'getw(w, p + 360) == 0 or getw(w, p + 248) == 0 or getw(w, p + 24) == 0',
        'getw(w, p + 360) == 0 or gradient_binding(w,parameter) == 0')
    momentum=once(original['optimizer_new_momentum'],'    q = dptr(w, t)\n',
        '    q = dptr(w, t)\n    g = gradient_binding(w,parameter)\n    if g == 0:\n        return 0\n    gp = dptr(w,g)\n')
    momentum=once(momentum,'is_dense(w, parameter)','is_dense(w, g)')
    edits['optimizer_new_momentum']=once(momentum,'getw(w, p + 128 + j * 8)','getw(w, gp + 128 + j * 8)')
    authoring='\n\n'.join(edits.values())+'\n'+inputs[HERE/'gradients.py'].decode()+'\n'+inputs[HERE/'lifetime.py'].decode()
    (out/'overlay-authoring.py').write_text(authoring)
    try:fragment,boundaries=emit(authoring)
    except Exception as error:
        (out/'refusal.json').write_text(json.dumps({'stage':'authoring','error':repr(error),'native_attempted':False},indent=2)+'\n');raise
    combined=dict(old);combined.update(functions(fragment.encode()))
    # Exact callee substitution retains the established scalar optimizer arithmetic.
    substitutions={
        'adam_element':('  v15:u64 = call(v0,v14) {"function":"get4"}',
                        '  v15:u64 = call(v0,v1,v4) {"function":"tgrad"}'),
        'sgd':('  v139:u64 = call(v120,v138) {"function":"get4"}',
               '  v139:u64 = call(v120,v121,v125) {"function":"tgrad"}'),
    }
    # The first get4 reads the parameter. Only the pinned second read is its grad.
    substitutions['momentum_element']=(
        '  v15:u64 = call(v0,v14) {"function":"get4"}',
        '  v15:u64 = call(v0,v1,v4) {"function":"tgrad"}')
    native_edits={}
    for name,(before,after) in substitutions.items():
        original_body=combined[name]
        combined[name]=once(original_body.decode(),before,after).encode()
        native_edits[name]={'before_sha256':sha(original_body),'after_sha256':sha(combined[name]),'before':before,'after':after}
    raw=b'hia-asm.v1 hepta.hia.program.v8 gradient_tensor_objects\n'+b''.join(combined.values())
    (out/'runtime.hepta').write_bytes(raw);(out/'overlay.hepta').write_text(fragment)
    (out/'authoring-boundaries.json').write_text(json.dumps({'baseline_reemission_byte_identical':True,
        'baseline_boundaries':bounds,'new_boundaries':boundaries,'exact_native_callee_edits':native_edits},indent=2)+'\n')
    external={'rng_seed','rng_restore','rng_next','rng_next64','rng_unit32','rng_unit64'}
    ordered=[];done=set();active=set()
    def visit(name):
        if name in external:return
        assert name in combined,name
        assert name not in active,('recursive native call',name)
        if name in done:return
        active.add(name)
        for target in sorted({n.decode() for n in CALL.findall(combined[name])}):visit(target)
        active.remove(name);done.add(name);ordered.append(name)
    for name in combined:visit(name)
    groups=[];group=[];ins=0;blocks=0
    for name in ordered:
        body=combined[name];ni=body.count(b' = ');nb=len(re.findall(rb'^block ',body,re.M))
        if group and (ins+ni>3600 or blocks+nb>1000 or len(group)>=65):
            groups.append(group);group=[];ins=0;blocks=0
        group.append(name);ins+=ni;blocks+=nb
    if group:groups.append(group)
    units=[('rng03',inputs[base/'rng03.hepta'])]
    for i,names in enumerate(groups):
        calls={n.decode() for name in names for n in CALL.findall(combined[name])};imports=sorted(calls-set(names));declarations=[]
        for name in imports:
            body=rng[name] if name in external else combined[name]
            declarations.append(re.sub(rb' entry\(\w+\)$',b' external()',body.split(b'\n')[0])+b'\n')
        assert len(names)+len(imports)<=256
        unit=f'gradient_object_{i:02d}'
        source=f'hia-asm.v1 hepta.hia.program.v8 {unit}\nexports({",".join(names)})\n'.encode()+b''.join(declarations)+b''.join(combined[n] for n in names)
        units.append((unit,source))
    units.append(('owners',inputs[base/'owners.hepta']))
    changed=[name for name in old if combined[name]!=old[name]]
    rows=[];report={'schema':'heptatorch.gradient-object-native-build.v1','status':'BUILDING',
        'baseline_source_sha256':BASE_SHA,'source_sha256':sha(raw),'functions':len(combined),
        'blocks':len(re.findall(rb'^block ',raw,re.M)),'instructions':raw.count(b' = '),
        'baseline_functions_unchanged':len(old)-len(changed),'changed_functions':changed,
        'added_functions':[name for name in combined if name not in old],
        'state_format':'HTGRAD01','backing_format':'HTOGRAD1','descriptor_bytes':448,
        'compiler_sha256':COMPILER_SHA,'linker_sha256':LINKER_SHA,
        'inputs_sha256':{str(p):sha(b) for p,b in inputs.items()},'units':rows}
    def save():(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    save();objects=[]
    for name,source in units:
        (out/(name+'.hepta')).write_bytes(source)
        result=subprocess.run([str(compiler)],input=source,capture_output=True,env={'PATH':''},timeout=120)
        (out/(name+'.o')).write_bytes(result.stdout);(out/(name+'.stderr')).write_bytes(result.stderr)
        row={'name':name,'source_sha256':sha(source),'object_sha256':sha(result.stdout),
             'functions':len(functions(source)),'blocks':len(re.findall(rb'^block ',source,re.M)),
             'instructions':source.count(b' = '),'compiler_exit':result.returncode};rows.append(row);save()
        if result.returncode!=0 or result.stderr or not result.stdout.startswith(b'\x7fELF'):
            report['status']='NATIVE_COMPILER_REFUSAL';save();raise AssertionError(row)
        objects.append(result.stdout)
    request=b'HIASTKS1'+struct.pack('<3Q',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(obj))+obj for obj in objects)
    (out/'link.request').write_bytes(request)
    result=subprocess.run([str(linker)],input=request,capture_output=True,env={'PATH':''},timeout=120)
    (out/'link.stderr').write_bytes(result.stderr);report['link_exit']=result.returncode;save()
    if result.returncode!=0 or result.stderr or not result.stdout.startswith(b'\x7fELF'):
        (out/'link.refusal.stdout').write_bytes(result.stdout);report['status']='NATIVE_LINKER_REFUSAL';save();raise AssertionError(report['link_exit'])
    app=out/'tensor-gradient-object-session';app.write_bytes(result.stdout);app.chmod(0o500)
    for p,b in inputs.items():assert p.read_bytes()==b,p
    assert sha(compiler.read_bytes())==COMPILER_SHA and sha(linker.read_bytes())==LINKER_SHA
    report.update(status='PASS',native_sha256=sha(result.stdout),native_bytes=len(result.stdout));save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs_sha256','units')},indent=2))

if __name__=='__main__':main()
