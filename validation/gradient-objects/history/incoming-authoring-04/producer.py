"""Source-bound HIA incoming-gradient Storage/Version transfer.

Host code only authors and compiles HIA. All gradient values, ownership, copies,
merges and Version updates execute in the linked native program.
"""
from pathlib import Path
import argparse,ast,copy,hashlib,importlib.util,json,re,struct,subprocess,sys

HERE=Path(__file__).resolve().parent
FOUNDATION_SHA='07c2fb69ea61d48466a57e47b80d4cf85be8ad88c9c12a2f771e0402cc319dcc'
FOUNDATION_AUTHOR_SHA='50c6ec6b0b36a07e779076032a61921cbf84b520820e556e6833fbdfaf84eb60'
COMPILER_SHA='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER_SHA='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
FUNCTION=re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)',re.M|re.S)
CALL=re.compile(rb'\{"function":"(\w+)"\}')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
functions=lambda raw:{name.decode():body for body,name in FUNCTION.findall(raw)}

def once(source,old,new):
    assert source.count(old)==1,(old,source.count(old))
    return source.replace(old,new,1)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('foundation','foundation-authoring','repository','upstream','compiler','linker','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    base=a.foundation.resolve();authors=a.foundation_authoring.resolve()
    repo=a.repository.resolve();upstream=a.upstream.resolve()
    compiler=a.compiler.resolve();linker=a.linker.resolve()
    paths=[Path(__file__),HERE/'incoming.py',base/'runtime.hepta',base/'rng03.hepta',base/'owners.hepta',
           base/'overlay-authoring.py',authors/'baseline-authoring/manifest.json',
           repo/'tools/owned_tensors/authoring.py',repo/'tools/owned_tensors/profile.json']
    manifest=json.loads((authors/'baseline-authoring/manifest.json').read_bytes())
    paths.extend(authors/'baseline-authoring'/name for name in manifest['files'])
    inputs={p:p.read_bytes() for p in paths}
    assert sha(inputs[base/'runtime.hepta'])==FOUNDATION_SHA
    assert sha(inputs[base/'overlay-authoring.py'])==FOUNDATION_AUTHOR_SHA
    assert sha(inputs[base/'rng03.hepta'])=='a3fa507339b40435285dc1fa65e3cd519b6a0921a4c93bccf6ea834bb24763a3'
    assert sha(inputs[base/'owners.hepta'])=='9c02b75c44eee8459c476049f4d9ebe10935d9d9ed8b27f5842c699f8e6a0884'
    assert sha(inputs[authors/'baseline-authoring/manifest.json'])=='f0118f3990b4e387a64be0a41c35401d60970953d8889b5ce9a923763bf19d51'
    assert sha(inputs[repo/'tools/owned_tensors/authoring.py'])=='2845fc2e3f03ccbadd5ea6c59734d0be6db6477bf8434d288259fab5ee570ba6'
    assert sha(inputs[repo/'tools/owned_tensors/profile.json'])=='69642a730f9da8567a01cc5afd9d15963a49ff1ab83f7761039bb301d0d2d242'
    for name,pin in manifest['files'].items():
        assert sha(inputs[authors/'baseline-authoring'/name])==pin,name
    profile=json.loads(inputs[repo/'tools/owned_tensors/profile.json'])
    for relative,pin in profile['upstream_authoring_sources'].items():
        p=upstream/relative;data=p.read_bytes();assert sha(data)==pin,p;inputs[p]=data
    assert sha(compiler.read_bytes())==COMPILER_SHA and sha(linker.read_bytes())==LINKER_SHA
    (out/'producer.py').write_bytes(inputs[Path(__file__)])
    (out/'incoming-authoring.py').write_bytes(inputs[HERE/'incoming.py'])
    sys.path.insert(0,str(upstream))
    spec=importlib.util.spec_from_file_location('incoming_emitter',repo/'tools/owned_tensors/authoring.py')
    emitter=importlib.util.module_from_spec(spec);spec.loader.exec_module(emitter)
    spec=importlib.util.spec_from_file_location('incoming_compactor',authors/'baseline-authoring/compact_functions.py')
    compactor=importlib.util.module_from_spec(spec);spec.loader.exec_module(compactor)
    class Builder(emitter.NativeBuilder):
        def expr(self,node):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='move_within':
                w,dst,src,n=[self.expr(arg) for arg in node.args]
                self.op('buffer.move_within',[w,self.cast(dst,'usize'),self.cast(src,'usize'),self.cast(n,'usize')])
                return self.c(0)
            return super().expr(node)
    old=functions(inputs[base/'runtime.hepta']);rng=functions(inputs[base/'rng03.hepta'])
    assert len(old)==330 and len(rng)==17
    nodes={}
    for name in manifest['layers']:
        nodes.update({n.name:n for n in ast.parse(inputs[authors/'baseline-authoring'/name]).body if isinstance(n,ast.FunctionDef)})
    nodes.update({n.name:n for n in ast.parse(inputs[base/'overlay-authoring.py']).body if isinstance(n,ast.FunctionDef)})
    header=b'hia-asm.v1 hepta.hia.program.v8 gradient_incoming_context\n'
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
              'graph_reset','gradient_audit','partial_execute','backward_execute')
    original={name:ast.unparse(nodes[name]) for name in selected}
    regenerated,bounds=emit('\n\n'.join(original.values()))
    regenerated_functions=functions(regenerated.encode())
    for name in selected:
        assert regenerated_functions[name]==old[name],('foundation authoring differs',name)
    edits={}
    edits['dptr']=once(original['dptr'],'* 448','* 576')
    edits['descriptor_page']=once(original['descriptor_page'],'114688','147456')
    edits['descriptor']=once(original['descriptor'],'memory_fill(w, p, 448, 0)','memory_fill(w, p, 576, 0)')
    edits['graph_reset']=once(original['graph_reset'],'memory_fill(w, p, 448, 0)','memory_fill(w, p, 576, 0)')
    for name in ('init_state','state_header'):
        edits[name]=once(original[name],str(int.from_bytes(b'HTGRAD01','little')),str(int.from_bytes(b'HTGRAD02','little')))
    edits['state_header']=once(edits['state_header'],'114688','147456')
    edits['state_header']=once(edits['state_header'],'    next = getw(w, 8)\n',
        '    if getw(w,3992) != 0 or getw(w,4000) != 0 or getw(w,4008) != 0:\n        return 7\n    next = getw(w, 8)\n')
    for name in ('so_state_init','so_state_header'):
        edits[name]=once(original[name],str(int.from_bytes(b'HTOGRAD1','little')),str(int.from_bytes(b'HTOGRAD2','little')))
    edits['gradient_audit']=once(original['gradient_audit'],'    status = 0\n',
        '    status = 0\n    if getw(w,3992) != 0 or getw(w,4000) != 0 or getw(w,4008) != 0:\n        return 9\n')
    edits['gradient_audit']=once(edits['gradient_audit'],'        if getw(w, p + 416) != 0',
        '        field = 448\n        while field < 576:\n            if getw(w,p+field) != 0:\n                status = 9\n            field = field+8\n        if getw(w, p + 416) != 0')
    for name in ('partial_execute','backward_execute'):
        edits[name]=once(original[name],'    putw(w, p + 336, 1)\n',
            '    if incoming_seed(w,t,seed) == 0:\n        return 0\n    putw(w, p + 336, 1)\n')
    edits['backward_execute']=once(edits['backward_execute'],'            partial_apply_node(w, current)\n',
        '            gradient_apply(w, current, scratch)\n')
    edits['backward_execute']=once(edits['backward_execute'],
        '    if retain == 0:\n        return release_backward_scratch(w, limit, epoch)\n    return 1',
        '    return release_backward_scratch(w, limit, epoch)')
    authoring='\n\n'.join(edits.values())+'\n'+inputs[HERE/'incoming.py'].decode()
    (out/'overlay-authoring.py').write_text(authoring)
    try:
        fragment,boundaries=emit(authoring)
    except Exception as error:
        (out/'refusal.json').write_text(json.dumps({'stage':'authoring','error':repr(error),'native_attempted':False},indent=2)+'\n')
        raise
    combined=dict(old);combined.update(functions(fragment.encode()))
    routing={}
    for name in ('back_binary','conv_term','back_matmul_point','back_indexed_add'):
        body=combined[name]
        before=b'{"function":"gadd"}'
        assert body.count(before)==2,name
        altered=body.replace(before,b'{"function":"incoming_gadd_first"}',1).replace(before,b'{"function":"incoming_gadd_second"}',1)
        assert altered.replace(b'{"function":"incoming_gadd_first"}',before).replace(b'{"function":"incoming_gadd_second"}',before)==body
        combined[name]=altered
        routing[name]={'old_sha256':sha(body),'new_sha256':sha(altered),'old_gadd_calls':2,'inverse_byte_equal':True}
    for name,before,after in (
        ('concat_part','gadd','incoming_gadd_selected'),
        ('back_concat','concat_part','incoming_concat_part')):
        body=combined[name]
        needle=('{"function":"'+before+'"}').encode();replacement=('{"function":"'+after+'"}').encode()
        assert body.count(needle)==1,name
        altered=body.replace(needle,replacement)
        assert altered.replace(replacement,needle)==body
        combined[name]=altered
        routing[name]={'old_sha256':sha(body),'new_sha256':sha(altered),'old_calls':1,'inverse_byte_equal':True}
    raw=b'hia-asm.v1 hepta.hia.program.v8 gradient_incoming_storage\n'+b''.join(combined.values())
    (out/'runtime.hepta').write_bytes(raw)
    (out/'authoring-boundaries.json').write_text(json.dumps({'foundation_reemission_byte_identical':True,
        'foundation_boundaries':bounds,'new_boundaries':boundaries,'edge_callee_routing':routing},indent=2)+'\n')
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
    changed=[name for name in old if combined[name]!=old[name]]
    rows=[];report={'schema':'heptatorch.gradient-incoming-native-build.v1','status':'BUILDING',
        'foundation_source_sha256':FOUNDATION_SHA,'source_sha256':sha(raw),'functions':len(combined),
        'blocks':len(re.findall(rb'^block ',raw,re.M)),'instructions':raw.count(b' = '),
        'foundation_functions_unchanged':len(old)-len(changed),'changed_functions':changed,
        'added_functions':[name for name in combined if name not in old],
        'state_format':'HTGRAD02','backing_format':'HTOGRAD2','descriptor_bytes':576,
        'compiler_sha256':COMPILER_SHA,'linker_sha256':LINKER_SHA,
        'inputs_sha256':{str(p):sha(b) for p,b in inputs.items()},'units':rows,
        'scope':'Actual first-order incoming gradient Storage and Version ownership; hooks, create_graph and unqualified derivative allocation-layout paths remain open.'}
    def save():(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    save();objects=[]
    for name,body in units:
        (out/(name+'.hepta')).write_bytes(body)
        result=subprocess.run([str(compiler)],input=body,capture_output=True,env={'PATH':''},timeout=120)
        (out/(name+'.o')).write_bytes(result.stdout);(out/(name+'.stderr')).write_bytes(result.stderr)
        row={'name':name,'source_sha256':sha(body),'object_sha256':sha(result.stdout),
             'functions':len(functions(body)),'blocks':len(re.findall(rb'^block ',body,re.M)),
             'instructions':body.count(b' = '),'compiler_exit':result.returncode};rows.append(row);save()
        if result.returncode or result.stderr or not result.stdout.startswith(b'\x7fELF'):
            report['status']='NATIVE_COMPILER_REFUSAL';save();raise AssertionError(row)
        objects.append(result.stdout)
    request=b'HIASTKS1'+struct.pack('<3Q',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(obj))+obj for obj in objects)
    (out/'link.request').write_bytes(request)
    result=subprocess.run([str(linker)],input=request,capture_output=True,env={'PATH':''},timeout=120)
    (out/'link.stderr').write_bytes(result.stderr);report['link_exit']=result.returncode;save()
    if result.returncode or result.stderr or not result.stdout.startswith(b'\x7fELF'):
        (out/'link.refusal.stdout').write_bytes(result.stdout)
        report['status']='NATIVE_LINKER_REFUSAL';save();raise AssertionError(report['link_exit'])
    app=out/'tensor-gradient-incoming-session';app.write_bytes(result.stdout);app.chmod(0o500)
    for p,b in inputs.items():assert p.read_bytes()==b,p
    assert sha(compiler.read_bytes())==COMPILER_SHA and sha(linker.read_bytes())==LINKER_SHA
    report.update(status='PASS',native_sha256=sha(result.stdout),native_bytes=len(result.stdout));save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs_sha256','units')},indent=2))

if __name__=='__main__':
    main()
