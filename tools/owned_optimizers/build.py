"""Emit checked native Optimizer overlays over frozen RNG + nullable HIA."""
from pathlib import Path
import argparse, ast, copy, hashlib, importlib.util, json, re, struct, subprocess, sys

HERE=Path(__file__).resolve().parent
CORE_SHA='490fc144160f6e4440aed001363f86ccff56c335f9781e38b058fc6106b1be09'
COMPILER_SHA='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER_SHA='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
FUNCTION=re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)',re.M|re.S)
CALL=re.compile(rb'\{"function":"(\w+)"\}')
EXTERNAL_RNG={'rng_seed','rng_restore','rng_next','rng_next64','rng_unit32','rng_unit64'}
sha=lambda raw:hashlib.sha256(raw).hexdigest()
functions=lambda raw:{n.decode():body for body,n in FUNCTION.findall(raw)}

def replace_once(source,old,new):
    assert source.count(old)==1,(old,source.count(old))
    return source.replace(old,new,1)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('baseline','objects','common-authoring','nullable-authoring','partial-authoring','materialization-authoring',
                'repository','source-root','compiler','linker','output'):
        ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    base=a.baseline.resolve();repo=a.repository.resolve();upstream=a.source_root.resolve()
    compiler=a.compiler.resolve();linker=a.linker.resolve()
    layers=[a.objects.resolve()/'overlay-authoring.py',a.partial_authoring.resolve(),a.materialization_authoring.resolve(),
            a.common_authoring.resolve(),a.nullable_authoring.resolve()]
    paths=[Path(__file__),HERE/'profile.json',HERE/'identity.py',HERE/'optimizer.py',HERE/'audit.py',HERE/'edges.py',HERE/'compact_functions.py',
           base/'runtime.hepta',base/'rng03.hepta',base/'owners.hepta',
           repo/'tools/owned_tensors/authoring.py',repo/'tools/owned_tensors/profile.json',
           repo/'docs/functions.json']+layers
    inputs={p:p.read_bytes() for p in paths}
    frozen=json.loads(inputs[HERE/'profile.json'])
    for name,pin in frozen['authoring_sha256'].items():assert sha(inputs[HERE/name])==pin,name
    for path,name in zip(layers,('objects','partial','materialization','common','nullable')):
        assert sha(inputs[path])==frozen['authoring_layer_sha256'][name],path
    pins={base/'runtime.hepta':CORE_SHA,
          base/'rng03.hepta':'a3fa507339b40435285dc1fa65e3cd519b6a0921a4c93bccf6ea834bb24763a3',
          base/'owners.hepta':'9c02b75c44eee8459c476049f4d9ebe10935d9d9ed8b27f5842c699f8e6a0884',
          repo/'tools/owned_tensors/authoring.py':'2845fc2e3f03ccbadd5ea6c59734d0be6db6477bf8434d288259fab5ee570ba6',
          repo/'tools/owned_tensors/profile.json':'69642a730f9da8567a01cc5afd9d15963a49ff1ab83f7761039bb301d0d2d242',
          repo/'docs/functions.json':'6ecffeb9a13a0a8fe6b78f75728fc46fef1bed06f1c49196f2d7d855ee7effca',
          layers[0]:'4f3fa887e6372f19777ecde955d39716ae633a1bb9be7408c065d6e94e0fffbd',
          layers[1]:'21f758fa6a0fd040d71b89c9aaa96279c9df2a3ff4e879dfd360e15a143581c1',
          layers[2]:'8385161dd8719d53385759e0119441739cbe82ce5fea2b896d5cdd3ce9b31c78'}
    for path,pin in pins.items():assert sha(inputs[path])==pin,path
    profile=json.loads(inputs[repo/'tools/owned_tensors/profile.json'])
    for rel,pin in profile['upstream_authoring_sources'].items():
        path=upstream/rel;data=path.read_bytes();assert sha(data)==pin,path;inputs[path]=data
    assert sha(compiler.read_bytes())==COMPILER_SHA and sha(linker.read_bytes())==LINKER_SHA
    (out/'producer.py').write_bytes(inputs[Path(__file__)])
    for name in ('identity','optimizer','audit','edges'):
        (out/(name+'-authoring.py')).write_bytes(inputs[HERE/(name+'.py')])
    sys.path.insert(0,str(upstream))
    spec=importlib.util.spec_from_file_location('owned_optimizer_emitter',repo/'tools/owned_tensors/authoring.py')
    emitter=importlib.util.module_from_spec(spec);spec.loader.exec_module(emitter)
    spec=importlib.util.spec_from_file_location('optimizer_bounded_compaction',HERE/'compact_functions.py')
    compactor=importlib.util.module_from_spec(spec);spec.loader.exec_module(compactor)
    class NativeBuilder(emitter.NativeBuilder):
        def expr(self,node):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='move_within':
                w,dest,source,size=[self.expr(value) for value in node.args]
                self.op('buffer.move_within',[w,self.cast(dest,'usize'),self.cast(source,'usize'),self.cast(size,'usize')])
                return self.c(0)
            return super().expr(node)
    old=functions(inputs[base/'runtime.hepta']);rng=functions(inputs[base/'rng03.hepta'])
    assert len(old)==275 and len(rng)==17
    old_nodes={}
    for path in layers:
        old_nodes.update({n.name:n for n in ast.parse(inputs[path].decode()).body if isinstance(n,ast.FunctionDef)})
    tag_pairs={int.from_bytes(b'HTRNGP01','little'):int.from_bytes(b'HTRNGN01','little'),
               int.from_bytes(b'HTORNGP1','little'):int.from_bytes(b'HTORNGN1','little')}
    class Tags(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,int) and node.value in tag_pairs:
                return ast.copy_location(ast.Constant(tag_pairs[node.value]),node)
            return node
    for name in ('init_state','state_header','so_state_init','so_state_header'):
        old_nodes[name]=Tags().visit(copy.deepcopy(old_nodes[name]))
    aliases={}
    for name in ('record','indexed_add_backward_check','release_backward_saved'):
        alias=name+'_body'
        body=old[name].replace(b'fn '+name.encode()+b'(',b'fn '+alias.encode()+b'(',1)
        if name!='record':body=body.replace(b'{"function":"getw"}',b'{"function":"backward_getw"}')
        aliases[alias]=body
    signatures={}
    for name,body in {**old,**rng,**aliases}.items():
        m=re.search(r'^fn \w+\((.*?)\) -> \((.*?)\)',body.split(b'\n',1)[0].decode())
        signatures[name]=([p.split(':')[1] for p in m[1].split(',') if p],m[2])
    header=b'hia-asm.v1 hepta.hia.program.v8 optimizer_authoring_context\n'
    parsed=[emitter._parse_assembly_document((header+body).decode()) for body in {**old,**rng}.values()]
    context=copy.deepcopy(parsed[0]);context['functions']=[doc['functions'][0] for doc in parsed]
    alias_documents=[emitter._parse_assembly_document((header+body).decode())['functions'][0] for body in aliases.values()]
    def emit(source,bounded=False,extra=False):
        nodes=ast.parse(source).body;names={n:n for n in signatures};sigs=dict(signatures)
        for node in nodes:
            names[node.name]=node.name;sigs[node.name]=([v.annotation.id for v in node.args.args],node.returns.id)
        total=emitter.total_scalar_functions(nodes);new=[]
        for node in nodes:
            b=NativeBuilder(node,names,sigs,total);b.statements(node.body)
            new.append({'name':node.name,'params':[b.param(v) for v in b.params],
                        'returns':[b.returns],'effects':['read','write','trap'],'entry':'b0','blocks':b.blocks})
        selected={n.name for n in nodes};doc=copy.deepcopy(context)
        doc['functions']=[f for f in doc['functions'] if f['name'] not in selected]+new
        if extra:doc['functions']+=copy.deepcopy(alias_documents)
        records=[]
        if bounded:
            doc['functions'],records=compactor.compact(doc['functions'],selected)
        else:
            doc=emitter.compact_authoring_cfg(doc)
            doc['functions']=[f for f in doc['functions'] if f['name'] in selected]
        text=emitter._render_assembly_document(doc).split('\n',1)[1]
        mapping={n.name:{'name':n.name,'params':[{'name':v.arg,'type':v.annotation.id} for v in n.args.args],
                         'returns':n.returns.id} for n in nodes}
        return text,mapping,records
    layout_names=['object_stats','so_data_base','so_state_init','so_state_header','so_slot_lookup',
                  'so_extent_room','so_extent_remove','so_extent_allocate','so_extent_release','so_state_migrate',
                  'so_audit_count','so_audit_records','so_audit_references','so_audit_free','so_audit_extent',
                  'so_audit_extents','so_audit']
    regenerated={name:ast.unparse(old_nodes[name]) for name in layout_names+['init_state','state_header','descriptor','graph_node_release','entry','inspect_command','partial_apply_node','ensure_backward_work']}
    verified,_,_=emit('\n\n'.join(regenerated.values()))
    verified_functions=functions(verified.encode())
    for name in regenerated:
        assert verified_functions[name]==old[name],('baseline re-emission differs',name)
    independently_bounded,_,baseline_boundaries=emit('\n\n'.join(regenerated.values()),bounded=True)
    assert functions(independently_bounded.encode())==verified_functions,'new authoring boundary changed baseline bodies'
    edited=[]
    for name in layout_names:
        source=regenerated[name]
        source=source.replace('* 448','* 512').replace('* 464','* 528').replace('* 488','* 552')
        source=source.replace('kind > 7','kind > 8').replace('kind <= 7','kind <= 8').replace('kind < 7','kind < 8')
        if name in ('so_state_init','so_state_header'):
            source=replace_once(source,str(int.from_bytes(b'HTORNGN1','little')),str(int.from_bytes(b'HTOOPT02','little')))
        if name=='so_audit':
            source=replace_once(source,'    status = so_audit_records(w)\n',
                '    status = optimizer_audit(w)\n    if status == 0:\n        status = so_audit_records(w)\n')
        edited.append(source)
    for name in ('init_state','state_header'):
        source=regenerated[name]
        source=replace_once(source,str(int.from_bytes(b'HTRNGN01','little')),str(int.from_bytes(b'HTOPT002','little')))
        if name=='state_header':source=replace_once(source,'139264 + 2096 + 131072','139264 + 2352 + 131072')
        edited.append(source)
    descriptor=regenerated['descriptor']
    descriptor=replace_once(descriptor,'    if slot >= 65536:\n        return terror(w, 3)\n',
        '    if slot >= 65536:\n        return terror(w, 3)\n    if getw(w,120) == 18446744073709551615:\n        return terror(w,8)\n')
    descriptor=replace_once(descriptor,'    putw(w, p + 304, 1)\n',
        '    putw(w, p + 304, 1)\n    putw(w,120,getw(w,120)+1)\n    putw(w,p+320,getw(w,120))\n')
    edited.append(descriptor)
    release=replace_once(regenerated['graph_node_release'],'    hold_release(w, getw(w, p + 320))\n','')
    release=replace_once(release,'    p = dptr(w, t)\n',
        '    p = dptr(w, t)\n    if getw(w,p+312) != 0:\n        return terror(w,9)\n')
    entry=replace_once(regenerated['entry'],'        if code >= 60 and code <= 69:\n',
        '        if (code >= 71 and code <= 79) or code == 90:\n            next = owned_optimizer_command(s,o,w,32+i*64,at,end)\n        elif code >= 60 and code <= 69:\n')
    edited.append(entry)
    inspect=replace_once(regenerated['inspect_command'],'if getw(w, p + 40) != 0 or getw(w, p + 32) == 0:',
                         'if getw(w, p + 40) != 0:')
    edited.append(inspect)
    leaf_apply=replace_once(regenerated['partial_apply_node'],'    if op == 0:\n',
        '    if op == 0:\n        if getw(w,p+32) == 0:\n            return 1\n')
    edited.append(leaf_apply)
    ensure=replace_once(regenerated['ensure_backward_work'],'if getw(w, p + 40) == 0 and getw(w, p + 248) == 0:',
        'if getw(w, p + 40) == 0 and getw(w,p+32) != 0 and getw(w, p + 248) == 0:')
    edited.append(ensure)
    authoring='\n\n'.join(edited)+'\n'+''.join(inputs[HERE/(name+'.py')].decode()+'\n' for name in ('identity','optimizer','audit','edges'))
    (out/'overlay-authoring.py').write_text(authoring)
    try:fragment,mapping,boundaries=emit(authoring,bounded=True,extra=True)
    except Exception as error:
        (out/'refusal.json').write_text(json.dumps({'stage':'source authoring','error':repr(error),'native_attempted':False},indent=2)+'\n');raise
    replacement=functions(fragment.encode());combined=dict(old);combined.update(aliases);combined.update(replacement)
    derivative_read_overlays={}
    for name in old:
        if name.startswith('back_') or name in ('concat_copy','conv_point','pool_point'):
            before=combined[name]
            after=before.replace(b'{"function":"getw"}',b'{"function":"backward_getw"}')
            if after!=before:
                derivative_read_overlays[name]={'before_sha256':sha(before),'after_sha256':sha(after),
                    'getw_calls_rebound':before.count(b'{"function":"getw"}')}
                combined[name]=after
    raw=b'hia-asm.v1 hepta.hia.program.v8 tensor_owned_optimizer_runtime\n'+b''.join(combined.values())
    assert sha(raw)==frozen['source_sha256']
    (out/'runtime.hepta').write_bytes(raw);(out/'overlay.hepta').write_text(fragment)
    (out/'functions.json').write_text(json.dumps(mapping,indent=2)+'\n')
    (out/'authoring-boundaries.json').write_text(json.dumps({'schema':'heptatorch.bounded-authoring-bodies.v1',
        'aggregate_document_admission_claimed':False,'native_module_admission_required':True,
        'baseline_old_and_new_boundary_function_bytes_identical':True,
        'baseline_boundary_records':baseline_boundaries,'new_boundary_records':boundaries,
        'baseline_function_bytes_sha256':{name:sha(body) for name,body in old.items()},
        'emitted_function_bytes_sha256':{name:sha(body) for name,body in combined.items()},
        'real_body_aliases':{name:sha(body) for name,body in aliases.items()},
        'captured_edge_derivative_read_overlays':derivative_read_overlays},indent=2)+'\n')
    changed=[name for name in old if old[name]!=combined[name]]
    ordered=[];active=set();done=set()
    def visit(name):
        if name in EXTERNAL_RNG:return
        assert name in combined,name
        assert name not in active,('recursion',name)
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
        local=set(names);calls={n.decode() for name in names for n in CALL.findall(combined[name])}
        imports=sorted(calls-local);declarations=[]
        for name in imports:
            body=rng[name] if name in EXTERNAL_RNG else combined[name]
            declarations.append(re.sub(rb' entry\(\w+\)$',b' external()',body.split(b'\n',1)[0])+b'\n')
        assert len(names)+len(imports)<=256
        unit=f'optimizer_{i:02d}'
        source=f'hia-asm.v1 hepta.hia.program.v8 {unit}\nexports({",".join(names)})\n'.encode()+b''.join(declarations)+b''.join(combined[n] for n in names)
        units.append((unit,source))
    units.append(('owners',inputs[base/'owners.hepta']))
    objects=[];rows=[]
    report={'schema':'heptatorch.owned-optimizer-native-build.v1','status':'BUILDING',
            'baseline_source_sha256':CORE_SHA,'source_sha256':sha(raw),'functions':len(combined),
            'state_format':'HTOPT002','backing_format':'HTOOPT02','wire_request':'HTORCH03','wire_response':'HTORSP03',
            'blocks':len(re.findall(rb'^block ',raw,re.M)),'instructions':raw.count(b' = '),
            'baseline_functions_unchanged':len(old)-len(changed),'changed_baseline_functions':changed,
            'added_functions':[name for name in combined if name not in old],
            'compiler_sha256':COMPILER_SHA,'linker_sha256':LINKER_SHA,
            'inputs_sha256':{str(path):sha(data) for path,data in inputs.items()},'units':rows}
    def save():(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    for name,source in units:
        (out/(name+'.hepta')).write_bytes(source)
        result=subprocess.run([str(compiler)],input=source,capture_output=True,env={'PATH':''},timeout=120)
        (out/(name+'.o')).write_bytes(result.stdout);(out/(name+'.stderr')).write_bytes(result.stderr)
        row={'name':name,'source_sha256':sha(source),'source_bytes':len(source),'compiler_exit':result.returncode,
             'object_sha256':sha(result.stdout),'functions':len(functions(source)),
             'blocks':len(re.findall(rb'^block ',source,re.M)),'instructions':source.count(b' = '),
             'function_interfaces':[line.decode() for line in source.splitlines() if line.startswith(b'fn ')],
             'exports':[line.decode() for line in source.splitlines() if line.startswith(b'exports(')]}
        rows.append(row);save()
        if result.returncode!=0 or result.stderr or not result.stdout.startswith(b'\x7fELF'):
            report['status']='NATIVE_COMPILER_REFUSAL';save();raise AssertionError(row)
        objects.append(result.stdout)
    request=b'HIASTKS1'+struct.pack('<3Q',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(o))+o for o in objects)
    (out/'link.request').write_bytes(request)
    result=subprocess.run([str(linker)],input=request,capture_output=True,env={'PATH':''},timeout=120)
    (out/'link.stderr').write_bytes(result.stderr);report['link_exit']=result.returncode;save()
    if result.returncode!=0 or result.stderr or not result.stdout.startswith(b'\x7fELF'):
        (out/'link.refusal.stdout').write_bytes(result.stdout)
        report['status']='NATIVE_LINKER_REFUSAL';save();raise AssertionError(report['link_exit'])
    assert sha(result.stdout)==frozen['native_sha256']
    assert [(u['name'],u['source_sha256'],u['object_sha256']) for u in rows]==[(u['name'],u['source_sha256'],u['object_sha256']) for u in frozen['units']]
    app=out/'tensor-optimizer-session';app.write_bytes(result.stdout);app.chmod(0o500)
    for path,data in inputs.items():assert path.read_bytes()==data,path
    for (name,source),obj in zip(units,objects):
        assert (out/(name+'.hepta')).read_bytes()==source and (out/(name+'.o')).read_bytes()==obj
    assert (out/'runtime.hepta').read_bytes()==raw and app.read_bytes()==result.stdout
    assert sha(compiler.read_bytes())==COMPILER_SHA and sha(linker.read_bytes())==LINKER_SHA
    report.update(status='PASS',native_sha256=sha(result.stdout),native_bytes=len(result.stdout));save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs_sha256','units')},indent=2))

if __name__=='__main__':main()
