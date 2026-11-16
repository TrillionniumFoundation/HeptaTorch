"""Emit and natively compile retained-gradient HIA over the frozen combined05.

Python verifies source custody and emits HIA. All numerical, ownership,
reference-count, allocator and Tensor execution is performed by native HIA.
"""
from pathlib import Path
import argparse, ast, copy, hashlib, importlib.util, json, re, struct, subprocess, sys

HERE=Path(__file__).resolve().parent
PARENT='1fed8ea7ba5bf755a9df9ca1e61d34cfd4712cc2fd66e0728493091c81c24dae'
COMPILER='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
FUNCTION=re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)',re.M|re.S)
CALL=re.compile(rb'\{"function":"(\w+)"\}')
EXTERNAL={'rng_seed','rng_restore','rng_next','rng_next64','rng_unit32','rng_unit64'}
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def functions(raw):
    pairs=FUNCTION.findall(raw);result={n.decode():b for b,n in pairs}
    assert len(pairs)==len(result)
    return result
def once(source,old,new):
    assert source.count(old)==1,(old,source.count(old))
    return source.replace(old,new,1)
def literal(tag):return b'{"value":'+str(int.from_bytes(tag,'little')).encode()+b'}'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','repository','upstream','compactor','compiler','linker','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    parent=a.parent.resolve();repo=a.repository.resolve();up=a.upstream.resolve()
    compiler=a.compiler.resolve();linker=a.linker.resolve();compactor=a.compactor.resolve()
    paths=[Path(__file__).resolve(),HERE/'originals.py',HERE/'original-authoring-custody.json',
        HERE/'retained.py',HERE/'layout.py',parent/'runtime.hepta',parent/'rng03.hepta',parent/'owners.hepta',
        repo/'tools/owned_tensors/authoring.py',repo/'tools/owned_tensors/profile.json',compactor]
    inputs={path:path.read_bytes() for path in paths}
    expected={parent/'runtime.hepta':PARENT,
        parent/'rng03.hepta':'a3fa507339b40435285dc1fa65e3cd519b6a0921a4c93bccf6ea834bb24763a3',
        parent/'owners.hepta':'9c02b75c44eee8459c476049f4d9ebe10935d9d9ed8b27f5842c699f8e6a0884',
        repo/'tools/owned_tensors/authoring.py':'2845fc2e3f03ccbadd5ea6c59734d0be6db6477bf8434d288259fab5ee570ba6',
        repo/'tools/owned_tensors/profile.json':'69642a730f9da8567a01cc5afd9d15963a49ff1ab83f7761039bb301d0d2d242',
        compactor:'36730b55b58c29b210d921a426c62eab50d9f579ad5d7a1e4199b38799218f37'}
    custody=json.loads(inputs[HERE/'original-authoring-custody.json'])
    expected[HERE/'originals.py']=custody['selected_authoring_sha256']
    for path,pin in expected.items():assert sha(inputs[path])==pin,(path,sha(inputs[path]))
    assert sha(compiler.read_bytes())==COMPILER and sha(linker.read_bytes())==LINKER
    profile=json.loads(inputs[repo/'tools/owned_tensors/profile.json'])
    for name,pin in profile['upstream_authoring_sources'].items():
        path=up/name;inputs[path]=path.read_bytes();assert sha(inputs[path])==pin,path
    for name in ('build.py','originals.py','original-authoring-custody.json','retained.py','layout.py'):
        (out/name).write_bytes(inputs[HERE/name])
    sys.path.insert(0,str(up))
    spec=importlib.util.spec_from_file_location('retained_emitter',repo/'tools/owned_tensors/authoring.py')
    emitter=importlib.util.module_from_spec(spec);spec.loader.exec_module(emitter)
    spec=importlib.util.spec_from_file_location('retained_compactor',compactor)
    compact=importlib.util.module_from_spec(spec);spec.loader.exec_module(compact)
    old=functions(inputs[parent/'runtime.hepta']);rng=functions(inputs[parent/'rng03.hepta'])
    assert (len(old),len(rng))==(428,17)
    combined=dict(old)
    aliases={'gradient_gc_enqueue':'retained_gc_enqueue_original',
        'tensor_row_release':'retained_row_release_original','publish_tensor':'retained_publish_original'}
    alias_receipt={}
    for name,alias in aliases.items():
        combined[alias]=once(old[name],('fn '+name+'(').encode(),('fn '+alias+'(').encode())
        assert once(combined[alias],('fn '+alias+'(').encode(),('fn '+name+'(').encode())==old[name]
        alias_receipt[name]={'original_sha256':sha(old[name]),'alias':alias,'alias_sha256':sha(combined[alias]),'inverse_exact':True}
    context=[];signatures={}
    for name,body in {**combined,**rng}.items():
        match=re.match(rb'fn \w+\(([^)]*)\) -> \(([^)]*)\)',body)
        signatures[name]=([x.split(b':')[1].decode() for x in match[1].split(b',') if x],match[2].decode())
        context.extend(emitter._parse_assembly_document((b'hia-asm.v1 hepta.hia.program.v8 retained_context\n'+body).decode())['functions'])
    records=[]
    def emit(source):
        nodes=ast.parse(source).body;sigs=dict(signatures)
        sigs.update({n.name:([p.annotation.id for p in n.args.args],n.returns.id) for n in nodes})
        names={n:n for n in sigs};total=emitter.total_scalar_functions(nodes);emitted=[]
        for node in nodes:
            builder=emitter.NativeBuilder(node,names,sigs,total);builder.statements(node.body)
            emitted.append({'name':node.name,'params':[builder.param(p) for p in builder.params],
                'returns':[builder.returns],'effects':['read','write','trap'],'entry':'b0','blocks':builder.blocks})
        selected={n.name for n in nodes}
        document=[f for f in copy.deepcopy(context) if f['name'] not in selected]+emitted
        checked,checks=compact.compact(document,selected);records.append(checks)
        return emitter._render_assembly_document({'schema':'hepta.hia.program.v8','module':'retained_overlay','functions':checked}).encode()
    originals={n.name:ast.unparse(n) for n in ast.parse(inputs[HERE/'originals.py']).body}
    patches=dict(originals)
    for name in ('tgrad','zero_grad','optimizer_fill_group','owned_adam_fill_group'):
        patches[name]=once(patches[name],'if getw(w, p + 40) != 0:',
            'if retained_parameter_blocker(w, t) != 0:')
    patches['zero_grad_none']=once(patches['zero_grad_none'],'if getw(w, dptr(w, t) + 40) != 0:',
        'if retained_parameter_blocker(w, t) != 0:')
    for name in ('optimizer_audit_entry','owned_adam_audit_entry'):
        patches[name]=once(patches[name],'if p == 0 or getw(w, p + 40) != 0:',
            'if p == 0 or retained_parameter_blocker(w, node) != 0:')
    patches['so_audit_records']=once(patches['so_audit_records'],'so_uget(w, p + 56) != 0 or so_uget(w, p + 48) == 0',
        'so_uget(w, p + 56) > 1 or so_uget(w, p + 48) == 0')
    patches['gradient_object_command']=once(patches['gradient_object_command'],'        else:\n            terror(w, 4)',
        '        elif code == 93:\n            result = retained_enable(w, resolve_handle(w, a))\n        elif code == 94:\n            result = retained_query(w, resolve_handle(w, a))\n        else:\n            terror(w, 4)')
    partial=patches['partial_execute']
    partial=once(partial,'    queue = alloc(w, queue_bytes + stage_bytes)',
        '    retained_bytes = (t + 1) * 8\n    queue = alloc(w, queue_bytes + retained_bytes + stage_bytes)')
    partial=once(partial,'    if ensure_backward_work(w, t, epoch) == 0:\n        return 0',
        '    if ensure_backward_work(w, t, epoch) == 0:\n        return 0\n    retained_table = queue + queue_bytes\n    if retained_reserve(w, retained_table, t, epoch) == 0:\n        return 0')
    partial=once(partial,'        if getw(w, 0) == 0:\n            partial_check_node(w, current)',
        '        if getw(w, 0) == 0:\n            retained_apply(w, retained_table, current)\n        if getw(w, 0) == 0:\n            partial_check_node(w, current)')
    partial=once(partial,'gradient_apply(w, current, queue + queue_bytes)',
        'gradient_apply(w, current, queue + queue_bytes + retained_bytes)')
    partial=once(partial,'    if hold_release(w, queue_owner) == 0:',
        '    if retained_release_reservations(w, retained_table, t) == 0:\n        return 0\n    if hold_release(w, queue_owner) == 0:')
    patches['partial_execute']=partial
    backward=patches['backward_execute']
    backward=once(backward,'    scratch = alloc(w, stage_bytes + 1)',
        '    retained_bytes = (limit + 1) * 8\n    scratch = alloc(w, stage_bytes + 1 + retained_bytes)')
    backward=once(backward,'    scratch_owner = getw(w, 80)',
        '    scratch_owner = getw(w, 80)\n    if retained_reserve(w, scratch, limit, epoch) == 0:\n        return 0')
    backward=once(backward,'            if op != 0:\n                gradient_apply(w, current, scratch)',
        '            if op != 0:\n                if retained_apply(w, scratch, current) == 0:\n                    return 0\n                gradient_apply(w, current, scratch + retained_bytes)')
    backward=once(backward,'            gradient_apply(w, current, scratch)',
        '            gradient_apply(w, current, scratch + retained_bytes)')
    backward=once(backward,'    if hold_release(w, scratch_owner) == 0:\n        return 0\n    return release_backward_scratch(w, limit, epoch)',
        '    if release_backward_scratch(w, limit, epoch) == 0:\n        return 0\n    if retained_release_reservations(w, scratch, limit) == 0:\n        return 0\n    return hold_release(w, scratch_owner)')
    patches['backward_execute']=backward
    try:
        for name,source in originals.items():
            emitted=functions(emit(source))
            assert emitted=={name:old[name]},('pinned authoring differs',name,sha(emitted.get(name,b'')),sha(old[name]))
        authored='\n\n'.join(patches.values())+'\n\n'+inputs[HERE/'retained.py'].decode()+'\n\n'+inputs[HERE/'layout.py'].decode()
        (out/'overlay-authoring.py').write_text(authored)
        fragment=emit(authored);overlay=functions(fragment)
    except Exception as error:
        (out/'refusal.json').write_text(json.dumps({'stage':'authoring','error':repr(error),'native_attempted':False},indent=2)+'\n');raise
    combined.update(overlay)
    tag_edits={}
    for name in ('init_state','state_header','so_state_init','so_state_header'):
        oldtag=b'HTGAD002' if name in ('init_state','state_header') else b'HTOGAD02'
        newtag=b'HTRET001' if name in ('init_state','state_header') else b'HTORET01'
        before=combined[name];combined[name]=once(before,literal(oldtag),literal(newtag))
        assert once(combined[name],literal(newtag),literal(oldtag))==before
        tag_edits[name]={'before_sha256':sha(before),'after_sha256':sha(combined[name]),'only_literal_changed':True}
    raw=b'hia-asm.v1 hepta.hia.program.v8 retained_gradient_runtime\n'+b''.join(combined.values())
    (out/'runtime.hepta').write_bytes(raw);(out/'overlay.hepta').write_bytes(fragment)
    (out/'authoring-custody.json').write_text(json.dumps({'all_selected_originals_reemit_byte_identically':True,
        'parent_sha256':PARENT,'compaction_records':records,'alias_inverse':alias_receipt,'tags':tag_edits,
        'changes':{name:{'before':originals[name],'after':patches[name]} for name in originals}},indent=2)+'\n')
    mapping=[{'name':n,'sha256':sha(b),'parent_sha256':sha(old[n]) if n in old else None,
        'unchanged_parent':b==old.get(n)} for n,b in combined.items()]
    (out/'source-map.json').write_text(json.dumps(mapping,indent=2)+'\n')
    ordered=[];done=set();active=set()
    def visit(name):
        if name in EXTERNAL:return
        assert name in combined and name not in active,('missing/recursive function',name)
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
            body=rng[name] if name in EXTERNAL else combined[name]
            declarations.append(re.sub(rb' entry\(\w+\)$',b' external()',body.split(b'\n')[0])+b'\n')
        assert len(names)+len(imports)<=256
        unit=f'retained_gradient_{i:02d}'
        body=f'hia-asm.v1 hepta.hia.program.v8 {unit}\nexports({",".join(names)})\n'.encode()+b''.join(declarations)+b''.join(combined[n] for n in names)
        units.append((unit,body))
    units.append(('owners',inputs[parent/'owners.hepta']))
    report={'schema':'heptatorch.retained-gradient-native-build.v1','status':'BUILDING',
        'parent_source_sha256':PARENT,'source_sha256':sha(raw),'functions':len(combined),
        'blocks':len(re.findall(rb'^block ',raw,re.M)),'instructions':raw.count(b' = '),
        'state_format':'HTRET001','backing_format':'HTORET01','descriptor_bytes':576,
        'compiler_sha256':COMPILER,'linker_sha256':LINKER,'units':[],
        'inputs_sha256':{str(path):sha(data) for path,data in inputs.items()},
        'unchanged_parent_functions':sum(r['unchanged_parent'] for r in mapping),
        'reference_status':'Hosted reference exists; numerical and lifecycle qualification of this candidate is separate.',
        'full_rewrite_complete':False,'pytorch_api_denominator':2590}
    def save():(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    save();objects=[]
    for name,body in units:
        (out/(name+'.hepta')).write_bytes(body)
        result=subprocess.run([str(compiler)],input=body,capture_output=True,env={'PATH':''},timeout=120)
        (out/(name+'.o')).write_bytes(result.stdout);(out/(name+'.stderr')).write_bytes(result.stderr)
        report['units'].append({'name':name,'source_sha256':sha(body),'object_sha256':sha(result.stdout),
            'functions':len(functions(body)),'blocks':len(re.findall(rb'^block ',body,re.M)),
            'instructions':body.count(b' = '),'compiler_exit':result.returncode});save()
        assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF'),(name,result.returncode)
        objects.append(result.stdout)
    request=b'HIASTKS1'+struct.pack('<3Q',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(o))+o for o in objects)
    (out/'link.request.bin').write_bytes(request)
    result=subprocess.run([str(linker)],input=request,capture_output=True,env={'PATH':''},timeout=120)
    (out/'link.stderr').write_bytes(result.stderr);report['link_exit']=result.returncode;save()
    assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
    binary=out/'tensor-retained-session';binary.write_bytes(result.stdout);binary.chmod(0o500)
    assert all(path.read_bytes()==data for path,data in inputs.items())
    assert sha(compiler.read_bytes())==COMPILER and sha(linker.read_bytes())==LINKER
    report.update(status='PASS',native_sha256=sha(result.stdout),native_bytes=len(result.stdout),module_count=len(units));save()
    print(json.dumps({k:report[k] for k in ('status','source_sha256','native_sha256','functions','blocks','instructions','module_count')}))

if __name__=='__main__':main()
