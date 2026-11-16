"""Compose actual gradient-owned, Adam and distribution HIA in one checked owner.

The host verifies and emits native HIA. It does not execute Tensor, gradient,
optimizer, random, ownership, allocation or numerical operations.
"""
from pathlib import Path
import argparse, ast, copy, hashlib, importlib.util, json, re, struct, subprocess, sys

HERE=Path(__file__).resolve().parent
BASE='7c68a96b08aa9a21a52e2fd12be912c5f47dbde889e462f9a0dbb38cb3c09edf'
GRAD='1b18da2cf9791e1b20982f9706434c1359ea194dd983dc6c66eb005f152e77b5'
ADAM_DIST='5d3a3c047d024a3880d257a317a04c2166a2e87bbc6b248bef62649a88546873'
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
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','gradient-build','adam-distribution-build','public-build','incoming-authoring','repository','upstream','compiler','linker','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    base=a.baseline.resolve();grad_dir=a.gradient_build.resolve();ad_dir=a.adam_distribution_build.resolve()
    public_dir=a.public_build.resolve();incoming=a.incoming_authoring.resolve()
    repo=a.repository.resolve();up=a.upstream.resolve();compiler=a.compiler.resolve();linker=a.linker.resolve()
    paths=[Path(__file__).resolve(),HERE/'zero_layout.py',HERE/'preserve_cache.py',HERE/'pointwise_layout.py',
           HERE/'scalar_add_bits.py',HERE/'binary-original.py',HERE/'scalar_add_preserving_bits.qualified.hepta',
           HERE/'scalar_add_nograd.qualified.hepta',base/'runtime.hepta',grad_dir/'runtime.hepta',ad_dir/'runtime.hepta',
           grad_dir/'rng03.hepta',grad_dir/'owners.hepta',ad_dir/'rng03.hepta',ad_dir/'owners.hepta',
           public_dir/'overlay-authoring.py',incoming/'incoming.py',incoming.parent/'gradient-object-extension/baseline-authoring/compact_functions.py',
           repo/'tools/owned_tensors/authoring.py',repo/'tools/owned_tensors/profile.json']
    inputs={path:path.read_bytes() for path in paths}
    expected={base/'runtime.hepta':BASE,grad_dir/'runtime.hepta':GRAD,ad_dir/'runtime.hepta':ADAM_DIST,
        HERE/'binary-original.py':'7246a5c3f3b51478860ade5f3f110971aa95c22161b31909638d6580c6010602',
        HERE/'scalar_add_bits.py':'01058054297bceabf42d1c34d91cdb15a042cc1560e89dd50387f71bc98fcc0c',
        HERE/'scalar_add_preserving_bits.qualified.hepta':'b9d272f526920879e66970641f60cfcc0fc1d71aaf3c285e2781a5a909c6a436',
        HERE/'scalar_add_nograd.qualified.hepta':'230b6c97281c694f0d4dbc19c251291d26604716266e6f05b061d486feb070de',
        public_dir/'overlay-authoring.py':'50c6ec6b0b36a07e779076032a61921cbf84b520820e556e6833fbdfaf84eb60',
        incoming/'incoming.py':'1b84dc71a77381f4fdce1fd5f470d2509086bb1fd1e527e6b01a1a2956d884b5',
        incoming.parent/'gradient-object-extension/baseline-authoring/compact_functions.py':'36730b55b58c29b210d921a426c62eab50d9f579ad5d7a1e4199b38799218f37',
        repo/'tools/owned_tensors/authoring.py':'2845fc2e3f03ccbadd5ea6c59734d0be6db6477bf8434d288259fab5ee570ba6',
        repo/'tools/owned_tensors/profile.json':'69642a730f9da8567a01cc5afd9d15963a49ff1ab83f7761039bb301d0d2d242'}
    for directory in (grad_dir,ad_dir):
        expected[directory/'rng03.hepta']='a3fa507339b40435285dc1fa65e3cd519b6a0921a4c93bccf6ea834bb24763a3'
        expected[directory/'owners.hepta']='9c02b75c44eee8459c476049f4d9ebe10935d9d9ed8b27f5842c699f8e6a0884'
    for path,pin in expected.items():assert sha(inputs[path])==pin,(path,sha(inputs[path]))
    assert sha(compiler.read_bytes())==COMPILER and sha(linker.read_bytes())==LINKER
    profile=json.loads(inputs[repo/'tools/owned_tensors/profile.json'])
    for name,pin in profile['upstream_authoring_sources'].items():
        path=up/name;inputs[path]=path.read_bytes();assert sha(inputs[path])==pin,path
    (out/'producer.py').write_bytes(inputs[Path(__file__).resolve()])
    (out/'zero-layout-authoring.py').write_bytes(inputs[HERE/'zero_layout.py'])
    (out/'preserve-cache-authoring.py').write_bytes(inputs[HERE/'preserve_cache.py'])
    (out/'pointwise-layout-authoring.py').write_bytes(inputs[HERE/'pointwise_layout.py'])
    (out/'scalar-add-bits-authoring.py').write_bytes(inputs[HERE/'scalar_add_bits.py'])
    (out/'binary-original-authoring.py').write_bytes(inputs[HERE/'binary-original.py'])
    sys.path.insert(0,str(up))
    spec=importlib.util.spec_from_file_location('combined_gradient_emitter',repo/'tools/owned_tensors/authoring.py')
    emitter=importlib.util.module_from_spec(spec);spec.loader.exec_module(emitter)
    spec=importlib.util.spec_from_file_location('combined_gradient_compactor',incoming.parent/'gradient-object-extension/baseline-authoring/compact_functions.py')
    compact=importlib.util.module_from_spec(spec);spec.loader.exec_module(compact)
    old=functions(inputs[base/'runtime.hepta']);grad=functions(inputs[grad_dir/'runtime.hepta']);ad=functions(inputs[ad_dir/'runtime.hepta'])
    rng=functions(inputs[grad_dir/'rng03.hepta'])
    assert (len(old),len(grad),len(ad),len(rng))==(315,355,378,17)
    conflicts={name for name in old if grad[name]!=old[name] and ad[name]!=old[name] and grad[name]!=ad[name]}
    assert conflicts=={'entry','init_state','state_header','so_state_init','so_state_header','partial_apply_node'}
    combined=dict(grad)
    for name,body in ad.items():
        if name not in old or (body!=old[name] and name not in conflicts):
            assert name not in grad or grad[name]==old[name],name
            combined[name]=body
    assert len(combined)==418
    prior_alias=combined.pop('partial_apply_node_adam')
    assert prior_alias.replace(b'fn partial_apply_node_adam(',b'fn partial_apply_node(',1)==old['partial_apply_node']
    alias=once(grad['partial_apply_node'],b'fn partial_apply_node(',b'fn partial_apply_node_gradient(')
    combined['partial_apply_node_gradient']=alias
    public_nodes={n.name:n for n in ast.parse(inputs[public_dir/'overlay-authoring.py']).body if isinstance(n,ast.FunctionDef)}
    incoming_nodes={n.name:n for n in ast.parse(inputs[incoming/'incoming.py']).body if isinstance(n,ast.FunctionDef)}
    original_entry=ast.unparse(public_nodes['entry'])
    original_describe=ast.unparse(incoming_nodes['incoming_describe'])
    original_binary=inputs[HERE/'binary-original.py'].decode()
    binary=once(original_binary,'    n = getw(w, dptr(w, t) + 8)\n',
        '    if pointwise_layout(w,t,a,b) == 0:\n        return 0\n    n = getw(w, dptr(w, t) + 8)\n')
    binary=once(binary,'put4(w, data + i * 4, scalar_binary(op, x, y))',
        'put4(w, taddr(w,t,i), scalar_binary(op, x, y))')
    entry=once(original_entry,'elif code >= 71 and code <= 79 or code == 90:',
        'elif code >= 71 and code <= 79 or code == 90 or code == 95:')
    entry=once(entry,'        elif code >= 60 and code <= 69:',
        '        elif code >= 80 and code <= 89:\n            if code == 86 or code == 87:\n                next = dist_partial_command(s,o,w,scratch,32+i*64,at)\n            else:\n                next = dist_command(s,o,w,scratch,32+i*64,at)\n        elif code >= 60 and code <= 69:')
    entry=once(entry,'if getr(s, 32 + i * 64) == 58 and getw(o, at + 16) != 0:',
        'if (code == 58 or code == 86 or code == 87) and getw(o, at + 16) != 0:')
    describe=once(original_describe,'    if op == 26 or op == 27:',
        '    if op == 41:\n        return incoming_zero_layout(w,t,3856)\n    if op == 26 or op == 27:')
    originals={name:ast.unparse(incoming_nodes[name]) for name in
        ('incoming_construct','incoming_drop_cache','incoming_can_accumulate','ensure_backward_work')}
    patches={}
    patches['ensure_backward_work']=once(originals['ensure_backward_work'],'    i = 1\n',
        '    if incoming_mark_preserve_targets(w,limit,epoch) == 0:\n        return 0\n    i = 1\n')
    patches['ensure_backward_work']=once(patches['ensure_backward_work'],
        '            putw(w, p + 536, so_uget(w, hp + 16))\n',
        '            putw(w, p + 536, so_uget(w, hp + 16))\n            if getw(w,p+544) == 1:\n                if incoming_reserve_preserve_storage(w,i) == 0:\n                    return 0\n')
    patches['incoming_drop_cache']=once(originals['incoming_drop_cache'],'    putw(w, p + 160, 0)\n',
        '    if hold_release(w,getw(w,p+552)) == 0:\n        return 0\n    other = getw(w,p+568)\n    if other != 0 and so_version_release(w,other) != 0:\n        return terror(w,9)\n    putw(w,p+544,0)\n    putw(w,p+552,0)\n    putw(w,p+560,0)\n    putw(w,p+568,0)\n    putw(w, p + 160, 0)\n')
    patches['incoming_can_accumulate']=once(originals['incoming_can_accumulate'],
        '    return choose(so_uget(w, sp + 40) == expected, 1, 0)',
        '    if getw(w,p+552) != 0 and getw(w,p+560) == getw(w,p+448):\n        expected = expected+1\n    return choose(so_uget(w, sp + 40) == expected, 1, 0)')
    patches['incoming_construct']=once(originals['incoming_construct'],'    if getw(w, 3808) == 0:\n',
        '    physical = 0\n    if getw(w, 3808) == 0:\n')
    patches['incoming_construct']=once(patches['incoming_construct'],'        origin = parent\n',
        '        origin = parent\n        if incoming_contiguous(w,parent,3856) == 0:\n            sid = getw(w,p+560)\n            vid = getw(w,p+568)\n            if getw(w,p+552) == 0 or sid == 0 or vid == 0 or sid == getw(w,p+536):\n                return terror(w,9)\n            physical = 1\n')
    patches['incoming_construct']=once(patches['incoming_construct'],
        '    if getw(w, 3808) == 0 and getw(w, 3888) != 0:\n',
        '    if physical != 0:\n        j = 0\n        while j < getw(w,p+8):\n            put4(w,incoming_address(w,parent,j),get4(w,getw(w,p+160)+j*4))\n            j = j+1\n    if getw(w, 3808) == 0 and getw(w, 3888) != 0:\n')
    wrapper='''def partial_apply_node(w:mutbuf, t:u64) -> u64:
    if getw(w,dptr(w,t)+40) == 41:
        dist_backward_bernoulli(w,t)
        if getw(w,0) != 0:
            return 0
        return 1
    return partial_apply_node_gradient(w,t)
'''
    context=[];signatures={}
    for name,body in {**combined,**rng}.items():
        m=re.match(rb'fn \w+\(([^)]*)\) -> \(([^)]*)\)',body)
        signatures[name]=([x.split(b':')[1].decode() for x in m[1].split(b',') if x],m[2].decode())
        context.extend(emitter._parse_assembly_document((b'hia-asm.v1 hepta.hia.program.v8 gradient_composition_context\n'+body).decode())['functions'])
    boundaries=[]
    def emit(source):
        nodes=ast.parse(source).body;sigs=dict(signatures)
        sigs.update({n.name:([p.annotation.id for p in n.args.args],n.returns.id) for n in nodes})
        names={n:n for n in sigs};total=emitter.total_scalar_functions(nodes);emitted=[]
        for node in nodes:
            builder=emitter.NativeBuilder(node,names,sigs,total);builder.statements(node.body)
            emitted.append({'name':node.name,'params':[builder.param(p) for p in builder.params],'returns':[builder.returns],
                'effects':['read','write','trap'],'entry':'b0','blocks':builder.blocks})
        selected={n.name for n in nodes}
        document=copy.deepcopy(context)
        document=[f for f in document if f['name'] not in selected]+emitted
        checked,records=compact.compact(document,selected);boundaries.append(records)
        return emitter._render_assembly_document({'schema':'hepta.hia.program.v8','module':'gradient_composition_overlay','functions':checked}).encode()
    try:
        for name,source in [('entry',original_entry),('incoming_describe',original_describe),('binary',original_binary),*originals.items()]:
            assert functions(emit(source))=={name:grad[name]},('pinned authoring differs',name)
        authored=entry+'\n\n'+wrapper+'\n\n'+describe+'\n\n'+binary+'\n\n'+'\n\n'.join(patches.values())+'\n\n'+inputs[HERE/'zero_layout.py'].decode()+'\n\n'+inputs[HERE/'preserve_cache.py'].decode()+'\n\n'+inputs[HERE/'pointwise_layout.py'].decode()+'\n\n'+inputs[HERE/'scalar_add_bits.py'].decode()
        (out/'overlay-authoring.py').write_text(authored)
        fragment=emit(authored);overlay=functions(fragment)
    except Exception as error:
        (out/'refusal.json').write_text(json.dumps({'stage':'authoring','error':repr(error),'native_attempted':False},indent=2)+'\n');raise
    assert set(overlay)=={'entry','partial_apply_node','incoming_describe','incoming_zero_axis_order','incoming_zero_layout',
        'incoming_mark_preserve_targets','incoming_reserve_preserve_storage','binary',
        'pointwise_compare_axes','pointwise_axis_order','pointwise_channels_last','pointwise_write_order','pointwise_layout',
        'scalar_add_preserving_bits',*originals}
    combined.update(overlay)
    old_scalar_add=combined['scalar_add_nograd']
    assert sha(old_scalar_add)=='0753da5f592b0a53e114437a4c468cde53d5295f8ea63fb1e0ba21b55f564218'
    combined['scalar_add_nograd']=once(old_scalar_add,b'{"function":"addf"}',b'{"function":"scalar_add_preserving_bits"}')
    assert once(combined['scalar_add_nograd'],b'{"function":"scalar_add_preserving_bits"}',b'{"function":"addf"}')==old_scalar_add
    assert combined['scalar_add_preserving_bits']==inputs[HERE/'scalar_add_preserving_bits.qualified.hepta']
    assert combined['scalar_add_nograd']==inputs[HERE/'scalar_add_nograd.qualified.hepta']
    tag_edits={}
    for name in ('init_state','state_header','so_state_init','so_state_header'):
        oldtag=b'HTGRAD02' if name in ('init_state','state_header') else b'HTOGRAD2'
        newtag=b'HTGAD002' if name in ('init_state','state_header') else b'HTOGAD02'
        before=combined[name];combined[name]=once(before,literal(oldtag),literal(newtag))
        assert once(combined[name],literal(newtag),literal(oldtag))==before
        tag_edits[name]={'before_sha256':sha(before),'after_sha256':sha(combined[name]),'only_literal_changed':True}
    assert len(combined)==428
    assert sum(name.startswith('dist_') for name in combined)==25
    assert all(combined[name]==body for name,body in ad.items() if name.startswith('dist_'))
    assert combined['owned_optimizer_gradient_value']==ad['owned_optimizer_gradient_value']
    assert combined['tgrad']==grad['tgrad']
    assert combined['owned_adam_new_moment']==ad['owned_adam_new_moment']
    assert combined['optimizer_new_momentum']==grad['optimizer_new_momentum']
    raw=b'hia-asm.v1 hepta.hia.program.v8 tensor_gradient_adam_distributions_runtime\n'+b''.join(combined.values())
    (out/'runtime.hepta').write_bytes(raw);(out/'overlay.hepta').write_bytes(fragment)
    (out/'body-compaction.json').write_text(json.dumps({'original_grad_entry_and_description_byte_identical':True,
        'selected_real_bodies_compacted_without_checker_limit_change':True,'records':boundaries,'tags':tag_edits,
        'old_adam_alias_replaced_by_real_gradient_dispatcher':True,'all_25_distribution_bodies_unchanged':True,
        'original_binary_author_reemits_byte_identically':True,
        'binary_changes_only_output_stride_selection_and_physical_destination_addressing':True,
        'scalar_add_nograd_change_is_one_reversible_callee_substitution':True,
        'scalar_add_nograd_original_body_sha256':sha(old_scalar_add),
        'all_finite_scalar_add_arithmetic_still_uses_original_addf':True},indent=2)+'\n')
    mapping=[]
    for name,body in combined.items():
        mapping.append({'name':name,'sha256':sha(body),'baseline_sha256':sha(old[name]) if name in old else None,
            'gradient_sha256':sha(grad[name]) if name in grad else None,'adam_distribution_sha256':sha(ad[name]) if name in ad else None,
            'unchanged_gradient_body':body==grad.get(name),'unchanged_adam_distribution_body':body==ad.get(name)})
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
    units=[('rng03',inputs[grad_dir/'rng03.hepta'])]
    for i,names in enumerate(groups):
        calls={v.decode() for name in names for v in CALL.findall(combined[name])};imports=sorted(calls-set(names));declarations=[]
        for name in imports:
            body=rng[name] if name in EXTERNAL else combined[name]
            declarations.append(re.sub(rb' entry\(\w+\)$',b' external()',body.split(b'\n')[0])+b'\n')
        assert len(names)+len(imports)<=256
        unit=f'gradient_adam_distributions_{i:02d}'
        body=f'hia-asm.v1 hepta.hia.program.v8 {unit}\nexports({",".join(names)})\n'.encode()+b''.join(declarations)+b''.join(combined[n] for n in names)
        units.append((unit,body))
    units.append(('owners',inputs[grad_dir/'owners.hepta']))
    report={'schema':'heptatorch.gradient-adam-distributions-native-build.v1','status':'BUILDING',
        'baseline_source_sha256':BASE,'gradient_source_sha256':GRAD,'adam_distribution_source_sha256':ADAM_DIST,
        'source_sha256':sha(raw),'functions':len(combined),'blocks':len(re.findall(rb'^block ',raw,re.M)),'instructions':raw.count(b' = '),
        'state_format':'HTGAD002','backing_format':'HTOGAD02','descriptor_bytes':576,
        'compiler_sha256':COMPILER,'linker_sha256':LINKER,'units':[],
        'inputs_sha256':{str(path):sha(data) for path,data in inputs.items()},
        'scope':'Actual same owned state for public gradient ownership, owned SGD/Adam/AdamW, Generator and qualified CPU distributions. Execution qualification is separate.',
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
    binary=out/'tensor-gradient-adam-distributions-session';binary.write_bytes(result.stdout);binary.chmod(0o500)
    assert all(path.read_bytes()==data for path,data in inputs.items())
    assert sha(compiler.read_bytes())==COMPILER and sha(linker.read_bytes())==LINKER
    report.update(status='PASS',native_sha256=sha(result.stdout),native_bytes=len(result.stdout),module_count=len(units));save()
    print(json.dumps({k:report[k] for k in ('status','source_sha256','native_sha256','functions','blocks','instructions','module_count')}))

if __name__=='__main__':main()
