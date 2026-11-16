"""Checked HIA lifecycle overlay, native ABI3 admission, and native linking."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,re,struct,subprocess,sys

HERE=Path(__file__).resolve().parent
CORE_SHA='4e2cc7f7612c826ec35280432be527cf98075af7d0f2c35a560df67e35c3904e'
AUTHORING_SHA='4f3fa887e6372f19777ecde955d39716ae633a1bb9be7408c065d6e94e0fffbd'
COMPILER_SHA='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER_SHA='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
FUNCTION=re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)',re.M|re.S)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
functions=lambda raw:{n.decode():body for body,n in FUNCTION.findall(raw)}
py_functions=lambda source:{n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for option in ('repository','source-root','baseline','compiler','linker','output'):
        ap.add_argument('--'+option,type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    REPO=args.repository.resolve();UPSTREAM=args.source_root.resolve()
    BASE=args.baseline.resolve();PACKAGE=REPO/'tools/tensor_objects'
    COMPILER=args.compiler.resolve();LINKER=args.linker.resolve()
    inputs={p:p.read_bytes() for p in [Path(__file__),HERE/'release.py',BASE/'runtime.hepta',BASE/'overlay-authoring.py',BASE/'objects_owners.hepta',PACKAGE/'profile.json',REPO/'tools/owned_tensors/authoring.py',REPO/'docs/functions.json']}
    assert sha(inputs[BASE/'runtime.hepta'])==CORE_SHA
    assert sha(inputs[BASE/'overlay-authoring.py'])==AUTHORING_SHA
    assert sha(inputs[PACKAGE/'profile.json'])=='6bad69660dae8c63a2b4e2268d6d52dfff6f156d7b3883547eeedcf05dabe4f9'
    assert sha(inputs[REPO/'tools/owned_tensors/authoring.py'])=='2845fc2e3f03ccbadd5ea6c59734d0be6db6477bf8434d288259fab5ee570ba6'
    frozen=json.loads((HERE/'profile.json').read_text())
    inputs[HERE/'profile.json']=(HERE/'profile.json').read_bytes()
    profile=json.loads((REPO/'tools/owned_tensors/profile.json').read_text())
    inputs[REPO/'tools/owned_tensors/profile.json']=(REPO/'tools/owned_tensors/profile.json').read_bytes()
    for rel,digest in profile['upstream_authoring_sources'].items():
        path=UPSTREAM/rel;data=path.read_bytes();assert sha(data)==digest,rel;inputs[path]=data
    assert sha(COMPILER.read_bytes())==COMPILER_SHA and sha(LINKER.read_bytes())==LINKER_SHA
    sys.path.insert(0,str(UPSTREAM))
    spec=importlib.util.spec_from_file_location('release_emitter',REPO/'tools/owned_tensors/authoring.py')
    emitter=importlib.util.module_from_spec(spec);spec.loader.exec_module(emitter)
    old=functions(inputs[BASE/'runtime.hepta']);nodes=py_functions(inputs[BASE/'overlay-authoring.py'].decode())
    seeds={'backward_seed','mark_graph','object_command'}
    seeds|={node.func.id for node in ast.walk(ast.parse(inputs[HERE/'release.py']))
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in old}
    needed=set()
    def context_visit(name):
        if name in needed:return
        needed.add(name)
        for target in re.findall(rb'\{"function":"(\w+)"\}',old[name]):context_visit(target.decode())
    for name in seeds:context_visit(name)
    context_source=b'hia-asm.v1 hepta.hia.program.v8 release_authoring_context\n'+b''.join(body for name,body in old.items() if name in needed)
    context=out/'authoring-context';(context/'src').mkdir(parents=True);(context/'docs').mkdir()
    (context/'src/heptatorch.hepta').write_bytes(context_source)
    (context/'docs/functions.json').write_bytes(inputs[REPO/'docs/functions.json'])
    emitter.ROOT=context
    signatures={}
    for name,body in old.items():
        m=re.search(r'^fn \w+\((.*?)\) -> \((.*?)\)',body.decode())
        signatures[name]=([p.split(':')[1] for p in m[1].split(',') if p],m[2])
    for name in ['backward_seed','mark_graph','object_command']:
        emitted,_=emitter.emit(ast.unparse(nodes[name]),signatures)
        assert functions(emitted.encode())[name]==old[name],('baseline mismatch',name)
    backward=ast.unparse(nodes['backward_seed'])
    backward=backward.replace('def backward_seed(w: mutbuf, t: u64, seed: u64)',
                              'def backward_execute(w: mutbuf, t: u64, seed: u64, retain: u64)',1)
    assert 'def backward_execute' in backward
    needle='    epoch = getw(w, 24) + 1\n'
    assert backward.count(needle)==1
    backward=backward.replace(needle,needle+'    if epoch == 0:\n        return terror(w,8)\n',1)
    needle='    limit = t\n'
    assert backward.count(needle)==1
    backward=backward.replace(needle,needle+'    if ensure_backward_work(w,limit,epoch) == 0:\n        return 0\n',1)
    needle='            if op == 39 or op == 40:\n                back_indexed_add(w, current)\n'
    assert backward.count(needle)==1
    backward=backward.replace(needle,needle+'            if getw(w,0) != 0:\n                return 0\n            if retain == 0:\n                if release_backward_saved(w,current) == 0:\n                    return 0\n',1)
    assert backward.endswith('    return 1')
    backward=backward[:-len('    return 1')]+'    if retain == 0:\n        return release_backward_scratch(w,limit,epoch)\n    return 1\n'
    mark=ast.unparse(nodes['mark_graph'])
    needle='            if graph_saved_check(w, getw(w, p + 280)) == 0:\n'
    assert mark.count(needle)==1
    mark=mark.replace(needle,'            if getw(w,p+328) != 0:\n                return terror(w,10)\n'+needle,1)
    command=ast.unparse(nodes['object_command'])
    assert command.count('code > 56')==1
    command=command.replace('code > 56','code > 57',1)
    needle='    if code == 56:\n        result = unpack_saved(w, a)\n'
    assert command.count(needle)==1
    command=command.replace(needle,needle+'    if code == 57:\n        result = backward_command(w,a,b,c)\n',1)
    authoring=inputs[HERE/'release.py'].decode()+'\n'+backward+'\n'+mark+'\n'+command+'\n'
    (out/'overlay-authoring.py').write_text(authoring)
    fragment,mapping=emitter.emit(authoring,signatures)
    replacement=functions(fragment.encode());combined=dict(old);combined.update(replacement)
    # The released-work/Saved invariants are incompatible with the frozen
    # HTSTATE3 implementation even though both owners have HIA type buffer.
    # Only these two exact state-tag constants change; the public wire stays v3.
    old_tag=b'{"value":3694451708891714632}'
    new_tag=b'{"value":3766509302929642568}'
    for name in ('init_state','state_header'):
        assert combined[name].count(old_tag)==1,name
        combined[name]=combined[name].replace(old_tag,new_tag)
    raw=b'hia-asm.v1 hepta.hia.program.v8 tensor_backward_release\n'+b''.join(combined.values())
    assert sha(raw)==frozen['source_sha256']
    (out/'runtime.hepta').write_bytes(raw);(out/'overlay.hepta').write_text(fragment)
    (out/'functions.json').write_text(json.dumps(mapping,indent=2)+'\n')
    changed=[name for name in old if old[name]!=combined[name]]
    assert set(changed)=={'backward_seed','mark_graph','object_command','init_state','state_header'}
    complete=b'hia-asm.v1 hepta.hia.program.v8 complete_backward_release_overlay\n'+b''.join(
        body for name,body in combined.items() if name not in old or body!=old[name])
    (out/'complete-overlay.hepta').write_bytes(complete)
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
        unit=f'release_{i:02d}'
        source=f'hia-asm.v1 hepta.hia.program.v8 {unit}\nexports({",".join(names)})\n'.encode()+b''.join(declarations)+b''.join(combined[n] for n in names)
        units.append((unit,source))
    units.append(('release_owners',inputs[BASE/'objects_owners.hepta']))
    objects=[];rows=[]
    report={'schema':'heptatorch.backward-release-native-build.v1','status':'BUILDING',
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
    assert sha(result.stdout)==frozen['native_sha256']
    app=out/'tensor-release-session';app.write_bytes(result.stdout);app.chmod(0o500)
    for p,data in inputs.items():assert p.read_bytes()==data,p
    for (name,source),obj in zip(units,objects):
        assert (out/(name+'.hepta')).read_bytes()==source
        assert (out/(name+'.o')).read_bytes()==obj
    assert (out/'runtime.hepta').read_bytes()==raw and app.read_bytes()==result.stdout
    assert sha(COMPILER.read_bytes())==COMPILER_SHA and sha(LINKER.read_bytes())==LINKER_SHA
    report.update(status='PASS',native_sha256=sha(result.stdout),native_bytes=len(result.stdout));save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','units')},indent=2))

if __name__=='__main__':main()
