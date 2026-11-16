"""Native owned Adam overlay over the exact frozen Optimizer08 HIA bodies."""
from pathlib import Path
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import re
import struct
import subprocess
import sys

HERE=Path(__file__).resolve().parent
CORE_SHA='7c68a96b08aa9a21a52e2fd12be912c5f47dbde889e462f9a0dbb38cb3c09edf'
COMPILER_SHA='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER_SHA='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
FUNCTION=re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]* entry\([A-Za-z_][A-Za-z_0-9]*\)\n.*?^end\n)',re.M|re.S)
CALL=re.compile(rb'\{"function":"(\w+)"\}')
EXTERNAL_RNG={'rng_seed','rng_restore','rng_next','rng_next64','rng_unit32','rng_unit64'}
sha=lambda raw:hashlib.sha256(raw).hexdigest()
functions=lambda raw:{name.decode():body for body,name in FUNCTION.findall(raw)}


def replace_once(source,old,new):
    assert source.count(old)==1,(old,source.count(old))
    return source.replace(old,new,1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','repository','source-root','compiler','linker','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    base=a.baseline.resolve();repo=a.repository.resolve();upstream=a.source_root.resolve()
    compiler=a.compiler.resolve();linker=a.linker.resolve()
    paths=[Path(__file__),HERE/'profile.json',HERE/'adam.py',HERE/'audit.py',HERE/'mutation.py',HERE/'compact_functions.py',
           base/'runtime.hepta',base/'rng03.hepta',base/'owners.hepta',base/'overlay-authoring.py',
           repo/'tools/owned_tensors/authoring.py',repo/'tools/owned_tensors/profile.json',repo/'docs/functions.json']
    inputs={path:path.read_bytes() for path in paths}
    frozen=json.loads(inputs[HERE/'profile.json'])
    assert frozen['baseline_source_sha256']==CORE_SHA
    for name,pin in frozen['authoring_sha256'].items():assert sha(inputs[HERE/name])==pin,name
    pins={base/'runtime.hepta':CORE_SHA,
          base/'overlay-authoring.py':'687d2c4e5e76848a86f824459bcc55d7e4ed9fcdce236e9570c5e8ec0f3d66cc',
          base/'rng03.hepta':'a3fa507339b40435285dc1fa65e3cd519b6a0921a4c93bccf6ea834bb24763a3',
          base/'owners.hepta':'9c02b75c44eee8459c476049f4d9ebe10935d9d9ed8b27f5842c699f8e6a0884',
          HERE/'compact_functions.py':'36730b55b58c29b210d921a426c62eab50d9f579ad5d7a1e4199b38799218f37',
          repo/'tools/owned_tensors/authoring.py':'2845fc2e3f03ccbadd5ea6c59734d0be6db6477bf8434d288259fab5ee570ba6',
          repo/'tools/owned_tensors/profile.json':'69642a730f9da8567a01cc5afd9d15963a49ff1ab83f7761039bb301d0d2d242',
          repo/'docs/functions.json':'6ecffeb9a13a0a8fe6b78f75728fc46fef1bed06f1c49196f2d7d855ee7effca'}
    for path,pin in pins.items():assert sha(inputs[path])==pin,path
    profile=json.loads(inputs[repo/'tools/owned_tensors/profile.json'])
    for relative,pin in profile['upstream_authoring_sources'].items():
        path=upstream/relative;data=path.read_bytes();assert sha(data)==pin,path;inputs[path]=data
    assert sha(compiler.read_bytes())==COMPILER_SHA and sha(linker.read_bytes())==LINKER_SHA
    (out/'producer.py').write_bytes(inputs[Path(__file__)])
    for name in ('adam','audit','mutation'):(out/(name+'-authoring.py')).write_bytes(inputs[HERE/(name+'.py')])
    sys.path.insert(0,str(upstream))
    spec=importlib.util.spec_from_file_location('owned_adam_emitter',repo/'tools/owned_tensors/authoring.py')
    emitter=importlib.util.module_from_spec(spec);spec.loader.exec_module(emitter)
    spec=importlib.util.spec_from_file_location('owned_adam_compactor',HERE/'compact_functions.py')
    compactor=importlib.util.module_from_spec(spec);spec.loader.exec_module(compactor)
    class NativeBuilder(emitter.NativeBuilder):
        def expr(self,node):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='move_within':
                w,dest,source,size=[self.expr(value) for value in node.args]
                self.op('buffer.move_within',[w,self.cast(dest,'usize'),self.cast(source,'usize'),self.cast(size,'usize')])
                return self.c(0)
            return super().expr(node)
    old=functions(inputs[base/'runtime.hepta']);rng=functions(inputs[base/'rng03.hepta'])
    assert len(old)==315 and len(rng)==17
    aliases={}
    for original,new in (('owned_optimizer_command','owned_sgd_command'),
                         ('optimizer_audit_record','sgd_optimizer_audit_record'),
                         ('optimizer_new_momentum','owned_adam_new_moment')):
        aliases[new]=replace_once(old[original],b'fn '+original.encode()+b'(',b'fn '+new.encode()+b'(')
        assert replace_once(aliases[new],b'fn '+new.encode()+b'(',b'fn '+original.encode()+b'(')==old[original]
    kernel=replace_once(old['adam_element'],b'fn adam_element(',b'fn owned_adam_element(')
    kernel_patches=[
        (b'  v15:u64 = call(v0,v14) {"function":"get4"}',
         b'  v15:u64 = call(v0,v1,v4) {"function":"owned_optimizer_gradient_value"}'),
        (b'  v16:u64 = call(v0,v7,v15,v5) {"function":"adam_gradient"}',
         b'  v16:u64 = call(v0,v7,v15,v5) {"function":"owned_adam_gradient"}'),
        (b'  v21:u64 = int.add.wrap(v20,v13)',b'  v21:u64 = call(v0,v2,v4) {"function":"taddr"}'),
        (b'  v25:u64 = int.add.wrap(v24,v13)',b'  v25:u64 = call(v0,v3,v4) {"function":"taddr"}')]
    for before,after in kernel_patches:kernel=replace_once(kernel,before,after)
    reversed_kernel=kernel
    for before,after in reversed(kernel_patches):reversed_kernel=replace_once(reversed_kernel,after,before)
    assert replace_once(reversed_kernel,b'fn owned_adam_element(',b'fn adam_element(')==old['adam_element']
    aliases['owned_adam_element']=kernel
    context_functions={**old,**rng,**aliases}
    signatures={}
    for name,body in context_functions.items():
        match=re.search(r'^fn \w+\((.*?)\) -> \((.*?)\)',body.split(b'\n',1)[0].decode())
        signatures[name]=([p.split(':')[1] for p in match[1].split(',') if p],match[2])
    header=b'hia-asm.v1 hepta.hia.program.v8 owned_adam_authoring_context\n'
    parsed=[emitter._parse_assembly_document((header+body).decode()) for body in context_functions.values()]
    context=copy.deepcopy(parsed[0]);context['functions']=[doc['functions'][0] for doc in parsed]
    def emit(source):
        nodes=ast.parse(source).body;names={name:name for name in signatures};sigs=dict(signatures)
        for node in nodes:
            names[node.name]=node.name;sigs[node.name]=([v.annotation.id for v in node.args.args],node.returns.id)
        total=emitter.total_scalar_functions(nodes);new=[]
        for node in nodes:
            b=NativeBuilder(node,names,sigs,total);b.statements(node.body)
            new.append({'name':node.name,'params':[b.param(v) for v in b.params],
                        'returns':[b.returns],'effects':['read','write','trap'],'entry':'b0','blocks':b.blocks})
        selected={node.name for node in nodes};doc=copy.deepcopy(context)
        doc['functions']=[fn for fn in doc['functions'] if fn['name'] not in selected]+new
        doc['functions'],records=compactor.compact(doc['functions'],selected)
        text=emitter._render_assembly_document(doc).split('\n',1)[1]
        mapping={node.name:{'name':node.name,'params':[{'name':v.arg,'type':v.annotation.id} for v in node.args.args],
                           'returns':node.returns.id} for node in nodes}
        return text,mapping,records
    base_nodes={node.name:node for node in ast.parse(inputs[base/'overlay-authoring.py'].decode()).body if isinstance(node,ast.FunctionDef)}
    entry=ast.unparse(base_nodes['entry'])
    # Check the real original entry against all original signatures, rather than
    # falsely filling missing helpers with stub bodies for aggregate admission.
    verification_context=context
    context=copy.deepcopy(context)
    context['functions']=[fn for fn in context['functions'] if fn['name'] not in aliases]
    before,_,baseline_boundary=emit(entry)
    assert functions(before.encode())['entry']==old['entry']
    context=verification_context
    entry=replace_once(entry,'code >= 71 and code <= 79 or code == 90:',
                       'code >= 71 and code <= 79 or code == 90 or code == 95:')
    authoring=entry+'\n\n'+''.join(inputs[HERE/(name+'.py')].decode()+'\n' for name in ('adam','audit','mutation'))
    (out/'overlay-authoring.py').write_text(authoring)
    try:fragment,mapping,boundaries=emit(authoring)
    except Exception as error:
        (out/'refusal.json').write_text(json.dumps({'stage':'bounded authoring','error':repr(error),'native_attempted':False},indent=2)+'\n');raise
    combined=dict(old);combined.update(aliases);combined.update(functions(fragment.encode()))
    tag_pairs={int.from_bytes(b'HTOPT002','little'):int.from_bytes(b'HTADAM01','little'),
               int.from_bytes(b'HTOOPT02','little'):int.from_bytes(b'HTOADAM1','little')}
    tag_names=['init_state','state_header','so_state_init','so_state_header']
    for name in tag_names:
        found=0
        for original,replacement in tag_pairs.items():
            before=b'{"value":'+str(original).encode()+b'}';after=b'{"value":'+str(replacement).encode()+b'}'
            if before in combined[name]:combined[name]=replace_once(combined[name],before,after);found+=1
        assert found==1,name
    raw=b'hia-asm.v1 hepta.hia.program.v8 tensor_owned_adam_runtime\n'+b''.join(combined.values())
    assert sha(raw)==frozen['source_sha256']
    (out/'runtime.hepta').write_bytes(raw);(out/'overlay.hepta').write_text(fragment)
    (out/'functions.json').write_text(json.dumps(mapping,indent=2)+'\n')
    (out/'numerical-kernel-overlay.json').write_text(json.dumps({
        'original_function_sha256':sha(old['adam_element']),'new_function_sha256':sha(kernel),
        'function_rename':['adam_element','owned_adam_element'],
        'patches':[{'before':x.decode(),'after':y.decode()} for x,y in kernel_patches],
        'exact_inverse_restores_original_bytes':True,'floating_arithmetic_unchanged':True,
        'gradient_source':'tgrad via owned_optimizer_gradient_value',
        'moment_address_source':'taddr logical indexing','maximize':'owned_adam_gradient before legacy adam_gradient'},indent=2)+'\n')
    changed=[name for name in old if old[name]!=combined[name]]
    (out/'authoring-boundaries.json').write_text(json.dumps({
        'schema':'heptatorch.bounded-authoring-bodies.v1','aggregate_document_admission_claimed':False,
        'native_module_admission_required':True,'baseline_entry_bytes_identical':True,
        'baseline_boundary_records':baseline_boundary,'new_boundary_records':boundaries,
        'baseline_function_bytes_sha256':{name:sha(body) for name,body in old.items()},
        'emitted_function_bytes_sha256':{name:sha(body) for name,body in combined.items()},
        'real_body_aliases':{name:sha(body) for name,body in aliases.items()}},indent=2)+'\n')
    ordered=[];active=set();done=set()
    def visit(name):
        if name in EXTERNAL_RNG:return
        assert name in combined,name
        assert name not in active,('recursion',name)
        if name in done:return
        active.add(name)
        for target in sorted({item.decode() for item in CALL.findall(combined[name])}):visit(target)
        active.remove(name);done.add(name);ordered.append(name)
    for name in combined:visit(name)
    groups=[];group=[];instructions=0;blocks=0
    for name in ordered:
        body=combined[name];ni=body.count(b' = ');nb=len(re.findall(rb'^block ',body,re.M))
        if group and (instructions+ni>3600 or blocks+nb>1000 or len(group)>=65):
            groups.append(group);group=[];instructions=0;blocks=0
        group.append(name);instructions+=ni;blocks+=nb
    if group:groups.append(group)
    units=[('rng03',inputs[base/'rng03.hepta'])]
    for i,names in enumerate(groups):
        local=set(names);calls={target.decode() for name in names for target in CALL.findall(combined[name])}
        imports=sorted(calls-local);declarations=[]
        for name in imports:
            body=rng[name] if name in EXTERNAL_RNG else combined[name]
            declarations.append(re.sub(rb' entry\(\w+\)$',b' external()',body.split(b'\n',1)[0])+b'\n')
        assert len(names)+len(imports)<=256
        unit=f'adam_{i:02d}'
        source=f'hia-asm.v1 hepta.hia.program.v8 {unit}\nexports({",".join(names)})\n'.encode()+b''.join(declarations)+b''.join(combined[name] for name in names)
        units.append((unit,source))
    units.append(('owners',inputs[base/'owners.hepta']))
    objects=[];rows=[]
    report={'schema':'heptatorch.owned-adam-native-build.v1','status':'BUILDING',
        'baseline_source_sha256':CORE_SHA,'source_sha256':sha(raw),'functions':len(combined),
        'state_format':'HTADAM01','backing_format':'HTOADAM1','wire_request':'HTORCH03','wire_response':'HTORSP03',
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
    request=b'HIASTKS1'+struct.pack('<3Q',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(obj))+obj for obj in objects)
    (out/'link.request').write_bytes(request)
    result=subprocess.run([str(linker)],input=request,capture_output=True,env={'PATH':''},timeout=120)
    (out/'link.stderr').write_bytes(result.stderr);report['link_exit']=result.returncode;save()
    if result.returncode!=0 or result.stderr or not result.stdout.startswith(b'\x7fELF'):
        (out/'link.refusal.stdout').write_bytes(result.stdout);report['status']='NATIVE_LINKER_REFUSAL';save();raise AssertionError(report['link_exit'])
    assert sha(result.stdout)==frozen['native_sha256']
    assert [(u['name'],u['source_sha256'],u['object_sha256']) for u in rows]==[(u['name'],u['source_sha256'],u['object_sha256']) for u in frozen['units']]
    app=out/'tensor-adam-session';app.write_bytes(result.stdout);app.chmod(0o500)
    for path,data in inputs.items():assert path.read_bytes()==data,path
    for (name,source),obj in zip(units,objects):
        assert (out/(name+'.hepta')).read_bytes()==source and (out/(name+'.o')).read_bytes()==obj
    assert (out/'runtime.hepta').read_bytes()==raw and app.read_bytes()==result.stdout
    assert sha(compiler.read_bytes())==COMPILER_SHA and sha(linker.read_bytes())==LINKER_SHA
    report.update(status='PASS',native_sha256=sha(result.stdout),native_bytes=len(result.stdout));save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs_sha256','units')},indent=2))


if __name__=='__main__':main()
