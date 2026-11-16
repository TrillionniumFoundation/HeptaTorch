#!/usr/bin/env python3
"""Build a distinct seven-kind Tensor/Generator owner profile with native HIA."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SHA=lambda raw:hashlib.sha256(raw).hexdigest()
BASE_SHA='4e2cc7f7612c826ec35280432be527cf98075af7d0f2c35a560df67e35c3904e'
AUTHOR_SHA='4f3fa887e6372f19777ecde955d39716ae633a1bb9be7408c065d6e94e0fffbd'
OWNER_SHA='993651c88e69be93dea49ef473b8409b9f609e5b226c6fcec10de230ef4af99e'
RNG_SHA='c1de63f5f046bfb20aa4d4f65cbffa6217a38a36c0dc63b05f08c7ddc7612032'
COMPILER_SHA='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER_SHA='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
CONTRACT_SHA='8cbf1595eeaef6280621120dd407066ae4f992ff9f9129f3404f8c36965a304c'
FUNCTION=re.compile(rb'^(fn (\w+)\([^\n]* entry\(\w+\)\n.*?^end\n)',re.M|re.S)
def functions(raw):return {name.decode():body for body,name in FUNCTION.findall(raw)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--rng-source',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--stack-build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();base=a.baseline.resolve();up=a.upstream.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    compiler=a.stack_build.resolve()/'native/stack-module-compiler';linker=a.stack_build.resolve()/'native/stack-linker-session'
    assert SHA(compiler.read_bytes())==COMPILER_SHA and SHA(linker.read_bytes())==LINKER_SHA
    snapshots={base/'runtime.hepta':BASE_SHA,base/'overlay-authoring.py':AUTHOR_SHA,base/'objects_owners.hepta':OWNER_SHA,a.rng_source.resolve():RNG_SHA}
    local_sources={path:path.read_bytes() for path in (Path(__file__).resolve(),HERE/'rng.authoring')}
    for path,pin in snapshots.items():assert SHA(path.read_bytes())==pin,path
    raw=(base/'runtime.hepta').read_bytes();baseline=functions(raw);combined=dict(baseline)
    authoring=(base/'overlay-authoring.py').read_text();nodes={n.name:n for n in ast.parse(authoring).body}
    rng=functions(a.rng_source.read_bytes());assert len(rng)==18;rng.pop('entry')
    exported_rng=['rng_seed','rng_restore','rng_next','rng_next64','rng_unit32','rng_unit64']
    contract_raw=(up/'docs/HIA_SELFHOST_CORE_SUPPORT.json').read_bytes();assert SHA(contract_raw)==CONTRACT_SHA
    contract=json.loads(contract_raw)['source_identity']['files']
    # The verified closure is exactly the existing HIA compiler authoring closure.
    relatives=[name for name in contract if name.startswith('hepta/') or name.startswith('tools/')]
    verified={name:SHA((up/name).read_bytes()) for name in relatives if (up/name).is_file()}
    for name,pin in verified.items():assert pin==contract[name],name
    sys.path.insert(0,str(up))
    from tools.generate_hia_selfhost import Builder,compact_authoring_cfg,total_scalar_functions
    from hepta.hia_agent_assembly import assemble_program,_render_assembly_document,_parse_assembly_document

    class NativeBuilder(Builder):
        def expr(self,node):
            if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.FloorDiv,ast.Mod)):
                x,y=self.expr(node.left),self.expr(node.right)
                return self.op('int.div.checked' if isinstance(node.op,ast.FloorDiv) else 'int.rem.checked',[x,y],x[1])
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                name=node.func.id
                if name in ('idiv','irem'):
                    x,y=[self.expr(x) for x in node.args]
                    return self.op('int.div.checked' if name=='idiv' else 'int.rem.checked',[x,y],x[1])
                if name in ('copy_range','copy_from_mut'):
                    target,source,at,old,size=[self.expr(x) for x in node.args];loan=None
                    if name=='copy_from_mut':loan=self.op('borrow.read',[source],'readbuf');source=loan
                    self.op('buffer.copy_range',[target,source,self.cast(at,'usize'),self.cast(old,'usize'),self.cast(size,'usize')])
                    if loan is not None:self.op('loan.end',[loan])
                    return self.c(0)
                if name=='move_within':
                    w,dest,src,size=[self.expr(x) for x in node.args]
                    self.op('buffer.move_within',[w,self.cast(dest,'usize'),self.cast(src,'usize'),self.cast(size,'usize')])
                    return self.c(0)
            return super().expr(node)

    replacements={}
    changes=[]
    new_backing=int.from_bytes(b'HTORNGS1','little');new_main=int.from_bytes(b'HTSTRNG1','little')
    for name,node in nodes.items():
        text=ast.unparse(node)
        if name.startswith('so_') or name=='object_stats':
            revised=text.replace(' * 384',' * 448').replace(' * 424',' * 488').replace(' * 400',' * 464')
            if name=='so_slot_lookup':revised=revised.replace('kind > 6','kind > 7')
            if name=='so_state_migrate':revised=revised.replace('kind < 6','kind < 7')
            if name=='so_audit_records':
                revised=revised.replace('kind <= 6','kind <= 7')
                needle='                if kind == 3:\n'
                assert revised.count(needle)==1
                revised=revised.replace(needle,'''                if kind == 7:
                    hid = so_uget(w,p+16)
                    address = private_hold(w,hid,5056)
                    if address == 0 or rng_state_canonical(w,address) == 0:
                        return 9
                    j = 24
                    while j < 64:
                        if so_uget(w,p+j) != 0:
                            return 9
                        j = j+8
                    j = 1
                    while j < i:
                        other = so_slot_at(w,7,j)
                        if so_uget(w,other+8) == 1 and so_uget(w,other+16) == hid:
                            return 9
                        j = j+1
'''+needle)
            if name=='so_state_init':
                revised=revised.replace('3544425623597306952',str(new_backing))
                revised=revised.replace('so_uput(w, 8, 1)','so_uput(w, 8, 2)')
            if name=='so_state_header':
                revised=revised.replace('3544425623597306952',str(new_backing))
                revised=revised.replace('so_uget(w, 8) != 1','so_uget(w, 8) != 2')
            if revised!=text:replacements[name]=revised;changes.append(name)
        if name in ('init_state','state_header'):
            revised=text.replace('3694451708891714632',str(new_main)).replace('139264 + 1840 + 131072','139264 + 2096 + 131072')
            assert revised!=text;replacements[name]=revised;changes.append(name)
        if name in ('so_tensor_live','so_saved_live'):
            needle='    if offset > so_uget(w, storage + 24) or size > so_uget(w, storage + 24) - offset:\n        return 0'
            assert text.count(needle)==1
            revised=text.replace(needle,'''    if offset > 1073741824:
        return 0
    if offset > so_uget(w,storage+24):
        if size != 0:
            return 0
    else:
        if size > so_uget(w,storage+24)-offset:
            return 0''')
            if name=='so_tensor_live':
                revised=revised.replace('        if size != 0:\n            return 0','''        node = so_uget(w,p+48)
        if size != 0 or node == 0 or node >= getw(w,8):
            return 0
        q = dptr(w,node)
        if getw(w,q+8) != 0 or getw(w,q+232) != offset:
            return 0''')
            replacements[name]=revised
            if name not in changes:changes.append(name)
        if name in ('bind_alias','saved_create'):
            if name=='bind_alias':
                needle='    so_uput(w, tp + 40, so_uget(w, sp + 24) - getw(w, q + 232))'
                revised=text.replace(needle,'''    remaining = 0
    if getw(w,q+232) <= so_uget(w,sp+24):
        remaining = so_uget(w,sp+24)-getw(w,q+232)
    so_uput(w,tp+40,remaining)''')
            else:
                needle='    so_uput(w, sp + 48, so_uget(w, storage + 24) - getw(w, p + 232))'
                revised=text.replace(needle,'''    remaining = 0
    if getw(w,p+232) <= so_uget(w,storage+24):
        remaining = so_uget(w,storage+24)-getw(w,p+232)
    so_uput(w,sp+48,remaining)''')
            assert revised!=text;replacements[name]=revised;changes.append(name)
    entry=ast.unparse(nodes['entry'])
    entry=entry.replace('w: mutbuf) -> u32:','w: mutbuf, scratch: mutbuf) -> u32:',1)
    needle='        next = command(s, o, w, 32 + i * 64, at, end)'
    assert entry.count(needle)==1
    entry=entry.replace(needle,'''        code = getr(s,32+i*64)
        if code >= 60 and code <= 69:
            next = rng_command(s,o,w,scratch,32+i*64,at,end)
        else:
            next = command(s,o,w,32+i*64,at,end)''')
    replacements['entry']=entry;changes.append('entry')
    source='\n\n'.join(replacements.values())+'\n\n'+(HERE/'rng.authoring').read_text()
    source+='''
def rng_state_canonical(w:mutbuf,address:u64) -> u64:
    left = get4(w,address+8)
    next = getw(w,address+16)
    if get4(w,address+12) != 1 or left == 0 or left > 624 or next > 624:
        return 0
    if (left != 1 or next != 0) and (next == 0 or left+next != 625):
        return 0
    if getw(w,address+5016) != 0 or getw(w,address+5032) != 0 or get4(w,address+5044) != 0:
        return 0
    double_flag = get4(w,address+5040)
    float_flag = get4(w,address+5052)
    if double_flag > 1 or float_flag > 1:
        return 0
    if double_flag == 0 and getw(w,address+5024) != 0:
        return 0
    if float_flag == 0 and get4(w,address+5048) != 0:
        return 0
    i = 0
    while i < 624:
        if getw(w,address+24+i*8) > 4294967295:
            return 0
        i = i+1
    return 1
'''
    (out/'overlay-authoring.py').write_text(source)
    parsed=ast.parse(source).body
    signature={}
    for name,body in {**baseline,**rng}.items():
        h=re.match(rb'fn \w+\(([^)]*)\) -> \(([^)]*)\)',body)
        signature[name]=([x.split(b':')[1].decode() for x in h[1].split(b',') if x],h[2].decode())
    signature.update({n.name:([x.annotation.id for x in n.args.args],n.returns.id) for n in parsed})
    names={name:name for name in signature};total=total_scalar_functions(parsed);fns=[]
    for n in parsed:
        b=NativeBuilder(n,names,signature,total);b.statements(n.body)
        fns.append({'name':n.name,'params':[b.param(v) for v in b.params],'returns':[b.returns],
                    'effects':['read','write','trap'],'entry':'b0','blocks':b.blocks})
    # Host CFG compaction verifies call interfaces. Native per-unit admission
    # resolves private RNG helpers in their actual separate module namespace.
    context={'schema':'hepta.hia.program.v8','module':'tensor_rng_authoring_context','functions':[]}
    for body in baseline.values():
        parsed_body=_parse_assembly_document(b'hia-asm.v1 hepta.hia.program.v8 tensor_rng_authoring_context\n'+body)
        context['functions'].extend(parsed_body['functions'])
    replace_names={n.name for n in parsed}
    context['functions']=[f for f in context['functions'] if f['name'] not in replace_names]
    rng_doc=assemble_program(a.rng_source.read_bytes()).as_dict()
    context['functions'] += [f for f in rng_doc['functions'] if f['name'] not in baseline and f['name']!='entry']
    context['functions']+=fns
    doc=compact_authoring_cfg(context)
    doc['functions']=[f for f in doc['functions'] if f['name'] in replace_names]
    fragment=_render_assembly_document(doc).encode();combined.update(functions(fragment))
    (out/'overlay.hepta').write_bytes(fragment)
    runtime=b'hia-asm.v1 hepta.hia.program.v8 tensor_rng_runtime\n'+b''.join(combined.values())
    (out/'runtime.hepta').write_bytes(runtime)
    # Add a genuine temporary scratch owner without copying any owner descriptor.
    original_owners=(base/'objects_owners.hepta').read_text()
    original_step=functions(original_owners.encode())['step'].decode()
    active=None;lines=[]
    for line in original_step.splitlines():
        if line.startswith('fn step('):line=line.replace('fn step(','fn step_impl(',1).replace(') -> (buffer,u32)',',rw_b0:mutbuf) -> (buffer,u32)',1)
        elif line.startswith('block '):
            active=line.split(' ',1)[1].split('(',1)[0];line=line[:-1]+',rw_'+active+':mutbuf)'
        elif line.strip().startswith(('jump ','branch ')):
            line=re.sub(r'(\w+)\(([^()]*)\)',lambda m:m[1]+'('+m[2]+',rw_'+active+')',line)
        elif '"function":"entry"' in line:
            line=line.replace('call(s8,o8,m8)','call(s8,o8,m8,rw_'+active+')')
        lines.append(line)
    adapted='\n'.join(lines)+'\n'
    owner_source=original_owners.replace(original_step,adapted)
    owner_source=owner_source.replace('fn entry(s:readbuf,o:mutbuf,w:mutbuf)','fn entry(s:readbuf,o:mutbuf,w:mutbuf,r:mutbuf)')
    owner_source+='''fn step(s:readbuf,o:mutbuf,state:buffer) -> (buffer,u32) effects(alloc,read,write,trap) entry(b0)
block b0(s:readbuf,o:mutbuf,state:buffer)
  cap:usize = buffer.len(o)
  minimum:usize = const() {"value":32}
  short:bool = compare.lt(cap,minimum)
  branch short no_output(state) allocate(s,o,state)
block no_output(st0:buffer)
  zero0:u32 = const() {"value":0}
  ret(st0,zero0)
block allocate(s1:readbuf,o1:mutbuf,st1:buffer)
  amount1:usize = const() {"value":5056}
  scratch1:buffer,ok1:bool = buffer.try_new(amount1)
  branch ok1 execute(s1,o1,st1,scratch1) failure(o1,st1,scratch1)
block failure(o2:mutbuf,st2:buffer,scratch2:buffer)
  _ = owner.drop(scratch2)
  error2:u64 = const() {"value":3}
  n2:u64 = call(o2,error2) {"function":"reply_error"}
  size2:u32 = int.cast(n2) {"mode":"wrap"}
  ret(st2,size2)
block execute(s3:readbuf,o3:mutbuf,st3:buffer,scratch3:buffer)
  work3:mutbuf = borrow.mut(scratch3)
  returned3:buffer,n3:u32 = call(s3,o3,st3,work3) {"function":"step_impl"}
  _ = loan.end(work3)
  _ = owner.drop(scratch3)
  ret(returned3,n3)
end
'''
    ordered=[];active=set();done=set()
    def visit(name):
        if name in exported_rng:return
        assert name in combined,name
        if name in done:return
        assert name not in active,name;active.add(name)
        for child in sorted({x.decode() for x in re.findall(rb'\{"function":"(\w+)"\}',combined[name])}):visit(child)
        active.remove(name);done.add(name);ordered.append(name)
    for name in combined:visit(name)
    groups=[];group=[];ins=blocks=0
    for name in ordered:
        body=combined[name];ni=body.count(b' = ');nb=len(re.findall(rb'^block ',body,re.M))
        if group and (ins+ni>3600 or blocks+nb>1000 or len(group)>=65):groups.append(group);group=[];ins=blocks=0
        group.append(name);ins+=ni;blocks+=nb
    if group:groups.append(group)
    rng_unit=b'hia-asm.v1 hepta.hia.program.v8 tensor_rng_frozen03\nexports('+','.join(exported_rng).encode()+b')\n'+b''.join(rng.values())
    units=[('rng03',rng_unit)]
    for i,group in enumerate(groups):
        imported=sorted({x.decode() for n in group for x in re.findall(rb'\{"function":"(\w+)"\}',combined[n])}-set(group))
        source=('hia-asm.v1 hepta.hia.program.v8 tensor_rng_'+str(i)+'\nexports('+','.join(group)+')\n').encode()
        for n in imported:
            body=rng[n] if n in exported_rng else combined[n]
            source+=re.sub(rb' entry\(\w+\)$',b' external()',body.split(b'\n',1)[0])+b'\n'
        source+=b''.join(combined[n] for n in group);units.append(('tensor_rng_'+str(i),source))
    units.append(('owners',owner_source.encode()))
    rows=[];objects=[]
    report={'schema':'heptatorch.tensor-rng-native-build.v1','status':'BUILDING','baseline_source_sha256':BASE_SHA,
        'build_script_sha256':SHA(local_sources[Path(__file__).resolve()]),'rng_authoring_sha256':SHA(local_sources[HERE/'rng.authoring']),
        'rng_source_sha256':RNG_SHA,'compiler_sha256':COMPILER_SHA,'linker_sha256':LINKER_SHA,
        'runtime_source_sha256':SHA(runtime),'changed_baseline_functions':changes,
        'baseline_function_body_map':{n:{'before':SHA(b),'after':SHA(combined[n]),'unchanged':b==combined[n]} for n,b in baseline.items()},
        'frozen_rng_function_bodies_unchanged':{n:SHA(b) for n,b in rng.items()},'units':rows,
        'layout':{'kinds':7,'table_stride':64,'data_base':'144+488*slots','free_base':'128+448*slots',
                  'audit_scratch_base':'144+464*slots','state_prefix_bytes':139264,'main_magic':'HTSTRNG1','backing_magic':'HTORNGS1'},
        'fma_pipeline_is_pending_fixed_torch_qualification':False,
        'fixed_torch_arithmetic_reference':{'probe_sha256':'a27756cc3be9603dbaa6b21465a3258d148f6ff3fde1f5df9ed7c8b0d93f9876',
            'report_sha256':'098899a6085d4b8645aec515bb94dfe7c2a1b1fc79c263ab5aa7842332684dd7',
            'fused_mismatches':0,'separate_mismatches':4914,'values':30720,
            'report_is_native_execution':False},'full_pytorch_rng_complete':False}
    def save():(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    for name,source in units:
        (out/(name+'.hepta')).write_bytes(source)
        result=subprocess.run([str(compiler)],input=source,capture_output=True,env={'PATH':''},timeout=60)
        (out/(name+'.compile.stderr')).write_bytes(result.stderr);(out/(name+'.compile.stdout')).write_bytes(result.stdout)
        row={'name':name,'source_sha256':SHA(source),'source_bytes':len(source),'functions':len(re.findall(rb'^fn ',source,re.M)),
            'blocks':len(re.findall(rb'^block ',source,re.M)),'instructions':source.count(b' = '),'compile_exit':result.returncode}
        rows.append(row);save()
        assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF'),(name,result.returncode,result.stderr)
        row['object_sha256']=SHA(result.stdout);objects.append(result.stdout);(out/(name+'.o')).write_bytes(result.stdout)
    request=b'HIASTKS1'+struct.pack('<3Q',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(x))+x for x in objects)
    (out/'link.request').write_bytes(request);result=subprocess.run([str(linker)],input=request,capture_output=True,env={'PATH':''},timeout=90)
    (out/'link.stdout').write_bytes(result.stdout);(out/'link.stderr').write_bytes(result.stderr);report['link_exit']=result.returncode;save()
    assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF'),(result.returncode,result.stderr)
    image=out/'tensor-rng-session';image.write_bytes(result.stdout);image.chmod(0o700)
    for path,pin in snapshots.items():assert SHA(path.read_bytes())==pin,path
    for path,original in local_sources.items():assert path.read_bytes()==original,path
    loaded={}
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).resolve().is_relative_to(up):
            relative=str(Path(path).resolve().relative_to(up));assert relative in verified
            loaded[relative]=SHA(Path(path).read_bytes());assert loaded[relative]==verified[relative]
    report.update(status='NATIVE_BUILD_PASS',executable_sha256=SHA(result.stdout),executable_bytes=len(result.stdout),
                  upstream_loaded_sources=loaded,host_production_rng_or_tensor_runtime=False)
    save();print(json.dumps({k:v for k,v in report.items() if k not in ('baseline_function_body_map','upstream_loaded_sources')},indent=2))

if __name__=='__main__':main()
