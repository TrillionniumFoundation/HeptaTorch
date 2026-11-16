#!/usr/bin/env python3
"""Author sources and invoke only native HIA compilation and native linking."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
COMPILER_SHA = 'b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
TOOL_LINKER_SHA = 'fa3065b5dc42bfe09fd29cf773c5adc5a9296c125a6a161daecf2fa885ea160b'
RNG_SHA = 'c1de63f5f046bfb20aa4d4f65cbffa6217a38a36c0dc63b05f08c7ddc7612032'
BASE_LINKER_SHA = 'b0adabf3db5bdda3ee46c0c74630938c0f1532e97bc532d65529c3ea0af83ac7'
BASE_STARTUP_SHA = '1271a7805412930cf682d1f08f5aeed2955862f40747d2a4253784f4055fe8b0'
CONTRACT_SHA = '8cbf1595eeaef6280621120dd407066ae4f992ff9f9129f3404f8c36965a304c'
AUTHORING_FILES = '''hepta/__init__.py
hepta/artifact_publication.py
hepta/hia_agent_assembly.py
hepta/hia_capability_flow.py
hepta/hia_core_admission_native.py
hepta/hia_function_shape_native.py
hepta/hia_instruction_policy_native.py
hepta/hia_json_canonical_native.py
hepta/hia_json_decode.py
hepta/hia_json_decode_native.py
hepta/hia_json_preflight.py
hepta/hia_json_preflight_native.py
hepta/hia_machine_x64.py
hepta/hia_opcodes.py
hepta/hia_program.py
hepta/hia_program_shape_native.py
hepta/hia_program_tools.py
hepta/hia_schema.py
hepta/hia_sha256_native.py
tools/generate_hia_selfhost.py
tools/generate_hia_elf_linker.py
tools/hia_selfhost_chain_elf.py
tools/hia_selfhost_cli.py
tools/hia_selfhost_effects.py
tools/hia_selfhost_exports.py
tools/hia_selfhost_keywords.py
tools/hia_selfhost_limits.py
tools/hia_selfhost_multi_results.py
tools/hia_selfhost_names.py
tools/hia_selfhost_owned_elf.py
tools/hia_selfhost_returned_loans.py
tools/hia_selfhost_signed.py
tools/hia_selfhost_traps.py'''.splitlines()

def sha(raw): return hashlib.sha256(raw).hexdigest()

def replace_once(source, old, new):
    assert source.count(old) == 1, (old,source.count(old))
    return source.replace(old,new)

def replace_function(source,name,body):
    node = next(n for n in ast.parse(source).body if n.name == name)
    return replace_once(source,ast.get_source_segment(source,node),body.rstrip())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--stack-build',type=Path,required=True)
    p.add_argument('--rng-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',type=Path,default=ROOT/'profile.json')
    a=p.parse_args();up=a.upstream.resolve();base=a.stack_build.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    profile_raw=a.profile.read_bytes();profile=json.loads(profile_raw)
    own_inputs={name:(ROOT/name).read_bytes() for name in ('build.py','model.authoring','owners.hia.in',
                'sources/rng.hepta','sources/model.hepta','sources/owners.hepta')}
    compiler=base/'native/stack-module-compiler';tool_linker=base/'native/stack-linker-linker'
    assert sha(compiler.read_bytes())==COMPILER_SHA
    assert sha(tool_linker.read_bytes())==TOOL_LINKER_SHA
    inputs={a.rng_source.resolve():RNG_SHA,base/'sources/linker-session/authoring.txt':BASE_LINKER_SHA,
            base/'sources/linker-session/startup-authoring.txt':BASE_STARTUP_SHA}
    for path,pin in inputs.items():assert sha(path.read_bytes())==pin,path
    contract_raw=(up/'docs/HIA_SELFHOST_CORE_SUPPORT.json').read_bytes();assert sha(contract_raw)==CONTRACT_SHA
    contract=json.loads(contract_raw)['source_identity']['files']
    custody={rel:sha((up/rel).read_bytes()) for rel in AUTHORING_FILES}
    for rel,pin in custody.items():assert pin==contract[rel],rel
    sys.path.insert(0,str(up))
    from tools.generate_hia_selfhost import Builder,compact_authoring_cfg,total_scalar_functions
    from tools.generate_hia_elf_linker import LinkerBuilder
    from hepta.hia_agent_assembly import _render_assembly_document,render_assembly,assemble_program

    class NativeBuilder(Builder):
        def expr(self,value):
            if isinstance(value,ast.BinOp) and isinstance(value.op,(ast.FloorDiv,ast.Mod)):
                left,right=self.expr(value.left),self.expr(value.right)
                return self.op('int.div.checked' if isinstance(value.op,ast.FloorDiv) else 'int.rem.checked',[left,right],left[1])
            if isinstance(value,ast.Call) and value.func.id in ('copy_range','copy_from_mut'):
                dest,source,dat,sat,size=[self.expr(x) for x in value.args]
                loan=None
                if value.func.id=='copy_from_mut':
                    loan=self.op('borrow.read',[source],'readbuf');source=loan
                self.op('buffer.copy_range',[dest,source,self.cast(dat,'usize'),self.cast(sat,'usize'),self.cast(size,'usize')])
                if loan is not None:self.op('loan.end',[loan])
                return self.c(0)
            return super().expr(value)

    def native(tool,raw,path):
        result=subprocess.run([str(tool)],input=raw,capture_output=True,env={'PATH':''},timeout=90)
        path.with_suffix(path.suffix+'.stderr').write_bytes(result.stderr)
        path.with_suffix(path.suffix+'.exit').write_text(str(result.returncode)+'\n')
        path.write_bytes(result.stdout)
        assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF'),(path,result.returncode,result.stderr)
        return result.stdout

    def link(tool,magic,objects,path,entry=b'f0'):
        request=magic+struct.pack('<3Q',len(objects),len(entry),0)+entry+b''.join(struct.pack('<Q',len(x))+x for x in objects)
        path.with_suffix('.request').write_bytes(request)
        raw=native(tool,request,path);path.chmod(0o700);return raw

    # A distinct entropy-bearing startup, based on the frozen guarded owned ABI.
    startup=(base/'sources/linker-session/startup-authoring.txt').read_text()
    entropy='''    # Entropy slot3 is trusted, readonly, and is never supplied by frame bytes.
    emit('49 8d 84 24');word(256)
    emit('49 89 84 24');word(144)
    emit('31 c0');zero_slot(152);zero_slot(256)
    emit('49 83 7c 24 08 40');branch('0f 82','entropy_done')
    emit('48 b8');quad(int.from_bytes(b'HTCGEN01','little'))
    emit('49 39 84 24');word(INPUT);branch('0f 85','entropy_done')
    emit('49 83 bc 24');word(INPUT+8);emit('03');branch('0f 85','entropy_done')
    emit('4d 8d ac 24');word(256)
    emit('41 be 08 00 00 00')
    label('entropy_read')
    emit('b8 3e 01 00 00 4c 89 ef 4c 89 f6 31 d2 0f 05')
    emit('48 83 f8 fc');branch('0f 84','entropy_read')
    emit('48 85 c0');branch('0f 8e','entropy_failed')
    emit('4c 39 f0');branch('0f 87','entropy_failed')
    emit('49 01 c5 49 29 c6');branch('0f 85','entropy_read')
    emit('49 c7 84 24');word(152);word(8)
    branch('e9','entropy_done')
    label('entropy_failed')
    emit('31 c0');zero_slot(152);zero_slot(256)
    label('entropy_done')
'''
    startup=replace_once(startup,'    # Private ABI arguments occupy [96,144): readbuf, mutbuf, owned buffer.\n',
                         entropy+'    # Four private ABI arguments occupy [96,160).\n')
    startup=replace_once(startup,'for offset in (128,136,160,168,176,184):zero_slot(offset)',
                         'for offset in (128,136,144,152,160,168,176,184,256):zero_slot(offset)')
    startup=replace_once(startup,"'argument_area':[96,144]","'argument_area':[96,160]")
    startup=replace_once(startup,"'state':[128,144]},","'state':[128,144],'entropy':[144,160]},")
    startup=replace_once(startup,"'parameters':['readbuf','mutbuf','buffer']","'parameters':['readbuf','mutbuf','buffer','readbuf']")
    namespace=dict(hashlib=hashlib,struct=struct,MIB=1<<20,CONTROL=4096,INPUT=4096,OUTPUT=4096+(1<<20),
                   MAPPING=4096+(2<<20),REQUEST_MAGIC=b'HTOWNR01',RESPONSE_MAGIC=b'HTOWNS01',STACK_BYTES=8388608)
    exec(compile(startup,'<cpu-generator-entropy-startup-authoring>','exec'),namespace)
    prefix,step,close,metadata=namespace['owned_session_template']()
    metadata.update(profile='heptatorch.cpu-generator.entropy-session.v1',entropy={'syscall':318,'bytes':8,'private_offset':256,
        'request_operation':3,'failure_length':0,'retry_eintr':True,'partial_read_loop':True,'frame_can_supply_entropy':False})
    (out/'startup-authoring.txt').write_text(startup);(out/'startup-prefix.bin').write_bytes(prefix)
    (out/'startup.json').write_text(json.dumps(metadata,indent=2)+'\n')
    source=(base/'sources/linker-session/authoring.txt').read_text()
    source=replace_once(source,str(int.from_bytes(b'HIASTKS1','little')),str(int.from_bytes(b'HIACGS01','little')))
    entry=ast.get_source_segment(source,next(n for n in ast.parse(source).body if n.name=='module_entry'))
    entry=replace_once(entry,'uint(s, at + 16, 8) != 3','uint(s, at + 16, 8) != 4')
    entry=replace_once(entry,'uint(s, at + 64, 8) != 8:','uint(s, at + 64, 8) != 8 or uint(s, at + 72, 8) != 6:')
    source=replace_function(source,'module_entry',entry)
    header=['def headers(o:mutbuf,w:mutbuf) -> u64:']
    for at in range(0,len(prefix),8):
        part=prefix[at:at+8];header.append(f'    write(o,{at},{int.from_bytes(part,"little")},{len(part)})')
    header+=['    put(o,96,get(w,64))','    put(o,104,get(w,64))',f'    write(o,{step},get(w,88)-{step+4},4)',
             f'    write(o,{close},get(w,96)-{close+4},4)',f'    put(o,{metadata["stack"]["budget_hole"]},get(w,104)+4096)','    return 0']
    source=replace_function(source,'headers','\n'.join(header))
    (out/'linker-authoring.txt').write_text(source)
    nodes=ast.parse(source).body
    names={n.name:'f'+str(i) for i,n in enumerate(nodes)}
    sigs={n.name:([a.annotation.id for a in n.args.args],n.returns.id) for n in nodes}
    total=total_scalar_functions(nodes);fns=[]
    for n in nodes:
        b=LinkerBuilder(n,names,sigs,total);b.statements(n.body)
        fns.append({'name':names[n.name],'params':[b.param(v) for v in b.params],'returns':[b.returns],
                    'effects':['read','write','trap'],'entry':'b0','blocks':b.blocks})
    doc=compact_authoring_cfg({'schema':'hepta.hia.program.v8','module':'cpu_generator_entropy_linker','functions':fns})
    raw=render_assembly(doc).encode();first,rest=raw.split(b'\n',1);raw=first+b'\nexports(f0)\n'+rest
    (out/'linker.hepta').write_bytes(raw)
    linker_obj=native(compiler,raw,out/'linker.o')
    linker=out/'generator-session-linker'
    link(tool_linker,b'HIASTKL1',[linker_obj],linker)

    # Reuse every non-entry frozen RNG function byte for byte.
    frozen=a.rng_source.read_bytes()
    bodies={m[2].decode():m[1] for m in re.finditer(rb'^(fn (\w+)\([^\n]*\n.*?^end\n)',frozen,re.M|re.S)}
    assert len(bodies)==18 and 'entry' in bodies
    bodies.pop('entry')
    signatures={}
    for name,body in bodies.items():
        match=re.match(rb'fn \w+\(([^)]*)\) -> \(([^)]*)\)',body)
        signatures[name]=([p.split(b':')[1].decode() for p in match[1].split(b',') if p],match[2].decode())
    rng=b'hia-asm.v1 hepta.hia.program.v8 cpu_generator_rng\nexports('+','.join(bodies).encode()+b')\n'+b''.join(bodies.values())
    sources={'rng':rng}
    constants={name:int.from_bytes(value,'little') for name,value in {'STATE_MAGIC':b'HTCGST01','REQUEST_MAGIC':b'HTCGEN01','RESPONSE_MAGIC':b'HTCGRS01'}.items()}
    source=(ROOT/'model.authoring').read_text()
    for key,value in constants.items():source=source.replace(key,str(value))
    nodes=ast.parse(source).body
    signatures.update({n.name:([a.annotation.id for a in n.args.args],n.returns.id) for n in nodes})
    names={n:n for n in signatures};total=total_scalar_functions(nodes);fns=[]
    for n in nodes:
        b=NativeBuilder(n,names,signatures,total);b.statements(n.body)
        fns.append({'name':n.name,'params':[b.param(v) for v in b.params],'returns':[b.returns],
                    'effects':['read','write','trap'],'entry':'b0','blocks':b.blocks})
    frozen_doc=assemble_program(frozen).as_dict()
    closure_functions=[f for f in frozen_doc['functions'] if f['name']!='entry']
    doc=compact_authoring_cfg({'schema':'hepta.hia.program.v8','module':'cpu_generator_model','functions':fns+closure_functions})
    doc['functions']=[f for f in doc['functions'] if f['name'] not in bodies]
    fns=doc['functions']
    def external(name):
        params,result=signatures[name]
        return 'fn '+name+'('+','.join('p'+str(i)+':'+t for i,t in enumerate(params))+') -> ('+result+') effects(read,write,trap) external()\n'
    local={f['name'] for f in fns};calls={op['attrs']['function'] for f in fns for b in f['blocks'] for op in b['ops'] if op['op']=='call'}
    model=_render_assembly_document(dict(doc,exports=[n.name for n in nodes]));head,exports,tail=model.split('\n',2)
    sources['model']=(head+'\n'+exports+'\n'+''.join(external(n) for n in sorted(calls-local))+tail).encode()
    owners=(ROOT/'owners.hia.in').read_text()
    calls=set(re.findall(r'\{"function":"(\w+)"\}',owners))
    sources['owners']=owners.replace('EXTERNALS\n',''.join(external(n) for n in sorted(calls-{'cg_error'}))).encode()
    objects=[];module_rows={}
    for name,raw in sources.items():
        raw=b'\n'.join(x for x in raw.splitlines() if x.strip())+b'\n'
        if name=='rng':
            emitted={m[2].decode():m[1] for m in re.finditer(rb'^(fn (\w+)\([^\n]*\n.*?^end\n)',raw,re.M|re.S)}
            assert emitted==bodies
        assert raw==own_inputs['sources/'+name+'.hepta'],name
        (out/(name+'.hepta')).write_bytes(raw)
        obj=native(compiler,raw,out/(name+'.o'));objects.append(obj)
        module_rows[name]={'source_sha256':sha(raw),'source_bytes':len(raw),'object_sha256':sha(obj),
            'definitions':len(re.findall(rb'^fn .* entry\(',raw,re.M)),'external_declarations':len(re.findall(rb'^fn .* external\(',raw,re.M)),
            'blocks':len(re.findall(rb'^block ',raw,re.M)),'instructions':raw.count(b' = ')}
    image=link(linker,b'HIACGS01',objects,out/'generator-session',entry=b'step')
    loaded={}
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path:
            path=Path(path).resolve()
            if path.is_relative_to(up):loaded[str(path.relative_to(up))]=sha(path.read_bytes())
    assert loaded==custody,(set(loaded)-set(custody),set(custody)-set(loaded))
    for path,pin in inputs.items():assert sha(path.read_bytes())==pin,path
    for name,snapshot in own_inputs.items():assert (ROOT/name).read_bytes()==snapshot,name
    assert a.profile.read_bytes()==profile_raw
    assert sha(compiler.read_bytes())==COMPILER_SHA and sha(tool_linker.read_bytes())==TOOL_LINKER_SHA
    report={'schema':'heptatorch.cpu-generator-native-build.v1','status':'PASS',
        'compiler_sha256':COMPILER_SHA,'tool_linker_sha256':TOOL_LINKER_SHA,
        'rng_source_sha256':RNG_SHA,'rng_function_bodies_sha256':{name:sha(body) for name,body in bodies.items()},
        'rng_all_17_nonentry_bodies_unchanged':True,'upstream_authoring_closure':custody,
        'own_authoring_sources_sha256':{name:sha(raw) for name,raw in own_inputs.items()},
        'base_linker_authoring_sha256':BASE_LINKER_SHA,'base_startup_authoring_sha256':BASE_STARTUP_SHA,
        'generator_linker_source_sha256':sha((out/'linker.hepta').read_bytes()),
        'generator_linker_sha256':sha(linker.read_bytes()),'startup_sha256':sha(prefix),
        'executable_sha256':sha(image),'modules':module_rows,'host_production_rng_or_owner_runtime':False,
        'frozen_profiles_modified':False,'request_magic':'HIACGS01','full_pytorch_generator_compatibility_claimed':False}
    for key in ('compiler_sha256','tool_linker_sha256','rng_source_sha256','generator_linker_source_sha256',
                'generator_linker_sha256','startup_sha256','executable_sha256','modules'):
        assert report[key]==profile[key],key
    report['profile_sha256']=sha(profile_raw)
    (out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
