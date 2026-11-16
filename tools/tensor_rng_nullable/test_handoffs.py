#!/usr/bin/env python3
"""Native owner moves between the common runtime and four frozen predecessors."""
import argparse,hashlib,importlib.util,json,re,struct,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True

SHA=lambda b:hashlib.sha256(b).hexdigest()
COMPILER='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'

def rename(raw,prefix,reverse=False):
    def symbol(n):
        if reverse:assert n.startswith(prefix);return n[len(prefix):]
        return prefix+n
    raw=re.sub(rb'^(hia-asm\.v1 hepta\.hia\.program\.v8 )(\w+)$',lambda m:m[1]+symbol(m[2]),raw,flags=re.M)
    raw=re.sub(rb'^(fn )(\w+)(\()',lambda m:m[1]+symbol(m[2])+m[3],raw,flags=re.M)
    raw=re.sub(rb'^exports\(([^)]*)\)$',lambda m:b'exports('+b','.join(symbol(n) for n in m[1].split(b','))+b')',raw,flags=re.M)
    return re.sub(rb'(\{"function":")(\w+)("\})',lambda m:m[1]+symbol(m[2])+m[3],raw)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build','rng-package','stack-build','peer-pins','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--peer',action='append',required=True,help='label=actual frozen build directory')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    roots=dict(x.split('=',1) for x in a.peer);pins=json.loads(a.peer_pins.read_text())
    assert set(roots)==set(pins['peers'])
    compiler=a.stack_build.resolve()/'native/stack-module-compiler';linker=a.stack_build.resolve()/'native/stack-linker-session'
    assert SHA(compiler.read_bytes())==COMPILER and SHA(linker.read_bytes())==LINKER
    package=a.rng_package.resolve();spec=importlib.util.spec_from_file_location('handoff_wire',package/'protocol.py')
    wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
    current=a.build.resolve();receipt=json.loads((current/'build.json').read_text())
    current_pin=SHA((current/'runtime.hepta').read_bytes());assert current_pin==pins['current_source_sha256']
    assert SHA((current/'tensor-rng-session').read_bytes())==pins['current_executable_sha256']
    sources={a.peer_pins.resolve():a.peer_pins.read_bytes()};common=out/'common';common.mkdir();bindings=[]
    def compile_unit(name,source,folder):
        (folder/(name+'.hepta')).write_bytes(source)
        r=subprocess.run([str(compiler)],input=source,capture_output=True,env={'PATH':''},timeout=60)
        (folder/(name+'.o')).write_bytes(r.stdout);(folder/(name+'.stderr')).write_bytes(r.stderr)
        assert r.returncode==0 and not r.stderr and r.stdout.startswith(b'\x7fELF')
        return r.stdout
    new_objects=[]
    for row in receipt['units']:
        path=current/(row['name']+'.hepta');raw=path.read_bytes();assert SHA(raw)==row['source_sha256'];sources[path]=raw
        transformed=rename(raw,b'new_');assert rename(transformed,b'new_',True)==raw
        new_objects.append(compile_unit('new_'+row['name'],transformed,common))
        bindings.append({'role':'common','name':row['name'],'original_sha256':SHA(raw),'renamed_sha256':SHA(transformed),'inverse_byte_identical':True})
    controller=(package/'state_handoff.hepta').read_bytes();sources[package/'state_handoff.hepta']=controller
    controller_object=compile_unit('controller',controller,common)
    report={'schema':'heptatorch.tensor-rng-common-profile-handoffs.v1','status':'RUNNING','current_source_sha256':current_pin,
        'current_executable_sha256':pins['current_executable_sha256'],'compiler_sha256':COMPILER,'linker_sha256':LINKER,
        'source_bindings':bindings,'peers':[],'frames':0,'raw_pointer_import':False}
    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    for label,pin in pins['peers'].items():
        root=Path(roots[label]).resolve();directory=out/label;directory.mkdir()
        raw=(root/'runtime.hepta').read_bytes();assert SHA(raw)==pin['source_sha256'];sources[root/'runtime.hepta']=raw
        objects=[]
        for row in pin['units']:
            path=root/(row['name']+'.hepta');raw=path.read_bytes();assert SHA(raw)==row['source_sha256'];sources[path]=raw
            transformed=rename(raw,b'old_');assert rename(transformed,b'old_',True)==raw
            objects.append(compile_unit('old_'+row['name'],transformed,directory))
            bindings.append({'role':label,'name':row['name'],'original_sha256':SHA(raw),'renamed_sha256':SHA(transformed),'inverse_byte_identical':True})
        objects+=new_objects+[controller_object]
        request=b'HIASTKS1'+struct.pack('<3Q',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(x))+x for x in objects)
        r=subprocess.run([str(linker)],input=request,capture_output=True,env={'PATH':''},timeout=90)
        (directory/'link.stderr').write_bytes(r.stderr);assert r.returncode==0 and not r.stderr and r.stdout.startswith(b'\x7fELF')
        image=directory/'handoff-session';image.write_bytes(r.stdout);image.chmod(0o500)
        s=wire.Session(image,directory=directory/'frames')
        def route(which,commands=(),error=0):
            before=s.owner() if error else None
            status,rows=wire.parse(s.exchange(struct.pack('<Q',which)+(wire.packet(commands) if which<2 else b'')))
            assert status==error
            if error:assert before==s.owner()
            return rows
        commands=[wire.full((3,),7,1),(46,wire.reference(1)),(53,)]
        if pin['has_generator']:commands += [(60,19),(63,wire.reference(4))]
        rows=route(0,commands);tensor,saved=rows[0]['value'],rows[1]['value'];old=s.owner()
        assert old['bytes'][64:72]==pin['state_magic'].encode()
        route(1,[(60,99),(68,tensor,wire.reference(1),0,wire.bits(1),1)],7)
        got=route(0,[(39,tensor,0,3),(49,saved,0,3),(53,)])
        assert got[0]['payload']==got[1]['payload']==struct.pack('<3f',7,7,7)
        if pin['has_generator']:assert route(0,[(63,rows[3]['value'])])[0]['payload']==rows[4]['payload']
        route(2);assert s.owner()['length']==0 and not s.mapped(old['pointer'])
        rows=route(1,[wire.full((3,),0,1),(60,99),(68,wire.reference(1),wire.reference(2),0,wire.bits(1),1),wire.full((3,),1),(58,wire.reference(1),wire.reference(4),0),(70,wire.reference(1)),(63,wire.reference(2)),(59,wire.reference(1)),(53,)])
        tensor,generator=rows[0]['value'],rows[1]['value'];state=rows[6]['payload'];assert rows[7]['value']==0;new=s.owner();assert new['bytes'][64:72]==b'HTRNGN01'
        route(0,[(12,tensor,0,wire.bits(999)),(33,0)],7)
        got=route(1,[(63,generator),(13,tensor),(53,),(59,tensor)])
        assert got[0]['payload']==state and wire.meta(got[1])['version']==1 and got[3]['value']==0
        route(3);assert s.owner()['length']==0 and not s.mapped(new['pointer'])
        s.finish();report['frames']+=s.frames
        report['peers'].append({'name':label,'state_magic':pin['state_magic'],'status':'PASS','frames':s.frames,
            'peer_source_sha256':pin['source_sha256'],'qualification_executable_sha256':SHA(r.stdout),
            'modules':len(objects),'both_directions_refuse_and_recover':True,'owner_pointer_length_and_bytes_preserved':True,
            'both_closed_owner_mappings_absent':True,'new_state_has_genuinely_cleared_nullable_leaf':True});save()
    for path,raw in sources.items():assert path.read_bytes()==raw
    assert SHA(compiler.read_bytes())==COMPILER and SHA(linker.read_bytes())==LINKER
    report.update(status='PASS',cases=len(report['peers'])*2,all_source_custody_unchanged=True);save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_bindings','peers')}))

if __name__=='__main__':main()
