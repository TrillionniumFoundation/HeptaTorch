#!/usr/bin/env python3
"""Compile two complete runtimes and a checked real-owner handoff controller."""
import argparse,hashlib,json,re,struct,subprocess
from pathlib import Path

SHA=lambda b:hashlib.sha256(b).hexdigest()
HERE=Path(__file__).resolve().parent
BASE_SHA='4e2cc7f7612c826ec35280432be527cf98075af7d0f2c35a560df67e35c3904e'
OWNER_SHA='993651c88e69be93dea49ef473b8409b9f609e5b226c6fcec10de230ef4af99e'
COMPILER_SHA='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER_SHA='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
FUNCTION=re.compile(rb'^(fn (\w+)\([^\n]* entry\(\w+\)\n.*?^end\n)',re.M|re.S)

def rename(raw,prefix,reverse=False):
    def symbol(name):
        if reverse:
            assert name.startswith(prefix);return name[len(prefix):]
        return prefix+name
    raw=re.sub(rb'^(hia-asm\.v1 hepta\.hia\.program\.v8 )(\w+)$',lambda m:m[1]+symbol(m[2]),raw,flags=re.M)
    raw=re.sub(rb'^(fn )(\w+)(\()',lambda m:m[1]+symbol(m[2])+m[3],raw,flags=re.M)
    raw=re.sub(rb'^exports\(([^)]*)\)$',lambda m:b'exports('+b','.join(symbol(n) for n in m[1].split(b','))+b')',raw,flags=re.M)
    return re.sub(rb'(\{"function":")(\w+)("\})',lambda m:m[1]+symbol(m[2])+m[3],raw)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--stack-build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    compiler=a.stack_build.resolve()/'native/stack-module-compiler';linker=a.stack_build.resolve()/'native/stack-linker-session'
    assert SHA(compiler.read_bytes())==COMPILER_SHA and SHA(linker.read_bytes())==LINKER_SHA
    old=a.baseline.resolve();new=a.build.resolve()
    assert SHA((old/'runtime.hepta').read_bytes())==BASE_SHA
    assert SHA((old/'objects_owners.hepta').read_bytes())==OWNER_SHA
    expected={name:body for body,name in FUNCTION.findall((old/'runtime.hepta').read_bytes()+(old/'objects_owners.hepta').read_bytes())}
    old_units=sorted(old.glob('objects_[0-9][0-9].hepta'))+[old/'objects_owners.hepta']
    actual={name:body for path in old_units for body,name in FUNCTION.findall(path.read_bytes())}
    assert actual==expected and len(old_units)==5
    receipt=json.loads((new/'build.json').read_text());assert receipt['status']=='NATIVE_BUILD_PASS'
    assert SHA((new/'tensor-rng-session').read_bytes())==receipt['executable_sha256']
    assert SHA((new/'runtime.hepta').read_bytes())==receipt['runtime_source_sha256']
    new_units=[new/(row['name']+'.hepta') for row in receipt['units']]
    for path,row in zip(new_units,receipt['units']):assert SHA(path.read_bytes())==row['source_sha256']
    units=[];bindings=[];sources={}
    for prefix,paths in [('old_',old_units),('new_',new_units)]:
        for path in paths:
            source=path.read_bytes();sources[path]=source;changed=rename(source,prefix.encode())
            assert rename(changed,prefix.encode(),True)==source
            name=prefix+path.stem;units.append((name,changed))
            bindings.append({'original':path.name,'source_sha256':SHA(source),'renamed':name,
                'renamed_sha256':SHA(changed),'inverse_symbol_rename_byte_identical':True})
    controller=(HERE/'state_handoff.hepta').read_bytes();units.append(('state_handoff',controller))
    report={'schema':'heptatorch.tensor-rng-owned-handoff-build.v1','status':'BUILDING','qualification_only':True,
        'compiler_sha256':COMPILER_SHA,'linker_sha256':LINKER_SHA,'old_source_sha256':BASE_SHA,
        'new_source_sha256':receipt['runtime_source_sha256'],'source_bindings':bindings,'units':[]}
    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    objects=[];save()
    for name,raw in units:
        (out/(name+'.hepta')).write_bytes(raw);r=subprocess.run([str(compiler)],input=raw,capture_output=True,env={'PATH':''},timeout=60)
        (out/(name+'.o')).write_bytes(r.stdout);(out/(name+'.stderr')).write_bytes(r.stderr)
        report['units'].append({'name':name,'source_sha256':SHA(raw),'compiler_exit':r.returncode,'object_sha256':SHA(r.stdout)})
        save();assert r.returncode==0 and not r.stderr and r.stdout.startswith(b'\x7fELF');objects.append(r.stdout)
    request=b'HIASTKS1'+struct.pack('<3Q',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(x))+x for x in objects)
    (out/'link.request').write_bytes(request);r=subprocess.run([str(linker)],input=request,capture_output=True,env={'PATH':''},timeout=90)
    (out/'link.stdout').write_bytes(r.stdout);(out/'link.stderr').write_bytes(r.stderr);assert r.returncode==0 and not r.stderr
    image=out/'state-handoff-session';image.write_bytes(r.stdout);image.chmod(0o500)
    for path,raw in sources.items():assert path.read_bytes()==raw
    assert (HERE/'state_handoff.hepta').read_bytes()==controller
    assert SHA(compiler.read_bytes())==COMPILER_SHA and SHA(linker.read_bytes())==LINKER_SHA
    report.update(status='PASS',executable_sha256=SHA(r.stdout),executable_bytes=len(r.stdout),module_count=len(objects))
    save();print(json.dumps({k:v for k,v in report.items() if k not in ('source_bindings','units')}))

if __name__=='__main__':main()
