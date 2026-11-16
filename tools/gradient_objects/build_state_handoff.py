"""Build a qualification-only controller with both complete checked owner steps.

Only symbol prefixes change. Every original unit is verified against its build
receipt and inverse renaming restores its exact bytes. The controller is the
already qualified native owner transfer controller, not a fabricated owner.
"""
from pathlib import Path
import argparse, hashlib, json, re, struct, subprocess

COMPILER='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
CONTROLLER='4093b8a1a4ce1b19d6943f457649355c950cfba50743ac7afe8a8396437cc7ee'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def rename(raw,prefix,reverse=False):
    def symbol(value):
        if reverse:
            assert value.startswith(prefix);return value[len(prefix):]
        return prefix+value
    raw=re.sub(rb'^(hia-asm\.v1 hepta\.hia\.program\.v8 )(\w+)$',lambda m:m[1]+symbol(m[2]),raw,flags=re.M)
    raw=re.sub(rb'^(fn )(\w+)(\()',lambda m:m[1]+symbol(m[2])+m[3],raw,flags=re.M)
    raw=re.sub(rb'^exports\(([^)]*)\)$',lambda m:b'exports('+b','.join(symbol(n) for n in m[1].split(b','))+b')',raw,flags=re.M)
    return re.sub(rb'(\{"function":")(\w+)("\})',lambda m:m[1]+symbol(m[2])+m[3],raw)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('old','new','controller','compiler','linker','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('old-source-sha256','new-source-sha256','old-tag','new-tag'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    compiler=a.compiler.resolve();linker=a.linker.resolve()
    assert sha(compiler.read_bytes())==COMPILER and sha(linker.read_bytes())==LINKER
    assert len(a.old_tag)==len(a.new_tag)==8 and a.old_tag!=a.new_tag
    inputs={Path(__file__).resolve():Path(__file__).read_bytes(),a.controller.resolve():a.controller.read_bytes()}
    assert sha(a.controller.read_bytes())==CONTROLLER
    report={'schema':'heptatorch.gradient-owner-format-handoff-build.v1','status':'BUILDING','qualification_only':True,
        'old_core_sha256':a.old_source_sha256,'new_core_sha256':a.new_source_sha256,
        'old_tag':a.old_tag,'new_tag':a.new_tag,'old_has_optimizer':True,
        'compiler_sha256':COMPILER,'linker_sha256':LINKER,'source_bindings':[],'units':[]}
    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    units=[]
    for label,directory,pin in [('old',a.old.resolve(),a.old_source_sha256),('new',a.new.resolve(),a.new_source_sha256)]:
        core=directory/'runtime.hepta';receipt=directory/'build.json';inputs[core]=core.read_bytes();inputs[receipt]=receipt.read_bytes()
        assert sha(inputs[core])==pin
        original=json.loads(inputs[receipt]);assert original['status']=='PASS'
        for unit in original['units']:
            path=directory/(unit['name']+'.hepta');raw=path.read_bytes();inputs[path]=raw
            assert sha(raw)==unit['source_sha256']
            renamed=rename(raw,(label+'_').encode());assert rename(renamed,(label+'_').encode(),True)==raw
            name=label+'_'+unit['name'];units.append((name,renamed))
            report['source_bindings'].append({'side':label,'name':name,'original_sha256':sha(raw),
                'renamed_sha256':sha(renamed),'inverse_symbol_rename_byte_identical':True})
    units.append(('state_handoff',a.controller.read_bytes()))
    report['inputs_sha256']={str(path):sha(raw) for path,raw in inputs.items()};save();objects=[]
    for name,source in units:
        (out/(name+'.hepta')).write_bytes(source)
        result=subprocess.run([str(compiler)],input=source,capture_output=True,timeout=120,env={'PATH':''})
        (out/(name+'.o')).write_bytes(result.stdout);(out/(name+'.stderr')).write_bytes(result.stderr)
        report['units'].append({'name':name,'source_sha256':sha(source),'source_bytes':len(source),
            'compiler_exit':result.returncode,'object_sha256':sha(result.stdout),'object_bytes':len(result.stdout)})
        save();assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
        objects.append(result.stdout)
    request=b'HIASTKS1'+struct.pack('<3Q',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(obj))+obj for obj in objects)
    (out/'link.request.bin').write_bytes(request)
    result=subprocess.run([str(linker)],input=request,capture_output=True,timeout=120,env={'PATH':''})
    (out/'link.stderr').write_bytes(result.stderr);report['link_exit']=result.returncode;save()
    assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
    binary=out/'state-handoff-session';binary.write_bytes(result.stdout);binary.chmod(0o500)
    assert all(path.read_bytes()==raw for path,raw in inputs.items())
    assert sha(compiler.read_bytes())==COMPILER and sha(linker.read_bytes())==LINKER
    report.update(status='PASS',native_sha256=sha(result.stdout),native_bytes=len(result.stdout),module_count=len(objects),link_request_bytes=len(request));save()
    print(json.dumps({k:report[k] for k in ('status','native_sha256','native_bytes','module_count')}))
if __name__=='__main__':main()
