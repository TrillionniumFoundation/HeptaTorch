"""Build a qualification-only checked HIA owner handoff between two real runtimes.

All old/new unit source bytes are reconstructed by reversible symbol renaming.
The controller owns no foreign pointer and never manufactures a nonempty owner.
"""
from pathlib import Path
import argparse,hashlib,json,re,struct,subprocess

SHA=lambda data:hashlib.sha256(data).hexdigest()
HERE=Path(__file__).resolve().parent
COMPILER_SHA='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER_SHA='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'

def rename(raw,prefix,reverse=False):
    def symbol(name):
        if reverse:
            assert name.startswith(prefix)
            return name[len(prefix):]
        return prefix+name
    raw=re.sub(rb'^(hia-asm\.v1 hepta\.hia\.program\.v8 )(\w+)$',
               lambda m:m[1]+symbol(m[2]),raw,flags=re.M)
    raw=re.sub(rb'^(fn )(\w+)(\()',lambda m:m[1]+symbol(m[2])+m[3],raw,flags=re.M)
    raw=re.sub(rb'^exports\(([^)]*)\)$',
               lambda m:b'exports('+b','.join(symbol(n) for n in m[1].split(b','))+b')',raw,flags=re.M)
    return re.sub(rb'(\{"function":")(\w+)("\})',lambda m:m[1]+symbol(m[2])+m[3],raw)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--old',type=Path,required=True)
    ap.add_argument('--new',type=Path,required=True)
    ap.add_argument('--compiler',type=Path,required=True)
    ap.add_argument('--linker',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    COMPILER=args.compiler.resolve();LINKER=args.linker.resolve()
    assert SHA(COMPILER.read_bytes())==COMPILER_SHA
    assert SHA(LINKER.read_bytes())==LINKER_SHA
    sources={Path(__file__).resolve():Path(__file__).read_bytes(),HERE/'state_handoff.hepta':(HERE/'state_handoff.hepta').read_bytes()}
    units=[];bindings=[]
    expected={'old':'e9e325374153f64744fbddfd2ccee00e570585cc42e4ad239361ce035ad4d2c2',
              'new':'302677af52d3e04520e6f132aaf17b98cbe5d2cf404aec040bf5144e324d2d4d'}
    for label,directory,stem in [('old',args.old.resolve(),'release_'),('new',args.new.resolve(),'partial_')]:
        core=directory/'runtime.hepta';sources[core]=core.read_bytes();assert SHA(sources[core])==expected[label]
        report_path=directory/'build.json';sources[report_path]=report_path.read_bytes()
        paths=sorted(directory.glob(stem+'[0-9][0-9].hepta'))+[directory/(stem+'owners.hepta')]
        assert len(paths)==5
        for path in paths:
            raw=path.read_bytes();sources[path]=raw
            transformed=rename(raw,(label+'_').encode())
            assert rename(transformed,(label+'_').encode(),True)==raw
            name=label+'_'+path.stem;units.append((name,transformed))
            bindings.append({'label':label,'original_unit':path.name,'original_sha256':SHA(raw),
                             'renamed_unit':name,'renamed_sha256':SHA(transformed),
                             'inverse_symbol_rename_byte_identical':True})
    units.append(('state_handoff',sources[HERE/'state_handoff.hepta']))
    report={'schema':'heptatorch.owned-state-cross-profile-build.v1','qualification_only':True,
            'status':'BUILDING','old_core_sha256':expected['old'],'new_core_sha256':expected['new'],
            'compiler_sha256':COMPILER_SHA,'linker_sha256':LINKER_SHA,'source_bindings':bindings,
            'input_sha256':{str(p):SHA(b) for p,b in sources.items()},'units':[]}
    def save(): (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save();objects=[]
    for name,source in units:
        (out/(name+'.hepta')).write_bytes(source)
        result=subprocess.run([str(COMPILER)],input=source,capture_output=True,timeout=120,env={'PATH':''})
        (out/(name+'.o')).write_bytes(result.stdout);(out/(name+'.stderr')).write_bytes(result.stderr)
        row={'name':name,'source_sha256':SHA(source),'source_bytes':len(source),
             'compiler_exit':result.returncode,'object_sha256':SHA(result.stdout),'object_bytes':len(result.stdout)}
        report['units'].append(row);save()
        assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF'),row
        objects.append(result.stdout)
    request=b'HIASTKS1'+struct.pack('<3Q',len(objects),4,0)+b'step'+b''.join(struct.pack('<Q',len(o))+o for o in objects)
    (out/'link.request.bin').write_bytes(request)
    result=subprocess.run([str(LINKER)],input=request,capture_output=True,timeout=120,env={'PATH':''})
    (out/'link.stdout').write_bytes(result.stdout);(out/'link.stderr').write_bytes(result.stderr)
    report['link_exit']=result.returncode;save()
    assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
    binary=out/'state-handoff-session';binary.write_bytes(result.stdout);binary.chmod(0o500)
    for p,data in sources.items():assert p.read_bytes()==data,p
    for (name,source),obj in zip(units,objects):
        assert (out/(name+'.hepta')).read_bytes()==source
        assert (out/(name+'.o')).read_bytes()==obj
    assert SHA(COMPILER.read_bytes())==COMPILER_SHA and SHA(LINKER.read_bytes())==LINKER_SHA
    report.update(status='PASS',native_sha256=SHA(result.stdout),native_bytes=len(result.stdout),
                  module_count=len(objects),link_request_bytes=len(request));save()
    print(json.dumps({k:report[k] for k in ('status','native_sha256','native_bytes','module_count','link_request_bytes')},indent=2))

if __name__=='__main__':main()
