#!/usr/bin/env python3
"""Build a persistent native runtime from the single authoritative HIA source.

Only the checked session entry is substituted in build storage. The 142 other
function bodies, including init and every numerical kernel, remain byte exact.
"""
from pathlib import Path
import argparse,hashlib,json,re,struct,subprocess,sys

SPEC=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
FUNCTION=re.compile(rb'^(fn ([A-Za-z_][A-Za-z_0-9]*)\([^\n]*\n.*?^end\n)',re.M|re.S)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--mode',choices=['modular','monolithic'],default='modular')
    p.add_argument('--compiler',type=Path)
    p.add_argument('--module-compiler',type=Path)
    p.add_argument('--flat-linker',type=Path)
    p.add_argument('--packer',type=Path,required=True)
    p.add_argument('--module-tool-dir',type=Path,default=SPEC.parent/'modules')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    profile_raw=(SPEC/'profile.json').read_bytes()
    pins=json.loads(profile_raw)
    source_path=args.source.resolve();source=source_path.read_bytes()
    assert sha(source)==pins['canonical_source_sha256']
    replacement=(SPEC/'entry.hepta').read_bytes();assert sha(replacement)==pins['session_entry_sha256']
    functions={name.decode():body for body,name in FUNCTION.findall(source)}
    assert len(functions)==pins['canonical_function_count']
    assert sha(functions['entry'])==pins['original_entry_sha256']
    assert replacement.startswith(b'fn entry(') and replacement.endswith(b'end\n')
    assert source.count(functions['entry'])==1
    overlay=source.replace(functions['entry'],replacement,1)
    changed={name.decode():body for body,name in FUNCTION.findall(overlay)}
    assert list(changed)==list(functions)
    assert [name for name in functions if functions[name]!=changed[name]]==['entry']
    assert sha(overlay)==pins['overlay_source_sha256']
    (out/'session-overlay.hepta').write_bytes(overlay)
    packer=args.packer.resolve();assert sha(packer.read_bytes())==pins['wide_packer_sha256']
    producers={packer:pins['wide_packer_sha256']}
    unit_snapshots=[]
    def run(exe,raw,label):
        result=subprocess.run([str(exe)],input=raw,capture_output=True,env={'PATH':''},timeout=60)
        (out/(label+'.stdout')).write_bytes(result.stdout);(out/(label+'.stderr')).write_bytes(result.stderr)
        assert result.returncode==0 and not result.stderr and result.stdout,(label,result.returncode,result.stderr)
        return result.stdout
    objects=[];split_sha=None
    if args.mode=='monolithic':
        assert args.compiler is not None
        compiler=args.compiler.resolve();assert sha(compiler.read_bytes())==pins['compiler_sha256']
        producers[compiler]=pins['compiler_sha256']
        flat=overlay.replace(b'hia-asm.v1 hepta.hia.program.v8 selfhost_exec\n',
                             b'hia-asm.v1 hepta.hia.program.v8 storage_session_kernel\n',1)
        assert sha(flat)==pins['flat_reference_source_sha256']
        (out/'flat-kernel.hepta').write_bytes(flat)
        image=run(compiler,flat,'compile-flat')
    else:
        assert args.module_compiler is not None and args.flat_linker is not None
        compiler=args.module_compiler.resolve();linker=args.flat_linker.resolve()
        assert sha(compiler.read_bytes())==pins['module_compiler_sha256']
        assert sha(linker.read_bytes())==pins['flat_linker_sha256']
        producers[compiler]=pins['module_compiler_sha256']
        producers[linker]=pins['flat_linker_sha256']
        splitter=args.module_tool_dir.resolve()/'split_runtime.py'
        assert sha(splitter.read_bytes())==pins['splitter_sha256']
        producers[splitter]=pins['splitter_sha256']
        result=subprocess.run([sys.executable,'-I','-S','-B',str(splitter),
                               '--source',str(out/'session-overlay.hepta'),
                               '--output',str(out/'units')],capture_output=True,timeout=30)
        (out/'split.stdout').write_bytes(result.stdout);(out/'split.stderr').write_bytes(result.stderr)
        assert result.returncode==0 and not result.stderr
        manifest=json.loads((out/'units/split-manifest.json').read_text())
        assert manifest['source_sha256']==sha(overlay) and manifest['partition_complete']
        split_sha=sha((out/'units/split-manifest.json').read_bytes())
        # This is file transport, not an object parser or host link gate.
        for name in ('ht_runtime','ht_kernel'):
            unit_path=out/'units'/f'{name}.hepta'
            unit=unit_path.read_bytes();unit_snapshots.append((unit_path,unit))
            obj=run(compiler,unit,name)
            assert unit_path.read_bytes()==unit
            (out/f'{name}.o').write_bytes(obj);objects.append(obj)
        request=b'HIAFLK01'+struct.pack('<QQQ',len(objects),5,0)+b'entry'+b''.join(
            struct.pack('<Q',len(obj))+obj for obj in objects)
        (out/'link-request.bin').write_bytes(request)
        image=run(linker,request,'native-link')
    (out/'session.image').write_bytes(image)
    request=b'HIAFLT01'+struct.pack('<Q',len(image))+image
    (out/'packing-request.bin').write_bytes(request)
    executable=run(packer,request,'native-pack')
    assert sha(executable)==pins[args.mode+'_session_sha256']
    application=out/'application';application.write_bytes(executable);application.chmod(0o500)
    assert source_path.read_bytes()==source
    assert (SPEC/'entry.hepta').read_bytes()==replacement
    assert (SPEC/'profile.json').read_bytes()==profile_raw
    assert (out/'session-overlay.hepta').read_bytes()==overlay
    assert application.read_bytes()==executable
    for path,expected in producers.items():assert sha(path.read_bytes())==expected
    for path,original in unit_snapshots:assert path.read_bytes()==original
    if args.mode=='modular':
        for name,obj in zip(('ht_runtime','ht_kernel'),objects):
            assert (out/f'{name}.o').read_bytes()==obj
    report={'schema':'heptatorch.persistent-runtime-build.v1','pass':True,'mode':args.mode,
            'source_sha256':sha(source),'overlay_sha256':sha(overlay),
            'session_entry_sha256':sha(replacement),'changed_functions':['entry'],
            'unchanged_function_count':len(functions)-1,'init_unchanged':True,
            'unchanged_function_hashes':{n:sha(functions[n]) for n in functions if n!='entry'},
            'native_compiler_sha256':sha(compiler.read_bytes()),
            'native_linker_sha256':None if args.mode=='monolithic' else sha(linker.read_bytes()),
            'object_sha256':[sha(obj) for obj in objects],'split_manifest_sha256':split_sha,
            'image_sha256':sha(image),'image_bytes':len(image),
            'packer_sha256':sha(packer.read_bytes()),'application_sha256':sha(executable),
            'application_bytes':len(executable),'native_compilation':True,
            'native_linking':args.mode=='modular','native_packing':True,
            'producer_source_entry_splitter_hashes_verified_before_and_after':True,
            'unit_snapshots_unchanged':True,
            'host_numeric_or_linker_runtime':False}
    (out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='unchanged_function_hashes'},indent=2))


if __name__=='__main__':main()
