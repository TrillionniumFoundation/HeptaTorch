"""Source/wire transport to the actual checked owned HIA compiler/linkers."""
from pathlib import Path
import argparse,hashlib,json,struct,subprocess

PINS={'compiler':'0ccdb524ea327ab290b2e70feaac49a4789abb2fee6da8746a095f435c17af73',
      'linker':'55a2b0a762c8db574a65d0d0e6a18e3cedc668c01d20080a11c26cf1ccb601b3',
      'qualification_linker':'4381b6d3425b501c8a1b5e2a5496ab0f27037e18f0e74727b321006327e6a9ee'}
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('compiler','linker','qualification-linker','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('sources',type=Path,nargs='+')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    producers={name:getattr(args,name).resolve() for name in PINS}
    for name,path in producers.items():assert sha(path.read_bytes())==PINS[name]
    def run(exe,data,label):
        result=subprocess.run([str(exe)],input=data,capture_output=True,env={'PATH':''},timeout=60)
        (out/(label+'.stdout')).write_bytes(result.stdout);(out/(label+'.stderr')).write_bytes(result.stderr)
        assert result.returncode==0 and not result.stderr and result.stdout,(label,result.returncode,result.stderr)
        return result.stdout
    objects=[];sources=[];rows=[]
    for index,path in enumerate(args.sources):
        source=path.read_bytes();sources.append(source)
        snapshot=out/f'unit-{index}.hepta';snapshot.write_bytes(source)
        obj=run(producers['compiler'],source,f'unit-{index}-compile')
        assert obj.startswith(b'\x7fELF') and snapshot.read_bytes()==source and path.read_bytes()==source
        (out/f'unit-{index}.o').write_bytes(obj);objects.append(obj)
        rows.append({'source':str(path.resolve()),'source_sha256':sha(source),
                     'object_sha256':sha(obj),'object_bytes':len(obj)})
    executables={}
    for name,magic,exe_name in [('linker',b'HIAOWNS1','service'),
                              ('qualification_linker',b'HIAOWNT1','service-close-probe')]:
        request=magic+struct.pack('<QQQ',len(objects),4,0)+b'step'+b''.join(
            struct.pack('<Q',len(obj))+obj for obj in objects)
        (out/(name+'.request.bin')).write_bytes(request)
        executable=run(producers[name],request,name)
        assert executable.startswith(b'\x7fELF')
        path=out/exe_name;path.write_bytes(executable);path.chmod(0o500)
        executables[exe_name]={'sha256':sha(executable),'bytes':len(executable),
                              'request_sha256':sha(request),'wire_magic':magic.decode()}
    for name,path in producers.items():assert sha(path.read_bytes())==PINS[name]
    for index,path in enumerate(args.sources):
        assert path.read_bytes()==sources[index] and (out/f'unit-{index}.hepta').read_bytes()==sources[index]
        assert (out/f'unit-{index}.o').read_bytes()==objects[index]
    report={'schema':'heptatorch.owned-service-native-link.v1','pass':True,
            'producer_pins':PINS,'sources':rows,'executables':executables,
            'same_compiled_objects_for_production_and_probe':True,
            'source_producer_hashes_verified_before_and_after':True,
            'native_compilation_and_linking':True,'no_host_linker_or_numerical_runtime':True}
    (out/'build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
