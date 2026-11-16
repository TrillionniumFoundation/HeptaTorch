#!/usr/bin/env python3
"""Compile and pack the flat-linker profile with unchanged native compilers."""
from pathlib import Path
import argparse,hashlib,json,struct,subprocess

sha=lambda raw:hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('compiler','module-compiler','generated','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    compiler=args.compiler.resolve();other=args.module_compiler.resolve();generated=args.generated.resolve()
    pins=json.loads((Path(__file__).resolve().parent/'profile.json').read_text())
    assert sha(compiler.read_bytes())==pins['compiler_sha256']
    assert sha(other.read_bytes())==pins['module_compiler_sha256']
    source=(generated/'flat-linker.hepta').read_bytes()
    pack_source=(generated/'linker-packer.hepta').read_bytes()
    assert sha(source)==pins['flat_linker_source_sha256']
    assert sha(pack_source)==pins['linker_packer_source_sha256']
    def run(exe,data,label):
        result=subprocess.run([str(exe)],input=data,capture_output=True,env={'PATH':''},timeout=60)
        (out/(label+'.stdout')).write_bytes(result.stdout);(out/(label+'.stderr')).write_bytes(result.stderr)
        assert result.returncode==0 and not result.stderr and result.stdout,(label,result.returncode,result.stderr)
        return result.stdout
    image=run(compiler,source,'image-c5');second=run(other,source,'image-e250')
    assert image==second and sha(image)==pins['flat_linker_image_sha256']
    (out/'flat-linker.image').write_bytes(image)
    packer=run(compiler,pack_source,'packer-c5');packer2=run(other,pack_source,'packer-e250')
    assert packer==packer2 and sha(packer)==pins['linker_packer_sha256']
    path=out/'linker-packer';path.write_bytes(packer);path.chmod(0o500)
    request=b'HIAFLT01'+struct.pack('<Q',len(image))+image
    (out/'packing-request.bin').write_bytes(request)
    executable=run(path,request,'native-pack')
    assert sha(executable)==pins['flat_linker_sha256']
    path=out/'flat-linker';path.write_bytes(executable);path.chmod(0o500)
    assert sha(compiler.read_bytes())==pins['compiler_sha256']
    assert sha(other.read_bytes())==pins['module_compiler_sha256']
    assert (generated/'flat-linker.hepta').read_bytes()==source
    assert (generated/'linker-packer.hepta').read_bytes()==pack_source
    assert sha((out/'linker-packer').read_bytes())==pins['linker_packer_sha256']
    assert sha(path.read_bytes())==pins['flat_linker_sha256']
    report={'schema':'heptatorch.flat-linker-native-build.v1','pass':True,
            'source_sha256':sha(source),'image_sha256':sha(image),
            'packer_source_sha256':sha(pack_source),'packer_sha256':sha(packer),
            'executable_sha256':sha(executable),
            'compiler_sha256':sha(compiler.read_bytes()),'module_compiler_sha256':sha(other.read_bytes()),
            'both_sources_compile_byte_identically':True,'native_compilation':True,'native_packing':True,
            'producer_and_source_hashes_verified_before_and_after':True}
    (out/'build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
