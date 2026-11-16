#!/usr/bin/env python3
"""Compile and pack the wide HIA packer through pinned native HIA tools."""
from pathlib import Path
import argparse,hashlib,json,struct,subprocess

C5='c5a9c2b95a3061e080a21298f5b26000936ed9dac519690d845b0699a15744c6'
E250='e250d93bc0b2a5ff878d2188c95780abf7e7ebaf186e4531b136fa2735b2af59'
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--module-compiler',type=Path,required=True)
    p.add_argument('--generated',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    pins=json.loads((Path(__file__).resolve().parent/'profile.json').read_text())
    compiler=args.compiler.resolve();other=args.module_compiler.resolve();generated=args.generated.resolve()
    assert sha(compiler.read_bytes())==C5 and sha(other.read_bytes())==E250
    metadata=json.loads((generated/'authoring.json').read_text())
    source=(generated/'session-packer.hepta').read_bytes()
    tool_source=(generated/'tool-packer.hepta').read_bytes()
    assert sha(source)==metadata['source_sha256'] and sha(tool_source)==metadata['tool_packer_source_sha256']
    assert sha(source)==pins['wide_packer_source_sha256'] and sha(tool_source)==pins['tool_packer_source_sha256']
    def run(exe,data,label):
        result=subprocess.run([str(exe)],input=data,capture_output=True,env={'PATH':''},timeout=60)
        (out/(label+'.stdout')).write_bytes(result.stdout);(out/(label+'.stderr')).write_bytes(result.stderr)
        assert result.returncode==0 and not result.stderr and result.stdout,(label,result.returncode,result.stderr)
        return result.stdout
    def publish(name,raw):
        path=out/name;path.write_bytes(raw);path.chmod(0o500);return path
    image=run(compiler,source,'image-c5');image2=run(other,source,'image-e250')
    assert image==image2
    assert sha(image)==pins['wide_packer_image_sha256']
    (out/'session-packer.image').write_bytes(image)
    tool=run(compiler,tool_source,'tool-c5');tool2=run(other,tool_source,'tool-e250')
    assert tool==tool2
    assert sha(tool)==pins['tool_packer_sha256']
    toolpath=publish('tool-packer',tool);toolpath2=publish('tool-packer-reproduced',tool2)
    request=b'HIAFLT01'+struct.pack('<Q',len(image))+image
    (out/'packing-request.bin').write_bytes(request)
    executable=run(toolpath,request,'packed-c5')
    executable2=run(toolpath2,b'HIAFLT01'+struct.pack('<Q',len(image2))+image2,'packed-e250')
    assert executable==executable2
    assert sha(executable)==pins['wide_packer_sha256']
    publish('session-packer',executable)
    assert sha(compiler.read_bytes())==C5 and sha(other.read_bytes())==E250
    assert (generated/'session-packer.hepta').read_bytes()==source
    assert (generated/'tool-packer.hepta').read_bytes()==tool_source
    assert sha(toolpath.read_bytes())==pins['tool_packer_sha256']
    assert sha(toolpath2.read_bytes())==pins['tool_packer_sha256']
    assert sha((out/'session-packer').read_bytes())==pins['wide_packer_sha256']
    report={'schema':'heptatorch.wide-session-packer-native-build.v1','pass':True,
            'source_sha256':sha(source),'tool_packer_source_sha256':sha(tool_source),
            'compiler_sha256':C5,'module_compiler_sha256':E250,
            'image_sha256':sha(image),'tool_packer_executable_sha256':sha(tool),
            'executable_sha256':sha(executable),'executable_bytes':len(executable),
            'both_sources_compile_byte_identically':True,'both_native_packing_results_byte_identical':True,
            'native_compilation_performed':True,'native_packing_performed':True,
            'producer_and_source_hashes_verified_before_and_after':True,
            'runtime_host_dependency':False,
            'authoring_receipt_sha256':sha((generated/'authoring.json').read_bytes()),
            'limits':{k:v for k,v in metadata.items() if k.endswith('_bytes') or k.endswith('_copies')}}
    (out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
