"""Observe real wide native packing, exact bounds and atomic refused output."""
from pathlib import Path
import argparse,hashlib,json,struct,subprocess,sys

M=1<<20;MAXIMUM=8*M
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--generated',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    exe=args.executable.resolve();image=args.image.resolve();generated=args.generated.resolve()
    metadata=json.loads((generated/'authoring.json').read_text())
    prefix=(generated/'session-prefix.bin').read_bytes();hole=metadata['session_call_hole']
    assert len(prefix)==774 and hole==465 and prefix[hole-1]==0xe8
    assert prefix[:7]==b'\x7fELF\x02\x01\x01'
    assert struct.unpack_from('<HHI',prefix,16)==(2,62,1)
    assert struct.unpack_from('<Q',prefix,24)[0]==0x4000b0
    assert struct.unpack_from('<II',prefix,64)==(1,5)
    assert struct.unpack_from('<II',prefix,120)==(0x6474e551,6)
    rows=[];cases={}
    def pack_frame(payload):return b'HIAFLT01'+struct.pack('<Q',len(payload))+payload
    def expected(payload):
        header=bytearray(prefix);size=len(prefix)+len(payload)
        struct.pack_into('<Q',header,96,size);struct.pack_into('<Q',header,104,size)
        struct.pack_into('<i',header,hole,len(prefix)-hole-4)
        return bytes(header)+payload
    def cli(name,request,status,wanted=None):
        path=out/(name+'.request.bin');path.write_bytes(request)
        result=subprocess.run([str(exe)],input=request,capture_output=True,env={'PATH':''},timeout=30)
        (out/(name+'.actual.bin')).write_bytes(result.stdout)
        assert result.returncode==status and not result.stderr,(name,result.returncode,result.stderr)
        if wanted is None:assert not result.stdout,(name,len(result.stdout))
        else:assert result.stdout==wanted,(name,len(result.stdout),len(wanted))
        row={'case':name,'kind':'actual-native-packer-wrapper','pass':True,'exit':status,
             'input_bytes':len(request),'output_bytes':len(result.stdout),
             'input_sha256':sha(request),'output_sha256':sha(result.stdout)}
        rows.append(row);cases[name]=(path,wanted)
    for size in (1,M+1,MAXIMUM-1,MAXIMUM):
        # An independently chosen opaque payload; it is packed, never executed.
        payload=(bytes(range(256))*((size+255)//256))[:size]
        cli('image-'+str(size),pack_frame(payload),0,expected(payload))
    cli('wrapper-one-byte-over-image-limit',pack_frame(bytes(MAXIMUM+1)),3)
    cli('empty-input',b'',2)
    cli('short-header',b'HIAFLT01'+bytes(7),2)
    cli('zero-image',pack_frame(b''),2)
    cli('wrong-magic',b'BADPACK!'+struct.pack('<Q',1)+b'\xc3',2)
    cli('truncated-body',b'HIAFLT01'+struct.pack('<Q',9)+b'\xc3',2)
    cli('overflow-declared-length',b'HIAFLT01'+struct.pack('<Q',(1<<64)-1)+b'\xc3',2)
    cli('extra-body',b'HIAFLT01'+struct.pack('<Q',1)+b'\xc3\x90',2)

    observer=Path(__file__).resolve().with_name('observe_buffers.py')
    def guarded(name,request_name,capacity,fill,returned,wanted=None):
        request=cases[request_name][0]
        spec={'image':str(image),'image_sha256':sha(image.read_bytes()),
              'request':str(request),'request_sha256':sha(request.read_bytes()),
              'capacity':capacity,'workspace':4096,'workspace_fill':fill}
        spec_path=out/(name+'.json');spec_path.write_text(json.dumps(spec,indent=2)+'\n')
        result=subprocess.run([sys.executable,'-I','-S','-B',str(observer),str(spec_path)],
                              capture_output=True,text=True,timeout=30)
        assert result.returncode==0,(name,result.returncode,result.stderr)
        value=json.loads(result.stdout)
        assert value['result']==returned,(name,value,returned)
        assert all(value[k] for k in ['source_unchanged','source_guards_unchanged',
            'output_guards_unchanged','workspace_guards_unchanged','workspace_unchanged'])
        assert not value['forbidden_modules']
        if wanted is None:assert value['output_unchanged']
        else:assert value['produced_sha256']==sha(wanted)
        rows.append({'case':name,'kind':'guarded-native-flat-kernel','pass':True,
                     'workspace_fill':fill,'output_capacity':capacity,**value})
    maximum_name='image-'+str(MAXIMUM);maximum=cases[maximum_name][1]
    for fill in (0,90):
        guarded('maximum-image-guarded-'+str(fill),maximum_name,len(maximum),fill,len(maximum),maximum)
    guarded('maximum-image-short-output',maximum_name,len(maximum)-1,90,0xfffffffe)
    for name in ('empty-input','short-header','zero-image','wrong-magic','truncated-body',
                 'overflow-declared-length','extra-body'):
        guarded(name+'-atomic',name,4096,90,0xffffffff)
    report={'schema':'heptatorch.wide-session-packer-review.v1','pass':True,'cases':len(rows),
            'rows':rows,'executable_sha256':sha(exe.read_bytes()),'image_sha256':sha(image.read_bytes()),
            'maximum_image_bytes_actually_packed':MAXIMUM,
            'maximum_input_bytes_actually_admitted':MAXIMUM+16,
            'maximum_output_bytes_actually_produced':MAXIMUM+len(prefix),
            'generated_authoring_receipt_sha256':sha((generated/'authoring.json').read_bytes()),
            'native_wrapper_and_kernel_both_exercised':True,
            'expected_output':'independent payload bytes, ELF fields and source-defined prefix/call placement',
            'runtime_path':'empty','host_numerical_runtime':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
