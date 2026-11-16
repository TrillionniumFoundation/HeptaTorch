"""Independent malformed-object and public buffer ABI review of native modules."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

CALLER = b'''hia-asm.v1 hepta.hia.program.v8 review_caller
exports(entry)
fn apply(s:readbuf,w:mutbuf) -> (u32) effects(read,write,trap) external()
fn entry(s:readbuf,o:mutbuf,w:mutbuf) -> (u32) effects(read,write,trap) entry(b0)
block b0(s:readbuf,o:mutbuf,w:mutbuf)
  n:u32 = call(s,w) {"function":"apply"}
  z:usize = const() {"value":0}
  one:usize = const() {"value":1}
  v:u8 = buffer.load(w,z)
  a:u8 = buffer.load(s,z)
  _ = buffer.store(o,z,v)
  _ = buffer.store(o,one,a)
  size:u32 = const() {"value":2}
  ret(size)
end
'''
PROVIDER = b'''hia-asm.v1 hepta.hia.program.v8 review_provider
exports(apply)
fn apply(s:readbuf,w:mutbuf) -> (u32) effects(read,write,trap) entry(b0)
block b0(s:readbuf,w:mutbuf)
  z:usize = const() {"value":0}
  a:u8 = buffer.load(s,z)
  k:u8 = const() {"value":17}
  b:u8 = int.add.wrap(a,k)
  _ = buffer.store(w,z,b)
  size:u32 = const() {"value":1}
  ret(size)
end
'''


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def frame(objects, entry=b'entry'):
    return b'HIABLK01'+struct.pack('<3Q',len(objects),len(entry),0)+entry+b''.join(struct.pack('<Q',len(o))+o for o in objects)


def u64(raw, offset):
    return struct.unpack_from('<Q',raw,offset)[0]


def patched(raw, offset, value, width=8):
    out=bytearray(raw)
    out[offset:offset+width]=int(value).to_bytes(width,'little')
    return bytes(out)


def layout(raw):
    table=u64(raw,40)
    fields={i:(u64(raw,table+i*64+24),u64(raw,table+i*64+32)) for i in range(1,8)}
    return table,fields


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--linker',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    identities={'compiler_sha256':digest(args.compiler.read_bytes()),'linker_sha256':digest(args.linker.read_bytes())}
    rows=[]
    def run(exe,request):
        return subprocess.run([str(exe.resolve())],input=request,capture_output=True,env={'PATH':''},timeout=10)
    objects=[]
    for name,source in [('caller',CALLER),('provider',PROVIDER)]:
        result=run(args.compiler,source)
        assert result.returncode==0 and not result.stderr and result.stdout.startswith(b'\x7fELF')
        objects.append(result.stdout)
        (args.output/(name+'.hepta')).write_bytes(source)
        (args.output/(name+'.o')).write_bytes(result.stdout)
    for reverse in (False,True):
        request=frame(objects[::-1] if reverse else objects)
        result=run(args.linker,request)
        assert result.returncode==0 and not result.stderr
        executable=args.output/('positive-'+str(int(reverse)))
        executable.write_bytes(result.stdout);executable.chmod(0o700)
        for value in (0,1,127,239,255):
            actual=run(executable,bytes([value]))
            assert actual.returncode==0 and not actual.stderr and actual.stdout==bytes([(value+17)&255,value])
            rows.append({'case':'actual-borrowed-call-'+str(int(reverse))+'-'+str(value),
                         'kind':'positive-native-execution','status':'PASS','request_sha256':digest(request),
                         'executable_sha256':digest(result.stdout)})
    def refuse(name,request):
        result=run(args.linker,request)
        row={'case':name,'kind':'malformed-object-refusal','request_sha256':digest(request),
             'exit':result.returncode,'stdout_bytes':len(result.stdout),'stderr_bytes':len(result.stderr),
             'status':'PASS' if result.returncode==2 and not result.stdout and not result.stderr else 'FAIL'}
        rows.append(row)
        if row['status']!='PASS':
            (args.output/(name+'.request')).write_bytes(request)
            (args.output/'first-failure.json').write_text(json.dumps(row,indent=2)+'\n')
            raise AssertionError(row)
    # Canonical object structure and unsigned offset-overflow adversaries.
    for number,original in enumerate(objects):
        table,sections=layout(original)
        def corrupt(name,offset,value,width=8):
            changed=objects.copy();changed[number]=patched(original,offset,value,width)
            refuse('object'+str(number)+'-'+name,frame(changed))
        for value in (0,1,(1<<63),(1<<64)-1):
            corrupt('section-table-'+str(value),40,value)
        for section in range(1,8):
            for field in (24,32):
                for value in ((1<<63),(1<<64)-1):
                    corrupt('section'+str(section)+'-field'+str(field)+'-'+str(value),table+section*64+field,value)
        meta=sections[7][0]
        for field,values in [(0,[0]),(8,[0,2]),(16,[0,257,(1<<64)-1]),(24,[0,257]),(32,[0,257]),(40,[257]),(48,[128,257]),(56,[1,(1<<64)-1])]:
            for value in values:
                corrupt('interface-header-'+str(field)+'-'+str(value),meta+field,value)
        for record in range(u64(original,meta+16)):
            at=meta+64+record*256
            for field,values in [(0,[0,(1<<64)-1]),(8,[2]),(16,[17,(1<<64)-1]),(24,[2]),(32,[1,16,(1<<64)-1]),(40,[0,6,8]),(48,[0,8,9,16])]:
                for value in values:
                    corrupt('record'+str(record)+'-field'+str(field)+'-'+str(value),at+field,value)
            for field in range(176,256,8):
                corrupt('record'+str(record)+'-reserved'+str(field),at+field,1)
            parameters=u64(original,at+16)
            assert parameters<16
            corrupt('record'+str(record)+'-unused-parameter',at+48+parameters*8,1)
        # A syntactically legal but different complete signature must fail at link.
        if number==1:
            corrupt('provider-effects-disagree',meta+64+32,12)
            corrupt('provider-read-mut-disagree',meta+64+48,7)
            corrupt('provider-result-width-disagree',meta+64+40,4)
        relocs,size=sections[6]
        if size:
            for field,value,width in [(0,0,8),(0,(1<<64)-1,8),(8,2,4),(12,0,4),(16,0,8)]:
                corrupt('relocation-'+str(field)+'-'+str(value),relocs+field,value,width)
            offset=u64(original,relocs)
            corrupt('relocation-not-a-call',64+offset-1,0x90,1)
            corrupt('relocation-not-zero-placeholder',64+offset,1,4)
    valid=frame(objects)
    for cut in (0,7,31,32,36,len(valid)-1):
        refuse('truncated-frame-'+str(cut),valid[:cut])
    refuse('duplicate-provider',frame([objects[0],objects[1],objects[1]]))
    refuse('no-provider',frame([objects[0]]))
    refuse('wrong-entry',frame(objects,b'apply'))
    report={'schema':'heptatorch.modules-independent-contract-review.v1','status':'PASS',**identities,
            'reviewer_sha256':digest(Path(__file__).read_bytes()),'cases':len(rows),
            'positive_execution_cases':sum(r['kind']=='positive-native-execution' for r in rows),
            'malformed_object_refusals':sum(r['kind']=='malformed-object-refusal' for r in rows),
            'object_sha256':[digest(o) for o in objects],
            'limitation':'Object code authenticity comes from pinned compiler and build custody; this is not a machine-code verifier.',
            'rows':rows}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':
    main()
