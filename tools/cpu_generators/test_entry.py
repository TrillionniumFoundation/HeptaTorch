#!/usr/bin/env python3
"""Native ABI entry admission and actual owned-step boundary execution."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from client import Session,packet,MASK

SHA=lambda b:hashlib.sha256(b).hexdigest()
COMPILER='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
RUNTIME_LINKER='0c5176e3aa968631ae592b887c5dc3cf1e8e3f48f73593f076e9d7965ffacfd7'
OLD_SESSION='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'

def request(magic,objects,name=b'step'):
    return magic+struct.pack('<3Q',len(objects),len(name),0)+name+b''.join(struct.pack('<Q',len(x))+x for x in objects)

def stub(params,returns=('buffer','u32'),effects='alloc,read,write,trap',close=True):
    names=['s','o','state','e','extra'];formal=','.join(n+':'+t for n,t in zip(names,params))
    drops=''
    for n,t in zip(names,params):
        if t in ('buffer','chain') and n!='state':drops+='  _ = owner.drop('+n+')\n'
    values=['state' if t=='buffer' else 'zero' for t in returns]
    raw='hia-asm.v1 hepta.hia.program.v8 generator_gate_probe\nexports(step'+(',close' if close else '')+')\n'
    raw+='fn step('+formal+') -> ('+','.join(returns)+') effects('+effects+') entry(b0)\nblock b0('+formal+')\n'
    raw+=drops+'  zero:u32 = const() {"value":0}\n  ret('+','.join(values)+')\nend\n'
    if close:
        raw+='fn close(state:buffer) -> () effects(alloc,trap) entry(b0)\nblock b0(state:buffer)\n  _ = owner.drop(state)\n  ret()\nend\n'
    return raw.encode()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--stack-build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build=a.build.resolve();stack=a.stack_build.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    compiler=stack/'native/stack-module-compiler';runtime=stack/'native/stack-linker-runtime';old=stack/'native/stack-linker-session'
    assert SHA(compiler.read_bytes())==COMPILER and SHA(runtime.read_bytes())==RUNTIME_LINKER and SHA(old.read_bytes())==OLD_SESSION
    identity=json.loads((build/'build.json').read_text());linker=build/'generator-session-linker';image=build/'generator-session'
    assert SHA(linker.read_bytes())==identity['generator_linker_sha256'] and SHA(image.read_bytes())==identity['executable_sha256']
    rows=[]
    def run(tool,data,name,ok=True):
        proc=subprocess.run([str(tool)],input=data,capture_output=True,env={'PATH':''},timeout=30)
        (out/(name+'.input')).write_bytes(data);(out/(name+'.stdout')).write_bytes(proc.stdout);(out/(name+'.stderr')).write_bytes(proc.stderr)
        if ok:assert proc.returncode==0 and not proc.stderr,(name,proc.returncode,proc.stderr)
        else:assert proc.returncode==2 and not proc.stdout and not proc.stderr,(name,proc.returncode,proc.stdout[:32],proc.stderr)
        return proc.stdout
    objects=[(build/(n+'.o')).read_bytes() for n in ('rng','model','owners')]
    source=Path(__file__).with_name('probe.hia').read_bytes();obj=run(compiler,source,'probe-compile')
    (out/'probe.o').write_bytes(obj)
    raw=run(runtime,request(b'HIASTKR1',objects+[obj],b'probe'),'probe-link')
    probe=out/'step-probe';probe.write_bytes(raw);probe.chmod(0o700)
    with Session(image) as s:
        s.words('create',count=1,a=19);pool=s.dump()
    def invoke(name,st,body,cap=64,entropy=b'',expected_status=0,unchanged=True,expected_return=None):
        data=struct.pack('<4Q',len(st),len(body),cap,len(entropy))+st+body+entropy
        response=run(probe,data,name)
        size,after_bytes,actualcap=struct.unpack_from('<3Q',response)
        assert actualcap==cap and len(response)==24+cap+after_bytes
        output=response[24:24+cap];after=response[24+cap:]
        if expected_return is not None:assert size==expected_return,(name,size,expected_return)
        if cap<64:assert size==0 and output==bytes(cap)
        else:
            assert size>=64 and output[:8]==b'HTCGRS01' and struct.unpack_from('<Q',output,8)[0]==expected_status,(name,size,output[:64])
            assert output[size:]==bytes(cap-size),(name,'trailing output changed')
        if unchanged:assert st==after,(name,SHA(st),SHA(after))
        rows.append({'case':name,'status':'PASS','original_owner_bytes':len(st),'returned_owner_bytes':len(after),
            'output_capacity':cap,'returned_bytes':size,'expected_status':None if cap<64 else expected_status,
            'retained_state_exact':unchanged,'request_sha256':SHA(data),'response_sha256':SHA(response)})
        return output,after
    for cap in (0,1,7,23,24,31,32,63):invoke('short-output-'+str(cap),pool,packet('create',count=1),cap,expected_return=0)
    for cap in (64,71):invoke('insufficient-result-'+str(cap),pool,packet('create',count=1),cap,expected_status=3)
    invoke('exact-create-output',pool,packet('create',count=1,a=20),72,unchanged=False,expected_return=72)
    invoke('exact-get-state-output',pool,packet('get_state',1),5120,expected_return=5120)
    for n in (0,1,7,8,16,63):invoke('short-input-'+str(n),pool,packet('stats')[:n],64,expected_status=1)
    for n in (1,7,8,15,16,23,24,63):invoke('short-internal-owner-'+str(n),bytes(n),packet('stats'),64,expected_status=10)
    for at,value in ((0,0),(16,1025),(32,2),(40,5087),(48,1),(56,1)):
        st=bytearray(pool);struct.pack_into('<Q',st,at,value)
        invoke('invalid-internal-header-'+str(at),st,packet('stats'),64,expected_status=10)
    epoch=bytearray(pool);struct.pack_into('<Q',epoch,24,MASK)
    invoke('epoch-exhausted-draw',epoch,packet('raw32',1,1),72,expected_status=8)
    invoke('epoch-exhausted-create',epoch,packet('create',count=1),72,expected_status=8)
    invoke('epoch-exhausted-zero-draw',epoch,packet('raw32',1,0),64)
    invoke('epoch-exhausted-read',epoch,packet('get_state',1),5120)
    handles=bytearray(pool);struct.pack_into('<Q',handles,8,MASK)
    invoke('handle-exhausted-create',handles,packet('create',count=1),72,expected_status=8)
    invoke('handle-exhausted-clone',handles,packet('clone_state',1,1),72,expected_status=8)
    output,after=invoke('handle-exhausted-release',handles,packet('release',1),64,unchanged=False)
    assert len(after)==64 and struct.unpack_from('<Q',after,8)[0]==MASK
    one=bytearray(pool);struct.pack_into('<Q',one,8,MASK-1)
    output,after=invoke('last-handle-valid',one,packet('clone_state',1,1),72,unchanged=False)
    assert struct.unpack_from('<Q',output,64)[0]==MASK
    invoke('last-handle-reject-next',after,packet('create',count=1),72,expected_status=8)
    for size in (0,1,7,9):invoke('entropy-size-'+str(size),pool,packet('seed',1),72,entropy=bytes(size),expected_status=9)
    entropy=struct.pack('<Q',0xfedcba9876543210)
    output,after=invoke('trusted-entropy-full-u64',pool,packet('seed',1),72,entropy=entropy,unchanged=False)
    assert output[64:72]==entropy and after[96:104]==entropy
    malformed=bytearray(pool[96:]);struct.pack_into('<I',malformed,5040,2)
    invoke('invalid-cache-full-owner-rollback',pool,packet('set_state',1,payload=malformed),64,expected_status=2)
    # The two entry profiles refuse each other's parameter contracts.
    run(old,request(b'HIASTKS1',objects),'old-profile-rejects-four-parameters',False)
    rows.append({'case':'old-profile-rejects-four-parameters','status':'PASS','exit':2})
    variants=[('three-parameters',('readbuf','mutbuf','buffer'),('buffer','u32'),'alloc,read,write,trap',True),
              ('entropy-mutbuf',('readbuf','mutbuf','buffer','mutbuf'),('buffer','u32'),'alloc,read,write,trap',True),
              ('entropy-scalar',('readbuf','mutbuf','buffer','u64'),('buffer','u32'),'alloc,read,write,trap',True),
              ('entropy-owner',('readbuf','mutbuf','buffer','buffer'),('buffer','u32'),'alloc,read,write,trap',True),
              ('five-parameters',('readbuf','mutbuf','buffer','readbuf','u64'),('buffer','u32'),'alloc,read,write,trap',True),
              ('wrong-output-type',('readbuf','readbuf','buffer','readbuf'),('buffer','u32'),'alloc,read,write,trap',True),
              ('reordered-results',('readbuf','mutbuf','buffer','readbuf'),('u32','buffer'),'alloc,read,write,trap',True),
              ('missing-alloc-effect',('readbuf','mutbuf','buffer','readbuf'),('buffer','u32'),'read,write,trap',True),
              ('missing-close',('readbuf','mutbuf','buffer','readbuf'),('buffer','u32'),'alloc,read,write,trap',False)]
    for name,params,results,effects,close in variants:
        src=stub(params,results,effects,close);compiled=run(compiler,src,'gate-'+name+'-compile')
        run(linker,request(b'HIACGS01',[compiled]),'gate-'+name,False)
        rows.append({'case':'gate-'+name,'status':'PASS','exit':2,'source_sha256':SHA(src)})
    valid=run(compiler,stub(('readbuf','mutbuf','buffer','readbuf')),'gate-positive-compile')
    run(linker,request(b'HIACGS01',[valid]),'gate-positive')
    rows.append({'case':'gate-positive','status':'PASS','exit':0})
    reordered=run(linker,request(b'HIACGS01',objects[::-1]),'reordered-service')
    executable=out/'reordered-service';executable.write_bytes(reordered);executable.chmod(0o700)
    with Session(executable) as s:
        assert s.words('create',count=1,a=5489)==[1]
        assert s.words('raw32',1,3)==[3499211612,581869302,3890346734]
    rows.append({'case':'reordered-linked-real-execution','status':'PASS','image_sha256':SHA(reordered)})
    report={'schema':'heptatorch.cpu-generator-native-entry-qualification.v1','status':'PASS','cases':rows,
        'case_count':len(rows),'production_image_sha256':SHA(image.read_bytes()),'linker_sha256':SHA(linker.read_bytes()),
        'compiler_sha256':COMPILER,'probe_source_sha256':SHA(source),'probe_image_sha256':SHA(raw),
        'test_source_sha256':SHA(Path(__file__).read_bytes()),'caller_created_owners_in_qualification_only':True,
        'production_accepts_native_pointers_from_packets':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))

if __name__=='__main__':main()
