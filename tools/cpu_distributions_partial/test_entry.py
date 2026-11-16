#!/usr/bin/env python3
"""Execute new partial commands through the real owned ABI with bounded output."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import resource
import struct
import subprocess

SHA=lambda raw:hashlib.sha256(raw).hexdigest()
COMPILER='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER='0c5176e3aa968631ae592b887c5dc3cf1e8e3f48f73593f076e9d7965ffacfd7'
PROTOCOL='7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
PROBE='54f81a65652e2c2ba9eb3f04d608e89b71c92c5ad33ff7061e9a03b89f45f9ea'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build','atomic-tools','stack-build','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();build=a.build.resolve();atomic=a.atomic_tools.resolve();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    compiler=a.stack_build.resolve()/'native/stack-module-compiler'
    linker=a.stack_build.resolve()/'native/stack-linker-runtime'
    image=build/'tensor-distributions-session';source=atomic/'entry_probe.hepta'
    inputs={path:path.read_bytes() for path in (compiler,linker,image,source,build/'build.json',
             atomic/'protocol.py',atomic/'owner_client.py',Path(__file__).resolve())}
    receipt=json.loads(inputs[build/'build.json'])
    assert receipt['status']=='PASS' and SHA(inputs[image])==receipt['executable_sha256']
    for path,digest in [(compiler,COMPILER),(linker,LINKER),(source,PROBE),(atomic/'protocol.py',PROTOCOL)]:
        assert SHA(inputs[path])==digest
    spec=importlib.util.spec_from_file_location('partial_entry_wire',atomic/'protocol.py')
    wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
    compiled=subprocess.run([str(compiler)],input=inputs[source],capture_output=True,env={'PATH':''},timeout=60)
    (out/'probe.o').write_bytes(compiled.stdout);(out/'probe.compile.stderr').write_bytes(compiled.stderr)
    assert compiled.returncode==0 and not compiled.stderr
    objects=[]
    for row in receipt['units']:
        path=build/(row['name']+'.o');raw=path.read_bytes();inputs[path]=raw
        assert SHA(raw)==row['object_sha256'];objects.append(raw)
    objects.append(compiled.stdout)
    request=b'HIASTKR1'+struct.pack('<3Q',len(objects),5,0)+b'probe'+b''.join(struct.pack('<Q',len(x))+x for x in objects)
    linked=subprocess.run([str(linker)],input=request,capture_output=True,env={'PATH':''},timeout=90)
    (out/'probe.link.request').write_bytes(request);(out/'probe.link.stderr').write_bytes(linked.stderr)
    assert linked.returncode==0 and not linked.stderr
    probe=out/'entry-probe';probe.write_bytes(linked.stdout);probe.chmod(0o500)
    report={'schema':'heptatorch.cpu-distributions-partial-owned-entry.v1','status':'RUNNING',
            'runtime_source_sha256':receipt['runtime_source_sha256'],'executable_sha256':receipt['executable_sha256'],
            'probe_source_sha256':PROBE,'probe_sha256':SHA(linked.stdout),'checks':[],
            'raw_processes':0,'live_frames':0,'complete_owner_bytes_compared':0,
            'qualification_only_owned_byte_faults':True,'foreign_owner_pointer_imported':False}

    def run(state,commands,capacity):
        body=wire.packet(commands);raw=struct.pack('<3Q',len(state),len(body),capacity)+state+body
        result=subprocess.run([str(probe)],input=raw,capture_output=True,env={'PATH':''},timeout=30)
        name='probe-'+str(report['raw_processes']);report['raw_processes']+=1
        (out/(name+'.input')).write_bytes(raw);(out/(name+'.stdout')).write_bytes(result.stdout)
        (out/(name+'.stderr')).write_bytes(result.stderr)
        assert result.returncode==0 and not result.stderr,(name,result.returncode)
        n,size,cap=struct.unpack_from('<3Q',result.stdout)
        assert cap==capacity and len(result.stdout)==24+cap+size
        return n,result.stdout[24:24+cap],result.stdout[24+cap:]

    for code,error in [(86,0),(86,7),(87,0),(87,5)]:
        name=f'op{code}-error{error}';session=wire.Session(image,directory=out/name)
        x,p,g=[r['value'] for r in session.run([wire.full((8,),1,int(error==5)),wire.full((8,),.5),(60,99)])]
        if error==7:session.run([(12,p,2,0x7fc00000)])
        command=(86,x,g,p,1) if code==86 else (87,x,g,0x3fe0000000000000,1,int(error!=5))
        before=session.owner()['bytes']
        status,records=wire.parse(session.exchange(wire.packet([command])))
        assert status==0 and len(records)==1 and records[0]['value']==error
        expected=session.owner()['bytes'];assert expected!=before
        for cap in (0,31,32,63,64,95):
            n,response,after=run(before,[command],cap)
            assert after==before
            report['complete_owner_bytes_compared']+=len(before)
            if cap<32:assert n==0 and response==bytes(cap)
            else:assert n==32 and wire.parse(response[:n])==(3,[])
        n,response,after=run(before,[command],96)
        assert n==96 and wire.parse(response)==(status,records) and after==expected
        report['complete_owner_bytes_compared']+=len(expected)
        if error:
            n,response,after=run(before,[command]+[(63,g)]*207,96)
            assert n==96 and wire.parse(response)==(status,records) and after==expected
            report['complete_owner_bytes_compared']+=len(expected)
        else:
            n,response,after=run(before,[command,(999,)],128)
            assert n==32 and wire.parse(response[:n])==(4,[]) and after==before
            report['complete_owner_bytes_compared']+=len(before)
            modified=bytearray(before)
            slots=struct.unpack_from('<Q',modified,139264+16)[0]
            version=wire.meta(session.run([(13,x)])[0])['version_id']
            at=139264+128+2*slots*64+((version&0xffffff)-1)*64+16
            struct.pack_into('<Q',modified,at,(1<<64)-1)
            n,response,after=run(modified,[command],96)
            assert n==32 and wire.parse(response[:n])==(8,[]) and after==modified
            report['complete_owner_bytes_compared']+=len(modified)
        session.run([(53,)]);session.finish();report['live_frames']+=session.frames
        report['checks'].append({'name':name,'output_capacity_refusals':6,'exact96bytes':True,
                                 'returned_owner_matches_interactive_execution':True,
                                 'prefix_halt_at_exact_capacity':bool(error),
                                 'late_ordinary_error_rolls_back':not error,'version_overflow_checked':not error})
    session=wire.Session(image,directory=out/'kernel-oom')
    x,p,g=[r['value'] for r in session.run([wire.full((8,),1),wire.full((8,),.5),(60,7)])]
    session.run([(12,p,2,0x7fc00000)])
    before=session.owner()
    vm=int(re.search(r'^VmSize:\s+(\d+) kB$',Path(f'/proc/{session.pid}/status').read_text(),re.M)[1])*1024
    old=resource.prlimit(session.pid,resource.RLIMIT_AS)
    resource.prlimit(session.pid,resource.RLIMIT_AS,(vm+8192,old[1]))
    try:status,records=wire.parse(session.exchange(wire.packet([(86,x,g,p,1)])))
    finally:resource.prlimit(session.pid,resource.RLIMIT_AS,old)
    assert status==3 and not records and session.owner()==before
    report['complete_owner_bytes_compared']+=before['length']
    status,records=wire.parse(session.exchange(wire.packet([(86,x,g,p,1)])))
    assert status==0 and struct.unpack('<4Q',records[0]['payload'])==(7,x,2,2)
    session.run([(53,)]);session.finish();report['live_frames']+=session.frames
    report['checks'].append({'name':'real-kernel-owner-allocation-refusal','owner_identity_and_bytes_unchanged':True,
                             'later_partial_error_remains_usable':True,'vm_size':vm,'margin':8192})
    for path,raw in inputs.items():assert path.read_bytes()==raw,path
    report.update(status='PASS',input_custody_after=True,
                  inputs={str(path):SHA(raw) for path,raw in inputs.items()})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('status','raw_processes','live_frames',
                                                'complete_owner_bytes_compared')},indent=2))


if __name__=='__main__':main()
