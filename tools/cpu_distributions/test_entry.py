#!/usr/bin/env python3
"""Real owned ABI/output guards, normal OOM rollback and numerical work limits."""
import argparse,hashlib,json,os,re,resource,struct,subprocess,time
from pathlib import Path
from protocol import Session,packet,parse,full,bits,meta

SHA=lambda b:hashlib.sha256(b).hexdigest()
COMPILER_SHA='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER_SHA='0c5176e3aa968631ae592b887c5dc3cf1e8e3f48f73593f076e9d7965ffacfd7'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--stack-build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    build=a.build.resolve();receipt=json.loads((build/'build.json').read_text());image=build/'tensor-distributions-session'
    assert SHA(image.read_bytes())==receipt['executable_sha256']
    compiler=a.stack_build.resolve()/'native/stack-module-compiler';linker=a.stack_build.resolve()/'native/stack-linker-runtime'
    assert SHA(compiler.read_bytes())==COMPILER_SHA and SHA(linker.read_bytes())==LINKER_SHA
    source=Path(__file__).with_name('entry_probe.hepta').read_bytes()
    c=subprocess.run([str(compiler)],input=source,capture_output=True,env={'PATH':''},timeout=60)
    (out/'probe.o').write_bytes(c.stdout);(out/'probe.compile.stderr').write_bytes(c.stderr);assert c.returncode==0 and not c.stderr
    objects=[]
    for row in receipt['units']:
        raw=(build/(row['name']+'.o')).read_bytes();assert SHA(raw)==row['object_sha256'];objects.append(raw)
    objects.append(c.stdout)
    request=b'HIASTKR1'+struct.pack('<3Q',len(objects),5,0)+b'probe'+b''.join(struct.pack('<Q',len(x))+x for x in objects)
    (out/'probe.link.request').write_bytes(request)
    l=subprocess.run([str(linker)],input=request,capture_output=True,env={'PATH':''},timeout=90)
    (out/'probe.link.stdout').write_bytes(l.stdout);(out/'probe.link.stderr').write_bytes(l.stderr);assert l.returncode==0 and not l.stderr
    probe=out/'entry-probe';probe.write_bytes(l.stdout);probe.chmod(0o500)
    s=Session(image,directory=out/'live')
    r=s.run([full((8,),0),(60,99)]);t,g=[x['value'] for x in r];snapshot=s.owner()['bytes'];checks=[];counter=0
    def run(st,body,cap):
        nonlocal counter
        raw=struct.pack('<3Q',len(st),len(body),cap)+st+body
        result=subprocess.run([str(probe)],input=raw,capture_output=True,env={'PATH':''},timeout=30)
        name='probe-'+str(counter);counter+=1
        (out/(name+'.input')).write_bytes(raw);(out/(name+'.stdout')).write_bytes(result.stdout);(out/(name+'.stderr')).write_bytes(result.stderr)
        assert result.returncode==0 and not result.stderr,(name,result.returncode,result.stderr)
        n,size,capacity=struct.unpack_from('<3Q',result.stdout);assert capacity==cap and len(result.stdout)==24+cap+size
        return n,result.stdout[24:24+cap],result.stdout[24+cap:]
    for cap in range(32):
        for state in (b'',snapshot):
            n,response,after=run(state,packet([(80,t,g,4602678819172646912,1)]),cap)
            assert n==0 and response==bytes(cap) and after==state
    checks.append({'name':'all-short-output-capacities','cases':64,'unchanged_output_and_owner':True})
    for length in (1,7,8,31,63,139263):
        state=b'Z'*length;n,response,after=run(state,packet([]),32)
        assert n==32 and parse(response)[0]==7 and after==state
    checks.append({'name':'short-state-owners','cases':6})
    for length in (0,1,7,8,16,23,31):
        body=packet([])[:length];n,response,after=run(snapshot,body,32)
        assert n==32 and parse(response)[0]==100 and after==snapshot
    checks.append({'name':'short-input','cases':7})
    for cap,commands in [(32,[(80,t,g,4602678819172646912,1)]),(63,[(80,t,g,4602678819172646912,1)]),
                         (64,[(80,t,g,4602678819172646912,1),(63,g)]),(5119,[(63,g)])]:
        n,response,after=run(snapshot,packet(commands),cap)
        assert n==32 and parse(response[:n])[0]==3 and after==snapshot
    checks.append({'name':'output-space-after-trial-work','cases':4})
    n,response,after=run(snapshot,packet([(63,g)]),5120)
    assert n==5120 and parse(response)[0]==0 and after==snapshot
    n,response,after=run(snapshot,packet([(80,t,g,4602678819172646912,1)]),64)
    assert n==64 and parse(response)[0]==0 and after!=snapshot
    checks.append({'name':'exact-output-capacity-edges','cases':2})
    # Mutate a genuine producer byte-state only inside this private native test
    # harness. It allocates a new checked owner; no process pointer is imported.
    modified=bytearray(snapshot);slots=struct.unpack_from('<Q',modified,139264+16)[0]
    info=meta(s.run([(13,t)])[0]);version=info['version_id']
    version_at=139264+128+2*slots*64+((version&0xffffff)-1)*64+16
    struct.pack_into('<Q',modified,version_at,(1<<64)-1)
    n,response,after=run(modified,packet([(80,t,g,4602678819172646912,1)]),64)
    assert n==32 and parse(response[:n])[0]==8 and after==modified
    checks.append({'name':'version-overflow-admission','cases':1,'qualified_test_only_state_fault':True})
    old=s.owner();status_text=Path(f'/proc/{s.pid}/status').read_text()
    size=int(re.search(r'^VmSize:\s+(\d+) kB$',status_text,re.M)[1])*1024
    limit=resource.prlimit(s.pid,resource.RLIMIT_AS)
    resource.prlimit(s.pid,resource.RLIMIT_AS,(size+8192,limit[1]))
    try:status,records=parse(s.exchange(packet([(80,t,g,4602678819172646912,1)])));assert status==3 and not records
    finally:resource.prlimit(s.pid,resource.RLIMIT_AS,limit)
    assert s.owner()==old
    s.run([(80,t,g,4602678819172646912,1),(53,)])
    checks.append({'name':'real-kernel-allocation-refusal','cases':1,'vm_size':size,'extra_address_space':8192,'entire_owner_unchanged':True})
    # An already executed fill is discarded when a later response exceeds 1MiB.
    s.error([(80,t,g,4602678819172646912,1)]+[(63,g)]*207,3)
    checks.append({'name':'real-session-output-overflow','cases':1,'entire_owner_unchanged':True})
    s.run([(53,)]);s.finish()
    report={'schema':'heptatorch.cpu-distributions-owned-entry-guards.v1','status':'PASS','checks':checks,
        'cases':sum(x['cases'] for x in checks),'probe_processes':counter,'live_frames':s.frames,
        'executable_sha256':receipt['executable_sha256'],'probe_source_sha256':SHA(source),'probe_sha256':SHA(l.stdout),
        'compiler_sha256':COMPILER_SHA,'runtime_linker_sha256':LINKER_SHA,'native_production_uses_host_reference':False}
    assert SHA(image.read_bytes())==receipt['executable_sha256'];assert SHA(compiler.read_bytes())==COMPILER_SHA and SHA(linker.read_bytes())==LINKER_SHA
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
