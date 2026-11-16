#!/usr/bin/env python3
"""Persistent native Generator state/model, guard, rollback and reclamation tests."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import shutil
import struct
import subprocess
import sys
import traceback

from client import Session,packet,MASK
from reference import Reference,untemper

sha=lambda b:hashlib.sha256(b).hexdigest()

def entropy_denial():
    # Real kernel seccomp returns -EAGAIN only for getrandom, not an RNG shim.
    class Filter(ctypes.Structure):_fields_=[('code',ctypes.c_ushort),('jt',ctypes.c_ubyte),('jf',ctypes.c_ubyte),('k',ctypes.c_uint)]
    class Program(ctypes.Structure):_fields_=[('len',ctypes.c_ushort),('filter',ctypes.POINTER(Filter))]
    code=(Filter*4)(Filter(0x20,0,0,0),Filter(0x15,0,1,318),Filter(0x06,0,0,0x50000|11),Filter(0x06,0,0,0x7fff0000))
    libc=ctypes.CDLL(None,use_errno=True);program=Program(4,code)
    if libc.prctl(38,1,0,0,0)!=0 or libc.prctl(22,2,ctypes.byref(program),0,0)!=0:os._exit(120)

def ranges(pid):
    rows=[]
    for line in Path(f'/proc/{pid}/maps').read_text().splitlines():
        f=line.split();lo,hi=(int(x,16) for x in f[0].split('-'));rows.append((lo,hi,f[1],f[5:] if len(f)>5 else []))
    return rows

def owner_observation(session):
    pid=session.process.pid;found=[]
    with open(f'/proc/{pid}/mem','rb',buffering=0) as mem:
        for lo,hi,perms,name in ranges(pid):
            if perms!='rw-p' or name or hi-lo>16<<20:continue
            mem.seek(lo);raw=mem.read(hi-lo);at=raw.find(b'HTOWNR01')
            while at>=0:
                if at+48<=len(raw) and raw[at+16:at+24]==b'HTOWNS01':
                    pointer,size=struct.unpack_from('<2Q',raw,at+32)
                    if pointer and size>=64:
                        mem.seek(pointer)
                        if mem.read(8)==b'HTCGST01':found.append((lo+at,pointer,size))
                at=raw.find(b'HTOWNR01',at+1)
    assert len(found)==1,found
    control,pointer,size=found[0]
    return dict(control=control,pointer=pointer,bytes=size)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    build=a.build.resolve();image=build/'generator-session';identity=json.loads((build/'build.json').read_text())
    assert sha(image.read_bytes())==identity['executable_sha256']
    cpp=out/'std-mt-reference';compiler=shutil.which('g++');assert compiler
    compiled=subprocess.run([compiler,'-std=c++17','-O2',str(Path(__file__).with_name('reference.cpp')),'-o',str(cpp)],capture_output=True)
    (out/'reference-build.stdout').write_bytes(compiled.stdout);(out/'reference-build.stderr').write_bytes(compiled.stderr)
    assert compiled.returncode==0 and not compiled.stderr
    rows=[];frames=0;values=0;state_checks=0;rollbacks=0
    def record(name,detail):rows.append(dict(name=name,status='PASS',**detail))
    def exact_state(s,h,m):
        nonlocal state_checks
        actual=s.call('get_state',h)['payload'];assert actual==m.state(),h;state_checks+=1;return actual
    def draw(s,h,m,op,n,a=0,b=0):
        nonlocal values
        result=s.words(op,h,n,a,b);expected=m.draw(op,n,a,b);assert result==expected,(op,h,n,result[:8],expected[:8])
        values+=len(result);exact_state(s,h,m)
    def rollback(s,body,status):
        nonlocal rollbacks
        before=s.dump();r=s.raw(body);after=s.dump();assert r['status']==status and not r['payload'] and before==after,('rollback',r['status'],status,len(r['payload']),sha(before),sha(after))
        rollbacks+=1
    try:
        with Session(image) as s:
            assert s.call('stats')['allocation']==0
            handles=s.words('create',count=4,a=0xfedcba9876543210);assert handles==[1,2,3,4]
            refs={h:Reference(cpp,seed=0xfedcba9876543210) for h in handles}
            for h in handles:exact_state(s,h,refs[h])
            for h in handles:draw(s,h,refs[h],'raw32',625)
            defaults=s.words('create_default',count=1);assert defaults==[5]
            refs[5]=Reference(cpp,seed=67280421310721);exact_state(s,5,refs[5])
            assert s.words('current_seed',5)==[67280421310721]
            record('construction-independent-std-mt-and-default',{'generators':5})
            for op in ('raw32','raw64','rand_f32','rand_f64','rand_f16','rand_bf16','advance'):
                for n in (0,1,623,624,625):
                    before=s.dump() if n==0 else None
                    draw(s,1,refs[1],op,n)
                    if n==0:assert s.dump()==before
            for low,high in [(0,(1<<28)-1),(-99,(1<<28)-99),(-(1<<63),(1<<63)-1),(-7,8),((1<<63)-2,(1<<63)-1)]:
                draw(s,2,refs[2],'rand_int64',257,low,high)
            record('dtype-consumption-thresholds-and-zero-noop',{'dtype_modes':7,'int64_ranges':5})
            raw=bytearray(refs[3].state())
            struct.pack_into('<Q',raw,16,struct.unpack_from('<Q',raw,16)[0]|(0xabad1dea<<32))
            struct.pack_into('<I',raw,12,0xff00ff00)
            for i in range(624):struct.pack_into('<Q',raw,24+8*i,struct.unpack_from('<Q',raw,24+8*i)[0]|(0xa1b2c3d4<<32))
            raw[5016:5024]=b'ABCDEFGH';raw[5032:5040]=b'abcdefgh';raw[5044:5048]=b'pad!';raw[5053:5056]=b'xyz'
            struct.pack_into('<Q',raw,5024,0xfff0000000000123);struct.pack_into('<I',raw,5040,1)
            struct.pack_into('<I',raw,5048,0x7f800123);raw[5052]=1
            s.call('set_state',3,payload=raw);refs[3].restore(raw);exact_state(s,3,refs[3])
            clones=s.words('clone_state',3,3);assert clones==[6,7,8]
            for h in clones:refs[h]=Reference(cpp,state=refs[3].state());exact_state(s,h,refs[h])
            draw(s,3,refs[3],'raw64',700)
            for h in clones:exact_state(s,h,refs[h])
            for h in clones:draw(s,h,refs[h],'raw64',700)
            s.call('manual_seed',3,a=MASK);refs[3].seed(MASK);exact_state(s,3,refs[3]);assert s.words('current_seed',3)==[MASK]
            record('normalized-import-caches-clone-and-reseed',{'clone_handles':clones})
            crafted=bytearray(refs[4].state());struct.pack_into('<IIQ',crafted,8,624,1,1)
            struct.pack_into('<Q',crafted,32,untemper(0x00ffffff))
            for op in ('rand_f16','rand_bf16'):
                s.call('set_state',4,payload=crafted);refs[4].restore(crafted)
                got=s.words(op,4,1);expected=refs[4].draw(op,1);assert got==expected==[0];exact_state(s,4,refs[4])
            record('low-precision-rounded-upper-endpoint-clamps',{'exact_zero_results':2})
            for _ in range(5):
                seed=s.words('seed',4)[0];refs[4].seed(seed);assert s.words('current_seed',4)==[seed]
                exact_state(s,4,refs[4]);draw(s,4,refs[4],'raw32',17)
            record('real-kernel-nondeterministic-seed-state-coherence',{'calls':5,'distinctness_used_as_assertion':False})
            before_count=rollbacks
            bad=[]
            for size in (0,1,7,8,16,63):bad.append((packet('stats')[:size],1))
            for at in (0,56):
                q=bytearray(packet('stats'));q[at]^=1;bad.append((q,1))
            bad += [(packet(0),5),(packet(20),5),(packet('stats')+b'x',1),
                    (packet('raw32',999,1),4),(packet('create',count=0),3),(packet('create',count=1025),3),
                    (packet('create',handle=1,count=1),1),(packet('create',count=1,b=1),1),
                    (packet('raw32',1,100001),3),(packet('advance',1,1000001),3),
                    (packet('rand_int64',1,1,3,3),1),(packet('rand_int64',1,1,4,3),1),
                    (packet('get_state',1,count=1),1),(packet('clone_state',1,count=1,a=1),1),
                    (packet('set_state',1,payload=b'x'*5055),1),(packet('set_state',1,payload=b'x'*5057),1),
                    (packet('dump',count=MASK,a=0),1),(packet('dump',count=1,a=MASK),1)]
            for at,width,value in [(8,4,0),(8,4,625),(12,4,0),(16,8,625),(16,8,0),(5040,4,2),(5052,1,2)]:
                state=bytearray(refs[1].state());state[at:at+width]=value.to_bytes(width,'little');bad.append((packet('set_state',1,payload=state),2))
            for body,status in bad:rollback(s,body,status)
            record('invalid-requests-preserve-entire-owned-pool',{'rollbacks':rollbacks-before_count})
            chooser=random.Random(0x1635b5ad)
            next_id=8
            for i in range(350):
                h=chooser.choice(sorted(refs));action=chooser.randrange(10)
                if action<6:
                    op=chooser.choice(['raw32','raw64','rand_f32','rand_f64','rand_f16','rand_bf16','advance'])
                    draw(s,h,refs[h],op,chooser.randrange(40))
                elif action==6:
                    seed=chooser.getrandbits(64);s.call('manual_seed',h,a=seed);refs[h].seed(seed);exact_state(s,h,refs[h])
                elif action==7 and len(refs)<12:
                    new=s.words('clone_state',h,1)[0];next_id+=1;assert new==next_id
                    refs[new]=Reference(cpp,state=refs[h].state());exact_state(s,new,refs[new])
                elif action==8 and len(refs)>3:
                    s.call('release',h);del refs[h];rollback(s,packet('raw32',h,1),4)
                else:
                    seed=s.words('seed',h)[0];refs[h].seed(seed);exact_state(s,h,refs[h])
                assert s.call('stats')['live']==len(refs)
                if i%50==0:
                    for handle,m in refs.items():exact_state(s,handle,m)
            record('mixed-multiple-generator-model',{'operations':350,'live_final':len(refs),'last_id':next_id})
            frames+=s.frames
        with Session(image) as s:
            h=s.words('create',count=1,a=17)[0];m=Reference(cpp,17)
            draw(s,h,m,'raw32',100000);draw(s,h,m,'advance',1000000)
            record('maximum-output-and-work',{'returned_values':100000,'advance_words':1000000})
            frames+=s.frames
        with Session(image) as s:
            ids=s.words('create',count=1024,a=99);assert ids==list(range(1,1025))
            assert s.call('stats')['allocation']==64+1024*5088
            rollback(s,packet('create',count=1),3);rollback(s,packet('clone_state',1,1),3)
            m=Reference(cpp,99)
            for h in (1,512,1024):exact_state(s,h,m)
            before=owner_observation(s);s.call('release',512);after=owner_observation(s)
            assert after['bytes']==before['bytes']-5088 and after['pointer']!=before['pointer']
            assert not any(lo<=before['pointer']<hi for lo,hi,_,_ in ranges(s.process.pid))
            rollback(s,packet('get_state',512),4)
            assert s.words('create',count=1,a=22)==[1025]
            exact_state(s,1024,m);exact_state(s,1025,Reference(cpp,22))
            record('capacity-and-interior-release-real-unmap',{'maximum_live':1024,'maximum_state_bytes':64+1024*5088,
                'old_owner_unmapped_while_process_alive':True,'old_owner_bytes':before['bytes'],'new_owner_bytes':after['bytes']})
            frames+=s.frames
        with Session(image,preexec_fn=entropy_denial) as s:
            h=s.words('create',count=1,a=43)[0]
            rollback(s,packet('seed',h),9)
            m=Reference(cpp,43);draw(s,h,m,'raw64',100)
            record('getrandom-kernel-error-atomicity',{'seccomp_errno':11,'returned_status':9,'later_numeric_execution':True})
            frames+=s.frames
        # Set a real kernel AS quota after startup. Basing it on observed VMA
        # size avoids guessing the host's initial Linux process stack size.
        with Session(image) as s:
            s.words('create',count=900,a=77)
            vm_pages=int(Path(f'/proc/{s.process.pid}/statm').read_text().split()[0])
            limit=vm_pages*os.sysconf('SC_PAGE_SIZE')+(1<<20)
            old_limit=resource.prlimit(s.process.pid,resource.RLIMIT_AS)
            resource.prlimit(s.process.pid,resource.RLIMIT_AS,(limit,old_limit[1]))
            rollback(s,packet('create',count=124,a=88),3)
            exact_state(s,1,Reference(cpp,77))
            resource.prlimit(s.process.pid,resource.RLIMIT_AS,old_limit)
            s.call('release',450)
            record('mmap-allocation-failure-preserves-existing-generators',{'address_space_limit':limit,'live_before':900,'failed_growth':124,'later_release_after_quota_restore':True})
            frames+=s.frames
        report={'schema':'heptatorch.cpu-generator-native-qualification.v1','status':'PASS','cases':rows,
            'frames':frames,'exact_output_values':values,'full_rng_state_checks':state_checks,'full_pool_failure_rollbacks':rollbacks,
            'image_sha256':sha(image.read_bytes()),'build_receipt_sha256':sha((build/'build.json').read_bytes()),
            'test_sources_sha256':{name:sha(Path(__file__).with_name(name).read_bytes()) for name in ['qualify.py','client.py','reference.py','reference.cpp']},
            'cpp_reference_sha256':sha(cpp.read_bytes()),'reference':'CPython MT plus independent C++ standard MT19937 initialization',
            'real_pytorch_run':False,'generator_tensor_integration_complete':False,'full_pytorch_generator_complete':False}
    except Exception:
        report={'schema':'heptatorch.cpu-generator-native-qualification.v1','status':'FAIL','cases':rows,
                'frames':frames,'exact_output_values':values,'full_rng_state_checks':state_checks,'full_pool_failure_rollbacks':rollbacks,
                'image_sha256':sha(image.read_bytes()),'error':traceback.format_exc()}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');raise
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
