#!/usr/bin/env python3
"""Actual native Tensor/Generator execution against independent exact models."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import signal
import struct
import subprocess
import traceback

from protocol import Session,Payload,packet,parse,full,bits,meta,reference as ref
from reference import Reference,untemper
from reference_fp import uniform,physical_indices

SHA=lambda raw:hashlib.sha256(raw).hexdigest()
WORDS=lambda values:struct.pack('<'+'I'*len(values),*values)
PROBE_SHA='098899a6085d4b8645aec515bb94dfe7c2a1b1fc79c263ab5aa7842332684dd7'

def view(tensor,shape,strides,offset):
    assert len(shape)==len(strides)<=4
    cfg=(len(shape),*shape,*([0]*(4-len(shape))),*strides,*([0]*(4-len(shape))),offset)
    return (69,tensor,Payload(struct.pack('<10Q',*cfg)),80)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True);p.add_argument('--torch-observation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    build=a.build.resolve();image=build/'tensor-rng-session';identity=json.loads((build/'build.json').read_text())
    image_sha=SHA(image.read_bytes());assert image_sha==identity['executable_sha256']
    probe_raw=a.torch_observation.read_bytes();assert SHA(probe_raw)==PROBE_SHA
    observation=json.loads(probe_raw);assert observation['oracle_provenance']['status']=='PASS'
    assert len(observation['oracle_provenance']['checks'])==11 and all(observation['oracle_provenance']['checks'].values())
    cpp=out/'std-mt-reference'
    result=subprocess.run([shutil.which('g++'),'-std=c++17','-O2',str(Path(__file__).with_name('reference.cpp')),'-o',str(cpp)],capture_output=True)
    (out/'reference-build.stdout').write_bytes(result.stdout);(out/'reference-build.stderr').write_bytes(result.stderr)
    assert result.returncode==0 and not result.stderr
    sources={p.name:SHA(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')}
    report={'schema':'heptatorch.tensor-rng-native-qualification.v1','status':'RUNNING','executable_sha256':image_sha,
        'build_receipt_sha256':SHA((build/'build.json').read_bytes()),'torch_observation_sha256':PROBE_SHA,
        'actual_torch_reexecuted_here':False,'host_tensor_or_rng_production_runtime':False,
        'mathematical_reference':'std::mt19937 seed state + CPython MT + exact signed-dyadic FMA/RNE',
        'cases':[],'frames':0,'exact_values':0,'full_mt_states':0,'whole_owner_rollbacks':0,'source_sha256':sources}
    def save():(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def state(s,g,m):
        raw=s.run([(63,g)])[0]['payload'];assert raw==m.state(),('state',g,SHA(raw),SHA(m.state()))
        report['full_mt_states']+=1;return raw
    def values(s,t,expected):
        raw=s.run([(39,t,0,len(expected))])[0]['payload']
        assert raw==WORDS(expected),('values',len(expected),list(struct.unpack('<'+'I'*min(8,len(expected)),raw[:32])),expected[:8])
        report['exact_values']+=len(expected);return raw
    def rollback(s,commands,error):
        s.error(commands,error);report['whole_owner_rollbacks']+=1
    def case(name,fn):
        s=Session(image,directory=out/name);row={'name':name,'status':'RUNNING'};report['cases'].append(row);save()
        try:
            detail=fn(s) or {};s.finish();row.update(status='PASS',frames=s.frames,**detail);report['frames']+=s.frames
        except BaseException as e:
            row.update(status='FAIL',error=repr(e),traceback=traceback.format_exc());report['status']='FAIL';save()
            if s.poll() is None:os.kill(s.pid,signal.SIGKILL);os.waitpid(s.pid,0)
            raise
        finally:
            (out/name/'assertions.json').write_text(json.dumps(row,indent=2)+'\n');save()

    def arithmetic(s):
        for number,row in enumerate(observation['arithmetic']):
            lo,hi=row['from_bits'],row['to_bits'];n=row['values'];seed=1729+number;m=Reference(cpp,seed)
            result=s.run([full((n,),0),(60,seed),(68,ref(1),ref(2),lo,hi),(39,ref(1),0,n),(63,ref(2)),(13,ref(1)),(53,)])
            expected=[uniform(m.word(),lo,hi) for _ in range(n)];raw=result[3]['payload']
            assert raw==WORDS(expected) and SHA(raw)==row['observed_bits_sha256'],row['case']
            assert result[4]['payload']==m.state() and SHA(m.state())==row['state_sha256'],row['case']
            assert meta(result[5])['version']==1
            report['exact_values']+=n;report['full_mt_states']+=1
            s.run([(41,result[0]['value']),(66,result[1]['value']),(33,0),(53,)])
        return {'ranges':30,'actual_pinned_torch_bit_hashes_matched':30,'fused_pipeline':True}
    case('exact-fma-and-actual-torch-recorded-corpus',arithmetic)

    def traversal(s):
        for row in observation['traversal']:
            shape,strides,offset=row['shape'],row['strides'],row['storage_offset'];n=math.prod(shape);m=Reference(cpp,99)
            r=s.run([full((row['storage_elements'],),-7),(60,99),view(ref(1),shape,strides,offset)])
            t,g,v=[x['value'] for x in r]
            if row['error']:
                rollback(s,[(68,v,g,0,bits(1))],2)
                expected=[bits(-7)]*row['storage_elements']
            else:
                s.run([(68,v,g,0,bits(1))]);expected=[bits(-7)]*row['storage_elements']
                for at in physical_indices(shape,strides,offset):expected[at]=uniform(m.word(),0,bits(1))
                assert expected==row['actual_storage_bits'],row['case']
            values(s,t,expected);state(s,g,m)
            r=s.run([(13,t),(13,v),(53,)])
            a,b=meta(r[0]),meta(r[1]);assert a['version']==b['version']==row['version_after']
            assert a['storage_id']==b['storage_id'] and a['version_id']==b['version_id']
            assert b['shape'][:len(shape)]==shape and b['strides'][:len(shape)]==strides and b['storage_offset']==offset*4
            s.run([(41,t),(41,v),(66,g),(33,0),(53,)])
        return {'layouts':12,'exact_actual_torch_layouts':12,'empty_offset_999':True}
    case('all-observed-layouts-and-alias-versions',traversal)

    def bounds(s):
        r=s.run([full((0,),0),(60,17),full((3,),0,1)]);empty,g,leaf=[x['value'] for x in r];m=Reference(cpp,17)
        s.run([(68,empty,g,0,bits(1))]);state(s,g,m)
        assert meta(s.run([(13,empty)])[0])['version']==1
        rollback(s,[(68,empty,g,bits(2),bits(1))],7)
        rollback(s,[(68,empty,g,0xff7fffff,0x7f7fffff)],7)
        rollback(s,[(68,leaf,g,0,bits(1))],5)
        s.run([(68,leaf,g,0,bits(1),1)]);values(s,leaf,[uniform(m.word(),0,bits(1)) for _ in range(3)]);state(s,g,m)
        assert meta(s.run([(13,leaf)])[0])['version']==1
        saved=s.run([(46,leaf)])[0]['value'];s.run([(68,leaf,g,bits(2),bits(2),1)])
        values(s,leaf,[bits(2)]*3)
        for _ in range(3):m.word()
        state(s,g,m);rollback(s,[(47,saved)],6)
        for lo,hi in [(0x7f800000,0x7f800000),(0xff800000,0),(0,0x7fc00001),(0xff7fffff,0x3f800000),(0xbf800000,0x7f7fffff)]:
            rollback(s,[(68,empty,g,lo,hi)],7)
        rollback(s,[(68,leaf,g,0,bits(1),2)],7)
        s.run([(53,)])
        return {'pinned_boundaries':5,'explicit_no_grad_flag':1,'saved_version_invalidated':True}
    case('empty-range-no-grad-and-version-boundaries',bounds)

    def empty_lifetime(s):
        r=s.run([full((8,),-7),(60,12),view(ref(1),(0,3),(0,1),999),(46,ref(3)),(56,ref(4)),(19,ref(3)),(18,ref(3)),(55,ref(3)),(53,)])
        base,g,v,saved,unpacked,detached,cloned,data=[x['value'] for x in r[:8]]
        for t in (v,unpacked,detached,cloned,data):values(s,t,[])
        assert s.run([(49,saved,0,0)])[0]['payload']==b''
        s.run([(68,unpacked,g,0,bits(1))]);state(s,g,Reference(cpp,12))
        rollback(s,[(49,saved,0,0)],6)
        for t in (v,unpacked,detached):assert meta(s.run([(13,t)])[0])['version']==1
        assert meta(s.run([(13,data)])[0])['version']==0
        rollback(s,[view(base,(1,),(1,),999)],2)
        rollback(s,[view(base,(0,),(1,),268435457)],3)
        extreme=s.run([view(base,(0,),(1,),268435456),(53,)])[0]['value']
        assert meta(s.run([(13,extreme)])[0])['storage_offset']==1073741824
        s.run([(48,saved)]+[(41,t) for t in (base,v,unpacked,detached,cloned,data,extreme)]+[(66,g),(53,)])
        return {'empty_aliases':6,'zero_element_offsets_bounded_bytes':1073741824}
    case('empty-outside-storage-lifetime-and-access-bounds',empty_lifetime)

    def clone_and_caches(s):
        r=s.run([full((625,),0),(60,0xfedcba9876543210)]);t,g=[x['value'] for x in r];m=Reference(cpp,0xfedcba9876543210)
        raw=bytearray(m.state());struct.pack_into('<Q',raw,16,0xdeadbeef00000000);struct.pack_into('<I',raw,12,7)
        for i in range(624):struct.pack_into('<Q',raw,24+i*8,struct.unpack_from('<Q',raw,24+i*8)[0]|0xbeefcafe00000000)
        raw[5016:5024]=b'deadbits';raw[5032:5040]=b'rhodead!';raw[5044:5048]=b'pad!';raw[5053:5056]=b'xyz'
        struct.pack_into('<Q',raw,5024,0xfff0000000000123);struct.pack_into('<I',raw,5040,1)
        struct.pack_into('<I',raw,5048,0x7f800123);raw[5052]=1
        s.run([(64,g,Payload(raw),5056)]);m.restore(raw);state(s,g,m)
        c=s.run([(65,g)])[0]['value'];other=Reference(cpp,state=m.state())
        s.run([(68,t,g,0,bits(1))]);values(s,t,[uniform(m.word(),0,bits(1)) for _ in range(625)]);state(s,g,m);state(s,c,other)
        s.run([(68,t,c,0,bits(1))]);values(s,t,[uniform(other.word(),0,bits(1)) for _ in range(625)]);state(s,c,other)
        s.run([(61,g,(1<<64)-1)]);m.seed((1<<64)-1);state(s,g,m)
        assert s.run([(62,g)])[0]['value']==(1<<64)-1
        crafted=bytearray(m.state());struct.pack_into('<IIQ',crafted,8,624,1,1);struct.pack_into('<Q',crafted,32,untemper(0x00ffffff))
        s.run([(64,g,Payload(crafted),5056)]);m.restore(crafted)
        scalar=s.run([full((),0)])[0]['value'];lo,hi=0x3f800001,0x3f800002
        s.run([(68,scalar,g,lo,hi)]);expected=uniform(m.word(),lo,hi);assert expected==lo
        values(s,scalar,[expected]);state(s,g,m);s.run([(53,)])
        return {'twist_boundary':625,'cached_normals_preserved':True,'endpoint_clamp_exact':True}
    case('clone-caches-normalization-reseed-and-clamp',clone_and_caches)

    def rollback_and_growth(s):
        r=s.run([full((8,),0),(60,99)]);t,g=[x['value'] for x in r];m=Reference(cpp,99)
        failures=[([(68,t,g,0,bits(1)),(999,)],4),
            ([(68,t,g,0,bits(1)),(66,g),(68,t,g,0,bits(1))],1),
            ([(68,t,g,0,bits(1)),(61,g,123),(69,t,0,80)],100),
            ([(68,t,g,0,bits(1)),(60,123),(62,ref(2)+1)],1)]
        for commands,error in failures:rollback(s,commands,error)
        state(s,g,m);values(s,t,[0]*8)
        old=s.owner();r=s.run([(68,t,g,0,bits(1))]+[(60,1000+i) for i in range(100)]+[(53,)])
        now=s.owner();assert now['length']>old['length'] and now['pointer']!=old['pointer'] and not s.mapped(old['pointer'])
        values(s,t,[uniform(m.word(),0,bits(1)) for _ in range(8)]);state(s,g,m)
        all_generators=[x['value'] for x in r[1:-1]]
        for i in (0,49,99):state(s,all_generators[i],Reference(cpp,1000+i))
        for x in all_generators:s.run([(66,x)])
        s.run([(53,)])
        return {'capacity_retry_after_rng_draw':True,'old_owner_really_unmapped':True,'generators_created_in_one_frame':100}
    case('tensor-generator-joint-rollback-and-growth-replay',rollback_and_growth)

    def attacks(s):
        r=s.run([full((4,),1),(60,11)]);t,g=[x['value'] for x in r];m=Reference(cpp,11)
        bad=[([(68,t,t,0,bits(1))],1),([(68,g,g,0,bits(1))],1),([(63,t)],1),([(67,g,1000001)],3),
             ([(68,t,g,1<<32,bits(1))],7),([(68,t,g,0,bits(1),0,1)],7),([(60,1,1)],7),
             ([(63,g,1)],7),([(64,g,Payload(bytes(5055)),5055)],100),([(64,g,0,5056)],100),
             ([(69,t,Payload(bytes(80)),79)],100),([view(t,(2,2),(1,1),4)],2),
             ([view(t,(0,),((1<<64)-1,),0)],3),([view(t,(262145,0),(0,1),0)],3)]
        for at,width,val in [(8,4,0),(8,4,625),(12,4,0),(16,8,625),(5040,4,2),(5052,1,2)]:
            raw=bytearray(m.state());raw[at:at+width]=val.to_bytes(width,'little');bad.append(([(64,g,Payload(raw),5056)],7))
        for commands,error in bad:rollback(s,commands,error)
        old=s.owner();state_bytes=old['bytes'];slots=struct.unpack_from('<Q',state_bytes,139264+16)[0]
        record=139264+128+6*slots*64+((g&0xffffff)-1)*64
        private=struct.unpack_from('<Q',state_bytes,record+16)[0]
        assert private and (private>>24)&255==6
        rollback(s,[(50,private,0,0)],1);rollback(s,[(45,private)],1)
        s.run([(66,g)]);rollback(s,[(63,g)],1)
        new=s.run([(60,11)])[0]['value'];assert new!=g and (new&0xffffff)==(g&0xffffff)
        rollback(s,[(63,g)],1);state(s,new,m);s.run([(53,)])
        return {'malformed_or_stale_requests':len(bad)+5,'private_generator_hold_is_not_public':True}
    case('typed-handles-payload-guards-and-stale-generations',attacks)

    def sequence(s):
        r=s.run([full((3,4),0),(60,17),(60,23)]);t,g0,g1=[x['value'] for x in r]
        gens={g0:Reference(cpp,17),g1:Reference(cpp,23)};choice=random.Random(0x924f);version=0
        for i in range(100):
            g=choice.choice([g0,g1]);lo=bits(-3.25);hi=bits(7.75)
            if i%9==0:
                n=choice.randrange(625);s.run([(67,g,n)])
                for _ in range(n):gens[g].word()
            else:
                s.run([(68,t,g,lo,hi)]);version+=1
                values(s,t,[uniform(gens[g].word(),lo,hi) for _ in range(12)])
            state(s,g,gens[g])
            assert meta(s.run([(13,t)])[0])['version']==version
        for g,m in gens.items():state(s,g,m)
        s.run([(53,)])
        return {'interleaved_steps':100,'tensor_version':version,'independent_generators':2}
    case('persistent-interleaved-generator-sequences',sequence)

    assert SHA(image.read_bytes())==image_sha
    assert {p.name:SHA(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')}==sources
    report['status']='PASS';save();print(json.dumps({k:v for k,v in report.items() if k not in ('cases','source_sha256')}))

if __name__=='__main__':main()
