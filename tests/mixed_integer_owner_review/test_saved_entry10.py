#!/usr/bin/env python3
"""Actual native owned step capacity and atomicity for original Saved-T access.

The unchanged checked HIA probe allocates a new real buffer and copies only
pool-relative state bytes. It never imports or memcpy-copies an owner descriptor.
Its real compiler/linker receipt is independently rebound before execution.
"""
import argparse
import json
import struct
import subprocess
import traceback
from pathlib import Path

import owner640_mixed10 as observer
from resource_support import Custody, State, create, integer_raw, load, sha, write_json, PROBE, COMPILER, LINKER


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('build','probe-build','protocol','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    custody=Custody(out)
    report={'status':'RUNNING','source_sha256':observer.SOURCE,'native_sha256':observer.IMAGE,
        'cases':[],'native_probe_processes':0,'interactive_frames':0,'complete_owner_bytes_compared':0,
        'native_probe_recompiled_this_invocation':False,'Torch_executed_here':False,
        'owner_descriptor_memcpy':False,'foreign_pointer_import':False,
        'whole_pytorch_gate':'HOLD','pytorch_api_denominator':2590,'closed_schema_count':0,
        'full_pytorch_rewrite_complete':False}
    session=None
    def save():write_json(out/'report.json',report)
    try:
        custody.bind(Path(__file__),copy='executed-test.py')
        custody.bind(Path(observer.__file__),copy='owner640_mixed10.py')
        wire,image,build,_=load(custody,a.build,a.protocol,
            expected_source=observer.SOURCE,expected_image=observer.IMAGE)
        prior=json.loads(custody.bind(a.probe_build/'report.json',copy='probe-build/report.json'))
        assert prior['status']=='PASS' and prior['case_count']==prior['passed']==5
        assert prior['source_sha256']==observer.SOURCE and prior['native_sha256']==observer.IMAGE
        assert prior['probe_source_sha256']==PROBE and prior['compiler_sha256']==COMPILER
        assert prior['runtime_linker_sha256']==LINKER and prior['linked_modules']==12
        custody.bind(a.probe_build/'inputs/entry_probe.hepta',PROBE,'probe-build/entry_probe.hepta')
        compiled=custody.bind(a.probe_build/'entry-probe.o',copy='probe-build/entry-probe.o')
        request=custody.bind(a.probe_build/'link.request',copy='probe-build/link.request')
        objects=[custody.bind(a.build/(u['name']+'.o'),u['object_sha256']) for u in build['units']]+[compiled]
        exact=b'HIASTKR1'+struct.pack('<3Q',len(objects),5,0)+b'probe'
        exact+=b''.join(struct.pack('<Q',len(b))+b for b in objects)
        assert request==exact and len(objects)==12
        for name in ('compile.exit.json','link.exit.json'):
            assert json.loads(custody.bind(a.probe_build/name,copy='probe-build/'+name))=={'exit_code':0}
        for name in ('compile.stderr','link.stderr'):
            assert custody.bind(a.probe_build/name,copy='probe-build/'+name)==b''
        probe=a.probe_build/'owned-entry-probe'
        custody.bind(probe,prior['probe_image_sha256'],'probe-build/owned-entry-probe')
        report.update(probe_source_sha256=PROBE,probe_image_sha256=prior['probe_image_sha256'],
            real_previous_compile_and_link_receipt_sha256=sha((a.probe_build/'report.json').read_bytes()),linked_modules=12)

        def invoke(before,commands,capacity):
            packet=wire.packet(commands)
            raw=struct.pack('<3Q',len(before),len(packet),capacity)+before+packet
            assert len(raw)<=1<<20
            n=report['native_probe_processes'];report['native_probe_processes']+=1
            result=subprocess.run([str(probe)],input=raw,capture_output=True,env={'PATH':''},timeout=30)
            prefix=out/f'probe-{n:03d}'
            for suffix,data in (('.input',raw),('.stdout',result.stdout),('.stderr',result.stderr)):
                Path(str(prefix)+suffix).write_bytes(data)
            write_json(Path(str(prefix)+'.exit.json'),{'exit_code':result.returncode})
            assert result.returncode==0 and result.stderr==b''
            returned,size,cap=struct.unpack_from('<3Q',result.stdout)
            assert cap==capacity and len(result.stdout)==24+cap+size
            return returned,result.stdout[24:24+cap],result.stdout[24+cap:]

        session=wire.Session(image,directory=out/'transport')
        base=create(wire,session,2,[5],integer_raw([0,3,0,5,0]))
        config=struct.pack('<10Q',1,2,0,0,0,2,0,0,0,1)
        view=session.run([(69,base,wire.Payload(config),80)])[0]['value']
        x=create(wire,session,0,[2],struct.pack('<2f',2.,4.),1)
        gen=session.run([(60,1729)])[0]['value']
        root=session.run([(103,3,x,view)])[0]['value']
        initial=State(session.owner()['bytes']);vi,vn=initial.tensor_node(view)
        session.run([(41,base),(41,view)])
        original_rng=State(session.owner()['bytes']).rng(gen)
        before=session.owner()['bytes']
        assert State(before).node(vi)[200//8]==0
        response=session.exchange(wire.packet([(107,root,1)]))
        status,records=wire.parse(response);after=session.owner()['bytes']
        assert status==0 and len(response)==64 and len(records)==1 and records[0]['payload']==b''
        held=records[0]['value'];hi,hn=State(after).tensor_node(held)
        assert hi==vi and tuple(hn[j] for j in (216//8,224//8,320//8))==tuple(vn[j] for j in (216//8,224//8,320//8))
        assert State(after).rng(gen)==original_rng and before!=after
        capacities=[]
        for cap in (0,1,31,32,33,47,63):
            returned,output,owner=invoke(before,[(107,root,1)],cap)
            assert owner==before and State(owner).rng(gen)==original_rng
            report['complete_owner_bytes_compared']+=len(before)
            if cap<32:assert returned==0 and output==bytes(cap)
            else:assert returned==32 and wire.parse(output[:returned])==(3,[])
            capacities.append({'capacity':cap,'returned':returned,'owner_sha256':sha(owner)})
        returned,output,owner=invoke(before,[(107,root,1)],64)
        assert returned==64 and output==response and owner==after
        report['complete_owner_bytes_compared']+=len(owner)
        returned,output,owner=invoke(before,[(107,root,1),(999,)],96)
        assert returned==32 and wire.parse(output[:returned])==(4,[]) and owner==before
        report['complete_owner_bytes_compared']+=len(owner)
        report['cases'].append({'name':'capacity_before_publication_and_later_ordinary_rollback','status':'PASS',
            'short_capacities':capacities,'exact_capacity':64,'same_original_T_S_V_incarnation':True,
            'entire_owner_rollback_includes_publication_and_full5056_RNG':True})
        save()

        def ordinary_error(name,commands,error):
            original=session.owner();reply=session.exchange(wire.packet(commands))
            assert wire.parse(reply)==(error,[]) and session.owner()==original
            returned,output,owner=invoke(original['bytes'],commands,64)
            assert returned==32 and wire.parse(output[:returned])==(error,[]) and owner==original['bytes']
            assert State(owner).rng(gen)==original_rng
            report['complete_owner_bytes_compared']+=2*len(owner)
            report['cases'].append({'name':name,'status':'PASS','expected_error':error,
                'full_owner_bytes':len(owner),'owner_sha256':sha(owner),'real_interactive_and_owned_probe_both_atomic':True})
            save()
        ordinary_error('slot_greater_than_one',[(107,root,2)],7)
        ordinary_error('reserved_argument',[(107,root,1,1)],7)
        # Successful default backward spends this exact root's real Saved set.
        seed=create(wire,session,0,[2],struct.pack('<2f',1.,1.))
        result=session.run([(58,root,seed,0)])[0]
        assert struct.unpack('<4Q',result['payload'])[0]==0
        ordinary_error('spent_Saved_set',[(107,root,1)],10)
        original=session.owner()['bytes'];reply=session.exchange(wire.packet([(107,root,0)]))
        absent_after=session.owner()['bytes']
        assert wire.parse(reply)[0]==0 and wire.parse(reply)[1][0]['value']==0
        returned,output,owner=invoke(original,[(107,root,0)],64)
        assert returned==64 and output==reply and owner==absent_after
        assert State(owner).rng(gen)==original_rng
        report['complete_owner_bytes_compared']+=len(owner)
        report['cases'].append({'name':'absent_slot_remains_absent_after_default_release','status':'PASS','value':0})
        # A new root saves the same actual view and its current Version.
        newer=session.run([(103,3,x,held)])[0]['value']
        session.run([(106,held,1)])
        ordinary_error('Saved_Version_guard',[(107,newer,1)],6)
        assert session.run([(53,)])[0]['value']==1
        final=session.owner();observer.verify_snapshot(observer.decode_owner(final['bytes'],final['pointer']))
        report.update(status='PASS',case_count=len(report['cases']),passed=len(report['cases']))
    except BaseException as error:
        report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
    finally:
        if session is not None:
            report['interactive_frames']=session.frames
            try:session.finish();report['EOF_exit_code']=session.exit_code
            except BaseException as error:report.update(status='FAIL',transport_error=repr(error))
        try:report['inputs']=custody.finish();report['source_custody_after']=True
        except BaseException as error:report.update(status='FAIL',custody_error=repr(error))
        save()
    print(json.dumps({k:report.get(k) for k in ('status','case_count','passed','native_probe_processes','interactive_frames','error')}))
    return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
