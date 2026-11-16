"""Legitimate old/new owned step calls; no manufactured owners or host writes."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import struct

SHA=lambda data:hashlib.sha256(data).hexdigest()
Q=lambda words:struct.pack('<'+'Q'*len(words),*words)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('executable','build-receipt','protocol','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    snapshots={p.resolve():p.read_bytes() for p in (Path(__file__),a.executable,a.build_receipt,a.protocol)}
    build=json.loads(snapshots[a.build_receipt.resolve()])
    assert build['status']=='PASS' and build['qualification_only']
    assert SHA(snapshots[a.executable.resolve()])==build['native_sha256']
    assert SHA(snapshots[a.protocol.resolve()])=='5b99b0e508c2093fe830e509fffbb46b709f65e85b669ab6d79a32d425aa95ac'
    spec=importlib.util.spec_from_file_location('adam_handoff_wire',a.protocol)
    wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
    report={'schema':'heptatorch.adam-legitimate-owner-state-handoff.v1','status':'RUNNING',
        'qualification_only':True,'native_executed':True,'Torch_executed_here':False,
        'native_sha256':build['native_sha256'],'old_source_sha256':build['old_core_sha256'],
        'new_source_sha256':build['new_core_sha256'],'rows':[],
        'inputs_sha256':{str(p):SHA(raw) for p,raw in snapshots.items()}}
    def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    for own,other,closing in ((0,1,3),(1,0,2)):
        name='Optimizer08-owner-to-Adam-step' if own==0 else 'Adam-owner-to-Optimizer08-step'
        s=wire.Session(a.executable,directory=a.output/name,address_limit=64<<20)
        def run(route,commands,status=0):
            actual,records=wire.parse(s.exchange(Q([route])+wire.packet(commands)))
            assert actual==status and len(records)==(len(commands) if status==0 else 0)
            assert s.poll() is None
            return records
        def metadata(t):return wire.meta(run(own,[(13,t)])[0])
        empty=s.owner()
        actual,records=wire.parse(s.exchange(Q([own])+b'bad'))
        assert actual==100 and records==[] and s.owner()==empty
        p=run(own,[wire.tensor((2,),[2,3],1)])[0]['value']
        seed=run(own,[wire.tensor((2,),[1,1])])[0]['value']
        assert run(own,[(58,p,seed,1)])[0]['value']==0
        if own==0:
            config=[wire.bits(.125),wire.bits(.5),0,0,0,0]
            groups=[1,1,0]+[0]*6+[p]
        else:
            config=[wire.bits(.125)]+[0]*7
            groups=[1,1,0]+[0]*8+[p]
        opt=run(own,[(71,own,wire.Payload(Q(config)),wire.Payload(Q(groups)))])[0]['value']
        run(own,[(72,opt)])
        assert run(own,[(10,p)])[0]['payload']==struct.pack('<2f',1.875,2.875)
        fields=(0,) if own==0 else (1,2,3)
        states=[run(own,[(77,opt,0,0,field)])[0]['value'] for field in fields]
        assert [metadata(t)['version'] for t in states]==([0] if own==0 else [1,1,2])
        run(own,[(90,p,0),(70,p)])
        assert run(own,[(59,p)])[0]['value']==0
        before=s.owner()
        assert before['length']>0 and before['bytes'][64:72]==(build['old_tag'] if own==0 else build['new_tag']).encode()
        (a.output/name/'before.owner.bin').write_bytes(before['bytes'])
        run(other,[(72,opt)],7)
        after=s.owner();assert after==before
        (a.output/name/'after.owner.bin').write_bytes(after['bytes'])
        run(own,[(90,p,1),(58,p,seed,1),(72,opt),(53,)])
        assert [run(own,[(77,opt,0,0,field)])[0]['value'] for field in fields]==states
        assert [metadata(t)['version'] for t in states]==([2] if own==0 else [2,2,4])
        expected=[1.6875,2.6875] if own==0 else [1.75,2.75]
        assert run(own,[(10,p)])[0]['payload']==struct.pack('<2f',*expected)
        if own==1:assert run(own,[(10,states[0])])[0]['payload']==struct.pack('<f',2)
        pointer=s.owner()['pointer'];assert s.mapped(pointer)
        actual,records=wire.parse(s.exchange(Q([closing])))
        assert actual==0 and records==[]
        assert s.owner()=={'pointer':0,'length':0,'bytes':b'','live':1}
        assert not s.mapped(pointer) and s.poll() is None
        z=run(other,[wire.full((),7,1)])[0]['value']
        run(other,[(7,z)]);assert run(other,[(11,z)])[0]['payload']==struct.pack('<f',1)
        assert s.owner()['bytes'][64:72]==(build['new_tag'] if own==0 else build['old_tag']).encode()
        s.finish()
        report['rows'].append({'case':name,'status':'PASS','frames':s.frames,
            'old_step_route':0,'new_step_route':1,'wrong_profile_status':7,
            'pointer_length_and_owner_bytes_unchanged':True,
            'preserved_owner_bytes':before['length'],'before_sha256':SHA(before['bytes']),
            'after_sha256':SHA(after['bytes']),'real_state_fields':list(fields),
            'same_state_Tensor_tokens_and_Versions_resume':True,
            'cross_profile_close_unmapped_while_process_alive':True,
            'opposite_profile_initialized_from_legitimate_live_empty':True,
            'all_responses_before_stdin_eof':True})
        save()
    for path,raw in snapshots.items():assert path.read_bytes()==raw,path
    (a.output/'executed-test.py').write_bytes(snapshots[Path(__file__).resolve()])
    report.update(status='PASS',cases=len(report['rows']),frames=sum(row['frames'] for row in report['rows']),
        preserved_owner_bytes=sum(row['preserved_owner_bytes'] for row in report['rows']))
    save();print(json.dumps({key:report[key] for key in ('status','cases','frames','preserved_owner_bytes')}))


if __name__=='__main__':main()
