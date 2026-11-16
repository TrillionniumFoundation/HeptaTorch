"""Execute legal HIA owner handoffs across HTSTATE5/HTSTATE6 without memory writes."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,struct

SHA=lambda data:hashlib.sha256(data).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--executable',type=Path,required=True)
    ap.add_argument('--build-receipt',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    inputs={p.resolve():p.read_bytes() for p in (Path(__file__),args.executable,args.build_receipt,args.protocol)}
    build=json.loads(inputs[args.build_receipt.resolve()])
    assert build['status']=='PASS' and build['qualification_only'] is True
    assert SHA(inputs[args.executable.resolve()])==build['native_sha256']
    assert SHA(inputs[args.protocol.resolve()])=='5b99b0e508c2093fe830e509fffbb46b709f65e85b669ab6d79a32d425aa95ac'
    spec=importlib.util.spec_from_file_location('cross_state_wire',args.protocol)
    wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
    report={'schema':'heptatorch.owned-state-cross-profile-execution.v1','status':'RUNNING',
            'qualification_only':True,'native_executed':True,'pytorch_executed':False,
            'native_sha256':build['native_sha256'],'old_source_sha256':build['old_core_sha256'],
            'new_source_sha256':build['new_core_sha256'],
            'inputs':{str(p):SHA(b) for p,b in inputs.items()},'rows':[]}
    def save(): (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    for own_route,other_route,close_route in [(0,1,3),(1,0,2)]:
        name='old-state-to-new-step' if own_route==0 else 'new-state-to-old-step'
        directory=args.output/name;session=wire.Session(args.executable,directory=directory,address_limit=64<<20)
        def execute(route,commands,status=0):
            returned,records=wire.parse(session.exchange(struct.pack('<Q',route)+wire.packet(commands)))
            assert returned==status,(name,returned,status)
            assert len(records)==(len(commands) if status==0 else 0)
            assert session.poll() is None
            return records
        # The first malformed frame must return the exact live-empty owner.
        empty_before=session.owner()
        status,_=wire.parse(session.exchange(struct.pack('<Q',own_route)+b'bad'))
        assert status==100 and session.owner()==empty_before
        x=execute(own_route,[wire.tensor((2,),[2,3],1)])[0]['value']
        one=execute(own_route,[wire.full((),1)])[0]['value']
        added=execute(own_route,[(2,1,x,one)])[0]['value']
        loss=execute(own_route,[(4,7,added)])[0]['value']
        backward=(7,loss) if own_route==0 else (57,loss,0,0)
        execute(own_route,[backward])
        assert execute(own_route,[(11,x)])[0]['payload']==struct.pack('<2f',1,1)
        if own_route==1:
            execute(own_route,[(70,x)])
            assert execute(own_route,[(59,x)])[0]['value']==0
            assert execute(own_route,[(11,x)])[0]['payload']==struct.pack('<2f',0,0)
        before=session.owner();expected_tag=b'HTSTATE5' if own_route==0 else b'HTSTATE6'
        assert before['length']>0 and before['bytes'][64:72]==expected_tag
        (directory/'before-rejected-handoff.owner.bin').write_bytes(before['bytes'])
        # The checked controller consumes the one owner into the other actual
        # step; its rejection returns that same owner without cloning/mutation.
        execute(other_route,[(8,x)],7)
        after=session.owner();assert after==before
        (directory/'after-rejected-handoff.owner.bin').write_bytes(after['bytes'])
        execute(own_route,[backward,(53,)])
        resumed=[2,2] if own_route==0 else [1,1]
        assert execute(own_route,[(11,x)])[0]['payload']==struct.pack('<2f',*resumed)
        to_close=session.owner();pointer=to_close['pointer']
        assert session.mapped(pointer)
        closed_status,closed_records=wire.parse(session.exchange(struct.pack('<Q',close_route)))
        assert closed_status==0 and closed_records==[]
        empty=session.owner();assert empty['pointer']==0 and empty['length']==0 and empty['bytes']==b''
        assert not session.mapped(pointer) and session.poll() is None
        # Reinitialize the opposite implementation from the returned live-empty
        # owner; no serialized owner pointer is ever accepted from the test.
        next_tensor=execute(other_route,[wire.full((),7,1)])[0]['value']
        execute(other_route,[(7,next_tensor)])
        assert execute(other_route,[(11,next_tensor)])[0]['payload']==struct.pack('<f',1)
        after_reinit=session.owner()
        assert after_reinit['bytes'][64:72]==(b'HTSTATE6' if own_route==0 else b'HTSTATE5')
        session.finish()
        report['rows'].append({'case':name,'status':'PASS','frames':session.frames,
            'old_route':own_route,'wrong_route':other_route,'cross_format_close_route':close_route,
            'rejection_status':7,'entire_owner_bytes_unchanged':before['length'],
            'owner_sha256_before':SHA(before['bytes']),'owner_sha256_after':SHA(after['bytes']),
            'pointer_and_length_unchanged':True,'resumed_gradient':resumed,
            'cross_format_close_unmapped_while_process_alive':True,
            'reinitialized_opposite_from_live_empty':True,'all_responses_before_stdin_eof':True,
            'only_read_only_proc_observation':True})
        save()
    for p,data in inputs.items():assert p.read_bytes()==data,p
    report.update(status='PASS',cases=len(report['rows']),frames=sum(row['frames'] for row in report['rows']),
                  entire_owner_bytes_unchanged=sum(row['entire_owner_bytes_unchanged'] for row in report['rows']))
    save();print(json.dumps({k:report[k] for k in ('status','cases','frames','entire_owner_bytes_unchanged')},indent=2))

if __name__=='__main__':main()
