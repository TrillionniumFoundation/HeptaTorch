"""Native ready-queue and partial-error checks; no host numeric runtime is used."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,signal,struct,traceback

SHA=lambda b:hashlib.sha256(b).hexdigest()
OBSERVATION_SHA='83bc3a03a51c5e2eb9499fb24ad73b9ab73da7452eb8ff4dc5b57e18a3958c7d'

def main():
    ap=argparse.ArgumentParser()
    for name in ('executable','protocol','observations','output'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    inputs={p.resolve():p.read_bytes() for p in (Path(__file__),args.executable,args.protocol,args.observations)}
    assert SHA(inputs[args.protocol.resolve()])=='5b99b0e508c2093fe830e509fffbb46b709f65e85b669ab6d79a32d425aa95ac'
    assert SHA(inputs[args.observations.resolve()])==OBSERVATION_SHA
    observations={r['case']:r['observation'] for r in json.loads(inputs[args.observations.resolve()])['rows']}
    spec=importlib.util.spec_from_file_location('partial_wire',args.protocol)
    wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
    f=lambda v:struct.pack('<'+'f'*len(v),*v)
    words=lambda *v:struct.pack('<'+'Q'*len(v),*v)
    def create(s,shape,values,requires=0):return s.run([wire.tensor(shape,values,requires)])[0]['value']
    def full(s,shape,value,requires=0):return s.run([wire.full(shape,value,requires)])[0]['value']
    def binary(s,op,a,b):return s.run([(2,op,a,b)])[0]['value']
    def total(s,x):return s.run([(4,7,x)])[0]['value']
    def gradient(s,x,values):assert s.run([(11,x)])[0]['payload']==f(values)
    def stats(s):return struct.unpack('<12Q',s.run([(54,)])[0]['payload'])
    def execute(s,t,seed=0,retain=0,expected=0,completed=None,leaves=None,op=None):
        row=s.run([(58,t,seed,retain)])[0]
        assert row['value']==expected and len(row['payload'])==32,row
        detail=struct.unpack('<4Q',row['payload']);assert detail[0]==expected
        for actual,wanted in zip(detail[1:],(completed,leaves,op)):
            if wanted is not None:assert actual==wanted,(detail,(completed,leaves,op))
        return detail
    report={'schema':'heptatorch.partial-backward-native.v1','status':'RUNNING',
        'native_sha256':SHA(inputs[args.executable.resolve()]),'native_executed':True,
        'pytorch_executed':False,'actual_observations_sha256':OBSERVATION_SHA,
        'inputs':{str(p):SHA(b) for p,b in inputs.items()},'rows':[],'frames':0,'rollback_bytes':0}
    def save():(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def case(name,fn,limit=64<<20):
        s=wire.Session(args.executable,directory=args.output/name,address_limit=limit)
        row={'case':name,'status':'RUNNING'};report['rows'].append(row);save()
        try:
            rollback,detail=fn(s);s.run([(53,)]);s.finish()
            row.update(status='PASS',frames=s.frames,details=detail,rollback_bytes=rollback,
                       all_responses_before_eof=True)
            report['frames']+=s.frames;report['rollback_bytes']+=rollback
        except BaseException as error:
            row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc());report['status']='FAIL'
            if s.poll() is None:os.kill(s.pid,signal.SIGKILL)
            try:s.finish(expected=-signal.SIGKILL)
            except BaseException:pass
            save();raise
        finally:
            (args.output/name/'assertions.json').write_text(json.dumps(row,indent=2)+'\n');save()

    def upstream_after_downstream(s):
        reference=observations['failed-upstream-version-check-after-downstream-release']
        assert 'modified by an inplace operation' in reference['call']['message']
        assert 'Trying to backward through the graph a second time' in reference['retry']['message']
        assert reference['call']['gradients_after_call']==[None] and reference['retry']['gradients_after_call']==[None]
        x=create(s,(3,),[1,2,3],1);a=binary(s,3,x,x);b=binary(s,3,a,a);loss=total(s,b)
        s.run([(12,x,0,wire.bits(2))]);before=stats(s)
        first=execute(s,loss,expected=6,completed=2,leaves=0,op=3)
        middle=stats(s);assert middle[9]==before[9]-2
        retry=execute(s,loss,retain=1,expected=10,completed=1,leaves=0,op=3)
        gradient(s,x,[0,0,0])
        rollback=s.error([(57,loss,0,0)],10)
        return rollback,{'actual_Torch_error_codes':[6,10],'native_payloads':[first,retry],
            'leaf_presence_limit':'Existing protocol exposes allocated zeros, not a grad-None object.',
            'downstream_private_saved_released':2,'legacy57_spent_status':10}
    case('actual-upstream-version-after-downstream-release',upstream_after_downstream)

    def branch(s,good_later,shared=False):
        x=full(s,(),2,1);z=x if shared else full(s,(),3,1)
        c=full(s,(),5);d=full(s,(),7)
        if good_later:
            bad=binary(s,3,x,c);good=binary(s,3,z,d)
            left=total(s,bad);right=total(s,good)
        else:
            good=binary(s,3,z,d);bad=binary(s,3,x,c)
            right=total(s,good);left=total(s,bad)
        root=binary(s,1,left,right);s.run([(12,c,0,wire.bits(6))])
        expected_leaf=1 if good_later and not shared else 0
        result=execute(s,root,expected=6,leaves=expected_leaf,op=3)
        gradient(s,x,[0])
        if not shared:gradient(s,z,[7 if good_later else 0])
        return 0,{'good_branch_created_later':good_later,'shared_leaf':shared,'payload':result,
            'expected_good_leaf_gradient':7 if expected_leaf else 0,
            'reference':'Independent dependency/ready-priority contract from pinned Engine; new live Torch branch probe pending.'}
    case('leaf-ready-before-lower-sequence-error',lambda s:branch(s,True))
    case('higher-sequence-error-prevents-leaf-ready',lambda s:branch(s,False))
    case('shared-leaf-waits-for-all-parent-edges',lambda s:branch(s,True,True))

    def duplicate(s):
        x=create(s,(2,),[2,3],1);square=binary(s,3,x,x);loss=total(s,square)
        execute(s,loss,completed=3,leaves=1);gradient(s,x,[4,6])
        x2=create(s,(2,),[1,2],1)
        joined=s.run([(34,0,wire.Payload(words(2,x2,x2)))])[0]['value'];loss2=total(s,joined)
        execute(s,loss2,completed=3,leaves=1);gradient(s,x2,[2,2])
        execute(s,loss2,completed=3,leaves=1);gradient(s,x2,[4,4])
        return 0,{'duplicate_binary_and_variadic_edges_counted_individually':True,'leaf_applies_once_per_pass':True}
    case('duplicate-binary-and-variadic-parent-dependencies',duplicate)

    def retains(s):
        x=create(s,(2,),[2,3],1);c=create(s,(2,),[4,5]);y=binary(s,3,x,c);loss=total(s,y)
        weak=s.run([(42,c)])[0]['value'];s.run([(41,c)])
        for n,flag in enumerate([1,1,0],1):
            execute(s,loss,retain=flag,completed=3,leaves=1);gradient(s,x,[4*n,5*n])
            if flag:
                hold=s.run([(43,weak)])[0]['value'];s.run([(45,hold)])
        rollback=s.error([(43,weak)],6)
        execute(s,loss,retain=1,expected=10,completed=1,leaves=0)
        return rollback,{'retain_sequence':[1,1,0,1],'final_gradient':[12,15]}
    case('retained-and-default-execution-lifetimes',retains)

    def stop_frame(s):
        x=full(s,(),2,1);coefficient=full(s,(),3);changed=full(s,(),4)
        loss=binary(s,3,x,coefficient);s.run([(12,coefficient,0,wire.bits(5))])
        commands=[(12,changed,0,wire.bits(9)),(58,loss,0,0),(12,changed,0,wire.bits(100)),(99,)]
        status,records=wire.parse(s.exchange(wire.packet(commands)))
        assert status==0 and len(records)==2 and records[1]['opcode']==58 and records[1]['value']==6
        assert struct.unpack('<4Q',records[1]['payload'])==(6,0,0,3)
        assert s.run([(10,changed)])[0]['payload']==f([9]);gradient(s,x,[0])
        # A legacy frame error still rolls back a prior write in its entirety.
        rollback=s.error([(12,changed,0,wire.bits(10)),(57,loss,0,0)],6)
        assert s.run([(10,changed)])[0]['payload']==f([9])
        return rollback,{'runtime_error_committed_preceding_write':9,'later_write_and_unknown_opcode_not_executed':True,
            'response_record_count':2,'input_record_count':4,'legacy57_whole_frame_rollback_preserved':True}
    case('runtime-error-stops-frame-and-commits-completed-work',stop_frame)

    def admission(s):
        x=create(s,(2,),[2,3],1);square=binary(s,3,x,x);loss=total(s,square)
        seed=full(s,(1,2),1);nongrad=full(s,(),1)
        rollback=0
        for command,error in [((58,loss,0,2),7),((58,square,seed,0),2),((58,square,0,0),5),
                              ((58,nongrad,0,0),5),((58,1,0,0),1)]:rollback+=s.error([command],error)
        execute(s,loss,completed=3,leaves=1);gradient(s,x,[4,6])
        return rollback,{'all_admission_failures_unchanged':True,'Saved_remained_usable':True}
    case('admission-errors-remain-frame-atomic',admission)

    def gradient_shape(s):
        x=full(s,(2,3),0,1);src=full(s,(2,3),2,1)
        config=wire.Payload(words(2,2,2,2,0,1,1,2))
        y=s.run([(36,x,src,1,config)])[0]['value'];seed=full(s,(2,3),1)
        before=stats(s)[9]
        for _ in range(2):execute(s,y,seed,expected=5,completed=0,leaves=0,op=39)
        assert stats(s)[9]==before;gradient(s,x,[0]*6);gradient(s,src,[0]*6)
        return 0,{'failing_node_not_released':True,'gradient_shape_error_repeat':[5,5]}
    case('invalid-produced-gradient-retains-failing-node',gradient_shape)

    def wide(s):
        ids=[r['value'] for r in s.run([wire.full((1,),1,1) for _ in range(300)])]
        y=s.run([(34,0,wire.Payload(words(300,*ids)))])[0]['value'];seed=full(s,(300,),1)
        execute(s,y,seed,completed=301,leaves=300)
        rows=s.run([(11,x) for x in ids]);assert all(row['payload']==f([1]) for row in rows)
        execute(s,y,seed,completed=301,leaves=300)
        rows=s.run([(11,x) for x in ids]);assert all(row['payload']==f([2]) for row in rows)
        return 0,{'simultaneously_ready_leaf_nodes':300,'heap_crosses_old_descriptor_boundary':True,'leaf_values_compared':600}
    case('wide-ready-heap-and-page-boundary',wide)

    def leaf(s):
        x=full(s,(),3,1);seed=full(s,(),-2)
        execute(s,x,completed=1,leaves=1);gradient(s,x,[1])
        execute(s,x,seed,completed=1,leaves=1);gradient(s,x,[-1])
        s.run([(7,x)]);gradient(s,x,[0])
        return 0,{'root_leaf_ready_immediately':True,'legacy_rehydration_after58':True}
    case('root-leaf-and-legacy-work-rehydration',leaf)

    for p,data in inputs.items():assert p.read_bytes()==data,p
    report.update(status='PASS',cases=len(report['rows']))
    save();print(json.dumps({k:report[k] for k in ('status','cases','frames','rollback_bytes','native_sha256')},indent=2))

if __name__=='__main__':main()
