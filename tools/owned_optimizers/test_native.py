"""Independent dyadic/native ownership checks; Python only serializes and observes."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, signal, struct, traceback
from fractions import Fraction

SHA=lambda raw:hashlib.sha256(raw).hexdigest()
Q=lambda words:struct.pack('<'+'Q'*len(words),*words)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','source','protocol','output'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--expected-sha256',required=True);ap.add_argument('--expected-source-sha256',required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    inputs={p.resolve():p.read_bytes() for p in (Path(__file__),a.binary,a.source,a.protocol)}
    assert SHA(inputs[a.binary.resolve()])==a.expected_sha256
    assert SHA(inputs[a.source.resolve()])==a.expected_source_sha256
    assert SHA(inputs[a.protocol.resolve()])=='5b99b0e508c2093fe830e509fffbb46b709f65e85b669ab6d79a32d425aa95ac'
    spec=importlib.util.spec_from_file_location('optimizer_wire',a.protocol)
    wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
    def config(lr=.125,momentum=0,dampening=0,weight_decay=0,nesterov=0,maximize=0):
        return [wire.bits(v) for v in (lr,momentum,dampening,weight_decay)]+[nesterov,maximize]
    def group(ids,overrides=None):
        fields=('lr','momentum','dampening','weight_decay','nesterov','maximize');words=[0]*6;mask=0
        for key,value in (overrides or {}).items():
            i=fields.index(key);mask|=1<<i;words[i]=wire.bits(value) if i<4 else value
        return Q([len(ids),mask]+words+list(ids))
    def create(s,ids=None,groups=None,**kwargs):
        if groups is None:groups=[group(ids)]
        record=s.run([(71,0,wire.Payload(Q(config(**kwargs))),wire.Payload(Q([len(groups)])+b''.join(groups)))])[0]
        return record['value'],struct.unpack('<4Q',record['payload'])
    def values(s,t,expected,gradient=False):
        record=s.run([(11 if gradient else 10,t)])[0]
        target=struct.pack('<'+'f'*len(expected),*[float(v) for v in expected])
        assert record['value']==len(expected) and record['payload']==target,(t,record,target.hex())
        return len(expected)
    def metadata(s,t):return wire.meta(s.run([(13,t)])[0])
    def state(s,opt,g=0,p=0):
        record=s.run([(76,opt,g,p)])[0]
        return record['value'],struct.unpack('<4Q',record['payload'])
    def state_tensor(s,opt,g=0,p=0):return s.run([(77,opt,g,p)])[0]['value']
    def seed(s,t,shape,values):
        v=s.run([wire.tensor(shape,values)])[0]['value'];s.run([(58,t,v,0)]);return v
    report={'schema':'heptatorch.owned-optimizer-native-qualification.v1','status':'RUNNING',
            'native_executable_sha256':a.expected_sha256,'native_source_sha256':a.expected_source_sha256,
            'inputs_sha256':{str(p):SHA(data) for p,data in inputs.items()},'native_executed':True,
            'pinned_Torch_executed':False,'rows':[],'frames':0,'exact_values':0,'rollback_bytes':0}
    def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def case(name,body):
        s=wire.Session(a.binary,directory=a.output/name,address_limit=128<<20)
        row={'case':name,'status':'RUNNING'};report['rows'].append(row);save()
        try:
            exact,rollback,details=body(s);s.run([(53,)]);s.finish()
            row.update(status='PASS',frames=s.frames,exact_values=exact,rollback_bytes=rollback,
                       details=details,all_responses_before_stdin_eof=True)
            for key in ('frames','exact_values','rollback_bytes'):report[key]+=row[key]
        except BaseException as error:
            row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc());report['status']='FAIL'
            if s.poll() is None:os.kill(s.pid,signal.SIGKILL)
            try:s.finish(expected=-signal.SIGKILL)
            except BaseException:pass
            save();raise
        finally:(a.output/name/'assertions.json').write_text(json.dumps(row,indent=2)+'\n');save()

    def lazy(s):
        p=s.run([wire.tensor((2,),[1,-2],1)])[0]['value'];before=metadata(s,p)
        opt,inventory=create(s,[p],momentum=.5);assert inventory==(1,1,1,0)
        absent=state(s,opt);assert absent[0]==0 and absent[1][1:]==(0,0,0)
        s.run([(72,opt)]);assert metadata(s,p)==before and state(s,opt)==absent
        seed(s,p,(2,),[2,-1]);s.run([(72,opt)])
        buf=state_tensor(s,opt);m=metadata(s,buf);assert m['version']==0 and not m['requires_grad']
        assert m['storage_id']!=before['storage_id'] and m['version_id']!=before['version_id']
        exact=values(s,p,[Fraction(3,4),Fraction(-15,8)])+values(s,buf,[2,-1])
        s.run([(72,opt)]);assert state_tensor(s,opt)==buf and metadata(s,buf)['version']==2
        exact+=values(s,p,[Fraction(3,8),Fraction(-27,16)])+values(s,buf,[3,Fraction(-3,2)])
        s.run([(73,opt,1)]);snapshot=s.owner();s.run([(72,opt)])
        assert s.run([(59,p)])[0]['value']==0 and metadata(s,buf)['version']==2
        exact+=values(s,p,[Fraction(3,8),Fraction(-27,16)])
        s.run([(41,p)]);rollback=s.error([(13,p)],1)
        renewed=s.run([(75,opt,0,0)])[0]['value'];assert renewed!=p
        now=metadata(s,renewed);assert now['storage_id']==before['storage_id'] and now['version_id']==before['version_id']
        rollback+=s.error([(13,p)],1)
        incarnation=state(s,opt)[1][0];s.run([(33,0)])
        newest=s.run([(75,opt,0,0)])[0]['value'];assert newest not in (p,renewed)
        assert state(s,opt)[1][0]==incarnation
        newbuf=state_tensor(s,opt);assert newbuf!=buf and metadata(s,newbuf)['version']==2
        seed(s,newest,(2,),[1,1]);s.run([(72,opt)])
        exact+=values(s,newbuf,[Fraction(5,2),Fraction(1,4)])
        assert metadata(s,newbuf)['version']==4
        return exact,rollback,{'real_first_state_version':0,'later_versions':[2,4],
            'closed_parameter_tokens_not_reactivated':True,'optimizer_private_roots_survive_reset':True}
    case('lazy-state-private-parameter-republication-and-reset',lazy)

    def duplicate(s):
        p=s.run([wire.tensor((),[1],1)])[0]['value'];seed(s,p,(),[2])
        opt,inventory=create(s,[p,p],momentum=.5,dampening=.25,weight_decay=.25)
        assert inventory==(1,2,1,1)
        expected_p=Fraction(1);expected_buf=None;exact=0
        for step in range(2):
            before_buf=expected_buf
            buffers=[]
            for occurrence in range(2):
                grad=Fraction(2)+Fraction(1,4)*expected_p
                if before_buf is None:current=grad
                else:
                    current=Fraction(1,2)*expected_buf+Fraction(3,4)*grad
                    expected_buf=current
                expected_p-=Fraction(1,8)*current;buffers.append(current)
            expected_buf=buffers[-1]
            s.run([(72,opt)]);buf=state_tensor(s,opt)
            assert state_tensor(s,opt,0,1)==buf
            assert metadata(s,p)['version']==2*(step+1)
            assert metadata(s,buf)['version']==(0 if step==0 else 4)
            exact+=values(s,p,[expected_p])+values(s,buf,[expected_buf])
        return exact,0,{'first_occurrences_both_snapshot_absent':True,'second_state_version':4}
    case('duplicate-occurrences-snapshot-before-functional-loop',duplicate)

    def groups(s):
        p=s.run([wire.tensor((),[1],1)])[0]['value'];q=s.run([wire.tensor((),[2],1)])[0]['value']
        opt,_=create(s,groups=[group([p],{'lr':-.125,'momentum':.5}),group([])])
        seed(s,p,(),[2]);s.run([(72,opt)]);exact=values(s,p,[Fraction(5,4)])
        s.run([(79,opt,wire.Payload(group([q],{'lr':.25,'momentum':.75})))])
        assert state(s,opt,2,0)[0]==0
        seed(s,q,(),[1]);s.run([(73,opt,1)]);assert s.run([(59,q)])[0]['value']==0
        rollback=s.error([(79,opt,wire.Payload(group([p])))],7)
        rollback+=s.error([(74,opt,3)],7)+s.error([(75,opt,1,0)],7)
        rollback+=s.error([(71,0,wire.Payload(Q(config(lr=-.125))),wire.Payload(Q([1])+group([q])))],7)
        return exact,rollback,{'negative_group_lr_accepted_default_negative_rejected':True,
            'empty_explicit_group_accepted_cross_group_duplicate_rejected':True}
    case('groups-overrides-empty-and-atomic-rejections',groups)

    def strided(s):
        base=s.run([wire.tensor((2,3),[1,-2,3,.5,2,-1])])[0]['value']
        view=s.run([(6,base,0,1)])[0]['value']
        # Transpose without a grad-producing parent has no backward node requirement;
        # detach makes its leaf identity explicit, preserving Storage/Version/strides.
        p=s.run([(19,view)])[0]['value'];before=metadata(s,p)
        assert before['strides'][:2]==[1,3]
        assert s.run([(90,p,1)])[0]['value']==p
        changed=metadata(s,p);assert changed==dict(before,requires_grad=1)
        opt,_=create(s,[p],momentum=.5,dampening=.25)
        seed(s,p,(3,2),[1,2,3,-1,-2,-3]);s.run([(72,opt)])
        buf=state_tensor(s,opt);assert metadata(s,buf)['strides'][:2]==[1,3]
        exact=values(s,buf,[1,2,3,-1,-2,-3]);assert metadata(s,buf)['version']==0
        s.run([(90,p,0)]);assert s.run([(59,p)])[0]['value']==1
        s.run([(72,opt)]);assert metadata(s,buf)['version']==2
        exact+=values(s,buf,[Fraction(5,4),Fraction(5,2),Fraction(15,4),Fraction(-5,4),Fraction(-5,2),Fraction(-15,4)])
        s.run([(73,opt,1)]);assert s.run([(59,p)])[0]['value']==0
        return exact,0,{'first_momentum_stride':[1,3],'requires_false_preserves_existing_grad_and_step_consumes_it':True}
    case('transposed-leaf-state-layout-and-requires-flag',strided)

    for path,data in inputs.items():assert path.read_bytes()==data,path
    (a.output/'executed-test.py').write_bytes(inputs[Path(__file__).resolve()])
    report['status']='PASS';save();print(json.dumps({k:report[k] for k in ('status','frames','exact_values','rollback_bytes')}))

if __name__=='__main__':main()
