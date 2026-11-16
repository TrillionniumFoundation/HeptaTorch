"""Independent dyadic ownership controls for native owned Adam/AdamW."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, signal, struct, traceback
from fractions import Fraction

SHA=lambda data:hashlib.sha256(data).hexdigest()
Q=lambda words:struct.pack('<'+'Q'*len(words),*words)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','protocol','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--expected-sha256',required=True)
    ap.add_argument('--expected-source-sha256',required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    snapshots={p.resolve():p.read_bytes() for p in (Path(__file__),a.binary,a.source,a.protocol)}
    assert SHA(snapshots[a.binary.resolve()])==a.expected_sha256
    assert SHA(snapshots[a.source.resolve()])==a.expected_source_sha256
    assert SHA(snapshots[a.protocol.resolve()])=='5b99b0e508c2093fe830e509fffbb46b709f65e85b669ab6d79a32d425aa95ac'
    spec=importlib.util.spec_from_file_location('owned_adam_wire',a.protocol)
    wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
    report={'schema':'heptatorch.owned-adam-native-controls.v1','status':'RUNNING',
        'source_sha256':a.expected_source_sha256,'native_sha256':a.expected_sha256,
        'native_executed':True,'Torch_executed_here':False,'cases':[],
        'frames':0,'exact_values':0,'rollback_bytes':0,
        'inputs_sha256':{str(p):SHA(b) for p,b in snapshots.items()}}
    def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def config(lr=.125,beta1=0,beta2=0,eps=0,weight_decay=0,maximize=0):
        return [wire.bits(x) for x in (lr,beta1,beta2,eps,weight_decay)]+[0,maximize,0]
    def group(ids,mask=0,words=None):
        return Q([len(ids),mask]+(words or [0]*8)+list(ids))
    def create(s,ids,algorithm=1,groups=None,**kwargs):
        groups=groups or [group(ids)]
        record=s.run([(71,algorithm,wire.Payload(Q(config(**kwargs))),
                       wire.Payload(Q([len(groups)])+b''.join(groups)))])[0]
        return record['value'],struct.unpack('<4Q',record['payload'])
    def metadata(s,t):return wire.meta(s.run([(13,t)])[0])
    def inventory(s,opt,g=0,p=0):
        record=s.run([(76,opt,g,p)])[0]
        assert len(record['payload'])==64
        return record['value'],struct.unpack('<8Q',record['payload'])
    def states(s,opt,g=0,p=0):
        return tuple(row['value'] for row in s.run([(77,opt,g,p,field) for field in (1,2,3)]))
    def values(s,t,expected):
        row=s.run([(10,t)])[0]
        target=struct.pack('<'+'f'*len(expected),*[float(x) for x in expected])
        assert row['value']==len(expected) and row['payload']==target,(row,target.hex())
        return len(expected)
    def seed(s,p,shape,vals):
        tensor=s.run([wire.tensor(shape,vals)])[0]['value']
        result=s.run([(58,p,tensor,0)])[0]
        assert result['value']==0
        return tensor
    def case(name,body):
        s=wire.Session(a.binary,directory=a.output/name,address_limit=128<<20)
        row={'case':name,'status':'RUNNING'};report['cases'].append(row);save()
        try:
            exact,rollback,details=body(s)
            s.run([(53,)]);s.finish()
            row.update(status='PASS',frames=s.frames,exact_values=exact,rollback_bytes=rollback,
                       all_responses_before_stdin_eof=True,details=details)
            for k in ('frames','exact_values','rollback_bytes'):report[k]+=row[k]
        except BaseException as error:
            row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
            report['status']='FAIL';save()
            if s.poll() is None:os.kill(s.pid,signal.SIGKILL)
            try:s.finish(expected=-signal.SIGKILL)
            except BaseException:pass
            raise
        finally:
            (a.output/name/'assertions.json').write_text(json.dumps(row,indent=2)+'\n');save()

    def lazy(s):
        p=s.run([wire.tensor((2,),[1,-2],1)])[0]['value']
        before=metadata(s,p);opt,inv=create(s,[p]);assert inv==(1,1,1,0)
        absent=inventory(s,opt);assert absent[0]==0 and absent[1][1:]==(0,0,0,0,0,0,0)
        assert states(s,opt)==(0,0,0)
        s.run([(72,opt)]);assert metadata(s,p)==before and inventory(s,opt)==absent
        seed(s,p,(2,),[2,-1]);s.run([(72,opt)])
        step,avg,sq=states(s,opt);assert len(set((p,step,avg,sq)))==4
        metas=[metadata(s,t) for t in (p,step,avg,sq)]
        assert [m['version'] for m in metas]==[1,1,1,2]
        assert len({m['storage_id'] for m in metas})==4 and len({m['version_id'] for m in metas})==4
        assert metas[1]['rank']==0 and metas[1]['numel']==1
        assert all(m['requires_grad']==0 for m in metas[1:])
        exact=values(s,p,[Fraction(7,8),Fraction(-15,8)])+values(s,step,[1])
        exact+=values(s,avg,[2,-1])+values(s,sq,[4,1])
        assert inventory(s,opt)[1][1]==14
        s.run([(90,p,0),(72,opt)])
        assert states(s,opt)==(step,avg,sq)
        assert [metadata(s,t)['version'] for t in (p,step,avg,sq)]==[2,2,2,4]
        exact+=values(s,p,[Fraction(3,4),Fraction(-7,4)])+values(s,step,[2])
        s.run([(95,step,wire.bits(.5)),(95,avg,wire.bits(.5))])
        assert metadata(s,step)['version']==3 and metadata(s,avg)['version']==3
        exact+=values(s,step,[Fraction(5,2)])+values(s,avg,[Fraction(5,2),Fraction(-1,2)])
        rollback=s.error([(72,opt)],7)
        s.run([(95,step,wire.bits(-.5)),(72,opt)])
        assert metadata(s,step)['version']==5
        exact+=values(s,p,[Fraction(5,8),Fraction(-13,8)])+values(s,step,[3])
        s.run([(73,opt,1)])
        assert s.run([(59,p)])[0]['value']==0
        unchanged=[metadata(s,t) for t in (p,step,avg,sq)]
        s.run([(72,opt)])
        assert [metadata(s,t) for t in (p,step,avg,sq)]==unchanged
        incarnations=inventory(s,opt)[1][::2]
        storage_versions=[(m['storage_id'],m['version_id']) for m in unchanged]
        s.run([(33,0)])
        fresh=s.run([(75,opt,0,0)])[0]['value']
        newstates=states(s,opt)
        assert all(x!=y for x,y in zip((fresh,)+newstates,(p,step,avg,sq)))
        assert inventory(s,opt)[1][::2]==incarnations
        assert [(metadata(s,t)['storage_id'],metadata(s,t)['version_id']) for t in (fresh,)+newstates]==storage_versions
        rollback+=s.error([(13,p)],1)+s.error([(13,step)],1)
        return exact,rollback,{'true_scalar_step_and_moment_objects':True,
            'None_skips_all_fields':True,'float32_step_is_actual_counter':True,
            'fractional_resume_explicitly_refused':True,'public_reset_tokens_fresh_same_objects':True}
    case('lazy-state-real-counter-public-alias-and-reset',lazy)

    def duplicates(s):
        p=s.run([wire.tensor((),[1],1)])[0]['value'];seed(s,p,(),[2])
        opt,inv=create(s,[p,p]);assert inv==(1,2,1,1)
        exact=0;st=None
        for k in (1,2):
            s.run([(72,opt)])
            current=states(s,opt);assert states(s,opt,0,1)==current
            if st is not None:assert st==current
            st=current
            assert [metadata(s,t)['version'] for t in (p,)+st]==[2*k,2*k,2*k,4*k]
            exact+=values(s,p,[Fraction(1)-Fraction(k,4)])+values(s,st[0],[2*k])
            exact+=values(s,st[1],[2])+values(s,st[2],[4])
        cfg=[wire.bits(.125),wire.bits(.5),0,0,0,0]
        sg=s.run([(71,0,wire.Payload(Q(cfg)),wire.Payload(Q([1,1,0]+[0]*6+[p])))])[0]['value']
        s.run([(72,sg)])
        momentum=s.run([(77,sg,0,0,0)])[0]['value']
        exact+=values(s,p,[Fraction(1,4)])+values(s,momentum,[2])
        assert metadata(s,momentum)['version']==0
        s.run([(72,opt)])
        exact+=values(s,p,[0])+values(s,st[0],[6])
        assert metadata(s,p)['version']==7 and metadata(s,st[2])['version']==12
        s.run([(78,sg),(78,opt)])
        exact+=values(s,st[1],[2])+values(s,momentum,[2])
        return exact,0,{'one_shared_Adam_state_per_key_before_functional_loop':True,
            'ordered_duplicate_updates':True,'SGD_and_Adam_share_parameter_without_state_coalescing':True}
    case('duplicate-shared-state-and-cross-algorithm-parameter',duplicates)

    def layout(s):
        base=s.run([wire.tensor((2,3),[1,-2,3,.5,2,-1])])[0]['value']
        p=s.run([(6,base,0,1)])[0]['value'];s.run([(90,p,1)])
        before=metadata(s,p);assert before['strides'][:2]==[1,3]
        seed(s,p,(3,2),[1,2,3,-1,-2,-3])
        opt,_=create(s,[p],algorithm=2,weight_decay=.25)
        s.run([(72,opt)]);step,avg,sq=states(s,opt)
        assert metadata(s,avg)['strides'][:2]==[1,3] and metadata(s,sq)['strides'][:2]==[1,3]
        assert metadata(s,p)['version']==2 and metadata(s,base)['version']==2
        expected=[Fraction(27,32),Fraction(23,64),Fraction(-33,16),Fraction(33,16),Fraction(97,32),Fraction(-27,32)]
        exact=values(s,p,expected)+values(s,avg,[1,2,3,-1,-2,-3])+values(s,sq,[1,4,9,1,4,9])
        s.run([(95,p,wire.bits(.5))])
        assert metadata(s,p)['version']==3 and metadata(s,base)['version']==3
        exact+=values(s,p,[x+Fraction(1,2) for x in expected])
        e=s.run([wire.full((0,3),0,1)])[0]['value'];seed(s,e,(0,3),[])
        emptyopt,_=create(s,[e],algorithm=2,weight_decay=-0.0)
        s.run([(72,emptyopt)])
        es,em,ev=states(s,emptyopt)
        assert [metadata(s,t)['version'] for t in (e,es,em,ev)]==[1,1,1,2]
        exact+=values(s,es,[1])+values(s,em,[])+values(s,ev,[])
        s.run([(95,e,wire.bits(.5))])
        assert metadata(s,e)['version']==2
        s.run([(73,emptyopt,1),(72,emptyopt)])
        assert [metadata(s,t)['version'] for t in (e,es,em,ev)]==[2,1,1,2]
        return exact,0,{'strided_moment_physical_layout_and_no_grad_add':True,
            'AdamW_nonzero_decay_two_versions_negative_zero_one':True,
            'defined_empty_participates_None_empty_skips':True}
    case('strided-physical-state-empty-and-shared-Version',layout)

    for path,data in snapshots.items():assert path.read_bytes()==data,path
    (a.output/'executed-test.py').write_bytes(snapshots[Path(__file__).resolve()])
    report['status']='PASS';save()
    print(json.dumps({k:report[k] for k in ('status','frames','exact_values','rollback_bytes')}))

if __name__=='__main__':main()
