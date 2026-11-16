"""Native default-backward lifetime checks with exact dyadic references."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,struct,traceback

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--executable',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest()=='5b99b0e508c2093fe830e509fffbb46b709f65e85b669ab6d79a32d425aa95ac'
    spec=importlib.util.spec_from_file_location('release_protocol',args.protocol)
    wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
    tensor,full,bits,Payload,meta=wire.tensor,wire.full,wire.bits,wire.Payload,wire.meta
    f=lambda values:struct.pack('<'+'f'*len(values),*values)
    words=lambda *values:struct.pack('<'+'Q'*len(values),*values)
    def stats(s):return struct.unpack('<12Q',s.run([(54,)])[0]['payload'])
    report={'schema':'heptatorch.default-backward-native.v1','status':'RUNNING',
            'native_sha256':hashlib.sha256(args.executable.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
            'cases':[],'frames':0,'rollback_bytes':0,'pytorch_executed':False}
    def save(): (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def case(name,body,limit=64<<20):
        s=wire.Session(args.executable,directory=args.output/name,address_limit=limit)
        row={'name':name,'status':'RUNNING'};report['cases'].append(row);save()
        try:
            rollback,detail=body(s)
            s.finish();row.update(status='PASS',frames=s.frames,rollback_bytes=rollback,
                                 detail=detail,stdin_open_before_every_response=True)
            report['frames']+=s.frames;report['rollback_bytes']+=rollback
        except BaseException as error:
            row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc());report['status']='FAIL'
            save();raise
        finally:
            (args.output/name/'assertions.json').write_text(json.dumps(row,indent=2)+'\n');save()

    def released_result(s):
        x=s.run([tensor((3,),[-2,1,3],1)])[0]['value']
        y=s.run([(3,5,x)])[0]['value'];square=s.run([(2,3,y,y)])[0]['value']
        loss=s.run([(4,7,square)])[0]['value'];weak=s.run([(42,y)])[0]['value']
        s.run([(41,y),(41,square)])
        held=s.run([(43,weak)])[0]['value'];s.run([(45,held)])
        s.run([(57,loss,0,0),(53,)])
        assert s.run([(11,x)])[0]['payload']==f([0,2,6])
        after=stats(s);assert after[9]==0 and after[10]==2,after
        rollback=s.error([(43,weak)],6)
        rollback+=s.error([(57,loss,0,1)],10)
        rollback+=s.error([(7,loss)],10)
        seed=s.run([full((),1)])[0]['value'];rollback+=s.error([(29,loss,seed)],10)
        assert s.run([(11,x)])[0]['payload']==f([0,2,6])
        s.run([(41,loss),(44,weak),(53,)])
        return rollback,{'expected_leaf_gradient':[0,2,6],'remaining_holds':'one descriptor page plus persistent leaf gradient'}
    case('saved-output-freed-and-legacy-reentry-refused',released_result)

    def metadata_only(s):
        x=s.run([tensor((2,2),[1,2,3,4],1)])[0]['value']
        one=s.run([full((),1)])[0]['value'];y=s.run([(2,1,x,one)])[0]['value']
        t=s.run([(6,y,0,1)])[0]['value'];c=s.run([(18,t)])[0]['value']
        loss=s.run([(4,7,c)])[0]['value']
        for step,command in enumerate([(57,loss,0,0),(57,loss,0,0),(7,loss),(57,loss,0,1),(57,loss,0,0)],1):
            s.run([command,(53,)])
            assert s.run([(11,x)])[0]['payload']==f([step]*4)
        info=meta(s.run([(13,x)])[0]);assert info['version']==0
        after=stats(s);assert after[9]==0 and after[10]==2,after
        return 0,{'backward_calls':5,'expected_leaf_gradient':[5]*4,'no_saved_graph_reallocated_work':True}
    case('add-clone-transpose-repeat-after-work-release',metadata_only)

    def leaf(s):
        x=s.run([full((),3,1)])[0]['value'];seed=s.run([full((),-2)])[0]['value']
        for call,expected in [((57,x,0,0),1),((57,x,seed,0),-1),((7,x),0)]:
            s.run([call]);assert s.run([(11,x)])[0]['payload']==f([expected])
        return 0,{'leaf_accumulator_persisted':True}
    case('leaf-only-repeated-backward',leaf)

    def retain_sequence(s):
        x=s.run([tensor((2,),[1,2],1)])[0]['value'];factor=s.run([tensor((2,),[3,4])])[0]['value']
        y=s.run([(2,3,x,factor)])[0]['value'];loss=s.run([(4,7,y)])[0]['value']
        weak=s.run([(42,factor)])[0]['value'];s.run([(41,factor),(41,y)])
        for n,retain in enumerate([1,1,0],1):
            s.run([(57,loss,0,retain)])
            assert s.run([(11,x)])[0]['payload']==f([3*n,4*n])
            if retain:
                h=s.run([(43,weak)])[0]['value'];s.run([(45,h)])
        rollback=s.error([(43,weak)],6)+s.error([(57,loss,0,1)],10)
        return rollback,{'retain_sequence':[True,True,False,True],'final_call_status':10,'expected_gradient':[9,12]}
    case('retained-coefficient-then-default-release',retain_sequence)

    def public_saved(s):
        x=s.run([tensor((2,),[-1,2],1)])[0]['value'];y=s.run([(3,5,x)])[0]['value']
        loss=s.run([(4,7,y)])[0]['value'];saved=s.run([(46,y)])[0]['value'];weak=s.run([(42,y)])[0]['value']
        s.run([(41,y),(57,loss,0,0),(53,)])
        assert stats(s)[9]==1
        assert s.run([(49,saved,0,2)])[0]['payload']==f([0,2])
        alias=s.run([(56,saved)])[0]['value']
        h=s.run([(43,weak)])[0]['value'];s.run([(45,h)])
        rollback=s.error([(57,loss,0,0)],10)
        s.run([(12,alias,0,bits(5))]);rollback+=s.error([(47,saved)],6)
        s.run([(48,saved),(41,alias),(53,)])
        rollback+=s.error([(43,weak)],6)
        return rollback,{'public_saved_independent_of_AD_release':True}
    case('public-saved-remains-after-private-release',public_saved)

    def indexed_self(s):
        for code in [36,37]:
            x=s.run([tensor((2,3),list(range(6)),1)])[0]['value']
            source=s.run([full((2,2),1)])[0]['value']
            config=Payload(words(2,2,2,2,0,1,1)) if code==36 else Payload(words(1,2,2,0))
            call=(36,x,source,1,config) if code==36 else (37,x,source,1,config,bits(.5))
            y=s.run([call])[0]['value'];loss=s.run([(4,7,y)])[0]['value'];s.run([(41,source)])
            s.run([(57,loss,0,0),(53,),(57,loss,0,0),(53,)])
            assert s.run([(11,x)])[0]['payload']==f([2]*6)
        return 0,{'opcodes':[36,37],'self_only_second_backward_succeeded_after_index_storage_release':True}
    case('indexed-add-self-only-no-saved-repeat',indexed_self)

    def empty_indices(s):
        x=s.run([tensor((3,),[1,2,3],1)])[0]['value']
        y=s.run([(30,x,0,0,Payload(b''))])[0]['value'];seed=s.run([full((0,),0)])[0]['value']
        s.run([(57,y,seed,0)]);assert s.run([(11,x)])[0]['payload']==f([0,0,0])
        rollback=s.error([(57,y,seed,0)],10)
        a=s.run([full((2,3),0)])[0]['value'];b=s.run([full((2,0),0,1)])[0]['value']
        result=s.run([(36,a,b,1,Payload(words(2,2,0)))])[0]['value'];loss=s.run([(4,7,result)])[0]['value']
        s.run([(57,loss,0,0)]);rollback+=s.error([(57,loss,0,0)],10)
        return rollback,{'empty_index_select_and_scatter_source_saved_roles_rejected_second_call':True}
    case('empty-saved-index-roles-still-release',empty_indices)

    def branches(s):
        x=s.run([tensor((2,),[1,2],1)])[0]['value'];one=s.run([full((),1)])[0]['value']
        shared=s.run([(2,3,x,x)])[0]['value'];left=s.run([(2,1,shared,one)])[0]['value']
        right=s.run([(2,1,shared,one)])[0]['value'];a=s.run([(4,7,left)])[0]['value'];b=s.run([(4,7,right)])[0]['value']
        unvisited=s.run([(2,3,x,x)])[0]['value'];c=s.run([(4,7,unvisited)])[0]['value']
        assert stats(s)[9]==4
        s.run([(57,a,0,0)]);assert stats(s)[9]==2
        rollback=s.error([(57,b,0,1)],10)
        s.run([(57,c,0,0)]);assert stats(s)[9]==0
        assert s.run([(11,x)])[0]['payload']==f([4,8])
        return rollback,{'shared_saved_spent':True,'unvisited_saved_remained_usable':True}
    case('shared-and-unvisited-roots',branches)

    def atomic(s):
        x=s.run([tensor((2,),[1,2],1)])[0]['value'];y=s.run([(2,3,x,x)])[0]['value'];loss=s.run([(4,7,y)])[0]['value']
        rollback=s.error([(57,loss,0,0),(99,)],4)
        assert stats(s)[9]==2
        assert s.run([(11,x)])[0]['payload']==f([0,0])
        rollback+=s.error([(57,loss,0,2)],7)
        seed=s.run([full((2,),1)])[0]['value'];rollback+=s.error([(57,loss,seed,0)],2)
        s.run([(57,loss,0,1),(57,loss,0,0),(53,)])
        assert s.run([(11,x)])[0]['payload']==f([4,8])
        rollback+=s.error([(57,loss,0,0)],10)
        return rollback,{'successful_release_then_later_frame_error_rolled_back':True}
    case('entire-frame-release-and-gradient-rollback',atomic)

    def allocation(s):
        x=s.run([full((256,512),1,1)])[0]['value'];one=s.run([full((),1)])[0]['value']
        y=s.run([(2,1,x,one)])[0]['value'];loss=s.run([(4,7,y)])[0]['value']
        s.run([(57,loss,0,0)])
        filler=s.run([full((1024,1024),0)])[0]['value']
        s.run([full((512,1024),0)])
        before=stats(s);assert before[1]==8<<20,before
        rollback=s.error([(57,loss,0,0)],3)
        assert s.run([(40,x,0,131072)])[0]['payload']==f([1]*131072)
        s.run([(41,filler),(57,loss,0,0),(53,)])
        assert s.run([(40,x,0,131072)])[0]['payload']==f([2]*131072)
        return rollback,{'RLIMIT_AS':32<<20,'actual_try_new_failure_on_work_rehydration':True,
                         'retry_after_releasing_filler_succeeded':True,'elements_checked_each_time':131072}
    case('work-rehydration-allocation-failure-and-recovery',allocation,32<<20)
    assert hashlib.sha256(args.executable.read_bytes()).hexdigest()==report['native_sha256']
    report['status']='PASS';save()
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))

if __name__=='__main__':main()
