"""Native nullable-gradient storage, rematerialization and rollback tests.

The tests import only a pinned wire/transport helper. Raw owner inspection is
read-only and checks documented relative offsets, never writes native state.
The numerical examples use exact binary32 words or exact dyadic arithmetic.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,signal,struct,traceback

SHA=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','protocol','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--expected-source-sha256',required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    inputs={p.resolve():p.read_bytes() for p in (Path(__file__),args.binary,args.source,args.protocol)}
    assert SHA(inputs[args.binary.resolve()])==args.expected_sha256
    assert SHA(inputs[args.source.resolve()])==args.expected_source_sha256
    assert SHA(inputs[args.protocol.resolve()])=='5b99b0e508c2093fe830e509fffbb46b709f65e85b669ab6d79a32d425aa95ac'
    spec=importlib.util.spec_from_file_location('nullable_wire',args.protocol)
    wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
    pack=lambda values:struct.pack('<'+'I'*len(values),*values)
    def tensor(s,shape,values,requires=0):
        return s.run([(1,len(shape),requires,*shape,*([0]*(4-len(shape))),wire.Payload(pack(values),4))])[0]['value']
    def full(s,shape,value,requires=0):return s.run([wire.full(shape,value,requires)])[0]['value']
    def presence(s,t):return s.run([(59,t)])[0]['value']
    def meta(s,t):return s.run([(13,t)])[0]['payload']
    def stats(s):return struct.unpack('<12Q',s.run([(54,)])[0]['payload'])
    def grad(s,t,expected):
        actual=s.run([(11,t)])[0]
        assert actual['value']==len(expected) and actual['payload']==pack(expected)
        return len(expected)
    def nodes(s):
        raw=s.owner()['bytes'];q=lambda at:struct.unpack_from('<Q',raw,at)[0]
        assert raw[64:72]==b'HTSTATE6'
        base=139264+144+424*q(139264+16);result={}
        for i in range(1,q(8)):
            page=q(4096+(i>>8)*8);assert page>>63
            p=base+(page&((1<<63)-1))+(i&255)*384
            if q(p+304):
                record={off:q(p+off) for off in (8,24,32,40,160,192,216,224,248,256,336,360,368,376)}
                assert [record[k] for k in (336,368,376)]==[0,0,0]
                if record[40]!=0 or record[360]==0:
                    assert record[24]==record[248]==0,(i,record)
                else:
                    assert record[32]==1 and record[360]==1 and record[24]>>63 and record[248]
                result[record[192]]=record
        return result
    report={'schema':'heptatorch.nullable-gradient-native-qualification.v1','status':'RUNNING',
            'native_executable_sha256':args.expected_sha256,'native_source_sha256':args.expected_source_sha256,
            'input_sha256':{str(p):SHA(b) for p,b in inputs.items()},'native_executed':True,'pinned_Torch_executed':False,
            'rows':[],'frames':0,'exact_gradient_words':0,'rollback_bytes':0,'committed_node_checks':0}
    def save():(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def case(name,body,limit=64<<20):
        s=wire.Session(args.binary,directory=args.output/name,address_limit=limit)
        row={'case':name,'status':'RUNNING'};report['rows'].append(row);save()
        try:
            words,rollback,checks,details=body(s);s.run([(53,)]);checks+=len(nodes(s));s.finish()
            row.update(status='PASS',frames=s.frames,exact_gradient_words=words,rollback_bytes=rollback,
                       committed_node_checks=checks,details=details,all_responses_before_stdin_eof=True)
            for key in ('frames','exact_gradient_words','rollback_bytes','committed_node_checks'):report[key]+=row[key]
        except BaseException as error:
            row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc());report['status']='FAIL'
            if s.poll() is None:os.kill(s.pid,signal.SIGKILL)
            try:s.finish(expected=-signal.SIGKILL)
            except BaseException:pass
            save();raise
        finally:(args.output/name/'assertions.json').write_text(json.dumps(row,indent=2)+'\n');save()

    def initial(s):
        x=full(s,(4,),2,1);m=meta(s,x);n=nodes(s);checks=len(n)
        assert n[x][248]==0 and presence(s,x)==0
        before=stats(s);s.run([(8,x)]);assert presence(s,x)==0
        checks+=len(nodes(s));count=grad(s,x,[0]*4)
        read=s.run([(40,x,1,2)])[0];assert read['payload']==pack([0,0]);count+=2
        s.run([(70,x)]);s.run([(70,x)]);after=stats(s)
        assert before==after and m==meta(s,x);checks+=len(nodes(s))
        assert presence(s,x)==0
        return count,0,checks,{'constructor_gradient_hold':0,'clear_None_and_zero_None_do_not_allocate':True,
                               'compatibility_grad_read_zero_with_presence_false':True}
    case('fresh-None-idempotent-clear-and-zero-readout',initial)

    def rematerialization(s):
        count=0;checks=0;details=[]
        for entry in (7,29,57,58):
            x=full(s,(),2,1);coefficient=tensor(s,(),[0x80000000]);y=s.run([(2,3,x,coefficient)])[0]['value']
            seed=full(s,(),1);m=meta(s,x)
            call=(7,y) if entry==7 else ((29,y,seed) if entry==29 else (entry,y,seed,1))
            s.run([call]);count+=grad(s,x,[0x80000000]);a=nodes(s);checks+=len(a)
            old_hold=a[x][248];old_stats=stats(s)
            s.run([(70,x)]);assert presence(s,x)==0;after=nodes(s);checks+=len(after)
            new_stats=stats(s)
            assert new_stats[6]==old_stats[6]-1 and new_stats[10]==old_stats[10]-1
            assert new_stats[4]==old_stats[4]+16 and m==meta(s,x)
            s.run([(8,x)]);assert presence(s,x)==0
            s.run([call]);count+=grad(s,x,[0x80000000]);b=nodes(s);checks+=len(b)
            assert b[x][248]!=old_hold and presence(s,x)==1 and m==meta(s,x)
            s.run([(8,x)]);count+=grad(s,x,[0]);assert presence(s,x)==1
            s.run([call]);count+=grad(s,x,[0]);checks+=len(nodes(s))
            details.append({'backward_entry':entry,'old_hold':old_hold,'new_hold':b[x][248],
                            'clear_releases_one_Storage_and_Hold':True,'first_rebuilt_gradient_word':0x80000000})
        return count,0,checks,details
    case('four-backward-entries-rebuild-storage-and-preserve-negative-zero',rematerialization)

    def empty(s):
        x=full(s,(0,),2,1);seed=full(s,(0,),1);before=stats(s)
        s.run([(58,x,seed,0)]);assert presence(s,x)==1
        first=nodes(s);checks=len(first);assert first[x][24]>>63 and first[x][248]
        defined=stats(s);s.run([(70,x)]);cleared=stats(s)
        assert presence(s,x)==0 and cleared[4]==defined[4]
        assert cleared[6]==defined[6]-1 and cleared[10]==defined[10]-1
        checks+=len(nodes(s));s.run([(58,x,seed,0)]);assert presence(s,x)==1
        checks+=len(nodes(s));assert grad(s,x,[])==0
        return 0,0,checks,{'zero_length_gradient_has_real_owner_when_defined':True,
                           'clear_changes_object_counts_without_free_byte_delta':True}
    case('empty-gradient-owner-clear-and-rebuild',empty)

    def failures(s):
        x=full(s,(),2,1);constant=full(s,(),3);loss=s.run([(2,3,x,constant)])[0]['value']
        s.run([(58,loss,0,1)]);count=grad(s,x,[wire.bits(3)])
        rollback=s.error([(70,x),(999,)],4);assert presence(s,x)==1
        count+=grad(s,x,[wire.bits(3)]);checks=len(nodes(s))
        for command,code in [((70,1),1),((70,constant),5),((70,loss),5),((70,x,1),7),((70,x,0,0,0,0,0,1),7)]:
            rollback+=s.error([command],code)
        seed=full(s,(),1)
        rollback+=s.error([(70,x),(58,x,seed,0),(999,)],4)
        assert presence(s,x)==1;count+=grad(s,x,[wire.bits(3)]);checks+=len(nodes(s))
        # A runtime exception has different commit semantics from admission.
        s.run([(12,constant,0,wire.bits(4))])
        raw=s.exchange(wire.packet([(70,x),(58,loss,0,0),(999,)]));status,records=wire.parse(raw)
        assert status==0 and len(records)==2 and records[1]['value']==6
        assert presence(s,x)==0;count+=grad(s,x,[0]);checks+=len(nodes(s))
        return count,rollback,checks,{'failed_frame_restores_released_gradient':True,
                                      'runtime_error_commits_prefix_clear_and_does_not_run_suffix':True}
    case('admission-validation-and-clear-transaction-boundaries',failures)

    def partial_failure(s):
        x=full(s,(),2,1);z=full(s,(),3,1);c=full(s,(),5);d=full(s,(),7)
        bad=s.run([(2,3,x,c)])[0]['value'];good=s.run([(2,3,z,d)])[0]['value']
        a=s.run([(4,7,bad)])[0]['value'];b=s.run([(4,7,good)])[0]['value']
        root=s.run([(2,1,a,b)])[0]['value'];s.run([(12,c,0,wire.bits(6))])
        result=s.run([(58,root,0,0)])[0];assert result['value']==6
        assert presence(s,x)==0 and presence(s,z)==1
        n=nodes(s);checks=len(n);assert n[x][248]==n[x][24]==0
        count=grad(s,x,[0])+grad(s,z,[wire.bits(7)])
        s.run([(70,z)]);n=nodes(s);checks+=len(n);assert presence(s,z)==0
        s.run([(58,z,0,0)]);count+=grad(s,z,[wire.bits(1)]);checks+=len(nodes(s))
        return count,0,checks,{'unready_leaf_reservation_released':True,'completed_leaf_keeps_real_gradient':True,
                               'completed_leaf_can_be_cleared_and_rebuilt_after_partial_failure':True}
    case('partial-error-cleans-only-unmaterialized-reserves',partial_failure)

    def large_oom(s):
        x=full(s,(256,512),2,1);loss=s.run([(4,7,x)])[0]['value']
        s.run([(58,loss,0,0)]);s.run([(70,x)]);assert presence(s,x)==0
        large=full(s,(1024,1024),1);fillers=[full(s,(256,1024),1) for _ in range(2)]
        checks=len(nodes(s));rollback=s.error([(58,loss,0,0)],3)
        assert presence(s,x)==0 and nodes(s)[x][248]==0;checks+=len(nodes(s))
        s.run([(41,fillers[-1])]);s.run([(58,loss,0,0)])
        assert presence(s,x)==1;s.read_constant(x,131072,1,gradient=True)
        checks+=len(nodes(s));s.run([(70,x)]);assert presence(s,x)==0;checks+=len(nodes(s))
        return 131072,rollback,checks,{'real_address_limit_bytes':32<<20,'entire_owner_unchanged_after_failed_rebuild':True,
                                      'rebuild_succeeds_after_unrelated_storage_release':True,'large_exact_gradient_values':131072}
    case('real-allocation-failure-during-gradient-recreation',large_oom,limit=32<<20)
    for p,data in inputs.items():assert p.read_bytes()==data,p
    (args.output/'executed-test.py').write_bytes(inputs[Path(__file__).resolve()])
    report['status']='PASS';save();print(json.dumps({k:report[k] for k in ('status','frames','exact_gradient_words','rollback_bytes','committed_node_checks')}))

if __name__=='__main__':main()
