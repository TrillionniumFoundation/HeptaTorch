"""Native gradient presence, optimizer skip, output and real OOM qualification."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,signal,struct,traceback
SHA=lambda data:hashlib.sha256(data).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    for name in ('executable','protocol','output'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    inputs={p.resolve():p.read_bytes() for p in (Path(__file__),args.executable,args.protocol)}
    assert SHA(inputs[args.protocol.resolve()])=='5b99b0e508c2093fe830e509fffbb46b709f65e85b669ab6d79a32d425aa95ac'
    spec=importlib.util.spec_from_file_location('presence_wire',args.protocol)
    wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
    f=lambda values:struct.pack('<'+'f'*len(values),*values)
    words=lambda *values:struct.pack('<'+'Q'*len(values),*values)
    def full(s,shape,value,requires=0):return s.run([wire.full(shape,value,requires)])[0]['value']
    def present(s,t):
        r=s.run([(59,t)])[0];assert r['value'] in (0,1) and r['payload']==b''
        return r['value']
    def version(s,t):return wire.meta(s.run([(13,t)])[0])['version']
    def value(s,t,expected):assert s.run([(10,t)])[0]['payload']==f(expected)
    def back(s,t,seed=0,retain=0,error=0):
        r=s.run([(58,t,seed,retain)])[0]
        assert r['value']==error and struct.unpack('<4Q',r['payload'])[0]==error
        return struct.unpack('<4Q',r['payload'])
    def adam(s,p,m,v,mode,step=1):
        cfg=wire.Payload(words(wire.bits(.125),wire.bits(.5),wire.bits(.75),wire.bits(.5),wire.bits(.25),mode))
        return (24,p,m,v,step,cfg)
    def momentum(s,p,m,step=1):
        cfg=wire.Payload(words(wire.bits(.125),wire.bits(.5),0,wire.bits(.25),0))
        return (25,p,m,0,step,cfg)
    report={'schema':'heptatorch.gradient-presence-and-partial-bounds.v1','status':'RUNNING',
        'native_sha256':SHA(inputs[args.executable.resolve()]),'native_executed':True,'pytorch_executed':False,
        'inputs':{str(p):SHA(b) for p,b in inputs.items()},'rows':[],'frames':0,'rollback_bytes':0}
    def save():(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def case(name,fn,limit=64<<20):
        s=wire.Session(args.executable,directory=args.output/name,address_limit=limit)
        row={'case':name,'status':'RUNNING'};report['rows'].append(row);save()
        try:
            rollback,details=fn(s);s.run([(53,)]);s.finish()
            row.update(status='PASS',frames=s.frames,rollback_bytes=rollback,details=details,
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

    def initial_and_legacy(s):
        checked=[]
        for opcode in (7,29,57,58):
            x=full(s,(),2,1);assert present(s,x)==0
            s.run([(8,x)]);assert present(s,x)==0
            zero=full(s,(),0);y=s.run([(2,3,x,zero)])[0]['value']
            assert present(s,y)==0 and present(s,zero)==0
            command=(opcode,y) if opcode==7 else ((29,y,full(s,(),1)) if opcode==29 else (opcode,y,0,0))
            s.run([command]);assert present(s,x)==1 and present(s,y)==0
            assert s.run([(11,x)])[0]['payload']==f([0])
            s.run([(8,x)]);assert present(s,x)==1
            checked.append(opcode)
        return 0,{'backward_entrypoints_marked_real_zero_gradients':checked,'zero_grad_does_not_create_or_remove_presence':True}
    case('initial-None-and-zero-valued-grad-across-all-backward-entries',initial_and_legacy)

    def empty_and_reset(s):
        x=full(s,(0,),2,1);seed=full(s,(0,),1);assert present(s,x)==0
        detail=back(s,x,seed);assert detail[2]==1 and present(s,x)==1
        s.run([(33,1)]);assert present(s,x)==1
        y=full(s,(),1,1);assert present(s,y)==0
        large=full(s,(1024,1024),1);assert present(s,x)==1 and present(s,y)==0
        s.run([(41,large)])
        return 0,{'empty_leaf_grad_is_present':True,'presence_survives_reset_and_pool_growth':True,
                  'new_reused_node_starts_absent':True}
    case('empty-grad-reset-reuse-and-owner-migration',empty_and_reset)

    def partial_leaf(s):
        x=full(s,(),2,1);z=full(s,(),3,1);c=full(s,(),5);d=full(s,(),7)
        bad=s.run([(2,3,x,c)])[0]['value'];good=s.run([(2,3,z,d)])[0]['value']
        left=s.run([(4,7,bad)])[0]['value'];right=s.run([(4,7,good)])[0]['value']
        root=s.run([(2,1,left,right)])[0]['value'];s.run([(12,c,0,wire.bits(6))])
        detail=back(s,root,error=6);assert detail[2]==1
        assert present(s,x)==0 and present(s,z)==1
        s.run([(9,x,wire.bits(.125)),(9,z,wire.bits(.125))])
        value(s,x,[2]);value(s,z,[2.125]);assert version(s,x)==0 and version(s,z)==1
        return 0,{'unready_leaf_presence':0,'ready_leaf_presence':1,'unready_parameter_unchanged':2,
                  'ready_parameter_after_SGD':2.125,'parameter_versions':[0,1]}
    case('partial-leaf-presence-controls-SGD-update',partial_leaf)

    def optimizer_absent(s):
        p=full(s,(),2,1);m=full(s,(),.5);v=full(s,(),.25)
        assert present(s,p)==0
        for call in ((9,p,wire.bits(.125)),momentum(s,p,m),adam(s,p,m,v,0),adam(s,p,m,v,1)):
            s.run([call]);value(s,p,[2]);value(s,m,[.5]);value(s,v,[.25])
            assert [version(s,t) for t in (p,m,v)]==[0,0,0]
        badcfg=wire.Payload(words(wire.bits(.125),wire.bits(.5),wire.bits(.75),wire.bits(.5),wire.bits(.25),2))
        rollback=s.error([(24,p,m,v,1,badcfg)],7)
        rollback+=s.error([(9,p,wire.bits(-1))],7)
        assert present(s,p)==0
        return rollback,{'all_four_optimizers_skip_None':True,'nonzero_state_preserved':[.5,.25],
                         'versions_unchanged':[0,0,0],'invalid_configuration_not_hidden_by_None':True}
    case('None-skips-SGD-momentum-Adam-and-AdamW',optimizer_absent)

    def optimizer_zero(s):
        expected=[]
        for mode in ('sgd','momentum','adam','adamw'):
            p=full(s,(),2,1);zero=full(s,(),0);y=s.run([(2,3,p,zero)])[0]['value'];back(s,y)
            assert present(s,p)==1 and s.run([(11,p)])[0]['payload']==f([0])
            m=full(s,(),0);v=full(s,(),0)
            if mode=='sgd':call=(9,p,wire.bits(.125));pv=1;mv=0;vv=0;target=2
            elif mode=='momentum':call=momentum(s,p,m);pv=1;mv=1;vv=0;target=1.9375
            else:call=adam(s,p,m,v,mode=='adamw');pv=2 if mode=='adamw' else 1;mv=1;vv=2;target=1.9375
            s.run([call]);value(s,p,[target]);assert [version(s,t) for t in (p,m,v)]==[pv,mv,vv]
            expected.append({'optimizer':mode,'parameter':target,'versions':[pv,mv,vv]})
        return 0,{'present_zero_gradient_still_updates_and_decays':expected}
    case('present-zero-gradient-is-not-None',optimizer_zero)

    def admission_presence(s):
        x=full(s,(),2,1);y=s.run([(2,3,x,x)])[0]['value']
        rollback=s.error([(58,y,0,0),(99,)],4);assert present(s,x)==0
        rollback+=s.error([(57,y,0,0),(99,)],4);assert present(s,x)==0
        back(s,y);assert present(s,x)==1
        rollback+=s.error([(59,1)],1)
        return rollback,{'presence_rolls_back_with_rejected_frame':True,'invalid_public_handle_rejected':True}
    case('presence-is-transactional-and-typed',admission_presence)

    def output_bound(s):
        x=full(s,(),2,1);y=s.run([(2,3,x,x)])[0]['value']
        commands=[(13,x)]*6553+[(8,x),(58,y,0,0)]
        rollback=s.error(commands,3);assert present(s,x)==0
        back(s,y);assert present(s,x)==1
        assert s.run([(11,x)])[0]['payload']==f([4])
        return rollback,{'record_header_fits_but_32_byte_execution_payload_does_not':True,
                         'Saved_and_gradient_presence_unchanged_on_refusal':True}
    case('short-result-capacity-before-execution',output_bound)

    def allocation_bound(s):
        x=full(s,(256,512),2,1);one=full(s,(),1)
        y=s.run([(2,1,x,one)])[0]['value'];loss=s.run([(4,7,y)])[0]['value'];back(s,loss)
        large=full(s,(1024,1024),1)
        fillers=[full(s,(256,512),1) for _ in range(4)]
        assert present(s,x)==1
        rollback=s.error([(58,loss,0,0)],3)
        s.read_constant(x,131072,1,gradient=True);assert present(s,x)==1
        s.run([(41,large)]);back(s,loss)
        s.read_constant(x,131072,2,gradient=True)
        return rollback,{'actual_RLIMIT_AS_bytes':32<<20,'work_allocation_failed_before_node_execution':True,
                         'entire_owner_unchanged':True,'recovered_after_free':True,'gradient_values_compared':262144}
    case('real-OOM-before-execution-then-recovery',allocation_bound,32<<20)

    for p,data in inputs.items():assert p.read_bytes()==data,p
    report.update(status='PASS',cases=len(report['rows']))
    save();print(json.dumps({k:report[k] for k in ('status','cases','frames','rollback_bytes','native_sha256')},indent=2))

if __name__=='__main__':main()
