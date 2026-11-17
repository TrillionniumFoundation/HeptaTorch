#!/usr/bin/env python3
"""One fixed native backward private-preserve-cache allocation failure control.

HIA executes all Tensor and autograd behavior. Python only transports commands,
decodes recorded owner state, verifies resource bounds and applies a temporary
address-space limit to its own child. No Torch or compiler executes here.
"""
import argparse, hashlib, importlib.util, json, re, resource, struct, sys, traceback
from pathlib import Path
sys.dont_write_bytecode=True
SOURCE='c436e21c45355891f8423f174a47e7c40fc8e1d6aef9a112043fac8787255d7a'
IMAGE='0799496559ffdfb051e34bc7a543db238b81d09c970a5cb4a8235151e0335b74'
ORIGINAL_DRIVER='9925244a326107793255fe8ea2681797e8f7bf00adf6d1b03022a94c6f3a3e72'
ORIGINAL_MANIFEST='a499508477bd78e6fa28fd09fab364af6eb38ebdcd56c79672d37d57f0e9f031'
ALIGN=lambda n:(n+15)&~15
PAGE=lambda n:(n+4095)&~4095
sha=lambda b:hashlib.sha256(b).hexdigest()

def guard(ok,label,**facts):
    if not ok:raise AssertionError({'resource_setup_rule':label,**facts})

def original_module(directory):
    directory=directory.resolve()
    driver=directory/'test_wrapped_scalar_resources.py'
    guard(sha(driver.read_bytes())==ORIGINAL_DRIVER,'original-driver-pin')
    guard(sha((directory/'source-manifest.json').read_bytes())==ORIGINAL_MANIFEST,'original-source-manifest-pin')
    sys.path.insert(0,str(directory))
    spec=importlib.util.spec_from_file_location('original07b_resource_driver',driver)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def slot_headroom(state):
    available={}
    for kind in range(1,9):
        rows=[struct.unpack_from('<8Q',state.raw,139264+128+((kind-1)*state.slots+s)*64)
              for s in range(state.slots)]
        available[kind]=sum(row[1]==0 and row[0]<4294967295 for row in rows)
    return available

def consume_extent_budget(spans,amount,label):
    """Read-only resource accounting for the pinned first-fit extent allocator."""
    amount=ALIGN(amount)
    for i,(offset,size) in enumerate(spans):
        if size>=amount:
            result={'label':label,'requested_rounded_bytes':amount,'offset':offset}
            if size==amount:spans.pop(i)
            else:spans[i]=[offset+amount,size-amount]
            return result
    raise AssertionError({'resource_setup_rule':'pre-target-extent-cannot-fit',
                          'label':label,'bytes':amount,'remaining':spans})

def geometry(c,x,y,seed,count):
    state=c.state();xn,xw=state.tensor_node(x);yn,yw=state.tensor_node(y)
    sn,sw=state.tensor_node(seed)
    guard(xn<yn,'leaf-precedes-output',leaf=xn,output=yn)
    guard(xw[0]==yw[0]==sw[0]==1 and xw[1]==yw[1]==sw[1]==count,'fixed-rank-count')
    guard(xw[12]==yw[12]==sw[12]==count,'fixed-shape')
    guard(xw[16]==yw[16]==1 and sw[16]==2,'noncontiguous-seed-actual-strides')
    guard(xw[72]==yw[72]==sw[72]==0,'actual-F32')
    guard(xw[4]==yw[4]==1 and sw[4]==0,'actual-requires-flags')
    guard(xw[5]==0 and yw[5]==3,'direct-wrapped-multiply-and-leaf')
    guard(yw[6]==xn and yw[9]==1 and yw[10]==0,'single-enabled-leaf-edge')
    scalar=yw[7];guard(scalar!=0,'original-Saved-scalar-present')
    scalarw=state.node(scalar)
    guard(scalarw[38]==1 and scalarw[72]==3 and scalarw[0]==0 and scalarw[1]==1,
          'actual-live-rank0-dtype3-Saved')
    guard(yw[45]==0 and not any(yw[80:84]) and not any(xw[80:84]),'no-retained-or-ViewInfo-path')
    guard(xw[30]==1 and xw[48]!=0,'existing-real-leaf-gradient')
    gradient=state.node(xw[48])
    guard(gradient[38]==1 and gradient[40]==xw[49] and gradient[72]==0,
          'real-gradient-incarnation')
    guard(all(xw[q]==0 for q in (32,56,60,66,68,69,70,71)),
          'leaf-normal-and-preserve-scratch-empty',words={str(q):xw[q] for q in (32,56,60,66,68,69,70,71)})
    guard(state.q(3984)==0 and state.q(24)<(1<<64)-1,'no-pending-mark-or-epoch-overflow')
    guard(state.next<240 and state.q(4096)!=0,'existing-descriptor-page')
    slots=slot_headroom(state)
    guard(min(slots[k] for k in (1,2,3,6))>=8,'no-pre-target-table-growth',available=slots)
    # Only op3 output and its enabled leaf are visited; the disabled dtype3
    # operand and non-requiring-grad seed/fillers are not active graph nodes.
    workspace=24+24*state.next+72+5*count
    return state,{'leaf_node':xn,'output_node':yn,'seed_node':sn,'scalar_node':scalar,
                  'gradient_node':xw[48],'gradient_incarnation':xw[49],
                  'highest_descriptor':state.next-1,'workspace_requested':workspace,
                  'workspace_rounded':ALIGN(workspace),'normal_buffer_rounded':ALIGN(4*count),
                  'preserve_target_rounded':ALIGN(4*count),'table_reusable_slots':slots,
                  'pool':state.pool,'raw_leaf_words':list(xw),'raw_output_words':list(yw),
                  'raw_seed_words':list(sw),'raw_saved_scalar_words':list(scalarw)}

def run_case(c,row,obligations,mod):
    count=65536
    c.row['target_allocation_qualified']=False
    x=c.full([count],1.0,1)
    y=c.one('actual wrapped Scalar64 multiply',114,x,int(row['scalar_binary64_hex'],16))['value']
    backing=c.full([2*count],1.0,0)
    seed=c.view(backing,[count],[2],0)
    canary=c.typed(0,[4],raw=mod.selected_words(row,'input_binary32_hex'))
    command=(58,y,seed,1)
    c.row['exact_backward_command']=list(command)
    warm=c.one('warmup existing gradient with retained graph',*command)
    c.check('warmup successful return',warm['value'],0)
    c.check('warmup complete diagnostics',len(warm['payload']),32)
    c.check('warmup semantic status',struct.unpack('<4Q',warm['payload'])[0],0)
    state,proof=geometry(c,x,y,seed,count)
    pool=state.pool
    source_sid=c.meta(x)['storage_id'];seed_sid=c.meta(seed)['storage_id']
    source_before=c.state().storage_bytes(source_sid)
    seed_before=c.state().storage_bytes(seed_sid);canary_before=c.logical(canary)
    fillers=[]
    # Only ordinary public allocations are used to consume existing free space.
    # Recompute the exact workspace length after every new descriptor.
    for _ in range(24):
        state,proof=geometry(c,x,y,seed,count)
        pre=proof['workspace_rounded']+proof['normal_buffer_rounded']
        target=proof['preserve_target_rounded'];free=sum(size for _,size in state.free())
        if pre+target//4<=free<=pre+3*target//4:break
        guard(free>pre+3*target//4,'sufficient-initial-budget',free=free,pre=pre,target=target)
        amount=min(262144,((free-(pre+target//2))//4)&~3)
        guard(amount>0,'positive-filler-size',amount=amount)
        fillers.append(c.full([amount],1.0,0))
        c.check('filler stays in original pool',c.state().pool,pool)
    state,proof=geometry(c,x,y,seed,count)
    pre=proof['workspace_rounded']+proof['normal_buffer_rounded']
    target=proof['preserve_target_rounded'];free=sum(size for _,size in state.free())
    guard(pre+target//4<=free<=pre+3*target//4,'final-free-budget',free=free,pre=pre,target=target)
    original_extents=[list(pair) for pair in state.free()]
    budget=[pair[:] for pair in original_extents]
    allocations=[consume_extent_budget(budget,proof['workspace_rounded'],'partial_execute workspace'),
                 consume_extent_budget(budget,proof['normal_buffer_rounded'],'enabled leaf normal backward buffer')]
    guard(sum(size for _,size in budget)<target and max((size for _,size in budget),default=0)<target,
          'first-capacity-growth-is-private-preserve-buffer',remaining=budget,target=target)
    proof.update(original_free_extents=original_extents,pre_target_extent_allocations=allocations,
                 remaining_extents_before_target=budget,total_free_bytes=free,
                 filler_count=len(fillers),target_function='incoming_reserve_preserve_storage',
                 target_source_sha256=obligations['functions']['incoming_reserve_preserve_storage'],
                 source_obligations=obligations['pre_target_order'])
    before=c.s.owner();old_limit=resource.prlimit(c.s.pid,resource.RLIMIT_AS)
    status=Path('/proc/%d/status'%c.s.pid).read_bytes();(c.out/'before-limit.status').write_bytes(status)
    vm=int(next(line.split()[1] for line in status.decode().splitlines() if line.startswith('VmSize:')))*1024
    headroom=PAGE(before['length']+16)+PAGE(5056+16)+65536
    soft=vm+headroom
    # storage_capacity doubles pool when the next 4*N extent does not fit.
    next_owner_bytes=len(state.raw)-state.pool+2*state.pool
    guard(headroom<PAGE(next_owner_bytes+16),'larger-owner-cannot-fit',headroom=headroom,next_owner_bytes=next_owner_bytes)
    guard(old_limit[1]==resource.RLIM_INFINITY or soft<old_limit[1],
          'temporary-limit-below-original-hard-limit',soft=soft,original=list(old_limit))
    proof.update(current_vm=vm,current_owner_bytes=before['length'],native_pid=c.s.pid,
                 original_limit=list(old_limit),temporary_soft_limit=soft,
                 permitted_transaction_headroom=headroom,first_growth_owner_bytes=next_owner_bytes)
    c.row['allocation_setup']=proof
    c.row['target_allocation_qualified']=True
    mod.dump(c.out/'allocation-setup.json',proof)
    resource.prlimit(c.s.pid,resource.RLIMIT_AS,(soft,old_limit[1]))
    try:
        (c.out/'limited.limits').write_bytes(Path('/proc/%d/limits'%c.s.pid).read_bytes())
        c.check('same-capacity control under limit',c.one('94 under constrained growth',94,x)['value'],0)
        c.refusal('backward_private_preserve_growth_OOM',[command],3,maps=True)
    finally:
        resource.prlimit(c.s.pid,resource.RLIMIT_AS,old_limit)
        c.row['original_limit_restored']=resource.prlimit(c.s.pid,resource.RLIMIT_AS)==old_limit
    c.check('OS child limit actually restored',c.row['original_limit_restored'],True)
    c.check('post-refusal native audit',c.one('audit after backward refusal',53)['value'],1)
    c.check('source bytes after refusal',c.state().storage_bytes(source_sid),source_before)
    c.check('full noncontiguous seed Storage after refusal',c.state().storage_bytes(seed_sid),seed_before)
    retry=c.one('identical backward after restoring limit',*command)
    c.check('identical-command recovery status',retry['value'],0)
    c.check('identical-command complete diagnostics',len(retry['payload']),32)
    c.check('identical-command semantic result',struct.unpack('<4Q',retry['payload'])[0],0)
    c.check('restored recovery actually grew pool',c.state().pool>pool,True)
    c.check('source bytes after successful retry',c.state().storage_bytes(source_sid),source_before)
    c.check('seed full Storage after successful retry',c.state().storage_bytes(seed_sid),seed_before)
    c.check('canary bytes after successful retry',c.logical(canary),canary_before)
    _,after_x=c.node(x)
    c.check('same existing leaf gradient root and incarnation',after_x[48:50],
            (proof['gradient_node'],proof['gradient_incarnation']))
    c.row['recovery_diagnostics']=list(struct.unpack('<4Q',retry['payload']))
    c.row['recovery_scope']='Successful identical native command and ownership/resource integrity; numeric parity remains the separately admitted original76.'
    c.reset_and_close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('resource-source','native-build','binary','protocol','reference','probe','obligations','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-native-sha256',default=IMAGE)
    p.add_argument('--expected-build-sha256',required=True)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report={'schema':'heptatorch.scalar07b.backward-private-preserve-resource.v1','status':'RUNNING',
            'cases':[],'native_executed':False,'Torch_executed':False,'compiler_executed':False,
            'source_sha256':SOURCE,'native_sha256':IMAGE,'whole_pytorch_gate':'HOLD','closed_schema_count':0,
            'numeric_oracle_computed_here':False,'original_resource3_reexecuted':False,
            'resource_scope':'One controlled native backward preserve allocation refusal plus identical-command recovery.'}
    c=None;custody=None;mod=None
    try:
        guard(a.expected_native_sha256==IMAGE,'exact-native-image')
        guard(re.fullmatch(r'[0-9a-f]{64}',a.expected_build_sha256) is not None,'actual-build-pin-format')
        mod=original_module(a.resource_source);custody=mod.Custody()
        custody.bind(Path(__file__))
        obligations=json.loads(custody.bind(a.obligations))
        guard(obligations['source_sha256']==SOURCE,'source-obligations-pin')
        wire,binary,build,row,old_evidence=mod.load_inputs(a,custody,out)
        raw=custody.bind(a.native_build/'runtime.hepta',SOURCE)
        functions={m.group(1).decode():m.group(0) for m in re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n',raw,re.M|re.S)}
        extra=out/'backward-bound-functions';extra.mkdir()
        for name,pin in obligations['functions'].items():
            guard(sha(functions[name])==pin,'exact-HIA-allocation-function',name=name)
            (extra/(name+'.hepta')).write_bytes(functions[name])
        report['bound_function_count']=len(obligations['functions'])
        report['original_native_build_sha256']=mod.BUILD
        report['native_build_sha256']=a.expected_build_sha256
        report['unit_count']=len(build['units'])
        report['function_count']=build['functions']
        if a.prepare_only:report['status']='PREPARED_NOT_RUN'
        else:
            report['native_executed']=None;report['native_start_attempted']=True
            c=mod.Case.__new__(mod.Case)
            c.__init__('backward_private_preserve_allocation_oom_recovery',wire,binary,out,report,128<<20)
            report['native_executed']=True
            run_case(c,row,obligations,mod);c.row['status']='PASS'
            report['status']='PASS'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
        if c is not None and hasattr(c,'row'):c.row.update(status='FAIL',error=repr(error))
        if c is not None and hasattr(c,'s'):
            report['native_executed']=True
            try:c.row['failed_owner']=mod.raw_file(c.out,c.s.owner()['bytes'])
            except BaseException as extra:report['failed_owner_error']=repr(extra)
    finally:
        if c is not None and hasattr(c,'s'):
            try:c.finish()
            except BaseException as error:
                report.update(status='FAIL',finish_error=repr(error))
                c.row.update(status='FAIL',finish_error=repr(error))
            mod.dump(c.out/'report.json',c.row)
        if custody is not None:
            try:report['inputs']=custody.finish();report['all_inputs_unchanged']=True
            except BaseException as error:report.update(status='FAIL',custody_error=repr(error))
        if report['status']=='PASS' and not (
                len(report['cases'])==1 and report['cases'][0]['status']=='PASS' and
                report['cases'][0].get('EOF_exit_code')==0):
            report.update(status='FAIL',final_EOF_error='complete-owned-child-EOF')
        report['case_count']=len(report['cases'])
        report['passed']=sum(row['status']=='PASS' for row in report['cases'])
        report['frames_before_EOF']=sum(row.get('frames_before_EOF',0) for row in report['cases'])
        report['checks']=sum(row.get('checks',0) for row in report['cases'])
        report['full_owner_rollback_bytes']=sum(row.get('rollback_bytes',0) for row in report['cases'])
        report['full_rng_bytes']=sum(row.get('full_rng_bytes',0) for row in report['cases'])
        report['committed_84Q_snapshots']=sum(row.get('committed_84Q_snapshots',0) for row in report['cases'])
        (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','case_count','passed','checks','frames_before_EOF','error')}))
    return int(report['status'] not in ('PASS','PREPARED_NOT_RUN'))

if __name__=='__main__':raise SystemExit(main())
