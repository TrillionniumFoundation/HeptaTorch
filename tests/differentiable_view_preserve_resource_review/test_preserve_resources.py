"""Three independent native R3 preserve-resource controls; no Torch execution.

The old six recipes are preserved but not run. The old Case transport and every
84Q committed-state predicate are retained; only profile identities change.
This is committed-state verification, not the legacy isolated-transition gate.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import struct
import traceback

from case_transport import Case as OriginalCase
from owner_state import State
import owner_view05 as observer
from support import Custody, dump, sha, raw_file, typed_bytes

SOURCE='3c2c0a125bcac6d47075bf58fc3f78c0e7b9a0b1370757981b99dcd3793b21c6'
IMAGE='7841b19051960b6fa13cb98841dbb8e4d6ead97f03e2ad65af415e1a35b03221'
PROFILE='7f06102c97fae80643b57a737112272bb3897b46659bd35a0343436adcf59836'
REFERENCE='f4abe6a750ecc791b3a7bc7aa507b4c77eb0a2fcccf45b0c6a828db0e01490d3'
PROBE='cf569335f48624c1edf2f97a649200f2e792ebcef5b6f1df6bac73293ce02efc'
PROTOCOL='7a30d7bca140e3a4cc9d63a3e6930b92d2ffd60ad4503895419d8a63e67c8bd9'
CLIENT='dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
COMPILER='b1c851a0896fa6071f800e1146c3c27d0c6cf85059e9a6281b46d42908a3f59f'
LINKER='aae650e8822ef0b1e74d15108ad1d0f17ec791205ed5a158ec60b709f48f0e16'
HERE=Path(__file__).resolve().parent
ALIGN=lambda n:(n+15)&~15
PAGE=lambda n:(n+4095)&~4095

def load_inputs(a,custody,out):
    manifest=json.loads(custody.bind(HERE/'source-manifest.json'))
    for name,record in manifest['files'].items():
        raw=custody.bind(HERE/name,record['sha256'])
        assert len(raw)==record['bytes'] and (HERE/name).stat().st_mode&511==record['mode']
    profile=json.loads(custody.bind(a.profile,PROFILE))
    assert profile['functions']==627 and profile['native_units']==13 and profile['descriptor_bytes']==672
    assert profile['source_sha256']==SOURCE and profile['native_sha256']==IMAGE
    native=a.native_build.resolve();image=native/'tensor-differentiable-view-session'
    authored=custody.bind(native/'runtime.hepta',SOURCE)
    custody.bind(image,IMAGE)
    info=json.loads(custody.bind(native/'build.json'))
    assert info['status']=='PASS' and info['source_sha256']==SOURCE and info['native_sha256']==IMAGE
    assert info['state_format']=='HTVAD003' and info['backing_format']=='HTOVAD03'
    assert info['descriptor_bytes']==672 and info['functions']==627 and len(info['units'])==13
    assert info['link_exit']==0 and info['compiler_sha256']==COMPILER and info['linker_sha256']==LINKER
    assert len({u['name'] for u in info['units']})==13
    assert [u['name'] for u in info['units']]==[u['name'] for u in profile['units']]
    for unit,pinned in zip(info['units'],profile['units']):
        assert unit['compiler_exit']==0
        assert all(unit[k]==pinned[k] for k in ('name','source_sha256','object_sha256'))
        for suffix,key in (('.hepta','source_sha256'),('.o','object_sha256')):
            raw=custody.bind(native/(unit['name']+suffix),unit[key]);assert len(raw)==pinned['source_bytes' if suffix=='.hepta' else 'object_bytes']
    functions={m.group(1).decode():m.group(0) for m in re.finditer(rb'^fn ([A-Za-z_][A-Za-z0-9_]*)\(.*?^end\n',authored,re.M|re.S)}
    assert len(functions)==627
    required=('gradient_workspace_size','history_prepare_leaf','partial_execute','ensure_backward_work',
        'incoming_reserve_preserve_storage','incoming_drop_cache','dview_preserve_capacity','dview_trim_preserve')
    source_dir=out/'bound-functions';source_dir.mkdir()
    for name in required:(source_dir/(name+'.hepta')).write_bytes(functions[name])
    dump(source_dir/'manifest.json',{n:sha(functions[n]) for n in required})
    for n,pin in {'partial_execute':'11e3c5ef304443c63444ddccb6305b77607cea6c59ca7ca709932e7d79014fbb',
         'ensure_backward_work':'6bc14c91e736f304bb671d8c491e5901a559df45671f1897574ebb1fd0b842b9',
         'incoming_reserve_preserve_storage':'f9018f7560ff7d047e55f800d7926fed1e87910384b03a1415fce7d00c431436',
         'incoming_drop_cache':'166f2578ea3e7453eaa4c692dbca0da37e54fe7cee790878babde4301d23e9d4'}.items():
        assert sha(functions[n])==pin
    protocol=a.protocol.resolve();custody.bind(protocol,PROTOCOL);custody.bind(protocol.with_name('owner_client.py'),CLIENT)
    spec=importlib.util.spec_from_file_location('r3_resource_protocol',protocol)
    wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
    ref=json.loads(custody.bind(a.reference,REFERENCE))
    assert ref['status']=='OBSERVED' and len(ref['rows'])==4 and ref['probe_sha256']==PROBE
    assert ref['probe_error_count']==0 and ref['source_files_unchanged'] is True
    checks=ref['oracle_provenance']['checks'];assert ref['oracle_provenance']['status']=='PASS'
    assert set(checks)=={'python_matches','platform_matches','version_backend_matches','git_matches_oracle_build_pin',
        'distribution_version_matches','direct_url_matches','direct_url_archive_sha256_matches','pip_report_url_matches',
        'pip_report_archive_sha256_matches','pip_report_version_matches','pip_report_requested'} and all(v is True for v in checks.values())
    for h,s in ref['states'].items():
        raw=bytes.fromhex(s['hex']);assert len(raw)==s['bytes']==5056 and sha(raw)==h
    return wire,image,info,ref,functions

def reference_gradient(ref,row,role):
    event=next(e for e in ref['rows'][row]['events'] if e['label']=='backward_phase_0_finite')
    assert event['accepted'] is True
    return event['after']['tensors'][role]['gradient']['tensor']

class Case(OriginalCase):
    def mark(self,label):
        snap=super().mark(label)
        state=self.state();state.partition()
        self.row['committed_84Q_snapshots']=self.row.get('committed_84Q_snapshots',0)+1
        if hasattr(self,'rng0') and self.gen in state.table(7):
            self.check('full RNG at '+label,state.rng(self.gen),self.rng0)
            self.row['full_rng_compared_bytes']=self.row.get('full_rng_compared_bytes',0)+5056
        return snap
    def full(self,shape,value=1.,requires=0):
        return self.one('real full Tensor',*self.w.full(shape,value,requires))['value']
    def audit(self,label):
        self.check(label+' native audit',self.one(label,53)['value'],1)
    def logical(self,t):
        n=self.meta(t)['numel'];return self.read(t,n)
    def expected_gradient(self,t,expected,label):
        m=self.meta(t);r=len(expected['shape'])
        self.check(label+' shape',m['shape'][:r],expected['shape'])
        self.check(label+' strides',m['strides'][:r],expected['strides'])
        self.check(label+' rank',m['rank'],r)
        self.check(label+' Version',m['version'],expected['version'])
        self.check(label+' byte extent',m['storage_bytes'],expected['storage_bytes'])
        self.check(label+' storage offset',m['storage_offset'],expected['storage_offset_bytes'])
        self.check(label+' requires',m['requires_grad'],int(expected['requires_grad']))
        self.check(label+' logical words',self.read(t,m['numel']),bytes.fromhex(expected['logical_bytes_hex']))
        self.check(label+' entire physical Storage',self.state().storage_bytes(m['storage_id']),bytes.fromhex(expected['storage_raw_hex']))
    def backward(self,t,seed,retain=0):
        rec=self.one('actual op44 backward',58,t,seed,retain)
        values=struct.unpack('<4Q',rec['payload'])
        self.check('backward header value',rec['value'],0);self.check('backward error',values[0],0)
        self.row.setdefault('backward_results',[]).append(list(values))
        return values
    def make_graph(self,shape,strides,offset,outshape,outstrides,outoffset,backing_count):
        backing=self.full([backing_count],1.)
        tmp=self.view(backing,shape,strides,offset)
        x=self.one('detach into an independent own leaf',19,tmp)['value']
        self.one('release temporary true view',41,tmp)
        self.one('explicit leaf requires',90,x,1)
        q=self.view(x,outshape,outstrides,outoffset)
        self.check('actual child op44',self.node(q)[1][5],44)
        self.check('own leaf has no true base',self.node(x)[1][80:84],(0,0,0,0))
        return backing,x,q
    def clear(self,*tensors):
        for t in tensors:self.one('clear actual G binding',92,t,0)
    def reset_and_close(self):
        self.one('release all public graph roots',33,0);self.closed()

def tail_release_reuse(c,ref,functions):
    original=ref['rows'][0]
    c.check('original input shape',original['input_size'],[2,3])
    c.check('original input strides',original['input_stride'],[0,1])
    c.check('original output shape',original['view_size'],[3])
    canary=c.typed(0,[8],raw=bytes.fromhex('0000803f0000004000004040000080400000a0400000c0400000e04000000041'))
    canary_raw=c.logical(canary)
    backing,x,q=c.make_graph([2,3],[0,1],0,[3],[1],0,3)
    c.one('retain real output gradient',93,q)
    expected_x=reference_gradient(ref,0,'input');expected_q=reference_gradient(ref,0,'view')
    words=original['seed_phases'][0]['words'];assert len(words)==3
    seed=c.typed(0,[3],raw=struct.pack('<3I',*words))
    baseline=c.state();baseline_counts=baseline.counts();baseline_free=baseline.free()
    c.row['preserve_geometry']={'capacity':24,'capacity_rounded':32,'span':12,'span_rounded':16,'released_tail':16,
        'qualification':'Source-bound actual op44 call path plus complete committed owner/extent checks; no intermediate-PC sampling.'}
    c.backward(q,seed,1)
    g1=c.one('hold first real input gradient',91,x)['value'];v1=c.one('hold first retained output gradient',91,q)['value']
    c.expected_gradient(g1,expected_x,'first input G');c.expected_gradient(v1,expected_q,'first view G')
    sid1=c.meta(g1)['storage_id'];weak=c.one('Weak first G Storage',42,g1)['value']
    c.clear(x,q)
    c.backward(q,seed,0)
    g2=c.one('hold independent second input gradient',91,x)['value'];v2=c.one('hold independent second view gradient',91,q)['value']
    c.expected_gradient(g2,expected_x,'second input G');c.expected_gradient(v2,expected_q,'second view G')
    c.check('new G has independent live Storage',c.meta(g2)['storage_id']!=sid1,True)
    c.expected_gradient(g1,expected_x,'old held input G remains unchanged')
    hold=c.one('old held G remains lockable',43,weak)['value'];c.one('release lock Hold',45,hold)
    c.clear(x,q)
    for g in (g2,v2,g1,v1):c.one('release explicitly held G',41,g)
    c.refusal('first_G_expired',[(43,weak)],6)
    c.one('release Weak tombstone',44,weak)
    c.check('no leftover live resource rows',c.state().counts(),baseline_counts)
    c.check('all trimmed and temporary extents returned',c.state().free(),baseline_free)
    c.check('canary physical bytes unchanged',c.logical(canary),canary_raw)
    c.audit('post-tail-release');c.reset_and_close()

def reachable(snapshot,root):
    seen=set();todo=[root]
    while todo:
        n=todo.pop()
        if n in seen:continue
        seen.add(n)
        todo.extend(p for p,enabled in snapshot['edges'][str(n)] if p and enabled)
    return seen

def preserve_capacity_oom_recovery(c,ref,functions):
    k=131072
    canary=c.typed(0,[4],[13.,-7.,5.,3.]);canary_raw=c.logical(canary)
    backing,x,q=c.make_graph([2,2],[k,1],0,[2,2],[k,1],1,k+3)
    expected_x=reference_gradient(ref,2,'input');expected_q=reference_gradient(ref,2,'view')
    words=ref['rows'][2]['seed_phases'][0]['words'];assert len(words)==4
    seed=c.typed(0,[2,2],raw=struct.pack('<4I',*words))
    initial=c.state();pool=initial.pool
    backing_sid=c.meta(backing)['storage_id'];original_backing=c.state().storage_bytes(backing_sid)
    fillers=[]
    for _ in range(8):
        state=c.state();free=sum(n for _,n in state.free())
        if 60*1024<=free<=68*1024:break
        assert free>68*1024,('SETUP_FAIL free too small',free)
        count=min(262144,((free-64*1024)//4)&~3)
        assert 0<count<16777216
        fillers.append(c.full([count],1.))
        c.check('filling existing extents cannot grow pool',c.state().pool,pool)
    state=c.state();snap=c.mark('bounded preserve-OOM setup');high=state.next-1
    root=c.node(q)[0];nodes=reachable(snap,root)
    assert nodes and high<128
    assert all(0<state.node(n)[1]<=4 for n in nodes)
    assert all(state.row(state.node(n)[24],1)[7]==0 for n in nodes if state.node(n)[30])
    # Source-derived conservative bound is finalized with the frozen budget proof.
    budget=json.loads((HERE/'oom-budget.json').read_text())
    for name,pin in budget['source_sha256'].items():assert sha(functions[name])==pin
    assert all(state.node(n)[5] in (0,44) for n in nodes)
    capacities={n:4*state.node(n)[1] for n in nodes}
    for child in nodes:
        cn=state.node(child)
        if cn[5]!=44:continue
        pairs=snap['edges'][str(child)];assert len(pairs)==2 and pairs[1]==[0,0] and pairs[0][1]==1
        parent=pairs[0][0];assert parent in nodes
        pn=state.node(parent)
        def end(d):return d[29]+4+sum((d[12+a]-1)*d[16+a]*4 for a in range(d[0]))
        span=max(end(cn),end(pn))-min(cn[29],pn[29])
        capacities[parent]=max(capacities[parent],span)
    assert sorted(v for v in capacities.values() if v>16)==[4*(k+3)]
    assert all(v<=16 or v==4*(k+3) for v in capacities.values())
    workspace_bound=24+24*(high+1)+budget['workspace_scratch_upper']
    small_bound=ALIGN(workspace_bound)+budget['per_small_extent_upper']*3*len(nodes)
    free=sum(size for _,size in state.free());preserve=ALIGN(4*(k+3))
    assert 60*1024<=free<=68*1024 and small_bound<free<preserve
    assert max(size for _,size in state.free())>=small_bound
    remaining={kind:state.slots-len(state.table(kind)) for kind in range(1,9)}
    assert min(remaining.values())>=3*len(nodes)+4 and (high+1)%256+len(nodes)<256
    assert all(state.q(4096+8*((high+1+i)>>8))!=0 for i in range(len(nodes)))
    assert len(nodes)<=4 and len(fillers)<8
    c.row['oom_setup']={'highest':high,'reachable_nodes':sorted(nodes),'reachable_numels':{str(n):state.node(n)[1] for n in nodes},
        'pool':pool,'free':free,'preserve_requested':4*(k+3),'preserve_rounded':preserve,'workspace_upper':workspace_bound,
        'all_pre_preserve_small_allocations_upper':small_bound,'table_free_slots':remaining,'page_headroom':256-(high+1)%256,
        'candidate_preserve_capacities':capacities,'filler_count':len(fillers),'setup_budget_source':budget}
    owner=c.s.owner();original_limit=resource.prlimit(c.s.pid,resource.RLIMIT_AS)
    status_path=Path('/proc/%d/status'%c.s.pid)
    before_status=status_path.read_bytes();(c.out/'before-limit.status').write_bytes(before_status)
    current_vm=int(next(line.split()[1] for line in before_status.decode().splitlines() if line.startswith('VmSize:')))*1024
    soft=current_vm+PAGE(owner['length']+16)+PAGE(5056+16)+65536
    assert original_limit[1]==resource.RLIM_INFINITY or soft<original_limit[1]
    c.row['oom_setup'].update(original_rlimit=list(original_limit),current_vm=current_vm,soft_limit=soft)
    resource.prlimit(c.s.pid,resource.RLIMIT_AS,(soft,original_limit[1]))
    try:
        (c.out/'limited.limits').write_bytes(Path('/proc/%d/limits'%c.s.pid).read_bytes())
        c.check('same-capacity owner query still succeeds',c.one('query under limited address space',94,q)['value'],0)
        c.refusal('preserve_growth_OOM',[(58,q,seed,0)],3,maps=True)
    finally:
        resource.prlimit(c.s.pid,resource.RLIMIT_AS,original_limit)
        c.row['original_limit_restored']=resource.prlimit(c.s.pid,resource.RLIMIT_AS)==original_limit
    c.check('failed backward leaves leaf G absent',c.one('query absent G after real OOM',91,x)['value'],0)
    c.audit('post-OOM')
    c.backward(q,seed,0)
    g=c.one('hold recovered G',91,x)['value'];c.expected_gradient(g,expected_x,'recovered G uses same two-row mapping')
    c.check('restored limit permits actual pool growth',c.state().pool>pool,True)
    c.check('entire original backing Storage unchanged',c.state().storage_bytes(backing_sid),original_backing)
    c.check('canary unchanged across failed and successful trial',c.logical(canary),canary_raw)
    c.clear(x);c.one('drop recovered G handle',41,g)
    c.audit('post-recovery');c.reset_and_close()

def tail_then_saved_error_prefix(c,ref,functions):
    x=c.typed(0,[3],[2.,3.,4.],requires=1);factor=c.typed(0,[3],[3.,5.,7.])
    m=c.one('actual upstream Saved multiplication',2,3,x,factor)['value']
    q=c.view(m,[2,3],[0,1],0);t=c.view(q,[3],[1],0)
    c.one('retain intermediate overlapping view',93,q);c.one('retain output view',93,t)
    seed=c.typed(0,[3],[1.,1.,1.]);marker=c.typed(0,[1],[7.])
    c.one('invalidate upstream Saved Version',95,factor,0x3f800000)
    snap=c.mark('pre-bound-error-traversal');start=c.node(t)[0];path=[];current=start
    for operation in (44,44,3):
        path.append(current);c.check('pre-state operation',snap['nodes'][str(current)][5],operation)
        if operation==44:
            edges=snap['edges'][str(current)];c.check('single real enabled view parent',edges[1],[0,0]);c.check('enabled view parent',edges[0][1],1);current=edges[0][0]
    c.row['pre_error_path']={'node_ids':path,'operations':[snap['nodes'][str(n)][5] for n in path],'payload_fourth_field':'failing_operation_code_from_descriptor_offset40_not_node_ordinal'}
    expected_partial=(6,len(path)-1,0,snap['nodes'][str(path[-1])][5])
    raw=c.s.exchange(c.w.packet([(95,marker,0x3f800000),(58,t,seed,0),(95,marker,0x42c80000)]))
    status,records=c.w.parse(raw)
    c.check('partial normal header',status,0);c.check('suffix stopped',[r['opcode'] for r in records],[95,58])
    c.check('Saved failure result value',records[-1]['value'],6)
    result=struct.unpack('<4Q',records[-1]['payload'])
    c.check('two completed op44 nodes, no leaf, failing Mul',result,expected_partial)
    c.row['partial_result']=list(result);c.mark('partial_saved_error_after_two_trimmed_views')
    c.check('prefix executed once',c.logical(marker),struct.pack('<f',8.))
    c.check('marker Version once',c.meta(marker)['version'],1)
    gq=c.one('publish committed q prehook G',91,q)['value'];gt=c.one('publish committed t prehook G',91,t)['value']
    c.check('q dyadic gradient control',c.logical(gq),struct.pack('<6f',*([.5]*6)))
    c.check('t seed gradient control',c.logical(gt),struct.pack('<3f',1.,1.,1.))
    for name,g in [('q',gq),('t',gt)]:c.check(name+' prehook G Version0',c.meta(g)['version'],0)
    c.check('unvisited leaf retains None',c.one('leaf gradient after upstream failure',91,x)['value'],0)
    c.audit('partial commit and unvisited cleanup')
    c.refusal('atomic57_same_Saved_failure',[(95,marker,0x3f800000),(57,t,seed,0),(95,marker,0x42c80000)],6)
    c.check('atomic failure keeps committed marker',c.logical(marker),struct.pack('<f',8.))
    c.check('atomic failure keeps marker Version',c.meta(marker)['version'],1)
    c.one('drop public q G',41,gq);c.one('drop public t G',41,gt)
    c.reset_and_close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-build','profile','protocol','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--case',choices=('all','tail_release_reuse','preserve_capacity_oom_recovery','tail_then_saved_error_prefix'),default='all')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    custody=Custody();report={'schema':'heptatorch.dview-r3-independent-preserve-resources.v1','status':'RUNNING',
        'source_sha256':SOURCE,'native_sha256':IMAGE,'reference_sha256':REFERENCE,'cases':[],
        'native_executed':False,'Torch_executed_here':False,'whole_pytorch_gate':'HOLD','closed_schemas':0,
        'old_six_recipes_executed':False,'isolated_transition_gate_executed':False,'requested_case':a.case,
        'scope':'Three new resource compositions, full committed 84Q owner accounting. Exact existing finite reference words only where explicitly named; no new Torch observations.'}
    try:
        wire,image,build,ref,functions=load_inputs(a,custody,out)
        if a.prepare_only:
            report['status']='PREPARED_NOT_RUN'
        else:
            for f in (tail_release_reuse,preserve_capacity_oom_recovery,tail_then_saved_error_prefix):
                if a.case!='all' and a.case!=f.__name__:continue
                c=None
                try:
                    c=Case.__new__(Case);report['native_executed']=True
                    c.__init__(f.__name__,wire,image,out,report,128<<20)
                    f(c,ref,functions);c.row['status']='PASS'
                except BaseException as e:
                    row=c.row if c and hasattr(c,'row') else {'name':f.__name__}
                    if not c or not hasattr(c,'row'):report['cases'].append(row)
                    row.update(status='FAIL',error=repr(e),traceback=traceback.format_exc())
                    if c and hasattr(c,'s'):
                        try:row['failed_owner']=raw_file(c.out,c.s.owner()['bytes'])
                        except BaseException as secondary:row['failed_owner_read_error']=repr(secondary)
                finally:
                    if c and hasattr(c,'s'):
                        try:c.finish()
                        except BaseException as e:c.row.update(status='FAIL',transport_error=repr(e),transport_traceback=traceback.format_exc())
                        dump(c.out/'report.json',c.row)
                    dump(out/'report.json',report)
            report['status']='PASS' if len(report['cases'])==(3 if a.case=='all' else 1) and all(r['status']=='PASS' for r in report['cases']) else 'FAIL'
    except BaseException as e:report.update(status='FAIL',preparation_error=repr(e),traceback=traceback.format_exc())
    try:report['inputs']=custody.finish();report['all_inputs_unchanged']=True
    except BaseException as e:report.update(status='FAIL',custody_error=repr(e))
    report.update(case_count=len(report['cases']),passed=sum(r['status']=='PASS' for r in report['cases']),
        frames_before_EOF=sum(r.get('frames_before_EOF',0) for r in report['cases']),checks=sum(r.get('checks',0) for r in report['cases']),
        full_owner_rollback_bytes=sum(r.get('rollback_bytes',0) for r in report['cases']),
        committed_84Q_snapshots=sum(r.get('committed_84Q_snapshots',0) for r in report['cases']),
        full_rng_compared_bytes=sum(r.get('full_rng_compared_bytes',0) for r in report['cases']))
    dump(out/'report.json',report)
    print(json.dumps({k:report.get(k) for k in ('status','case_count','passed','frames_before_EOF','checks','preparation_error')}))
    return int(report['status'] not in ('PASS','PREPARED_NOT_RUN'))

if __name__=='__main__':raise SystemExit(main())
