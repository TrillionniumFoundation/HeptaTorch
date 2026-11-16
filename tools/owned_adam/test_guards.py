"""Independent Adam wire/capacity/owner-rollback controls; no Torch import."""
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
    for name in ('binary','source','protocol','build-report','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    snapshots={p.resolve():p.read_bytes() for p in (Path(__file__),a.binary,a.source,a.protocol,a.build_report)}
    build=json.loads(snapshots[a.build_report.resolve()])
    assert SHA(snapshots[a.binary.resolve()])==build['native_sha256']
    assert SHA(snapshots[a.source.resolve()])==build['source_sha256']
    assert SHA(snapshots[a.protocol.resolve()])=='5b99b0e508c2093fe830e509fffbb46b709f65e85b669ab6d79a32d425aa95ac'
    spec=importlib.util.spec_from_file_location('adam_guard_wire',a.protocol)
    wire=importlib.util.module_from_spec(spec);spec.loader.exec_module(wire)
    default=[wire.bits(.125),0,0,0,0,0,0,0]
    def group(ids,mask=0,words=None):return Q([len(ids),mask]+(words or [0]*8)+list(ids))
    def create(groups,config=default,algorithm=1):
        return (71,algorithm,wire.Payload(Q(config)),wire.Payload(Q([len(groups)])+b''.join(groups)))
    def meta(s,t):return wire.meta(s.run([(13,t)])[0])
    def state(s,opt,field=1,group=0,occurrence=0):
        return s.run([(77,opt,group,occurrence,field)])[0]['value']
    def seed(s,p,shape,values):
        t=s.run([wire.tensor(shape,values)])[0]['value'];assert s.run([(58,p,t,0)])[0]['value']==0
    report={'schema':'heptatorch.owned-adam-guards.v1','status':'RUNNING',
        'native_sha256':build['native_sha256'],'source_sha256':build['source_sha256'],
        'native_executed':True,'Torch_executed_here':False,'cases':[],
        'frames':0,'rejections':0,'rollback_bytes':0,
        'inputs_sha256':{str(p):SHA(data) for p,data in snapshots.items()}}
    def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def case(name,body,limit=64<<20):
        s=wire.Session(a.binary,directory=a.output/name,address_limit=limit)
        row={'case':name,'status':'RUNNING','rejections':0,'rollback_bytes':0}
        report['cases'].append(row);save()
        def reject(commands,status):
            row['rollback_bytes']+=s.error(commands,status);row['rejections']+=1
        try:
            row['details']=body(s,reject);s.run([(53,)]);s.finish()
            row.update(status='PASS',frames=s.frames,all_responses_before_stdin_eof=True)
            for key in ('frames','rejections','rollback_bytes'):report[key]+=row[key]
        except BaseException as error:
            row.update(status='FAIL',error=repr(error));report['status']='FAIL';save();raise
        finally:save()

    def malformed(s,reject):
        p=s.run([wire.tensor((2,),[1,-2],1)])[0]['value']
        opt=s.run([create([group([p])])])[0]['value']
        nongrad=s.run([(2,1,p,p)])[0]['value']
        for field in (0,1,2,3,4):
            for bits in (wire.bits(-.125),0x7fc00001):
                bad=list(default);bad[field]=bits;reject([create([group([p])],bad)],7)
        for field in (1,2):
            for bits in (wire.bits(1),0x7f800000):
                bad=list(default);bad[field]=bits;reject([create([group([p])],bad)],7)
        for field,value in ((0,1<<32),(5,1),(6,2),(7,1)):
            bad=list(default);bad[field]=value;reject([create([group([p])],bad)],7)
        reject([create([group([p])],algorithm=3)],7)
        reject([create([])],7)
        reject([create([group([p],128)])],7)
        reject([create([group([p],0,[1]+[0]*7)])],7)
        reject([create([group([p],1,[1<<32]+[0]*7)])],7)
        reject([create([group([p],32,[0]*5+[1,0,0])])],7)
        reject([create([group([p]),group([p])])],7)
        reject([create([group([nongrad])])],5)
        reject([create([group([opt])])],1)
        reject([(72,opt,1)],7);reject([(73,opt,2)],7)
        reject([(74,opt,1)],7);reject([(75,opt,0,1)],7)
        reject([(76,opt,1,0)],7);reject([(77,opt,0,0,0)],7);reject([(77,opt,0,0,4)],7)
        reject([(71,1,96,32)],100);reject([(71,1,97,104)],100);reject([(79,opt,96)],100)
        reject([(72,p)],1)
        reject([(95,p,1<<32)],7);reject([(95,p,0,1)],7);reject([(95,opt,0)],1)
        reject([(95,p,wire.bits(.5)),(999,)],4)
        assert meta(s,p)['version']==0
        for field in (0,3,4):
            cfg=list(default);cfg[field]=0x7f800000
            admitted=s.run([create([group([])],cfg)])[0]['value'];s.run([(72,admitted),(78,admitted)])
        cfg=list(default);cfg[0]=0x80000000;cfg[1]=0x80000000;cfg[2]=0x80000000;cfg[3]=0x80000000;cfg[4]=0x80000000
        admitted=s.run([create([group([])],cfg)])[0]['value'];s.run([(78,admitted)])
        seed(s,p,(2,),[2,-1])
        negative=s.run([create([group([p],1,[wire.bits(-.125)]+[0]*7)])])[0]['value']
        s.run([(72,negative)])
        assert s.run([(10,p)])[0]['payload']==struct.pack('<2f',1.125,-2.125)
        s.run([(78,negative),(78,opt)])
        reject([(72,opt)],1)
        return {'exact_constructor_comparisons_NaNs_infinity_negative_zero':True,
            'typed_reserved_payload_and_membership_rejections':True,
            'per_group_negative_lr_uses_native_update':True,
            '95_single_Version_and_later_error_full_rollback':True}
    case('configuration-types-payload-and-no-grad-mutation',malformed)

    def capacities(s,reject):
        p=s.run([wire.full((0,),0)])[0]['value']
        row=s.run([create([group([p]*4096)])])[0]
        assert struct.unpack('<4Q',row['payload'])==(1,4096,1,4095)
        opt=row['value'];inv=s.run([(74,opt,0)])[0]
        assert inv['value']==4096 and len(inv['payload'])==80+4096*16
        pairwords=struct.unpack('<'+'Q'*(4096*2),inv['payload'][80:])
        assert len(set(pairwords[::2]))==1 and all(x==p for x in pairwords[1::2])
        s.run([(72,opt)]);assert state(s,opt,3,0,4095)==0
        reject([create([group([p]*4097)])],7)
        reject([create([group([p]*4096),group([p])])],3)
        reject([(79,opt,wire.Payload(group([]))),(79,opt,wire.Payload(group([p])))],3)
        s.run([(78,opt)])
        groups=[group([]) for _ in range(1024)]
        row=s.run([create(groups)])[0];opt=row['value']
        assert struct.unpack('<4Q',row['payload'])==(1024,0,0,0)
        assert s.run([(74,opt,1023)])[0]['value']==0
        s.run([(72,opt),(73,opt,1)])
        reject([create(groups+[group([])])],7)
        reject([(79,opt,wire.Payload(group([])))],3)
        s.run([(78,opt)])
        return {'exact_occurrence_boundary':4096,'exact_group_boundary':1024,
            'one_beyond_and_late_append_frame_rollback':True}
    case('exact-configuration-capacities',capacities)

    def references(s,reject):
        records=s.run([wire.full((),1,1),create([group([0xfe000001])]),(72,0xfe000002),(76,0xfe000002,0,0)])
        p,opt=records[0]['value'],records[1]['value']
        assert (p>>24)&255==1 and (opt>>24)&255==8
        assert records[2]['value']==1 and records[3]['value']==0
        assert struct.unpack('<8Q',records[3]['payload'])[1:]==(0,0,0,0,0,0,0)
        reject([(72,opt),(95,0xfe000001,wire.bits(.5))],1)
        reject([create([group([0xfe000002])])],1)
        reject([(79,opt,wire.Payload(group([0xfe000001])))],1)
        seed(s,p,(),[2]);s.run([(72,opt)])
        avg=state(s,opt,2)
        rows=s.run([(77,opt,0,0,2),(95,0xfe000001,wire.bits(.5)),(10,0xfe000002)])
        assert rows[0]['value']==avg and rows[1]['value']==avg and rows[2]['payload']==struct.pack('<f',2.5)
        saved=s.run([(46,avg)])[0]['value'];s.run([(95,avg,0)])
        reject([(47,saved)],6);reject([(49,saved,0,0)],6)
        s.run([(48,saved)])
        return {'same_frame_Tensor_Optimizer_and_mutation_results_are_real_handles':True,
            'scalar_or_future_result_cannot_forge_Tensor':True,
            'zero_add_still_invalidates_public_Saved_once':True}
    case('typed-result-references-and-public-Saved',references)

    def memory(s,reject):
        p=s.run([wire.tensor((2,),[1,-2],1)])[0]['value'];seed(s,p,(2,),[2,-1])
        opt=s.run([create([group([p])])])[0]['value']
        before=s.owner()
        reject([(72,opt),wire.full((262144,64),7)],3)
        assert s.owner()==before and state(s,opt,1)==0
        assert meta(s,p)['version']==0
        s.run([(72,opt)]);step,avg,sq=[state(s,opt,i) for i in (1,2,3)]
        before=s.owner()
        reject([(95,avg,wire.bits(.5)),(72,opt),wire.full((262144,64),7)],3)
        assert s.owner()==before
        assert [meta(s,t)['version'] for t in (p,step,avg,sq)]==[1,1,1,2]
        assert s.run([(10,p)])[0]['payload']==struct.pack('<2f',.875,-1.875)
        s.run([(72,opt)])
        assert s.run([(10,p)])[0]['payload']==struct.pack('<2f',.75,-1.75)
        assert s.run([(10,step)])[0]['payload']==struct.pack('<f',2)
        return {'real_RLIMIT_AS':64<<20,'legal_requested_data_bytes':64<<20,
            'lazy_creation_and_existing_state_mutation_both_rolled_back':True,
            'next_success_reads_original_actual_step':True}
    case('real-allocation-failure-with-lazy-and-existing-state',memory)

    def private_growth(s,reject):
        parameters=[row['value'] for row in s.run([wire.full((0,3),0,1) for _ in range(300)])]
        empty_seed=s.run([wire.full((0,3),0)])[0]['value']
        assert all(row['value']==0 for row in s.run([(58,p,empty_seed,0) for p in parameters]))
        opt=s.run([create([group(parameters)])])[0]['value']
        before=s.owner()['length'];s.run([(72,opt)]);after=s.owner()['length']
        assert after>before
        last=parameters[-1];handles=[state(s,opt,i,0,299) for i in (1,2,3)]
        identities=[meta(s,t) for t in [last]+handles]
        assert [m['version'] for m in identities]==[1,1,1,2]
        stable=struct.unpack('<8Q',s.run([(76,opt,0,299)])[0]['payload'])[::2]
        s.run([(33,0),(72,opt)])
        fresh=s.run([(75,opt,0,299)])[0]['value'];states=[state(s,opt,i,0,299) for i in (1,2,3)]
        current=[meta(s,t) for t in [fresh]+states]
        assert all(old!=new for old,new in zip([last]+handles,[fresh]+states))
        assert [(m['storage_id'],m['version_id']) for m in identities]==[(m['storage_id'],m['version_id']) for m in current]
        assert [m['version'] for m in current]==[2,2,2,4]
        assert struct.unpack('<8Q',s.run([(76,opt,0,299)])[0]['payload'])[::2]==stable
        reject([(13,last)],1);reject([(13,handles[0])],1)
        s.run([(78,opt)])
        assert s.run([(10,states[0])])[0]['payload']==struct.pack('<f',2)
        return {'private_parameter_count':300,'real_private_state_Tensors':900,
            'owner_length_before_state':before,'owner_length_after_state':after,
            'descriptor_pages_and_tables_grow_then_reset_keeps_actual_roots':True,
            'public_alias_survives_Optimizer_drop':True}
    case('nine-hundred-state-Tensors-grow-reset-and-republish',private_growth,128<<20)
    for path,data in snapshots.items():assert path.read_bytes()==data,path
    (a.output/'executed-test.py').write_bytes(snapshots[Path(__file__).resolve()])
    report.update(status='PASS',case_count=len(report['cases']));save()
    print(json.dumps({key:report[key] for key in ('status','case_count','frames','rejections','rollback_bytes')}))


if __name__=='__main__':main()
