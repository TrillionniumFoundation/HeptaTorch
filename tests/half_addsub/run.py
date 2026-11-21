#!/usr/bin/env python3
"""Qualify native Half addition/subtraction bytes against fresh, physically retained Torch output."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import struct
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests'))
from half_addsub.cases import CASES,CASE_SHA256,raw_input,overwrite_words,SUCCESSORS,BOOL_REFUSALS,MUL_CONTROL_SUCCESSORS,LEGACY_FE_SUCCESSORS,ORIGINAL_CASE_SHA256
from numeric_predicates.native import Observer as BaseObserver,Live,Replay,Memory,load_wire,dump
from numeric_predicates.run import Inputs
import current_runtime_half_div_controls as division_controls

sha=lambda raw:hashlib.sha256(raw).hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def module(path,name,bind):
    bind(path);spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


class Observer(BaseObserver):
    def setup(self,spec):
        if 'cast' in spec:
            source,handles=self.setup(spec['source'])
            result=self.one(108,source,spec['cast'],0,0)['value']
            source,handles=result,[result]+handles
        else:
            raw=raw_input(spec)
            if spec['strides'] is None:
                source=self.create(spec['dtype'],spec['shape'],raw);handles=[source]
            else:
                backing=self.create(spec['dtype'],[spec['storage_count']],raw)
                shape,strides=spec['shape'],spec['strides']
                cfg=struct.pack('<10Q',len(shape),*shape,*([0]*(4-len(shape))),*strides,*([0]*(4-len(shape))),spec['offset'])
                source=self.one(111,backing,self.wire.Payload(cfg),80)['value'];handles=[source,backing]
        for index,word in enumerate(overwrite_words(spec)):self.one(12,source,index,word)
        return source,handles



def read_reference(directory,execution,inputs):
    report=json.loads(inputs.bind(directory/'report.json'))
    process=json.loads(inputs.bind(execution/'process.json'))
    require(report['schema']=='heptatorch.half-addsub.reference.v2' and report['status']=='OBSERVED_HALF_ADDSUB'
        and report['cases']==report['expected_cases']==len(report['rows'])==len(CASES)==1006
        and report['case_sha256']==CASE_SHA256 and not report['failures']
        and report['Torch_executed'] is True and report['native_executed'] is False
        and report['host_expected_values_computed'] is False and report['inputs_unchanged'] is True
        and report['rng_unchanged'] is True and report['cpu_capability']=='AVX2'
        and report['requested_cpu_capability']=='avx2' and report['cpu_environment']=='avx2'
        and report['torch_threads']==1,
        'complete fixed-wheel Half addition/subtraction observation')
    provenance=report['oracle_provenance']
    require(provenance['status']=='PASS' and len(provenance['checks'])==11
        and all(value is True for value in provenance['checks'].values()),'eleven original fixed-wheel checks')
    require(process['started'] is True and process['exit_code']==0
        and process['stdout_EOF'] is True and process['stderr_EOF'] is True
        and process['forced_cleanup'] is False and not process.get('timed_out')
        and not process.get('error') and not process.get('execution_error')
        and not process['cleanup_errors'] and not process['running_descendants_after_cleanup']
        and not process['terminated_process_groups'],'actual clean reference process completion')
    argv=process['argv']
    require(argv[1:]==['-B',str(ROOT/'tests/half_addsub/reference.py'),'--output',str(directory)]
        and report['actual_process']['argv']==argv and report['actual_process']['pid']==process['pid'],
        'actual reference process argument and PID binding')
    interpreter=Path(argv[0]).resolve(strict=True)
    for channel in ('stdout','stderr'):
        stream=inputs.bind(execution/('runner.'+channel),process[channel+'_sha256'])
        require(len(stream)==process[channel+'_bytes'],'complete reference stream')
    require(inputs.bind(execution/'runner.stderr')==b'','clean reference stderr')
    require(json.loads(inputs.bind(execution/'runner.stdout'))==dict(status='OBSERVED_HALF_ADDSUB',
        cases=1006,expected_cases=1006,error=None),'complete actual reference stdout')
    expected_files={str(p.resolve()) for p in [ROOT/'tests/half_addsub/reference.py',ROOT/'tests/half_addsub/__init__.py',
        ROOT/'tests/half_addsub/cases.py',ROOT/'tests/numeric_predicates/reference.py',ROOT/'tests/numeric_predicates/cases.py',
        ROOT/'tests/numeric_predicates/__init__.py',ROOT/'tests/pytorch_oracle.py',ROOT/'compatibility/pytorch-oracle-lock.json',
        ROOT/'compatibility/upstream-lock.json',ROOT/'build/pytorch-install.json',interpreter]}
    require(set(report['inputs'])==expected_files,'exact full reference input inventory')
    for path,record in report['inputs'].items():
        raw=inputs.bind(path,record['sha256'])
        require(record==dict(path=str(Path(path).resolve()),bytes=len(raw),sha256=sha(raw),mode=stat.S_IMODE(Path(path).stat().st_mode)),
            'original reference input bytes and mode')
    require(report['actual_process']['python']==report['inputs'][str(interpreter)],'exact interpreter identity')
    def blob(item):
        require(set(item)=={'path','bytes','sha256'} and item['path']=='blobs/'+item['sha256']+'.bin','canonical observed blob')
        raw=inputs.bind(directory/item['path'],item['sha256']);require(len(raw)==item['bytes'],'complete observed blob')
        return raw
    for index,(row,spec) in enumerate(zip(report['rows'],CASES)):
        require(row==json.loads(inputs.bind(directory/('case-%03d.json'%index))) and row['index']==index
            and row['spec']==spec and row['inputs_unchanged'] is True and row['rng_unchanged'] is True
            and row['accepted']==(not spec.get('expected_error',False)),
            'exact literal input and independently retained operation outcome')
        require(len(row['inputs'])==2,'both complete Torch operands')
        values=list(row['inputs'])
        if row['accepted']:
            require(row['fresh_tensor'] is True and row['fresh_storage'] is True,'independent output ownership')
            values.append(row['result'])
        else:
            if spec['expected_error']=='bool_subtraction':
                require(row['error']['type']=='NotImplementedError' and 'bool tensor' in row['error']['message'],
                    'actual Torch Bool subtraction rejection')
            else:
                require(spec['expected_error']=='broadcast' and row['error']['type']=='RuntimeError'
                    and 'size' in row['error']['message'] and 'match' in row['error']['message'],
                    'actual Torch broadcasting error')
        for value in values:blob(value['logical']);blob(value['storage'])

    return dict(report=report,blob=blob,report_sha256=sha(inputs.bind(directory/'report.json')))


def factory_packet(n,spec,invalid_suffix=False):
    left,right=spec['left'],spec['right'];lr,rr=raw_input(left),raw_input(right)
    require(left['strides'] is None and right['strides'] is None,'dense literal same-frame input')
    count=3+int(invalid_suffix);end=32+64*count;other=(end+64+len(lr)+7)&~7;total=other+64+len(rr)
    packet=b'HTORCH03'+struct.pack('<3Q',count,total,0)
    packet+=struct.pack('<8Q',97,left['dtype'],end,0,0,0,0,0)
    packet+=struct.pack('<8Q',97,right['dtype'],other,0,0,0,0,0)
    packet+=struct.pack('<8Q',2,spec['operation'],n.wire.reference(1),n.wire.reference(2),0,0,0,0)
    if invalid_suffix:packet+=struct.pack('<8Q',125,n.wire.reference(1),0,0,0,0,0,0)
    for item,raw,at in ((left,lr,end),(right,rr,other)):
        shape=item['shape'];packet+=b'\x00'*(at-len(packet))
        packet+=struct.pack('<8Q',len(shape),0,*shape,*([0]*(4-len(shape))),at+64,len(raw))+raw
    return packet


def allocation_control(n,row):
    spec=row['spec'];generator,rng=n.begin()
    left,lh=n.setup(spec['left']);right,rh=n.setup(spec['right'])
    original=[n.tensor(left)[0],n.tensor(right)[0]]
    n.check('both OOM operands equal actual Torch',original,[n.expected(item) for item in row['inputs']])
    n.s.lower_limit(n.s.owner());live=[];command=n.wire.packet([(2,spec['operation'],left,right)])
    for attempt in range(32):
        status,records,_,_=n.issue(command)
        if status:
            n.check('actual native add/sub allocation refusal',status,3)
            break
        live.append(records[0]['value'])
    else:raise ValueError('no genuine allocation refusal within fixed32 attempt budget')
    n.check('complete RNG survives real allocation refusal',n.one(63,generator)['payload'],rng)
    n.s.restore_limit();status,records,_,_=n.issue(command)
    n.check('identical add/sub request succeeds after actual limit restoration',status,0)
    result=records[0]['value'];live.append(result)
    n.check('retry result equals independently retained Torch bytes',n.tensor(result)[0],n.expected(row['result']))
    n.check('allocation failure and identical retry preserve both complete operands',[n.tensor(left)[0],n.tensor(right)[0]],original)
    for token in live+lh+rh:n.one(41,token)
    n.end(generator,rng)
    return dict(reference_index=row['index'],attempts=attempt+1,allocation_refusals=1,
        complete_owner_rollback=True,identical_request_succeeded_after_limit_restore=True)


def legacy_fe_packet(n,spec,right,invalid_suffix=False):
    """The original228-byte factory97/FE-add request, plus a rollback control."""
    label=spec['legacy_fe_successor_label'];dtype=dict(LEGACY_FE_SUCCESSORS)[label]
    require(spec['operation']==1 and spec['left']['dtype']==dtype
        and spec['left']['shape']==[2] and spec['left']['strides'] is None
        and raw_input(spec['left'])==struct.pack('<2H',1,2)
        and spec['right']['dtype']==0 and spec['right']['shape']==[2]
        and raw_input(spec['right'])==struct.pack('<2I',0x3f800000,0x80000000)
        and spec['same_handle'] is False,'the complete original FE request recipe')
    count=2+int(invalid_suffix);end=32+64*count;raw=raw_input(spec['left'])
    body=b'HTORCH03'+struct.pack('<3Q',count,end+64+len(raw),0)
    body+=struct.pack('<8Q',97,dtype,end,0,0,0,0,0)
    body+=struct.pack('<8Q',2,1,n.wire.reference(1),right,0,0,0,0)
    if invalid_suffix:body+=struct.pack('<8Q',125,n.wire.reference(1),0,0,0,0,0,0)
    body+=struct.pack('<8Q',1,0,2,0,0,0,end+64,len(raw))+raw
    require(len(body)==(292 if invalid_suffix else 228),'exact complete FE packet length')
    return body


def live_owner_rows(memory):
    rows={}
    for kind in range(1,9):
        for slot in range(1,memory.slots+1):
            offset=139264+128+((kind-1)*memory.slots+slot-1)*64
            row=struct.unpack_from('<8Q',memory.raw,offset)
            if row[1]==1:
                token=(row[0]<<32)|(kind<<24)|slot
                rows[(kind,token)]=memory.row(kind,token)
    return rows


def compare_legacy_fe(n,row):
    spec=row['spec'];label=spec['legacy_fe_successor_label'];generator,rng=n.begin()
    right,handles=n.setup(spec['right']);original,right_node=n.tensor(right)
    n.check('original FE pre-existing F32 operand equals actual Torch',original,n.expected(row['inputs'][1]))
    request=legacy_fe_packet(n,spec,right)
    status,records,before,after=n.issue(request)
    n.check('original FE factory and addition both succeed',status,0)
    n.check('original FE complete response sequence',[record['opcode'] for record in records],[97,2])
    bm,am=Memory(before['bytes']),Memory(after['bytes'])
    old_rows,new_rows=live_owner_rows(bm),live_owner_rows(am)
    tensor_storages={bm.node(token)[27] for kind,token in old_rows if kind==1}
    for (kind,token),old_row in old_rows.items():
        n.check('FE preserves every pre-existing live owner table row',new_rows.get((kind,token)),old_row)
        if kind==1:n.check('FE preserves every84-word pre-existing Tensor descriptor',am.node(token),bm.node(token))
        # Descriptor pages are internal Storage objects whose free descriptor
        # slots legitimately change on allocation. Preserve all their live
        # table rows and all existing Tensor descriptors, and compare complete
        # payloads for the Storages actually exposed by pre-existing Tensors.
        if kind==2 and token in tensor_storages:
            old_storage=bm.raw[bm.start+old_row[2]:bm.start+old_row[2]+old_row[3]]
            new_row=new_rows[(kind,token)]
            new_storage=am.raw[am.start+new_row[2]:am.start+new_row[2]+new_row[3]]
            n.check('FE preserves all existing complete Storage bytes',new_storage,old_storage)
    added=set(new_rows)-set(old_rows)
    n.check('exactly two new Tensor Storage and Version objects, no other live edges',
        [sum(kind==category for kind,_ in added) for category in range(1,9)],[2,2,2,0,0,0,0,0])
    left,result=[record['value'] for record in records]
    actual_left,left_node=n.tensor(left);actual,result_node=n.tensor(result)
    n.check('same-frame raw Half factory equals independently observed literal subnormals',actual_left,n.expected(row['inputs'][0]))
    n.check('same-frame F32 addition result bytes and full geometry equal actual Torch',actual,n.expected(row['result']))
    n.check('pre-existing F32 operand unchanged after the complete FE transaction',n.tensor(right)[0],original)
    n.check('both FE outputs have fresh independent Tensor Storage Version and incarnation',
        [len({left,result,right})==3,len({left_node[27],result_node[27],right_node[27]})==3,
         len({left_node[28],result_node[28],right_node[28]})==3,
         len({left_node[40],result_node[40],right_node[40]})==3],[True]*4)
    n.check('reported outputs bind the complete new Tensor/Storage/Version sets',added,
        {(kind,handle) for node,token in ((left_node,left),(result_node,result))
            for kind,handle in ((1,token),(2,node[27]),(3,node[28]))})
    no_edges=(32,40,48,56,72,80,88,176,184,248,280,288,296,312,360,384,392,400,408,584,592,600,608,616,624,640,648,656,664)
    for node in (left_node,result_node):
        n.check('both published FE tensors have no AD Saved input or view retention edges',
            [node[offset//8] for offset in no_edges],[0]*len(no_edges))
    n.check('complete RNG survives the original successful FE transaction',n.one(63,generator)['payload'],rng)
    maps=n.s.maps();refused,_,_,_=n.issue(legacy_fe_packet(n,spec,right,True))
    n.check('full successful FE factory/add prefix rolls back on invalid suffix',refused,4)
    n.check('all mappings unchanged after FE prefix rollback',n.s.maps(),maps)
    n.check('published FE outputs survive the later rollback',
        [n.tensor(left)[0],n.tensor(result)[0],n.tensor(right)[0]],[actual_left,actual,original])
    n.one(41,left)
    for token in handles:n.one(41,token)
    n.check('FE result survives releasing both factory input and pre-existing input',n.tensor(result)[0],actual)
    n.one(41,result);n.end(generator,rng)
    return dict(index=row['index'],name=spec['name'],operation=1,accepted=True,expected_error=None,
        elements=actual['numel'],bitwise=True,controls=1,allocation_control=None,
        legacy_fe_successor_label=label,same_frame_request_sha256=sha(request),
        complete_existing_owner_rows_preserved=len(old_rows),new_Tensor_Storage_Version_counts=[2,2,2])


def compare(n,row,division_reference):
    if 'legacy_fe_successor_label' in row['spec']:return compare_legacy_fe(n,row)
    spec=row['spec'];generator,rng=n.begin();left,lh=n.setup(spec['left'])
    right,rh=(left,[]) if spec.get('same_handle') else n.setup(spec['right'])
    left_before,left_node=n.tensor(left);right_before,right_node=n.tensor(right)
    n.check('complete independent left geometry and bits',left_before,n.expected(row['inputs'][0]))
    n.check('complete independent right geometry and bits',right_before,n.expected(row['inputs'][1]))
    command=(2,spec['operation'],left,right)
    status,records,before,after=n.issue(n.wire.packet([command]))
    expected_status=0 if row['accepted'] else (7 if spec['expected_error']=='bool_subtraction' else 2)
    n.check('actual Torch acceptance or exact public refusal mapping',status,expected_status)
    n.check('all84 left descriptor words unchanged',Memory(before['bytes']).node(left),Memory(after['bytes']).node(left))
    n.check('all84 right descriptor words unchanged',Memory(before['bytes']).node(right),Memory(after['bytes']).node(right))
    n.check('complete left unchanged',n.tensor(left)[0],left_before)
    n.check('complete right unchanged',n.tensor(right)[0],right_before)
    elements=controls=0;division_successors=[]
    if row['accepted']:
        n.check('exact add/sub response',[r['opcode'] for r in records],[2])
        result=records[0]['value'];actual,node=n.tensor(result)
        n.check('all add/sub result bytes and metadata equal actual Torch',actual,n.expected(row['result']))
        elements=actual['numel']
        n.check('fresh Tensor Storage Version identities',
            [result not in (left,right),node[27] not in (left_node[27],right_node[27]),node[28] not in (left_node[28],right_node[28])],[True]*3)
        no_edges=(32,40,48,56,72,80,88,176,184,248,280,288,296,312,360,384,392,400,408,584,592,600,608,616,624,640,648,656,664)
        n.check('result has no AD Saved input or view retention edges',[node[o//8] for o in no_edges],[0]*len(no_edges))
        if spec.get('controls'):
            for side in (2,3):
                for invalid in (0,1,generator):
                    bad=list(command);bad[side]=invalid
                    refused,_,_,_=n.issue(n.wire.packet([bad]));n.check('invalid operand handle rollback',refused,1);controls+=1
            stale=n.create(spec['left']['dtype'],[1],b'\x00\x00');n.one(41,stale)
            for side in (2,3):
                bad=list(command);bad[side]=stale
                refused,_,_,_=n.issue(n.wire.packet([bad]));n.check('stale operand handle rollback',refused,1);controls+=1
            for operation in (0,4,5,(1<<64)-1):
                if operation==4:
                    elements+=division_controls.compare(n,row,left,right,division_reference,'half_addsub')
                    division_successors.append(spec['name'])
                else:
                    refused,_,_,_=n.issue(n.wire.packet([(2,operation,left,right)]))
                    n.check('remaining unimplemented Half arithmetic explicit refusal',refused,7)
                controls+=1
            refused,_,_,_=n.issue(n.wire.packet([command,(125,left)]))
            n.check('successful add/sub then invalid suffix rolls back allocation',refused,4);controls+=1
            grad=n.create(0,[1],struct.pack('<I',0x3f800000),True)
            for operands in ((left,grad),(grad,right)):
                refused,_,_,_=n.issue(n.wire.packet([(2,spec['operation'],*operands)]))
                n.check('Half AD remains explicitly unsupported with complete rollback',refused,5);controls+=1
            n.one(41,grad)
            accepted,fe,_,_=n.issue(factory_packet(n,spec))
            n.check('same-frame public factories and FE arithmetic accepted',accepted,0)
            n.check('all same-frame records',[r['opcode'] for r in fe],[97,97,2])
            n.check('same-frame left equals actual Torch',n.tensor(fe[0]['value'])[0],n.expected(row['inputs'][0]))
            n.check('same-frame right equals actual Torch',n.tensor(fe[1]['value'])[0],n.expected(row['inputs'][1]))
            n.check('same-frame output equals actual Torch',n.tensor(fe[2]['value'])[0],n.expected(row['result']))
            for rec in fe:n.one(41,rec['value'])
            controls+=1
            maps=n.s.maps();refused,_,_,_=n.issue(factory_packet(n,spec,True))
            n.check('same-frame allocated factories and arithmetic prefix rollback',refused,4)
            n.check('complete mappings restored after failed FE transaction',n.s.maps(),maps);controls+=1
            n.check('controls preserve both complete input tensors',[n.tensor(left)[0],n.tensor(right)[0]],[left_before,right_before])
        for token in lh+rh:n.one(41,token)
        n.check('result survives complete input release',n.tensor(result)[0],actual)
        n.one(41,result)
    else:
        for token in lh+rh:n.one(41,token)
    n.end(generator,rng);allocation=None
    if spec.get('allocation_control'):
        allocation=allocation_control(n,row);elements+=actual['numel']
    return dict(index=row['index'],name=spec['name'],operation=spec['operation'],accepted=row['accepted'],
        expected_error=spec.get('expected_error'),elements=elements,bitwise=True,controls=controls,allocation_control=allocation,
        division_successors=division_successors)


def qualify(report,repository,build,build_sha,output,inputs):
    """Reapply every comparison to the original retained physical execution."""
    require(Path(repository).resolve()==ROOT,'actual Half addition/subtraction source repository')
    qualification=module(ROOT/'tools/current_runtime/qualification.py','half_addsub_readback_build',inputs.bind)
    built=qualification.admit_build(build,inputs.bind,build_sha)
    require(report==json.loads(inputs.bind(output/'report.json'))
        and report['schema']=='heptatorch.half-addsub.native-observation.v2'
        and report['status']=='PASS' and report['group']=='half_addsub'
        and report['cases']==report['expected_cases']==len(CASES)==1006
        and report['case_sha256']==CASE_SHA256 and not report['failures']
        and report['source_sha256']==built['source_sha256'] and report['native_sha256']==built['native_sha256']
        and report['actual_build_sha256']==build_sha and report['native_executed'] is True
        and report['Torch_executed'] is True and report['host_expected_values_computed'] is False
        and report['all_inputs_unchanged'] is True,'complete current Half addition/subtraction native execution')
    for path,record in report['inputs'].items():
        raw=inputs.bind(path,record['sha256'])
        require(record==dict(bytes=len(raw),sha256=sha(raw),mode=stat.S_IMODE(Path(path).stat().st_mode)),
            'unchanged complete original Half addition/subtraction observer input')
    reference=read_reference(Path(report['reference_directory']).resolve(strict=True),
        Path(report['reference_process_directory']).resolve(strict=True),inputs)
    require(reference['report_sha256']==report['reference_sha256'],'exact independent original reference')
    division_reference=division_controls.read_reference(ROOT,
        Path(report['division_reference_directory']).resolve(strict=True),
        Path(report['division_reference_process_directory']).resolve(strict=True),inputs)
    require(division_reference['report_sha256']==report['division_reference_sha256'],
        'same complete actual1196 reference for original division controls')
    wire=load_wire(ROOT,inputs.bind);frames=checks=owners=elements=mappings=allocation_refusals=protocol_controls=0
    fe_successors=[];division_successors=[]
    require(len(report['results'])==len(CASES),'all Half addition/subtraction native child cases')
    for row,saved in zip(reference['report']['rows'],report['results']):
        require(saved==json.loads(inputs.bind(output/saved['name']/'observation.json'))
            and saved['status']=='PASS' and saved['name']=='case-%03d'%row['index'],'complete native per-case record')
        session=Replay(build/'heptatorch',wire,output/saved['name'],inputs.bind)
        n=Observer(session,wire,reference);observed=compare(n,row,division_reference);finish=session.finish()
        require(observed==saved['result'] and n.checks==saved['checks'] and finish==saved['finish'],
            'every original bitwise geometry lifetime predicate reapplied to raw physical evidence')
        frames+=finish['frames'];owners+=finish['owner_reads'];checks+=n.checks;elements+=observed['elements']
        mappings+=finish['mapping_reads'];protocol_controls+=observed['controls']
        division_successors.extend(observed.get('division_successors',[]))
        if 'legacy_fe_successor_label' in observed:fe_successors.append(observed['legacy_fe_successor_label'])
        if observed['allocation_control'] is not None:allocation_refusals+=observed['allocation_control']['allocation_refusals']
    require((frames,owners,checks,mappings)==(report['frames'],report['owner_reads'],report['checks'],report['mapping_reads']),'complete physical replay census')
    require(allocation_refusals==4 and protocol_controls==70,'four real OOM recoveries and all70 protocol controls')
    require(fe_successors==[label for label,_ in LEGACY_FE_SUCCESSORS],
        'both original FE requests physically replayed exactly once')
    require(division_successors==list(division_controls.LABELS['half_addsub']),
        'all and only original division controls physically replayed against the shared reference')
    inputs.recheck()
    return dict(cases=1006,frames=frames,assertions=checks,owner_snapshots=owners,native_sessions=1006,
        bitwise_elements=elements,mapping_reads=mappings,actual_allocation_refusals=allocation_refusals,
        identical_request_recoveries=allocation_refusals,protocol_controls=protocol_controls,physical_replay=True,native_executed_by_qualifier=False,
        Torch_executed_by_qualifier=False,reference_sha256=reference['report_sha256'],
        cpu_capability=reference['report']['cpu_capability'],requested_cpu_capability='avx2',torch_threads=1,
        original_refusal_successors=[row['spec']['legacy_refusal_label'] for row in reference['report']['rows']
            if 'legacy_refusal_label' in row['spec'] and row['accepted']],
        original_bool_subtraction_refusals=[row['spec']['legacy_refusal_label'] for row in reference['report']['rows']
            if 'legacy_refusal_label' in row['spec'] and not row['accepted']],
        multiplication_refusal_successors=[row['spec']['multiplication_refusal_label'] for row in reference['report']['rows']
            if 'multiplication_refusal_label' in row['spec']],
        original_fe_successors=fe_successors,original1004_case_sha256=ORIGINAL_CASE_SHA256,
        original1004_cases_unchanged=True,
        division_reference_sha256=division_reference['report_sha256'],
        replaced_division_refusals=division_successors,
        full_project_status='HOLD',host_expected_values_computed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-directory',type=Path,required=True)
    parser.add_argument('--expected-build-sha256',required=True)
    parser.add_argument('--torch-python',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-directory',type=Path)
    parser.add_argument('--reference-execution',type=Path)
    parser.add_argument('--half-div-reference-directory',type=Path,required=True)
    parser.add_argument('--half-div-reference-execution',type=Path,required=True)
    args=parser.parse_args();build=args.build_directory.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False);inputs=Inputs();started=time.monotonic()
    report=dict(schema='heptatorch.half-addsub.native-observation.v2',status='FAIL',group='half_addsub',
        cases=len(CASES),expected_cases=len(CASES),case_sha256=CASE_SHA256,results=[],failures=[],
        native_executed=False,Torch_executed=False,host_expected_values_computed=False,frames=0,checks=0,owner_reads=0,mapping_reads=0,
        actual_build_sha256=args.expected_build_sha256,full_project_status='HOLD')
    try:
        require(not sys.flags.optimize,'all independent predicates enabled')
        inputs.bind(Path(sys.executable).resolve())
        for directory in ('half_addsub','numeric_predicates','half_logical'):
            for file in sorted((ROOT/'tests'/directory).glob('*.py')):inputs.bind(file)
        qualification=module(ROOT/'tools/current_runtime/qualification.py','half_addsub_build',inputs.bind)
        built=qualification.admit_build(build,inputs.bind,args.expected_build_sha256)
        report.update(source_sha256=built['source_sha256'],native_sha256=built['native_sha256'])
        require((args.reference_directory is None)==(args.reference_execution is None),
            'shared reference and actual process receipt must be supplied together')
        if args.reference_directory is None:
            supervisor=module(ROOT/'tests/current_runtime_process.py','half_addsub_process',inputs.bind)
            reference_directory=output/'reference';execution=output/'reference-execution';execution.mkdir()
            process=supervisor.execute([str(args.torch_python.absolute()),'-B',str(ROOT/'tests/half_addsub/reference.py'),
                '--output',str(reference_directory)],execution,300,cwd=ROOT)
            dump(execution/'process.json',process)
        else:
            reference_directory=args.reference_directory.resolve(strict=True)
            execution=args.reference_execution.resolve(strict=True)
        report.update(reference_directory=str(reference_directory),reference_process_directory=str(execution))
        reference=read_reference(reference_directory,execution,inputs)
        require(Path(args.torch_python).resolve(strict=True)==Path(reference['report']['actual_process']['python']['path']),
            'configured Torch interpreter equals the actual retained reference process')
        report.update(Torch_executed=True,reference_sha256=reference['report_sha256'],cpu_capability=reference['report']['cpu_capability'],
            requested_cpu_capability=reference['report']['requested_cpu_capability'],torch_threads=reference['report']['torch_threads'])
        division_directory=args.half_div_reference_directory.resolve(strict=True)
        division_execution=args.half_div_reference_execution.resolve(strict=True)
        division_reference=division_controls.read_reference(ROOT,division_directory,division_execution,inputs)
        require(Path(args.torch_python).resolve(strict=True)==Path(division_reference['report']['actual_process']['python']['path']),
            'same pinned actual interpreter for original arithmetic and division controls')
        report.update(division_reference_directory=str(division_directory),division_reference_process_directory=str(division_execution),
            division_reference_sha256=division_reference['report_sha256'])
        wire=load_wire(ROOT,inputs.bind)
        for row in reference['report']['rows']:
            name='case-%03d'%row['index'];saved=dict(name=name,status='FAIL');session=None;n=None
            try:
                session=Live(build/'heptatorch',wire,output/name,inputs.bind);report['native_executed']=True
                n=Observer(session,wire,reference);saved['result']=compare(n,row,division_reference);saved['status']='PASS'
            except BaseException as error:
                saved['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            finally:
                if session is not None:
                    try:saved['finish']=session.finish()
                    except BaseException as error:
                        saved.update(status='FAIL',finish_error=dict(type=type(error).__name__,message=str(error)))
                saved['checks']=n.checks if n is not None else 0
                if (output/name).exists():dump(output/name/'observation.json',saved)
            report['results'].append(saved)
            report['frames']+=saved.get('finish',{}).get('frames',0)
            report['owner_reads']+=saved.get('finish',{}).get('owner_reads',0)
            report['mapping_reads']+=saved.get('finish',{}).get('mapping_reads',0)
            report['checks']+=saved['checks']
            if saved['status']!='PASS':report['failures'].append(name)
            if row['index']%32==0 or saved['status']!='PASS':print(json.dumps(saved),flush=True)
        inputs.recheck();report['all_inputs_unchanged']=True
        if len(report['results'])==1006 and not report['failures']:report['status']='PASS'
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    report['inputs']=inputs.records;report['elapsed_seconds']=time.monotonic()-started
    dump(output/'report.json',report)
    if report['status']=='PASS':
        try:
            proof=qualify(report,ROOT,build,args.expected_build_sha256,output,Inputs())
            dump(output/'physical-qualification.json',proof)
        except BaseException as error:
            dump(output/'physical-qualification.json',dict(status='FAIL',error=str(error),traceback=traceback.format_exc()))
            print(json.dumps(dict(status='FAIL_PHYSICAL_REPLAY',error=str(error))),flush=True)
            return 1
    print(json.dumps({k:report.get(k) for k in ('status','cases','frames','checks','failures','error','physical_replay')}),flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
