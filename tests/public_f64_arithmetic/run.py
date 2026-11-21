#!/usr/bin/env python3
"""Qualify native mixed public F64 arithmetic bytes against fresh, physically retained Torch output."""

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
from public_f64_arithmetic.cases import CASES,CASE_SHA256,CASE_COUNT,GROUP_COUNTS,LEGACY_CONTROLS,WIDTHS,dtype,raw_input,overwrite_words
from numeric_predicates.native import Observer as BaseObserver,Live,Replay,Memory,load_wire,dump
from numeric_predicates.run import Inputs

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
    require(report['schema']=='heptatorch.public-f64-arithmetic.reference.v1' and report['status']=='OBSERVED_PUBLIC_F64_ARITHMETIC'
        and report['cases']==report['expected_cases']==len(report['rows'])==len(CASES)==CASE_COUNT
        and report['case_sha256']==CASE_SHA256 and not report['failures']
        and report['Torch_executed'] is True and report['native_executed'] is False
        and report['host_expected_values_computed'] is False and report['inputs_unchanged'] is True
        and report['rng_unchanged'] is True and report['cpu_capability']=='AVX2'
        and report['requested_cpu_capability']=='avx2' and report['cpu_environment']=='avx2'
        and report['torch_threads']==1,
        'complete fixed-wheel mixed public F64 arithmetic observation')
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
    require(argv[1:]==['-B',str(ROOT/'tests/public_f64_arithmetic/reference.py'),'--output',str(directory)]
        and report['actual_process']['argv']==argv and report['actual_process']['pid']==process['pid'],
        'actual reference process argument and PID binding')
    interpreter=Path(argv[0]).resolve(strict=True)
    for channel in ('stdout','stderr'):
        stream=inputs.bind(execution/('runner.'+channel),process[channel+'_sha256'])
        require(len(stream)==process[channel+'_bytes'],'complete reference stream')
    require(inputs.bind(execution/'runner.stderr')==b'','clean reference stderr')
    require(json.loads(inputs.bind(execution/'runner.stdout'))==dict(status='OBSERVED_PUBLIC_F64_ARITHMETIC',
        cases=CASE_COUNT,expected_cases=CASE_COUNT,error=None),'complete actual reference stdout')
    expected_files={str(p.resolve()) for p in [ROOT/'tests/public_f64_arithmetic/reference.py',ROOT/'tests/public_f64_arithmetic/__init__.py',
        ROOT/'tests/public_f64_arithmetic/cases.py',ROOT/'tests/numeric_predicates/reference.py',ROOT/'tests/numeric_predicates/cases.py',
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
            require(spec['expected_error'] in ('broadcast','bool_subtraction'),
                'actual Torch original literal error case')
            message=row['error']['message'].lower()
            if spec['expected_error']=='broadcast':
                require(row['error']['type']=='RuntimeError' and 'size' in message and 'match' in message,
                    'actual Torch exact RuntimeError broadcasting error')
            else:
                require(row['error']['type']=='NotImplementedError' and 'bool' in message and 'subtraction' in message,
                    'actual Torch exact NotImplementedError Bool subtraction error')
        for value in values:blob(value['logical']);blob(value['storage'])

    return dict(report=report,blob=blob,report_sha256=sha(inputs.bind(directory/'report.json')))


def source_inverse(built,inputs):
    combined=module(ROOT/'tools/current_runtime/index_signbit_source_transition.py','arithmetic_index_signbit_inverse',inputs.bind)
    current,combined_proof=combined.restore_source(inputs.bind(ROOT/'tools/current_runtime/source/runtime.hepta'),inputs.bind)
    require(combined_proof['source_sha256']==built['source_sha256'] and combined_proof['functions']==800
        and combined_proof['parent_functions']==775 and combined_proof['unchanged_parent_functions']==755
        and combined_proof['changed_functions']==['entry', 'linear', 'reduce_axis', 'softmax', 'layernorm', 'batchnorm_eval', 'conv2d', 'pool2d', 'matmul', 'index_select', 'gather', 'tensor_join', 'indexed_add', 'bind_alias', 'so_tensor_live', 'to_command', 'to_strided_view', 'f64_dtype_audit', 'predicate_bits', 'predicate_command']
        and combined_proof['appended_functions']==['typed_index_select_previous', 'typed_index_select', 'typed_index_select_offset', 'indexed_gather_forward_source', 'indexed_add_forward_inputs', 'indexed_add_forward_location', 'matmul_forward_index', 'matmul_forward_point', 'softmax_forward_group', 'axis_source_forward', 'conv_term_forward', 'conv_no_grad_forward', 'pool_point_forward', 'pool_no_grad_forward', 'matmul_output_version', 'concat_forward_target', 'concat_forward_part', 'concat_forward_copy', 'concat_forward_layout', 'concat_forward', 'linear_bias_output_version', 'normalization_output_layout', 'public_f64_empty_strided_view', 'public_f64_empty_descriptor', 'so_f64_empty_offset_live']
        and combined_proof['whole_source_inverse_exact'] is True
        and combined_proof['whole_round_trip_exact'] is True
        and combined_proof['admitted_new_opcode']==126 and combined_proof['opcode125_remains_rejected'] is True,
        'complete800 source inverse precedes every unchanged775 predecessor predicate')
    relu=module(ROOT/'tools/current_runtime/typed_relu_source_transition.py','typed_relu_before_arithmetic_inverse',inputs.bind)
    same773,relu_proof=relu.restore_source(current,inputs.bind)
    require(relu_proof['source_sha256']==combined_proof['parent_source_sha256'] and relu_proof['functions']==775
        and relu_proof['parent_functions']==773 and relu_proof['changed_functions']==['unary']
        and relu_proof['whole_source_inverse_exact'] is True and relu_proof['whole_round_trip_exact'] is True,
        'complete actual typed ReLU775 inverse before the frozen same-F64 source')
    same=module(ROOT/'tools/current_runtime/same_f64_arithmetic_source_transition.py','same_f64_before_mixed_inverse',inputs.bind)
    previous,same_proof=same.restore_source(same773,inputs.bind)
    transition=module(ROOT/'tools/current_runtime/public_f64_arithmetic_source_transition.py','public_f64_arithmetic_exact_inverse',inputs.bind)
    original,proof=transition.restore_source(previous,inputs.bind)
    require(same_proof['source_sha256']==relu_proof['parent_source_sha256'] and same_proof['functions']==773
        and same_proof['parent_functions']==773 and same_proof['whole_source_inverse_exact'] is True
        and same_proof['whole_round_trip_exact'] is True and same_proof['same_f64_op1_to4_route_changed'] is True
        and same_proof['same_f64_old_fallback_preserved'] is False
        and proof['source_sha256']==same_proof['parent_source_sha256'] and proof['parent_functions']==771
        and proof['whole_source_inverse_exact'] is True and proof['whole_round_trip_exact'] is True,
        'actual same-F64 successor and exact mixed773 then771 predecessor recovery')
    return dict(combined_proof,typed_relu_predecessor_source_inverse=
        dict(relu_proof,same_f64_predecessor_source_inverse=
            dict(same_proof,mixed_predecessor_source_inverse=proof)))


def legacy_alias_packet(n,left,right,invalid_suffix=False):
    raw=raw_input(left);shape=left['shape'];count=2+int(invalid_suffix);end=32+64*count
    commands=struct.pack('<8Q',97,4,end,0,0,0,0,0)
    commands+=struct.pack('<8Q',2,1,n.wire.reference(1),right,0,0,0,0)
    if invalid_suffix:commands+=struct.pack('<8Q',125,n.wire.reference(1),0,0,0,0,0,0)
    config=struct.pack('<8Q',len(shape),0,*shape,*([0]*(4-len(shape))),end+64,len(raw))
    return b'HTORCH03'+struct.pack('<3Q',count,end+64+len(raw),0)+commands+config+raw


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
            n.check('actual native mixed public F64 arithmetic allocation refusal',status,3)
            break
        live.append(records[0]['value'])
    else:raise ValueError('no genuine allocation refusal within fixed32 attempt budget')
    n.check('complete RNG survives real allocation refusal',n.one(63,generator)['payload'],rng)
    n.s.restore_limit();status,records,_,_=n.issue(command)
    n.check('identical mixed public F64 arithmetic request succeeds after actual limit restoration',status,0)
    result=records[0]['value'];live.append(result)
    n.check('retry result equals independently retained Torch bytes',n.tensor(result)[0],n.expected(row['result']))
    n.check('allocation failure and identical retry preserve both complete operands',[n.tensor(left)[0],n.tensor(right)[0]],original)
    for token in live+lh+rh:n.one(41,token)
    n.end(generator,rng)
    return dict(reference_index=row['index'],attempts=attempt+1,allocation_refusals=1,
        complete_owner_rollback=True,identical_request_succeeded_after_limit_restore=True)








def compare(n,row):
    spec=row['spec'];generator,rng=n.begin();left,lh=n.setup(spec['left'])
    right,rh=(left,[]) if spec.get('same_handle') else n.setup(spec['right'])
    left_before,left_node=n.tensor(left);right_before,right_node=n.tensor(right)
    n.check('complete independent left geometry and bits',left_before,n.expected(row['inputs'][0]))
    n.check('complete independent right geometry and bits',right_before,n.expected(row['inputs'][1]))
    command=(2,spec['operation'],left,right)
    status,records,before,after=n.issue(n.wire.packet([command]))
    expected_status=0 if row['accepted'] else (2 if spec['expected_error']=='broadcast' else 7)
    n.check('actual Torch acceptance or exact public refusal mapping',status,expected_status)
    n.check('all84 left descriptor words unchanged',Memory(before['bytes']).node(left),Memory(after['bytes']).node(left))
    n.check('all84 right descriptor words unchanged',Memory(before['bytes']).node(right),Memory(after['bytes']).node(right))
    n.check('complete left unchanged',n.tensor(left)[0],left_before)
    n.check('complete right unchanged',n.tensor(right)[0],right_before)
    elements=controls=0
    if row['accepted']:
        n.check('exact mixed public F64 arithmetic response',[r['opcode'] for r in records],[2])
        result=records[0]['value'];actual,node=n.tensor(result)
        n.check('all mixed public F64 arithmetic result bytes and metadata equal actual Torch',actual,n.expected(row['result']))
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
            stale=n.create(spec['left']['dtype'],[1],b'\x00'*WIDTHS[spec['left']['dtype']]);n.one(41,stale)
            for side in (2,3):
                bad=list(command);bad[side]=stale
                refused,_,_,_=n.issue(n.wire.packet([bad]));n.check('stale operand handle rollback',refused,1);controls+=1
            for operation in (0,5,(1<<64)-1):
                refused,_,_,_=n.issue(n.wire.packet([(2,operation,left,right)]))
                n.check('original unsupported mixed F64 operation refusal',refused,7);controls+=1
            refused,_,_,_=n.issue(n.wire.packet([command,(125,left)]))
            n.check('successful mixed public F64 arithmetic then invalid suffix rolls back allocation',refused,4);controls+=1
            grad=n.create(0,[1],struct.pack('<I',0x3f800000),True)
            f64=left if dtype(spec['left'])==4 else right
            for operands in ((f64,grad),(grad,f64)):
                refused,_,_,_=n.issue(n.wire.packet([(2,spec['operation'],*operands)]))
                n.check('mixed F64 AD remains explicitly unsupported with complete rollback',refused,5);controls+=1
            n.one(41,grad)
            for position in range(4,8):
                unchanged=list(command)+[0]*4;unchanged[position]=1
                accepted,extra,prior,later=n.issue(n.wire.packet([unchanged]))
                n.check('original command2 unused words preserve default arithmetic',accepted,0)
                n.check('one default arithmetic response with unused word',[item['opcode'] for item in extra],[2])
                other=extra[0]['value'];observed,other_node=n.tensor(other)
                n.check('unused words do not supply alpha or alter any result bytes or metadata',observed,n.expected(row['result']))
                n.check('unused words preserve all84 left words',Memory(prior['bytes']).node(left),Memory(later['bytes']).node(left))
                n.check('unused words preserve all84 right words',Memory(prior['bytes']).node(right),Memory(later['bytes']).node(right))
                n.check('default result has fresh Tensor Storage Version identities',
                    [other not in (left,right,result),other_node[27] not in (left_node[27],right_node[27],node[27]),
                     other_node[28] not in (left_node[28],right_node[28],node[28])],[True]*3)
                n.check('unused words add no AD Saved input or view retention edges',[other_node[o//8] for o in no_edges],[0]*len(no_edges))
                n.one(41,other);controls+=1
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
        if spec.get('legacy_same_frame'):
            accepted,fe,_,_=n.issue(legacy_alias_packet(n,spec['left'],right))
            n.check('original F64 factory and existing F32 FE pair accepted',accepted,0)
            n.check('original two FE response records',[r['opcode'] for r in fe],[97,2])
            n.check('original FE literal factory bits and metadata',n.tensor(fe[0]['value'])[0],n.expected(row['inputs'][0]))
            n.check('original FE mixed arithmetic exact Torch result',n.tensor(fe[1]['value'])[0],n.expected(row['result']))
            n.check('original FE preserves the existing complete F32 tensor',n.tensor(right)[0],right_before)
            for record in fe:n.one(41,record['value'])
            controls+=1
            mappings=n.s.maps();refused,_,_,_=n.issue(legacy_alias_packet(n,spec['left'],right,True))
            n.check('original FE prefix plus invalid suffix restores complete owner',refused,4)
            n.check('original FE prefix rollback restores complete mappings',n.s.maps(),mappings);controls+=1
        for token in lh+rh:n.one(41,token)
        n.check('result survives complete input release' ,n.tensor(result)[0],actual)
        n.one(41,result)
    else:
        for token in lh+rh:n.one(41,token)
    n.end(generator,rng);allocation=None
    if spec.get('allocation_control'):
        allocation=allocation_control(n,row);elements+=actual['numel']
    return dict(index=row['index'],name=spec['name'],operation=spec['operation'],accepted=row['accepted'],
        expected_error=spec.get('expected_error'),elements=elements,bitwise=True,controls=controls,allocation_control=allocation)




def qualify(report,repository,build,build_sha,output,inputs):
    """Reapply every comparison to the original retained physical execution."""
    require(Path(repository).resolve()==ROOT,'actual mixed public F64 arithmetic source repository')
    qualification=module(ROOT/'tools/current_runtime/qualification.py','public_f64_arithmetic_readback_build',inputs.bind)
    built=qualification.admit_build(build,inputs.bind,build_sha)
    inverse=source_inverse(built,inputs)
    require(report.get('source_inverse')==inverse,'exact source inverse from the actually executed mixed F64 image')
    require(report==json.loads(inputs.bind(output/'report.json'))
        and report['schema']=='heptatorch.public-f64-arithmetic.native-observation.v1'
        and report['status']=='PASS' and report['group']=='public_f64_arithmetic'
        and report['cases']==report['expected_cases']==len(CASES)==CASE_COUNT
        and report['case_sha256']==CASE_SHA256 and not report['failures']
        and report['source_sha256']==built['source_sha256'] and report['native_sha256']==built['native_sha256']
        and report['actual_build_sha256']==build_sha and report['native_executed'] is True
        and report['Torch_executed'] is True and report['host_expected_values_computed'] is False
        and report['all_inputs_unchanged'] is True,'complete current mixed public F64 arithmetic native execution')
    for path,record in report['inputs'].items():
        raw=inputs.bind(path,record['sha256'])
        require(record==dict(bytes=len(raw),sha256=sha(raw),mode=stat.S_IMODE(Path(path).stat().st_mode)),
            'unchanged complete original mixed public F64 arithmetic observer input')
    reference=read_reference(Path(report['reference_directory']).resolve(strict=True),
        Path(report['reference_process_directory']).resolve(strict=True),inputs)
    require(reference['report_sha256']==report['reference_sha256'],'exact independent original reference')
    wire=load_wire(ROOT,inputs.bind);frames=checks=owners=elements=mappings=allocation_refusals=protocol_controls=0
    require(len(report['results'])==len(CASES),'all mixed public F64 arithmetic native child cases')
    for row,saved in zip(reference['report']['rows'],report['results']):
        require(saved==json.loads(inputs.bind(output/saved['name']/'observation.json'))
            and saved['status']=='PASS' and saved['name']=='case-%03d'%row['index'],'complete native per-case record')
        session=Replay(build/'heptatorch',wire,output/saved['name'],inputs.bind)
        n=Observer(session,wire,reference);observed=compare(n,row);finish=session.finish()
        require(observed==saved['result'] and n.checks==saved['checks'] and finish==saved['finish'],
            'every original bitwise geometry lifetime predicate reapplied to raw physical evidence')
        frames+=finish['frames'];owners+=finish['owner_reads'];checks+=n.checks;elements+=observed['elements']
        mappings+=finish['mapping_reads'];protocol_controls+=observed['controls']
        if observed['allocation_control'] is not None:allocation_refusals+=observed['allocation_control']['allocation_refusals']
    require((frames,owners,checks,mappings)==(report['frames'],report['owner_reads'],report['checks'],report['mapping_reads']),
        'complete physical replay census')
    require(allocation_refusals==4 and protocol_controls==82,'all four actual allocation failures and all82 protocol controls')
    require([row['spec']['legacy_refusal_label'] for row in reference['report']['rows'] if 'legacy_refusal_label' in row['spec']]==[row[0] for row in LEGACY_CONTROLS], 'all25 exact original F64 mixed entry controls')
    inputs.recheck()
    return dict(cases=CASE_COUNT,frames=frames,assertions=checks,owner_snapshots=owners,native_sessions=CASE_COUNT,
        bitwise_elements=elements,mapping_reads=mappings,actual_allocation_refusals=allocation_refusals,
        identical_request_recoveries=allocation_refusals,protocol_controls=protocol_controls,
        physical_replay=True,native_executed_by_qualifier=False,
        Torch_executed_by_qualifier=False,reference_sha256=reference['report_sha256'],
        cpu_capability=reference['report']['cpu_capability'],requested_cpu_capability='avx2',torch_threads=1,
        original_entry_controls=[row['spec']['legacy_refusal_label'] for row in reference['report']['rows']
            if 'legacy_refusal_label' in row['spec']],
        original_refusal_successors=[row['spec']['legacy_refusal_label'] for row in reference['report']['rows']
            if 'legacy_refusal_label' in row['spec'] and row['accepted']],
        retained_bool_subtraction_refusals=[row['spec']['legacy_refusal_label'] for row in reference['report']['rows']
            if 'legacy_refusal_label' in row['spec'] and not row['accepted']],
        group_counts=GROUP_COUNTS, same_f64_nan_payload_qualified=False,legacy_same_f64_route_unchanged=False,
        full_project_status='HOLD',host_expected_values_computed=False)



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-directory',type=Path,required=True)
    parser.add_argument('--expected-build-sha256',required=True)
    parser.add_argument('--torch-python',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-directory',type=Path)
    parser.add_argument('--reference-execution',type=Path)
    args=parser.parse_args();build=args.build_directory.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False);inputs=Inputs();started=time.monotonic()
    report=dict(schema='heptatorch.public-f64-arithmetic.native-observation.v1',status='FAIL',group='public_f64_arithmetic',
        cases=len(CASES),expected_cases=len(CASES),case_sha256=CASE_SHA256,results=[],failures=[],
        native_executed=False,Torch_executed=False,host_expected_values_computed=False,frames=0,checks=0,owner_reads=0,mapping_reads=0,
        actual_build_sha256=args.expected_build_sha256,full_project_status='HOLD')
    try:
        require(not sys.flags.optimize,'all independent predicates enabled')
        inputs.bind(Path(sys.executable).resolve())
        for directory in ('public_f64_arithmetic','numeric_predicates','half_logical'):
            for file in sorted((ROOT/'tests'/directory).glob('*.py')):inputs.bind(file)
        qualification=module(ROOT/'tools/current_runtime/qualification.py','public_f64_arithmetic_build',inputs.bind)
        built=qualification.admit_build(build,inputs.bind,args.expected_build_sha256)
        report.update(source_sha256=built['source_sha256'],native_sha256=built['native_sha256'],source_inverse=source_inverse(built,inputs))
        require((args.reference_directory is None)==(args.reference_execution is None),
            'shared reference and actual process receipt must be supplied together')
        if args.reference_directory is None:
            supervisor=module(ROOT/'tests/current_runtime_process.py','public_f64_arithmetic_process',inputs.bind)
            reference_directory=output/'reference';execution=output/'reference-execution';execution.mkdir()
            process=supervisor.execute([str(args.torch_python.absolute()),'-B',str(ROOT/'tests/public_f64_arithmetic/reference.py'),
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
        wire=load_wire(ROOT,inputs.bind)
        for row in reference['report']['rows']:
            name='case-%03d'%row['index'];saved=dict(name=name,status='FAIL');session=None;n=None
            try:
                session=Live(build/'heptatorch',wire,output/name,inputs.bind);report['native_executed']=True
                n=Observer(session,wire,reference);saved['result']=compare(n,row);saved['status']='PASS'
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
        if len(report['results'])==CASE_COUNT and not report['failures']:report['status']='PASS'
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
