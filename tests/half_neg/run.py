#!/usr/bin/env python3
"""Qualify native Half negation bytes against fresh, physically retained Torch output."""

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
from half_neg.cases import CASES,CASE_SHA256,raw_input,overwrite_words
from numeric_predicates.native import Observer as BaseObserver,Live,Replay,Memory,load_wire,dump,allocation_control
from numeric_predicates.run import Inputs
import current_runtime_typed_relu_controls as relu_controls

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

    def command(self,spec,source):
        return (3,6,source)


def factory_packet(n,spec,*,requires=False,negate=False,invalid_suffix=False):
    raw=raw_input(spec);shape=spec['shape']
    require(spec['strides'] is None,'dense original transaction input')
    count=1+int(negate)+int(invalid_suffix);end=32+64*count
    packet=b'HTORCH03'+struct.pack('<3Q',count,end+64+len(raw),0)
    packet+=struct.pack('<8Q',97,spec['dtype'],end,0,0,0,0,0)
    if negate:packet+=struct.pack('<8Q',3,6,n.wire.reference(1),0,0,0,0,0)
    if invalid_suffix:packet+=struct.pack('<8Q',125,n.wire.reference(1),0,0,0,0,0,0)
    packet+=struct.pack('<8Q',len(shape),int(requires),*shape,*([0]*(4-len(shape))),end+64,len(raw))+raw
    return packet


def read_reference(directory,execution,inputs):
    report=json.loads(inputs.bind(directory/'report.json'))
    process=json.loads(inputs.bind(execution/'process.json'))
    require(report['schema']=='heptatorch.half-neg.reference.v1' and report['status']=='OBSERVED_HALF_NEG'
        and report['cases']==report['expected_cases']==len(report['rows'])==len(CASES)==371
        and report['case_sha256']==CASE_SHA256 and not report['failures']
        and report['Torch_executed'] is True and report['native_executed'] is False
        and report['host_expected_values_computed'] is False and report['inputs_unchanged'] is True
        and report['rng_unchanged'] is True and report['cpu_capability']=='AVX2'
        and report['requested_cpu_capability']=='avx2' and report['cpu_environment']=='avx2'
        and report['torch_threads']==1,
        'complete fixed-wheel Half negation observation')
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
    require(argv[1:]==['-B',str(ROOT/'tests/half_neg/reference.py'),'--output',str(directory)]
        and report['actual_process']['argv']==argv and report['actual_process']['pid']==process['pid'],
        'actual reference process argument and PID binding')
    interpreter=Path(argv[0]).resolve(strict=True)
    for channel in ('stdout','stderr'):
        stream=inputs.bind(execution/('runner.'+channel),process[channel+'_sha256'])
        require(len(stream)==process[channel+'_bytes'],'complete reference stream')
    require(inputs.bind(execution/'runner.stderr')==b'','clean reference stderr')
    require(json.loads(inputs.bind(execution/'runner.stdout'))==dict(status='OBSERVED_HALF_NEG',
        cases=371,expected_cases=371,error=None),'complete actual reference stdout')
    expected_files={str(p.resolve()) for p in [ROOT/'tests/half_neg/reference.py',ROOT/'tests/half_neg/__init__.py',
        ROOT/'tests/half_neg/cases.py',ROOT/'tests/numeric_predicates/reference.py',ROOT/'tests/numeric_predicates/cases.py',
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
            and row['spec']==spec and row['input_unchanged'] is True and row['rng_unchanged'] is True
            and row['accepted'] is True,
            'exact literal input and independently retained operation outcome')
        require(row['fresh_tensor'] is True and row['fresh_storage'] is True,'independent output ownership')
        values=[row['input'],row['result']]
        for value in values:blob(value['logical']);blob(value['storage'])

    return dict(report=report,blob=blob,report_sha256=sha(inputs.bind(directory/'report.json')))


def compare(n,row,typed_relu_reference):
    spec=row['spec'];generator,rng=n.begin();source,handles=n.setup(spec['input'])
    original,source_node=n.tensor(source)
    n.check('complete independent source geometry and bits',original,n.expected(row['input']))
    command=(3,6,source)
    status,records,before,after=n.issue(n.wire.packet([command]))
    n.check('actual Torch negation acceptance',status,0)
    n.check('exact negation response',[r['opcode'] for r in records],[3])
    n.check('all84 input descriptor words unchanged',Memory(before['bytes']).node(source),Memory(after['bytes']).node(source))
    n.check('complete source unchanged',n.tensor(source)[0],original)
    result=records[0]['value'];actual,node=n.tensor(result)
    n.check('all negation result bytes and metadata equal actual Torch',actual,n.expected(row['result']))
    n.check('fresh Tensor Storage Version identities',
        [result!=source,node[27]!=source_node[27],node[28]!=source_node[28]],[True]*3)
    no_edges=(32,40,48,56,72,80,88,176,184,248,280,288,296,312,360,384,392,400,408,584,592,600,608,616,624,640,648,656,664)
    n.check('result has no AD Saved input or view retention edges',[node[o//8] for o in no_edges],[0]*len(no_edges))
    controls=0;typed_relu_successors=[]
    if spec.get('controls'):
        for invalid in (0,1,generator):
            refused,_,_,_=n.issue(n.wire.packet([(3,6,invalid)]));n.check('invalid operand handle rollback',refused,1);controls+=1
        stale=n.create(spec['input']['dtype'],[1],b'\x00\x00');n.one(41,stale)
        refused,_,_,_=n.issue(n.wire.packet([(3,6,stale)]));n.check('stale operand handle rollback',refused,1);controls+=1
        for operation in (0,1,2,3,4,5,7,11,12,13,14,15,(1<<64)-1):
            if operation==5:
                typed_relu_successors.append(relu_controls.compare(n,row,source,result,generator,typed_relu_reference,'half_neg'))
            else:
                refused,_,_,_=n.issue(n.wire.packet([(3,operation,source)]))
                n.check('unimplemented Half unary remains an explicit refusal',refused,7)
            controls+=1
        refused,_,_,_=n.issue(factory_packet(n,spec['input'],requires=True))
        n.check('Half requires-grad factory remains an explicit boundary',refused,5);controls+=1
        grad=n.create(0,[2],struct.pack('<2I',0x3f800000,0x40000000),True)
        for opcode in (108,109):
            refused,_,_,_=n.issue(n.wire.packet([(opcode,grad,spec['input']['dtype'],0,0)]))
            n.check('Half AD conversion remains unsupported with full rollback',refused,5);controls+=1
        n.one(41,grad)
        refused,_,_,_=n.issue(n.wire.packet([command,(125,source)]))
        n.check('successful negation then invalid suffix rolls back allocation',refused,4);controls+=1
        accepted,alias_records,_,_=n.issue(factory_packet(n,spec['input'],negate=True))
        n.check('same-frame public factory and FE negation accepted',accepted,0)
        n.check('exact same-frame records',[r['opcode'] for r in alias_records],[97,3])
        n.check('same-frame FE source exact actual Torch',n.tensor(alias_records[0]['value'])[0],n.expected(row['input']))
        n.check('same-frame FE result exact actual Torch',n.tensor(alias_records[1]['value'])[0],n.expected(row['result']))
        n.one(41,alias_records[0]['value']);n.one(41,alias_records[1]['value']);controls+=1
        maps=n.s.maps();refused,_,_,_=n.issue(factory_packet(n,spec['input'],negate=True,invalid_suffix=True))
        n.check('same-frame allocated factory-negation prefix rolls back',refused,4)
        n.check('same-frame rollback restores complete mappings',n.s.maps(),maps);controls+=1
        n.check('all packet controls leave original input unchanged',n.tensor(source)[0],original)
    for token in handles:n.one(41,token)
    n.check('result survives complete input release',n.tensor(result)[0],actual)
    n.one(41,result);n.end(generator,rng)
    allocation=None;elements=actual['numel']+sum(item['elements'] for item in typed_relu_successors)
    if spec.get('allocation_control'):
        allocation_row=dict(row,spec=spec['input'])
        allocation=allocation_control(n,allocation_row);elements+=actual['numel']
    return dict(index=row['index'],name=spec['name'],accepted=True,elements=elements,
        bitwise=True,controls=controls,allocation_control=allocation,typed_relu_successors=typed_relu_successors)


def qualify(report,repository,build,build_sha,output,inputs):
    """Reapply every comparison to the original retained physical execution."""
    require(Path(repository).resolve()==ROOT,'actual Half negation source repository')
    qualification=module(ROOT/'tools/current_runtime/qualification.py','half_neg_readback_build',inputs.bind)
    built=qualification.admit_build(build,inputs.bind,build_sha)
    require(report==json.loads(inputs.bind(output/'report.json'))
        and report['schema']=='heptatorch.half-neg.native-observation.v2'
        and report['status']=='PASS' and report['group']=='half_neg'
        and report['cases']==report['expected_cases']==len(CASES)==371
        and report['case_sha256']==CASE_SHA256 and not report['failures']
        and report['source_sha256']==built['source_sha256'] and report['native_sha256']==built['native_sha256']
        and report['actual_build_sha256']==build_sha and report['native_executed'] is True
        and report['Torch_executed'] is True and report['host_expected_values_computed'] is False
        and report['all_inputs_unchanged'] is True,'complete current Half negation native execution')
    for path,record in report['inputs'].items():
        raw=inputs.bind(path,record['sha256'])
        require(record==dict(bytes=len(raw),sha256=sha(raw),mode=stat.S_IMODE(Path(path).stat().st_mode)),
            'unchanged complete original Half negation observer input')
    reference=read_reference(Path(report['reference_directory']).resolve(strict=True),
        Path(report['reference_process_directory']).resolve(strict=True),inputs)
    require(reference['report_sha256']==report['reference_sha256'],'exact independent original reference')
    typed_relu_reference=relu_controls.read_reference(ROOT,
        Path(report['typed_relu_reference_directory']).resolve(strict=True),
        Path(report['typed_relu_reference_process_directory']).resolve(strict=True),inputs)
    require(typed_relu_reference['report_sha256']==report['typed_relu_reference_sha256']
        and typed_relu_reference['admission']['reference_process_sha256']==report['typed_relu_reference_process_sha256']
        and typed_relu_reference['admission']==report['typed_relu_reference_admission']
        and typed_relu_reference['report']['actual_process']['python']==reference['report']['actual_process']['python'],
        'same complete actual776 reference, process, fixed interpreter and exact four original recipes')
    typed_relu_successors=[]
    wire=load_wire(ROOT,inputs.bind);frames=checks=owners=elements=mappings=allocation_refusals=protocol_controls=0
    require(len(report['results'])==len(CASES),'all Half negation native child cases')
    for row,saved in zip(reference['report']['rows'],report['results']):
        require(saved==json.loads(inputs.bind(output/saved['name']/'observation.json'))
            and saved['status']=='PASS' and saved['name']=='case-%03d'%row['index'],'complete native per-case record')
        session=Replay(build/'heptatorch',wire,output/saved['name'],inputs.bind)
        n=Observer(session,wire,reference);observed=compare(n,row,typed_relu_reference);finish=session.finish()
        require(observed==saved['result'] and n.checks==saved['checks'] and finish==saved['finish'],
            'every original bitwise geometry lifetime predicate reapplied to raw physical evidence')
        frames+=finish['frames'];owners+=finish['owner_reads'];checks+=n.checks;elements+=observed['elements']
        mappings+=finish['mapping_reads'];protocol_controls+=observed['controls']
        typed_relu_successors.extend(observed['typed_relu_successors'])
        if observed['allocation_control'] is not None:allocation_refusals+=observed['allocation_control']['allocation_refusals']
    require((frames,owners,checks,mappings)==(report['frames'],report['owner_reads'],report['checks'],report['mapping_reads']),
        'complete physical replay census')
    require(allocation_refusals==2 and protocol_controls==46,'both actual allocation failures and all46 protocol controls')
    typed_relu_proof=relu_controls.qualify_successors(typed_relu_reference,typed_relu_successors,'half_neg')
    inputs.recheck()
    return dict(cases=371,frames=frames,assertions=checks,owner_snapshots=owners,native_sessions=371,
        bitwise_elements=elements,mapping_reads=mappings,actual_allocation_refusals=allocation_refusals,
        identical_request_recoveries=allocation_refusals,protocol_controls=protocol_controls,
        physical_replay=True,native_executed_by_qualifier=False,
        Torch_executed_by_qualifier=False,reference_sha256=reference['report_sha256'],
        typed_relu_reference_sha256=typed_relu_reference['report_sha256'],
        typed_relu_reference_process_sha256=typed_relu_reference['admission']['reference_process_sha256'],
        typed_relu_control_successors=typed_relu_proof,
        cpu_capability=reference['report']['cpu_capability'],requested_cpu_capability='avx2',torch_threads=1,
        original_refusal_successors=[row['spec']['legacy_refusal_label'] for row in reference['report']['rows']
            if 'legacy_refusal_label' in row['spec']],
        full_project_status='HOLD',host_expected_values_computed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-directory',type=Path,required=True)
    parser.add_argument('--expected-build-sha256',required=True)
    parser.add_argument('--torch-python',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-directory',type=Path)
    parser.add_argument('--reference-execution',type=Path)
    parser.add_argument('--typed-relu-reference-directory',type=Path,required=True)
    parser.add_argument('--typed-relu-reference-execution',type=Path,required=True)
    args=parser.parse_args();build=args.build_directory.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False);inputs=Inputs();started=time.monotonic()
    report=dict(schema='heptatorch.half-neg.native-observation.v2',status='FAIL',group='half_neg',
        cases=len(CASES),expected_cases=len(CASES),case_sha256=CASE_SHA256,results=[],failures=[],
        native_executed=False,Torch_executed=False,host_expected_values_computed=False,frames=0,checks=0,owner_reads=0,mapping_reads=0,
        actual_build_sha256=args.expected_build_sha256,full_project_status='HOLD')
    try:
        require(not sys.flags.optimize,'all independent predicates enabled')
        inputs.bind(Path(sys.executable).resolve())
        for directory in ('half_neg','numeric_predicates','half_logical'):
            for file in sorted((ROOT/'tests'/directory).glob('*.py')):inputs.bind(file)
        qualification=module(ROOT/'tools/current_runtime/qualification.py','half_neg_build',inputs.bind)
        built=qualification.admit_build(build,inputs.bind,args.expected_build_sha256)
        report.update(source_sha256=built['source_sha256'],native_sha256=built['native_sha256'])
        require((args.reference_directory is None)==(args.reference_execution is None),
            'shared reference and actual process receipt must be supplied together')
        if args.reference_directory is None:
            supervisor=module(ROOT/'tests/current_runtime_process.py','half_neg_process',inputs.bind)
            reference_directory=output/'reference';execution=output/'reference-execution';execution.mkdir()
            process=supervisor.execute([str(args.torch_python.absolute()),'-B',str(ROOT/'tests/half_neg/reference.py'),
                '--output',str(reference_directory)],execution,300,cwd=ROOT)
            dump(execution/'process.json',process)
        else:
            reference_directory=args.reference_directory.resolve(strict=True)
            execution=args.reference_execution.resolve(strict=True)
        report.update(reference_directory=str(reference_directory),reference_process_directory=str(execution))
        reference=read_reference(reference_directory,execution,inputs)
        typed_relu_reference=relu_controls.read_reference(ROOT,args.typed_relu_reference_directory,
            args.typed_relu_reference_execution,inputs)
        require(typed_relu_reference['report']['actual_process']['python']==reference['report']['actual_process']['python'],
            'the shared ReLU and original negation observations used the identical fixed interpreter')
        report.update(typed_relu_reference_directory=typed_relu_reference['admission']['reference_directory'],
            typed_relu_reference_process_directory=typed_relu_reference['admission']['reference_process_directory'],
            typed_relu_reference_sha256=typed_relu_reference['report_sha256'],
            typed_relu_reference_process_sha256=typed_relu_reference['admission']['reference_process_sha256'],
            typed_relu_reference_admission=typed_relu_reference['admission'])
        require(Path(args.torch_python).resolve(strict=True)==Path(reference['report']['actual_process']['python']['path']),
            'configured Torch interpreter equals the actual retained reference process')
        report.update(Torch_executed=True,reference_sha256=reference['report_sha256'],cpu_capability=reference['report']['cpu_capability'],
            requested_cpu_capability=reference['report']['requested_cpu_capability'],torch_threads=reference['report']['torch_threads'])
        wire=load_wire(ROOT,inputs.bind)
        for row in reference['report']['rows']:
            name='case-%03d'%row['index'];saved=dict(name=name,status='FAIL');session=None;n=None
            try:
                session=Live(build/'heptatorch',wire,output/name,inputs.bind);report['native_executed']=True
                n=Observer(session,wire,reference);saved['result']=compare(n,row,typed_relu_reference);saved['status']='PASS'
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
        if len(report['results'])==371 and not report['failures']:report['status']='PASS'
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
