#!/usr/bin/env python3
"""Qualify native Half multiplication bytes against fresh, physically retained Torch output."""

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
from half_mul.cases import CASES,CASE_SHA256,raw_input
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
            return result,[result]+handles
        raw=raw_input(spec)
        if spec['strides'] is None:
            source=self.create(spec['dtype'],spec['shape'],raw)
            return source,[source]
        backing=self.create(spec['dtype'],[spec['storage_count']],raw)
        shape,strides=spec['shape'],spec['strides']
        cfg=struct.pack('<10Q',len(shape),*shape,*([0]*(4-len(shape))),*strides,*([0]*(4-len(shape))),spec['offset'])
        source=self.one(111,backing,self.wire.Payload(cfg),80)['value']
        return source,[source,backing]


def read_reference(directory,execution,inputs):
    report=json.loads(inputs.bind(directory/'report.json'))
    process=json.loads(inputs.bind(execution/'process.json'))
    require(report['schema']=='heptatorch.half-mul.reference.v1' and report['status']=='OBSERVED_HALF_MUL'
        and report['cases']==report['expected_cases']==len(report['rows'])==len(CASES)==312
        and report['case_sha256']==CASE_SHA256 and not report['failures']
        and report['Torch_executed'] is True and report['native_executed'] is False
        and report['host_expected_values_computed'] is False and report['inputs_unchanged'] is True
        and report['rng_unchanged'] is True and report['cpu_capability']=='AVX2'
        and report['requested_cpu_capability']=='avx2' and report['cpu_environment']=='avx2'
        and report['torch_threads']==1,
        'complete fixed-wheel Half multiplication observation')
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
    require(argv[1:]==['-B',str(ROOT/'tests/half_mul/reference.py'),'--output',str(directory)]
        and report['actual_process']['argv']==argv and report['actual_process']['pid']==process['pid'],
        'actual reference process argument and PID binding')
    interpreter=Path(argv[0]).resolve(strict=True)
    for channel in ('stdout','stderr'):
        stream=inputs.bind(execution/('runner.'+channel),process[channel+'_sha256'])
        require(len(stream)==process[channel+'_bytes'],'complete reference stream')
    require(inputs.bind(execution/'runner.stderr')==b'','clean reference stderr')
    require(json.loads(inputs.bind(execution/'runner.stdout'))==dict(status='OBSERVED_HALF_MUL',
        cases=312,expected_cases=312,error=None),'complete actual reference stdout')
    expected_files={str(p.resolve()) for p in [ROOT/'tests/half_mul/reference.py',ROOT/'tests/half_mul/__init__.py',
        ROOT/'tests/half_mul/cases.py',ROOT/'tests/half_mul/legacy_controls.py',ROOT/'tests/numeric_predicates/reference.py',ROOT/'tests/numeric_predicates/cases.py',
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
            require(row['error']['type']=='RuntimeError' and 'size' in row['error']['message']
                and 'match' in row['error']['message'],'actual Torch broadcasting error')
        for value in values:blob(value['logical']);blob(value['storage'])

    return dict(report=report,blob=blob,report_sha256=sha(inputs.bind(directory/'report.json')))


def compare(n,row):
    spec=row['spec'];generator,rng=n.begin()
    left,left_handles=n.setup(spec['left']);right,right_handles=n.setup(spec['right'])
    left_before,left_node=n.tensor(left);right_before,right_node=n.tensor(right)
    n.check('complete independent left geometry and bits',left_before,n.expected(row['inputs'][0]))
    n.check('complete independent right geometry and bits',right_before,n.expected(row['inputs'][1]))
    command=(2,3,left,right)
    status,records,owner_before,owner_after=n.issue(n.wire.packet([command]))
    n.check('actual Torch multiplication acceptance or broadcasting error',status,0 if row['accepted'] else 2)
    n.check('left descriptor remains byte identical',Memory(owner_before['bytes']).node(left),Memory(owner_after['bytes']).node(left))
    n.check('right descriptor remains byte identical',Memory(owner_before['bytes']).node(right),Memory(owner_after['bytes']).node(right))
    n.check('complete left unchanged',n.tensor(left)[0],left_before)
    n.check('complete right unchanged',n.tensor(right)[0],right_before)
    elements=0;controls=0
    if row['accepted']:
        n.check('exact multiplication response',[r['opcode'] for r in records],[2])
        result=records[0]['value'];actual,result_node=n.tensor(result)
        n.check('all multiplication result bytes and metadata equal actual Torch',actual,n.expected(row['result']))
        elements=actual['numel']
        n.check('fresh Tensor Storage Version identities',
            [result not in (left,right),result_node[27] not in (left_node[27],right_node[27]),result_node[28] not in (left_node[28],right_node[28])],[True]*3)
        no_edges=(32,40,48,56,72,80,88,176,184,248,280,288,296,312,360,384,392,400,408,584,592,600,608,616,624,640,648,656,664)
        n.check('result has no AD Saved input or view retention edges',[result_node[o//8] for o in no_edges],[0]*len(no_edges))
        if spec.get('controls'):
            for side in (2,3):
                for invalid in (0,1,generator):
                    bad=list(command);bad[side]=invalid
                    refused,_,_,_=n.issue(n.wire.packet([bad]));n.check('invalid operand handle rollback',refused,1);controls+=1
            for operation in (0,1,2,4,5):
                refused,_,_,_=n.issue(n.wire.packet([(2,operation,left,right)]))
                n.check('unimplemented Half arithmetic remains an explicit refusal',refused,7);controls+=1
            refused,_,_,_=n.issue(n.wire.packet([command,(125,left)]))
            n.check('successful multiplication then invalid suffix rolls back allocation',refused,4);controls+=1
            grad=n.create(0,[1],struct.pack('<I',0x3f800000),True)
            for operands in ((left,grad),(grad,right)):
                refused,_,_,_=n.issue(n.wire.packet([(2,3,*operands)]))
                n.check('Half AD remains explicitly unsupported with full rollback',refused,5);controls+=1
            n.one(41,grad)
        for token in left_handles+right_handles:n.one(41,token)
        n.check('result survives complete input release',n.tensor(result)[0],actual)
        n.one(41,result)
    else:
        for token in left_handles+right_handles:n.one(41,token)
    n.end(generator,rng)
    return dict(index=row['index'],name=spec['name'],accepted=row['accepted'],elements=elements,bitwise=True,controls=controls)


def qualify(report,repository,build,build_sha,output,inputs):
    """Reapply every comparison to the original retained physical execution."""
    require(Path(repository).resolve()==ROOT,'actual Half multiplication source repository')
    qualification=module(ROOT/'tools/current_runtime/qualification.py','half_mul_readback_build',inputs.bind)
    built=qualification.admit_build(build,inputs.bind,build_sha)
    require(report==json.loads(inputs.bind(output/'report.json'))
        and report['schema']=='heptatorch.half-mul.native-observation.v1'
        and report['status']=='PASS' and report['group']=='half_mul'
        and report['cases']==report['expected_cases']==len(CASES)==312
        and report['case_sha256']==CASE_SHA256 and not report['failures']
        and report['source_sha256']==built['source_sha256'] and report['native_sha256']==built['native_sha256']
        and report['actual_build_sha256']==build_sha and report['native_executed'] is True
        and report['Torch_executed'] is True and report['host_expected_values_computed'] is False
        and report['all_inputs_unchanged'] is True,'complete current Half multiplication native execution')
    for path,record in report['inputs'].items():
        raw=inputs.bind(path,record['sha256'])
        require(record==dict(bytes=len(raw),sha256=sha(raw),mode=stat.S_IMODE(Path(path).stat().st_mode)),
            'unchanged complete original Half multiplication observer input')
    reference=read_reference(Path(report['reference_directory']).resolve(strict=True),
        Path(report['reference_process_directory']).resolve(strict=True),inputs)
    require(reference['report_sha256']==report['reference_sha256'],'exact independent original reference')
    wire=load_wire(ROOT,inputs.bind);frames=checks=owners=elements=0
    require(len(report['results'])==len(CASES),'all Half multiplication native child cases')
    for row,saved in zip(reference['report']['rows'],report['results']):
        require(saved==json.loads(inputs.bind(output/saved['name']/'observation.json'))
            and saved['status']=='PASS' and saved['name']=='case-%03d'%row['index'],'complete native per-case record')
        session=Replay(build/'heptatorch',wire,output/saved['name'],inputs.bind)
        n=Observer(session,wire,reference);observed=compare(n,row);finish=session.finish()
        require(observed==saved['result'] and n.checks==saved['checks'] and finish==saved['finish'],
            'every original bitwise geometry lifetime predicate reapplied to raw physical evidence')
        frames+=finish['frames'];owners+=finish['owner_reads'];checks+=n.checks;elements+=observed['elements']
    require((frames,owners,checks)==(report['frames'],report['owner_reads'],report['checks']),'complete physical replay census')
    inputs.recheck()
    return dict(cases=312,frames=frames,assertions=checks,owner_snapshots=owners,native_sessions=312,
        bitwise_elements=elements,physical_replay=True,native_executed_by_qualifier=False,
        Torch_executed_by_qualifier=False,reference_sha256=reference['report_sha256'],
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
    args=parser.parse_args();build=args.build_directory.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False);inputs=Inputs();started=time.monotonic()
    report=dict(schema='heptatorch.half-mul.native-observation.v1',status='FAIL',group='half_mul',
        cases=len(CASES),expected_cases=len(CASES),case_sha256=CASE_SHA256,results=[],failures=[],
        native_executed=False,Torch_executed=False,host_expected_values_computed=False,frames=0,checks=0,owner_reads=0,
        actual_build_sha256=args.expected_build_sha256,full_project_status='HOLD')
    try:
        require(not sys.flags.optimize,'all independent predicates enabled')
        inputs.bind(Path(sys.executable).resolve())
        for directory in ('half_mul','numeric_predicates','half_logical'):
            for file in sorted((ROOT/'tests'/directory).glob('*.py')):inputs.bind(file)
        qualification=module(ROOT/'tools/current_runtime/qualification.py','half_mul_build',inputs.bind)
        built=qualification.admit_build(build,inputs.bind,args.expected_build_sha256)
        report.update(source_sha256=built['source_sha256'],native_sha256=built['native_sha256'])
        require((args.reference_directory is None)==(args.reference_execution is None),
            'shared reference and actual process receipt must be supplied together')
        if args.reference_directory is None:
            supervisor=module(ROOT/'tests/current_runtime_process.py','half_mul_process',inputs.bind)
            reference_directory=output/'reference';execution=output/'reference-execution';execution.mkdir()
            process=supervisor.execute([str(args.torch_python.absolute()),'-B',str(ROOT/'tests/half_mul/reference.py'),
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
            report['checks']+=saved['checks']
            if saved['status']!='PASS':report['failures'].append(name)
            if row['index']%32==0 or saved['status']!='PASS':print(json.dumps(saved),flush=True)
        inputs.recheck();report['all_inputs_unchanged']=True
        if len(report['results'])==312 and not report['failures']:report['status']='PASS'
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
