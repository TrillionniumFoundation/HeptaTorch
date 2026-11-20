#!/usr/bin/env python3
"""Qualify native BF16 copy bytes against fresh, physically retained Torch output."""

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
from bf16_copy.cases import CASES,CASE_SHA256,raw_input
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
        raw=raw_input(spec)
        if spec['strides'] is None:
            source=self.create(0,spec['shape'],raw)
            return source,[source]
        backing=self.create(0,[spec['input_count']],raw)
        shape,strides=spec['shape'],spec['strides']
        cfg=struct.pack('<10Q',len(shape),*shape,*([0]*(4-len(shape))),*strides,*([0]*(4-len(shape))),spec['offset'])
        source=self.one(111,backing,self.wire.Payload(cfg),80)['value']
        return source,[source,backing]

    def command(self,spec,source):
        if spec['opcode']==108:return (108,source,6,0,spec['memory_format'])
        return (109,source,6,spec['memory_format'])


def read_reference(directory,execution,inputs):
    report=json.loads(inputs.bind(directory/'report.json'))
    process=json.loads(inputs.bind(execution/'process.json'))
    require(report['schema']=='heptatorch.bf16-copy.reference.v1' and report['status']=='OBSERVED_BF16_COPY'
        and report['cases']==report['expected_cases']==len(report['rows'])==len(CASES)==280
        and report['case_sha256']==CASE_SHA256 and not report['failures']
        and report['Torch_executed'] is True and report['native_executed'] is False
        and report['host_expected_values_computed'] is False and report['inputs_unchanged'] is True
        and report['rng_unchanged'] is True and report['cpu_capability'] in ('AVX2','AVX512'),
        'complete fixed-wheel BF16 observation')
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
    require(argv[1:]==['-B',str(ROOT/'tests/bf16_copy/reference.py'),'--output',str(directory)]
        and report['actual_process']['argv']==argv and report['actual_process']['pid']==process['pid'],
        'actual reference process argument and PID binding')
    interpreter=Path(argv[0]).resolve(strict=True)
    for channel in ('stdout','stderr'):
        stream=inputs.bind(execution/('runner.'+channel),process[channel+'_sha256'])
        require(len(stream)==process[channel+'_bytes'],'complete reference stream')
    require(inputs.bind(execution/'runner.stderr')==b'','clean reference stderr')
    require(json.loads(inputs.bind(execution/'runner.stdout'))==dict(status='OBSERVED_BF16_COPY',
        cases=280,expected_cases=280,error=None),'complete actual reference stdout')
    expected_files={str(p.resolve()) for p in [ROOT/'tests/bf16_copy/reference.py',ROOT/'tests/bf16_copy/__init__.py',
        ROOT/'tests/bf16_copy/cases.py',ROOT/'tests/numeric_predicates/reference.py',ROOT/'tests/numeric_predicates/cases.py',
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
            and row['spec']==spec and all(row[k] is True for k in ('input_unchanged','rng_unchanged','fresh_tensor','fresh_storage')),
            'exact literal input and independently retained output')
        for value in (row['input'],row['result']):blob(value['logical']);blob(value['storage'])
    return dict(report=report,blob=blob,report_sha256=sha(inputs.bind(directory/'report.json')))


def compare(n,row):
    spec=row['spec'];generator,rng=n.begin();source,inputs=n.setup(spec)
    before,node=n.tensor(source)
    n.check('complete independent input geometry and bits',before,n.expected(row['input']))
    status,records,owner_before,owner_after=n.issue(n.wire.packet([n.command(spec,source)]))
    n.check('conversion accepts valid recipe',status,0)
    n.check('exact conversion response',[r['opcode'] for r in records],[spec['opcode']])
    result=records[0]['value'];actual,result_node=n.tensor(result)
    n.check('all BF16 result bytes and metadata equal actual Torch',actual,n.expected(row['result']))
    n.check('all input descriptor words unchanged',Memory(owner_before['bytes']).node(source),Memory(owner_after['bytes']).node(source))
    n.check('complete source unchanged',n.tensor(source)[0],before)
    n.check('fresh owned Tensor Storage Version',[result!=source,node[27]!=result_node[27],node[28]!=result_node[28]],[True]*3)
    for token in inputs:n.one(41,token)
    n.check('result survives source release',n.tensor(result)[0],actual)
    n.one(41,result);n.end(generator,rng)
    return dict(index=row['index'],name=spec['name'],opcode=spec['opcode'],memory_format=spec['memory_format'],
                elements=actual['numel'],bitwise=True)


def qualify(report,repository,build,build_sha,output,inputs):
    """Reapply every comparison to the original retained physical execution."""
    require(Path(repository).resolve()==ROOT,'actual BF16 source repository')
    qualification=module(ROOT/'tools/current_runtime/qualification.py','bf16_copy_readback_build',inputs.bind)
    built=qualification.admit_build(build,inputs.bind,build_sha)
    require(report==json.loads(inputs.bind(output/'report.json'))
        and report['schema']=='heptatorch.bf16-copy.native-observation.v1'
        and report['status']=='PASS' and report['group']=='bf16_copy'
        and report['cases']==report['expected_cases']==len(CASES)==280
        and report['case_sha256']==CASE_SHA256 and not report['failures']
        and report['source_sha256']==built['source_sha256'] and report['native_sha256']==built['native_sha256']
        and report['actual_build_sha256']==build_sha and report['native_executed'] is True
        and report['Torch_executed'] is True and report['host_expected_values_computed'] is False
        and report['all_inputs_unchanged'] is True,'complete current BF16 native execution')
    for path,record in report['inputs'].items():
        raw=inputs.bind(path,record['sha256'])
        require(record==dict(bytes=len(raw),sha256=sha(raw),mode=stat.S_IMODE(Path(path).stat().st_mode)),
            'unchanged complete original BF16 observer input')
    reference=read_reference(output/'reference',output/'reference-execution',inputs)
    require(reference['report_sha256']==report['reference_sha256'],'exact independent original reference')
    wire=load_wire(ROOT,inputs.bind);frames=checks=owners=elements=0
    require(len(report['results'])==len(CASES),'all BF16 native child cases')
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
    return dict(cases=280,frames=frames,assertions=checks,owner_snapshots=owners,native_sessions=280,
        bitwise_elements=elements,physical_replay=True,native_executed_by_qualifier=False,
        Torch_executed_by_qualifier=False,reference_sha256=reference['report_sha256'],
        full_project_status='HOLD',host_expected_values_computed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-directory',type=Path,required=True)
    parser.add_argument('--expected-build-sha256',required=True)
    parser.add_argument('--torch-python',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build=args.build_directory.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False);inputs=Inputs();started=time.monotonic()
    report=dict(schema='heptatorch.bf16-copy.native-observation.v1',status='FAIL',group='bf16_copy',
        cases=len(CASES),expected_cases=len(CASES),case_sha256=CASE_SHA256,results=[],failures=[],
        native_executed=False,Torch_executed=False,host_expected_values_computed=False,frames=0,checks=0,owner_reads=0,
        actual_build_sha256=args.expected_build_sha256,full_project_status='HOLD')
    try:
        require(not sys.flags.optimize,'all independent predicates enabled')
        inputs.bind(Path(sys.executable).resolve())
        for directory in ('bf16_copy','numeric_predicates','half_logical'):
            for file in sorted((ROOT/'tests'/directory).glob('*.py')):inputs.bind(file)
        qualification=module(ROOT/'tools/current_runtime/qualification.py','bf16_copy_build',inputs.bind)
        built=qualification.admit_build(build,inputs.bind,args.expected_build_sha256)
        report.update(source_sha256=built['source_sha256'],native_sha256=built['native_sha256'])
        supervisor=module(ROOT/'tests/current_runtime_process.py','bf16_copy_process',inputs.bind)
        execution=output/'reference-execution';execution.mkdir()
        process=supervisor.execute([str(args.torch_python.absolute()),'-B',str(ROOT/'tests/bf16_copy/reference.py'),
            '--output',str(output/'reference')],execution,300,cwd=ROOT)
        dump(execution/'process.json',process)
        reference=read_reference(output/'reference',execution,inputs)
        report.update(Torch_executed=True,reference_sha256=reference['report_sha256'],cpu_capability=reference['report']['cpu_capability'])
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
        if len(report['results'])==280 and not report['failures']:report['status']='PASS'
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
