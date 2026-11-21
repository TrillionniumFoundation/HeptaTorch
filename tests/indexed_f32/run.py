#!/usr/bin/env python3
"""Execute native indexed F32 forward and replay every original frame and owner snapshot."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import stat
import struct
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests'))
from indexed_f32.cases import CASES,CASE_SHA256
from indexed_f32.native import Observer,Live,Replay,Memory,load_wire,dump,recipes
from numeric_predicates.evidence import SUPPORT,LOCK

sha=lambda raw:hashlib.sha256(raw).hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


class Inputs:
    def __init__(self):self.files={};self.records=self.files
    def bind(self,path,expected=None):
        path=Path(path).absolute();before=path.lstat()
        require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),'regular physical input: '+str(path))
        raw=path.read_bytes();after=path.stat()
        fields=lambda value:(value.st_dev,value.st_ino,value.st_mode,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
        require(fields(before)==fields(after),'input changed during read: '+str(path))
        row=dict(bytes=len(raw),sha256=sha(raw),mode=stat.S_IMODE(after.st_mode))
        require(expected is None or row['sha256']==expected,'exact input hash: '+str(path))
        require(str(path) not in self.files or self.files[str(path)]==row,'input changed: '+str(path))
        self.files[str(path)]=row
        return raw
    def finish(self):
        for path,row in list(self.files.items()):self.bind(path,row['sha256'])
    recheck=finish


def module(path,name,bind):
    bind(path);spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def profile(repository,build,build_sha,inputs):
    qualification=module(repository/'tools/current_runtime/qualification.py','indexed_f32_current_profile',inputs.bind)
    built=qualification.admit_build(build,inputs.bind,build_sha)
    for package in ('indexed_f32','numeric_predicates','half_logical'):
        for file in sorted((repository/'tests'/package).glob('*.py')):inputs.bind(file)
    transition=module(repository/'tools/current_runtime/indexed_forward_source_transition.py','indexed_f32_inverse',inputs.bind)
    source=inputs.bind(repository/'tools/current_runtime/source/runtime.hepta')
    parent,proof=transition.restore_source(source,inputs.bind)
    require(proof['source_sha256']==built['source_sha256'] and proof['functions']==781
        and proof['parent_functions']==778 and proof['unchanged_parent_functions']==776
        and proof['changed_functions']==['gather','indexed_add']
        and proof['appended_functions']==['indexed_gather_forward_source','indexed_add_forward_inputs','indexed_add_forward_location']
        and proof['whole_source_inverse_exact'] is True and proof['original_requires_grad_dispatch_preserved'] is True
        and proof['original_backward_functions_unchanged'] is True,'exact indexed forward source inverse and original AD dispatch')
    combined=module(repository/'tools/current_runtime/index_signbit_source_transition.py','indexed_f32_parent_inverse',inputs.bind)
    baseline,previous=combined.restore_source(parent,inputs.bind)
    require(previous['source_sha256']==proof['parent_source_sha256'] and previous['functions']==778
        and previous['parent_functions']==775 and previous['whole_source_inverse_exact'] is True
        and previous['whole_round_trip_exact'] is True,'complete unchanged778 to775 source chain')
    return built


def reference(repository,directory,execution,inputs,allow=False):
    raw=inputs.bind(directory/'report.json');report=json.loads(raw)
    require(report['schema']=='heptatorch.indexed-f32.reference.v1' and report['case_sha256']==CASE_SHA256
        and report['expected_cases']==report['cases']==len(report['rows'])==len(CASES)
        and report['Torch_executed'] is True and report['native_executed'] is False
        and report['compiler_executed'] is False and report['failures']==[]
        and report['host_expected_values_computed'] is False and report['inputs_unchanged'] is True
        and report['rng_unchanged'] is True and report['cpu_capability']=='AVX2' and report['torch_threads']==1
        and report['all_observed_blobs_unchanged'] is True,
        'complete actual index_select reference')
    wanted='OBSERVED_EXPLORATORY_INDEXED_F32' if allow else 'OBSERVED_INDEXED_F32'
    require(report['status']==wanted and report['production_oracle_qualified'] is (not allow),
        'strict fixed-wheel provenance is mandatory unless explicitly exploratory')
    provenance=report['oracle_provenance']
    require(provenance['status']==('EXPLORATORY_NOT_PINNED_WHEEL_QUALIFICATION' if allow else 'PASS')
        and len(provenance['checks'])==(5 if allow else 11)
        and all(value is True for value in provenance['checks'].values())
        and provenance['lock_sha256']==LOCK,'actual installed Torch identity and original fixed-wheel controls')
    inputs.bind(repository/'tests/pytorch_oracle.py',SUPPORT)
    lock=json.loads(inputs.bind(repository/'compatibility/pytorch-oracle-lock.json',LOCK))
    require(provenance['expected']==lock['oracle'],'original official wheel lock unchanged')
    if allow:
        require(provenance['full_installation_provenance_verified'] is False,'exploratory results grant no CI qualification')
        for key in ('package','extension'):
            row=provenance[key];path=Path(row['path']);data=inputs.bind(path,row['sha256'])
            require(row==dict(path=str(path),bytes=len(data),sha256=sha(data),mode=stat.S_IMODE(path.stat().st_mode)),
                'actual local Torch binary and package retained unchanged')
    process=json.loads(inputs.bind(execution/'process.json'))
    require(process==json.loads(inputs.bind(execution/'runner.json')),'actual independent reference process receipt matches supervisor original')
    require(process['started'] is True and process['exit_code']==0
        and process['stdout_EOF'] is True and process['stderr_EOF'] is True
        and process['forced_cleanup'] is False and not process.get('timed_out')
        and not process.get('error') and not process.get('execution_error')
        and not process['cleanup_errors'] and not process['running_descendants_after_cleanup']
        and not process['terminated_process_groups'],'actual reference exit, both EOFs and normal completion')
    expected_args=['-B',str(repository/'tests/indexed_f32/reference.py'),'--output',str(directory)]
    if allow:expected_args.append('--allow-unverified-local')
    argv=process['argv']
    require(argv[1:]==expected_args and report['actual_process']['argv']==argv
        and report['actual_process']['pid']==process['pid'],'exact supervised independent Torch argv and PID')
    interpreter=Path(argv[0]).resolve(strict=True);python=report['actual_process']['python'];data=inputs.bind(interpreter)
    require(python==dict(path=str(interpreter),bytes=len(data),sha256=sha(data),mode=stat.S_IMODE(interpreter.stat().st_mode)),
        'actual exact independent Torch interpreter')
    for channel in ('stdout','stderr'):
        stream=inputs.bind(execution/('runner.'+channel),process[channel+'_sha256'])
        require(len(stream)==process[channel+'_bytes'],'complete actual reference '+channel)
    require(inputs.bind(execution/'runner.stderr')==b'','clean actual Torch reference stderr')
    require(json.loads(inputs.bind(execution/'runner.stdout'))==dict(status=wanted,cases=331,expected_cases=331,
        production_oracle_qualified=not allow,error=None),'complete original reference stdout')
    files=[repository/'tests/indexed_f32'/name for name in ('__init__.py','cases.py','reference.py')]
    files += [repository/'tests/numeric_predicates'/name for name in ('__init__.py','cases.py','reference.py')]
    files += [repository/'tests/pytorch_oracle.py',repository/'compatibility/pytorch-oracle-lock.json',
        repository/'compatibility/upstream-lock.json',interpreter]
    if not allow:files.append(repository/'build/pytorch-install.json')
    require(set(report['inputs'])=={str(path.resolve()) for path in files},'complete actual reference input census')
    for path,record in report['inputs'].items():
        data=inputs.bind(path,record['sha256'])
        require(record==dict(path=path,bytes=len(data),sha256=sha(data),mode=stat.S_IMODE(Path(path).stat().st_mode)),
            'complete unchanged original reference input bytes and modes')
    def blob(item):
        require(set(item)=={'path','bytes','sha256'} and item['path']=='blobs/'+item['sha256']+'.bin','canonical actual reference blob')
        raw=inputs.bind(directory/item['path'],item['sha256']);require(len(raw)==item['bytes'],'full actual reference blob');return raw
    for index,(row,spec) in enumerate(zip(report['rows'],CASES)):
        require(row==json.loads(inputs.bind(directory/('case-%04d.json'%index))) and row['index']==index and row['spec']==spec,
            'exact literal case input and original observed record')
        require(row['inputs_unchanged'] is True and row['rng_unchanged'] is True,
            'actual reference preserved original input and global RNG')
        if row['accepted']:require(row['fresh_tensor'] is True and row['fresh_storage'] is True,'actual independent result ownership')
        for key in ('input_a','input_b','result','seed','gradient_a','gradient_b'):
            if row.get(key) is not None:
                for field in ('logical','storage'):blob(row[key][field])
    return dict(report=report,blob=blob,root=directory,execution=execution,report_sha256=sha(raw))


def qualify(report,repository,build,build_sha,output,inputs,allow_unverified_local=False):
    """Reapply every numerical assertion to original physical frames and Owner bytes."""
    built=profile(repository,build,build_sha,inputs)
    require(report==json.loads(inputs.bind(output/'report.json')),'unchanged original native report')
    wanted='PASS_EXPLORATORY_NATIVE_ONLY' if allow_unverified_local else 'PASS'
    require(report['schema']=='heptatorch.indexed-f32.native.v1' and report['group']=='indexed_f32'
        and report['status']==wanted and report['cases']==report['expected_cases']==len(CASES)==331
        and report['case_sha256']==CASE_SHA256 and not report['failures']
        and report['source_sha256']==built['source_sha256'] and report['native_sha256']==built['native_sha256']
        and report['actual_build_sha256']==build_sha and report['native_executed'] is True
        and report['Torch_executed'] is True and report['compiler_executed'] is False
        and report['host_expected_values_computed'] is False and report['original_reference_modified'] is False
        and report['all_inputs_unchanged'] is True and report['full_pytorch_rewrite_complete'] is False
        and report['allow_unverified_local'] is allow_unverified_local
        and report['production_oracle_qualified'] is (not allow_unverified_local)
        and report['full_installation_provenance_verified'] is (not allow_unverified_local),
        'exact actual independent observation identity and qualification scope')
    for path,row in report['inputs'].items():
        raw=inputs.bind(path,row['sha256'])
        require(row==dict(bytes=len(raw),sha256=sha(raw),mode=stat.S_IMODE(Path(path).stat().st_mode)),
            'every original observer input retained unchanged')
    fixed=reference(repository,Path(report['reference_directory']),Path(report['reference_process_directory']),
        inputs,allow_unverified_local)
    require(fixed['report_sha256']==report['reference_sha256'],'exact original independent actual Torch reference')
    wire=load_wire(repository,inputs.bind);cases=recipes(fixed)
    require(len(report['results'])==len(cases),'complete original numerical and control observation census')
    totals=dict(frames=0,owner_reads=0,mapping_reads=0,checks=0)
    for saved,(name,runner,row) in zip(report['results'],cases):
        require(saved==json.loads(inputs.bind(output/name/'observation.json'))
            and saved['name']==name and saved['status']=='PASS','exact original per-session observation')
        session=Replay(build/'heptatorch',wire,output/name,inputs.bind);observer=Observer(session,wire,fixed)
        result=runner(observer,row);finish=session.finish()
        require(result==saved['result'] and finish==saved['finish'] and observer.checks==saved['checks'],
            'every original numerical metadata ownership lifetime and resource assertion reapplied')
        for key in totals:totals[key]+=observer.checks if key=='checks' else finish[key]
    require(totals==report['totals'] and all(report[key]==totals[key] for key in totals)
        and totals['mapping_reads']==30,'complete physical frame Owner and mapping census')
    ordinary=report['results'][:331];allocations=report['results'][331:]
    require(len(report['results'])==334 and len(allocations)==3
        and sum(row['result']['controls'] for row in ordinary)==56
        and sum(row['result']['gradient_checked'] for row in ordinary)==193
        and all(row['result']['actual_allocation_failures']==1 and row['result']['identical_request_recovered'] is True for row in allocations),
        'complete original protocol, actual backward and real OOM recovery coverage')
    return dict(cases=331,frames=totals['frames'],assertions=totals['checks'],native_sessions=334,
        owner_snapshots=totals['owner_reads'],mapping_snapshots=totals['mapping_reads'],physical_replay=True,
        native_executed_by_qualifier=False,Torch_executed_by_qualifier=False,
        complete_ci_admission=not allow_unverified_local,full_installation_provenance_verified=not allow_unverified_local,
        production_oracle_qualified=not allow_unverified_local,reference_sha256=fixed['report_sha256'],
        additional_protocol_controls=56,additional_successful_FE_transactions=3,additional_failed_FE_transactions=12,
        additional_allocation_recoveries=3,actual_autograd_cases=193,host_expected_values_computed=False,
        original_backward_functions_unchanged=True,original_requires_grad_dispatch_preserved=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-directory',type=Path,required=True)
    p.add_argument('--expected-build-sha256',required=True)
    p.add_argument('--torch-python',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-unverified-local',action='store_true',help='Exploratory work only; cannot pass complete CI admission.')
    args=p.parse_args();require(not sys.flags.optimize,'independent numerical predicates must be enabled')
    output=args.output.resolve();build=args.build_directory.resolve();output.mkdir(parents=True,exist_ok=False)
    inputs=Inputs();started=time.monotonic()
    report=dict(schema='heptatorch.indexed-f32.native.v1',group='indexed_f32',status='FAIL',results=[],failures=[],
        cases=len(CASES),expected_cases=len(CASES),case_sha256=CASE_SHA256,frames=0,owner_reads=0,mapping_reads=0,checks=0,
        native_executed=False,Torch_executed=False,compiler_executed=False,host_expected_values_computed=False,
        original_reference_modified=False,actual_build_sha256=args.expected_build_sha256,
        full_pytorch_rewrite_complete=False,full_project_status='HOLD',runtime_qualification_transferred=False,
        allow_unverified_local=args.allow_unverified_local,production_oracle_qualified=False,
        full_installation_provenance_verified=False)
    try:
        inputs.bind(Path(sys.executable).resolve())
        built=profile(ROOT,build,args.expected_build_sha256,inputs)
        report.update(source_sha256=built['source_sha256'],native_sha256=built['native_sha256'])
        supervisor=module(ROOT/'tests/current_runtime_process.py','index_select_reference_supervisor',inputs.bind)
        execution=output/'reference-execution';execution.mkdir()
        argv=[str(args.torch_python.absolute()),'-B',str(ROOT/'tests/indexed_f32/reference.py'),
            '--output',str(output/'reference')]
        if args.allow_unverified_local:argv.append('--allow-unverified-local')
        process=supervisor.execute(argv,execution,300,cwd=ROOT)
        dump(execution/'process.json',process)
        fixed=reference(ROOT,output/'reference',execution,inputs,args.allow_unverified_local)
        report.update(Torch_executed=True,reference_directory=str(fixed['root']),
            reference_process_directory=str(execution),reference_sha256=fixed['report_sha256'],
            production_oracle_qualified=fixed['report']['production_oracle_qualified'],
            full_installation_provenance_verified=fixed['report']['production_oracle_qualified'])
        wire=load_wire(ROOT,inputs.bind);cases=recipes(fixed)
        totals=dict(frames=0,owner_reads=0,mapping_reads=0,checks=0)
        for ordinal,(name,runner,case) in enumerate(cases):
            row=dict(name=name,status='FAIL');session=observer=None
            try:
                session=Live(build/'heptatorch',wire,output/name,inputs.bind);report['native_executed']=True
                observer=Observer(session,wire,fixed);row['result']=runner(observer,case);row['status']='PASS'
            except BaseException as error:
                row['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            finally:
                if session is not None:
                    try:row['finish']=session.finish()
                    except BaseException as error:
                        row['status']='FAIL';row['finish_error']=dict(type=type(error).__name__,message=str(error))
                row['checks']=observer.checks if observer is not None else 0
                if (output/name).exists():dump(output/name/'observation.json',row)
            report['results'].append(row)
            for key in totals:totals[key]+=row['checks'] if key=='checks' else row.get('finish',{}).get(key,0)
            if row['status']!='PASS':report['failures'].append(name)
            if ordinal%64==0 or row['status']!='PASS':
                print(json.dumps(dict(group='indexed_f32',recipe=name,status=row['status'],error=row.get('error'))),flush=True)
        inputs.finish();report.update(all_inputs_unchanged=True,totals=totals,**totals)
        if not report['failures'] and len(report['results'])==334:
            report['status']='PASS_EXPLORATORY_NATIVE_ONLY' if args.allow_unverified_local else 'PASS'
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    report.update(inputs=inputs.files,elapsed_seconds=time.monotonic()-started)
    dump(output/'report.json',report)
    if report['status'] in ('PASS','PASS_EXPLORATORY_NATIVE_ONLY'):
        try:
            replay_inputs=Inputs()
            proof=qualify(report,ROOT,build,args.expected_build_sha256,output,replay_inputs,args.allow_unverified_local)
            replay_inputs.finish();dump(output/'physical-qualification.json',proof)
        except BaseException as error:
            dump(output/'physical-qualification.json',dict(status='FAIL',error=str(error),traceback=traceback.format_exc()))
            print(json.dumps(dict(status='FAIL_PHYSICAL_REPLAY',error=str(error))),flush=True)
            return 1
    print(json.dumps({key:report.get(key) for key in ('group','status','cases','frames','checks','failures','error')}),flush=True)
    return 0 if report['status'] in ('PASS','PASS_EXPLORATORY_NATIVE_ONLY') else 1


if __name__=='__main__':raise SystemExit(main())
