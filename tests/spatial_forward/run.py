#!/usr/bin/env python3
"""Execute spatial/reduction cases and replay the complete original physical evidence."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests'))
from spatial_forward.cases import CASES,CASE_SHA256
from spatial_forward.native import Observer,Live,Replay,Memory,load_wire,dump,recipes
import pinned_reference_custody as custody

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
    # The unchanged indexed profile admits the entire795 ->781 ->778 ->775
    # source inverse, every cold-built object and all original tool receipts.
    previous=module(repository/'tests/indexed_f32/run.py','spatial_forward_current_profile',inputs.bind)
    built=previous.profile(repository,build,build_sha,inputs)
    for path in sorted((repository/'tests/spatial_forward').glob('*.py')):inputs.bind(path)
    inputs.bind(repository/'tests/pinned_reference_custody.py')
    for path in custody.source_files(repository):inputs.bind(path)
    return built


def reference(repository,directory,execution,inputs,allow=False):
    raw_report=inputs.bind(directory/'report.json');report=json.loads(raw_report)
    require(report['schema']=='heptatorch.spatial-forward.reference.v1' and report['case_sha256']==CASE_SHA256
        and report['expected_cases']==report['cases']==len(report['rows'])==len(CASES)==320
        and report['Torch_executed'] is True and report['native_executed'] is False
        and report['compiler_executed'] is False and report['failures']==[]
        and report['host_expected_values_computed'] is False and report['inputs_unchanged'] is True
        and report['rng_unchanged'] is True and report['cpu_capability']=='AVX2' and report['torch_threads']==1
        and report['all_observed_blobs_unchanged'] is True,
        'complete independent actual Torch spatial reference')
    wanted='OBSERVED_EXPLORATORY_SPATIAL_FORWARD' if allow else 'OBSERVED_SPATIAL_FORWARD'
    require(report['status']==wanted and report['production_oracle_qualified'] is (not allow),
        'strict original wheel provenance required unless explicitly exploratory')
    proof=custody.read(repository,directory,execution,report,inputs.bind,'tests/spatial_forward/reference.py',allow=allow)
    require(proof['original_eleven_checks_recomputed_from_raw_inputs'] is True
        and proof['actual_process_verified'] is True and proof['actual_supervisor_intent_verified'] is True
        and proof['actual_environment_transition_verified'] is True
        and proof['original_reference_script_bytes_verified'] is True
        and proof['full_installation_provenance_verified'] is (not allow),
        'all original eleven checks independently recomputed from complete raw installation entities')
    require(json.loads(inputs.bind(execution/'runner.stdout'))==dict(status=wanted,cases=320,expected_cases=320,
        production_oracle_qualified=not allow,error=None),'complete actual reference stdout')
    files=[repository/'tests/spatial_forward'/name for name in ('__init__.py','cases.py','reference.py')]
    files += [repository/'tests/numeric_predicates'/name for name in ('__init__.py','cases.py','reference.py')]
    files += [repository/'tests/pinned_reference_custody.py',repository/'tests/pytorch_oracle.py',
        repository/'compatibility/pytorch-oracle-lock.json',repository/'compatibility/upstream-lock.json',
        repository/'compatibility/pytorch-reference-requirements.txt',Path(report['actual_process']['python']['path'])]
    files+=custody.source_files(repository)
    if not allow:files.extend([repository/'build/pytorch-install.json',repository/'build/pytorch-reference-support-install.json'])
    require(set(report['inputs'])=={str(path.resolve()) for path in files},'complete actual reference source/input census')
    for path,record in report['inputs'].items():
        data=inputs.bind(path,record['sha256'])
        require(record==dict(path=path,bytes=len(data),sha256=sha(data),mode=stat.S_IMODE(Path(path).stat().st_mode)),
                'original reference input bytes and modes unchanged')
    def blob(item):
        require(set(item)=={'path','bytes','sha256'} and item['path']=='blobs/'+item['sha256']+'.bin','canonical actual reference blob')
        raw=inputs.bind(directory/item['path'],item['sha256']);require(len(raw)==item['bytes'],'complete actual reference blob')
        return raw
    for index,(row,spec) in enumerate(zip(report['rows'],CASES)):
        require(row==json.loads(inputs.bind(directory/('case-%04d.json'%index))) and row['index']==index and row['spec']==spec,
                'literal input recipe and complete original observation')
        require(row['inputs_unchanged'] is True and row['rng_unchanged'] is True and row['accepted'] is True
            and row['fresh_tensor'] is True and row['fresh_storage'] is True,'actual reference input/RNG and ownership predicates')
        require(len(row['inputs'])==len(spec['inputs']) and all((left is None)==(right is None)
            for left,right in zip(row['inputs'],spec['inputs'])),'exact optional input census')
        needs_grad=any(value is not None and value['requires_grad'] for value in spec['inputs'])
        require(('seed' in row)==needs_grad and row['result']['requires_grad'] is needs_grad,
                'actual reference backward participation matches original input requirements')
        if needs_grad:
            require(len(row['gradients'])==len(spec['inputs']) and all((observed is not None)==(value is not None and value['requires_grad'])
                for value,observed in zip(spec['inputs'],row['gradients'])),'complete independently observed input gradients')
        observed=row['inputs']+[row['result']]+([row['seed']]+row['gradients'] if needs_grad else [])
        for value in observed:
            if value is not None:
                for field in ('logical','storage'):blob(value[field])
    return dict(report=report,blob=blob,root=directory,execution=execution,report_sha256=sha(raw_report),custody=proof)


def qualify(report,repository,build,build_sha,output,inputs,allow_unverified_local=False):
    """Reapply every original predicate to raw frames/owners without native or Torch execution."""
    built=profile(repository,build,build_sha,inputs)
    require(report==json.loads(inputs.bind(output/'report.json')),'unchanged original native report')
    wanted='PASS_EXPLORATORY_NATIVE_ONLY' if allow_unverified_local else 'PASS'
    require(report['schema']=='heptatorch.spatial-forward.native.v1' and report['group']=='spatial_forward'
        and report['status']==wanted and report['cases']==report['expected_cases']==len(CASES)==320
        and report['case_sha256']==CASE_SHA256 and report['failures']==[]
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
                'every original observer source and input retained unchanged')
    fixed=reference(repository,Path(report['reference_directory']),Path(report['reference_process_directory']),inputs,allow_unverified_local)
    require(fixed['report_sha256']==report['reference_sha256'],'exact original independent actual Torch reference')
    wire=load_wire(repository,inputs.bind);cases=recipes(fixed)
    require(len(report['results'])==len(cases)==323,'complete numerical/control session census')
    totals=dict(frames=0,owner_reads=0,mapping_reads=0,checks=0)
    for saved,(name,runner,row) in zip(report['results'],cases):
        require(saved==json.loads(inputs.bind(output/name/'observation.json')) and saved['name']==name and saved['status']=='PASS',
                'exact original per-session observation')
        session=Replay(build/'heptatorch',wire,output/name,inputs.bind);observer=Observer(session,wire,fixed)
        result=runner(observer,row);finish=session.finish()
        require(result==saved['result'] and finish==saved['finish'] and observer.checks==saved['checks'],
                'every numerical metadata ownership lifetime and resource assertion reapplied')
        for key in totals:totals[key]+=observer.checks if key=='checks' else finish[key]
    require(totals==report['totals'] and all(report[key]==totals[key] for key in totals) and totals['mapping_reads']==24,
            'complete physical frame Owner and mapping census')
    ordinary=report['results'][:320];allocations=report['results'][320:]
    grad_cases=sum(any(value is not None and value['requires_grad'] for value in spec['inputs']) for spec in CASES)
    grad_tensors=sum(sum(value is not None and value['requires_grad'] for value in spec['inputs']) for spec in CASES)
    require(len(allocations)==3 and sum(row['result']['controls'] for row in ordinary)==28
        and sum(row['result']['gradient_checked'] for row in ordinary)==grad_cases
        and sum(row['result']['gradient_tensors_checked'] for row in ordinary)==grad_tensors
        and all(row['result']['actual_allocation_failures']==1 and row['result']['identical_request_recovered'] is True for row in allocations),
        'complete protocol, independent backward and actual OOM recovery coverage')
    return dict(cases=320,frames=totals['frames'],assertions=totals['checks'],native_sessions=323,
        owner_snapshots=totals['owner_reads'],mapping_snapshots=totals['mapping_reads'],physical_replay=True,
        native_executed_by_qualifier=False,Torch_executed_by_qualifier=False,
        complete_ci_admission=not allow_unverified_local,full_installation_provenance_verified=not allow_unverified_local,
        production_oracle_qualified=not allow_unverified_local,reference_sha256=fixed['report_sha256'],
        additional_protocol_controls=28,additional_successful_FE_transactions=3,additional_failed_FE_transactions=9,
        additional_allocation_recoveries=3,actual_autograd_cases=grad_cases,actual_gradient_tensors=grad_tensors,
        host_expected_values_computed=False,original_backward_functions_unchanged=True,
        original_requires_grad_dispatch_preserved=True,original_eleven_checks_recomputed_from_raw_inputs=True,
        actual_supervisor_intent_verified=True,actual_environment_transition_verified=True,
        original_reference_script_bytes_verified=True,portable_producer_path_roles=True,
        **{key:fixed['custody'][key] for key in ('official_wheel_sha256','official_member_anchor_sha256',
            'independent_official_wheel_member_anchor_verified','complete_official_wheel_archive_verified',
            'installed_wheel_members_verified','installed_wheel_bytes_verified','torch_shared_libraries_verified',
            'actual_mapped_torch_cpu_code_verified','all_actual_installed_inputs_unchanged','complete_installed_raw_snapshot_verified')})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-directory',type=Path,required=True);p.add_argument('--expected-build-sha256',required=True)
    p.add_argument('--torch-python',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-unverified-local',action='store_true',help='Exploratory only; never grants complete CI admission.')
    args=p.parse_args();require(not sys.flags.optimize,'all independent predicates must be enabled')
    output=args.output.resolve();build=args.build_directory.resolve();output.mkdir(parents=True,exist_ok=False)
    inputs=Inputs();started=time.monotonic()
    report=dict(schema='heptatorch.spatial-forward.native.v1',group='spatial_forward',status='FAIL',results=[],failures=[],
        cases=len(CASES),expected_cases=len(CASES),case_sha256=CASE_SHA256,frames=0,owner_reads=0,mapping_reads=0,checks=0,
        native_executed=False,Torch_executed=False,compiler_executed=False,host_expected_values_computed=False,
        original_reference_modified=False,actual_build_sha256=args.expected_build_sha256,
        full_pytorch_rewrite_complete=False,full_project_status='HOLD',runtime_qualification_transferred=False,
        allow_unverified_local=args.allow_unverified_local,production_oracle_qualified=False,
        full_installation_provenance_verified=False)
    try:
        inputs.bind(Path(sys.executable).resolve());built=profile(ROOT,build,args.expected_build_sha256,inputs)
        report.update(source_sha256=built['source_sha256'],native_sha256=built['native_sha256'])
        supervisor=module(ROOT/'tests/current_runtime_process.py','spatial_reference_supervisor',inputs.bind)
        execution=output/'reference-execution';execution.mkdir()
        argv=[str(args.torch_python.absolute()),'-B',str(ROOT/'tests/spatial_forward/reference.py'),'--output',str(output/'reference')]
        if args.allow_unverified_local:argv.append('--allow-unverified-local')
        process=custody.supervise(supervisor,argv,execution,300,ROOT,inputs.bind)
        dump(execution/'process.json',process)
        fixed=reference(ROOT,output/'reference',execution,inputs,args.allow_unverified_local)
        report.update(Torch_executed=True,reference_directory=str(fixed['root']),reference_process_directory=str(execution),
            reference_sha256=fixed['report_sha256'],production_oracle_qualified=not args.allow_unverified_local,
            full_installation_provenance_verified=not args.allow_unverified_local)
        wire=load_wire(ROOT,inputs.bind);cases=recipes(fixed);totals=dict(frames=0,owner_reads=0,mapping_reads=0,checks=0)
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
            if ordinal%32==0 or row['status']!='PASS':
                print(json.dumps(dict(group='spatial_forward',recipe=name,status=row['status'],error=row.get('error'))),flush=True)
        for value in list(sys.modules.values()):
            path=getattr(value,'__file__',None)
            if path and Path(path).is_file() and Path(path).resolve().is_relative_to(ROOT):inputs.bind(Path(path).resolve())
        inputs.finish();report.update(all_inputs_unchanged=True,totals=totals,**totals)
        if not report['failures'] and len(report['results'])==323:
            report['status']='PASS_EXPLORATORY_NATIVE_ONLY' if args.allow_unverified_local else 'PASS'
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    if report['status'] not in ('PASS','PASS_EXPLORATORY_NATIVE_ONLY'):
        report.update(production_oracle_qualified=False,full_installation_provenance_verified=False)
    report.update(inputs=inputs.files,elapsed_seconds=time.monotonic()-started);dump(output/'report.json',report)
    if report['status'] in ('PASS','PASS_EXPLORATORY_NATIVE_ONLY'):
        try:
            replay_inputs=Inputs();proof=qualify(report,ROOT,build,args.expected_build_sha256,output,replay_inputs,args.allow_unverified_local)
            replay_inputs.finish();dump(output/'physical-qualification.json',proof)
        except BaseException as error:
            dump(output/'physical-qualification.json',dict(status='FAIL',error=str(error),traceback=traceback.format_exc()))
            print(json.dumps(dict(status='FAIL_PHYSICAL_REPLAY',error=str(error))),flush=True);return 1
    print(json.dumps({key:report.get(key) for key in ('group','status','cases','frames','checks','failures','error')}),flush=True)
    return 0 if report['status'] in ('PASS','PASS_EXPLORATORY_NATIVE_ONLY') else 1


if __name__=='__main__':raise SystemExit(main())
