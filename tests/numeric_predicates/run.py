#!/usr/bin/env python3
"""Qualify five HIA predicates using a fresh fixed Torch reference and raw replay."""

import argparse
import importlib.util
import json
from pathlib import Path
import stat
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests'))
from numeric_predicates import evidence,native
from numeric_predicates.cases import CASES,CASE_SHA256

require,sha=evidence.require,evidence.sha


class Inputs:
    def __init__(self):
        self.records={}

    def bind(self,path,expected=None):
        path=Path(path).absolute();before=path.lstat()
        require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),'ordinary physical input: '+str(path))
        raw=path.read_bytes();after=path.stat()
        fields=lambda row:(row.st_dev,row.st_ino,row.st_mode,row.st_size,row.st_mtime_ns,row.st_ctime_ns)
        require(fields(before)==fields(after),'physical input changed while reading')
        row=evidence.identity(raw,stat.S_IMODE(after.st_mode))
        require(expected is None or sha(raw)==expected,'fixed input SHA: '+str(path))
        require(str(path) not in self.records or self.records[str(path)]==row,'previous physical input changed')
        self.records[str(path)]=row
        return raw

    def recheck(self):
        for path,row in list(self.records.items()):self.bind(path,row['sha256'])


def module(path,name,bind):
    bind(path);spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def profile(repository,build,build_sha,inputs):
    qualification=module(repository/'tools/current_runtime/qualification.py','predicate_current_profile',inputs.bind)
    built=qualification.admit_build(build,inputs.bind,build_sha)
    for file in sorted((repository/'tests/numeric_predicates').glob('*.py')):inputs.bind(file)
    for name in ('__init__.py','native.py','reference.py'):
        inputs.bind(repository/'tests/half_logical'/name)
    transition=module(repository/'tools/current_runtime/predicate_source_transition.py','predicate_source_proof',inputs.bind)
    bf16=module(repository/'tools/current_runtime/bf16_source_transition.py','predicate_bf16_source_proof',inputs.bind)
    mul=module(repository/'tools/current_runtime/half_mul_source_transition.py','predicate_half_mul_source_proof',inputs.bind)
    neg=module(repository/'tools/current_runtime/half_neg_source_transition.py','predicate_half_neg_source_proof',inputs.bind)
    addsub=module(repository/'tools/current_runtime/half_addsub_source_transition.py','predicate_half_addsub_source_proof',inputs.bind)
    current=inputs.bind(repository/'tools/current_runtime/source/runtime.hepta')
    combined=module(repository/'tools/current_runtime/index_signbit_source_transition.py','numeric_predicates_index_signbit_inverse',inputs.bind)
    current,combined_proof=combined.restore_source(current,inputs.bind)
    require(combined_proof['source_sha256']==built['source_sha256'] and combined_proof['functions']==778
        and combined_proof['parent_functions']==775 and combined_proof['unchanged_parent_functions']==771
        and combined_proof['changed_functions']==['entry','index_select','predicate_bits','predicate_command']
        and combined_proof['appended_functions']==['typed_index_select_previous','typed_index_select','typed_index_select_offset']
        and combined_proof['whole_source_inverse_exact'] is True
        and combined_proof['whole_round_trip_exact'] is True
        and combined_proof['admitted_new_opcode']==126 and combined_proof['opcode125_remains_rejected'] is True,
        'complete778 source inverse precedes every unchanged775 predecessor predicate')
    relu_path=repository/'tools/current_runtime/typed_relu_source_transition.py'
    inputs.bind(relu_path,'d06390c160833aa829c1a65cffa5ed8698d7aa23b72675c2942fca006f5350f9')
    relu=module(relu_path,'predicate_typed_relu_source_proof',inputs.bind)
    current,relu_extension=relu.restore_source(current,inputs.bind)
    require(relu_extension['source_sha256']==combined_proof['parent_source_sha256'] and relu_extension['functions']==775
        and relu_extension['parent_functions']==773 and relu_extension['unchanged_parent_functions']==772
        and relu_extension['changed_functions']==['unary']
        and relu_extension['appended_functions']==['typed_relu_value','typed_relu_tensor']
        and relu_extension['new_branch_operation']==5 and relu_extension['new_branch_dtypes']==[2,4,5,6]
        and relu_extension['whole_source_inverse_exact'] is True
        and relu_extension['whole_round_trip_exact'] is True,'exact typed ReLU775 successor source relation')
    same=module(repository/'tools/current_runtime/same_f64_arithmetic_source_transition.py','predicate_same_f64_arithmetic_source_proof',inputs.bind)
    current,same_extension=same.restore_source(current,inputs.bind)
    require(same_extension['source_sha256']==relu_extension['parent_source_sha256'] and same_extension['functions']==773
        and same_extension['parent_functions']==773 and same_extension['unchanged_parent_functions']==771
        and same_extension['changed_functions']==['half_mul_priority','public_f64_binary']
        and same_extension['whole_source_inverse_exact'] is True
        and same_extension['whole_round_trip_exact'] is True,'exact same-F64 arithmetic successor source relation')
    arithmetic=module(repository/'tools/current_runtime/public_f64_arithmetic_source_transition.py','predicate_public_f64_arithmetic_source_proof',inputs.bind)
    current,arithmetic_extension=arithmetic.restore_source(current,inputs.bind)
    require(arithmetic_extension['source_sha256']==same_extension['parent_source_sha256'] and arithmetic_extension['functions']==773
        and arithmetic_extension['parent_functions']==771
        and arithmetic_extension['changed_functions']==['binary','half_mul_cast']
        and arithmetic_extension['unchanged_parent_functions']==769
        and arithmetic_extension['whole_source_inverse_exact'] is True
        and arithmetic_extension['whole_round_trip_exact'] is True,'exact mixed public F64 source extension')
    division=module(repository/'tools/current_runtime/half_div_source_transition.py','predicate_half_div_source_proof',inputs.bind)
    current,division_extension=division.restore_source(current,inputs.bind)
    require(division_extension['source_sha256']==arithmetic_extension['parent_source_sha256'] and division_extension['functions']==771
        and division_extension['whole_source_inverse_exact'] is True,'exact Half true division source extension')
    wide=module(repository/'tools/current_runtime/wide_neg_source_transition.py','predicate_wide_neg_source_proof',inputs.bind)
    previous,wide_extension=wide.restore_source(current,inputs.bind)
    require(wide_extension['source_sha256']==division_extension['parent_source_sha256'] and wide_extension['functions']==767
        and wide_extension['whole_source_inverse_exact'] is True,'exact I64/F64 negation source extension')
    previous,addsub_extension=addsub.restore_source(previous,inputs.bind)
    require(addsub_extension['source_sha256']==wide_extension['parent_source_sha256'] and addsub_extension['functions']==765
        and addsub_extension['whole_source_inverse_exact'] is True,'exact Half add/sub source extension')
    previous,neg_extension=neg.restore_source(previous,inputs.bind)
    require(neg_extension['source_sha256']==addsub_extension['parent_source_sha256'] and neg_extension['functions']==761
        and neg_extension['whole_source_inverse_exact'] is True,'exact Half negation source extension')
    previous,mul_extension=mul.restore_source(previous,inputs.bind)
    require(mul_extension['source_sha256']==neg_extension['parent_source_sha256'] and mul_extension['functions']==758
        and mul_extension['whole_source_inverse_exact'] is True,'exact Half multiplication source extension')
    parent,extension=bf16.restore_source(previous,inputs.bind)
    require(extension['source_sha256']==mul_extension['parent_source_sha256'] and extension['functions']==742
        and extension['whole_source_inverse_exact'] is True,'exact BF16 copy source extension')
    proof=transition.proof(parent)
    require(proof['functions']==740 and proof['whole_source_inverse_exact'] is True
        and proof['dispatcher_guard']['exact_closed_range_verified'] is True
        and proof['old_path_delegation']['argument_order_preserved'] is True
        and proof['old_path_delegation']['complete_parent_body_unchanged'] is True,
        'exact old-source reconstruction and closed new route proof')
    return built


def qualify(report,repository,build,build_sha,output,inputs):
    """Reapply all comparisons to every original physical frame and full owner."""
    built=profile(repository,build,build_sha,inputs)
    require(report==json.loads(inputs.bind(output/'report.json')),'unchanged actual native observer report')
    require(report['schema']=='heptatorch.numeric-predicates.native-observation.v1'
        and report['status']=='PASS' and report['group']=='numeric_predicates'
        and report['cases']==report['expected_cases']==len(CASES)==485 and report['case_sha256']==CASE_SHA256
        and report['source_sha256']==built['source_sha256'] and report['native_sha256']==built['native_sha256']
        and report['actual_build_sha256']==build_sha and report['native_executed'] is True
        and report['Torch_executed'] is True and report['compiler_executed'] is False
        and report['host_expected_values_computed'] is False and report['original_reference_modified'] is False
        and report['all_inputs_unchanged'] is True and not report['failures'],'complete actual native observation identity and scope')
    for path,row in report['inputs'].items():
        raw=inputs.bind(path,row['sha256'])
        require(row==evidence.identity(raw,stat.S_IMODE(Path(path).stat().st_mode)),
                'all recorded original observer inputs retained unchanged')
    fixed=evidence.read(repository,Path(report['reference_directory']),Path(report['reference_process_directory']),inputs.bind)
    require(fixed['report_sha256']==report['reference_sha256'],'exact independent actual reference report retained')
    wire=native.load_wire(repository,inputs.bind);recipes=native.recipes(fixed)
    require(len(report['results'])==len(recipes),'all cases and additional controls completed')
    frames=checks=owner_reads=mapping_reads=0
    for saved,(name,recipe,parameters) in zip(report['results'],recipes):
        directory=output/name
        require(saved==json.loads(inputs.bind(directory/'observation.json'))
            and saved['name']==name and saved['status']=='PASS','complete original per-recipe observation')
        session=native.Replay(build/'heptatorch',wire,directory,inputs.bind)
        observer=native.Observer(session,wire,fixed);result=recipe(observer,*parameters);finish=session.finish()
        require(result==saved['result'] and observer.checks==saved['checks'] and finish==saved['finish'],
                'every original numerical metadata ownership and resource predicate reapplied')
        frames+=finish['frames'];checks+=observer.checks;owner_reads+=finish['owner_reads'];mapping_reads+=finish['mapping_reads']
    require(frames==report['frames'] and checks==report['checks'] and owner_reads==report['owner_reads']
        and mapping_reads==report['mapping_reads']==10,'complete original physical evidence census')
    inputs.recheck()
    return dict(cases=485,frames=frames,assertions=checks,native_sessions=len(recipes),owner_snapshots=owner_reads,
        mapping_snapshots=mapping_reads,physical_replay=True,native_executed_by_qualifier=False,
        Torch_executed_by_qualifier=False,reference_sha256=fixed['report_sha256'],
        additional_packet_controls=56,additional_successful_FE_transactions=5,
        additional_failed_FE_transactions=5,additional_allocation_recoveries=5,
        host_expected_values_computed=False,old_dispatcher_preserved=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-directory',type=Path,required=True)
    parser.add_argument('--expected-build-sha256',required=True)
    parser.add_argument('--torch-python',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require(not sys.flags.optimize,'independent predicates enabled')
    build,output=args.build_directory.resolve(),args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    inputs=Inputs();started=time.monotonic()
    report=dict(schema='heptatorch.numeric-predicates.native-observation.v1',status='FAIL',group='numeric_predicates',
        cases=len(CASES),expected_cases=len(CASES),case_sha256=CASE_SHA256,results=[],failures=[],
        native_executed=False,Torch_executed=False,compiler_executed=False,host_expected_values_computed=False,
        original_reference_modified=False,runtime_qualification_transferred=False,full_project_status='HOLD',
        frames=0,checks=0,owner_reads=0,mapping_reads=0,actual_build_sha256=args.expected_build_sha256)
    try:
        inputs.bind(Path(sys.executable).resolve())
        built=profile(ROOT,build,args.expected_build_sha256,inputs)
        report.update(source_sha256=built['source_sha256'],native_sha256=built['native_sha256'])
        supervisor=module(ROOT/'tests/current_runtime_process.py','predicate_reference_supervisor',inputs.bind)
        execution=output/'reference-execution';execution.mkdir()
        argv=[str(args.torch_python.absolute()),'-B',str(ROOT/'tests/numeric_predicates/reference.py'),
              '--repository',str(ROOT),'--output',str(output/'reference')]
        process=supervisor.execute(argv,execution,300,cwd=ROOT)
        native.dump(execution/'process.json',process)
        fixed=evidence.read(ROOT,output/'reference',execution,inputs.bind)
        report.update(Torch_executed=True,reference_directory=str(fixed['root']),
            reference_process_directory=str(execution),reference_sha256=fixed['report_sha256'])
        wire=native.load_wire(ROOT,inputs.bind);recipes=native.recipes(fixed)
        for ordinal,(name,recipe,parameters) in enumerate(recipes):
            row=dict(name=name,status='FAIL');session=observer=None
            try:
                session=native.Live(build/'heptatorch',wire,output/name,inputs.bind);report['native_executed']=True
                observer=native.Observer(session,wire,fixed);row['result']=recipe(observer,*parameters);row['status']='PASS'
            except BaseException as error:
                row['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            finally:
                if session is not None:
                    try:row['finish']=session.finish()
                    except BaseException as error:
                        row['status']='FAIL';row['finish_error']=dict(type=type(error).__name__,message=str(error))
                row['checks']=observer.checks if observer is not None else 0
                if (output/name).exists():native.dump(output/name/'observation.json',row)
            report['results'].append(row)
            for key in ('frames','owner_reads','mapping_reads'):report[key]+=row.get('finish',{}).get(key,0)
            report['checks']+=row['checks']
            if row['status']!='PASS':report['failures'].append(name)
            if ordinal%32==0 or row['status']!='PASS':
                print(json.dumps(dict(group='numeric_predicates',recipe=name,status=row['status'],error=row.get('error'))),flush=True)
        inputs.recheck();report['all_inputs_unchanged']=True
        report['status']='PASS' if not report['failures'] and len(report['results'])==len(recipes) else 'FAIL'
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    report['inputs']=inputs.records;report['elapsed_seconds']=time.monotonic()-started
    native.dump(output/'report.json',report)
    if report['status']=='PASS':
        try:
            proof=qualify(report,ROOT,build,args.expected_build_sha256,output,Inputs())
            native.dump(output/'physical-qualification.json',proof)
        except BaseException as error:
            native.dump(output/'physical-qualification.json',dict(status='FAIL',error=str(error),traceback=traceback.format_exc()))
            print(json.dumps(dict(status='FAIL_PHYSICAL_REPLAY',error=str(error))),flush=True)
            return 1
    print(json.dumps({key:report.get(key) for key in ('group','status','cases','frames','checks','failures','error')}),flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
