#!/usr/bin/env python3
"""Run one immutable original AsStrided reader with exact current admission."""
import argparse
import dis
from pathlib import Path
import os
import sys
import traceback
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
import importlib.util
_spec=importlib.util.spec_from_file_location('heptatorch_asstrided16_worker_common',HERE/'common.py')
common=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(common)
Inputs,SOURCE,IMAGE,SUITES,COUNTS=common.Inputs,common.SOURCE,common.IMAGE,common.SUITES,common.COUNTS
admit,derive,dump=common.admit,common.derive,common.dump
native_execution_fact,require,sha=common.native_execution_fact,common.require,common.sha


class FinishObservation:
    """Observe existing original return frames without replacing any function."""
    def __init__(self, original_client):
        self.original_client = str(original_client.resolve())
        self.events = []
        self.errors = []

    def __call__(self, frame, event, returned):
        if (event!='return' or frame.f_code.co_filename!=self.original_client
                or frame.f_code.co_name not in ('__init__','finish')):
            return
        try:
            values=frame.f_locals;session=values['self']
            original_class=frame.f_globals['OwnerSession']
            require(frame.f_code is original_class.__dict__[frame.f_code.co_name].__code__
                    and isinstance(session,original_class),'actual original method code identity')
            opcode=dis.opname[frame.f_code.co_code[frame.f_lasti]]
            row=dict(method=frame.f_code.co_name,pid=getattr(session,'pid',None),
                observer_pid=os.getpid(),directory=str(getattr(session,'directory',None)),
                executable_sha256=getattr(session,'binary_sha256',None),
                original_method_code_identity=True,return_opcode=opcode,
                normal_return=opcode in ('RETURN_VALUE','RETURN_CONST'))
            if frame.f_code.co_name=='finish':
                error,extra=values.get('error'),values.get('extra')
                row.update(native_exit_code=getattr(session,'exit_code',None),
                    stdout_EOF=extra==b'',stderr_EOF=error==b'',
                    stdout_bytes=len(extra) if isinstance(extra,bytes) else None,
                    stderr_bytes=len(error) if isinstance(error,bytes) else None,
                    stdout_sha256=sha(extra) if isinstance(extra,bytes) else None,
                    stderr_sha256=sha(error) if isinstance(error,bytes) else None,
                    input_closed=session.input is None,output_closed=session.output is None,
                    stderr_closed=session.stderr is None,
                    request_archive_closed=session.requests.closed,
                    response_archive_closed=session.responses.closed,
                    returned_None=returned is None)
            row['callback_completed']=True
            self.events.append(row)
        except BaseException as error:
            # A recording failure must never replace the original operation.
            # The independent admission below rejects incomplete observation.
            self.errors.append(repr(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('repository','build-directory','reference-directory','output'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--expected-build-sha256',required=True)
    parser.add_argument('--suite',required=True,choices=SUITES)
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    require(not sys.flags.optimize,'original assertions must stay enabled')
    repo,build,refs,out = (args.repository.resolve(),args.build_directory.resolve(),
                          args.reference_directory.resolve(),args.output.resolve())
    out.mkdir(parents=True,exist_ok=False)
    evidence = out/'adaptation';evidence.mkdir()
    actual_dir = out/'actual'
    inputs,edits = Inputs(),[]
    observation = FinishObservation(repo/'tests/retained_nonleaf_review/owner_client.py')
    report = dict(schema='heptatorch.asstrided16.current-worker.v1',suite=args.suite,status='RUNNING',
        source_sha256=SOURCE,native_sha256=IMAGE,actual_build_sha256=args.expected_build_sha256,
        cases=0,required_cases=COUNTS[args.suite],native_executed=False,driver_entered=False,
        Torch_executed=False,compiler_executed=False,old_pass_transferred=False,
        numeric_assertions_modified=False,owner_predicates_modified=False,
        full_pytorch_rewrite_complete=False)
    try:
        for file in sorted(HERE.glob('*.py')):inputs.bind(file)
        current,built,reference,reference_path,source_proof = admit(
            repo,build,args.expected_build_sha256,refs,args.suite,inputs)
        dump(evidence/'native-source-proof.json',source_proof)
        original_dir = repo/'tests/tensor_to_original16_review'
        observer_path = evidence/'native_observer_current.py'
        observer_raw,proof = derive(original_dir/'native_observer09.py',observer_path,[
            ('actual state tag','HTTOCP08','HTSINT02',1),
            ('actual backing tag','HTOTOCP8','HTOSIN02',1)],('Memory',),inputs)
        edits.append(proof)
        reader_path = original_dir/('replay_as_strided_'+args.suite+'.py')
        text = inputs.bind(reader_path).decode()
        start = text.index("    assert build['status'] == 'PASS'")
        end = text.index('    spec = importlib.util.spec_from_file_location',start)
        previous = text[start:end]
        require(sha(previous.encode())=='ace14ab9d8f9b629b69a26742adec59035a72bec1e08e62c55169fea15612539',
                'original build-only admission block changed')
        derived_path = evidence/reader_path.name
        reader_raw,proof = derive(reader_path,derived_path,[
            ('complete actual current cold-build admission',previous,
             '    asstrided16_admit(args, inputs)\n',1)],('main',),inputs)
        edits.append(proof)

        def reader_admit(reader_args, reader_inputs):
            require(reader_args.binary.resolve()==build/'heptatorch'
                    and reader_args.source.resolve()==build/'runtime.hepta'
                    and reader_args.build_report.resolve()==build/'build.json',
                    'original reader changed actual current build paths')
            require(reader_args.expected_source_sha256==SOURCE and reader_args.expected_native_sha256==IMAGE,
                    'original reader current identities differ')
            current.admit_build(build,inputs.bind,args.expected_build_sha256)
            for path,row in list(inputs.rows.items()):
                raw = inputs.bind(Path(path),row['sha256'])
                key = Path(path).resolve()
                require(key not in reader_inputs or reader_inputs[key]==raw,
                        'original reader previously bound different bytes')
                reader_inputs[key] = raw

        driver = types.ModuleType('original_asstrided16_'+args.suite)
        driver.__file__ = str(reader_path)
        driver.asstrided16_admit = reader_admit
        exec(compile(reader_raw,str(derived_path),'exec'),driver.__dict__)
        report['source_proof']=source_proof
        report['prepared_argv']=[str(reader_path),'--binary',str(build/'heptatorch'),
            '--source',str(build/'runtime.hepta'),'--build-report',str(build/'build.json'),
            '--protocol',str(repo/'tests/retained_nonleaf_review/protocol.py'),
            '--observer',str(observer_path),'--reference',str(reference_path),
            '--expected-source-sha256',SOURCE,'--expected-native-sha256',IMAGE,
            '--expected-observer-sha256',sha(observer_raw),'--output',str(actual_dir)]
        dump(evidence/'source-inverses.json',dict(edits=edits,whole_file_inverse_exact=True,
            only_build_admission_and_owner_tag_literals_changed=True,
            numeric_expectations_modified=False,recipes_modified=False))
        if args.prepare_only:
            report['status']='PREPARED_NOT_RUN'
        else:
            require(sys.getprofile() is None,'no preexisting profiler may be replaced')
            report['driver_entered']=True
            report['native_executed']=None
            dump(out/'worker.json',report)
            sys.argv=report['prepared_argv']
            sys.setprofile(observation)
            try:
                code=driver.main()
            finally:
                sys.setprofile(None)
            report['reader_exit_code']=code
            report['cases']=COUNTS[args.suite] if code==0 else None
            require(code==0,'original unchanged numerical reader returned failure')
            require(observation.errors==[], 'read-only original finish observation incomplete')
            report['status']='PASS_ORIGINAL_REPLAY'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
    finally:
        captured=dict(schema='heptatorch.asstrided16.original-finish-observation.v1',
            method='read-only Python return-frame profiler; original functions and syscalls unchanged',
            original_client_sha256='dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b',
            original_finish_sha256='487d4015daab8471dc8142431e68045e036aefa260aab0d501cad96ddbb05c40',
            events=observation.events,errors=observation.errors,
            original_finish_replaced=False,syscalls_replaced=False)
        dump(out/'original-finish-observations.json',captured)
        inputs.bind(out/'original-finish-observations.json')
        report['native_executed']=native_execution_fact(actual_dir,report['driver_entered'])
        try:
            inputs.recheck()
            report['all_inputs_unchanged']=True
        except BaseException as error:
            report.update(status='FAIL',all_inputs_unchanged=False,input_error=repr(error))
        report['inputs']=inputs.rows
        report['source_inverses']=edits
        dump(out/'worker.json',report)
    print(__import__('json').dumps({key:report.get(key) for key in
          ('status','suite','cases','native_executed','error')}))
    return int(report['status']=='FAIL')


if __name__=='__main__':
    raise SystemExit(main())
