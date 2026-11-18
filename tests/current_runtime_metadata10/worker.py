#!/usr/bin/env python3
"""Execute the fixed v2 reader with only two identity edits and passive evidence."""
import argparse
import ast
import dis
import errno
import importlib.util
import json
import lzma
import os
from pathlib import Path
import stat
import struct
import sys
import time
import traceback
import types
import warnings

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('metadata10_worker_common',HERE/'common.py')
common=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(common)


def code_bytes(code):
    """Serialize code values, never marshal's incidental reference-count flags."""
    def encode(value):
        kind=type(value)
        if value is None:return ['none']
        if value is Ellipsis:return ['ellipsis']
        if kind is bool:return ['bool',value]
        if kind is int:return ['int',value]
        if kind is str:return ['str',value]
        if kind is bytes:return ['bytes',value.hex()]
        if kind is float:return ['float64_bits',struct.pack('<d',value).hex()]
        if kind is complex:return ['complex128_bits',struct.pack('<2d',value.real,value.imag).hex()]
        if kind is tuple:return ['tuple',[encode(item) for item in value]]
        if kind is frozenset:
            return ['frozenset',sorted((encode(item) for item in value),key=common.canonical)]
        if kind is types.CodeType:
            fields={}
            for name in dir(value):
                if not name.startswith('co_'):continue
                item=getattr(value,name)
                if not callable(item):fields[name]=encode(item)
            return ['code',fields,list(value.co_lines()),list(value.co_positions())]
        raise ValueError('unsupported exact code constant: '+kind.__name__)
    common.require(type(code) is types.CodeType,'the observed identity must be a complete code object')
    # co_lnotab is derived but included for completeness; inspecting it must not
    # add a deprecation warning to the original worker's stderr stream.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',DeprecationWarning)
        return common.canonical(encode(code))


def fixed_code(path,qualified,pin):
    raw=path.read_bytes();tree=ast.parse(raw);class_name,method_name=qualified.split('.')
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==method_name)
    segment=ast.get_source_segment(raw.decode(),node).encode()
    common.require(common.sha(segment)==pin['sha256'] and len(segment)==pin['bytes'],
                   'entire original observed method source')
    top=compile(raw,str(path),'exec')
    compiled_class=next(v for v in top.co_consts if isinstance(v,types.CodeType) and v.co_name==class_name)
    code=next(v for v in compiled_class.co_consts if isinstance(v,types.CodeType) and v.co_name==method_name)
    return code,common.sha(code_bytes(code))


class PassiveObservations:
    """Observe exact original normal returns; never replace methods or send bytes."""
    def __init__(self,repository,output,pins):
        self.repository=repository;self.output=output;self.pins=pins
        self.events=[];self.errors=[];self.active={};self.calls=0;self.callback_seconds=0.0
        self.methods={};self.code_expectations={}
        (output/'owners').mkdir(parents=True,exist_ok=False)
        for name in ('Native.tensor','OwnerSession.finish'):
            pin=pins['function_sources'][name];path=(repository/pin['path']).resolve()
            code,digest=fixed_code(path,name,pin)
            self.methods[(str(path),name.split('.')[1])]=(name,code,digest)
            self.code_expectations[name]=dict(source_file=pin['path'],function_source_sha256=pin['sha256'],
                code_sha256=digest,code_identity_format='CANONICAL_CODE_ATTRIBUTES_V1',
                source_file_sha256=pins['dependencies'][pin['path']]['sha256'])

    def __call__(self,frame,event,arg):
        if event not in ('call','return'):return
        selected=self.methods.get((frame.f_code.co_filename,frame.f_code.co_name))
        if selected is None:return
        started=time.monotonic();self.calls+=1
        try:
            name,expected_code,expected_digest=selected
            class_name,method=name.split('.')
            original=getattr(frame.f_globals[class_name],method)
            common.require(frame.f_code is original.__code__,
                'the actual call must use the exact original method code object')
            common.require(frame.f_code==expected_code and code_bytes(frame.f_code)==code_bytes(expected_code),
                'all original code attributes, constants and instruction/position bytes must agree')
            instance=frame.f_locals['self']
            session=instance.s if name=='Native.tensor' else instance
            if event=='call':
                common.require(id(frame) not in self.active,'one observed entry for this frame')
                observed=dict(method=name,pid=session.pid,start_frame=session.frames,
                    case_directory=str(instance.folder if name=='Native.tensor' else session.directory.parent))
                if name=='OwnerSession.finish':
                    descriptors={key:getattr(session,key) for key in ('input','output','stderr')}
                    common.require(all(type(fd) is int and fd>=0 for fd in descriptors.values())
                        and len(set(descriptors.values()))==3,'all three original transport descriptors exist at finish entry')
                    identities={}
                    for key,fd in descriptors.items():
                        item=os.fstat(fd)
                        identities[key]=dict(fd=fd,device=item.st_dev,inode=item.st_ino,mode=item.st_mode)
                    observed['entry_descriptors']=identities
                self.active[id(frame)]=observed
                return
            observed=self.active.pop(id(frame))
            opcode=dis.opname[frame.f_code.co_code[frame.f_lasti]]
            observed.update(normal_return=opcode in ('RETURN_VALUE','RETURN_CONST'),return_opcode=opcode,
                original_method_code_identity=True,code_sha256=expected_digest,
                function_source_sha256=self.pins['function_sources'][name]['sha256'],
                end_frame=session.frames,observer_pid=os.getpid())
            if not observed['normal_return']:
                observed['callback_completed']=False;self.events.append(observed)
                raise ValueError('original observed method exited through an exception')
            if name=='Native.tensor':
                common.require(isinstance(arg,tuple) and len(arg)==2 and isinstance(arg[0],dict),
                               'the original tensor result must return normally')
                owner=frame.f_locals['owner'];raw=owner['bytes'];digest=common.sha(raw)
                target=self.output/'owners'/(digest+'.owner.xz')
                if not target.exists():
                    with target.open('xb') as stream:stream.write(lzma.compress(raw,preset=1))
                caller=frame.f_back
                observed.update(token=frame.f_locals['token'],tensor=json.loads(common.canonical(arg[0])),
                    descriptor_words=list(arg[1]),owner=dict(file='owners/'+target.name,pointer=owner['pointer'],
                        bytes=len(raw),sha256=digest,compressed_sha256=common.sha(target.read_bytes())),
                    caller=caller.f_code.co_name,caller_label=caller.f_locals.get('label'),
                    caller_token=caller.f_locals.get('token'),caller_gradient=caller.f_locals.get('gradient'))
                common.require(observed['end_frame']==observed['start_frame']+3,
                               'the unchanged tensor method must issue exactly 13/96/98')
            else:
                error,extra=frame.f_locals['error'],frame.f_locals['extra']
                common.require(type(error) is bytes and type(extra) is bytes,'the original finish has both actual reads')
                closed={}
                for key,item in observed['entry_descriptors'].items():
                    try:os.fstat(item['fd'])
                    except OSError as failure:
                        common.require(failure.errno==errno.EBADF,'the same original descriptor is closed')
                        closed[key]=True
                    else:raise ValueError('original transport descriptor remained open after normal finish')
                observed.update(returned_None=arg is None,exit_code=session.exit_code,
                    expected_exit=frame.f_locals['expected'],stderr_hex=error.hex(),stdout_tail_hex=extra.hex(),
                    stdin_closed=session.input is None,stdout_closed=session.output is None,
                    stderr_closed=session.stderr is None,requests_closed=session.requests.closed,
                    responses_closed=session.responses.closed,actual_binary_sha256=session.binary_sha256,
                    same_entry_descriptors_closed=closed)
                common.require(arg is None and session.exit_code==frame.f_locals['expected']==0
                    and error==extra==b'' and all(observed[key] is True for key in
                    ('stdin_closed','stdout_closed','stderr_closed','requests_closed','responses_closed')),
                    'both actual empty reads and all closed native transport descriptors')
            observed['callback_completed']=True;self.events.append(observed)
        except BaseException as error:
            self.errors.append(dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        finally:self.callback_seconds+=time.monotonic()-started

    def report(self):
        return dict(schema='heptatorch.cat-metadata10.passive-original-observations.v1',
            methods=self.code_expectations,events=self.events,errors=self.errors,
            unmatched_calls=len(self.active),callback_calls=self.calls,callback_seconds=self.callback_seconds,
            methods_replaced=False,syscalls_replaced=False,extra_native_commands=0,
            classification='EXACT_ORIGINAL_CODE_NORMAL_RETURN_OBSERVATIONS_ONLY')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,required=True)
    parser.add_argument('--build-directory',type=Path,required=True)
    parser.add_argument('--expected-build-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(argv)
    common.require(not sys.flags.optimize,'all original assertions remain enabled')
    repo,build,out=map(lambda value:value.resolve(),(args.repository,args.build_directory,args.output))
    out.mkdir(parents=True,exist_ok=False)
    binding_path=repo/'tests/current_runtime_original35/binding.py'
    spec=importlib.util.spec_from_file_location('metadata10_input_binding',binding_path)
    binding=importlib.util.module_from_spec(spec);spec.loader.exec_module(binding)
    inputs=binding.Inputs();original=None;observer=None;exit_code=None
    report=dict(schema='heptatorch.current-cat.metadata10-worker.v1',status='RUNNING',
        prepare_only=args.prepare_only,worker_pid=os.getpid(),worker_argv=list(sys.orig_argv),
        native_executed=False,Torch_executed=False,compiler_executed=False,
        original_reader_replaced=False,original_compare_changed=False,old_FAIL_modified=False,
        full_pytorch_rewrite_complete=False,schema_closure=False,full_project_status='HOLD')
    previous_argv,previous_path,previous_profile=sys.argv[:],sys.path[:],sys.getprofile()
    try:
        for name in ('worker.py','common.py','qualify.py','readback.py','source.json','frozen/native_v2.py'):inputs.bind(HERE/name)
        inputs.bind(binding_path)
        python=Path(sys.executable).resolve();python_raw=inputs.bind(python)
        report['worker_interpreter']=dict(requested_path=sys.executable,original_path=str(python),bytes=len(python_raw),sha256=common.sha(python_raw),
            mode=stat.S_IMODE(python.stat().st_mode),version=sys.version,role='THIS_WORKER_INTERPRETER_NOT_REFERENCE_INTERPRETER')
        pins,built=common.admit(repo,build,args.expected_build_sha256,inputs)
        reference,recipes,roles,reference_proof=common.reference_package(repo,out/'reference',inputs,restore=True)
        report.update(source_sha256=built['source_sha256'],native_sha256=built['native_sha256'],
            actual_build_sha256=args.expected_build_sha256,functions=built['functions'],units=built['units_planned'],
            reference_binding=reference_proof)
        source,proof=common.adapt_driver(built,inputs)
        adaptation=out/'adaptation';adaptation.mkdir()
        driver_path=adaptation/'native_v2_current.py'
        with driver_path.open('xb') as stream:stream.write(source)
        inputs.bind(driver_path,proof['derived_sha256']);common.dump(adaptation/'source-inverse.json',proof)
        driver=types.ModuleType('metadata10_exact_original_driver');driver.__file__=str(driver_path)
        exec(compile(source,str(driver_path),'exec'),driver.__dict__)
        observer=PassiveObservations(repo,out/'trace',pins)
        common.require(previous_profile is None,'no other profiler may be displaced')
        sys.argv=[str(driver_path),'--repository',str(repo),'--build-directory',str(build),
            '--expected-build-sha256',args.expected_build_sha256,'--reference',str(roles['report']),
            '--producer',str(roles['producer']),'--output',str(out/'actual')]
        if args.prepare_only:sys.argv.append('--prepare-only')
        report['native_executed']=False if args.prepare_only else None
        sys.setprofile(observer)
        try:exit_code=driver.main()
        finally:sys.setprofile(previous_profile)
        common.require(exit_code==0,'unchanged original metadata reader must return success')
        report['status']='PREPARED_NOT_EXECUTED' if args.prepare_only else 'PASS_METADATA10_EXECUTED_PENDING_PURE_ADMISSION'
    except BaseException as error:
        report.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:
        sys.setprofile(previous_profile);sys.argv[:]=previous_argv;sys.path[:]=previous_path
        if observer is not None:
            observations=observer.report();common.dump(out/'original-method-observations.json',observations)
            report['instrumentation']=dict(path='original-method-observations.json',
                sha256=common.sha((out/'original-method-observations.json').read_bytes()),
                events=len(observer.events),callback_seconds=observer.callback_seconds,extra_native_commands=0)
            if observer.errors or observer.active:report['status']='FAIL'
            if any(event.get('method')=='Native.tensor' and event.get('callback_completed') is True for event in observer.events):
                report['native_executed']=True
        if (out/'actual/report.json').is_file():
            raw=inputs.bind(out/'actual/report.json');original=json.loads(raw)
            report['original_report']=dict(path='actual/report.json',bytes=len(raw),sha256=common.sha(raw),status=original.get('status'))
            if original.get('native_executed') is True:report['native_executed']=True
            elif report['native_executed'] is not True and args.prepare_only:report['native_executed']=False
            required='PREPARED_NOT_EXECUTED' if args.prepare_only else 'MATCH_ALL_TEN_FIXED_REFERENCE_FIELDS'
            if original.get('status')!=required or original.get('all_inputs_unchanged') is not True:report['status']='FAIL'
        elif exit_code==0:report['status']='FAIL'
        changed=inputs.recheck();report.update(inputs=inputs.records,all_inputs_unchanged=not changed,input_recheck_errors=changed,
            original_reader_exit_code=exit_code)
        if changed:report['status']='FAIL'
        common.dump(out/'report.json',report)
    print(json.dumps(dict(status=report['status'],native_executed=report['native_executed'],
                          original_status=None if original is None else original.get('status'))))
    return 0 if report['status'] in ('PREPARED_NOT_EXECUTED','PASS_METADATA10_EXECUTED_PENDING_PURE_ADMISSION') else 1


if __name__=='__main__':raise SystemExit(main())
