#!/usr/bin/env python3
"""Pure physical reconstruction of the fixed ten cat metadata comparisons."""
import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path
import stat
import struct
import sys
import traceback

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('metadata10_qualification_common',HERE/'common.py')
common=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(common)


def qualify_metadata10(report,repository,build,build_sha,directory,inputs,execution_directory):
    common.require(not sys.flags.optimize,'all original assertions remain enabled')
    repository,build,directory,execution_directory=map(lambda p:Path(p).resolve(),
        (repository,build,directory,execution_directory))
    for name in ('worker.py','common.py','qualify.py','readback.py','source.json','frozen/native_v2.py'):inputs.bind(HERE/name)
    common.require(json.loads(inputs.bind(directory/'report.json'))==report,'original worker report bytes')
    common.require(report['schema']=='heptatorch.current-cat.metadata10-worker.v1'
        and report['status']=='PASS_METADATA10_EXECUTED_PENDING_PURE_ADMISSION'
        and report['prepare_only'] is False and report['native_executed'] is True
        and report['Torch_executed'] is False and report['compiler_executed'] is False
        and report['original_reader_replaced'] is False and report['original_compare_changed'] is False
        and report['old_FAIL_modified'] is False and report['all_inputs_unchanged'] is True
        and report['input_recheck_errors']==[] and report['original_reader_exit_code']==0
        and report['full_pytorch_rewrite_complete'] is False and report['schema_closure'] is False
        and report['full_project_status']=='HOLD' and not report.get('error'),
        'complete original metadata10 worker with no scope or expectation changes')
    runner=json.loads(inputs.bind(execution_directory/'runner.json'))
    common.require(runner['started'] is True and runner['pid']==report['worker_pid']
        and runner['exit_code']==0 and runner['stdout_EOF'] is True and runner['stderr_EOF'] is True
        and runner['stderr_bytes']==0 and all(not runner.get(key) for key in ('timed_out','forced_cleanup',
            'error','execution_error','cleanup_errors','running_descendants_after_cleanup')),
        'actual complete worker process, both EOFs and no cleanup')
    for stream in ('stdout','stderr'):
        raw=inputs.bind(execution_directory/('runner.'+stream),runner[stream+'_sha256'])
        common.require(len(raw)==runner[stream+'_bytes'],'complete original worker stream')
    for path,pin in report['inputs'].items():
        raw=inputs.bind(Path(path),pin['sha256'])
        common.require(len(raw)==pin['bytes'] and stat.S_IMODE(Path(path).stat().st_mode)==pin['mode'],
                       'every original worker input physically unchanged')
    python=report['worker_interpreter'];raw=inputs.bind(Path(python['original_path']),python['sha256'])
    common.require(len(raw)==python['bytes'] and stat.S_IMODE(Path(python['original_path']).stat().st_mode)==python['mode']
        and python['role']=='THIS_WORKER_INTERPRETER_NOT_REFERENCE_INTERPRETER',
        'actual worker interpreter identity remains distinct from fixed reference interpreter custody')
    argv=runner['argv']
    common.require(argv==report['worker_argv'] and len(argv)==13
        and argv[0]==python['requested_path'] and str(Path(argv[0]).resolve())==python['original_path']
        and argv[1:4]==['-I','-S','-B'] and argv[4]==str(HERE/'worker.py'),
        'actual isolated worker command, exact script and physically bound interpreter')
    pairs=list(zip(argv[5::2],argv[6::2]))
    common.require(len(pairs)==4 and len({key for key,value in pairs})==4
        and dict(pairs)=={'--repository':str(repository),'--build-directory':str(build),
            '--expected-build-sha256':build_sha,'--output':str(directory)},
        'actual worker argv binds this repository, exact cold receipt and original output directory without extra options')
    pins,built=common.admit(repository,build,build_sha,inputs)
    common.require(report['source_sha256']==built['source_sha256'] and report['native_sha256']==built['native_sha256']
        and report['actual_build_sha256']==build_sha and report['functions']==built['functions']
        and report['units']==built['units_planned'],'same strictly admitted actual complete cold image')
    reference,recipes,roles,reference_proof=common.reference_package(repository,directory/'reference',inputs)
    common.require(report['reference_binding']==reference_proof,'all original reference data custody must recompute exactly')
    derived,source_proof=common.adapt_driver(built,inputs)
    driver=directory/'adaptation/native_v2_current.py'
    common.require(inputs.bind(driver)==derived and json.loads(inputs.bind(directory/'adaptation/source-inverse.json'))==source_proof,
                   'entire executed v2 reader must invert by precisely two identity literals')
    original_raw=inputs.bind(directory/'actual/report.json',report['original_report']['sha256'])
    original=json.loads(original_raw)
    common.require(len(original_raw)==report['original_report']['bytes']
        and report['original_report']['path']=='actual/report.json'
        and original['schema']=='heptatorch.current-cat.actual-reference10.v1'
        and original['status']==report['original_report']['status']=='MATCH_ALL_TEN_FIXED_REFERENCE_FIELDS'
        and original['source_sha256']==built['source_sha256'] and original['native_sha256']==built['native_sha256']
        and original['actual_build_sha256']==build_sha and original['reference_sha256']==pins['reference_sha256']
        and original['producer_sha256']==pins['producer_sha256'] and original['native_executed'] is True
        and original['Torch_executed'] is False and original['compiler_executed'] is False
        and original['old_reports_modified'] is False and original['full_pytorch_rewrite_complete'] is False
        and original['all_inputs_unchanged'] is True and original['input_recheck_errors']==[],
        'all ten actual original comparisons must succeed on this source')
    observations=json.loads(inputs.bind(directory/'original-method-observations.json',report['instrumentation']['sha256']))
    common.require(observations['schema']=='heptatorch.cat-metadata10.passive-original-observations.v1'
        and observations['errors']==[] and observations['unmatched_calls']==0
        and observations['methods_replaced'] is False and observations['syscalls_replaced'] is False
        and observations['extra_native_commands']==report['instrumentation']['extra_native_commands']==0
        and observations['classification']=='EXACT_ORIGINAL_CODE_NORMAL_RETURN_OBSERVATIONS_ONLY'
        and observations['callback_calls']==2*len(observations['events'])
        and report['instrumentation']['events']==len(observations['events'])
        and report['instrumentation']['callback_seconds']==observations['callback_seconds']
        and math.isfinite(observations['callback_seconds']) and observations['callback_seconds']>=0,
        'strict passive observation with every exact original call and normal return')
    common.require(set(observations['methods'])=={'Native.tensor','OwnerSession.finish'},'only the two original observed methods')
    code_reader=common.load(HERE/'worker.py','metadata10_exact_code_identity_reader',inputs)
    for method,item in observations['methods'].items():
        source=pins['function_sources'][method]
        unused,expected_code_sha=code_reader.fixed_code(repository/source['path'],method,source)
        common.require(item['source_file']==source['path'] and item['function_source_sha256']==source['sha256']
            and item['source_file_sha256']==pins['dependencies'][source['path']]['sha256']
            and item['code_identity_format']=='CANONICAL_CODE_ATTRIBUTES_V1'
            and item['code_sha256']==expected_code_sha,
            'complete fixed source identity for each actually observed method')
    for event in observations['events']:
        common.require(event['method'] in observations['methods'] and event['normal_return'] is True
            and event['return_opcode'] in ('RETURN_VALUE','RETURN_CONST')
            and event['original_method_code_identity'] is True and event['callback_completed'] is True
            and event['observer_pid']==runner['pid'] and event['pid']!=runner['pid']
            and event['code_sha256']==observations['methods'][event['method']]['code_sha256']
            and event['function_source_sha256']==pins['function_sources'][event['method']]['sha256'],
            'every event comes from the exact original method and a normal return')
    saved_path=sys.path[:];names=('binding','prepare','current_runtime_fixtures')
    saved={name:sys.modules.get(name) for name in names}
    try:
        cat4=common.load(repository/'tests/current_runtime_cat4/qualify.py','metadata10_fixed_cat4_owner_custody',inputs,common.CAT4_QUALIFIER)
        captured=cat4.CapturedInputs(inputs)
        captured.bind(driver)
        common.load(repository/'tests/current_runtime_original35/binding.py','binding',captured)
        captured.bind(roles['report'],pins['reference_sha256']);captured.bind(roles['producer'],pins['producer_sha256'])
        adapter=common.load(repository/'tests/current_runtime270.py','metadata10_fixed_owner_adapter',captured)
        fixtures=common.load(repository/'tests/current_runtime_fixtures.py','current_runtime_fixtures',captured)
        package=directory/'actual/original-owner-package'
        package_proof=cat4.check_original_package(repository,package,fixtures,captured)
        rebuilt,original_binding=adapter.make_bindings(package,build,build_sha,captured)
        common.require(rebuilt==built,'the same entire cold receipt through the unchanged original adapter')
        owner,owner_proof,owner_raw=sys.modules['prepare'].owner_module(built['source_sha256'],built['native_sha256'])
        common.require(captured.bind(directory/'actual/owner-current.py')==owner_raw,
                       'all original complete owner predicates, with exact admitted identity only')
        common.require(json.loads(inputs.bind(directory/'actual/source-binding.json'))==dict(current=original_binding,owner=owner_proof),
                       'full actual original source/owner inverse must recompute')
        for relative in ('tests/logical_reductions/cases.py','tests/logical_reductions/native.py',
                         'tests/differentiable_view_native_review/protocol.py','tests/differentiable_view_native_review/owner_client.py'):
            captured.bind(repository/relative,pins['dependencies'][relative]['sha256'])
        common.require(captured.records==original['inputs'],'complete original v2 input census must physically match')
        wire=common.load(repository/'tests/differentiable_view_native_review/protocol.py','metadata10_readback_wire',inputs)
        native_raw=inputs.bind(repository/'tests/logical_reductions/native.py',pins['dependencies']['tests/logical_reductions/native.py']['sha256'])
        tree=ast.parse(native_raw)
        selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('require','node')]
        common.require([node.name for node in selected]==['require','node'],'both exact original descriptor helper functions')
        namespace={'struct':struct,'BASE':139264}
        exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(repository/'tests/logical_reductions/native.py'),'exec'),namespace)
        reader_module=common.load(HERE/'readback.py','metadata10_pure_frame_reader',inputs)
        compare=common.comparison_function(inputs)
        common.require(len(original['rows'])==len(reference['rows'])==len(recipes)==10,'all ten fixed recipes')
        rows=[];pids=set();all_trace_owners=set();used_events=0
        for ordinal,(recipe,row,wanted) in enumerate(zip(recipes,original['rows'],reference['rows'])):
            common.require(row['name']==wanted['name']==recipe['name'] and row['recipe']==wanted['recipe']==recipe
                and row['status']=='MATCH_FIXED_REFERENCE_FIELDS' and row['differences']==[]
                and row['native_executed'] is True and not row.get('error') and not row.get('cleanup_error')
                and wanted['observation_complete'] is True and wanted['rng_unchanged'] is True,
                'same complete actual and fixed-reference case identities')
            folder=directory/'actual'/('%02d-'%ordinal+recipe['name'])
            common.require(json.loads(inputs.bind(folder/'case.json'))==row,'complete original per-case report')
            common.require(row['process_id'] not in pids,'ten distinct original native children');pids.add(row['process_id'])
            events=[event for event in observations['events'] if event['pid']==row['process_id']]
            common.require(events and all(event['case_directory']==str(folder) for event in events),
                           'every physical event belongs to this actual native child and case')
            used_events+=len(events)
            reader=reader_module.Reader(common,wire,owner,namespace['node'],folder,directory/'trace',events,row,inputs)
            common.require(reader.ledger['executable_sha256']==built['native_sha256'],'actual native child image')
            actual,counts=reader.replay(recipe);differences=[]
            for label in ('inputs_before_forward','inputs_after_forward','inputs_after_backward_or_skip',
                          'inputs_after_releasing_output_and_seed'):
                common.require(len(actual[label])==len(wanted[label]),'complete original Tensor metadata list')
                for i,(native,expected) in enumerate(zip(actual[label],wanted[label])):
                    compare(native,expected,label+'/'+str(i),differences)
            common.require(wanted['forward']['status']=='RETURNED','fixed observed successful cat')
            compare(actual['forward_output'],wanted['forward']['output'],'forward/output',differences)
            if recipe['backward']:
                common.require(wanted['backward']['status']=='RETURNED','fixed observed successful backward')
                for native_key,ref_key in (('seed_before','seed_before'),('seed_after','seed_after'),('output_after_backward','output_after')):
                    compare(actual[native_key],wanted['backward'][ref_key],'backward/'+ref_key,differences)
            else:common.require(wanted['backward']['status']=='NOT_REQUESTED_FORWARD_ONLY','original forward-only recipe')
            common.require(differences==row['differences']==[], 'all original metadata including unshifted gradient versions must match fixed Torch')
            all_trace_owners.update(counts['trace_owner_archives']);rows.append(dict(name=recipe['name'],**counts))
        common.require(used_events==len(observations['events']) and len(pids)==10,'all ten complete child method event streams')
        common.require(all_trace_owners=={str(path.relative_to(directory/'trace')) for path in (directory/'trace/owners').glob('*.owner.xz')},
                       'every additional physical owner archive is used by an actual tensor read')
        return dict(cases=10,frames=sum(row['frames'] for row in rows),native_sessions=10,native_EOF_observations=20,
            tensor_observations=sum(row['tensor_observations'] for row in rows),owner_observations=sum(row['owner_observations'] for row in rows),
            original_assertions=sum(row['checks'] for row in rows),trace_owner_archives=len(all_trace_owners),
            original_inputs_rechecked=len(captured.records),original_compare_source_sha256=pins['function_sources']['compare']['sha256'],
            original_owner_source_inverse=owner_proof,original_owner_package=package_proof,source_inverse=source_proof,
            reference_binding=reference_proof,case_readbacks=rows,gradient_version_offsets_applied=False,
            new_native_executed_by_qualification=False,Torch_executed_by_qualification=False)
    finally:
        sys.path[:]=saved_path
        for name,value in saved.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repository','build-directory','result-directory','execution-directory','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-build-sha256',required=True);args=parser.parse_args(argv)
    common.require(not args.output.exists(),'new pure readback output required')
    path=args.repository.resolve()/'tests/current_runtime_original35/binding.py'
    spec=importlib.util.spec_from_file_location('metadata10_readback_inputs',path)
    binding=importlib.util.module_from_spec(spec);spec.loader.exec_module(binding)
    inputs=binding.Inputs();result=dict(schema='heptatorch.cat-metadata10.pure-readback.v1',status='FAIL',
        native_executed_by_readback=False,Torch_executed_by_readback=False,compiler_executed_by_readback=False)
    try:
        report=json.loads(inputs.bind(args.result_directory.resolve()/'report.json'))
        result['counts']=qualify_metadata10(report,args.repository,args.build_directory,args.expected_build_sha256,
            args.result_directory,inputs,args.execution_directory)
        result['status']='PASS_METADATA10_PHYSICAL_READBACK'
    except BaseException as error:
        result['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        errors=inputs.recheck();result.update(inputs=inputs.records,all_inputs_unchanged=not errors,input_recheck_errors=errors)
        if errors:result['status']='FAIL'
        args.output.parent.mkdir(parents=True,exist_ok=True);common.dump(args.output,result)
    print(json.dumps(dict(status=result['status'],cases=result.get('counts',{}).get('cases'),native_executed=False)))
    return 0 if result['status']=='PASS_METADATA10_PHYSICAL_READBACK' else 1


if __name__=='__main__':raise SystemExit(main())
