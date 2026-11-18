#!/usr/bin/env python3
"""Run complete original Bool27 or Integral24 readers on fixed 701.

Original expected refusals and original FAIL reports remain untouched. The
separate outer report admits only the explicitly enumerated contract changes.
It does not qualify new F32 numerical semantics or the entire PyTorch project.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parent))
from profile import (SOURCE,IMAGE,BOOL_PINS,INTEGRAL_SHA,SUPPORT_SHA,OWNER_SHA,
                     dump,require,sha,read_pin,make_profile,execute_worker)

BOOL_REFERENCE='21dd75aeb86414818f7aa381f8b866d78b16f9bb3f5738e9a60d27ac3d15dd28'
INTEGRAL_REFERENCE='6c9df4b579664c89c200b9d796367dd6d41f93f8260e8ff012f17d5041d2278b'


def parse_response(raw):
    require(len(raw)>=32,'response header complete')
    magic,version,count,status=struct.unpack_from('<8s3Q',raw)
    require(magic==b'HTORSP03' and version==3,'original v3 response')
    if status:
        require(count==0 and len(raw)==32,'refusal has no records')
        return status,[]
    at=32;records=[]
    for _ in range(count):
        require(at+32<=len(raw),'response record header complete')
        opcode,error,value,size=struct.unpack_from('<4Q',raw,at);at+=32
        require(error==0 and at+size<=len(raw),'response record payload complete')
        records.append(dict(opcode=opcode,value=value,payload=raw[at:at+size]));at+=size
    require(at==len(raw),'no response tail')
    return 0,records


def frames(path,magic):
    raw=path.read_bytes();at=0;result=[]
    while at<len(raw):
        require(at+16<=len(raw),'complete raw frame envelope')
        actual,size=struct.unpack_from('<8sQ',raw,at);at+=16
        require(actual==magic and 32<=size<=1<<20 and at+size<=len(raw),'complete actual raw frame')
        result.append(raw[at:at+size]);at+=size
    require(at==len(raw),'no raw stream tail')
    return result


def native_transports(directory,expected):
    paths=sorted(directory.rglob('transport.json'))
    require(len(paths)==expected,'complete original native-child denominator')
    result=[]
    for path in paths:
        raw=path.read_bytes();t=json.loads(raw)
        require(t['executable_sha256']==IMAGE and t['exit_code']==0 and t['stdout_eof'] is True
                and t['stderr_eof'] is True and t['responses_before_stdin_eof'] is True
                and t['forced_cleanup'] is False and t['cleanup_errors']==[],
                'actual same701 image/clean exit/double EOF')
        require(t['stderr_bytes']==t['extra_stdout_bytes']==0,'no unexplained native output')
        require((path.parent/'stderr.bin').read_bytes()==b'' and
                (path.parent/'extra-stdout-after-frames.bin').read_bytes()==b'',
                'actual empty trailing streams')
        requests=frames(path.parent/'requests.frames.bin',b'HTOWNR01')
        responses=frames(path.parent/'responses.frames.bin',b'HTOWNS01')
        require(len(requests)==len(responses)==len(t['rows'])==t['frames']==t['complete_owner_boundaries'],
                'all request/response/owner boundaries present')
        with (path.parent.parent/'owner-frames.jsonl').open() as stream:
            journal=[json.loads(line) for line in stream]
        require(len(journal)==t['frames'],'all original owner frame journal rows')
        for i,(request,response,row,boundary) in enumerate(zip(requests,responses,t['rows'],journal),1):
            require(row==dict(frame=i,pid=t['pid'],response_bytes=len(response),stdin_open=True),
                    'original transport row binds actual frame')
            status,records=parse_response(response)
            require(boundary['frame']==i and 'observer_error' not in boundary and
                    boundary['request_sha256']==sha(request) and boundary['response_sha256']==sha(response)
                    and boundary['request_bytes']==len(request) and boundary['response_bytes']==len(response)
                    and boundary['frame_status']==status and boundary['record_count']==len(records),
                    'original complete owner boundary binds exact raw pair')
        result.append(dict(path=str(path.relative_to(directory)),sha256=sha(raw),pid=t['pid'],
                           frames=t['frames'],exit_code=0,stdout_eof=True,stderr_eof=True))
    require(len({r['pid'] for r in result})==expected,'distinct actual native children')
    return dict(children=expected,frames=sum(r['frames'] for r in result),rows=result,
                all_exit_zero=True,all_stdout_eof=True,all_stderr_eof=True,complete_owner_boundaries=True)


def raw_observation(directory,request_hex,response_hex,expected_status,expected_count):
    request,response=bytes.fromhex(request_hex),bytes.fromhex(response_hex)
    requests=frames(directory/'raw/requests.frames.bin',b'HTOWNR01')
    responses=frames(directory/'raw/responses.frames.bin',b'HTOWNS01')
    matches=[i for i,pair in enumerate(zip(requests,responses)) if pair==(request,response)]
    require(matches,'historical control observation is in actual native raw stream')
    status,records=parse_response(response)
    require((status,len(records))==(expected_status,expected_count),'control actual wire result')
    require(request[:8]==b'HTORCH03','control actual request header')
    count,declared,reserved=struct.unpack_from('<3Q',request,8)
    require(count==1 and declared==len(request) and reserved==0,'one exact original control request')
    command=list(struct.unpack_from('<8Q',request,32))
    if not status:
        require(records[0]['opcode']==command[0] and records[0]['payload']==b'',
                'success has original opcode and no invented payload')
    return dict(actual_status=status,record_count=len(records),command=command,
                request_sha256=sha(request),response_sha256=sha(response),actual_frame_indices=[i+1 for i in matches])


def passing_case(row):
    require(row['status']=='PASS' and row['exit_code']==0 and row['error'] is None,'original case actually passed')
    require(not row.get('mismatches') and not row.get('blockers') and not row.get('unattempted_reference_events')
            and not row.get('terminal_blocker'),'no hidden original mismatch/blocker')
    require(all(c.get('passed',c.get('check')) is True for c in row.get('checks',[])),
            'all recorded original comparisons pass')


def common_report(report,config):
    require(report['source_sha256']==SOURCE and report['native_sha256']==IMAGE and report['native_executed'] is True
            and report['Torch_executed'] is False,'same701 actual native report')
    require(report['schema_closure'] is False and report['full_project_status']=='HOLD','bounded qualification')
    key=str(Path(config['build_directory'])/'build.json')
    require(report['inputs'][key]['sha256']==config['expected_build_sha256'],'same actual cold build input')
    require(report['status']=='FAIL','original FAIL status must remain FAIL')


def qualify_bool(report,config):
    common_report(report,config)
    require(report['schema']=='heptatorch.sum-bool.native27.v1' and report['error'] is None
            and report['inputs_unchanged'] is True and report['build_functions']==712 and report['cold_units']==15,
            'complete original Bool execution and inputs')
    require(report['reference_sha256']==BOOL_REFERENCE and report['passed']==report['expected_cases']==len(report['cases'])==27
            and report['failed']==0,'all27 original Bool cases')
    for row in report['cases']:passing_case(row)
    names=['Bool_explicit_Bool','I64_explicit_Bool','Bool_explicit_F32','I64_explicit_F32',
           'F32_explicit_I64','F32_explicit_Bool','wrapped_F64_explicit_Bool']
    require([r['name'] for r in report['dtype_controls']]==names,'all7 original dtype controls')
    changed=[]
    for index,row in enumerate(report['dtype_controls']):
        if index not in (2,3):
            passing_case(row);continue
        require(row['status']=='FAIL' and row['exit_code']==0 and row['error']['type']=='AssertionError'
                and 'EOF_failure' not in row['error'],'original explicit F32 refusal failure')
        failed=[c for c in row['checks'] if c['passed'] is not True]
        require(failed==[dict(label='documented-dtype-boundary',actual=0,expected=7,passed=False)],
                'only unchanged expected cause7 assertion may fail')
        require(row['error']['message']==str(dict(rule='native-comparison',label='documented-dtype-boundary',actual=0,expected=7)),
                'specific original Bool failure')
        require(len(row['operations'])==1 and row['operations'][0]['label']=='dtype-control'
                and row['operations'][0]['normalized_status']==row['operations'][0]['frame_status']==0,
                'original dtype operation actually completed successfully')
        op=row['operations'][0]
        wire=raw_observation(Path(config['original_output'])/f'dtype-{index:02}',op['request_hex'],op['response_hex'],0,1)
        require(wire['command'][0]==115 and wire['command'][5]==0,'only explicitly added F32 target')
        changed.append(dict(index=index,name=row['name'],original_status='FAIL',original_expected_status=7,**wire))
    require(len(report['entry_controls'])==len(report['allocation_controls'])==1,'original packet/OOM controls present')
    for row in report['entry_controls']+report['allocation_controls']:passing_case(row)
    entry=report['entry_controls'][0]['operations']
    require([r['label'] for r in entry]==['invalid_public_capability','invalid_keepdim','unsupported_dtype_code',
         'reserved_word_six','reserved_word_seven','truncated_axis_payload','None_nonzero_payload',
         'empty_axes_nonzero_payload','axis_payload_before_payload_region','stale-public-capability']
         and [r['normalized_status'] for r in entry]==[1,7,7,7,7,2,7,7,2,1],'all10 original entry assertions')
    allocation=report['allocation_controls'][0]['operations']
    require([r['label'] for r in allocation]==['Bool-result-Storage-allocation-failure','same-Bool-operation-after-resource-recovery']
            and [r['normalized_status'] for r in allocation]==[3,0],'original Bool actual OOM/recovery')
    return dict(original_cases_passed=27,original_dtype_control_count=7,original_dtype_passed=5,original_dtype_failed=2,
                original_entry_controls_passed=10,original_Bool_allocation_control_passed=1,
                explicitly_changed_control_indices=[2,3],changed_controls=changed,original_report_status='FAIL')


def qualify_integral(report,config):
    common_report(report,config)
    require(report['schema']=='heptatorch.current-runtime.integral24.v1' and report['functions']==712
            and report['native_units']==15 and report['actual_build_sha256']==config['expected_build_sha256']
            and report['all_inputs_unchanged'] is True and report['original_input_recheck_errors']==[],
            'complete Integral original execution/custody')
    require(report['reference_sha256']==INTEGRAL_REFERENCE and report['passed']==report['expected_cases']==len(report['cases'])==24
            and report['failed']==report['blocked']==0 and report['completed_reference_events']==report['original_event_count']==260,
            'all24 original cases and260 events')
    for index,row in enumerate(report['cases']):
        passing_case(row)
        require(row['index']==index and row['completed_reference_events']==list(range(row['reference_event_count'])),
                'all original ordered Integral events')
    require(sum(len(r['completed_reference_events']) for r in report['cases'])==260,'actual complete260 event index')
    require(report['error']['type']=='AssertionError' and report['error']['message']==str(
        dict(rule='original-five-recorded-with-new-contract')),'only unchanged700 outer classifier rejects changed four controls')
    require('final_recheck_error' not in report,'no hidden custody failure')
    old=report['historical_controls']
    require(old['status']=='FAIL' and old['case_count']==len(old['cases'])==5 and old['passed']==1 and old['failed']==4,
            'original five remains FAIL with four unchanged refusals now succeeding')
    require([r['index'] for r in old['cases']]==list(range(5)),'all original control indices')
    changed=[]
    for index,row in enumerate(old['cases']):
        require(row['exit_code']==0,'original historical child exit')
        obs=row['observations'];expected=7 if index==4 else 0;count=0 if index==4 else 1
        wire=raw_observation(Path(config['original_output'])/'native/historical-controls'/f'control-{index:02}',
                             obs['request_hex'],obs['response_hex'],expected,count)
        require(obs['frame_status']==expected and obs['record_count']==count and obs['command']==wire['command'],
                'historical control exact raw observation')
        if index==4:
            passing_case(row);continue
        require(row['status']=='FAIL' and row['checks']==[dict(label='native-cause7',actual=0,expected=7,check=False)]
                and row['error']['type']=='AssertionError' and 'EOF_failure' not in row['error'],
                'only original refusal assertion differs')
        require(row['error']['message']==str(dict(reference_rule='bounded-api-control',control=row['name'],label='native-cause7')),
                'specific original Integral control failure')
        require(wire['command'][0]==115 and wire['command'][5]==(1 if index in (0,2) else 0),
                'only added Bool or F32 target succeeds')
        changed.append(dict(index=index,name=row['name'],original_status='FAIL',original_expected_status=7,**wire))
    return dict(original_cases_passed=24,original_events_completed=260,original_dtype_control_count=5,
                original_dtype_passed=1,original_dtype_failed=4,retained_refusal_indices=[4],
                explicitly_changed_control_indices=[0,1,2,3],changed_controls=changed,original_report_status='FAIL')


def run_worker(argv,directory):
    """Preserve both streams and reap the entire original-reader process group."""
    started=time.monotonic()
    runner=dict(argv=argv,started=False,stdout_EOF=False,stderr_EOF=False,timeout_seconds=1200,
                forced_cleanup=False,cleanup_errors=[])
    child=None;stdout=stderr=b''
    try:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,start_new_session=True)
        runner.update(started=True,pid=child.pid)
        stdout,stderr=child.communicate(timeout=runner['timeout_seconds'])
        runner.update(stdout_EOF=True,stderr_EOF=True)
    except BaseException as failure:
        runner['error']=dict(type=type(failure).__name__,message=str(failure))
        runner['timed_out']=isinstance(failure,subprocess.TimeoutExpired)
        if child is not None:
            runner['forced_cleanup']=True
            try:
                os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            except BaseException as cleanup:
                runner['cleanup_errors'].append(dict(operation='killpg',type=type(cleanup).__name__,message=str(cleanup)))
            try:
                stdout,stderr=child.communicate(timeout=30)
                runner.update(stdout_EOF=True,stderr_EOF=True)
            except BaseException as cleanup:
                if isinstance(cleanup,subprocess.TimeoutExpired):
                    stdout=cleanup.output or b'';stderr=cleanup.stderr or b''
                runner['cleanup_errors'].append(dict(operation='drain-and-wait',type=type(cleanup).__name__,message=str(cleanup)))
    runner.update(exit_code=child.returncode if child is not None else None,
        elapsed_seconds=time.monotonic()-started,stdout_bytes=len(stdout),stderr_bytes=len(stderr),
        stdout_sha256=sha(stdout),stderr_sha256=sha(stderr))
    (directory/'original-runner.stdout').write_bytes(stdout)
    (directory/'original-runner.stderr').write_bytes(stderr)
    dump(directory/'original-runner.json',runner)
    return runner,stderr


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=['bool27','integral24'],required=True)
    for name in ['repository','build-directory','bool-reference','references-directory','owner-directory','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-build-sha256',required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();require(not sys.flags.optimize,'assertions enabled')
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='heptatorch.712.original-readers-and-contract-delta.v1',status='PREPARING',
        suite=args.suite,source_sha256=SOURCE,native_sha256=IMAGE,actual_build_sha256=args.expected_build_sha256,
        native_executed=False,native_execution_requested=not args.prepare_only,Torch_executed=False,compiler_executed=False,
        original_reports_modified=False,original_comparisons_modified=False,original_expected_refusals_modified=False,
        F32_numerical_semantics_qualified_here=False,current_F32_gate='separate Sum F32 17+4 native reader',
        schema_closure=False,full_project_status='HOLD',error=None)
    code=1
    try:
        proof=make_profile(args.repository,args.output/'profile')
        config={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}
        config.update(profile_directory=str(args.output/'profile'),original_output=str(args.output/'original'),
            source_proof_sha256=sha((args.output/'profile/source-proof.json').read_bytes()),
            adapter_sources={name:sha(Path(__file__).with_name(name).read_bytes()) for name in ['run.py','profile.py']})
        dump(args.output/'worker-config.json',config)
        config_sha=sha((args.output/'worker-config.json').read_bytes())
        argv=[sys.executable,'-I','-S','-B',str(Path(__file__).resolve()),'--internal-worker',
              str(args.output/'worker-config.json'),config_sha]
        runner,stderr=run_worker(argv,args.output);report['original_runner']=runner
        if runner['started'] and not args.prepare_only:report['native_executed']=None
        require(runner['started'] and runner['stdout_EOF'] and runner['stderr_EOF']
                and not runner['forced_cleanup'] and not runner['cleanup_errors'] and 'error' not in runner
                and not stderr,'original runner complete, no cleanup/timeout/stderr')
        path=args.output/'original/report.json';raw=path.read_bytes();original=json.loads(raw)
        report['original_report']=dict(path='original/report.json',bytes=len(raw),sha256=sha(raw),status=original['status'])
        report['identity_adaptation']=proof
        if args.prepare_only:
            require(runner['exit_code']==0 and original['status']=='PREPARED_NOT_RUN' and original['native_executed'] is False
                    and original['cases']==[],'pure admission only, no reused PASS')
            report.update(status='PREPARED_NOT_RUN',original_control_status='NOT_RUN');code=0
        else:
            report['native_executed']=original.get('native_executed') is True
            require(runner['exit_code']==1,'original unchanged expected refusal must keep original FAIL exit1')
            facts=(qualify_bool if args.suite=='bool27' else qualify_integral)(original,config)
            report.update(facts)
            report['transport']=native_transports(args.output/'original',36 if args.suite=='bool27' else 29)
            require(path.read_bytes()==raw,'original report remains byte-exact after outer admission')
            report['status']='PASS_712_COMPARISONS_WITH_ORIGINAL_FAILURES_RETAINED';code=0
    except BaseException as error:
        report.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    dump(args.output/'report.json',report)
    print(json.dumps({k:report[k] for k in ['status','suite','native_executed','error']}))
    return code


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--internal-worker':
        raw=read_pin(Path(sys.argv[2]),sys.argv[3]);config=json.loads(raw)
        for name,pin in config['adapter_sources'].items():read_pin(Path(__file__).with_name(name),pin)
        raise SystemExit(execute_worker(config))
    raise SystemExit(main())
