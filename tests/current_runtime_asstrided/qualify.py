"""Independently repeat every original AsStrided16 source, owner and transport check."""
import argparse
import importlib.util
import sys
import traceback
import json
import lzma
from pathlib import Path
import re
import stat
import struct

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('heptatorch_asstrided16_qualification_common',HERE/'common.py')
common=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(common)
SOURCE,IMAGE,SUITES,COUNTS,SOURCES_SHA=common.SOURCE,common.IMAGE,common.SUITES,common.COUNTS,common.SOURCES_SHA
admit,frames,require,sha=common.admit,common.frames,common.require,common.sha
SOURCE_FILES=('common.py','worker.py','qualify.py','sources.json')


def check_source_inverses(repo, folder, suite, worker, inputs):
    proof=json.loads(inputs.bind(folder/'adaptation/source-inverses.json'))
    require(proof['whole_file_inverse_exact'] is True
        and proof['only_build_admission_and_owner_tag_literals_changed'] is True
        and proof['numeric_expectations_modified'] is False and proof['recipes_modified'] is False
        and proof['edits']==worker['source_inverses'] and len(proof['edits'])==2,
        'complete original reader and observer source inverses required')
    directory=repo/'tests/tensor_to_original16_review'
    reader_name='replay_as_strided_'+suite+'.py'
    for row,original_name,executed_name in zip(proof['edits'],
            ['native_observer09.py',reader_name],['native_observer_current.py',reader_name]):
        before=inputs.bind(directory/original_name,row['original_sha256'])
        after=inputs.bind(folder/'adaptation'/executed_name,row['derived_sha256'])
        require(row['original']==str(directory/original_name)
            and row['derived']==str(folder/'adaptation'/executed_name)
            and row['whole_file_inverse_exact'] is True,'original/executed source paths differ')
        if original_name=='native_observer09.py':
            expected=[(b'HTTOCP08',b'HTSINT02'),(b'HTOTOCP8',b'HTOSIN02')]
        else:
            start=before.index(b"    assert build['status'] == 'PASS'")
            end=before.index(b'    spec = importlib.util.spec_from_file_location',start)
            block=before[start:end]
            require(sha(block)=='ace14ab9d8f9b629b69a26742adec59035a72bec1e08e62c55169fea15612539',
                    'the complete original build-only block differs')
            expected=[(block,b'    asstrided16_admit(args, inputs)\n')]
        require([(bytes.fromhex(e['before_hex']),bytes.fromhex(e['after_hex'])) for e in row['edits']]
                ==expected and all(e['count']==1 for e in row['edits']),
                'an original comparison or recipe was changed')
        inverse=after
        for old,new in reversed(expected):
            require(inverse.count(new)==1,'ambiguous full source inverse')
            inverse=inverse.replace(new,old,1)
        require(inverse==before,'original source inverse differs')
    for path,pin in worker['inputs'].items():
        raw=inputs.bind(Path(path),pin['sha256'])
        require(len(raw)==pin['bytes'] and stat.S_IMODE(Path(path).stat().st_mode)==pin['mode'],
                'actual worker input bytes/mode changed')


def inspect_frame_pairs(requests, responses):
    require(len(requests)==len(responses),'request/response frame denominator')
    for (size,request),(response_size,response) in zip(requests,responses):
        require(32<=size<=1<<20 and request[:8]==b'HTORCH03', 'complete original tensor request')
        count,extent,reserved=struct.unpack_from('<3Q',request,8)
        require(extent==len(request) and reserved==0 and 1<=count<=16383
                and 32+64*count<=len(request),'complete original request command table')
        require(32<=response_size<=1<<20 and response[:8]==b'HTORSP03','complete original tensor response')
        version,returned,status=struct.unpack_from('<3Q',response,8)
        require(version==3,'original response version')
        if status:
            require(returned==0 and len(response)==32,'complete original refusal frame')
            continue
        require(returned==count,'all original commands require corresponding response records')
        at=32
        for index in range(returned):
            require(at+32<=len(response),'truncated original response record')
            opcode,error,value,length=struct.unpack_from('<4Q',response,at);at+=32
            require(opcode==struct.unpack_from('<Q',request,32+64*index)[0]
                    and error==0 and at+length<=len(response),'original record identity or extent')
            at+=length
        require(at==len(response),'extra original response bytes')


def inspect_owner_archives(case_directory, row, inputs):
    references={};observations=0
    def collect(value):
        nonlocal observations
        if isinstance(value,dict):
            if {'file','bytes','sha256'}<=value.keys():
                name=value['file']
                require(re.fullmatch('[0-9a-f]{64}\\.owner\\.xz',name) is not None,
                        'exact original owner filename')
                require(name not in references or references[name]==value,'conflicting original owner pin')
                references[name]=value;observations+=1
            for item in value.values():collect(item)
        elif isinstance(value,list):
            for item in value:collect(item)
    collect(row)
    require(references and {p.name for p in case_directory.glob('*.owner.xz')}==set(references),
            'complete original owner-archive census')
    total=0
    for name,pin in references.items():
        compressed=inputs.bind(case_directory/name)
        decoder=lzma.LZMADecompressor();raw=decoder.decompress(compressed)
        require(decoder.eof and not decoder.unused_data and len(raw)==pin['bytes']
                and sha(raw)==pin['sha256']==name[:-9],'original complete owner bytes differ')
        total+=len(raw)
    return dict(observations=observations,unique_archives=len(references),unique_raw_bytes=total)



def read_suite(repo,folder,suite,build_sha,inputs,reference,pins,expected_source_proof,receipt,prepare_only=False):
    worker_raw=inputs.bind(folder/'worker.json')
    worker=json.loads(worker_raw)
    row=dict(suite=suite,status='RUNNING',native_executed=worker.get('native_executed'),
             worker_sha256=sha(worker_raw),process=receipt)
    require(receipt['started'] and receipt['exit_code']==0 and receipt['stdout_EOF']
        and receipt['stderr_EOF'] and receipt['stderr_bytes']==0
        and all(not receipt.get(key) for key in ('timed_out','forced_cleanup',
            'execution_error','error','cleanup_errors','running_descendants_after_cleanup')),
        'complete original reader process and both EOF required')
    require(worker is not None and worker['all_inputs_unchanged'] is True
        and worker['source_sha256']==SOURCE and worker['native_sha256']==IMAGE
        and worker['actual_build_sha256']==build_sha,'worker identity or custody mismatch')
    check_source_inverses(repo,folder,suite,worker,inputs)
    source_proof=json.loads(inputs.bind(folder/'adaptation/native-source-proof.json'))
    require(source_proof==worker['source_proof']==expected_source_proof,
            'the actual current source/owner admission proof changed')
    observed=json.loads(inputs.bind(folder/'original-finish-observations.json'))
    require(observed['errors']==[] and observed['original_finish_replaced'] is False
        and observed['syscalls_replaced'] is False
        and observed['original_client_sha256']==pins['original_finish']['file_sha256']
        and observed['original_finish_sha256']==pins['original_finish']['sha256'],
        'actual original finish observation incomplete')
    if prepare_only:
        require(worker['status']=='PREPARED_NOT_RUN' and worker['native_executed'] is False
                and worker['cases']==0,'preparation must not execute native cases')
        require(observed['events']==[],'preparation must not create a native session')
        row['status']='PREPARED_NOT_RUN'
        return row
    require(worker['status']=='PASS_ORIGINAL_REPLAY' and worker['native_executed'] is True,
            'actual original reader did not pass')
    original_raw=inputs.bind(folder/'actual/report.json')
    original=json.loads(original_raw)
    require(original['status']=='PASS' and original['cases']==original['passed']==COUNTS[suite]
        and original['failed']==0 and original['all_inputs_unchanged'] is True
        and original['source_sha256']==SOURCE and original['native_sha256']==IMAGE,
        'complete unchanged original numerical report required')
    require(original['weak_lifetime_claimed'] is False
        and original['full_pytorch_rewrite_complete'] is False
        and original['reference_sha256']==pins['references'][suite]['reference_sha256']
        and original['probe_sha256']==pins['references'][suite]['probe_sha256']
        and original['reader_sha256']==pins['repository_files'][
            'tests/tensor_to_original16_review/replay_as_strided_'+suite+'.py']['sha256']
        and original['provenance_checks']==reference['oracle_provenance']['checks'],
        'complete original reference/reader/provenance identity required')
    require([(x['index'],x['name']) for x in original['rows']]==
            [(x['index'],x['name']) for x in reference['rows']],
            'original reference row identities changed')
    require(len(observed['events'])==2*COUNTS[suite],
            'all actual constructors and original finish calls must be observed')
    native=[]
    for ordinal,(actual,want) in enumerate(zip(original['rows'],reference['rows'])):
        require(actual['status']=='PASS' and actual['differences']==[]
            and actual['host_errors']==[] and 'transport_error' not in actual
            and 'transport' in actual and actual['transport'] is None,
            'original case failed or native finish did not complete')
        require([x['label'] for x in actual['events']]==[x['label'] for x in want['events']]
            and [x['label'] for x in actual['snapshots']]==[x['label'] for x in want['snapshots']],
            'original event or snapshot denominator changed')
        case=folder/'actual'/('case-'+str(actual['index']).zfill(2))
        ledger=json.loads(inputs.bind(case/'raw/transport.json'))
        require(ledger['executable_sha256']==IMAGE and ledger['exit_code']==0
            and ledger['responses_before_stdin_eof'] is True
            and ledger['frames']==actual['frames']==len(ledger['rows']),
            'actual original native session ledger required')
        created,finished=observed['events'][2*ordinal:2*ordinal+2]
        require([created['method'],finished['method']]==['__init__','finish']
            and all(event['normal_return'] is True and event['callback_completed'] is True
                and event['original_method_code_identity'] is True
                and event['observer_pid']==receipt['pid']
                and event['pid']==ledger['pid'] and event['pid']!=receipt['pid']
                and event['directory']==str(case/'raw')
                and event['executable_sha256']==IMAGE for event in (created,finished)),
            'exact original native child and normal method returns required')
        require(finished['native_exit_code']==ledger['exit_code']==0
            and finished['stdout_EOF'] is True and finished['stderr_EOF'] is True
            and finished['stdout_bytes']==finished['stderr_bytes']==0
            and finished['stdout_sha256']==finished['stderr_sha256']==sha(b'')
            and all(finished[key] is True for key in ('input_closed','output_closed','stderr_closed',
                'request_archive_closed','response_archive_closed','returned_None')),
            'both actual empty reads and successful original finish are required')
        request=inputs.bind(case/'raw/requests.frames.bin')
        response=inputs.bind(case/'raw/responses.frames.bin')
        req,res=frames(request,b'HTOWNR01'),frames(response,b'HTOWNS01')
        require(len(req)==len(res)==ledger['frames'],'complete actual request/response frame counts')
        inspect_frame_pairs(req,res)
        for index,((size,_),event) in enumerate(zip(res,ledger['rows']),1):
            require(event==dict(frame=index,pid=ledger['pid'],response_bytes=size,stdin_open=True),
                    'actual native response row changed')
        native.append(dict(index=actual['index'],name=actual['name'],pid=ledger['pid'],
            frames=ledger['frames'],exit_code=0,requests_sha256=sha(request),
            responses_sha256=sha(response),stdout_EOF=True,stderr_EOF=True,
            original_finish_observation=finished,
            owner_archives=inspect_owner_archives(case,actual,inputs)))
        denominator=pins['original_denominators'][suite]['rows'][ordinal]
        require(all(actual[key]==value for key,value in denominator.items()),
                'a complete original numerical/lifecycle denominator changed')
    for key in ('frames','checks','exact_words','rollback_bytes','rng_bytes'):
        require(original[key]==sum(case[key] for case in original['rows'])
                ==pins['original_denominators'][suite][key],
                'original aggregate denominator differs: '+key)
    row.update(status='PASS',cases=original['cases'],frames=original['frames'],
        checks=original['checks'],exact_words=original['exact_words'],
        rng_bytes=original['rng_bytes'],rollback_bytes=original['rollback_bytes'],
        original_report_sha256=sha(original_raw),native_sessions=native)
    return row


def read_execution(directory,suite,report,inputs,prepare_only=False):
    execution=directory/(suite+'-execution')
    receipt=json.loads(inputs.bind(execution/'runner.json'))
    expected=[report['python_executable'],'-I','-S','-B',str(Path(report['source_directory'])/'worker.py'),
        '--repository',report['repository'],'--build-directory',report['build_directory'],
        '--expected-build-sha256',report['actual_build_sha256'],
        '--reference-directory',report['reference_directory'],'--suite',suite,
        '--output',str(directory/suite)]
    if prepare_only:expected.append('--prepare-only')
    require(receipt['argv']==expected and receipt['stderr_mirrored'] is True,
            'exact original worker invocation must match its actual process receipt')
    for stream in ('stdout','stderr'):
        raw=inputs.bind(execution/('runner.'+stream),receipt[stream+'_sha256'])
        require(len(raw)==receipt[stream+'_bytes'],'complete supervised worker stream bytes')
    return receipt


def qualify_asstrided16(report,build_sha,directory,inputs,*,repository=None):
    """Read a complete actual execution; start no worker and change no evidence."""
    directory=Path(directory).resolve()
    repository=Path(repository).resolve() if repository is not None else HERE.parents[1]
    require(report['schema']=='heptatorch.current761.asstrided16.v1'
        and report['status']=='PASS_CURRENT_ASSTRIDED16'
        and report['source_sha256']==SOURCE and report['native_sha256']==IMAGE
        and report['actual_build_sha256']==build_sha and report['functions']==761 and report['units']==15,
        'complete same-current-image AsStrided16 qualification required')
    require(report['repository']==str(repository) and report['output_directory']==str(directory)
        and report['prepare_only'] is False and report['native_executed'] is True
        and report['Torch_executed'] is False and report['compiler_executed'] is False
        and report['old_pass_transferred'] is False and report['numeric_expectations_modified'] is False
        and report['full_pytorch_rewrite_complete'] is False and report['schema_closure'] is False
        and report['full_project_status']=='HOLD' and report['failures']==[]
        and report['all_inputs_unchanged'] is True,'actual finite execution and unchanged original inputs required')
    require(report['required_suites']==list(SUITES) and report['required_cases']==COUNTS
        and report['required_cases_total']==16
        and [row['suite'] for row in report['suites']]==list(SUITES)
        and all(row['status']=='PASS' for row in report['suites']),
        'all four complete original suites must be present in their original order')
    actual_source=Path(report['source_directory'])
    for name in SOURCE_FILES:
        current=inputs.bind(HERE/name)
        executed=inputs.bind(actual_source/name)
        require(executed==current,'actual executed AsStrided16 implementation differs: '+name)
    pins=json.loads(inputs.bind(HERE/'sources.json',SOURCES_SHA))
    build,reference_root=Path(report['build_directory']),Path(report['reference_directory'])
    repeated=[]
    for suite,recorded in zip(SUITES,report['suites']):
        _,_,reference,_,source_proof=admit(repository,build,build_sha,reference_root,suite,inputs)
        receipt=read_execution(directory,suite,report,inputs)
        row=read_suite(repository,directory/suite,suite,build_sha,inputs,reference,pins,source_proof,receipt)
        require(row==recorded,'complete original suite readback differs: '+suite)
        repeated.append(row)
    for key,expected in dict(cases=16,frames=1920,checks=10272,exact_words=21892,
                            rng_bytes=1860608,rollback_bytes=0).items():
        require(report[key]==sum(row[key] for row in repeated)==expected,
                'complete original AsStrided16 aggregate differs: '+key)
    require(report['inputs'],'complete actual input census is required')
    for path,pin in list(report['inputs'].items()):
        raw=inputs.bind(Path(path),pin['sha256'])
        require(len(raw)==pin['bytes'] and stat.S_IMODE(Path(path).stat().st_mode)==pin['mode'],
                'complete original AsStrided16 input bytes/mode differ')
    children=[child for row in repeated for child in row['native_sessions']]
    require(len(children)==16,'sixteen actual native children required')
    counts=dict(cases=16,frames=1920,assertions=10272,exact_words=21892,
        serialized_state_bytes=1860608,rollback_bytes=0,native_sessions=16,worker_sessions=4,
        native_EOF_count=32,worker_EOF_count=8,oracle_provenance_checks_per_group=11,
        original_report_status='PASS',owner_observations=sum(c['owner_archives']['observations'] for c in children),
        owner_unique_archives=sum(c['owner_archives']['unique_archives'] for c in children),
        owner_unique_raw_bytes=sum(c['owner_archives']['unique_raw_bytes'] for c in children))
    if 'readback' in report:require(report['readback']==counts,'stored independent AsStrided16 readback differs')
    return counts


def observed_execution(rows):
    facts=[row.get('native_executed') for row in rows]
    if any(value is True for value in facts):return True
    if any(value is not False for value in facts):return None
    return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,default=HERE.parents[1])
    parser.add_argument('--build-directory',type=Path,required=True)
    parser.add_argument('--reference-directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-build-sha256')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    require(not sys.flags.optimize,'all original assertions must stay enabled')
    repo,build,out=args.repository.resolve(),args.build_directory.resolve(),args.output.resolve()
    refs=(args.reference_directory or repo/'validation/tensor-to-original16-recovery').resolve()
    require(out!=build and not out.is_relative_to(build),'output must be outside actual build')
    out.mkdir(parents=True,exist_ok=False)
    build_sha=args.expected_build_sha256 or sha((build/'build.json').read_bytes())
    inputs,rows,failures=common.Inputs(),[],[]
    report=dict(schema='heptatorch.current761.asstrided16.v1',status='RUNNING',
        source_sha256=SOURCE,native_sha256=IMAGE,actual_build_sha256=build_sha,
        functions=761,units=15,required_suites=list(SUITES),required_cases=COUNTS,
        required_cases_total=16,prepare_only=args.prepare_only,suites=rows,failures=failures,
        repository=str(repo),build_directory=str(build),reference_directory=str(refs),
        output_directory=str(out),source_directory=str(HERE),python_executable=sys.executable,
        python_version=sys.version,cases=0,native_executed=False,Torch_executed=False,compiler_executed=False,
        all_inputs_unchanged=False,old_pass_transferred=False,numeric_expectations_modified=False,
        full_pytorch_rewrite_complete=False,schema_closure=False,full_project_status='HOLD')
    try:
        for name in SOURCE_FILES:inputs.bind(HERE/name)
        inputs.bind(Path(sys.executable).resolve())
        pins=json.loads(inputs.bind(HERE/'sources.json',SOURCES_SHA))
        references={};source_proofs={}
        for suite in SUITES:
            _,_,reference,_,source_proof=admit(repo,build,build_sha,refs,suite,inputs)
            references[suite]=reference;source_proofs[suite]=source_proof
        process=common.module(repo/'tests/current_runtime_process.py','asstrided16_current_process',inputs)
        for suite in SUITES:
            folder=out/suite;execution=out/(suite+'-execution');execution.mkdir()
            row=dict(suite=suite,status='RUNNING',native_executed=None)
            rows.append(row)
            command=[sys.executable,'-I','-S','-B',str(HERE/'worker.py'),
                '--repository',str(repo),'--build-directory',str(build),
                '--expected-build-sha256',build_sha,'--reference-directory',str(refs),
                '--suite',suite,'--output',str(folder)]
            if args.prepare_only:command.append('--prepare-only')
            try:
                try:
                    receipt=process.execute(command,execution,900,cwd=repo,mirror_stderr=True)
                finally:
                    if (folder/'worker.json').is_file():
                        worker_raw=inputs.bind(folder/'worker.json')
                        worker=json.loads(worker_raw);fact=worker.get('native_executed')
                        row['native_executed']=fact if type(fact) is bool else None
                        row['worker_sha256']=sha(worker_raw)
                row['process']=receipt
                require(receipt==read_execution(out,suite,report,inputs,args.prepare_only),
                        'actual worker receipt differs from preserved JSON')
                row.update(read_suite(repo,folder,suite,build_sha,inputs,references[suite],pins,
                                      source_proofs[suite],receipt,args.prepare_only))
            except BaseException as error:
                row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
                failures.append(dict(suite=suite,error=repr(error)))
            finally:
                report['native_executed']=observed_execution(rows)
                common.dump(out/'report.json',report)
        inputs.recheck();report['all_inputs_unchanged']=True
        if failures:report['status']='FAIL'
        elif args.prepare_only:report['status']='PREPARED_NOT_RUN'
        else:
            for key in ('cases','frames','checks','exact_words','rng_bytes','rollback_bytes'):
                report[key]=sum(row[key] for row in rows)
            report['status']='PASS_CURRENT_ASSTRIDED16'
            report['inputs']=inputs.rows
            report['readback']=qualify_asstrided16(report,build_sha,out,inputs,repository=repo)
    except BaseException as error:
        report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
    finally:
        report['native_executed']=observed_execution(rows)
        try:inputs.recheck();report['all_inputs_unchanged']=True
        except BaseException as error:
            report.update(status='FAIL',all_inputs_unchanged=False,input_error=repr(error))
        report['inputs']=inputs.rows
        common.dump(out/'report.json',report)
    print(json.dumps({key:report.get(key) for key in ('status','cases','frames','checks','native_executed','failures','error')}))
    return int(report['status']=='FAIL')


if __name__=='__main__':
    raise SystemExit(main())
