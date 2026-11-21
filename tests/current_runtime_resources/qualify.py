#!/usr/bin/env python3
"""Execute the original seven resource controls on one current cold-built image."""
import argparse
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tarfile
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = 'c73a7cb474981056404ac96bc214979d57aaa9a86e6474b1c140c3d98eb05779'
IMAGE = '1ea247df76ff96c307e293790aa38de4a0bf0ff94b8148a480653c924658679d'
FUSION_RESOURCE_SHA = '5fb33b970fb74792abdfb7fab7fe9b62196a9b6d08cfeb1f3ebb5330ebe19e97'
ARCHIVE = 'ce4aff6daaf5bb727fa6a520cbecf98dd8e09e67ffffa6cb60171a210b82e490'
ARCHIVE_MANIFEST = 'a3d49014e99b68a98a7cc41a787d4fa6102129b63aed720dc857b5d6e862ef4d'
ARCHIVE_LOCATION = '33df7752cb769610e6c30421e9154bca2168c696a0d67bf5422c8d699a05e444'
CASES = {
    'r4': ['entry_guards_and_relative_tokens','saved_scalar_and_temporary_lifetime',
           'result_allocation_oom_recovery'],
    'backward': ['backward_private_preserve_allocation_oom_recovery'],
    'r3': ['tail_release_reuse','preserve_capacity_oom_recovery','tail_then_saved_error_prefix'],
}
REFERENCES = {
    'f3ecc9dd7db2ce8fbd0e424557ad5b95ac2e899a1c09cf38dad47e4fecc2b0c6':('scalar32','actual.json',2867710),
    'c0ddafbae6ea81bd98f6066bc867f71e0b6e8d4f73e6c5d7f15f7fe3d29569a3':('scalar32','probe.py',17863),
    'f4abe6a750ecc791b3a7bc7aa507b4c77eb0a2fcccf45b0c6a828db0e01490d3':('input4','actual.json',5680715),
    'cf569335f48624c1edf2f97a649200f2e792ebcef5b6f1df6bac73293ce02efc':('input4','probe.py',23382),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def combined_native_execution(rows):
    """Execution facts are independent of whether a qualification passed."""
    states = [row.get('native_executed') for row in rows]
    if any(state is True for state in states):
        return True
    if any(state is not False for state in states):
        return None
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repository', required=True, type=Path)
    ap.add_argument('--build-directory', required=True, type=Path)
    ap.add_argument('--expected-build-sha256', help='Expected SHA of this actual cold-build receipt; otherwise computed before admission.')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--suite', choices=('all',*CASES), default='all')
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    require(not sys.flags.optimize, 'original assertions must remain enabled')
    repo, build, out = args.repository.resolve(), args.build_directory.resolve(), args.output.resolve()
    build_sha = args.expected_build_sha256 or sha((build/'build.json').read_bytes())
    require(out != build and not out.is_relative_to(build), 'output must be outside actual build')
    out.mkdir(parents=True, exist_ok=False)
    inputs, rows, failures = {}, [], []
    selected = list(CASES) if args.suite == 'all' else [args.suite]
    report = dict(schema='heptatorch.current797.resource7.v1',status='RUNNING',
        source_sha256=SOURCE,native_sha256=IMAGE,actual_build_sha256=build_sha,
        functions=797,units=15,required_cases=CASES,required_case_count=7,
        selected_suites=selected,prepare_only=args.prepare_only,suites=rows,failures=failures,
        native_executed=False,Torch_executed=False,compiler_executed=False,
        full_pytorch_rewrite_complete=False,schema_closure=False,old_pass_transferred=False)

    def bind(path, expected=None):
        path = path.resolve()
        raw = path.read_bytes()
        row = dict(bytes=len(raw),sha256=sha(raw),mode=path.stat().st_mode & 511)
        require(expected is None or row['sha256']==expected,'input SHA mismatch: '+str(path))
        require(str(path) not in inputs or inputs[str(path)]==row,'input changed: '+str(path))
        inputs[str(path)] = row
        return raw

    try:
        for path in sorted(HERE.glob('*.py')):
            bind(path)
        source_pins = json.loads(bind(HERE/'sources.json'))
        for relative, row in source_pins['files'].items():
            bind(repo/relative,row['sha256'])
        current = module(repo/'tools/current_runtime/qualification.py','resource7_qualification')
        current.admit_build(build,bind,build_sha)
        source_helper_path = repo/'tools/current_runtime/fusion_resource_transition.py'
        bind(source_helper_path,FUSION_RESOURCE_SHA)
        source_helper = module(source_helper_path,'resource778_independent_source_paths')
        expected_source_paths = source_helper.prove(bind(build/'runtime.hepta',SOURCE),bind)
        process = module(repo/'tests/current_runtime_process.py','resource7_process')
        archive_path = repo/'validation/current-runtime76-reference/complete.tar.xz'
        archive_manifest_path = repo/'validation/current-runtime76-reference/manifest.json'
        archive_location_path = repo/'validation/view-reshape-r4-native76/archive-location.json'
        archive_manifest = json.loads(bind(archive_manifest_path,ARCHIVE_MANIFEST))
        archive_location = json.loads(bind(archive_location_path,ARCHIVE_LOCATION))
        require(archive_manifest['schema']=='heptatorch.current-runtime.fixed-original76-references.v1'
            and archive_manifest['archive']['sha256']==ARCHIVE
            and archive_manifest['origin']['archive']['sha256']=='511dff328018a0a2d6a56a94fde4d2e0d64ff10ea66bb0cb71f89bb61ab2a3fb'
            and archive_location['original_git_file']['sha256']==archive_manifest['origin']['archive']['sha256']
            and archive_location['current_reference_subset']['sha256']==ARCHIVE,
            'compact reference and historical archive source identities')
        archive = bind(archive_path,ARCHIVE)
        restored, bindings = {}, []
        with tarfile.open(fileobj=io.BytesIO(archive),mode='r:xz') as packet:
            for member in packet:
                if not member.isfile() or member.size not in {v[2] for v in REFERENCES.values()}:
                    continue
                raw = packet.extractfile(member).read()
                digest = sha(raw)
                if digest not in REFERENCES:
                    continue
                suite,name,size = REFERENCES[digest]
                require((suite,name) not in restored and len(raw)==size,'duplicate original reference')
                path = out/'references'/suite/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(raw);path.chmod(member.mode)
                bind(path,digest)
                restored[suite,name] = path
                original = archive_manifest['files'][member.name]
                require((original['sha256'],original['bytes'],original['mode'])==
                        (digest,len(raw),member.mode),'compact member exact original identity')
                bindings.append(dict(archive_member=member.name,path=str(path),bytes=len(raw),
                                     sha256=digest,mode=member.mode,
                                     original_archive_member=original['original_archive_member']))
        require(len(restored)==4,'both original references and probes must be restored')
        dump(out/'reference-bindings.json',dict(archive=str(archive_path),archive_sha256=ARCHIVE,
             archive_manifest=dict(path=str(archive_manifest_path),sha256=ARCHIVE_MANIFEST),
             historical_archive_location=dict(path=str(archive_location_path),sha256=ARCHIVE_LOCATION),
             original_archive=archive_manifest['origin']['archive'],
             original_bytes_and_modes_preserved=True,matches=bindings))
        for suite in selected:
            directory = out/suite
            directory.mkdir()
            command = [sys.executable,'-I','-S','-B',str(HERE/'adapter.py'),
                '--repository',str(repo),'--build-directory',str(build),'--suite',suite,
                '--expected-current-build-sha256',build_sha,
                '--evidence',str(directory/'adaptation'),'--native-build',str(build),
                '--output',str(directory/'actual')]
            if args.prepare_only:
                command.append('--prepare-only')
            if suite in ('r4','backward'):
                parent = repo/'tests/wrapped_scalar_resource_review/portable07b'
                command += ['--binary',str(build/'heptatorch'),
                    '--protocol',str(parent/'protocol.py'),
                    '--reference',str(restored['scalar32','actual.json']),
                    '--probe',str(restored['scalar32','probe.py']),
                    '--expected-build-sha256',build_sha,'--expected-native-sha256',IMAGE]
                if suite=='backward':
                    command += ['--resource-source',str(parent),
                                '--obligations',str(repo/'tests/backward_private_cache_resource_review/obligations.json')]
            else:
                command += ['--profile',str(repo/'tools/differentiable_view_recovered_r3/profile.json'),
                    '--protocol',str(repo/'tests/differentiable_view_preserve_resource_review/protocol.py'),
                    '--reference',str(restored['input4','actual.json'])]
            row = dict(suite=suite,case_names=CASES[suite],status='RUNNING',native_executed=None)
            rows.append(row)
            try:
                actual = None
                try:
                    receipt = process.execute(command,directory,900,cwd=repo,mirror_stderr=True)
                finally:
                    # A nonzero worker can still have executed a real native
                    # child. Always observe its original report before judging
                    # the worker receipt; a missing report remains unknown.
                    actual_path = directory/'actual/report.json'
                    if actual_path.is_file():
                        actual_raw = bind(actual_path)
                        actual = json.loads(actual_raw)
                        observed = actual.get('native_executed')
                        row['native_executed'] = observed if type(observed) is bool else None
                        row['report_sha256'] = sha(actual_raw)
                row['process'] = receipt
                require(receipt['started'] and receipt['exit_code']==0 and receipt['stdout_EOF']
                    and receipt['stderr_EOF'] and not receipt['stderr_bytes']
                    and not receipt.get('forced_cleanup') and not receipt.get('timed_out')
                    and not receipt.get('execution_error') and not receipt.get('error')
                    and not receipt.get('cleanup_errors') and not receipt.get('running_descendants_after_cleanup'),
                    'complete successful worker EOF required')
                require(actual is not None,'original worker report is required')
                proof = json.loads(bind(directory/'adaptation/source-bindings.json'))
                require(proof['fusion_source_path_proof']==expected_source_paths,
                        'independent complete actual797 source path proof differs')
                require(actual['source_sha256']==SOURCE and actual['native_sha256']==IMAGE
                        and actual['all_inputs_unchanged'] is True,'actual current identity/custody')
                require(proof['all_original_inputs_unchanged'] is True and proof['whole_file_inverse_exact'] is True
                        and proof['numeric_expectations_modified'] is False
                        and proof['resource_thresholds_modified'] is False
                        and proof['owner_equations_modified'] is False,'exact adapter boundary')
                if args.prepare_only:
                    require(actual['status']=='PREPARED_NOT_RUN' and actual['native_executed'] is False
                            and actual['case_count']==0,'preparation must not execute native cases')
                    row['status']='PREPARED_NOT_RUN'
                else:
                    require(actual['status']=='PASS' and actual['native_executed'] is True
                        and [c['name'] for c in actual['cases']]==CASES[suite]
                        and actual['case_count']==actual['passed']==len(CASES[suite]),'complete original recipe denominator')
                    transports=[]
                    for case in actual['cases']:
                        require(case['status']=='PASS' and case['EOF_exit_code']==0,'original recipe and native exit')
                        if case['name'] in ('result_allocation_oom_recovery',
                                'backward_private_preserve_allocation_oom_recovery',
                                'preserve_capacity_oom_recovery'):
                            require(case['original_limit_restored'] is True,
                                    'original native address-space limit must be restored after OOM')
                        transport_dir=directory/'actual'/case['name']/'transport'
                        ledger=json.loads(bind(transport_dir/'transport.json'))
                        require(ledger['exit_code']==0 and ledger['executable_sha256']==IMAGE
                            and ledger['responses_before_stdin_eof'] is True
                            and ledger['frames']==case['frames_before_EOF']==len(ledger['rows']),
                            'real native transport ledger')
                        request=bind(transport_dir/'requests.frames.bin')
                        response=bind(transport_dir/'responses.frames.bin')
                        at=frames=0;opcodes=[]
                        while at<len(request):
                            require(at+16<=len(request),'truncated request header')
                            require(request[at:at+8]==b'HTOWNR01','native request frame magic')
                            size=struct.unpack_from('<Q',request,at+8)[0]
                            packet=request[at+16:at+16+size]
                            require(len(packet)==size,'truncated request body')
                            if packet[:8]==b'HTORCH03':
                                count=struct.unpack_from('<Q',packet,8)[0]
                                require(32+64*count<=len(packet),'truncated command table')
                                opcodes.extend(struct.unpack_from('<Q',packet,32+64*i)[0] for i in range(count))
                            at+=16+size;frames+=1
                        require(at==len(request) and frames==ledger['frames'],'actual frame denominator')
                        response_at=response_frames=0
                        while response_at<len(response):
                            require(response_at+16<=len(response),'truncated response header')
                            require(response[response_at:response_at+8]==b'HTOWNS01',
                                    'native response frame magic')
                            size=struct.unpack_from('<Q',response,response_at+8)[0]
                            require(32<=size<=1<<20 and response_at+16+size<=len(response),
                                    'complete bounded response body')
                            require(response_frames<len(ledger['rows']) and ledger['rows'][response_frames]==dict(
                                frame=response_frames+1,pid=ledger['pid'],response_bytes=size,stdin_open=True),
                                'actual native response frame must match original live-input ledger row')
                            response_at+=16+size;response_frames+=1
                        require(response_at==len(response) and response_frames==frames,
                                'complete response/request frame denominator')
                        require(not set(opcodes).intersection((115,116,117,118,119)),
                                'old recipes must use the proven old dispatcher delegation')
                        require(3 not in opcodes,
                                'original resource recipes must not enter the new wide unary route')
                        transports.append(dict(case=case['name'],pid=ledger['pid'],frames=frames,
                            requests_sha256=sha(request),responses_sha256=sha(response),
                            response_frames=response_frames,response_rows_exact=True,
                            opcodes=sorted(set(opcodes)),exit_code=0))
                    row.update(status='PASS',cases=actual['case_count'],checks=actual['checks'],
                        frames=actual['frames_before_EOF'],committed_84Q_snapshots=actual['committed_84Q_snapshots'],
                        rollback_bytes=actual['full_owner_rollback_bytes'],
                        full_rng_bytes=actual.get('full_rng_bytes',actual.get('full_rng_compared_bytes')),
                        native_transports=transports)
            except BaseException as error:
                row.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
                failures.append(dict(suite=suite,error=repr(error)))
            report['native_executed']=combined_native_execution(rows)
            dump(out/'report.json',report)
        for path,row in list(inputs.items()):
            bind(Path(path),row['sha256'])
        report.update(inputs=inputs,all_inputs_unchanged=True,
            native_executed=combined_native_execution(rows),
            cases=sum(row.get('cases',0) for row in rows),checks=sum(row.get('checks',0) for row in rows),
            frames=sum(row.get('frames',0) for row in rows))
        if failures:
            report['status']='FAIL'
        elif args.prepare_only:
            report['status']='PREPARED_NOT_RUN'
        elif args.suite=='all':
            require(report['cases']==7,'full resource7 denominator')
            report['status']='PASS_CURRENT_RESOURCE7'
        else:
            report['status']='PASS_SELECTED_RESOURCES'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc(),inputs=inputs)
    report['native_executed']=combined_native_execution(rows)
    dump(out/'report.json',report)
    print(json.dumps({key:report.get(key) for key in ('status','cases','checks','frames','failures','error')}))
    return int(report['status']=='FAIL')


if __name__=='__main__':
    raise SystemExit(main())
