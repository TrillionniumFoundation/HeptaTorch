#!/usr/bin/env python3
"""Pure reader: replay every raw RNG packet and borrowed-buffer observation.

No subprocess, native executable, C++ reference, or Torch operation is started.
Original comparison code is regenerated as data to verify the exact executed
files; the complete 35-case historical row payload must also remain identical.
"""
from pathlib import Path, PurePosixPath
import argparse
import contextlib
import io
import json
import struct
import sys
import tempfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from support import IMAGE, admit, require, sha
from process_evidence import read_context, read_processes
from reference_evidence import read_reference

IMPLEMENTATIONS=('build.py','support.py','prepare.py','prepare_custody.py','run.py','qualify.py',
                 'process_evidence.py','reference_evidence.py')


def independent_default_state():
    """Only the frozen original make_state() literal bytes, not RNG outcomes."""
    state=bytearray(5056)
    struct.pack_into('<QIIQ',state,0,0xfedcba9876543210,624,1,1)
    for i in range(624):
        struct.pack_into('<Q',state,24+8*i,(0x6c078965*(i+1)+0x12345678)&0xffffffff)
    struct.pack_into('<Q',state,5024,0xfff0000000000123)
    struct.pack_into('<I',state,5040,1)
    struct.pack_into('<I',state,5048,0x7f800123);state[5052]=1
    return bytes(state)


def qualify(repository,current_build,directory,allow_unverified_local=False):
    repo,current,root=map(lambda p:Path(p).resolve(),(repository,current_build,directory))
    checked={}
    def read(relative,expected=None):
        part=PurePosixPath(relative)
        require(not part.is_absolute() and str(part)==relative and '..' not in part.parts and '\\' not in relative,
                'normalized evidence-relative path')
        path=root.joinpath(*part.parts)
        require(not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(root)
                and all(not parent.is_symlink() for parent in path.parents if parent!=root and parent.is_relative_to(root)),
                'regular contained evidence file: '+relative)
        raw=path.read_bytes()
        require(expected is None or sha(raw)==expected,'evidence identity: '+relative)
        require(path not in checked or checked[path]==raw,'evidence changed')
        checked[path]=raw;return raw
    report_raw=read('report.json');report=json.loads(report_raw)
    require(report['schema']=='heptatorch.current-object-rng-consumer.v1','consumer report schema')
    local=report['local_observed_oracle']
    require(type(local) is bool, 'explicit boolean execution mode')
    if local:
        require(allow_unverified_local and report['status']=='PASS_EXPLORATORY_NATIVE_ONLY'
                and report['full_installation_provenance_verified'] is False
                and report['production_oracle_qualified'] is False,'exploratory results do not qualify the production consumer')
    else:
        require(report['status']=='PASS' and report['full_installation_provenance_verified'] is True
                and report['production_oracle_qualified'] is True,'strict production reference provenance')
    require(report['cases']==report['expected_cases']==185
            and report['expected_groups']=={'cpython74':74,'pytorch76':76,'flat10':10,'independent25':25},'all four full original suites')
    read('execution-context.json',report['execution_context_sha256'])
    context=read_context(read,local)
    migration=admit(repo,root/'packet-build',current)
    require(report['migration']==migration and report['all_inputs_unchanged'] is True,'same actual current object before/after complete original suites')
    require(report['historical143_restored'] is False and report['old_rng_runtime_executed'] is False
            and report['same_full_current_elf_executed'] is False and report['same_current_rng_object_executed'] is True
            and report['full_rng_gate_closed'] is False and report['whole_pytorch_rewrite_complete'] is False,
            'migration scope must not claim whole current ELF or full RNG closure')
    snapshot_paths=[row['path'] for row in report['input_snapshots']]
    require(len(snapshot_paths)==len(set(snapshot_paths)), 'unique complete input snapshots')
    for row in report['input_snapshots']:
        raw=read(row['path'],row['sha256'])
        require(len(raw)==row['bytes'] and type(row['mode']) is int,'complete input snapshot bytes and mode')
    for name in IMPLEMENTATIONS:
        require(read('inputs/implementation/'+name)==(HERE/name).read_bytes(),'actual executed implementation source')
    repo_paths=['tools/rng-cpu/rng.hepta','tools/rng-cpu/profile.json','tools/rng-cpu/upstream-sources.json',
               'tools/rng-cpu/THIRD_PARTY_NOTICES.txt','tools/dtype-f16/f16.hepta','tools/dtype-bf16/bf16.hepta',
               'tests/check_rng_custody.py','tests/rng_cpu_oracle.py','tests/rng_mt_reference.cpp',
               'tests/pytorch_oracle.py','compatibility/pytorch-oracle-lock.json','compatibility/upstream-lock.json',
               'compatibility/pytorch-reference-requirements.txt','validation/rng-independent-package.sha256']
    upstream=json.loads(read('inputs/repository/tools/rng-cpu/upstream-sources.json'))
    repo_paths.extend(row['path'] for row in upstream['sources'])
    for path in repo_paths:
        require(read('inputs/repository/'+path)==(repo/path).read_bytes(), 'every original repository input unchanged: '+path)
    package=read('inputs/repository/validation/rng-independent-package.sha256',
                 '51e1720a077540695cc19f31f64589f3581f9140e35c68b9780b1fce1205cd03')
    package_rows=[]
    for line in package.decode().splitlines():
        digest,path=line.split(None,1);raw=read('inputs/frozen-package/'+path,digest)
        package_rows.append({'path':path,'sha256':digest,'bytes':len(raw)})
    require(len(package_rows)==6 and report['frozen_package']==package_rows,'all original frozen package members')
    expected_snapshots={'inputs/implementation/'+name for name in IMPLEMENTATIONS}
    expected_snapshots|={'inputs/repository/'+path for path in repo_paths}
    expected_snapshots|={'inputs/frozen-package/'+row['path'] for row in package_rows}
    expected_snapshots.add('inputs/runtime/python.bin')
    optional_snapshots={'inputs/repository/build/pytorch-install.json',
                        'inputs/repository/build/pytorch-reference-support-install.json'}
    require(expected_snapshots <= set(snapshot_paths) <= expected_snapshots|optional_snapshots,
            'complete declared input census, without omitted source or arbitrary substitute')
    require(report['original_pip_install_report_available'] is
            ('inputs/repository/build/pytorch-install.json' in snapshot_paths), 'actual pip report availability')
    if not local:
        require(set(snapshot_paths)==expected_snapshots|optional_snapshots
                and not (root/'pytorch76/observed-torch.json').exists(), 'strict input census excludes exploratory receipt')
    frozen_raw=read('inputs/frozen-package/validation/native-rng-independent.json',
                    'bf6fe60a4099931ce3eb6f9ab565e5a93b04552f473e1e68edd37fe3a98e224b')
    frozen=json.loads(frozen_raw)
    require(frozen['status']=='PASS' and frozen['final_cases']==35 and frozen['retired_failure_observations']==10,
            'complete original independent comparison and retained former failure record')
    # Recompute exact allowed build/identity/ABI transformations. This executes
    # only the repository's source adapter, and starts no native or reference.
    import prepare
    import prepare_custody
    previous=sys.argv
    try:
        with tempfile.TemporaryDirectory(prefix='rng-source-reader-') as temporary:
            expected=Path(temporary)/'adapted'
            with contextlib.redirect_stdout(io.StringIO()):
                sys.argv=['prepare','--repository',str(repo),'--build',str(root/'packet-build'),
                          '--current-build',str(current),'--output',str(expected)]
                prepare.main()
                sys.argv=['prepare-custody','--repository',str(repo),'--output',str(expected/'check_rng_custody.py')]
                prepare_custody.main()
            for path in expected.iterdir():
                require(read('code/adapted/'+path.name)==path.read_bytes(),'exact permitted original source transformation: '+path.name)
            for name in ('build.py','support.py','process_evidence.py','reference_evidence.py'):
                require(read('code/'+name)==(expected.parent/name).read_bytes(),'actual imported admission snapshot')
    finally:sys.argv=previous
    source_inverse=json.loads(read('code/adapted/source-inverse.json'))
    require(source_inverse['original_eleven_provenance_assertions_retained_in_strict_branch'] is True
            and all(x['whole_file_inverse_exact'] is True for x in source_inverse['scripts']),
            'all original tests retained behind exact source inverse')
    custody=json.loads(read('custody.json'))
    require(custody['status']=='PASS' and custody['functions_verified']==18
            and custody['reused_function_bodies_identical']==8 and custody['primary_source_files']==6
            and custody['original_mt19937_notice_retained'] is True
            and custody['all_17_non_entry_bodies_equal'] is True and custody['historical143_source_restored'] is False,
            'all old body, dtype, source license and notice requirements retained')
    actuals=read_processes(read,report,context)
    require(read('custody.json')==read('processes/custody.stdout'), 'custody result is actual complete child stdout')
    build=json.loads(read('packet-build/build.json'))
    require(build['actual_process']==actuals['build'] and build['producer_paths']=={
        'repository':context['paths']['repository'],'current_build':context['paths']['current_build'],
        'output':context['paths']['output']+'/packet-build'}, 'exact actual build process context')
    build_files={'build.json','packet_adapter.hepta','rng03.hepta','rng03.o','packet_adapter.o','compiler','linker','rng-native'}
    build_files|={name+suffix for name in ('compile-0','compile-1','link-0','link-1')
                  for suffix in ('.request.bin','.stdout.bin','.stderr.bin','.intent.json')}
    require({p.name for p in (root/'packet-build').iterdir()}==build_files, 'complete exact native build raw artifact set')
    for name in sorted(build_files):read('packet-build/'+name)
    groups={row['name']:row for row in report['groups']}
    require(set(groups)==set(report['expected_groups']) and len(report['groups'])==4,'complete unique original suite set')
    results={}
    packet_frames=0;independent_frames=0;exact_values=0;state_bytes=0;flat_count=0
    for name,wanted_count in report['expected_groups'].items():
        group=groups[name]
        require(group['process_passed'] is True and group['executed'] is True and group['cases']==wanted_count,'complete actual group')
        raw=read(group['report_path'],group['report_sha256']);value=json.loads(raw)
        require(json.loads(read('processes/'+name+'.stdout'))==value,'complete original report is actual child stdout')
        expected_status='PASS_EXPLORATORY_NATIVE_ONLY' if name=='pytorch76' and local else 'PASS'
        require(value['status']==group['status']==expected_status and len(value['rows'])==wanted_count
                and len({r['case'] for r in value['rows']})==wanted_count
                and (name=='flat10' or all(r['status']=='PASS' for r in value['rows'])) and value['migration']==migration,
                'every original case and exact current build binding')
        results[name]=value
        if name in ('cpython74','pytorch76'):
            require(value['native_executable_sha256']==IMAGE and value['failed_cases']==0
                    and value['oracle_sha256']==sha(read('code/adapted/rng_cpu_oracle.py')),
                    'actual numerical oracle and native identities')
            require(value['allow_unverified_local']==(local and name=='pytorch76'), 'explicit local scope only')
            require(value['native_compilations_byte_identical'] is True and value['source_and_image_unchanged'] is True,
                    'retained deterministic native build and immutability requirements')
            expected_frames,expected_values,expected_state=(110,718619,434816) if name=='cpython74' else (114,719337,455040)
            require((value['native_invocations'],value['exact_values'],value['serialized_state_bytes_compared'])
                    ==(expected_frames,expected_values,expected_state),'complete original numerical/state denominator')
            frames=[frame for row in value['rows'] for frame in row.get('frames',[row])]
            require(len(frames)==len(value['captured_native_packets'])==expected_frames,'complete original raw packet set')
            for i,(expected,receipt) in enumerate(zip(frames,value['captured_native_packets'])):
                prefix=name+'/raw/%04d'%i
                require(json.loads(read(prefix+'.json'))==receipt and receipt['index']==i,'actual raw packet receipt')
                request=read(prefix+'.request.bin',receipt['request_sha256'])
                response=read(prefix+'.response.bin',receipt['response_sha256'])
                stderr=read(prefix+'.stderr.bin',receipt['stderr_sha256'])
                require(receipt['returncode']==0 and not stderr and receipt['stderr_bytes']==0
                        and len(request)==receipt['request_bytes'] and len(response)==receipt['response_bytes'],
                        'actual packet exit and complete binary streams')
                require(sha(request)==expected['request_sha256'] and sha(response)==expected['response_sha256'],
                        'every original frame request/response is captured')
                if 'native_status' in expected:
                    require(response==struct.pack('<3Q',expected['native_status'],0,0),'unchanged original packet rejection')
                else:
                    count=expected['values']
                    require(len(response)==5080+8*count and response[:24]==struct.pack('<3Q',0,count,5056),
                            'unchanged original response shape')
                    require(sha(response[24:5080])==expected['state_sha256'] and expected['state_bytes_exact']==5056
                            and sha(response[5080:])==expected['reference_values_sha256'],
                            'all independently compared original value words and serialized-state bytes')
            require(len(value['independent_cpp_cpython_seed_proofs'])==(15 if name=='cpython74' else 16)
                    and all(x['status']=='PASS' and x['independent_raw32_words']==4097 and x['cpp_and_cpython_final_state_equal'] is True
                            for x in value['independent_cpp_cpython_seed_proofs']), 'complete original C++/CPython engine qualification')
            read(name+'/mt-standard-reference',value['cpp_reference_executable_sha256'])
            require(value['cpp_reference_source_sha256']==sha(read('inputs/repository/tests/rng_mt_reference.cpp')),'exact independent standard-library source')
            packet_frames+=expected_frames;exact_values+=expected_values;state_bytes+=expected_state
        else:
            key='final_flat_boundary' if name=='flat10' else 'final_state_and_admission'
            require(value['rows']==frozen['observations'][key]['report']['rows'],'every complete original historical 35-case row remains identical')
            script='flat_probe.py' if name=='flat10' else 'review_rng.py'
            require(value['script_sha256']==sha(read('code/adapted/'+script)) and value['native_sha256']==IMAGE,
                    'actual independent comparator and native identities')
            native_responses={};native_requests={}
            if name=='independent25':
                read(name+'/state-reference',value['cpp_reference_image_sha256'])
                require(value['cpp_reference_source_sha256']==sha(read('code/adapted/state_reference.cpp')),
                        'exact independent inverse-tempering/state C++ source')
                paired=[]
                for original_row in value['rows']:
                    case=original_row['case']
                    if 'response_sha256' not in original_row and 'request_sha256' not in original_row:
                        continue
                    request=read(name+'/'+case+'.request.bin',original_row.get('request_sha256'))
                    response=read(name+'/'+case+'.response.bin',original_row.get('response_sha256'))
                    if 'native_status' in original_row:
                        require(response==struct.pack('<3Q',original_row['native_status'],0,0),
                                'complete independent invalid packet response')
                    else:
                        require(len(response)==5080+8*original_row['values']
                                and response[:24]==struct.pack('<3Q',0,original_row['values'],5056),
                                'complete independent values and state response')
                    if case=='reseed-invalidates-both-caches':
                        expected_request=b'HTRNG001'+struct.pack('<IIQQ',1,2,0,5056)+independent_default_state()
                        expected_request+=struct.pack('<IIQQQ',7,0,(1<<64)-1,0,0)+struct.pack('<IIQQQ',0,5,0,0,0)
                        require(request==expected_request, 'exact original reseed-and-cache packet')
                    native_responses[case]=response;native_requests[case]=request;paired.append(case)
                counterpart='reseed-fresh-counterpart'
                counterpart_request=read(name+'/'+counterpart+'.request.bin')
                counterpart_response=read(name+'/'+counterpart+'.response.bin')
                require(counterpart_request==b'HTRNG001'+struct.pack('<IIQQ',0,1,(1<<64)-1,0)
                        +struct.pack('<IIQQQ',0,5,0,0,0), 'original fresh-reseed comparison packet')
                require(counterpart_response==native_responses['reseed-invalidates-both-caches'],
                        'original complete fresh-reseed response equality')
                paired.append(counterpart)
                require(len(paired)==23 and {p.name for p in (root/name).glob('*.request.bin')}
                        =={case+'.request.bin' for case in paired}
                        and {p.name for p in (root/name).glob('*.response.bin')}=={case+'.response.bin' for case in paired},
                        'all 23 independent actual stdin packets are bound')
                independent_frames+=23
            expected_rows=[(r['case'],r) for r in value['rows']] if name=='flat10' else [(r['case'],r['flat_observation']) for r in value['rows'] if 'flat_observation' in r]
            require(len(expected_rows)==(10 if name=='flat10' else 15),'complete original borrowed-span observation set')
            directory=root/name/'flat-evidence'
            require(len(list(directory.glob('*.json')))==len(expected_rows),'no missing or extra flat observation')
            for i,(case,expected) in enumerate(expected_rows):
                row=json.loads(read(name+'/flat-evidence/%04d.json'%i))
                require(row['index']==i and row['native_sha256']==IMAGE,'actual ABI3 observation identity')
                original={k:v for k,v in expected.items() if k!='case'}
                require(row['observed']==original,'original observation belongs to complete actual raw memory')
                require(row['output_capacity']==original['output_capacity']
                        and row['workspace_capacity']==original['workspace_capacity'],'original declared borrowed-span capacities')
                blobs={}
                require(set(row['blobs'])=={'request','input_after','return','output','workspace'},
                        'complete exact raw borrowed-buffer blob roles')
                for key,b in row['blobs'].items():
                    require(b['path']=='%04d.%s.bin'%(i,key), 'unique fixed raw borrowed-buffer file name')
                    raw=read(name+'/flat-evidence/'+b['path'],b['sha256']);require(len(raw)==b['bytes'],'complete actual flat memory')
                    blobs[key]=raw
                require(len(blobs['output'])==len(blobs['workspace'])==8192 and len(blobs['return'])==4
                        and len(blobs['input_after'])==len(blobs['request']), 'complete original raw memory extents')
                if name=='flat10':
                    expected_request=b'HTRNG001'+struct.pack('<IIQQ',0,0,17,0)
                elif case in native_requests:
                    expected_request=native_requests[case]
                    require(sha(expected_request)==next(r['request_sha256'] for r in value['rows'] if r['case']==case),
                            'original frozen independent case request hash')
                else:
                    empty=b'HTRNG001'+struct.pack('<IIQQ',0,0,0,0)
                    requests={'workspace-one-byte-short':empty,
                              'output-one-value-short':b'HTRNG001'+struct.pack('<IIQQ',0,1,0,0)+struct.pack('<IIQQQ',0,1,0,0,0),
                              'input-header-one-byte-short':empty[:31]}
                    require(case in requests, 'known exact original capacity request')
                    expected_request=requests[case]
                require(blobs['request']==expected_request and blobs['input_after']==expected_request,
                        'actual complete flat request belongs to the exact original case')
                output,workspace=blobs['output'],blobs['workspace'];cap=row['output_capacity']
                if name=='flat10':
                    require(read(name+'/'+case+'.output.bin')==output,'complete original flat output artifact')
                elif case in native_responses:
                    require(output[:24]==native_responses[case], 'original flat and stdin failure responses equal')
                changed=[j for j,x in enumerate(output) if x!=0xa5]
                require(row['observed']['signal'] is None and row['observed']['exit_code']==0
                        and len(blobs['return'])==4 and struct.unpack('<I',blobs['return'])[0]==row['observed']['returned_u32'],
                        'actual ABI3 return and successful forked call')
                require(sha(output)==original['output_sha256'] and output[:32].hex()==original['output_prefix_hex']
                        and len(changed)==original['output_changed_bytes']
                        and ([min(changed),max(changed)] if changed else None)==original['output_changed_range'],
                        'all original output physical-byte predicates')
                require((workspace==b'\x5a'*len(workspace))==original['workspace_unchanged']
                        and (blobs['input_after']==blobs['request'])==original['input_unchanged'] is True
                        and (output[cap:]==b'\xa5'*(len(output)-cap))==original['outside_declared_output_capacity_unchanged'] is True,
                        'all original workspace atomicity, input preservation and output-capacity predicates')
                wanted_workspace=output[24:5080]+b'\x5a'*(8192-5056) if name=='flat10' and case=='output-capacity-5080' else b'\x5a'*8192
                require(workspace==wanted_workspace, 'complete exact original workspace contents and unwritten suffix')
                flat_count+=1
    torch=results['pytorch76']
    require(torch['full_installation_provenance_verified'] is (not local)
            and torch['production_oracle_qualified'] is (not local), 'actual reference qualification scope')
    reference=read_reference(read,torch,context,actuals['pytorch76'])
    expected_reference_files=set(json.loads(read('pytorch76/reference-inputs.json'))['inputs'])
    require({p.name for p in (root/'pytorch76/reference-inputs').iterdir()}==expected_reference_files,
            'all raw reference entities, no unbound extra or omitted file')
    for path,before in checked.items():require(path.read_bytes()==before,'evidence changed during pure replay')
    return {'schema':'heptatorch.current-object-rng-consumer-qualification.v1',
            'status':'PASS_EXPLORATORY_REPLAY_ONLY' if local else 'PASS_CURRENT_RNG_CONSUMER',
            'consumer_report_sha256':sha(report_raw),'cases':185,'packet_frames':packet_frames+independent_frames,
            'original_reference_packet_frames':packet_frames,'independent_packet_frames':independent_frames,
            'exact_values':exact_values,'serialized_state_bytes':state_bytes,
            'physical_borrowed_buffer_observations':flat_count,'complete_historical_rows_preserved':35,
            'original_74_76_10_25_denominators_retained':True,'migration':migration,
            'reference_custody':reference,'supervised_python_processes':8,'supervised_native_tool_processes':4,
            'exact_flat_case_requests_bound':25,'all_flat_buffer_extents_verified':True,
            'current_build_input_bindings':build['input_bindings'],
            'native_executed_by_reader':False,'Torch_executed_by_reader':False,
            'full_installation_provenance_verified':not local,'production_oracle_qualified':not local,
            'historical143_restored':False,'full_rng_gate_closed':False,'whole_pytorch_rewrite_complete':False,
            'inputs':{str(path.relative_to(root)):{'bytes':len(raw),'sha256':sha(raw)} for path,raw in checked.items()}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',type=Path,required=True)
    p.add_argument('--current-build',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-unverified-local',action='store_true')
    a=p.parse_args();result=qualify(a.repository,a.current_build,a.evidence,a.allow_unverified_local)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('status','cases','packet_frames','physical_borrowed_buffer_observations')}))


if __name__=='__main__':main()
