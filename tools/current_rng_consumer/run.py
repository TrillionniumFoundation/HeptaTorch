#!/usr/bin/env python3
"""Run every old CPU RNG check against the current native RNG object.

The default PyTorch branch retains all eleven original source checks. An
explicit local-only flag is available for exploratory development; the pure
qualifier rejects that result unless explicitly invoked in the same local mode.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import stat

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from support import admit, require, sha
from process_evidence import actual_process, plan
from reference_evidence import reference_wheel_inputs

PACKAGE='validation/rng-independent-package.sha256'
SCRIPTS=('build.py','support.py','prepare.py','prepare_custody.py','run.py','qualify.py',
         'process_evidence.py','reference_evidence.py')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,required=True)
    parser.add_argument('--current-build',type=Path,required=True)
    parser.add_argument('--flat-linker',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--original-package',type=Path,
                        help='Optional exact frozen package location for sparse local checkouts; every original hash is mandatory.')
    parser.add_argument('--local-observed-oracle',action='store_true')
    args=parser.parse_args()
    require(not sys.flags.optimize and sys.flags.isolated and sys.flags.no_site
            and sys.flags.ignore_environment and sys.flags.dont_write_bytecode,
            'production runner must use -I -S -B with assertions enabled')
    repo,current,out=map(lambda p:p.resolve(),(args.repository,args.current_build,args.output))
    require(not (repo/'src/heptatorch.hepta').exists() and not (repo/'src/heptatorch.hepta').is_symlink(),
            'this consumer must run without restoring historical143')
    out.mkdir(parents=True,exist_ok=False)
    (out/'processes').mkdir()
    package=(args.original_package or repo).resolve()
    context={'schema':'heptatorch.rng-consumer.execution-context.v2',
             'local_observed_oracle':args.local_observed_oracle,
             'home':str(Path.home()), 'producer':actual_process(__file__,include_environment=False),
             'paths':{'repository':str(repo),'current_build':str(current),'output':str(out),
                      'implementation':str(HERE),'flat_linker':str(args.flat_linker.resolve()),
                      'original_package':str(package)}}
    context_raw=(json.dumps(context,indent=2,sort_keys=True)+'\n').encode()
    (out/'execution-context.json').write_bytes(context_raw)
    snapshots={}
    reports=[]
    manifest={'schema':'heptatorch.current-object-rng-consumer.v1','status':'RUNNING',
              'expected_cases':185,'expected_groups':{'cpython74':74,'pytorch76':76,'flat10':10,'independent25':25},
              'local_observed_oracle':args.local_observed_oracle,
              'full_installation_provenance_verified':False,'production_oracle_qualified':False,
              'historical143_restored':False,'old_rng_runtime_executed':False,
              'same_full_current_elf_executed':False,'same_current_rng_object_executed':False,
              'full_rng_gate_closed':False,'whole_pytorch_rewrite_complete':False,
              'execution_context_sha256':sha(context_raw),
              'processes':reports,'input_snapshots':[],'shared_reference_inputs':{},'groups':[]}
    def save():
        (out/'report.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    def snapshot(path,role):
        path=path.resolve();raw=path.read_bytes()
        require(path not in snapshots or snapshots[path][0]==raw,'source changed during preparation')
        destination=out/'inputs'/role
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
        snapshots[path]=(raw,role)
        row={'path':'inputs/'+role,'bytes':len(raw),'sha256':sha(raw),
             'mode':stat.S_IMODE(path.stat().st_mode)}
        if row not in manifest['input_snapshots']:manifest['input_snapshots'].append(row)
        return raw
    def execute(name):
        wanted=plan(context,name)
        original=dict(wanted,python=context['producer']['python'],
                      supervisor_pid=context['producer']['pid'],started=False,
                      stdout_EOF=False,stderr_EOF=False)
        (out/'processes'/(name+'.intent.json')).write_text(json.dumps(original,indent=2,sort_keys=True)+'\n')
        intent=dict(original,python_sha256=original['python']['sha256'],
                    timed_out=False,execution_error=None)
        reports.append(intent);save()
        started=time.monotonic()
        process=subprocess.Popen(wanted['argv'],cwd=wanted['cwd'],env=wanted['environment'],
                                 stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        intent.update(started=True,pid=process.pid)
        try:
            stdout,stderr=process.communicate(timeout=wanted['timeout_seconds'])
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            stdout,stderr=process.communicate();intent['timed_out']=True
        for suffix,raw in [('stdout',stdout),('stderr',stderr)]:
            (out/'processes'/(name+'.'+suffix)).write_bytes(raw)
        intent.update(exit_code=process.returncode,stdout_EOF=True,stderr_EOF=True,
                      stdout_sha256=sha(stdout),stderr_sha256=sha(stderr),stdout_bytes=len(stdout),stderr_bytes=len(stderr),
                      elapsed_seconds=time.monotonic()-started,
                      stdout_path='processes/'+name+'.stdout',stderr_path='processes/'+name+'.stderr')
        actual_path=out/wanted['actual_process_path']
        if actual_path.is_file():
            actual_raw=actual_path.read_bytes()
            intent['actual_process_sha256']=sha(actual_raw)
        else:
            intent['execution_error']='Child did not leave its actual process receipt'
        (out/'processes'/(name+'.json')).write_text(json.dumps(intent,indent=2)+'\n')
        save()
        return process.returncode==0 and not intent['timed_out'] and intent['execution_error'] is None and not stderr
    try:
        for name in SCRIPTS:snapshot(HERE/name,'implementation/'+name)
        snapshot(Path(sys.executable),'runtime/python.bin')
        paths=['tools/rng-cpu/rng.hepta','tools/rng-cpu/profile.json','tools/rng-cpu/upstream-sources.json',
               'tools/rng-cpu/THIRD_PARTY_NOTICES.txt','tools/dtype-f16/f16.hepta','tools/dtype-bf16/bf16.hepta',
               'tests/check_rng_custody.py','tests/rng_cpu_oracle.py','tests/rng_mt_reference.cpp',
               'tests/pytorch_oracle.py','compatibility/pytorch-oracle-lock.json','compatibility/upstream-lock.json',
               'compatibility/pytorch-reference-requirements.txt',PACKAGE,
               'tests/pinned_reference_custody.py','compatibility/pytorch-official-wheel-members.json']
        upstream=json.loads((repo/'tools/rng-cpu/upstream-sources.json').read_bytes())
        paths.extend(row['path'] for row in upstream['sources'])
        for path in paths:snapshot(repo/path,'repository/'+path)
        package_rows=[]
        for line in (repo/PACKAGE).read_text().splitlines():
            expected,path=line.split(None,1)
            raw=snapshot(package/path,'frozen-package/'+path)
            require(sha(raw)==expected,'complete original frozen package: '+path)
            package_rows.append({'path':path,'sha256':expected,'bytes':len(raw)})
        require(len(package_rows)==6,'complete original RNG independent package')
        manifest['frozen_package']=package_rows
        pip=repo/'build/pytorch-install.json'
        if pip.is_file():snapshot(pip,'repository/build/pytorch-install.json')
        support_pip=repo/'build/pytorch-reference-support-install.json'
        if support_pip.is_file():snapshot(support_pip,'repository/build/pytorch-reference-support-install.json')
        manifest['shared_reference_inputs']=reference_wheel_inputs(repo)
        manifest['original_pip_install_report_available']=pip.is_file()
        require(execute('build'), 'native consumer build failed')
        manifest['migration']=admit(repo,out/'packet-build',current);save()
        code=out/'code/adapted'
        require(execute('prepare'),'original comparison admission failed')
        require(execute('prepare-custody'),'original custody admission failed')
        require(execute('custody'),'current RNG/old source/licensed source custody failed')
        (out/'custody.json').write_bytes((out/'processes/custody.stdout').read_bytes())
        for name in ('cpython74','flat10','independent25','pytorch76'):
            ok=execute(name)
            path=out/name/'report.json'
            row={'name':name,'executed':True,'process_passed':ok,'report_path':name+'/report.json'}
            if path.is_file():
                raw=path.read_bytes();value=json.loads(raw)
                row.update(report_sha256=sha(raw),status=value['status'],cases=len(value['rows']))
            else:row.update(status='MISSING_REPORT',cases=0)
            manifest['groups'].append(row);save()
        require(all(row['process_passed'] and row['cases']==manifest['expected_groups'][row['name']]
                    for row in manifest['groups']),'one or more full original suites failed')
        require(not (repo/'src/heptatorch.hepta').exists() and not (repo/'src/heptatorch.hepta').is_symlink(),
                'historical143 must remain absent')
        for path,(before,role) in snapshots.items():
            require(path.read_bytes()==before,'input changed during execution: '+role)
        require(admit(repo,out/'packet-build',current)==manifest['migration'],'same current build after all suites')
        require(reference_wheel_inputs(repo)==manifest['shared_reference_inputs'],
                'complete original official wheel inputs unchanged through all four suites')
        manifest.update(status='PASS_EXPLORATORY_NATIVE_ONLY' if args.local_observed_oracle else 'PASS',
                        cases=185,all_inputs_unchanged=True,same_current_rng_object_executed=True,
                        full_installation_provenance_verified=not args.local_observed_oracle,
                        production_oracle_qualified=not args.local_observed_oracle)
        save()
        from qualify import qualify
        proof=qualify(repo,current,out,allow_unverified_local=args.local_observed_oracle)
        (out/'qualification.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'status':manifest['status'],'cases':185,'proof_status':proof['status'],
                          'current_rng_object_sha256':manifest['migration']['current_rng_object_sha256'],
                          'full_installation_provenance_verified':manifest['full_installation_provenance_verified']}))
        return 0
    except Exception as error:
        manifest.update(status='FAIL',error_type=type(error).__name__,error=str(error),
                        full_installation_provenance_verified=False,production_oracle_qualified=False)
        save();print(json.dumps({'status':'FAIL','error':str(error)}),file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
