#!/usr/bin/env python3
"""Run the current HIA logical-reduction suite using exact frozen Torch results."""
import argparse
import hashlib
import importlib.util
import json
import lzma
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
REFERENCE_SHA='e2874986d52e62725d54767ea5824a76d066e948b5d54a99e3b24f9e32c0e425'
MANIFEST_SHA='8c9ec07dd94637fc7f1bd3dfc2f24728e4da6d19ab21a0287a05342c3d2f5256'
ARCHIVE_SHA='ef98c25d7fc8455e59737a70fa90de4e7ce62294648642a81c3f9d7dce314bcb'
sha=lambda b:hashlib.sha256(b).hexdigest()


def require(ok,rule):
    if not ok:raise ValueError(rule)


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


class Inputs:
    def __init__(self):self.rows={}

    def bind(self,path,expected=None):
        path=Path(path).absolute();require(not path.is_symlink(),'symlink input: '+str(path))
        before=path.stat();require(stat.S_ISREG(before.st_mode),'nonregular input: '+str(path))
        raw=path.read_bytes();after=path.stat()
        identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mode,s.st_mtime_ns,s.st_ctime_ns)
        require(identity(before)==identity(after),'input changed during read: '+str(path))
        row=dict(bytes=len(raw),sha256=sha(raw),mode=stat.S_IMODE(after.st_mode))
        require(expected is None or row['sha256']==expected,'input digest mismatch: '+str(path))
        require(str(path) not in self.rows or self.rows[str(path)]==row,'bound input changed: '+str(path))
        self.rows[str(path)]=row
        return raw

    def recheck(self):
        for path,row in list(self.rows.items()):self.bind(path,row['sha256'])


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def run_native(command,out):
    receipt=dict(argv=command,timeout_seconds=900,started=False,stdout_EOF=False,stderr_EOF=False)
    child=None;stdout=stderr=b'';start=time.monotonic()
    try:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            cwd=ROOT,start_new_session=True,env={'PATH':os.defpath,'LANG':'C.UTF-8','LC_ALL':'C.UTF-8'})
        receipt['started']=True
        try:stdout,stderr=child.communicate(timeout=900)
        except subprocess.TimeoutExpired:
            receipt['timed_out']=True;os.killpg(child.pid,signal.SIGKILL)
            stdout,stderr=child.communicate(timeout=10)
        receipt.update(exit_code=child.returncode,stdout_EOF=True,stderr_EOF=True)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL);stdout,stderr=child.communicate(timeout=10)
            receipt.update(forced_cleanup=True,exit_code=child.returncode)
        receipt.update(elapsed_seconds=time.monotonic()-start,stdout_bytes=len(stdout),stderr_bytes=len(stderr),
                       stdout_sha256=sha(stdout),stderr_sha256=sha(stderr))
        (out/'runner.stdout').write_bytes(stdout);(out/'runner.stderr').write_bytes(stderr)
        dump(out/'runner.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-directory',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();build=args.build_directory.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    require(not sys.flags.optimize,'assertions must be enabled')
    inputs=Inputs();report=dict(schema='heptatorch.logical-reductions.qualification.v1',status='RUNNING',
        native_executed=False,torch_executed=False,full_pytorch_rewrite_complete=False,schema_closure=False)
    exit_code=0
    try:
        inputs.bind(Path(__file__).resolve())
        profile_path=ROOT/'tools/current_runtime/qualification.py';inputs.bind(profile_path)
        profile=module(profile_path,'logical_current_profile')
        built=profile.admit_build(build,inputs.bind)
        cases_path=Path(__file__).with_name('cases.py');inputs.bind(cases_path)
        cases=module(cases_path,'logical_literal_cases')
        require(len(cases.CASES)==147,'all 147 literal cases are required')
        package=ROOT/'validation/logical-reductions-reference'
        manifest=json.loads(inputs.bind(package/'manifest.json',MANIFEST_SHA))
        archive=inputs.bind(package/'report.json.xz',ARCHIVE_SHA)
        raw=lzma.decompress(archive)
        require(len(archive)==9704 and len(raw)==223897 and sha(raw)==REFERENCE_SHA,'frozen reference changed')
        require(manifest['schema']=='heptatorch.logical-reductions.fixed-reference.v1' and
                manifest['status']=='OBSERVED' and manifest['original_torch_cases']==147 and
                manifest['source_report']==dict(bytes=len(raw),sha256=REFERENCE_SHA) and
                manifest['case_sha256']==cases.CASE_SHA256 and manifest['report_bytes_unchanged'] is True and
                manifest['native_qualification'] is False,'reference scope or source binding changed')
        previous=manifest['previous_reference']
        previous_raw=lzma.decompress(inputs.bind(package/previous['archive']['path'],previous['archive']['sha256']))
        require(sha(previous_raw)==previous['source_report']['sha256'] and
                json.loads(previous_raw)['rows']==json.loads(raw)['rows'][:138] and
                previous['rows_unchanged_in_current_reference']==138,'original 138 observations changed')
        inputs.bind(package/previous['execution']['path'],previous['execution']['sha256'])
        execution=json.loads(inputs.bind(package/'execution.json',manifest['execution']['sha256']))
        require(execution['execution_performed'] is True and execution['observed_exit_code']==0 and
                execution['stdout_EOF'] is True and execution['stderr_EOF'] is True and
                execution['process']['output_complete'] is True and execution['process']['state']=='succeeded' and
                execution['report']['sha256']==REFERENCE_SHA,'actual reference execution was incomplete')
        for rel,pin in manifest['source_files'].items():
            source=inputs.bind(ROOT/rel,pin['sha256']);require(len(source)==pin['bytes'],'reference source size changed')
        reference=out/'reference.json';reference.write_bytes(raw);reference.chmod(0o600);inputs.bind(reference,REFERENCE_SHA)
        script=Path(__file__).with_name('native.py');inputs.bind(script)
        inputs.bind(ROOT/'tests/differentiable_view_native_review/protocol.py')
        inputs.bind(ROOT/'tests/differentiable_view_native_review/owner_client.py')
        command=[sys.executable,'-E','-S','-B',str(script),'--repository',str(ROOT),
                 '--build-directory',str(build),'--reference',str(reference),'--output',str(out/'native')]
        receipt=run_native(command,out);report.update(native_executed=receipt['started'],runner=receipt)
        require(receipt.get('exit_code')==0 and receipt['stdout_EOF'] and receipt['stderr_EOF'] and
                not receipt.get('timed_out') and not receipt.get('forced_cleanup') and receipt['stderr_bytes']==0,
                'native runner failed or streams were incomplete')
        native_path=out/'native/report.json';native_raw=inputs.bind(native_path);native=json.loads(native_raw)
        require(native['schema']=='heptatorch.logical-reductions.native.v1' and native['status']=='PASS' and
                native['cases']==native['expected_cases']==len(native['rows'])==147 and
                [r['name'] for r in native['rows']]==[r['name'] for r in cases.CASES] and
                all(r['status']=='PASS' for r in native['rows']),'all native reference cases must pass')
        require(native['native_executed'] is True and native['torch_executed'] is False and
                native['inputs_unchanged'] is True and native['full_pytorch_rewrite_complete'] is False and
                native['source_sha256']==built['source_sha256'] and native['native_sha256']==built['native_sha256'] and
                native['inputs'][str(build/'build.json')]==sha(inputs.bind(build/'build.json')) and
                native['inputs'][str(reference)]==REFERENCE_SHA,'native source/build/reference binding changed')
        control=native['packet_controls'];alloc=native['allocation_controls']
        expected_controls=[name+str(op) for op in (118,119) for name in
            ('reserved48','reserved56','invalid-dtype','invalid-keepdim','too-many-axes','invalid-source',
             'missing-axis-payload','second-command-rollback')]+['stale-source118','stale-source119']
        require(control['status']=='PASS' and control['controls']==18 and control['names']==expected_controls,
                'all eighteen packet controls must pass')
        require(len(alloc)==2 and [a['opcode'] for a in alloc]==[118,119] and
                all(a['status']=='PASS' and a['full_owner_rollback'] is True and a['same_request_after_restore'] is True and
                    a['attempts'][-1]['status']==3 for a in alloc),'both real allocation failures and retries are required')
        folders=['case-%03d'%i for i in range(147)]+['packet-controls','allocation-118','allocation-119']
        require(len(list((out/'native').glob('*/raw/transport.json')))==150,'unexpected native child set')
        frames=0
        for folder in folders:
            transport=json.loads(inputs.bind(out/'native'/folder/'raw/transport.json'))
            observed=json.loads(inputs.bind(out/'native'/folder/'observations.json'))
            require(transport['exit_code']==observed['exit_code']==0 and observed['stdout_EOF'] is True and
                    observed['stderr_EOF'] is True and transport['executable_sha256']==built['native_sha256'] and
                    observed['frames']==transport['frames'],'native child exit, EOF or image binding failed: '+folder)
            frames+=transport['frames']
        require(frames==native['frames'],'native frame denominator mismatch')
        inputs.recheck()
        report.update(status='PASS',cases=147,packet_controls=18,allocation_controls=2,native_children=150,
            frames=frames,case_checks=native['checks'],
            total_checks=native['checks']+control['checks']+sum(a['checks'] for a in alloc),
            source_sha256=built['source_sha256'],native_sha256=built['native_sha256'],
            actual_build_sha256=sha(inputs.bind(build/'build.json')),reference_sha256=REFERENCE_SHA,
            native_report_sha256=sha(native_raw),all_native_EOF=True,all_inputs_unchanged=True)
    except BaseException as error:
        report.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        exit_code=1
    report['inputs']=inputs.rows;dump(out/'report.json',report)
    print(json.dumps({k:report.get(k) for k in ('status','cases','packet_controls','allocation_controls','frames','error')}))
    return exit_code

if __name__=='__main__':raise SystemExit(main())
