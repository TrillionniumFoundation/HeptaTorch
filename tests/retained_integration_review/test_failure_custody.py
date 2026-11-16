#!/usr/bin/env python3
"""A wrong tool pin must fail before native execution and preserve its evidence."""
from pathlib import Path
import argparse
import ast
import copy
import difflib
import hashlib
import json
import lzma
import stat
import subprocess
import sys
import tarfile


ORIGINAL = '1376074fb29bc9721018eb9fd05f90dc041ba135688e789f3ca330dab3860f51'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('qualifier','original-qualifier','repo','parent-build','linker','stack-build','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for name in vars(a):setattr(a,name,getattr(a,name).resolve())
    a.output.mkdir(parents=True,exist_ok=False)
    original,new=a.original_qualifier.read_bytes(),a.qualifier.read_bytes()
    assert sha(original)==ORIGINAL and new!=original
    (a.output/'original-qualifier.py').write_bytes(original)
    (a.output/'revised-qualifier.py').write_bytes(new)
    old_ast,new_ast=ast.parse(original),ast.parse(new)
    def main_handler(tree):
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        region=main.body[-1]
        assert isinstance(region,ast.Try) and len(region.handlers)==1
        return region.handlers[0]
    old_handler,new_handler=main_handler(old_ast),main_handler(new_ast)
    old_dump,new_dump=ast.dump(old_handler,include_attributes=False),ast.dump(new_handler,include_attributes=False)
    assert old_dump!=new_dump
    old_handler.body=[ast.Pass()];new_handler.body=[ast.Pass()]
    assert ast.dump(old_ast,include_attributes=False)==ast.dump(new_ast,include_attributes=False)
    proof={'schema':'heptatorch.retained-qualifier.except-only-delta.v1','status':'PASS',
           'original_sha256':sha(original),'revised_sha256':sha(new),
           'only_main_final_except_handler_body_changed':True,
           'all_success_path_and_existing_helper_AST_identical':True,
           'old_handler_normalized_AST_sha256':sha(old_dump.encode()),
           'new_handler_normalized_AST_sha256':sha(new_dump.encode()),
           'original_successful_execution_not_relabelled':True}
    dump(a.output/'source-delta.json',proof)
    (a.output/'except-only.diff').write_text(''.join(difflib.unified_diff(original.decode().splitlines(True),
        new.decode().splitlines(True),fromfile='successful-original-137607',tofile='failure-custody-revision')))
    bad=a.output/'wrong-compiler-pin.fixture'
    bad.write_bytes(b'WRONG_COMPILER_PIN_CONTROL_ONLY\nNEVER_EXECUTE\n')
    child=a.output/'failed-invocation'
    command=[sys.executable,'-I','-S','-B',str(a.qualifier),'--repo',str(a.repo),
             '--parent-build',str(a.parent_build),'--compiler',str(bad),'--linker',str(a.linker),
             '--stack-build',str(a.stack_build),'--output',str(child)]
    dump(a.output/'command.json',{'command':command,'revised_qualifier_sha256':sha(new),
                                 'wrong_fixture_sha256':sha(bad.read_bytes())})
    with (a.output/'stdout').open('wb') as stdout,(a.output/'stderr').open('wb') as stderr:
        run=subprocess.run(command,cwd=a.repo,stdout=stdout,stderr=stderr,check=False)
    result=json.loads((child/'report.json').read_bytes())
    assert run.returncode==1 and result['status']=='FAIL'
    assert result['commands']==[] and result['completed_outer_commands']==0
    assert result['requested_producer_mode']=='fresh'
    assert result['producer_executed_this_invocation'] is False and result['producer_receipt_reused'] is False
    assert result['successful_suite_completion_claimed'] is False
    assert 'input SHA' in result['error'] and str(bad) in result['error']
    assert 'input SHA' in result['traceback'] and str(bad) in result['traceback']
    assert 'evidence_archive_error' not in result and 'final_failure_report_error' not in result
    evidence=result['failed_execution_evidence']
    assert evidence['scope']=='PARTIAL_FAILED_EXECUTION_BYTES_ONLY'
    assert evidence['successful_suite_completion_claimed'] is False
    manifest=json.loads((child/'evidence-manifest.json').read_bytes())
    assert manifest['status']=='COMPLETE_BYTES_VERIFIED'
    assert manifest['excluded_top_level_only']==['evidence-manifest.json','raw-evidence.tar.xz','report.json']
    members=manifest['files'];assert set(members)=={'executed-integration.py'}
    assert members['executed-integration.py']['sha256']==sha(new)
    seen={}
    with tarfile.open(child/'raw-evidence.tar.xz','r:xz') as tar:
        for m in tar:
            assert m.isfile() and m.name in members and m.name not in seen
            raw=tar.extractfile(m).read()
            value={'sha256':sha(raw),'bytes':len(raw),'mode':m.mode}
            assert value==members[m.name]
            assert raw==(child/m.name).read_bytes()
            seen[m.name]=value
    assert seen==members
    with lzma.open(child/'raw-evidence.tar.xz','rb') as f:
        while f.read(1024*1024):pass
    for key in ('archive','manifest'):
        path=Path(evidence[key]['path'])
        assert path.stat().st_size==evidence[key]['bytes'] and sha(path.read_bytes())==evidence[key]['sha256']
    stdout=(a.output/'stdout').read_bytes()
    final=json.loads(stdout.splitlines()[-1]);assert final['status']=='FAIL' and final['error']==result['error']
    assert a.qualifier.read_bytes()==new and a.original_qualifier.read_bytes()==original
    report={'schema':'heptatorch.retained-qualifier.failure-custody-control.v1',
            'status':'PASS_EXPECTED_PREPARATION_REJECTION','qualifier_exit_code':run.returncode,
            'qualifier_status':result['status'],'primary_error':result['error'],
            'qualifier_report_sha256':sha((child/'report.json').read_bytes()),
            'revised_qualifier_sha256':sha(new),'original_qualifier_sha256':sha(original),
            'source_delta_sha256':sha((a.output/'source-delta.json').read_bytes()),
            'native_executed':False,'compiler_executed':False,'Torch_executed':False,
            'completed_outer_commands':0,'only_wrong_compiler_pin_injected':True,
            'primary_error_and_traceback_preserved':True,'partial_evidence_self_verified':True,
            'partial_evidence':evidence,'never_claimed_complete_suite':True,
            'all_success_path_AST_unchanged':True,'successful_six_suites_not_repeated':True,
            'stdout_sha256':sha(stdout),'stderr_sha256':sha((a.output/'stderr').read_bytes()),
            'full_rewrite_complete':False,'overall_decision':'HOLD','full_schema_parity':0}
    dump(a.output/'report.json',report)
    print(json.dumps({'status':report['status'],'revised_qualifier_sha256':sha(new),
                      'report_sha256':sha((a.output/'report.json').read_bytes())}))


if __name__=='__main__':main()
