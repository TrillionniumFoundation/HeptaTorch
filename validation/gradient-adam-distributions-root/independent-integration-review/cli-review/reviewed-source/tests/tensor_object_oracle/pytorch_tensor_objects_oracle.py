"""Real pinned PyTorch oracle for native owned numeric Tensor objects v3.

The complete original 28-case numerical suite is executed from hash-checked,
unchanged sources through a reversible transport adapter. Seven additional
object-lifetime cases are reported separately. PyTorch is only the reference;
every native result is obtained from a live HIA process before stdin EOF.
"""
from pathlib import Path
import argparse
import json
import os
import runpy
import sys
import time

import bridge
import lifecycle


def emit(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key:report[key] for key in ('schema','status','cases','frames','compared_values')
                      if key in report}, allow_nan=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--owned-reference', type=Path, required=True)
    parser.add_argument('--session-reference', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--expected-native-sha256', default=bridge.FINAL_NATIVE_SHA)
    parser.add_argument('--trace-directory', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args=parser.parse_args()
    for name in ('executable','report','owned_reference','session_reference','reference','lock','repo'):
        setattr(args,name,getattr(args,name).resolve())
    if args.trace_directory is not None:
        args.trace_directory=args.trace_directory.resolve()
    started=time.monotonic()
    sources={
        'frozen_28_oracle':args.owned_reference,
        'frozen_21_session':args.session_reference,
        'frozen_reference_support':args.reference,
        'oracle_lock':args.lock,
        'v3_entry':Path(__file__).resolve(),
        'v3_bridge':Path(bridge.__file__).resolve(),
        'v3_lifecycle':Path(lifecycle.__file__).resolve(),
    }
    custody={name:{'filename':path.name,'sha256':bridge.sha(path.read_bytes())}
             for name,path in sources.items()}
    native_before=bridge.sha(args.executable.read_bytes())
    assert native_before==args.expected_native_sha256, (native_before,args.expected_native_sha256)
    scope,definitions,registrations,owned_program,_,audit=bridge.prepare_sources(
        args.owned_reference,args.session_reference,args.reference)
    original_plan=audit['inherited_overlay_audit']['original_case_plan']+[
        {'case':name,'rtol':rtol,'atol':atol,'source':'unchanged frozen owned-v2 extension'}
        for name,_,rtol,atol in scope['EXTRA_CASES']]
    lifetime_plan=[{'case':name,'rtol':rtol,'atol':atol,'source':'new v3 object lifetime differential'}
                   for name,_,rtol,atol in lifecycle.CASES]
    plan=original_plan+lifetime_plan
    assert len(original_plan)==28 and len(lifetime_plan)==7 and len({p['case'] for p in plan})==35
    base={
        'schema':'heptatorch.live-pytorch-owned-tensor-objects-v3.v1',
        'status':'NOT_RUN','native_executable_sha256':native_before,
        'source_custody':custody,'overlay_audit':audit,'case_plan':plan,
        'expected_cases':35,'inherited_frozen_numeric_cases':28,'additional_lifetime_cases':7,
        'full_rewrite_complete':False,'pytorch_api_denominator':2590,
        'scope':'Real owned CPU binary32 rank 0..4 numeric Tensor objects, independent Storage and Version identities, '
                'typed public handles and actual multi-frame native execution. This is not complete PyTorch storage or autograd compatibility.',
        'handle_contract': 'The inherited suite uses unchanged logical client ledger ordinals. Only actual native returned handles '
                           'or checked earlier-record result references enter Tensor ID fields. All numeric bytes are preserved.',
        'known_open_contracts':[
            'Default native backward retains graph/Saved data until public root drop or graph_reset; Torch default releases Saved data.',
            'Explicit native optimizer state and torch.optim-created state retain different construction lifecycles; observed counters are preserved.',
            'Public Saved unpack restores non-grad logical layout and Storage/Version sharing, not full SavedVariable autograd metadata.',
            'The former v1 and v2 profile claims and unresolved gaps are not relabeled by this v3 run.',
        ],
    }
    if args.prepare_only:
        emit(args.report,dict(base,reason='Frozen source custody and reversible AST checks passed; no Torch or native test case executed.',
                              cases=0,frames=0,compared_values=0,source_overlay_checks='PASS'))
        return 0
    args.report.parent.mkdir(parents=True,exist_ok=True)
    provenance_path=args.report.with_name(args.report.stem+'-provenance.json')
    previous_argv,previous_directory=sys.argv,Path.cwd()
    try:
        os.chdir(args.repo)
        sys.argv=[str(args.reference),str(args.executable),str(provenance_path),str(args.lock)]
        support=runpy.run_path(str(args.reference),run_name='heptatorch_tensor_objects_reference_support')
    except SystemExit as error:
        observed=json.loads(provenance_path.read_text()) if provenance_path.exists() else {'error':str(error)}
        emit(args.report,dict(base,reason='Unchanged pinned reference support declined execution; no native case was run.',
                              reference_initialization=observed,cases=0,frames=0,compared_values=0))
        return 2
    finally:
        sys.argv=previous_argv
        os.chdir(previous_directory)
    provenance=support['PROVENANCE']
    assert provenance['status']=='PASS' and len(provenance['checks'])==11
    assert all(value is True for value in provenance['checks'].values())
    scope.update({name:support[name] for name in ('torch','F','bits','f32','PROVENANCE')})
    scope['REPORT']=args.report
    bridge.install(scope,definitions,owned_program,support['Program'],args.executable,args.trace_directory)
    exec(compile(registrations,str(args.session_reference),'exec'),scope)
    assert len(scope['ROWS'])==21
    extra_functions={
        'descriptor_pages':scope['descriptor_pages'],
        'large_chunked':scope['large_chunked'],
        'prefix_SGD':lambda session:scope['prefix_optimizer'](session,'SGD'),
        'prefix_Adam':lambda session:scope['prefix_optimizer'](session,'Adam'),
        'prefix_AdamW':lambda session:scope['prefix_optimizer'](session,'AdamW'),
        'latezero_matmul':scope['latezero_matmul'],
        'latezero_gather':scope['latezero_gather'],
    }
    for name,function,rtol,atol in scope['EXTRA_CASES']:
        scope['case'](name,extra_functions[function],rtol,atol)
    assert len(scope['ROWS'])==28
    additional_functions=lifecycle.build(scope)
    for name,function,rtol,atol in lifecycle.CASES:
        scope['case'](name,additional_functions[function],rtol,atol)
    rows=scope['ROWS']
    assert [(r['case'],r['rtol'],r['atol']) for r in rows]==[(p['case'],p['rtol'],p['atol']) for p in plan]
    unchanged={name:bridge.sha(path.read_bytes())==custody[name]['sha256'] for name,path in sources.items()}
    native_unchanged=native_before==bridge.sha(args.executable.read_bytes())

    def group(group_rows):
        return {'cases':len(group_rows),'passed':sum(r['status']=='PASS' for r in group_rows),
                'failed':sum(r['status']=='FAIL' for r in group_rows),
                'frames':sum(r['frames'] for r in group_rows),
                'compared_values':sum(r['compared_values'] for r in group_rows)}

    passed=len(rows)==35 and all(r['status']=='PASS' for r in rows) and native_unchanged and all(unchanged.values())
    report=dict(base,status='PASS' if passed else 'FAIL',oracle_provenance=provenance,
                cases=len(rows),failed_cases=sum(r['status']=='FAIL' for r in rows),
                frames=sum(r['frames'] for r in rows),compared_values=sum(r['compared_values'] for r in rows),
                groups={'unchanged_numeric_28':group(rows[:28]),'additional_object_lifetime_7':group(rows[28:])},
                direct_parameter_absolute_version_mismatches=sum(len(r.get('parameter_version_mismatches',[])) for r in rows),
                explicit_state_absolute_version_differences=sum(len(r.get('explicit_state_version_differences',[])) for r in rows),
                observed_open_policy_differences=[r['known_policy_difference'] for r in rows if 'known_policy_difference' in r],
                source_files_unchanged_during_run=unchanged,native_unchanged_during_run=native_unchanged,
                rows=rows,wall_seconds=time.monotonic()-started)
    emit(args.report,report)
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
