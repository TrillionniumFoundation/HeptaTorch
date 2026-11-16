"""Replay all 16 genuine pinned Torch partial-backward recipes on HIA.

Inputs and expected gradients/presence/errors come from a hash-bound hosted
report. Read-only Torch tracing is used only for its observed completed-node
and leaf counts; the hook-free run supplies numerical state. No native result
is used to synthesize reference values, and no Torch runtime is substituted.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import struct
import traceback

PROBE_SHA='2dc9a76462368f5bb227f02d5e401e10505e38676c9fe9b04460c1dae2d2c970'
OBSERVATIONS_SHA='61f9ad730d5ce2d02b737622bc05af683dd707ddbf041b492ad0987bbd3bb982'
CONTROLS_SHA='43fbb5a3abb62125191266f7eec9d020717ae9f6388c171edbda6957dda255a8'
OBSERVER_SHA='dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
CASE_NAMES=(
    'chain-upstream-version-error-default','chain-upstream-version-error-retained',
    'independent-good-newer-nonuniform-seed','independent-bad-newer',
    'independent-good-newer-retained-retry','independent-good-newer-preexisting-gradient',
    'shared-leaf-waits-for-failing-branch','shared-leaf-pending-and-independent-leaf-committed',
    'no-Saved-good-branch-repeats-before-version-error','shared-intermediate-diamond-and-independent-leaf',
    'gradient-shape-error-after-good-branch','gradient-shape-error-before-good-branch',
    'released-node-error-after-live-good-branch','shared-leaf-retained-failure-then-good-root',
    'mutated-output-Saved-error-after-good-branch','successful-shared-leaf-readiness-control')


def sha(raw): return hashlib.sha256(raw).hexdigest()


def status_class(call):
    if call['outcome']=='SUCCEEDED': return 0
    assert call['outcome']=='RAISED' and call['exception_type']=='RuntimeError'
    if 'Trying to backward through the graph a second time' in call['message']: return 10
    if 'has been modified by an inplace operation' in call['message']: return 6
    if 'returned an invalid gradient' in call['message']: return 5
    raise AssertionError(('unclassified actual Torch runtime error',call))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','controls','transport-observer','observations','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True);parser.add_argument('--expected-source-sha256',required=True)
    a=parser.parse_args()
    for name in ('binary','source','controls','transport_observer','observations','output'): setattr(a,name,getattr(a,name).resolve())
    paths={name:getattr(a,name) for name in ('binary','source','controls','transport_observer','observations')}
    hashes={name:sha(p.read_bytes()) for name,p in paths.items()}
    assert hashes=={'binary':a.expected_sha256,'source':a.expected_source_sha256,'controls':CONTROLS_SHA,
                    'transport_observer':OBSERVER_SHA,'observations':OBSERVATIONS_SHA}
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    b=load('independent_partial_codec',a.controls);transport=load('independent_partial_transport',a.transport_observer)
    reference=json.loads(a.observations.read_text())
    assert reference['status']=='OBSERVED' and reference['cases']==16 and reference['probe_errors']==0
    assert reference['probe_sha256']==PROBE_SHA and reference['target_execution'] is False
    assert reference['oracle_provenance']['status']=='PASS' and len(reference['oracle_provenance']['checks'])==11
    assert all(reference['oracle_provenance']['checks'].values())
    assert tuple(r['case'] for r in reference['rows'])==CASE_NAMES
    assert all(r['status']=='OBSERVED' and r['observation']['instrumentation_values_shapes_presence_and_saved_control_equal'] for r in reference['rows'])
    a.output.mkdir(parents=True,exist_ok=False);archive=a.output/'executed-sources';archive.mkdir()
    files=(Path(__file__),a.controls,a.transport_observer,a.observations)
    source_hashes={p.name:sha(p.read_bytes()) for p in files}
    for p in files: (archive/p.name).write_bytes(p.read_bytes())

    def metadata(c,tid,expected,check_version=True):
        row=c.one(13,tid,label='actual Torch operand/output metadata');m=b.words(row);rank=len(expected['shape'])
        assert len(m)==16 and row['value']==tid and m[0]==rank and m[1]==len(expected['values'])
        assert m[2]==int(expected['requires_grad']) and m[3:3+rank]==tuple(expected['shape']) and m[7:7+rank]==tuple(expected['strides'])
        if check_version: assert m[13]==expected['version']
        c.comparisons+=4+2*rank+int(check_version);c.values(tid,expected['values'])
        return m[13]

    def gradients(c,named,reference_gradients):
        out={}
        for name,expected in reference_gradients.items():
            row=c.one(59,named[name],label='strict None versus defined gradient presence')
            assert row['payload']==b'' and row['value']==int(expected is not None),(name,row,expected)
            c.comparisons+=1;out[name]={'native_present':row['value'],'torch_present':int(expected is not None)}
            if expected is not None:
                c.values(named[name],expected['values'],True)
                out[name]['compared_values']=expected['values']
        return out

    def build(c,index,trial):
        named={};opcodes={};before=trial['recipe']['named_values_before_mutation']
        def new(name):
            ref=before[name];tid=c.new(ref['shape'],ref['values'],ref['requires_grad']);named[name]=tid;opcodes['AccumulateGrad('+name+')']=0;return tid
        def put(name,code,*args):
            tid=c.make(code,*args);named[name]=tid
            opcodes[name]={2:args[0],3:args[0],4:args[0],36:39}[code]
            return tid
        def square(name,x): return put(name,2,3,x,x)
        def add(name,x,y): return put(name,2,1,x,y)
        def total(name,x): return put(name,4,7,x)
        if index not in (10,11): x=new('x')
        if index in (0,1):
            inner=square('inner',x);outer=square('outer',inner);total('loss',outer)
        elif index in (2,3,4,5,8):
            g=new('g')
            if index==5:
                two=c.new([], [2.]);prime=c.make(4,7,c.make(2,3,g,two));result=c.one(58,prime,0,0)
                assert result['value']==0
                gradients(c,named,trial['recipe']['setup'][0]['gradients_after'])
            if index==3:
                good=square('good',g);bad=square('bad',x)
            else:
                bad=square('bad',x);good=add('good',g,g) if index==8 else square('good',g)
            merged=add('merged',bad,good)
            if index!=2: total('loss',merged)
        elif index in (6,7,13):
            coefficient=new('coefficient');bad=put('bad',2,3,x,coefficient);good=square('good',x)
            if index==13: total('good_root',good)
            merged=add('merged',bad,good)
            if index==7:
                g=new('g');independent=square('independent',g);merged=add('all_branches',merged,independent)
            total('loss',merged)
        elif index==9:
            g=new('g');shared=square('shared',x);left=square('left',shared);right=add('right',shared,g)
            total('loss',add('merged',left,right))
        elif index in (10,11):
            x=new('self');source=new('source');g=new('g');config=b.Blob(struct.pack('<3Q4q',2,2,2,0,1,1,2))
            if index==11:
                good=square('good',g);bad=put('bad',36,x,source,1,config)
            else:
                bad=put('bad',36,x,source,1,config);good=square('good',g)
            total('loss',add('merged',bad,good))
        elif index==12:
            spent=square('spent',x);prior=c.make(4,7,spent);assert c.one(58,prior,0,0)['value']==0
            gradients(c,named,trial['recipe']['setup'][0]['gradients_after'])
            g=new('g');good=square('good',g);total('loss',add('merged',spent,good))
        elif index==14:
            bad=put('bad',3,5,x);g=new('g');good=square('good',g);total('loss',add('merged',bad,good))
        elif index==15:
            g=new('g');left=square('left',x);right=add('right',x,x);good=square('good',g)
            total('loss',add('all_branches',add('merged',left,right),good))
        else: raise AssertionError(index)
        assert set(named)==set(before),(set(named),set(before))
        for name,ref in before.items(): metadata(c,named[name],ref)
        mutation_versions=[]
        for mutation in trial['recipe']['mutations']:
            name=mutation['name'];target=named[name]
            if name=='bad' and index==14: target=c.make(19,target)
            after=trial['recipe']['named_values_after_mutation'][name]
            c.run([(12,target,i,b.bits(value)) for i,value in enumerate(after['values'])],label='scalar writes reproduce actual Torch vector mutation values')
            version=metadata(c,named[name],after,False)
            mutation_versions.append({'name':name,'native_version':version,'torch_version':after['version'],
                                      'mapping':'One Torch vector add_ is represented by existing native scalar writes; absolute counter counts differ, saved-staleness semantics are compared.'})
            if name=='bad' and index==14: c.one(41,target)
        return named,opcodes,mutation_versions

    rows=[]
    for index,case in enumerate(reference['rows']):
        directory=a.output/f'case-{index+1:02d}';session=transport.OwnerSession(a.binary,directory=directory,address_limit=64<<20)
        c=b.Check(session);row={'case':case['case'],'status':'FAIL'}
        try:
            o=case['observation'];plain=o['untraced'];traced=o['traced'];named,opcodes,versions=build(c,index,plain)
            initial=gradients(c,named,plain['initial']['gradients']);results=[]
            for observed_call,trace_call in zip(plain['calls'],traced['calls']):
                before_presence=gradients(c,named,observed_call['leaf_gradients_before'])
                seed_ref=observed_call['seed'];seed=0 if seed_ref is None else c.new(seed_ref['shape'],seed_ref['values'])
                expected_error=status_class(observed_call);failing=trace_call['entered_without_post']
                if expected_error: assert len(failing)==1
                else: assert failing==[]
                expected=(expected_error,len(trace_call['actual_finished_order']),
                          len(trace_call['actual_finished_leaf_accumulators']),opcodes[failing[0]] if failing else 0)
                output=named[observed_call['output_node']]
                record=c.one(58,output,seed,int(observed_call['retain_graph']),label='actual traced node counts and hook-free outcome')
                actual=b.words(record);assert record['value']==expected_error and actual==expected,(actual,expected,trace_call['actual_entered_order'])
                c.comparisons+=5
                after_presence=gradients(c,named,observed_call['leaf_gradients_after']);c.audit()
                results.append({'native_structured_result':list(actual),'expected_from_observed_Torch':list(expected),
                                'observed_Torch_entered_order':trace_call['actual_entered_order'],
                                'observed_Torch_finished_order':trace_call['actual_finished_order'],
                                'observed_Torch_failing_node':failing,'before_presence':before_presence,'after_presence':after_presence})
            session.finish();row.update(status='PASS',mutation_versions=versions,initial_presence=initial,calls=results)
        except Exception as error:
            row.update(exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            if session.poll() is None: os.kill(session.pid,signal.SIGKILL)
            try: session.finish(expected=-signal.SIGKILL)
            except Exception as cleanup: row['cleanup_exception']=repr(cleanup)
        row.update(frames=c.frames,exact_scalar_or_binary32_comparisons=c.comparisons)
        (directory/'report.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');rows.append(row)
    assert hashes=={name:sha(p.read_bytes()) for name,p in paths.items()}
    assert source_hashes=={p.name:sha(p.read_bytes()) for p in files}
    report={'schema':'heptatorch.native-replay-pinned-partial-backward.v1','status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
            'native_executable_sha256':hashes['binary'],'native_source_sha256':hashes['source'],
            'observations_sha256':OBSERVATIONS_SHA,'probe_sha256':PROBE_SHA,'test_source_hashes':source_hashes,
            'oracle_provenance':reference['oracle_provenance'],'cases':len(rows),'passed':sum(r['status']=='PASS' for r in rows),
            'frames':sum(len(r['frames']) for r in rows),'exact_scalar_or_binary32_comparisons':sum(r['exact_scalar_or_binary32_comparisons'] for r in rows),
            'gradient_presence_comparison':'Strict command59 == (actual Torch grad is not None), before and after every call; no zero-for-None substitution.',
            'schedule_comparison':'Native successful-node count, successful-leaf count, first failing opcode, runtime error class and partial gradient state are checked against actual Torch tracing. Native per-node hook events are not exposed and are not claimed measured.',
            'reference_execution':'The fixed report already ran on pinned Torch with tracing/untraced controls; this command executes only native HIA.',
            'pinned_Torch_executed_during_replay':False,'native_executed':True,'rtol':0.,'atol':0.,
            'limits':['Vector mutation adapters use several native scalar writes; absolute mutation version counts are explicitly recorded, not equated.',
                      'Native gradient-buffer Tensor object versions and user hooks are not part of this wire interface.',
                      'The matched scheduling scope is these CPU first-order built-in derivative recipes, with autograd multithreading disabled in the reference.'],
            'rows':rows,'full_rewrite_complete':False,'pytorch_api_denominator':2590}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','cases','passed','frames','exact_scalar_or_binary32_comparisons')}))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
