"""Replay cases 21--36 of the genuine pinned Torch backward-release report.

This executes only the native HIA target. Reference outcomes, gradient values,
storage expiry observations and Saved-property observations come from the
hash-bound prior Torch run. Command 57 deliberately rolls back a failed frame;
the one observed partial-release difference is reported as OPEN, not PASS.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import os
import signal
import struct
import traceback

from native_release_controls import Blob, Check, OBSERVER_SHA, bits, sha, words


PROBE_SHA='1192de6e175407756b01028278446067b22b70fe2d820f0045f6f5d623d46aa3'
CONTROLS_SHA='43fbb5a3abb62125191266f7eec9d020717ae9f6388c171edbda6957dda255a8'
CASE_NAMES=(
    'retain-true-true-false-true-sequence',
    'implicit-default-releases-saved',
    'no-saved-input-mutation-and-repeat',
    'saved-mutation-rejection-before-release',
    'failed-upstream-version-check-after-downstream-release',
    'saved-freed-marker-with-public-data-still-alive',
    'leaf-storage-held-by-AccumulateGrad-after-default-backward',
    'nongrad-coefficient-released-by-default-backward',
    'saved-output-weak-retain-then-default',
    'external-unpacked-alias-keeps-storage-but-not-saved-validity',
    'shared-subgraph-second-root-sees-freed-saved',
    'unvisited-independent-node-keeps-its-saved',
    'seed-shape-error-before-engine-preserves-saved',
    'backward-gradient-shape-error-preserves-failing-node-saved',
    'empty-index-select-still-has-saved-tensor-lifetime',
    'leaf-backward-twice-without-grad-function',
)
# Fixed before native execution. These two existing tanh examples exercise
# lifecycle, not a new bit-exact transcendental guarantee. All dyadic fixtures
# and all compared integer metadata retain exact equality.
TANH_RTOL=4e-5
TANH_ATOL=5e-6


def reference_status(call):
    if call['outcome']=='SUCCEEDED': return 0
    assert call['outcome']=='RAISED' and call['exception_type']=='RuntimeError',call
    message=call['message']
    if 'Trying to backward through the graph a second time' in message: return 10
    if 'has been modified by an inplace operation' in message: return 6
    if 'Mismatch in shape: grad_output' in message: return 2
    if 'returned an invalid gradient' in message: return 5
    raise AssertionError(('unclassified actual Torch exception',call))


def compare_values(c,handle,reference,gradient=False,approx=False):
    if not approx:
        c.values(handle,reference['values'],gradient)
        return
    row=c.one(11 if gradient else 10,handle,label='predeclared tanh tolerance')
    expected=reference['values']
    assert row['value']==len(expected) and len(row['payload'])==4*len(expected)
    actual=struct.unpack('<'+'f'*len(expected),row['payload'])
    for left,right in zip(actual,expected):
        assert math.isfinite(left) and math.isfinite(right)
        assert abs(left-right)<=TANH_ATOL+TANH_RTOL*abs(right),(left,right)
    c.approximate_comparisons+=len(expected)


def compare_gradients(c,leaves,call,approx=False):
    assert len(leaves)==len(call['gradients_after_call'])
    for (handle,shape,requires),expected in zip(leaves,call['gradients_after_call']):
        if expected is None:
            # The existing native protocol exposes allocated zero leaf-gradient
            # buffers before accumulation; Torch exposes None. Record this API
            # distinction explicitly, never claim object-presence equality.
            if requires:
                count=math.prod(shape)
                c.values(handle,[0.]*count,True)
                c.none_gradient_observations.append({'TensorID':handle,
                    'torch_gradient':None,'native_gradient_representation':'allocated zero buffer',
                    'elements':count})
        else:
            assert requires and expected['shape']==shape
            compare_values(c,handle,expected,True,approx)


def invoke(c,output,leaves,call,seed=None,approx=False,native_override=None):
    wanted=reference_status(call)
    if native_override is None: native_override=wanted
    if call['seed'] is None:
        assert seed is None
        seed_id=0
    elif seed is None:
        seed_id=c.new(call['seed']['shape'],call['seed']['values'])
    else:
        seed_id=seed
    retain=call['retain_graph_argument']
    assert retain in (True,False,'OMITTED')
    c.one(57,output,seed_id,int(retain is True),status=native_override,
          label='actual Torch outcome '+str(wanted)+'; command57 outcome '+str(native_override))
    compare_gradients(c,leaves,call,approx)
    c.call_comparisons.append({'torch_status_class':wanted,'native_status':native_override,
                               'outcome_equal':wanted==native_override,'retain_graph_argument':retain})


def leaf(c,values=(1.,2.,3.),shape=(3,)):
    return c.new(list(shape),list(values),True)


def square_loss(c,x):
    square=c.make(2,3,x,x)
    return square,c.make(4,7,square)


def mutate(c,x,values):
    c.run([(12,x,i,bits(value)) for i,value in enumerate(values)],label='explicit scalar writes matching Torch mutation values')


def retain_sequence(c,o):
    x=leaf(c,(1.,-2.,3.));y=c.make(2,3,x,x)
    seed=c.new([3],o['steps'][0]['seed']['values'])
    for row in o['steps']: invoke(c,y,[(x,[3],True)],row,seed)


def implicit_default(c,o):
    x=leaf(c,(1.,-2.,3.));_,loss=square_loss(c,x)
    for key in ('first','second'): invoke(c,loss,[(x,[3],True)],o[key])


def no_saved_mutation(c,o):
    x=leaf(c,(1.,-2.,3.));two=c.new([], [2.]);y=c.make(2,1,x,two);loss=c.make(4,7,y)
    invoke(c,loss,[(x,[3],True)],o['first'])
    mutate(c,x,o['input_after_mutation']['values'])
    compare_values(c,x,o['input_after_mutation'])
    invoke(c,loss,[(x,[3],True)],o['second'])
    return {'mutation_mapping':'One Torch add_ maps to three existing native scalar writes; values are identical, mutation counter increments are not asserted equal.'}


def saved_mutation(c,o):
    x=leaf(c,(1.,-2.,3.));b=c.new([3],[2.,3.,4.]);y=c.make(2,3,x,b);loss=c.make(4,7,y)
    mutate(c,b,[3.,4.,5.])
    for key in ('first','second'): invoke(c,loss,[(x,[3],True),(b,[3],False)],o[key])
    return {'mutation_mapping':'Three native scalar writes reproduce one Torch add_ value mutation; only stale-Saved rejection and unchanged gradient values are compared.'}


def partial_failure(c,o):
    x=leaf(c);middle=c.make(2,3,x,x);top=c.make(2,3,middle,middle);loss=c.make(4,7,top)
    mutate(c,x,[2.,3.,4.])
    stats_before=list(words(c.one(54)))
    invoke(c,loss,[(x,[3],True)],o['call'])
    stats_after=list(words(c.one(54)));assert stats_before==stats_after
    assert reference_status(o['retry'])==10
    invoke(c,loss,[(x,[3],True)],o['retry'],native_override=6)
    stats_retry=list(words(c.one(54)));assert stats_before==stats_retry
    return {'compatibility_status':'OPEN',
        'difference':'Torch released the successfully applied downstream node before an upstream version error; command57 rolls back its whole frame. The Torch retry reports freed Saved while native retry reports stale version.',
        'torch_statuses':[reference_status(o['call']),reference_status(o['retry'])],
        'native_statuses':[6,6],
        'torch_downstream_saved_after_failure':o['downstream_after_failure'],
        'torch_upstream_saved_after_failure':o['upstream_after_failure'],
        'native_stats_before_after_retry':[stats_before,stats_after,stats_retry],
        'next_implementation':'A separate explicit backward operation with structured runtime exceptions and per-node commit is required; this report does not claim it exists.'}


def payload_alive(c,o):
    x=leaf(c);y=c.make(3,5,x);seed=c.new([3],o['first']['seed']['values'])
    invoke(c,y,[(x,[3],True)],o['first'],seed)
    compare_values(c,y,o['public_output_after'])
    invoke(c,y,[(x,[3],True)],o['second'],seed)


def leaf_lifetime(c,o):
    x=leaf(c);one=c.new([], [1.]);middle=c.make(2,1,x,one);loss=c.make(4,7,middle);weak=c.make(42,x)
    c.run([(41,x),(41,middle)])
    c.expiry(weak,o['expired_after_public_drop'])
    invoke(c,loss,[],o['call'])
    c.expiry(weak,o['expired_after_default_backward'])
    c.one(41,loss);c.expiry(weak,o['expired_after_root_drop']);c.one(44,weak)


def coefficient_lifetime(c,o):
    x=leaf(c);b=c.new([3],[2.,3.,4.]);middle=c.make(2,3,x,b);loss=c.make(4,7,middle);weak=c.make(42,b)
    c.run([(41,b),(41,middle)]);c.expiry(weak,o['expired_after_public_drop'])
    invoke(c,loss,[(x,[3],True)],o['call']);c.expiry(weak,o['expired_after_default_backward']);c.one(44,weak)


def saved_output_lifetime(c,o):
    x=leaf(c,(-1.,0.,1.));y=c.make(3,15,x);square,loss=square_loss(c,y);weak=c.make(42,y)
    c.run([(41,y),(41,square)]);c.expiry(weak,o['expired_after_public_drop'])
    invoke(c,loss,[(x,[3],True)],o['retained'],approx=True);c.expiry(weak,o['expired_after_retained'])
    invoke(c,loss,[(x,[3],True)],o['default'],approx=True);c.expiry(weak,o['expired_after_default']);c.one(44,weak)


def unpacked_alias(c,o):
    x=leaf(c,(-.5,.25,1.));y=c.make(3,15,x);square,loss=square_loss(c,y)
    # The public detached alias has the same S/V as the successfully unpacked
    # Torch Saved tensor, but no artificial native AD-Saved introspection API.
    external=c.make(19,y);weak=c.make(42,y)
    c.run([(41,y),(41,square)])
    invoke(c,loss,[(x,[3],True)],o['call'],approx=True)
    c.expiry(weak,o['expired_with_external_alias']);compare_values(c,external,o['external_values'],approx=True)
    c.one(41,external);c.expiry(weak,o['expired_after_external_alias_drop']);c.one(44,weak)
    invoke(c,loss,[(x,[3],True)],o['retry'],approx=True)


def shared_roots(c,o):
    x=leaf(c);y,left=square_loss(c,x);one=c.new([], [1.]);right=c.make(4,7,c.make(2,1,y,one))
    invoke(c,left,[(x,[3],True)],o['left']);invoke(c,right,[(x,[3],True)],o['right'])
    compare_values(c,y,o['public_intermediate'])


def independent_nodes(c,o):
    x=leaf(c);a=c.make(2,3,x,x);b=c.make(2,3,x,x);left=c.make(4,7,a);right=c.make(4,7,b)
    invoke(c,left,[(x,[3],True)],o['first']);invoke(c,right,[(x,[3],True)],o['second'])


def invalid_seed(c,o):
    x=leaf(c);y=c.make(2,3,x,x)
    for key in ('invalid_seed_call','valid_seed_call'): invoke(c,y,[(x,[3],True)],o[key])


def invalid_gradient_shape(c,o):
    x=c.new([2,3],list(range(6)),True);b=c.new([2,3],[1.]*6,True)
    y=c.make(36,x,b,1,Blob(struct.pack('<3Q4q',2,2,2,0,1,1,2)))
    seed=c.new([2,3],o['first']['seed']['values'])
    for key in ('first','second'): invoke(c,y,[(x,[2,3],True),(b,[2,3],True)],o[key],seed)


def empty_index(c,o):
    x=c.new([2,3],[1.,-2.,3.,4.,-.5,2.],True);y=c.make(30,x,1,0,Blob(b''));seed=c.new([2,0],[])
    meta=words(c.one(13,y));assert meta[0]==2 and meta[1]==0 and meta[3:5]==(2,0) and meta[7:9]==tuple(o['output']['strides'])
    c.comparisons+=6
    for key in ('first','second'): invoke(c,y,[(x,[2,3],True)],o[key],seed)


def leaf_repeat(c,o):
    assert o['grad_fn_is_none'] is True
    x=leaf(c,(2.,),())
    for key in ('first','second'): invoke(c,x,[(x,[],True)],o[key])


FUNCTIONS=(retain_sequence,implicit_default,no_saved_mutation,saved_mutation,partial_failure,
           payload_alive,leaf_lifetime,coefficient_lifetime,saved_output_lifetime,unpacked_alias,
           shared_roots,independent_nodes,invalid_seed,invalid_gradient_shape,empty_index,leaf_repeat)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--expected-source-sha256',required=True)
    parser.add_argument('--observations',type=Path,required=True)
    parser.add_argument('--expected-observations-sha256',required=True)
    parser.add_argument('--transport-observer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    for name in ('binary','source','observations','transport_observer','output'):
        setattr(args,name,getattr(args,name).resolve())
    before={key:sha(getattr(args,key).read_bytes()) for key in ('binary','source','observations','transport_observer')}
    assert before=={'binary':args.expected_sha256,'source':args.expected_source_sha256,
                    'observations':args.expected_observations_sha256,'transport_observer':OBSERVER_SHA}
    controls=Path(__file__).with_name('native_release_controls.py');assert sha(controls.read_bytes())==CONTROLS_SHA
    reference=json.loads(args.observations.read_text())
    assert reference['status']=='OBSERVED' and reference['cases']==36 and reference['probe_errors']==0
    assert reference['probe_sha256']==PROBE_SHA and reference['target_execution'] is False
    assert reference['oracle_provenance']['status']=='PASS'
    checks=reference['oracle_provenance']['checks'];assert len(checks)==11 and all(v is True for v in checks.values())
    selected=reference['rows'][20:]
    assert tuple(r['case'] for r in selected)==CASE_NAMES and all(r['status']=='OBSERVED' for r in selected)
    spec=importlib.util.spec_from_file_location('backward_lifetime_replay_observer',args.transport_observer)
    transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
    args.output.mkdir(parents=True,exist_ok=False)
    files=(Path(__file__),controls,args.transport_observer,args.observations)
    source_hashes={p.name:sha(p.read_bytes()) for p in files};archive=args.output/'executed-sources';archive.mkdir()
    for p in files: (archive/p.name).write_bytes(p.read_bytes())
    rows=[]
    for index,(fn,case) in enumerate(zip(FUNCTIONS,selected),21):
        directory=args.output/f'case-{index:02d}';session=transport.OwnerSession(args.binary,directory=directory,address_limit=64<<20)
        c=Check(session);c.approximate_comparisons=0;c.none_gradient_observations=[];c.call_comparisons=[]
        row={'case':case['case'],'status':'FAIL'}
        try:
            details=fn(c,case['observation']);c.audit();session.finish()
            open_difference=bool(details and details.get('compatibility_status')=='OPEN')
            row.update(status='DECLARED_DIFFERENCE' if open_difference else 'PASS',native_contract_verified=True,details=details)
        except Exception as error:
            row.update(exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            if session.poll() is None: os.kill(session.pid,signal.SIGKILL)
            try: session.finish(expected=-signal.SIGKILL)
            except Exception as cleanup: row['cleanup_exception']=repr(cleanup)
        row.update(frames=c.frames,call_comparisons=c.call_comparisons,
                   exact_scalar_or_binary32_comparisons=c.comparisons,approximate_binary32_comparisons=c.approximate_comparisons,
                   none_gradient_representation_difference=c.none_gradient_observations,
                   entire_committed_owner_rollback_bytes=c.rollback_bytes)
        (directory/'report.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');rows.append(row)
    assert source_hashes=={p.name:sha(p.read_bytes()) for p in files}
    assert before=={key:sha(getattr(args,key).read_bytes()) for key in before}
    errors=sum(r['status']=='FAIL' for r in rows);differences=sum(r['status']=='DECLARED_DIFFERENCE' for r in rows)
    report={'schema':'heptatorch.native-replay-pinned-backward-lifetimes.v1',
        'status':'FAIL' if errors else 'QUALIFIED_WITH_OPEN_DIFFERENCE' if differences else 'PASS',
        'native_executable_sha256':before['binary'],'native_source_sha256':before['source'],
        'observations_sha256':before['observations'],'probe_sha256':PROBE_SHA,
        'oracle_provenance':reference['oracle_provenance'],'source_hashes':source_hashes,
        'reference_scope':'Last 16 cases of the actual 36-case pinned Torch report; command57 partial-error rollback remains explicitly different.',
        'reference_was_previously_executed':True,'pinned_Torch_executed_during_replay':False,'native_executed':True,
        'numeric_tolerances':{'dyadic_fixtures':{'rtol':0.,'atol':0.},'two_tanh_lifecycle_fixtures':{'rtol':TANH_RTOL,'atol':TANH_ATOL}},
        'comparison_limits':['No native grad-None presence API: allocated zeros are checked and the distinction is recorded.',
                             'A single Torch vector add_ maps to several existing native scalar writes; absolute mutation counters are not equated for that adapter.',
                             'Native AD Saved getters are not public; release is observed through repeat-backward behavior, weak lifetime and resource counts.'],
        'cases':len(rows),'pytorch_behavior_matched_cases':sum(r['status']=='PASS' for r in rows),
        'declared_difference_cases':differences,'native_contract_verified_cases':sum(r.get('native_contract_verified',False) for r in rows),
        'unexpected_failures':errors,'frames':sum(len(r['frames']) for r in rows),
        'exact_scalar_or_binary32_comparisons':sum(r['exact_scalar_or_binary32_comparisons'] for r in rows),
        'approximate_binary32_comparisons':sum(r['approximate_binary32_comparisons'] for r in rows),
        'entire_committed_owner_rollback_bytes':sum(r['entire_committed_owner_rollback_bytes'] for r in rows),
        'rows':rows,'full_rewrite_complete':False,'pytorch_api_denominator':2590}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','cases','pytorch_behavior_matched_cases','declared_difference_cases','unexpected_failures','frames','exact_scalar_or_binary32_comparisons','approximate_binary32_comparisons','entire_committed_owner_rollback_bytes')}))
    return 1 if errors else 0


if __name__=='__main__': raise SystemExit(main())
