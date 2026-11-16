#!/usr/bin/env python3
"""Observe empty native_dropout Tensor identity versus captured autograd nodes.

This is an additive actual-Torch reference. It does not execute HIA, alter the
frozen28 reference, or predict acceptance, errors, lifetime or gradient outcomes.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import runpy
import struct
import sys
import traceback
import warnings
import weakref

SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
PARENT_SHA = 'bc5a490f1623a4078050829798f502b0be934722f371adbcbb5612aaeb1a563c'
SOURCE_REF = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
CHECKS = {'python_matches','platform_matches','version_backend_matches',
          'git_matches_oracle_build_pin','distribution_version_matches','direct_url_matches',
          'direct_url_archive_sha256_matches','pip_report_url_matches',
          'pip_report_archive_sha256_matches','pip_report_version_matches','pip_report_requested'}
PLAN = [
    {'name':'default-failure-releases-mask-before-accumulator-error', 'focus':'Saved release and retry/mutation precedence'},
    {'name':'preexisting-gradient-survives-same-T-history-change', 'focus':'The old actual gradient Tensor remains bound'},
    {'name':'two-empty-dropouts-capture-distinct-history-nodes', 'focus':'One Tensor, two immutable old edges, one original accumulator'},
    {'name':'older-independent-root-still-targets-original-accumulator', 'focus':'Changing current Tensor history does not rewrite earlier edges'},
    {'name':'false-at-forward-then-true-remains-a-leaf', 'focus':'No retroactive edge under an initially disabled requires flag'},
    {'name':'requires-flag-on-T-after-history-change', 'focus':'False and True requests, Tensor identity, Version and existing history'},
    {'name':'retain-prehook-materializes-before-later-backward-error', 'focus':'Actual G replacement across retained/default/released failures'},
    {'name':'weak-lifetimes-with-and-without-history-detach', 'focus':'Storage weak handles after final public references are removed'},
    {'name':'nonleaf-input-captures-its-previous-Mul-history', 'focus':'Same-T history replacement does not turn a valid old nonleaf edge into a self-edge'},
]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference','lock','repo','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    inputs = {Path(__file__).resolve():Path(__file__).read_bytes(),
              a.reference.resolve():a.reference.read_bytes(), a.lock.resolve():a.lock.read_bytes()}
    assert sha(inputs[a.reference.resolve()]) == SUPPORT_SHA
    assert sha(inputs[a.lock.resolve()]) == LOCK_SHA
    common = {'schema':'heptatorch.empty-native-dropout-identity-probe.v1',
              'probe_sha256':sha(inputs[Path(__file__).resolve()]),
              'parent_observation_sha256':PARENT_SHA, 'parent_reference_modified':False,
              'primary_source_ref':SOURCE_REF, 'expected_cases':len(PLAN), 'case_plan':PLAN,
              'native_executed':False, 'native_values_compared':0,
              'full_pytorch_rewrite_complete':False, 'pytorch_api_denominator':2590,
              'comparison_policy':'Observe actual outcomes only; no expected result, exception, identity or lifetime is substituted.'}
    if a.prepare_only:
        report = {**common, 'status':'PREPARED_NOT_RUN', 'torch_executed':False,
                  'cases':0, 'rows':[], 'inputs_sha256':{str(k):sha(v) for k,v in inputs.items()}}
        a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'status':report['status'],'expected_cases':len(PLAN),'probe_sha256':common['probe_sha256']}))
        return 0
    assert Path.cwd().resolve() == a.repo.resolve(), 'Run from the exact repository containing the locked pip-install and inventory records'
    old_argv = sys.argv
    try:
        sys.argv = [str(a.reference), str(a.repo/'reference-only-no-native-execution'),
                    str(a.output.with_suffix('.provenance.json')), str(a.lock)]
        context = runpy.run_path(str(a.reference), run_name='empty_dropout_identity_provenance_only')
    finally:
        sys.argv = old_argv
    t, provenance = context['torch'], context['PROVENANCE']
    assert provenance['status'] == 'PASS' and set(provenance['checks']) == CHECKS
    assert all(v is True for v in provenance['checks'].values())
    states, rows, errors = {}, [], []

    def state():
        raw = bytes(t.get_rng_state().tolist())
        assert len(raw) == 5056
        key = sha(raw)
        states.setdefault(key, {'bytes':len(raw),'hex':raw.hex()})
        return key

    def words(x):
        z = x.detach().contiguous().reshape(-1)
        if x.dtype == t.float32:
            return [int(v)&0xffffffff for v in z.view(t.int32).tolist()]
        return [int(v) for v in z.tolist()]

    def tensor(x):
        if x is None:
            return None
        return {'dtype':str(x.dtype), 'shape':list(x.shape), 'strides':list(x.stride()),
                'element_size':x.element_size(), 'storage_bytes':x.untyped_storage().nbytes(),
                'storage_offset_elements':x.storage_offset(), 'tensor_identity':str(x._cdata),
                'storage_identity':str(x.untyped_storage()._cdata), 'version':x._version,
                'requires_grad':x.requires_grad, 'is_leaf':x.is_leaf,
                'retains_grad':x.retains_grad,
                'grad_fn':type(x.grad_fn).__name__ if x.grad_fn is not None else None,
                'binary_words':words(x)}

    def capture(operation):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                value = operation()
                result = {'accepted':True,'outcome':'SUCCEEDED','error':None}
            except Exception as e:
                value = None
                result = {'accepted':False,'outcome':'RAISED',
                          'error':{'type':type(e).__name__,'message':str(e)}}
        result['warnings'] = [{'type':type(w.message).__name__,'message':str(w.message)} for w in caught]
        return value, result

    def gradient(x):
        value, event = capture(lambda:tensor(x.grad))
        return {'tensor':value, 'observation':event}

    def snapshot(x, mask=None):
        return {'input':tensor(x),'gradient':gradient(x),'mask':tensor(mask),'state':state()}

    def empty(requires=True):
        return t.empty((0,),dtype=t.float32).as_strided((2,0,3),(17,101,2),7).detach().requires_grad_(requires)

    def seed(x):
        return t.empty(tuple(x.shape),dtype=t.float32)

    def dropout(x):
        return t.ops.aten.native_dropout.default(x,.5,True)

    def graph(node, depth=0):
        if node is None:
            return None
        if depth > 8:
            return {'depth_limit':True,'type':type(node).__name__}
        result = {'type':type(node).__name__, 'sequence_nr':int(node._sequence_nr()),
                  'next':[{'output_nr':int(nr),'node':graph(n,depth+1)} for n,nr in node.next_functions]}
        if hasattr(node,'variable'):
            result['accumulator_variable'] = tensor(node.variable)
        return result

    def saved(node):
        value, event = capture(lambda:tensor(node._saved_result1))
        return {'tensor':value, 'observation':event}

    def backward(x, root, retain):
        _, event = capture(lambda:root.backward(seed(root),retain_graph=retain))
        return {'call':event,'after':snapshot(x)}

    def no_grad_add(x):
        with t.no_grad():
            x.add_(.5)

    def observe(index):
        if index == 0:
            x = empty(); before = snapshot(x); out,mask = dropout(x); node = out.grad_fn
            r = {'before':before,'after_forward':snapshot(x,mask),'graph':graph(node),
                 'output_is_input_object':out is x,'output_is_input_Tensor':out._cdata==x._cdata}
            r['first_default'] = backward(x,out,False)
            r['saved_after_first'] = saved(node)
            _, r['mask_mutation'] = capture(lambda:no_grad_add(mask))
            r['after_mask_mutation'] = snapshot(x,mask)
            r['retry_default'] = backward(x,out,False)
            r['saved_after_retry'] = saved(node)
            return r
        if index == 1:
            x = empty(); r = {'materialize_before_forward':backward(x,x,False)}; held = x.grad
            r['held_before'] = tensor(held); out,mask = dropout(x)
            r['after_forward'] = snapshot(x,mask)
            r['same_existing_G_after_forward'] = x.grad is held
            r['retained_backward'] = backward(x,out,True)
            r['held_after'] = tensor(held); r['same_existing_G_after_failure'] = x.grad is held
            r['saved_after_failure'] = saved(out.grad_fn)
            return r
        if index == 2:
            x = empty(); out1,mask1 = dropout(x); n1 = out1.grad_fn
            first_graph = graph(n1); out2,mask2 = dropout(x); n2 = out2.grad_fn
            r = {'first_graph':first_graph,'second_graph':graph(n2),
                 'all_outputs_same_T':x._cdata==out1._cdata==out2._cdata,
                 'different_history_nodes':n1 is not n2,
                 'second_next_is_first':len(n2.next_functions)==1 and n2.next_functions[0][0] is n1,
                 'mask1':tensor(mask1),'mask2':tensor(mask2),'after_forward':snapshot(x)}
            r['default_backward'] = backward(x,out2,False)
            r['saved_first_after'] = saved(n1); r['saved_second_after'] = saved(n2)
            r['retry'] = backward(x,out2,False)
            return r
        if index == 3:
            x = empty(); old = x * 2.; old_node = old.grad_fn
            r = {'old_graph_before':graph(old_node)}
            out,mask = dropout(x)
            r.update(old_graph_after=graph(old_node),new_graph=graph(out.grad_fn),after_forward=snapshot(x,mask))
            r['old_root_backward'] = backward(x,old,True)
            r['new_root_backward'] = backward(x,out,False)
            return r
        if index == 4:
            x = empty(False); before = snapshot(x); out,mask = dropout(x)
            r = {'before':before,'after_forward':snapshot(x,mask),'same_T':out._cdata==x._cdata}
            _, r['enable_requires'] = capture(lambda:x.requires_grad_(True))
            r['after_enable'] = snapshot(x,mask)
            r['direct_backward'] = backward(x,x,False)
            return r
        if index == 5:
            x = empty(); out,mask = dropout(x)
            r = {'after_forward':snapshot(x,mask)}
            _, r['disable_requires'] = capture(lambda:x.requires_grad_(False))
            r['after_disable_attempt'] = snapshot(x,mask)
            _, r['enable_requires'] = capture(lambda:x.requires_grad_(True))
            r['after_enable_attempt'] = snapshot(x,mask)
            r['backward'] = backward(x,out,True)
            return r
        if index == 6:
            x = empty(); out,mask = dropout(x)
            _, retain = capture(lambda:x.retain_grad())
            r = {'retain_call':retain,'before_backward':snapshot(x,mask)}
            r['retained_failure'] = backward(x,out,True); held = x.grad
            r['held_after_first'] = tensor(held)
            r['default_failure'] = backward(x,out,False)
            r['G_same_as_held_after_default'] = x.grad is held
            r['held_after_default'] = tensor(held)
            r['released_retry'] = backward(x,out,False)
            r['G_same_as_held_after_retry'] = x.grad is held
            r['held_after_retry'] = tensor(held)
            return r
        if index == 7:
            result = []
            for action in ('drop_without_backward','drop_after_default_failure','detach_history_then_drop',
                           'drop_with_older_child_then_drop_child','drop_with_optimizer_then_drop_optimizer'):
                def construct():
                    x = empty(); x.grad = t.empty(tuple(x.shape),dtype=t.float32)
                    extra_roots = {}
                    if action == 'drop_with_older_child_then_drop_child':
                        extra_roots['older_child'] = x * 2.
                    if action == 'drop_with_optimizer_then_drop_optimizer':
                        extra_roots['optimizer'] = t.optim.SGD([x],lr=.125)
                    out,mask = dropout(x)
                    observed = {'before_action':snapshot(x,mask)}
                    tensor_refs = {'input':weakref.ref(x),'gradient':weakref.ref(x.grad),'mask':weakref.ref(mask)}
                    handles = {name:value.untyped_storage()._weak_ref() for name,value in
                               (('input',x),('gradient',x.grad),('mask',mask))}
                    if action == 'drop_after_default_failure':
                        observed['action'] = backward(x,out,False)
                    elif action == 'detach_history_then_drop':
                        _, observed['action'] = capture(lambda:x.detach_())
                    else:
                        observed['action'] = {'accepted':True,'outcome':'NO_BACKWARD_REQUESTED','error':None}
                    observed['before_drop'] = snapshot(x,mask)
                    return observed, tensor_refs, handles, extra_roots
                observed, refs, handles, extra_roots = construct()
                gc.collect()
                observed['after_Tensor_locals_dropped'] = {
                    'explicit_remaining_roots':sorted(extra_roots),
                    'Python_wrappers_alive':{name:ref() is not None for name,ref in refs.items()},
                    'Storage_expired':{name:bool(t.UntypedStorage._expired(h)) for name,h in handles.items()}}
                extra_roots.clear()
                gc.collect()
                observed['after_remaining_roots_dropped'] = {
                    'Python_wrappers_alive':{name:ref() is not None for name,ref in refs.items()},
                    'Storage_expired':{name:bool(t.UntypedStorage._expired(h)) for name,h in handles.items()}}
                for handle in handles.values():
                    t.UntypedStorage._free_weak_ref(handle)
                result.append({'action_name':action, **observed})
            return {'variants':result,'lifetime_observer_keeps_only_weak_handles_and_primitives':True}
        if index == 8:
            x = empty(); q = x * 2.; previous = q.grad_fn
            r = {'leaf_before':snapshot(x),'nonleaf_before':snapshot(q),'previous_graph':graph(previous)}
            out,mask = dropout(q)
            r.update(leaf_after_forward=snapshot(x),nonleaf_after_forward=snapshot(q,mask),
                     same_input_T=out._cdata==q._cdata,leaf_is_different_T=x._cdata!=q._cdata,
                     next_is_previous=len(out.grad_fn.next_functions)==1 and out.grad_fn.next_functions[0][0] is previous,
                     current_graph=graph(out.grad_fn))
            r['backward'] = backward(x,out,False)
            r['nonleaf_after_backward'] = snapshot(q,mask)
            r['retry'] = backward(x,out,False)
            return r
        raise AssertionError(index)

    with t.random.fork_rng(devices=[]):
        for i, spec in enumerate(PLAN):
            try:
                t.manual_seed(1729)
                before = state()
                observed = observe(i)
                after = state()
                rows.append({'index':i, **spec, 'status':'OBSERVED', 'observation':observed,
                             'state_before':before,'state_after':after,'state_unchanged':before==after})
            except BaseException as e:
                error = {'index':i,'name':spec['name'],'error':repr(e),'traceback':traceback.format_exc()}
                errors.append(error)
                rows.append({**error,'status':'PROBE_ERROR'})
    assert all(path.read_bytes()==raw for path,raw in inputs.items())
    report = {**common, 'status':'OBSERVED' if not errors else 'PROBE_ERROR', 'torch_executed':True,
              'oracle_provenance':provenance,'source_files_unchanged':True,
              'cases':len(rows),'rows':rows,'probe_errors':errors,'probe_error_count':len(errors),
              'states':states,'unique_complete_states':len(states),
              'cpu_capability':t.backends.cpu.get_cpu_capability(),'threads':t.get_num_threads(),
              'inputs_sha256':{str(path):sha(raw) for path,raw in inputs.items()},
              'scope':'9 additive recipes. Weak-lifetime recipe has5 explicitly listed actions. Empty tensors carry no numeric element payload; no result is inferred from the native target.'}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'cases':len(rows),'probe_error_count':len(errors),
                      'output_bytes':a.output.stat().st_size,'output_sha256':sha(a.output.read_bytes())}))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
