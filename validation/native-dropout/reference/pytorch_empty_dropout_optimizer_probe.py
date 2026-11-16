#!/usr/bin/env python3
"""Observe existing Optimizer membership after empty native_dropout changes T history.

Additive fixed-Torch reference only. No HIA execution, predicted outcome,
numeric expectation, exception substitution, or modification of the prior9.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import traceback
import warnings

SUPPORT_SHA = 'ef8fe89aa7dbff4ee03f9d5762d2736a3648d9e48e97ad5df7bec615676f04ae'
LOCK_SHA = 'ca2e54ce65083bbe0ab3d47b6878971271765ab35a9dd70725fa345fceed3944'
PARENT_PROBE_SHA = '973421e3f6a239bad6f46a076b68666108ba976b3d1c7e0a4064a7d3ebaa33b6'
PARENT_OBSERVATION_SHA = 'de7db2f985873be797bd7e167fb374f82b5690021e0a0541d2fb4f7a36b7ad83'
SOURCE_REF = '31a77fc9c0a7c6fff15a1ff58c743f53b1a9d457'
CHECKS = {'python_matches', 'platform_matches', 'version_backend_matches',
          'git_matches_oracle_build_pin', 'distribution_version_matches', 'direct_url_matches',
          'direct_url_archive_sha256_matches', 'pip_report_url_matches',
          'pip_report_archive_sha256_matches', 'pip_report_version_matches', 'pip_report_requested'}
PLAN = [{'name':algorithm.lower()+'-existing-member-'+presence,
         'algorithm':algorithm, 'initial_gradient':presence,
         'operations':['zero_grad_false', 'step_with_current_gradient',
                       'zero_grad_true', 'step_after_clear']}
        for algorithm in ('SGD', 'Adam', 'AdamW') for presence in ('none', 'defined-empty')]
CONFIG = {'lr':0.125, 'momentum':0.5, 'dampening':0.0, 'weight_decay':0.25,
          'nesterov':False, 'betas':[0.5,0.75], 'eps':0.01,
          'maximize':False, 'foreach':False, 'fused':False}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'lock', 'repo', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    assert not a.output.exists(), 'The reference output is create-only'
    inputs = {Path(__file__).resolve():Path(__file__).read_bytes(),
              a.reference.resolve():a.reference.read_bytes(), a.lock.resolve():a.lock.read_bytes()}
    assert sha(inputs[a.reference.resolve()]) == SUPPORT_SHA
    assert sha(inputs[a.lock.resolve()]) == LOCK_SHA
    common = {'schema':'heptatorch.empty-native-dropout-existing-optimizer-probe.v1',
              'probe_sha256':sha(inputs[Path(__file__).resolve()]),
              'parent_probe_sha256':PARENT_PROBE_SHA,
              'parent_observation_sha256':PARENT_OBSERVATION_SHA,
              'parent_reference_modified':False, 'primary_source_ref':SOURCE_REF,
              'expected_cases':len(PLAN), 'case_plan':PLAN, 'configuration':CONFIG,
              'native_executed':False, 'native_values_compared':0,
              'full_pytorch_rewrite_complete':False, 'pytorch_api_denominator':2590,
              'comparison_policy':'Actual outcomes only; no acceptance, exception, identity, Version, state or RNG result is substituted.'}
    if a.prepare_only:
        report = {**common, 'status':'PREPARED_NOT_RUN', 'torch_executed':False,
                  'cases':0, 'rows':[], 'inputs_sha256':{str(k):sha(v) for k,v in inputs.items()}}
        a.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'status':report['status'], 'expected_cases':len(PLAN),
                          'probe_sha256':common['probe_sha256']}))
        return 0
    assert Path.cwd().resolve() == a.repo.resolve(), 'Use the repository holding the locked pip-install and inventory records'
    old_argv = sys.argv
    try:
        sys.argv = [str(a.reference), str(a.repo/'reference-only-no-native-execution'),
                    str(a.output.with_suffix('.provenance.json')), str(a.lock)]
        context = runpy.run_path(str(a.reference), run_name='empty_dropout_optimizer_provenance_only')
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
        states.setdefault(key, {'bytes':len(raw), 'hex':raw.hex()})
        return key

    def words(x):
        z = x.detach().contiguous().reshape(-1)
        if x.dtype == t.float32:
            return [int(v)&0xffffffff for v in z.view(t.int32).tolist()]
        return [int(v) for v in z.tolist()]

    def tensor(x):
        if x is None:
            return None
        node = x.grad_fn
        return {'dtype':str(x.dtype), 'shape':list(x.shape), 'strides':list(x.stride()),
                'element_size':x.element_size(), 'storage_bytes':x.untyped_storage().nbytes(),
                'storage_offset_elements':x.storage_offset(), 'tensor_identity':str(x._cdata),
                'storage_identity':str(x.untyped_storage()._cdata), 'version':x._version,
                'requires_grad':x.requires_grad, 'is_leaf':x.is_leaf, 'retains_grad':x.retains_grad,
                'grad_fn':type(node).__name__ if node is not None else None,
                'grad_fn_sequence':int(node._sequence_nr()) if node is not None else None,
                'binary_words':words(x)}

    def capture(operation):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                value = operation()
                event = {'accepted':True, 'outcome':'SUCCEEDED', 'error':None}
            except Exception as error:
                value = None
                event = {'accepted':False, 'outcome':'RAISED',
                         'error':{'type':type(error).__name__, 'message':str(error)}}
        event['warnings'] = [{'type':type(w.message).__name__, 'message':str(w.message)} for w in caught]
        return value, event

    def graph(node, depth=0):
        if node is None:
            return None
        if depth > 8:
            return {'depth_limit':True, 'type':type(node).__name__}
        result = {'type':type(node).__name__, 'sequence_nr':int(node._sequence_nr()),
                  'next':[{'output_nr':int(nr), 'node':graph(n,depth+1)} for n,nr in node.next_functions]}
        if hasattr(node, 'variable'):
            result['accumulator_variable'] = tensor(node.variable)
        return result

    def optimizer_info(opt, x):
        # .get and iteration do not materialize an absent lazy state entry.
        entries = []
        for parameter, entry in opt.state.items():
            values = {}
            for name, value in entry.items():
                if isinstance(value, t.Tensor):
                    values[name] = {'kind':'Tensor', 'tensor':tensor(value)}
                else:
                    values[name] = {'kind':type(value).__name__, 'value':value}
            entries.append({'parameter_identity':str(parameter._cdata),
                            'parameter_is_input':parameter is x, 'values':values})
        groups = []
        for group in opt.param_groups:
            metadata = {key:value for key,value in group.items() if key != 'params'}
            groups.append({'metadata':metadata,
                           'parameters':[{'tensor_identity':str(value._cdata),
                                          'is_input_object':value is x} for value in group['params']]})
        return {'algorithm':type(opt).__name__, 'parameter_has_state_entry':x in opt.state,
                'state_entry_count':len(opt.state), 'groups':groups, 'state_entries':entries}

    def snapshot(x, opt, mask=None, history=None, held=None):
        gradient, gradient_event = capture(lambda:x.grad)
        saved, saved_event = (None, None)
        if history is not None:
            saved, saved_event = capture(lambda:tensor(history._saved_result1))
        return {'input':tensor(x), 'gradient':tensor(gradient), 'gradient_observation':gradient_event,
                'held_initial_gradient':tensor(held),
                'gradient_is_held':gradient is held if held is not None else None,
                'mask':tensor(mask), 'current_graph':graph(x.grad_fn),
                'current_history_is_saved_handle':x.grad_fn is history if history is not None else None,
                'saved_mask':saved, 'saved_mask_observation':saved_event,
                'optimizer':optimizer_info(opt,x), 'state':state()}

    def construct_optimizer(algorithm,x):
        if algorithm == 'SGD':
            return t.optim.SGD([x], lr=CONFIG['lr'], momentum=CONFIG['momentum'],
                               dampening=CONFIG['dampening'], weight_decay=CONFIG['weight_decay'],
                               nesterov=CONFIG['nesterov'], maximize=False, foreach=False, fused=False)
        cls = t.optim.Adam if algorithm == 'Adam' else t.optim.AdamW
        return cls([x], lr=CONFIG['lr'], betas=tuple(CONFIG['betas']), eps=CONFIG['eps'],
                   weight_decay=CONFIG['weight_decay'], maximize=False, foreach=False, fused=False)

    def observe(spec):
        x = t.empty((0,),dtype=t.float32).as_strided((2,0,3),(17,101,2),7).detach().requires_grad_(True)
        opt = construct_optimizer(spec['algorithm'],x)
        observed = {'before_gradient_materialization':snapshot(x,opt)}
        seed = None
        if spec['initial_gradient'] == 'defined-empty':
            seed = t.empty(tuple(x.shape),dtype=t.float32)
            _, observed['materialize_gradient'] = capture(lambda:x.backward(seed))
        else:
            observed['materialize_gradient'] = {'outcome':'NOT_REQUESTED'}
        held = x.grad
        observed['initial_seed'] = tensor(seed)
        observed['before_forward'] = snapshot(x,opt,held=held)
        pair, observed['native_dropout_call'] = capture(lambda:t.ops.aten.native_dropout.default(x,.5,True))
        if pair is None:
            observed['after_failed_forward'] = snapshot(x,opt,held=held)
            return observed
        out,mask = pair
        history = out.grad_fn
        observed['output_is_input_object'] = out is x
        observed['output_is_input_Tensor'] = out._cdata == x._cdata
        observed['after_forward'] = snapshot(x,opt,mask,history,held)
        operations = [('zero_grad_false',lambda:opt.zero_grad(set_to_none=False)),
                      ('step_with_current_gradient',lambda:opt.step()),
                      ('zero_grad_true',lambda:opt.zero_grad(set_to_none=True)),
                      ('step_after_clear',lambda:opt.step())]
        observed['events'] = []
        for name, operation in operations:
            before = snapshot(x,opt,mask,history,held)
            _, event = capture(operation)
            after = snapshot(x,opt,mask,history,held)
            observed['events'].append({'name':name, 'before':before, 'operation':event, 'after':after})
        observed['strong_observer_roots'] = ['input/output same T', 'optimizer', 'mask', 'history',
                                              'explicit held initial G when defined', 'initial seed when defined']
        observed['weak_lifetime_claim'] = False
        return observed

    with t.random.fork_rng(devices=[]):
        for index,spec in enumerate(PLAN):
            try:
                t.manual_seed(1729)
                before = state()
                observation = observe(spec)
                after = state()
                rows.append({'index':index, **spec, 'status':'OBSERVED', 'observation':observation,
                             'state_before':before, 'state_after':after, 'state_unchanged':before==after})
            except BaseException as error:
                record = {'index':index, 'name':spec['name'], 'error':repr(error),
                          'traceback':traceback.format_exc()}
                errors.append(record)
                rows.append({**record, 'status':'PROBE_ERROR'})
    assert all(path.read_bytes()==raw for path,raw in inputs.items())
    report = {**common, 'status':'OBSERVED' if not errors else 'PROBE_ERROR', 'torch_executed':True,
              'oracle_provenance':provenance, 'source_files_unchanged':True,
              'cases':len(rows), 'rows':rows, 'probe_errors':errors, 'probe_error_count':len(errors),
              'states':states, 'unique_complete_states':len(states),
              'cpu_capability':t.backends.cpu.get_cpu_capability(), 'threads':t.get_num_threads(),
              'inputs_sha256':{str(path):sha(raw) for path,raw in inputs.items()},
              'scope':'6 additive actual-Torch recipes; no native execution and no weak-lifetime claim. Existing members, true lazy state Tensors and four ordered operations are observed without asserting their outcomes.'}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'], 'cases':len(rows), 'probe_error_count':len(errors),
                      'output_bytes':a.output.stat().st_size, 'output_sha256':sha(a.output.read_bytes())}))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
