"""Replay eight actual Torch grad-presence/optimizer observations on HIA.

Parameter versions and grad presence are strict. Adam-family values use the
pre-existing session-oracle tolerances, fixed here before native execution;
SGD, zero/absent buffers and gradients are bit exact. Explicit native optimizer
state Tensor lifetime differences remain visible, including momentum versions.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import struct
import traceback

OBSERVATIONS_SHA='b88b3f7ebeb480427f02039ae221ea19764b03eb35065fde10898bb810ad77dd'
PROBE_SHA='b368fd752485bd40a109712fffe156f6dfb43e48c4c1c971f9836e54bc861306'
CONTROLS_SHA='43fbb5a3abb62125191266f7eec9d020717ae9f6388c171edbda6957dda255a8'
OBSERVER_SHA='dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
ADAM_RTOL=4e-5
ADAM_ATOL=5e-6


def sha(raw): return hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','controls','transport-observer','observations','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-sha256',required=True);p.add_argument('--expected-source-sha256',required=True);a=p.parse_args()
    for name in ('binary','source','controls','transport_observer','observations','output'): setattr(a,name,getattr(a,name).resolve())
    paths={name:getattr(a,name) for name in ('binary','source','controls','transport_observer','observations')}
    hashes={name:sha(path.read_bytes()) for name,path in paths.items()}
    assert hashes=={'binary':a.expected_sha256,'source':a.expected_source_sha256,'controls':CONTROLS_SHA,
                    'transport_observer':OBSERVER_SHA,'observations':OBSERVATIONS_SHA}
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    b=load('presence_replay_codec',a.controls);transport=load('presence_replay_transport',a.transport_observer)
    ref=json.loads(a.observations.read_text());assert ref['status']=='OBSERVED' and ref['cases']==8 and ref['probe_errors']==0
    assert ref['probe_sha256']==PROBE_SHA and len(ref['oracle_provenance']['checks'])==11 and all(ref['oracle_provenance']['checks'].values())
    a.output.mkdir(parents=True,exist_ok=False);archive=a.output/'executed-sources';archive.mkdir()
    files=(Path(__file__),a.controls,a.transport_observer,a.observations);source_hashes={p.name:sha(p.read_bytes()) for p in files}
    for path in files: (archive/path.name).write_bytes(path.read_bytes())

    def numerical(c,tid,expected,approx=False,gradient=False):
        record=c.one(11 if gradient else 10,tid,label='actual Torch values')
        count=len(expected);assert record['value']==count and len(record['payload'])==count*4
        actual=struct.unpack('<'+'f'*count,record['payload']);actual_bits=struct.unpack('<'+'I'*count,record['payload'])
        wanted_bits=tuple(b.bits(v) for v in expected)
        if approx:
            for x,y in zip(actual,expected): assert math.isfinite(x) and math.isfinite(y) and abs(x-y)<=ADAM_ATOL+ADAM_RTOL*abs(y),(x,y)
            c.approximate_comparisons+=count
            c.approximate_bit_differences+=sum(x!=y for x,y in zip(actual_bits,wanted_bits))
        else:
            assert actual_bits==wanted_bits,(actual_bits,wanted_bits);c.comparisons+=count
        return list(actual)

    def metadata(c,tid,expected,strict_version=True):
        row=c.one(13,tid);m=b.words(row);rank=len(expected['shape'])
        assert row['value']==tid and len(m)==16 and m[0]==rank and m[1]==len(expected['values'])
        assert m[2]==int(expected['requires_grad']) and m[3:3+rank]==tuple(expected['shape']) and m[7:7+rank]==tuple(expected['strides'])
        if strict_version: assert m[13]==expected['version'],(m[13],expected['version'],tid)
        c.comparisons+=4+2*rank+int(strict_version);return m[13]

    def versioned_vector_mutation(c,param,values):
        # A transparent single-frame composition of existing native operations:
        # mutate Storage through a temporary .data alias, bump the real shared
        # Version once with an identical-value write, then destroy the alias.
        before_stats=b.words(c.one(54));meta=b.words(c.one(13,param));sid,vid=meta[11:13]
        before_s=b.words(c.one(51,sid));before_v=b.words(c.one(52,vid));reference=0xfe000001
        commands=[(55,param),(13,reference)]+[(12,reference,i,b.bits(value)) for i,value in enumerate(values)]+[(12,param,0,b.bits(values[0])),(41,reference)]
        rows=c.run(commands,label='one committed vector-value mutation with exactly one original Version bump')
        temporary=rows[0]['value'];alias_meta=b.words(rows[1]);alias_version=alias_meta[12]
        assert alias_meta[11]==sid and alias_version!=vid
        after_stats=b.words(c.one(54));after_s=b.words(c.one(51,sid));after_v=b.words(c.one(52,vid))
        assert before_stats[5:11]==after_stats[5:11] and before_stats[4]==after_stats[4]
        assert before_s==after_s
        assert after_v[:2]==before_v[:2] and after_v[2]==before_v[2]+1 and after_v[3:]==before_v[3:]
        assert after_stats[11]==before_stats[11]+1
        c.one(13,temporary,status=1,label='temporary data alias is destroyed')
        c.one(52,alias_version,status=1,label='temporary independent Version is destroyed')
        c.audit();c.comparisons+=12
        return {'composition':'cmd55 data_alias; scalar writes through its separate Version; one same-value cmd12 on original; cmd41 drops temporary alias in the same frame.',
                'original_StorageID':sid,'original_VersionID':vid,'temporary_TensorID':temporary,'temporary_VersionID':alias_version,
                'before_stats':list(before_stats),'after_stats':list(after_stats),
                'live_T_S_V_Weak_Saved_Hold_counts_unchanged':True,'free_bytes_restored':True,'Storage_record_unchanged':True,
                'original_Version_counter_incremented_exactly_once':True,'temporary_Tensor_and_Version_invalid_after_drop':True,
                'one_consumed_internal_node_sequence_is_dead_and_not_in_backward_graph':True,
                'not_implemented_by_this_adapter':'Tensor.add_ API or general vector in-place operator'}

    def execute_case(c,index,o):
        kind=o.get('optimizer');actions=o['actions'];first=actions[0]['before'];params={};states={};steps={}
        for name,t in first['parameters'].items():
            if index==7 and name=='nonleaf': continue
            values=[1.,2.,3.] if 3<=index<=5 and name=='bad' else t['values']
            params[name]=c.new(t['shape'],values,t['requires_grad']);steps[name]=0
        if index==7: params['nonleaf']=c.make(2,3,params['x'],params['constant'])
        if kind:
            for name,param in params.items():
                t=first['parameters'][name];states[name]={'momentum_buffer' if kind=='SGD-momentum' else 'exp_avg':c.new(t['shape'],[0.]*len(t['values']))}
                if kind!='SGD-momentum': states[name]['exp_avg_sq']=c.new(t['shape'],[0.]*len(t['values']))
        loss=None;mutation=None
        if 3<=index<=5:
            bad=c.make(2,3,params['bad'],params['bad']);good=c.make(2,3,params['good'],params['good']);loss=c.make(4,7,c.make(2,1,bad,good))
            mutation=versioned_vector_mutation(c,params['bad'],first['parameters']['bad']['values'])
        if index==6: loss=c.make(4,7,params['p'])
        if index==7: loss=c.make(4,7,params['nonleaf'])
        state_gaps=[];state_presence_gaps=[];snapshots=[]

        def check_snapshot(reference,label):
            result={'stage':label,'parameters':{},'optimizer_state':{}}
            for name,t in reference['parameters'].items():
                tid=params[name];version=metadata(c,tid,t)
                grad=reference['gradients'][name];presence=c.one(59,tid)
                assert presence['value']==int(grad is not None) and presence['payload']==b'';c.comparisons+=1
                approximate=bool(kind in ('Adam','AdamW') and steps[name]>0 and grad is not None)
                values=numerical(c,tid,t['values'],approximate)
                if grad is not None: numerical(c,tid,grad['values'],gradient=True)
                result['parameters'][name]={'values':values,'version':version,'gradient_present':presence['value']}
                if not kind: continue
                state=reference['optimizer_state'][name];actual_states={}
                if not state['entry_exists']:
                    state_presence_gaps.append({'stage':label,'parameter':name,'torch_state_entry_exists':False,
                        'native_state':'Caller supplied matching-shape Tensor objects already exist; this is not a native Optimizer object.'})
                for field,handle in states[name].items():
                    sm=b.words(c.one(13,handle));version=sm[13]
                    if state['entry_exists']:
                        expected=state['fields'][field]['tensor']
                        metadata(c,handle,expected,strict_version=kind!='SGD-momentum')
                        values=numerical(c,handle,expected['values'],kind!='SGD-momentum')
                        if version!=expected['version']:
                            assert kind=='SGD-momentum'
                            state_gaps.append({'stage':label,'parameter':name,'field':field,'native_version':version,
                                'torch_version':expected['version'],'status':'OPEN',
                                'reason':'Explicit pre-existing native momentum Tensor is updated; Torch creates its momentum buffer by clone on the first step.'})
                    else:
                        assert version==0;c.comparisons+=1
                        values=numerical(c,handle,[0.]*len(t['values']))
                    actual_states[field]={'TensorID':handle,'values':values,'version':version}
                if state['entry_exists'] and 'step' in state['fields']:
                    expected_step=state['fields']['step']['tensor']['values'];assert expected_step==[float(steps[name])];c.comparisons+=1
                    actual_states['step_representation']={'native':'explicit command integer supplied by test client','value':steps[name],
                        'torch':state['fields']['step'],'object_lifetime_equal':False}
                result['optimizer_state'][name]={'torch_entry_exists':state['entry_exists'],'native_explicit_state':actual_states}
            return result

        def optimize(reference_before):
            config=o['config'];commands=[]
            for name,param in params.items():
                present=reference_before['gradients'][name] is not None
                step=steps[name]+1
                if kind=='SGD-momentum':
                    fields=[b.bits(config[k]) for k in ('lr','momentum','dampening','weight_decay')]+[int(config['nesterov'])]
                    commands.append((25,param,states[name]['momentum_buffer'],0,step,b.Blob(struct.pack('<5Q',*fields))))
                else:
                    fields=[b.bits(config['lr']),b.bits(config['betas'][0]),b.bits(config['betas'][1]),b.bits(config['eps']),b.bits(config['weight_decay']),int(kind=='AdamW')]
                    commands.append((24,param,states[name]['exp_avg'],states[name]['exp_avg_sq'],step,b.Blob(struct.pack('<6Q',*fields))))
                if present: steps[name]=step
            rows=c.run(commands,label='request optimizer for every parameter, including absent-gradient ones')
            assert all(r['value']==1 and r['payload']==b'' for r in rows)

        for action_index,action in enumerate(actions):
            before_check=check_snapshot(action['before'],str(action_index)+' before')
            label=action['action']
            if index<=2:
                if action_index in (0,2,4,6): optimize(action['before'])
                elif action_index in (1,5): c.one(8,params['p'])
                else:
                    zero=c.new([], [0.]);loss=c.make(4,7,c.make(2,3,params['p'],zero));assert c.one(58,loss,0,0)['value']==0
            elif 3<=index<=5:
                if action_index==0:
                    r=c.one(58,loss,0,0);assert r['value']==6 and b.words(r)==(6,4,1,3)
                    assert action['outcome']=='RAISED' and 'has been modified by an inplace operation' in action['message'];c.comparisons+=5
                elif action_index in (1,3): optimize(action['before'])
                else: c.run([(8,t) for t in params.values()])
            elif index==6:
                if action_index in (0,3): c.one(8,params['p'])
                elif action_index==1: assert c.one(58,loss,0,0)['value']==0
                else: optimize(action['before'])
            else: assert c.one(58,loss,0,0)['value']==0
            if not (3<=index<=5 and action_index==0): assert action['outcome']=='SUCCEEDED'
            after_check=check_snapshot(action['after'],str(action_index)+' after')
            snapshots.append({'action':label,'before':before_check,'after':after_check})
        c.audit()
        return {'snapshots':snapshots,'low_level_single_version_mutation':mutation,
                'state_version_differences':state_gaps,'state_creation_differences':state_presence_gaps,
                'step_counter_scope':'Existing protocol takes an explicit caller-supplied step integer; no native Optimizer/step Tensor object is claimed.'}

    rows=[]
    for index,case in enumerate(ref['rows']):
        directory=a.output/f'case-{index+1:02d}';session=transport.OwnerSession(a.binary,directory=directory,address_limit=64<<20)
        c=b.Check(session);c.approximate_comparisons=0;c.approximate_bit_differences=0;row={'case':case['case'],'status':'FAIL'}
        try:
            details=execute_case(c,index,case['observation']);session.finish()
            row.update(status='QUALIFIED_WITH_OPEN_STATE_LIFECYCLE' if case['observation'].get('optimizer') else 'PASS',details=details,
                       parameter_values_versions_gradient_presence_and_skip_passed=True)
        except Exception as error:
            row.update(exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            if session.poll() is None: os.kill(session.pid,signal.SIGKILL)
            try: session.finish(expected=-signal.SIGKILL)
            except Exception as cleanup: row['cleanup_exception']=repr(cleanup)
        row.update(frames=c.frames,exact_scalar_or_binary32_comparisons=c.comparisons,
                   adam_tolerance_comparisons=c.approximate_comparisons,adam_compared_values_with_different_bits=c.approximate_bit_differences,
                   entire_owner_rollback_bytes=c.rollback_bytes)
        (directory/'report.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');rows.append(row)
    assert hashes=={name:sha(path.read_bytes()) for name,path in paths.items()}
    assert source_hashes=={path.name:sha(path.read_bytes()) for path in files}
    errors=sum(r['status']=='FAIL' for r in rows)
    report={'schema':'heptatorch.native-replay-pinned-gradient-presence.v1','status':'FAIL' if errors else 'QUALIFIED_WITH_OPEN_STATE_LIFECYCLE',
        'native_executable_sha256':hashes['binary'],'native_source_sha256':hashes['source'],'observations_sha256':OBSERVATIONS_SHA,
        'probe_sha256':PROBE_SHA,'test_source_hashes':source_hashes,'oracle_provenance':ref['oracle_provenance'],
        'cases':len(rows),'parameter_values_versions_gradient_presence_and_skip_passed':sum(r.get('parameter_values_versions_gradient_presence_and_skip_passed',False) for r in rows),
        'unexpected_failures':errors,'frames':sum(len(r['frames']) for r in rows),
        'exact_scalar_or_binary32_comparisons':sum(r['exact_scalar_or_binary32_comparisons'] for r in rows),
        'adam_tolerance_comparisons':sum(r['adam_tolerance_comparisons'] for r in rows),
        'adam_compared_values_with_different_bits':sum(r['adam_compared_values_with_different_bits'] for r in rows),
        'state_version_difference_observations':sum(len(r.get('details',{}).get('state_version_differences',[])) for r in rows),
        'state_creation_difference_observations':sum(len(r.get('details',{}).get('state_creation_differences',[])) for r in rows),
        'numeric_policy':{'SGD_gradients_absent_parameters_and_zero_buffers':'binary32 bits exact','parameter_Version_and_presence':'integer exact',
                          'Adam_family_updated_parameter_and_state_values':{'rtol':ADAM_RTOL,'atol':ADAM_ATOL,'source':'Unchanged existing session oracle default policy, fixed before execution.'}},
        'native_executed':True,'pinned_Torch_executed_during_replay':False,
        'compatibility_limits':['Native explicit optimizer-state Tensor lifetime and scalar step input remain distinct from Torch Optimizer-owned state.',
                                'Momentum first-step state Version differences are recorded, not reset or hidden; these observations are not merged with the earlier separate 283-state counter gap.',
                                'Partial-case vector mutation is an explicitly disclosed single-frame data_alias/write/bump/drop construction; it is not a native Tensor.add_ implementation.',
                                'No grad=None clearing operation, nonleaf retain_grad or .grad Tensor-object version API is qualified.'],
        'rows':rows,'full_rewrite_complete':False,'pytorch_api_denominator':2590}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','cases','parameter_values_versions_gradient_presence_and_skip_passed','unexpected_failures','frames','exact_scalar_or_binary32_comparisons','adam_tolerance_comparisons','adam_compared_values_with_different_bits','state_version_difference_observations','state_creation_difference_observations')}))
    return 1 if errors else 0


if __name__=='__main__': raise SystemExit(main())
