"""Independent command58 structured-error controls and one genuine replay.

Case 1 replays actual pinned Torch case25 from the frozen 36-case report. The
other cases are distinct wire/transaction contract controls. This file does not
import the implementation, authoring helpers or an alternative numeric runtime.
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

CONTROLS_SHA='43fbb5a3abb62125191266f7eec9d020717ae9f6388c171edbda6957dda255a8'
OBSERVER_SHA='dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'
OBSERVATIONS_SHA='83bc3a03a51c5e2eb9499fb24ad73b9ab73da7452eb8ff4dc5b57e18a3958c7d'
CASE_NAMES=('actual-Torch-case25-now-reports-version-then-freed',
            'structured-runtime-error-commits-prefix-and-skips-suffix',
            'admission-and-later-command-errors-retain-frame-rollback',
            'same-frame-result-refs-and-resumption-after-runtime-error')


def sha(raw): return hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','source','controls','transport-observer','observations','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-sha256',required=True);p.add_argument('--expected-source-sha256',required=True)
    a=p.parse_args()
    for name in ('binary','source','controls','transport_observer','observations','output'): setattr(a,name,getattr(a,name).resolve())
    files={name:getattr(a,name) for name in ('binary','source','controls','transport_observer','observations')}
    hashes={name:sha(path.read_bytes()) for name,path in files.items()}
    assert hashes=={'binary':a.expected_sha256,'source':a.expected_source_sha256,'controls':CONTROLS_SHA,
                    'transport_observer':OBSERVER_SHA,'observations':OBSERVATIONS_SHA}
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    b=load('partial_control_codec',a.controls);transport=load('partial_control_transport',a.transport_observer)
    reference=json.loads(a.observations.read_text());assert reference['cases']==36 and reference['status']=='OBSERVED'
    observed=reference['rows'][24];assert observed['case']=='failed-upstream-version-check-after-downstream-release'
    o=observed['observation'];assert 'has been modified by an inplace operation' in o['call']['message']
    assert 'Trying to backward through the graph a second time' in o['retry']['message']
    a.output.mkdir(parents=True,exist_ok=False);archive=a.output/'executed-sources';archive.mkdir()
    for path in (Path(__file__),a.controls,a.transport_observer,a.observations): (archive/path.name).write_bytes(path.read_bytes())

    def back(c,root,seed,retain,wanted):
        row=c.one(58,root,seed,retain,label='structured backward runtime result')
        actual=b.words(row);assert row['value']==wanted[0] and actual==wanted,(row,actual,wanted)
        c.comparisons+=5;return {'value':row['value'],'payload':list(actual)}

    def chain(c):
        x=c.new([3],[1.,2.,3.],True);inner=c.make(2,3,x,x);outer=c.make(2,3,inner,inner);loss=c.make(4,7,outer)
        return x,inner,outer,loss

    def actual_case(c):
        x,inner,outer,loss=chain(c);c.run([(12,x,i,b.bits(v)) for i,v in enumerate([2.,3.,4.])])
        before=c.session.owner();first=back(c,loss,0,0,(6,2,0,3));after=c.session.owner()
        assert before!=after,'runtime exception did not commit any state'
        assert o['call']['gradients_after_call']==[None];c.values(x,[0.,0.,0.],True)
        second=back(c,loss,0,1,(10,1,0,3));c.values(x,[0.,0.,0.],True)
        c.one(57,loss,0,0,status=10,label='legacy atomic entry observes newly spent Saved')
        c.values(inner,[1.,4.,9.]);c.values(outer,[1.,16.,81.])
        return {'reference_case':observed['case'],'reference_status_classes':[6,10],
                'native_results':[first,second],'committed_owner_changed_after_runtime_exception':True,
                'pytorch_leaf_gradients':o['retry']['gradients_after_call'],
                'gradient_presence_limit':'Torch grad remains None; existing native API exposes an allocated zero buffer. Values and release/error behavior are compared, not grad object presence.'}

    def prefix_suffix(c):
        x,_,_,loss=chain(c);before=c.session.owner()
        commands=[(12,x,0,b.bits(9.)),(58,loss,0,0),(999,)]
        request=b.encode(commands);response=c.session.exchange(request);status,records=b.decode(response)
        assert status==0 and [r['code'] for r in records]==[12,58] and len(records)==2
        assert records[1]['value']==6 and b.words(records[1])==(6,2,0,3)
        c.frames.append({'frame':c.session.frames,'label':'structured runtime error terminates suffix and commits prior scalar mutation',
                         'expected_status':0,'actual_status':status,'request_sha256':sha(request),'response_sha256':sha(response),
                         'requested_commands':3,'returned_records':2})
        c.comparisons+=7;after=c.session.owner();assert before!=after
        c.values(x,[9.,2.,3.]);c.values(x,[0.,0.,0.],True)
        back(c,loss,0,1,(10,1,0,3))
        fresh=c.new([], [3.],True);back(c,fresh,0,0,(0,1,1,0));c.values(fresh,[1.],True)
        return {'successful_prefix_visible':True,'unknown_suffix_command_not_executed':True,'next_frame_executes_normally':True}

    def rollback(c):
        x,_,_,loss=chain(c);bad=c.new([1],[1.]);storage=b.words(c.one(13,x))[11]
        for command,status in [((58,loss,bad,0),2),((58,loss,0,2),7),((58,storage,0,0),1)]:
            c.run([(12,x,0,b.bits(9.)),command],status,'admission failure preserves full original owner')
        c.run([(58,loss,0,0),(999,)],4,'successful backward followed by a normal command error rolls back the full frame')
        c.values(x,[1.,2.,3.]);c.values(x,[0.,0.,0.],True)
        back(c,loss,0,1,(0,4,1,0));c.values(x,[4.,32.,108.],True)
        back(c,loss,0,0,(0,4,1,0));c.values(x,[8.,64.,216.],True)
        back(c,loss,0,1,(10,1,0,3))
        return {'ordinary_frame_errors_still_rollback':True,'structured_runtime_errors_are_distinct_results':True}

    def refs(c):
        ref=lambda index:0xfe000000|index
        commands=[b.tensor([2],[1.,2.],True),(2,3,ref(1),ref(1)),(4,7,ref(2)),b.tensor([], [1.]),(58,ref(3),ref(4),0),(11,ref(1))]
        rows=c.run(commands,label='new output and seed refs resolve before command58')
        assert rows[4]['value']==0 and b.words(rows[4])==(0,3,1,0)
        assert rows[5]['payload']==struct.pack('<2f',2.,4.);c.comparisons+=7
        x=rows[0]['value'];loss=rows[2]['value'];back(c,loss,0,0,(10,1,0,3))
        c.run([b.tensor([], [2.],True),(58,ref(1),0,0),(18,ref(2))],1,'structured success result is not a Tensor')
        c.values(x,[2.,4.],True)
        y=c.new([2],[2.,3.],True);z=c.make(2,1,y,y);seed=c.new([2],[.5,-1.])
        back(c,z,seed,0,(0,2,1,0));c.values(y,[1.,-2.],True)
        back(c,z,seed,0,(0,2,1,0));c.values(y,[2.,-4.],True)

    functions=(actual_case,prefix_suffix,rollback,refs);rows=[]
    for i,(name,fn) in enumerate(zip(CASE_NAMES,functions),1):
        directory=a.output/f'case-{i:02d}';session=transport.OwnerSession(a.binary,directory=directory,address_limit=64<<20)
        c=b.Check(session);row={'case':name,'status':'FAIL','reference_kind':'actual pinned Torch observation replay' if i==1 else 'independent native protocol control'}
        try:
            detail=fn(c);c.audit();session.finish();row.update(status='PASS',detail=detail)
        except Exception as error:
            row.update(exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            if session.poll() is None: os.kill(session.pid,signal.SIGKILL)
            try: session.finish(expected=-signal.SIGKILL)
            except Exception as error: row['cleanup_exception']=repr(error)
        row.update(frames=c.frames,exact_scalar_or_binary32_comparisons=c.comparisons,entire_owner_rollback_bytes=c.rollback_bytes)
        (directory/'report.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
    assert hashes=={name:sha(path.read_bytes()) for name,path in files.items()}
    report={'schema':'heptatorch.native-partial-backward-independent-controls.v1','status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
            'source_hashes':hashes,'test_sha256':sha(Path(__file__).read_bytes()),'cases':len(rows),'passed':sum(r['status']=='PASS' for r in rows),
            'case_count_split':{'actual_Torch_observation_replayed':1,'independent_native_protocol_controls':3},
            'native_executed':True,'pinned_Torch_executed_here':False,'frames':sum(len(r['frames']) for r in rows),
            'exact_scalar_or_binary32_comparisons':sum(r['exact_scalar_or_binary32_comparisons'] for r in rows),
            'entire_owner_rollback_bytes':sum(r['entire_owner_rollback_bytes'] for r in rows),'rows':rows,
            'full_rewrite_complete':False,'pytorch_api_denominator':2590}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','cases','passed','frames','exact_scalar_or_binary32_comparisons','entire_owner_rollback_bytes')}))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
