"""Replay the 20 operator cases from a genuine pinned Torch release report.

Inputs, seeds, output metadata, gradients and backward outcomes are read from
the hash-bound hosted report. Operator composition and index/configuration
fixtures are independently specified here. There is no local Torch fallback.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import signal
import struct
import traceback

from native_release_controls import Blob, Check, OBSERVER_SHA, bits, sha, words


PROBE_SHA='1192de6e175407756b01028278446067b22b70fe2d820f0045f6f5d623d46aa3'
KINDS=('add-tensor','clone','view','transpose','cat','stack','mul-shared-input',
       'mul-left-only','mul-empty','relu','index-select','gather','max-pool','avg-pool',
       'scatter-source-grad','scatter-self-only','index-add-source-grad','index-add-self-only',
       'conv2d','matmul-fold-copy')


def meta_and_values(c,handle,reference,label):
    record=c.one(13,handle,label=label+' metadata');observed=words(record)
    assert len(observed)==16 and record['value']==handle
    rank=len(reference['shape'])
    assert observed[0]==rank and observed[2]==int(reference['requires_grad'])
    assert observed[3:3+rank]==tuple(reference['shape'])
    assert observed[7:7+rank]==tuple(reference['strides'])
    assert observed[13]==reference['version']
    assert observed[1]==len(reference['values'])
    c.comparisons+=4+2*rank
    c.values(handle,reference['values'])


def build(c,kind,observation):
    leaves=[]
    for i,reference in enumerate(observation['leaves']):
        if kind=='matmul-fold-copy' and i==0:
            original=c.new([2,3,4],list(range(24)))
            handle=c.make(6,original,0,1)
        else:
            handle=c.new(reference['shape'],reference['values'],reference['requires_grad'])
        meta_and_values(c,handle,reference,'original operand '+str(i));leaves.append(handle)
    x=leaves[0]
    if kind=='add-tensor': output=c.make(2,1,x,leaves[1])
    elif kind=='clone': output=c.make(18,x)
    elif kind=='view': output=c.make(17,x,2,3,2)
    elif kind=='transpose': output=c.make(6,x,0,1)
    elif kind in ('cat','stack'):
        output=c.make(35 if kind=='stack' else 34,0 if kind=='stack' else 1,
                      Blob(struct.pack('<3Q',2,x,x)))
    elif kind in ('mul-shared-input','mul-empty'): output=c.make(2,3,x,x)
    elif kind=='mul-left-only': output=c.make(2,3,x,leaves[1])
    elif kind=='relu': output=c.make(3,5,x)
    elif kind=='index-select': output=c.make(30,x,1,3,Blob(struct.pack('<3q',2,0,2)))
    elif kind=='gather': output=c.make(31,x,1,Blob(struct.pack('<3Q4q',2,2,2,2,0,1,1)))
    elif kind in ('max-pool','avg-pool'):
        pair=(2<<32)|2
        output=c.make(23,24 if kind=='max-pool' else 23,x,pair,pair,0,1)
    elif kind.startswith('scatter-'):
        output=c.make(36,x,leaves[1],1,Blob(struct.pack('<3Q6q',2,2,3,2,0,2,1,1,0)))
    elif kind.startswith('index-add-'):
        output=c.make(37,x,leaves[1],1,Blob(struct.pack('<2Q2q',1,2,2,0)),bits(.5))
    elif kind=='conv2d': output=c.make(22,x,leaves[1],(1<<32)|1,0,(1<<32)|1,1)
    elif kind=='matmul-fold-copy': output=c.make(32,x,leaves[1])
    else: raise AssertionError(kind)
    meta_and_values(c,output,observation['output'],'forward output')
    return leaves,output


def reference_status(call):
    if call['outcome']=='SUCCEEDED': return 0
    assert call['outcome']=='RAISED' and call['exception_type']=='RuntimeError'
    assert 'Trying to backward through the graph a second time' in call['message'],call
    return 10


def execute(c,kind,observation):
    leaves,output=build(c,kind,observation)
    first_seed=observation['first']['seed']
    assert first_seed is not None
    seed=c.new(first_seed['shape'],first_seed['values'])
    meta_and_values(c,seed,first_seed,'nonuniform seed')
    stats=[]
    for label in ('first','second'):
        call=observation[label]
        assert call['retain_graph_argument'] is False and call['seed']==first_seed
        c.one(57,output,seed,0,status=reference_status(call),label='actual Torch '+label+' backward outcome')
        for native,expected,original in zip(leaves,call['gradients_after_call'],observation['leaves']):
            if expected is None:
                assert not original['requires_grad']
            else:
                assert expected['shape']==original['shape']
                c.values(native,expected['values'],gradient=True)
        stats.append(list(words(c.one(54,label='native post-'+label+' resource counts'))))
    c.audit()
    return {'native_stats_after_calls':stats,
            'pytorch_saved_properties_after_first':observation['saved_after_first']}


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
    reference=json.loads(args.observations.read_text())
    assert reference['status']=='OBSERVED' and reference['cases']==36 and reference['probe_errors']==0
    assert reference['probe_sha256']==PROBE_SHA and reference['target_execution'] is False
    assert reference['oracle_provenance']['status']=='PASS'
    checks=reference['oracle_provenance']['checks'];assert len(checks)==11 and all(v is True for v in checks.values())
    selected=reference['rows'][:20]
    assert [r['case'] for r in selected]==['repeat-default-'+kind for kind in KINDS]
    assert all(r['status']=='OBSERVED' for r in selected)
    spec=importlib.util.spec_from_file_location('backward_release_replay_observer',args.transport_observer)
    transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
    args.output.mkdir(parents=True,exist_ok=False)
    source_files=[Path(__file__),Path(__file__).with_name('native_release_controls.py'),args.transport_observer,args.observations]
    source_hashes={p.name:sha(p.read_bytes()) for p in source_files}
    archive=args.output/'executed-sources';archive.mkdir()
    for p in source_files: (archive/p.name).write_bytes(p.read_bytes())
    rows=[]
    for index,(kind,case) in enumerate(zip(KINDS,selected)):
        directory=args.output/f'case-{index+1:02d}';session=transport.OwnerSession(args.binary,directory=directory,address_limit=64<<20)
        check=Check(session);row={'case':case['case'],'status':'FAIL',
            'reference_outcomes':[case['observation'][key]['outcome'] for key in ('first','second')]}
        try:
            details=execute(check,kind,case['observation']);session.finish();row.update(status='PASS',details=details)
        except Exception as error:
            row.update(exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            if session.poll() is None: os.kill(session.pid,signal.SIGKILL)
            try: session.finish(expected=-signal.SIGKILL)
            except Exception as cleanup: row['cleanup_exception']=repr(cleanup)
        row.update(frames=check.frames,exact_scalar_or_binary32_comparisons=check.comparisons,
                   entire_committed_owner_rollback_bytes=check.rollback_bytes)
        (directory/'report.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');rows.append(row)
    assert source_hashes=={p.name:sha(p.read_bytes()) for p in source_files}
    assert before=={key:sha(getattr(args,key).read_bytes()) for key in before}
    report={'schema':'heptatorch.native-replay-pinned-backward-release.v1',
            'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
            'native_executable_sha256':before['binary'],'native_source_sha256':before['source'],
            'observations_sha256':before['observations'],'probe_sha256':PROBE_SHA,
            'oracle_provenance':reference['oracle_provenance'],'source_hashes':source_hashes,
            'reference_scope':'First 20 operator cases of the actual 36-case pinned Torch report; the remaining 16 are not counted by this replay.',
            'reference_was_previously_executed':True,'pinned_Torch_executed_during_replay':False,'native_executed':True,
            'rtol':0.,'atol':0.,'tolerance_reason':'Fixed small dyadic operator examples; all compared binary32 values and metadata must be exact.',
            'cases':len(rows),'passed':sum(r['status']=='PASS' for r in rows),
            'frames':sum(len(r['frames']) for r in rows),
            'exact_scalar_or_binary32_comparisons':sum(r['exact_scalar_or_binary32_comparisons'] for r in rows),
            'entire_committed_owner_rollback_bytes':sum(r['entire_committed_owner_rollback_bytes'] for r in rows),
            'rows':rows,'full_rewrite_complete':False,'pytorch_api_denominator':2590}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','cases','passed','frames','exact_scalar_or_binary32_comparisons','entire_committed_owner_rollback_bytes')}))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
