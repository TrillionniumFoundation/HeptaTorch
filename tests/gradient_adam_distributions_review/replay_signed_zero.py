"""Bitwise replay of the genuine pinned 13-case signed-zero gradient probe.

All authoritative input, seed and gradient binary32 words are read from the
hash-bound actual Torch report. Mismatches are retained without stopping later
calls or cases. This is not a floating tolerance test and never uses float==
to decide whether +0 and -0 are equal.
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

OBSERVATIONS_SHA='c8cc73a3f06ffc6ab4c30e4acd5e7e98c1b544dcd2736a25af41a2d9e4c15fb4'
PROBE_SHA='4e31fd8fcabf0349a31a3f833906776567c54a467e28f8eba833ca7d5df61d2c'
CONTROLS_SHA='43fbb5a3abb62125191266f7eec9d020717ae9f6388c171edbda6957dda255a8'
OBSERVER_SHA='dee3e4cb25ee1e5e06513e2f33478fbe274d9510ce2aaab9525815c72312946b'


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
    b=load('signed_zero_codec',a.controls);transport=load('signed_zero_transport',a.transport_observer)
    reference=json.loads(a.observations.read_text());assert reference['status']=='OBSERVED' and reference['cases']==13 and reference['probe_errors']==0
    assert reference['probe_sha256']==PROBE_SHA and len(reference['oracle_provenance']['checks'])==11 and all(reference['oracle_provenance']['checks'].values())
    a.output.mkdir(parents=True,exist_ok=False);archive=a.output/'executed-sources';archive.mkdir()
    files=(Path(__file__),a.controls,a.transport_observer,a.observations);source_hashes={p.name:sha(p.read_bytes()) for p in files}
    for path in files: (archive/path.name).write_bytes(path.read_bytes())

    def create(c,ref,requires=None):
        shape=ref['shape'];words=ref['binary32_words'];assert len(shape)<=4
        command=(1,len(shape),int(ref['requires_grad'] if requires is None else requires),*shape,*([0]*(4-len(shape))),
                 b.Blob(struct.pack('<'+'I'*len(words),*words)))
        return c.make(*command)

    def bit_values(c,tid,ref,stage,gradient=False):
        record=c.one(11 if gradient else 10,tid,label=stage)
        expected=ref['binary32_words'];assert record['value']==len(expected) and len(record['payload'])==4*len(expected)
        actual=list(struct.unpack('<'+'I'*len(expected),record['payload']))
        differences=[{'index':i,'actual':left,'expected':right,'actual_hex':f'{left:08x}','expected_hex':f'{right:08x}'}
                     for i,(left,right) in enumerate(zip(actual,expected)) if left!=right]
        c.word_comparisons+=len(expected)
        if differences: c.mismatches.append({'stage':stage,'differences':differences})
        return {'stage':stage,'actual_words':actual,'expected_words':expected,'equal':not differences}

    def presence(c,tid,ref,stage):
        record=c.one(59,tid,label=stage+' gradient presence');assert record['payload']==b''
        wanted=int(ref is not None);c.comparisons+=1
        assert record['value']==wanted,(record['value'],wanted)
        return None if ref is None else bit_values(c,tid,ref,stage,True)

    rows=[]
    for i,case in enumerate(reference['rows'],1):
        directory=a.output/f'case-{i:02d}';session=transport.OwnerSession(a.binary,directory=directory,address_limit=64<<20)
        c=b.Check(session);c.mismatches=[];c.word_comparisons=0;row={'case':case['case'],'status':'FAIL'}
        try:
            o=case['observation'];initial=o['initial'];op=o['operation'];x=create(c,initial['input']);output=x
            if op=='clone': output=c.make(18,x)
            elif op=='view': output=c.make(17,x,0)
            elif op=='transpose': output=c.make(6,x,0,1)
            elif op=='neg': output=c.make(3,6,x)
            elif op=='mul-one': output=c.make(2,3,x,c.new([], [1.]))
            elif op=='shared-add': output=c.make(2,1,x,x)
            elif op=='sum': output=c.make(4,7,x)
            elif op=='gather': output=c.make(31,x,0,b.Blob(struct.pack('<2Qq',1,1,0)))
            else: assert op=='leaf'
            forward=[bit_values(c,x,initial['input'],'input'),bit_values(c,output,initial['output'],'forward output')]
            presence(c,x,initial['gradient'],'initial');calls=[]
            for call_index,call in enumerate(o['calls']):
                before=presence(c,x,call['gradient_before'],str(call_index)+' before')
                seed=create(c,call['seed'],False);bit_values(c,seed,call['seed'],str(call_index)+' seed')
                record=c.one(58,output,seed,0,label='exact signed-zero backward')
                assert call['outcome']=='SUCCEEDED' and record['value']==0
                expected=(0,1 if op=='leaf' else 2,1,0);assert b.words(record)==expected;c.comparisons+=5
                after=presence(c,x,call['gradient_after'],str(call_index)+' after')
                calls.append({'before':before,'after':after,'native_backward_result':list(b.words(record))})
            c.audit();session.finish()
            row.update(status='FAIL' if c.mismatches else 'PASS',forward=forward,calls=calls,mismatches=c.mismatches)
        except Exception as error:
            row.update(exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),mismatches=c.mismatches)
            if session.poll() is None: os.kill(session.pid,signal.SIGKILL)
            try: session.finish(expected=-signal.SIGKILL)
            except Exception as cleanup: row['cleanup_exception']=repr(cleanup)
        row.update(frames=c.frames,exact_metadata_comparisons=c.comparisons,exact_binary32_word_comparisons=c.word_comparisons)
        (directory/'report.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');rows.append(row)
    assert hashes=={name:sha(path.read_bytes()) for name,path in paths.items()}
    assert source_hashes=={path.name:sha(path.read_bytes()) for path in files}
    report={'schema':'heptatorch.native-replay-pinned-gradient-signed-zero.v1','status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',
        'native_executable_sha256':hashes['binary'],'native_source_sha256':hashes['source'],'observations_sha256':OBSERVATIONS_SHA,
        'probe_sha256':PROBE_SHA,'test_source_hashes':source_hashes,'oracle_provenance':reference['oracle_provenance'],
        'cases':len(rows),'passed':sum(r['status']=='PASS' for r in rows),'failed':sum(r['status']=='FAIL' for r in rows),
        'frames':sum(len(r['frames']) for r in rows),'exact_metadata_comparisons':sum(r['exact_metadata_comparisons'] for r in rows),
        'exact_binary32_word_comparisons':sum(r['exact_binary32_word_comparisons'] for r in rows),
        'different_word_observations':sum(len(m['differences']) for r in rows for m in r.get('mismatches',[])),
        'comparison':'Unsigned binary32 word equality; +0 and -0 remain different; no tolerance.',
        'native_executed':True,'pinned_Torch_executed_during_replay':False,
        'rows':rows,'full_rewrite_complete':False,'pytorch_api_denominator':2590}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','cases','passed','failed','frames','exact_metadata_comparisons','exact_binary32_word_comparisons','different_word_observations')}))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
